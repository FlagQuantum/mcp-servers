"""End-to-end: the console script's HTTP transport, over a real socket.

The stdio entry point has its own test (``test_server_process.py``). This one
covers the second transport, because a transport is not exercised by importing
the app object: only a process that binds, listens and answers proves it. The
package can already build the app (``mcp.http_app()``) while the shipped entry
point cannot be reached that way — which is the whole reason this file exists.

Marked ``integration`` because it spawns processes, like its stdio twin.

The Host-header cases speak HTTP through ``http.client`` rather than through a
client library. That is deliberate: a client library reaches for the ambient
proxy configuration, and this machine's system proxy answers a request carrying
a non-localhost ``Host`` with its own 502 before the server ever sees it — which
is a measurement of the proxy, not of the server.
"""

from __future__ import annotations

import http.client
import json
import socket
import subprocess
import sys
import time
from collections.abc import Iterator

import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from tests.surface import EXPECTED_TOOLS

pytestmark = pytest.mark.integration

GHZ = [
    {"name": "h", "index": [0]},
    {"name": "cx", "index": [0, 1]},
    {"name": "cx", "index": [1, 2]},
]

STARTUP_TIMEOUT_SECONDS = 60.0


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _spawn(*args: str) -> subprocess.Popen[str]:
    """Launch the package as a module, so PATH does not matter."""
    return subprocess.Popen(
        [sys.executable, "-m", "flagquantum_mcp_server", *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _wait_until_listening(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + STARTUP_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"the server exited with {process.returncode} before listening:\n"
                f"{process.stdout.read() if process.stdout else ''}"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError:
            time.sleep(0.2)
    raise AssertionError(
        f"the server was not listening on {port} within {STARTUP_TIMEOUT_SECONDS}s"
    )


def _stop(process: subprocess.Popen[str]) -> None:
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - only on a stuck child
        process.kill()
        process.wait(timeout=10)


@pytest.fixture
def http_server() -> Iterator[int]:
    port = _free_port()
    process = _spawn("--transport", "http", "--host", "127.0.0.1", "--port", str(port))
    try:
        _wait_until_listening(port, process)
        yield port
    finally:
        _stop(process)


def _initialize(host_header: str, port: int) -> int:
    """Send one MCP initialize carrying an arbitrary Host header."""
    body = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "host-probe", "version": "0"},
            },
        }
    )
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    try:
        connection.putrequest("POST", "/mcp", skip_host=True)
        connection.putheader("Host", host_header)
        connection.putheader("Content-Type", "application/json")
        connection.putheader("Accept", "application/json, text/event-stream")
        connection.putheader("Content-Length", str(len(body)))
        connection.endheaders()
        connection.send(body.encode())
        return int(connection.getresponse().status)
    finally:
        connection.close()


async def test_the_http_transport_serves_the_full_surface(http_server: int) -> None:
    transport = StreamableHttpTransport(f"http://127.0.0.1:{http_server}/mcp")

    async with Client(transport) as client:
        tools = await client.list_tools()
        result = await client.call_tool(
            "analyze_circuit_tool",
            {"circuit": json.dumps(GHZ), "circuit_format": "qir"},
        )

    assert {tool.name for tool in tools} == set(EXPECTED_TOOLS)
    assert result.structured_content is not None
    assert result.structured_content["status"] == "success"
    assert result.structured_content["analysis"]["depth"] == 3


async def test_a_simulation_round_trips_over_http(http_server: int) -> None:
    """The capability the transport exists for: a real result over the wire."""
    transport = StreamableHttpTransport(f"http://127.0.0.1:{http_server}/mcp")

    async with Client(transport) as client:
        result = await client.call_tool(
            "simulate_circuit_tool",
            {
                "circuit": json.dumps(
                    [{"name": "h", "index": [0]}, {"name": "cx", "index": [0, 1]}]
                ),
                "circuit_format": "qir",
                "outputs": [{"kind": "counts", "wires": [0, 1]}],
                "options": {"shots": 256, "seed": 7},
            },
        )

    assert result.structured_content is not None
    assert result.structured_content["status"] == "success"
    counts = result.structured_content["outputs"][0]["value"][0]
    assert sum(counts.values()) == 256
    # A Bell state measured in the computational basis has no |01> or |10>.
    assert set(counts) <= {"00", "11"}


def test_an_allowlisted_host_is_served_and_a_bogus_one_is_not() -> None:
    """The allowlist is the reason a non-loopback bind is allowed at all."""
    port = _free_port()
    process = _spawn(
        "--transport",
        "http",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--allowed-host",
        f"quantum-mcp:{port}",
    )
    try:
        _wait_until_listening(port, process)
        assert _initialize(f"quantum-mcp:{port}", port) == 200
        assert _initialize("evil.example.com", port) == 421
    finally:
        _stop(process)


def test_a_non_loopback_bind_without_an_allowlist_refuses_to_start() -> None:
    """Binding the world with no host check is the failure this prevents.

    The refusal happens while parsing arguments, before the SDK is imported, so
    it is near-instant. A short budget is part of the assertion: if the process
    is still running after it, the server bound the interface, and a timeout is
    the accurate report of that — not a failure to start in time.
    """
    port = _free_port()
    process = _spawn("--transport", "http", "--host", "0.0.0.0", "--port", str(port))
    try:
        try:
            output = process.communicate(timeout=15)[0]
        except subprocess.TimeoutExpired:
            pytest.fail(
                "the server kept running instead of refusing to bind 0.0.0.0 without --allowed-host"
            )
    finally:
        _stop(process)

    assert process.returncode != 0, f"it exited 0 instead of refusing:\n{output}"
    assert "--allowed-host" in output, output
