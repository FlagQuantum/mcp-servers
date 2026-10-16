"""End-to-end: run the real console script and speak MCP to it over stdio.

This is the only test that proves the shipped entry point works — the packaging,
the console script, the stdio framing and the tool schemas, all at once. It is
marked ``integration`` because it spawns a process; CI runs it in its own job.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from tests.surface import (
    EXPECTED_PROMPTS,
    EXPECTED_RESOURCE_TEMPLATES,
    EXPECTED_RESOURCES,
    EXPECTED_TOOLS,
)

pytestmark = pytest.mark.integration

GHZ = [
    {"name": "h", "index": [0]},
    {"name": "cx", "index": [0, 1]},
    {"name": "cx", "index": [1, 2]},
]
_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def transport() -> StdioTransport:
    """Launch the package as a module, so PATH does not matter."""
    env = dict(os.environ)
    source_root = str(_ROOT / "src")
    inherited = env.get("PYTHONPATH")
    env["PYTHONPATH"] = source_root if not inherited else source_root + os.pathsep + inherited
    return StdioTransport(
        command=sys.executable,
        args=["-m", "flagquantum_mcp_server"],
        cwd=str(_ROOT),
        env=env,
    )


async def test_the_server_starts_and_lists_its_surface(transport: StdioTransport) -> None:
    async with Client(transport) as client:
        tools = await client.list_tools()
        resources = await client.list_resources()
        resource_templates = await client.list_resource_templates()
        prompts = await client.list_prompts()

    assert {tool.name for tool in tools} == set(EXPECTED_TOOLS)
    assert {str(resource.uri) for resource in resources} == set(EXPECTED_RESOURCES)
    assert {template.uriTemplate for template in resource_templates} == set(
        EXPECTED_RESOURCE_TEMPLATES
    )
    assert {prompt.name for prompt in prompts} == set(EXPECTED_PROMPTS)


async def test_research_discovery_replays_over_the_wire(transport: StdioTransport) -> None:
    async with Client(transport) as client:
        result = await client.call_tool("search_capabilities_tool", {"query": {}})
        contents = await client.read_resource("flagquantum://capabilities/index")

    assert result.structured_content is not None
    assert result.structured_content["status"] == "success"
    candidates = result.structured_content["result"]
    assert {item["capability_id"] for item in candidates} == {
        "ground_state.solve.vqe",
        "ground_state.solve.exact",
        "validate.energy_variance",
    }
    assert json.loads(contents[0].text) == candidates


async def test_a_tool_call_round_trips_over_the_wire(transport: StdioTransport) -> None:
    async with Client(transport) as client:
        result = await client.call_tool(
            "analyze_circuit_tool",
            {"circuit": json.dumps(GHZ), "circuit_format": "qir"},
        )

    assert result.structured_content is not None
    assert result.structured_content["status"] == "success"
    assert result.structured_content["analysis"]["depth"] == 3


async def test_a_failure_keeps_the_transport_healthy(transport: StdioTransport) -> None:
    """A rejected call must not kill the session for the calls after it."""
    async with Client(transport) as client:
        failed = await client.call_tool(
            "analyze_circuit_tool", {"circuit": "{}", "circuit_format": "ir"}
        )
        succeeded = await client.call_tool(
            "analyze_circuit_tool",
            {"circuit": json.dumps(GHZ), "circuit_format": "qir"},
        )

    assert failed.structured_content is not None
    assert failed.structured_content["status"] == "error"
    assert succeeded.structured_content is not None
    assert succeeded.structured_content["status"] == "success"


async def test_a_resource_reads_over_the_wire(transport: StdioTransport) -> None:
    async with Client(transport) as client:
        contents = await client.read_resource("flagquantum://gate-set")

    # The client hands back raw MCP content blocks, not the server's own
    # ResourceResult wrapper.
    payload = json.loads(contents[0].text)

    assert contents[0].mimeType == "application/json"
    assert payload["n_gates"] == 35
