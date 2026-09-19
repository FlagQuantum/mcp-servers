"""MCP server exposing the FlagQuantum SDK to any MCP-compatible client.

This package is an out-of-tree edge adapter. It is deliberately *not* part of
the FlagQuantum repository: that project's long-horizon architecture contract
names "the main repository has no production MCP transport dependency" as a
retirement condition, so the protocol gateway lives here instead.

Run it over stdio (the default), or over Streamable HTTP::

    flagquantum-mcp-server
    flagquantum-mcp-server --transport http --host 0.0.0.0 --port 8105 \\
        --allowed-host quantum-mcp:8105

The server is local, deterministic and read-only. It holds no credentials,
makes no outbound request, and submits nothing to hardware. The HTTP transport
adds an inbound listener; it does not add egress. See AGENTS.md rule 4.
"""

from __future__ import annotations

import argparse
import ipaddress
from collections.abc import Sequence

from flagquantum_mcp_server._version import __version__

__all__ = ["__version__", "main"]

DEFAULT_HTTP_PORT = 8105


def main() -> None:
    """Run the FlagQuantum MCP server over the selected transport.

    Both the SDK import and the argument parse are cheap: the server module is
    imported only once the transport is known, so ``--help`` and ``--version``
    do not pay for loading the MCP SDK.
    """
    args = _parse_args()

    from flagquantum_mcp_server.server import mcp

    if args.transport == "stdio":
        mcp.run(transport="stdio", show_banner=False)
        return

    if args.allowed_host or args.allowed_origin:
        _enable_host_origin_protection(args.allowed_host, args.allowed_origin)

    mcp.run(transport="http", host=args.host, port=args.port, show_banner=False)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="flagquantum-mcp-server",
        description="Serve the FlagQuantum SDK over MCP.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=__version__,
    )
    parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default="stdio",
        help=(
            "stdio (default) is launched and owned by one client; http listens on "
            "--host:--port for clients that connect to it, such as a gateway."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Interface for --transport http. Defaults to loopback; binding a "
            "non-loopback interface additionally requires --allowed-host."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_HTTP_PORT,
        help=f"Port for --transport http. Defaults to {DEFAULT_HTTP_PORT}.",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        metavar="HOST[:PORT]",
        help=(
            "A Host header the server will answer, repeatable. Passing any turns "
            "host validation on; loopback hosts are always answered. Required "
            "when --host is not a loopback address."
        ),
    )
    parser.add_argument(
        "--allowed-origin",
        action="append",
        default=[],
        metavar="ORIGIN",
        help="A browser Origin the server will answer, repeatable.",
    )
    args = parser.parse_args(argv)

    if args.transport == "http" and not _is_loopback(args.host) and not args.allowed_host:
        parser.error(
            f"--host {args.host} is not a loopback address, so the server would "
            "answer any Host header that reaches it. Name the hosts it should "
            "answer with --allowed-host HOST[:PORT], or bind 127.0.0.1."
        )
    return args


def _is_loopback(host: str) -> bool:
    """Whether binding ``host`` exposes the server to more than this machine.

    A name that is not an IP literal counts as non-loopback: resolving it to
    decide would make the check depend on DNS, and a name that resolves to
    loopback today can resolve elsewhere tomorrow.
    """
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _enable_host_origin_protection(
    allowed_hosts: Sequence[str], allowed_origins: Sequence[str]
) -> None:
    """Turn on FastMCP's request guard, which is off by default.

    Configured through the settings object on purpose. ``http_app(
    allowed_hosts=...)`` looks like the way to do this and is not: it appends to
    a list that is only consulted once the guard is on, so passing it alone
    enforces nothing, silently. Measured on fastmcp 3.4.7 — a bogus Host is
    answered 200 with that keyword and 421 through this path.
    """
    import fastmcp

    fastmcp.settings.http_host_origin_protection = True
    fastmcp.settings.http_allowed_hosts = list(allowed_hosts) or None
    fastmcp.settings.http_allowed_origins = list(allowed_origins) or None


if __name__ == "__main__":  # pragma: no cover - process entry point
    main()
