"""MCP server exposing the FlagQuantum SDK to any MCP-compatible client.

This package is an out-of-tree edge adapter. It is deliberately *not* part of
the FlagQuantum repository: that project's long-horizon architecture contract
names "the main repository has no production MCP transport dependency" as a
retirement condition, so the protocol gateway lives here instead.

Run it over stdio::

    flagquantum-mcp-server

The server is local, deterministic and read-only. It holds no credentials,
contacts no network, and submits nothing to hardware.
"""

from __future__ import annotations

from flagquantum_mcp_server._version import __version__

__all__ = ["__version__", "main"]


def main() -> None:
    """Run the FlagQuantum MCP server over stdio.

    The import is deferred so that ``--help`` and version inspection do not pay
    for loading the MCP SDK.
    """
    from flagquantum_mcp_server.server import mcp

    mcp.run(transport="stdio", show_banner=False)


if __name__ == "__main__":  # pragma: no cover - process entry point
    main()
