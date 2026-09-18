"""Drive the FlagQuantum MCP server over stdio, end to end.

Runs the real console script as a subprocess and speaks MCP to it, so this is
the same path an agent host takes. No API key and no LLM are involved: the
point is to show the tool sequence, not to call a model.

    python examples/stdio_client.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StdioTransport

# A GHZ state over three wires, written as the compact gate list this server
# accepts with circuit_format="qir".
GHZ = [
    {"name": "h", "index": [0]},
    {"name": "cx", "index": [0, 1]},
    {"name": "cx", "index": [1, 2]},
]


def payload(result: Any) -> dict[str, Any]:
    """Pull the structured payload out of an MCP tool result."""
    if isinstance(result.structured_content, dict):
        return result.structured_content
    return json.loads(result.content[0].text)


async def main() -> None:
    """Run one build, analyze, route, export and plan sequence."""
    # `python -m flagquantum_mcp_server` works wherever the package is
    # importable, so the example does not depend on the console script being
    # on PATH. An MCP host normally uses the console script or `uvx` instead.
    transport = StdioTransport(command=sys.executable, args=["-m", "flagquantum_mcp_server"])
    async with Client(transport) as client:
        tools = await client.list_tools()
        print(f"server exposes {len(tools)} tools")
        for tool in tools:
            print(f"  {tool.name}")

        circuit = json.dumps(GHZ)

        print("\n-- serialize --")
        canonical = payload(
            await client.call_tool(
                "serialize_circuit_tool", {"circuit": circuit, "circuit_format": "qir"}
            )
        )
        print(f"  ir_version    {canonical['ir_version']}")
        print(f"  content_hash  {canonical['content_hash']}")

        print("\n-- analyze --")
        analysis = payload(
            await client.call_tool(
                "analyze_circuit_tool", {"circuit": circuit, "circuit_format": "qir"}
            )
        )
        print(f"  depth         {analysis['analysis']['depth']}")
        print(f"  gate_counts   {analysis['analysis']['gate_counts']}")

        print("\n-- compare topologies --")
        comparison = payload(
            await client.call_tool(
                "compare_topologies_tool", {"circuit": circuit, "circuit_format": "qir"}
            )
        )
        for entry in comparison["topologies"]:
            print(
                f"  {entry['topology']['n_edges']:>2} edges  "
                f"{entry['n_instructions']} instructions  depth {entry['depth']}"
            )

        print("\n-- emit --")
        qasm = payload(
            await client.call_tool(
                "emit_openqasm_tool",
                {"circuit": circuit, "circuit_format": "qir", "version": 3.0},
            )
        )
        for line in qasm["text"].splitlines():
            print(f"  {line}")

        print("\n-- plan --")
        plan = payload(
            await client.call_tool(
                "plan_execution_tool",
                {
                    "circuit": circuit,
                    "circuit_format": "qir",
                    "options": {"shots": 1024},
                    "outputs": [{"kind": "counts", "wires": [0, 1, 2]}],
                },
            )
        )
        print(f"  mode          {plan['plan']['mode']}")
        print(f"  device        {plan['plan']['device']}")
        print(f"  state_bytes   {plan['summary']['state_bytes']}")

        print("\n-- error handling --")
        failure = payload(
            await client.call_tool(
                "analyze_circuit_tool", {"circuit": "{}", "circuit_format": "ir"}
            )
        )
        print(f"  status        {failure['status']}")
        print(f"  code          {failure['error']['code']}")


if __name__ == "__main__":
    asyncio.run(main())
