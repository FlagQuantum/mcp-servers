# Examples

## `stdio_client.py`

Runs the server as a real subprocess over stdio and walks one full sequence:
serialize → analyze → compare topologies → emit OpenQASM → plan → handle an
error. No API key, no model, no credentials.

```bash
python examples/stdio_client.py
```

This is the reference for how any MCP host drives the server. An agent host
does the same thing, with a model choosing the arguments instead of the script
hard-coding them.

## Driving it from an LLM

Any MCP-compatible client works by pointing it at the console script. There is
no credential to configure:

```json
{
  "mcpServers": {
    "flagquantum": { "command": "flagquantum-mcp-server", "args": [] }
  }
}
```

For a LangChain or LangGraph agent, wrap the same command with
`langchain-mcp-adapters`:

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

client = MultiServerMCPClient(
    {
        "flagquantum": {
            "transport": "stdio",
            "command": "flagquantum-mcp-server",
            "args": [],
        }
    }
)
tools = await client.get_tools()
```

That adapter is not a dependency of this package — install it in your own
project if you want it.

## What these examples deliberately avoid

The upstream reference suite ships a multi-agent example that drives real
quantum hardware. This server has no hardware path, by design: FlagQuantum's
released 0.2.0 ships no remote-submission entry point, and QPU submission is a
governed capability that belongs to a control plane, not to a local adapter
that any agent can call. An example that appeared to submit a job would be
misleading.
