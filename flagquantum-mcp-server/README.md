# FlagQuantum MCP Server

An [MCP](https://modelcontextprotocol.io) server that gives any MCP-compatible
agent local access to the [FlagQuantum](https://github.com/flagos-ai/FlagQuantum)
SDK: build, compile, route, serialize and plan quantum circuits, with no
credentials, no network access and no hardware submission.

Part of [`FlagQuantum/mcp-servers`](https://github.com/FlagQuantum/mcp-servers).

## What it does

Nine read-only tools over stdio:

| Tool | What it answers |
| --- | --- |
| `analyze_circuit_tool` | Gate counts, depth, wire usage, two-qubit gate count |
| `serialize_circuit_tool` | Canonical IR JSON plus its content hash |
| `deserialize_circuit_tool` | Is this IR valid, and does it round-trip unchanged? |
| `optimize_circuit_tool` | What did target-independent optimization change? |
| `route_circuit_tool` | What does this circuit cost on a line / ring / grid / custom topology? |
| `compare_topologies_tool` | Which connectivity is cheapest for this circuit? |
| `emit_openqasm_tool` | OpenQASM 2.0 or 3.0 text |
| `emit_qcis_tool` | QCIS text |
| `plan_execution_tool` | How would the SDK execute this — which mode, device, how much memory? |

Three resources: `flagquantum://version`, `flagquantum://gate-set`,
`flagquantum://ir-schema`.

Three prompts: `build_and_analyze_circuit`, `compile_for_topology`,
`export_circuit`.

## Install

```bash
pip install flagquantum-mcp-server
```

This pulls `flagquantum`, which depends on `torch`.

### Claude Code

```bash
claude mcp add flagquantum -- uvx flagquantum-mcp-server
```

### Claude Desktop / Cline

```json
{
  "mcpServers": {
    "flagquantum": {
      "command": "uvx",
      "args": ["flagquantum-mcp-server"]
    }
  }
}
```

### MCP Inspector

```bash
npx @modelcontextprotocol/inspector uvx flagquantum-mcp-server
```

## Circuit formats

Two input formats are accepted, both of them FlagQuantum's own serialization.

**`ir`** (canonical) — FlagQuantum IR JSON, as produced by
`CircuitIR.to_json()`. Versioned, hashable, and rejected if it carries unknown
fields:

```json
{
  "kind": "flagquantum.circuit_ir",
  "version": "1.0",
  "n_wires": 2,
  "dtype": "complex64",
  "shape": [4],
  "instructions": [
    {"opcode": "h", "wires": [0], "params": {}, "matrix": null, "metadata": {}},
    {"opcode": "cx", "wires": [0, 1], "params": {}, "matrix": null, "metadata": {}}
  ],
  "observables": [],
  "measurements": [],
  "metadata": {}
}
```

**`qir`** (convenience) — the compact gate list from `Circuit.to_qir()`, easier
to write by hand:

```json
[{"name": "h", "index": [0]}, {"name": "cx", "index": [0, 1]}]
```

**The two formats use different key names**, and this is the most common
mistake: the gate list calls a gate `name` and its wires `index`, while serialized
IR calls them `opcode` and `wires`. Sending IR keys as `qir` is rejected with a
message that says so by name. An empty gate list is also rejected, because the
wire count is inferred from the highest index — an empty list describes no
circuit. A bare integer is accepted for a single-wire gate
(`"index": 0` means `"index": [0]`).

Either format can be passed to any tool; `serialize_circuit_tool` converts
`qir` into canonical `ir`.

`circuit_format` is a closed set — the JSON schema publishes
`"enum": ["ir", "qir"]`, so a wrong value is rejected before any tool body runs.
**OpenQASM text is not a supported input.** FlagQuantum ships emitters but no
QASM parser, so there is nothing to convert it with; a caller holding OpenQASM
has to load it into FlagQuantum itself and send the resulting IR.

## Limits

Every bound is overridable by environment variable, so a deployment can tighten
them without a code change:

| Variable | Default | Bounds |
| --- | --- | --- |
| `FLAGQUANTUM_MCP_MAX_QUBITS` | 24 | Circuit width |
| `FLAGQUANTUM_MCP_MAX_GATES` | 10000 | Instruction count |
| `FLAGQUANTUM_MCP_MAX_IR_BYTES` | 262144 | Serialized circuit payload |
| `FLAGQUANTUM_MCP_MAX_QASM_CHARS` | 1000000 | Emitted program size |
| `FLAGQUANTUM_MCP_MAX_COMPARE_TOPOLOGIES` | 4 | Topologies per comparison |

## Errors

There are two layers, and which one answers depends on whether the schema could
describe the mistake.

**Schema violations** are caught by the MCP layer before any tool body runs and
come back as a protocol error. That covers a `circuit_format` outside the enum,
a missing required argument, and an argument of the wrong JSON type.

**Everything else** comes back as a structured envelope, so a caller can branch
on the code instead of parsing prose:

```json
{"status": "error", "error": {"code": "LIMIT_EXCEEDED", "message": "..."}}
```

Codes: `INVALID_INPUT`, `LIMIT_EXCEEDED`, `UNSUPPORTED_FORMAT`,
`SDK_UNAVAILABLE`, `INTERNAL_ERROR`.

Both layers reach the client as an error it can read; only the layer differs.
Nothing escapes as an unhandled exception that would break the transport — a
failed call leaves the session usable for the next one, which
`tests/test_server_process.py` asserts over a real stdio connection.

## What this server deliberately does not do

- **No execution.** Nothing runs a circuit, locally or remotely. Planning is
  `plan_execution_tool`; running is the caller's step, through `fq.run`.
- **No hardware, no credentials, no network.** FlagQuantum's own release 0.2.0
  ships no remote-submission entry point, and this adapter adds none.
- **No noise models.** A `NoiseModel` is a live SDK object rather than a
  serializable value, and every tool here takes and returns JSON.
- **No in-tree coupling.** This package must never be imported by the
  FlagQuantum repository. That project's long-horizon architecture contract
  names "the main repository has no production MCP transport dependency" as a
  retirement condition, and its `tests/team/services/test_service_boundaries.py`
  fails if `mcp` or `fastmcp` becomes importable on the core path. Keeping the
  gateway out of tree is what that contract asks for.

## Which contracts this rests on

FlagQuantum publishes a frozen `stable_exports` snapshot (34 names, each with a
named verification test) and separately describes `flagquantum.compiler` as its
"stable expert compiler interface". This server uses both, in two tiers:

| Tier | Surface | Tools |
| --- | --- | --- |
| Frozen snapshot | `Circuit`, `CircuitIR`, `Instruction`, `IR_VERSION`, `ExecutionOptions`, `ExecutionPlan`, `plan`, … | analyze, serialize, deserialize, plan |
| Public module (`__all__`) | `flagquantum.compiler`: `CouplingMap`, `optimize`, `route_to_topology` | optimize, route, compare |
| Public submodule (no `__all__`) | `flagquantum.compiler.openqasm.emit_openqasm`, `flagquantum.compiler.qcis.emit_qcis` | emit_openqasm, emit_qcis |

The third tier is the weakest: those two functions are public but are not
re-exported from `flagquantum.compiler`. They are resolved through a helper
that turns a relocation into a named error rather than an `AttributeError`
inside a tool call, and `tests/test_api_contract.py` pins both paths so a move
fails the build instead of failing a user.

The dependency is pinned to `flagquantum>=0.2,<0.3`. It is a version range,
never a git URL: a URL in the dependency table makes every environment that
installs a different upstream revision unresolvable.

## Development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/mypy --config-file ../mypy.ini src
.venv/bin/pytest -m "not integration"
```

See the [repository README](../README.md) and [CONTRIBUTING.md](../CONTRIBUTING.md).

## License

Apache-2.0.
