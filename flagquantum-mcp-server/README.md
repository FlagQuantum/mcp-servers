# FlagQuantum MCP Server

[![MCP Registry](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fregistry.modelcontextprotocol.io%2Fv0.1%2Fservers%2Fio.github.FlagQuantum%252Fflagquantum-mcp-server%2Fversions%2Flatest&query=%24.server.version&label=MCP%20Registry&logo=modelcontextprotocol)](https://registry.modelcontextprotocol.io/?q=io.github.FlagQuantum%2Fflagquantum-mcp-server)

<!-- mcp-name: io.github.FlagQuantum/flagquantum-mcp-server -->

An [MCP](https://modelcontextprotocol.io) server that gives any MCP-compatible
agent local access to the [FlagQuantum](https://github.com/flagos-ai/FlagQuantum)
SDK: build, compile, route, serialize and plan quantum circuits, with no
credentials, no network access and no hardware submission.

Part of [`FlagQuantum/mcp-servers`](https://github.com/FlagQuantum/mcp-servers).

The `mcp-name` comment above is not decoration: the MCP Registry reads it from
this README to verify that whoever publishes the registry entry also controls
the PyPI package. Removing it breaks registry publishing.

## What it does

Fifteen read-only tools over stdio.

**Build and inspect**

| Tool | What it answers |
| --- | --- |
| `describe_gate_set_tool` | Wire count, parameter names and aliases for named gates — or every gate |
| `analyze_circuit_tool` | Gate counts, depth, wire usage, two-qubit gate count |
| `describe_layers_tool` | Which gates run concurrently, and therefore where the depth comes from |
| `serialize_circuit_tool` | Canonical IR JSON plus its content hash |
| `deserialize_circuit_tool` | Is this IR valid, and does it round-trip unchanged? |

**Parameters**

| Tool | What it answers |
| --- | --- |
| `inspect_parameters_tool` | Is this circuit parameterized, and where does each symbol sit? |
| `bind_parameters_tool` | What does this ansatz look like once the symbols are numbers? |

**Compile and route**

| Tool | What it answers |
| --- | --- |
| `optimize_circuit_tool` | What did target-independent optimization change? |
| `route_circuit_tool` | What does this circuit cost on a line / ring / grid / custom topology? |
| `compare_topologies_tool` | Which connectivity is cheapest for this circuit? |
| `describe_topology_tool` | What is the connectivity, and how far apart are two wires? |

**Export and present**

| Tool | What it answers |
| --- | --- |
| `emit_openqasm_tool` | OpenQASM 2.0 or 3.0 text |
| `emit_qcis_tool` | QCIS text |
| `draw_circuit_tool` | An ASCII diagram of the circuit |
| `plan_execution_tool` | How would the SDK execute this — which mode, device, how much memory? |

Three resources: `flagquantum://version` (versions of the server, the SDK and
the IR contract), `flagquantum://gate-set` (every gate with its wire count and
parameter names) and `flagquantum://ir-schema` (the IR envelope, shown by
example from a real serialization).

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

A gate carries its arguments under **`parameters`**, which the example above
never shows because neither gate takes one:

```json
[{"name": "rz", "index": [0], "parameters": {"theta": 0.5}}]
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

One more naming trap: an IR payload's version field is spelled **`version`**,
and sending `ir_version` instead is rejected as an unknown key — but tool
results report that same field as `ir_version`. The payload key and the reported
key are not spelled the same way.

`circuit_format` is a closed set — the JSON schema publishes
`"enum": ["ir", "qir"]`, so a wrong value is rejected before any tool body runs.
**OpenQASM text is not a supported input.** FlagQuantum ships emitters but no
QASM parser, so there is nothing to convert it with; a caller holding OpenQASM
has to load it into FlagQuantum itself and send the resulting IR.

### On parameters

A gate parameter is either a number or a **symbol**, and a symbol is written as
a one-key object naming it:

```json
{"name": "ry", "index": [0], "parameters": {"theta": {"$parameter": "theta"}}}
```

This works in both input formats. A symbol is what makes a circuit an ansatz:
`inspect_parameters_tool` reports it, and `bind_parameters_tool` substitutes a
number for it.

**A bare string is not a symbol.** `{"theta": "theta"}` is a value, not a
reference; the SDK stores it as one, so the circuit reports itself as
unparameterized. Nothing else complains: `draw_circuit_tool` prints the value it
was handed, and a string prints as `RY(theta)` — indistinguishable from a real
symbol — while `plan_execution_tool` plans the circuit for execution. So the
input boundary rejects it, along with a misspelled marker (`{"$unknown": ...}`)
and a non-string symbol name, naming the gate and the parameter. The other
encodings a parameter may carry are `$expression`, `$complex` and `$tensor`; a
mapping carrying none of those four keys is rejected for the same reason. The
`flagquantum://ir-schema` resource carries a worked `parameter_example`.

One rough edge, inherited from the SDK: a *real* symbol is rendered as the SDK's
repr of it, so `RY(theta)` appears as `RY(Parameter(name='theta'))`.
FlagQuantum's `Parameter` defines no `__str__`, so `str()` falls through to
`__repr__`. The symbol is intact — only the diagram's spelling is clumsy — and
it affects both input formats equally.

### On an explicit unitary

A gate's matrix travels under a key that **differs by format**: the gate list
calls it `gate`, serialized IR calls it `matrix`.

```json
[{"name": "any", "index": [0], "gate": [[0, 1], [1, 0]]}]
```

Two things are rejected rather than half-honoured.

**The wrong key for the format.** `Circuit.from_qir` reads `gate` and ignores
`matrix` entirely, so a gate list spelling it the IR way would be built with no
matrix at all — silently. For a built-in name that means a different circuit
than the caller wrote, so the key mismatch is named.

**A matrix on a built-in opcode.** A matrix belongs to `any`, the opcode
FlagQuantum reserves for it. On a built-in name the SDK keeps the matrix in the
payload but lets the built-in's own definition win, which splits the tools:
`analyze_circuit_tool` reports the gate under the built-in's name while the
emitters refuse to lower it. Name the gate `any` instead.

### On a wire number

Wire numbers must be integers in both formats, and the `wires` list must be a
list. Both are worth stating because the SDK accepts more: it iterates whatever
it is handed and calls `int()` on each element, so `"1"`, `true`, `1.7` and
`0.9` are wires, and `"01"` is wires 0 and 1 while `{"0": 1}` is wire 0. The last
cases are why this is an error rather than a convenience — a value that means
nothing turns into a circuit that looks fine.

### On the envelope's own fields

The same rule, applied to the fields outside `instructions`. `n_wires` and the
entries of `shape` are read with `int()`, so `"2"`, `2.7` and `true` are all a
width; `dtype` is looked up as an attribute on `torch`, so a value that is not a
string reaches an attribute lookup inside the SDK.

The test for what is refused is **whether information is lost**, not whether the
JSON type matches the schema exactly. That is why `version: 1.0` is accepted —
`str(1.0)` reproduces it — while `n_wires: 2.7` is not: it silently becomes 2,
and `n_wires: true` silently becomes 1.

`dtype` is checked for being a *string* here. Which strings are legal stays with
the SDK, whose own message names the rule (`complex_dtype must be complex64 or
complex128`), so the two cannot drift apart.

### On emitted text

`emit_openqasm_tool` measures every wire unless `result_wires` names the ones you
want, so the emitted program normally carries measurements the source circuit
does not — a Bell circuit with no `measurements` still emits `c = measure q;`.
Declaring measurements in the source IR changes nothing: the emitted text is
chosen by `result_wires` alone. The `content_hash` in the result identifies the
**source circuit**, not the emitted text. `emit_qcis_tool` appends nothing.

### On `content_hash`

Every payload carries the SDK's `content_hash`, which is the SHA-256 of the
canonical JSON **including the `metadata` object**. An IR payload that omits
`metadata` is legal — it defaults to `{}` — but it hashes differently from the
same circuit carrying its runtime metadata. Treat the hash as identifying the
payload, not the gate sequence alone. The `flagquantum://ir-schema` resource
says the same thing where an agent will read it.

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
named verification test), describes `flagquantum.compiler` as its "stable expert
compiler interface", and lets each package declare its own `__all__`. This
server uses all three tiers, and `tests/test_api_contract.py` pins the members
of each:

| Tier | Surface | Used by |
| --- | --- | --- |
| 1. Frozen snapshot | `Circuit`, `CircuitIR`, `Instruction`, `IR_VERSION`, `ExecutionOptions`, `ExecutionPlan`, `plan`, `Parameter`, … | analyze, serialize, deserialize, plan, inspect/bind parameters |
| 2. Documented module | `flagquantum.compiler`: `CouplingMap`, `optimize`, `route_to_topology`, `schedule_layers` | optimize, route, compare, describe layers |
| 3. Public but not frozen | `flagquantum.compiler.openqasm.emit_openqasm`, `flagquantum.compiler.qcis.emit_qcis`, `flagquantum.drawer.draw`, `flagquantum.core.{operator_manifest,gate_info,canonical_opcode}` | emit_openqasm, emit_qcis, draw, gate validation |

**Tier 3 is the weakest, and it is not decoration.** Gate validation needs each
gate's wire count and parameter names, and the SDK's operator manifest is the
only authority for that — a caller cannot infer it from the circuit format, and
reimplementing it here would create a second source of truth that drifts. The
manifest is reached through a helper that turns a relocation into a named error
rather than an `AttributeError` inside a tool call, and every tier-3 name is
pinned by a test so a move upstream fails the build instead of failing a user.
The failures that guard against are quiet ones: a weaker gate check accepts a
misspelled parameter and drops it.

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
