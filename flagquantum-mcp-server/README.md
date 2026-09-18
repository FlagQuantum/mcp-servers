# FlagQuantum MCP Server

[![MCP Registry](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fregistry.modelcontextprotocol.io%2Fv0.1%2Fservers%2Fio.github.FlagQuantum%252Fflagquantum-mcp-server%2Fversions%2Flatest&query=%24.server.version&label=MCP%20Registry&logo=modelcontextprotocol)](https://registry.modelcontextprotocol.io/?q=io.github.FlagQuantum%2Fflagquantum-mcp-server)

<!-- mcp-name: io.github.FlagQuantum/flagquantum-mcp-server -->

An [MCP](https://modelcontextprotocol.io) server that gives any MCP-compatible
agent local access to the [FlagQuantum](https://github.com/flagos-ai/FlagQuantum)
SDK: build, compile, route, serialize and plan quantum circuits — and run them
locally — with no credentials, no network access and no hardware submission.

Part of [`FlagQuantum/mcp-servers`](https://github.com/FlagQuantum/mcp-servers).

The `mcp-name` comment above is not decoration: the MCP Registry reads it from
this README to verify that whoever publishes the registry entry also controls
the PyPI package. Removing it breaks registry publishing.

## What it does

Seventeen tools over stdio. Fifteen of them only read the circuit they are
given; the other two run it, one to measure it and one to improve it.

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

**Run**

| Tool | What it answers |
| --- | --- |
| `simulate_circuit_tool` | Counts, samples, marginal probabilities or an expectation value from a local statevector run |
| `train_parameters_tool` | The angles that lower a circuit's energy against a Pauli-sum Hamiltonian |

Three resources: `flagquantum://version` (versions of the server, the SDK and
the IR contract), `flagquantum://gate-set` (every gate with its wire count and
parameter names) and `flagquantum://ir-schema` (the IR envelope, shown by
example from a real serialization).

Three prompts: `build_and_analyze_circuit`, `compile_for_topology`,
`export_circuit`.

These are recipes for the clients that present them to a person — Claude
Desktop, the MCP Inspector, a host that turns them into slash commands. An
autonomous agent may receive none of them: several recorded sessions could not
enumerate them at all, and reported that they could not tell whether prompts
existed. So nothing an agent has to know lives only here. Every constraint the
prompts state is also in a tool summary, or in the server instructions, or in a
resource — the QCIS refusal and the "say what you assumed" rule were the two
exceptions, and they are in `emit_qcis_tool` and `bind_parameters_tool` now.
`PROMPT_CONSTRAINTS` in `tests/test_server_contract.py` keeps that true.

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

Binding is needed to **export**: the emitters write numbers, and refuse an
unbound circuit naming `bind_parameters`. Nothing else needs it. `analyze`,
`optimize`, `route`, `draw`, `describe_layers` and `plan_execution_tool` all
accept a parameterized circuit as it stands, because none of them reads a
parameter value — a plan is built from the payload's shape and dtype. Bind when
you want the emitted text or a concrete circuit to run, not before.

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

### On observables and measurements

Both are lists of objects, empty when the circuit has none, and an entry is
short — the SDK fills in the rest on load:

```json
{
  "observables": [{"name": "ZZ", "wires": [0, 1]}],
  "measurements": [{"kind": "counts", "wires": [0, 1], "shots": 1024}]
}
```

What the SDK stores adds the optional keys and lowercases the observable name,
so `"ZZ"` comes back as `"zz"` carrying `"coefficient": 1.0`. `coefficient` is
the term's weight, and it may be a symbol exactly as a gate angle may
(`{"$parameter": "w"}`), which is how a Hamiltonian term carries a variational
weight. `shots` is null when omitted.

An observable is what a VQE ansatz exists to measure, so it is worth knowing
that the shape is reachable without being wrong first — `flagquantum://ir-schema`
publishes both forms side by side as `observable_example`.

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
| `FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS` | 60 | Predicted cost of one training call |
| `FLAGQUANTUM_MCP_MAX_RESPONSE_VALUES` | 65536 | Values one result may carry back |
| `FLAGQUANTUM_MCP_MAX_HAMILTONIAN_TERMS` | 1024 | Pauli terms per expectation request |

The last one is the only bound on what this server *returns* rather than what it
accepts. A result travels into a model's context rather than into memory, and a
sampled result grows with `shots` — 200,000 draws is legal by every other bound
here and far more than an answer.

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

## What it runs, and what that does not prove

`simulate_circuit_tool` executes circuits in this process: a statevector
simulation on CPU. That is the whole of it — no remote target, no provider, no
token. It was not always here. Until it landed the server planned and never ran,
and the line above used to say so; a real task showed the cost, so the line
moved rather than the tool being smuggled in underneath it.

The result says where it ran, in the SDK's own fields rather than in a summary:

```json
"execution": {
  "execution_path": "local_statevector",
  "platform_provider": "pytorch_cpu",
  "mode": "statevector",
  "device": "cpu",
  "accuracy": {"metric": "not_measured", "...": "..."}
}
```

`accuracy.metric` is `not_measured`, and the plan behind the run reports
`release_gate_allowed: false`. An ideal simulation of a circuit is a statement
about that circuit, not about any device that would run it — the SDK declines to
license a stronger reading, and this server does not add one.

Three things are refused rather than reinterpreted, each because accepting
would produce a plausible-looking answer:

| Input | Why it is refused |
| --- | --- |
| A circuit with unbound parameters | The SDK's refusal is `planned execution failed`, which names nothing. The message here names the parameters and points at `bind_parameters_tool` |
| A circuit carrying `observables` | The default statevector path never reads the field, so the run would return no expectation value and no error either. Ask for the expectation directly with `pauli` or `terms` instead |
| `outputs` alongside the circuit's own `measurements` | The SDK accepts one or the other, and says so in vocabulary that names neither field |

The width where a full probability distribution stops being returnable is the
SDK's own contraction limit, and its message points at a setting this server
does not expose. That refusal keeps the SDK's text and adds what a caller can
actually do — name a few wires, or ask for `counts`, which report only the
outcomes that occurred.

### Measuring an energy

`{"kind": "expectation", "pauli": "ZZ"}` evaluates one term. A Hamiltonian is a
weighted sum, so it takes `terms` instead:

```json
{"kind": "expectation",
 "terms": [{"pauli": "ZZZZ", "coefficient": 1.0},
           {"pauli": "XIII", "coefficient": -1.0}]}
```

Each term comes back as **its own row**, carrying its `coefficient` and its
value, because that is what the SDK computes — it evaluates the sum term by term
and does not total it. So `⟨H⟩` is the caller's arithmetic:

```python
energy = sum(row["coefficient"] * row["value"][0] for row in rows)
```

That is deliberate rather than convenient. A total computed here would be a
number the SDK never produced and this server could not attribute, and the
per-term rows are what make the result auditable — you can see which term
dominated. `coefficient` appears only on rows the SDK put one on, which means
the terms of a weighted expectation.

A coefficient must be a real number. A **symbol** is refused: `Z(0) * Parameter`
builds a parameter expression rather than an observable, so there is nothing to
evaluate — bind the circuit's gates first instead. A single unweighted term is
still spelled `"pauli": "ZZ"`, and the two spellings are one code path, so they
cannot drift apart.

### Lowering an energy

`simulate_circuit_tool` evaluates an energy. `train_parameters_tool` improves
one, using the exact gradients a statevector simulation reports:

```json
{
  "circuit": "[{\"name\": \"ry\", \"index\": [0], \"parameters\": {\"theta\": {\"$parameter\": \"t0\"}}}]",
  "circuit_format": "qir",
  "hamiltonian": [{"pauli": "Z", "coefficient": -1.0}, {"pauli": "X", "coefficient": 1.0}],
  "steps": 100,
  "learning_rate": 0.1
}
```

It returns the loss after each step, the parameters it ended on, and the
parameters it started from. Call it again with `values` set to the `parameters`
it returned to continue the run rather than restart it.

The `hamiltonian` argument is the same term list an `expectation` output takes,
and it is required: without an objective the number being minimised is not an
energy. **One letter per wire** — the circuit above is one wire, so each term is
one letter. A two-wire circuit takes `"ZZ"`, and a term whose length does not
match the circuit is refused before anything runs.

Two limits are worth knowing before you call it. Training is far more expensive
than simulating — measured, a 16-qubit step costs 85 ms against a simulation's
few milliseconds — so a run is refused up front when its predicted cost exceeds
`FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS`, with the prediction, the width and the step
count in the message. The cost that dominates at the top of the width range is
holding the state: the model predicts 25.2 s per step at 24 wires before a single
gate is applied, so no 24-wire circuit gets more than two steps, and the layered
ansatz measured here — 71 instructions — is predicted at 119 s per step and gets
none. That is a statement about the
circuit, not about the width: one gate at 24 wires is still under the budget.

And the result is what it is: a loss curve and a set of angles. Whether the run
converged is your reading, not this tool's claim, and the SDK's
`accuracy.metric == "not_measured"` travels with the execution exactly as it
does for a simulation.

## Which contracts this rests on

FlagQuantum publishes a frozen `stable_exports` snapshot (31 names, each with a
named verification test), describes `flagquantum.compiler` as its "stable expert
compiler interface", and lets each package declare its own `__all__`. This
server uses all three tiers, and `tests/test_api_contract.py` pins the members
of each:

| Tier | Surface | Used by |
| --- | --- | --- |
| 1. Frozen snapshot | `Circuit`, `CircuitIR`, `Instruction`, `IR_VERSION`, `ExecutionOptions`, `ExecutionPlan`, `plan`, `Parameter`, `RuntimePolicy`, … | analyze, serialize, deserialize, plan, inspect/bind parameters, train_parameters |
| 2. Documented module | `flagquantum.compiler`: `CouplingMap`, `optimize`, `route_to_topology`, `schedule_layers` | optimize, route, compare, describe layers |
| 3. Public but not frozen | `flagquantum.compiler.openqasm.emit_openqasm`, `flagquantum.compiler.qcis.emit_qcis`, `flagquantum.drawer.draw`, `flagquantum.core.{operator_manifest,gate_info,canonical_opcode}`, `flagquantum.algorithms.{Hamiltonian, pauli_term}` | emit_openqasm, emit_qcis, draw, gate validation, train_parameters |

**Tier 3 is the weakest, and it is not decoration.** Gate validation needs each
gate's wire count and parameter names, and the SDK's operator manifest is the
only authority for that — a caller cannot infer it from the circuit format, and
reimplementing it here would create a second source of truth that drifts. The
manifest is reached through a helper that turns a relocation into a named error
rather than an `AttributeError` inside a tool call, and every tier-3 name is
pinned by a test so a move upstream fails the build instead of failing a user.
The failures that guard against are quiet ones: a weaker gate check accepts a
misspelled parameter and drops it.

The training tool is the one place where the tiers are not the whole story. Its
objective needs two names that are **not** frozen — `Hamiltonian` and
`pauli_term`, both from `flagquantum.algorithms` — and one that is:
`RuntimePolicy` is in the frozen snapshot, but the snapshot fixes only the name.
Whether the Hamiltonian reaches the run is decided by
`RuntimePolicy().observable`, whose default is `"z"`, so
`tests/test_api_contract.py` pins that default too rather than trusting the
snapshot to carry it.

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
