# Training a circuit's parameters — design

Status: proposed
Date: 2026-09-18

## The problem this solves

Four recorded agent sessions drove this server through the same VQE task. The
fourth produced an energy and said what the next step cost:

> this server has no optimizer or gradient interface at all ... I called
> `simulate_circuit_tool` about 50 times across 7 ansatz families ... **this is
> not a converged VQE, it is a manual scan.**

The scanning is not a documentation failure. The server can evaluate an energy
— that landed in 0.2.0 — and has no way to improve one, so the only search
available to a caller is blind enumeration. Fifty calls to approximate what a
gradient step does exactly.

FlagQuantum supports this already, at its highest maturity level:

```toml
[capabilities.local_statevector]
title            = "Local statevector simulation and training"
user_goals       = [..., "Train a parameterized quantum circuit",
                         "Run local VQE and quantum machine learning"]
public_apis      = ["Circuit", "run", "Module", "train"]
gradient_support = "exact"
level            = "production_supported"
```

All four declared APIs are in the frozen `stable_exports` snapshot — tier 1.
AGENTS.md rule 7 is satisfied by the manifest itself.

## What was verified before designing

Every claim below was measured against `flagquantum` 0.2.0 in this environment,
not read off a docstring.

**A serialized circuit can drive `Module` and `train`.** They want a Python
callable, not JSON, which looked like a dead end. Replaying the IR into a
closure works, and the replay is faithful: rebuilding a circuit from the
instructions in its own IR through
`Circuit.gate(name, wires, params=..., matrix=...)` reproduces that payload byte
for byte, matrix gates included. A parameterized circuit replayed with tensors
serializes those tensors as `$tensor` rather than as the `$parameter` symbol it
came from, which is the difference between a symbol and a value rather than a
difference in the circuit.

`Circuit.gate` is the single primitive — built-in gates, arbitrary matrices and
channels all go through it, so the replay layer is a loop and not a gate table.
It takes its arguments as a `params=` mapping, not positionally:
`gate("ry", [0], params={"theta": t})`, with the argument names the SDK's own
gate manifest publishes.

**`Module` accepts named parameters.** `parameters={"alpha": 1}` declares a
one-element parameter *group* called `alpha`, and `init={"alpha": 0.3}` gives it
its starting value. The builder receives a mapping and reads
`parameters["alpha"][0]`. Names are the circuit's own `$parameter` names, so the
replay does not have to maintain a positional order.

**The names come from the SDK, and they are sorted.** `Circuit.from_ir(ir)`
decodes the `$parameter` markers into live symbols and
`circuit.parameter_names` lists them — alphabetically, each name once however
many gates carry it. There is no need to compute an order here, and no need to
parse instructions looking for markers.

**A `{"$parameter": ...}` marker is not a symbol in the Python API.** Handed to
`Circuit.ry(theta={"$parameter": "t0"})` it is stored as an opaque dict and the
circuit reports `is_parameterized() == False`; the marker is a *serialization*
encoding that `Circuit.from_ir` decodes and the Python constructor does not.
This is the same silent failure `circuits.py` already refuses at the JSON
boundary, and it is why the replay reads names through `circuit_from_ir` rather
than from the raw instruction list.

**It converges.** A four-qubit layered ansatz against a transverse-field Ising
Hamiltonian: loss `+0.9527 → −1.0000` in 100 steps, 0.10 s.

**`Module` requires static circuit topology** — it traces the builder with
`make_fx` and refuses a builder that branches on tensor values. A serialized
ansatz satisfies this by construction.

**The objective is fixed.** With `hamiltonian=H`, `ExecutionResult.value` *is*
the energy expectation. A JSON interface cannot carry a callable, and it does
not need to: the objective is the Hamiltonian expectation and nothing else.

**Cost is a different order of magnitude from `simulate`.**

Per step, warm, on an `ry`/`cx`/`rz` ansatz with one Hamiltonian term per wire.
One layer of that ansatz has `3n - 1` instructions.

| width | instructions | per step | 100 steps |
| --- | --- | --- | --- |
| 4 qubits | 11 | 2.1 ms | 0.2 s |
| 8 qubits | 23 | 4.1 ms | 0.4 s |
| 12 qubits | 35 | 4.2–7.2 ms | 0.7 s |
| 16 qubits | 47 | 32–38 ms | 3.5 s |
| 20 qubits | 59 | 583 ms | 58 s |
| 24 qubits | 71 | 23 s | 38 min |

Two costs, and both are visible in that table. At small widths the per-step
cost is nearly flat and dominated by dispatch — ~2 ms whatever the width. Past
16 qubits the state the SDK carries takes over and grows as `2 ** n_wires`. Gate
count matters too, in both regimes: at 16 qubits, four layers cost 116 ms/step
against one layer's 38 ms.

A first call in a fresh process additionally pays ~0.5 s once, tracing the
builder with `make_fx`. That is why an unpractised measurement looks like a flat
~20 ms/step at small widths: the startup landed on 20 steps.

`simulate_circuit_tool`'s worst case at the existing bound is 2.4 s. Training
at 24 qubits is 38 minutes. That difference is the whole reason this design has
a budget model.

## Interface

One new tool, one new module.

```
train_parameters_tool(
    circuit: str,
    circuit_format: CircuitFormat = "ir",
    hamiltonian: Sequence[Mapping[str, Any]],        # required
    values: Mapping[str, float] | None = None,       # initial, defaults to zeros
    steps: int = 50,
    learning_rate: float = 0.1,
) -> dict
```

`hamiltonian` takes the **same `terms` shape** `outputs` already uses —
`[{"pauli": "ZZZZ", "coefficient": 1.0}, ...]`, one letter per wire — so the
two are one concept with one spelling, and the builder is shared with
`planning.py` rather than reimplemented.

`hamiltonian` is required. There is no sensible default: without one the value
is not an energy and training it would minimise an unnamed quantity.

`values` defaults to zeros, and **the result reports the values it started
from**, so a starting point is never implicit.

Returns, in the shape the session that needed it would want:

```json
{
  "status": "success",
  "initial_loss": 0.9527,
  "final_loss": -0.9996,
  "losses": [0.9527, 0.9004, "..."],
  "completed_steps": 50,
  "parameters": {"t0": 0.31, "u0": -1.2},
  "initial_parameters": {"t0": 0.0, "u0": 0.0},
  "circuit": {"n_qubits": 4, "n_instructions": 11, "content_hash": "..."},
  "execution": {"execution_path": "local_statevector", "...": "..."}
}
```

`parameters` is written in the `bind_parameters_tool`/`values` spelling, so the
continuation loop is mechanical: call again with `values=parameters`.

## The replay layer (new module `training.py`)

- `parameter_names(ir)` — the circuit's own names, from
  `circuit_from_ir(ir).parameter_names`. Sorted by the SDK, each name once.
- `replay_builder(ir, names)` — returns the callable `Module` traces. For each
  instruction it calls `Circuit.gate(name, wires, params=..., matrix=...)`,
  substituting `parameters[name][0]` for each symbolic argument and passing
  every other value through unchanged.
- `train_parameters(...)` — the tool body.

The replay carries no numerical logic. It translates a serialized circuit into
the callable shape the SDK asks for, and nothing else. It reads the instruction
list from `ir.to_dict()`, which is where a `Parameter` becomes the marker a
caller sees over the wire, rather than from the live objects, whose `params`
hold SDK types a JSON tool has no business inspecting.

## Budget model

A prediction decides before any work starts, because the alternative is a
stdio session that hangs for fifteen minutes with the client blocked.

```
startup         ≈ 0.5                                     seconds, once
per_step        ≈ max(0.01, n_instructions × 2 ** n_wires × 3e-8)
predicted_total ≈ startup + steps × per_step
```

Two terms, because cost has two regimes, and each term is the cost it models
rather than a fitted fudge. `startup` is the `make_fx` trace, paid once per
process. `per_step` is work per step: gates applied against a state of
`2 ** n_wires` amplitudes, so it is the product of the two. The `0.01` floor is
dispatch — ~2 ms measured, rounded up — which is what dominates at small widths.
The constant `3e-8` is the only calibrated number: the largest measured point is
71 instructions × 2²⁴ states at 23 s per step, which gives 1.9e-8, rounded up to
3e-8.

Checked against every measurement above, and against the cold first call:

| width | layers | steps | predicted | measured | over by |
| --- | --- | --- | --- | --- | --- |
| 4 | 1 | 20, cold | 0.70 s | 0.50 s | 1.4× |
| 4 | 1 | 20 | 0.70 s | 0.04 s | 17× |
| 8 | 1 | 20 | 0.70 s | 0.08 s | 8.5× |
| 12 | 1 | 20 | 0.70 s | 0.14 s | 4.9× |
| 12 | 4 | 20 | 0.84 s | 0.25 s | 3.3× |
| 16 | 1 | 20 | 2.35 s | 0.76 s | 3.1× |
| 16 | 4 | 20 | 7.89 s | 2.31 s | 3.4× |
| 20 | 1 | 20 | 37.6 s | 11.7 s | 3.2× |
| 24 | 1 | 2 | 72.0 s | 46.0 s | 1.6× |

Every row over-predicts, which is the direction that matters: a refusal that
sometimes declines work that would have fit is a better failure than a call that
blocks for an hour. The worst case for the decision is the last row, where the
margin is 1.6× rather than 3× — at 24 qubits the `0.5 s` startup is noise
against 71 s of work, so the whole prediction rests on the `3e-8` coefficient,
and the two-step measurement it was calibrated from is the one row with no
repetition behind it.

A first draft of this model had only a `2 ** n_wires` term and a `0.02` floor,
calibrated from cold measurements where the one-time trace had been divided
across the steps. It under-predicted at 16 qubits, and its stated table did not
reproduce: re-measuring warm gave 4 ms/step at 8 qubits where the draft said 20.
The model above is the one that survives being checked line by line.

If `predicted_total > FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS` (default 60), the tool
refuses and names the prediction, the width, the steps, and the variable that
raises the bound.

The description says this is an estimate rather than a measurement of the
caller's machine.

## Refusals

Each is a case where accepting would produce a plausible-looking answer.

| input | why it is refused |
| --- | --- |
| a circuit with no parameters | there is nothing to train; the loss would be constant |
| no `hamiltonian` | without one the value is not an energy |
| malformed `hamiltonian` terms | reuses the `terms` builder's refusals, naming the term index |
| `values` naming an unknown parameter | a typo would otherwise be silently ignored and the parameter left at zero |
| `values` missing a parameter the circuit has | the rest would silently train from zeros |
| `steps < 1` | a loop that cannot improve anything still reports success |
| predicted cost over the budget | see above |

## Tests

- the replay rebuilds a numeric circuit byte for byte, matrix gate included
- the replayed circuit evaluates to the same energy as the source circuit bound
  with the same numbers
- training reduces the loss on a circuit whose optimum is known
- the returned `parameters` fed back as `values` continues rather than restarts
- each refusal above, asserting on the message and not only the code
- the budget refusal fires **before** the run: assert the wall clock stays small
- every new assertion mutation-tested, per AGENTS.md

## The dependency this needs, and why it is allowed

`Module(hamiltonian=...)` takes a `flagquantum.algorithms.Hamiltonian`. That
class:

- lives in `flagquantum.algorithms`, which declares it in `__all__`;
- is **not** named in any capability's `public_apis`;
- is **not** reachable as `flagquantum.Hamiltonian`.

So it is tier 3: public, not frozen. AGENTS.md rule 3 admits tier 3 when there
is a reason and the names are pinned by a test, which is the situation here —
there is no other way to express the objective. The names go into the tier-3
table in `tests/test_api_contract.py` in the same change.

**This is an upstream gap worth recording.** `Module`'s own signature references
a type from a module the capability manifest does not declare, so a caller
following the manifest cannot construct the objective for the capability the
manifest advertises. `run_vqe`, `hardware_efficient_ansatz` and
`transverse_field_ising` are in the same position.

## What this deliberately does not do

- **No noise model.** Training against a noisy objective needs a `NoiseModel`,
  which is a live SDK object rather than a serializable value.
- **No distributed training.** `sharded_statevector_training` is
  `production_supported`, but it needs multiple ranks and this server is one
  process on one machine.
- **No optimizer choice.** Adam at a caller-set learning rate. A
  `torch.optim.Optimizer` is a live object and the caller cannot build one
  through JSON; exposing a menu of names would be inventing an interface the
  SDK does not have.
- **No convergence claim.** The result reports losses and parameters. Whether
  they are converged is the caller's reading, and the SDK's own
  `accuracy.metric == "not_measured"` travels with the execution.
