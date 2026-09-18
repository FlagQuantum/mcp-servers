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
closure works, and the replay is exact: rebuilding a circuit from its IR
through `Circuit.gate(name, wires, params=..., matrix=...)` reproduces the
source payload byte for byte, matrix gates included.

`Circuit.gate` is the single primitive — built-in gates, arbitrary matrices and
channels all go through it, so the replay layer is a loop and not a gate table.

**`Module` accepts named parameters.** `parameters={"alpha": 1}` and
`init={"alpha": 0.3}` bind by name, matching the circuit's own `$parameter`
names. The replay does not have to maintain a positional order.

**It converges.** A four-qubit layered ansatz against a transverse-field Ising
Hamiltonian: loss `+0.9527 → −1.0000` in 100 steps, 0.10 s.

**`Module` requires static circuit topology** — it traces the builder with
`make_fx` and refuses a builder that branches on tensor values. A serialized
ansatz satisfies this by construction.

**The objective is fixed.** With `hamiltonian=H`, `ExecutionResult.value` *is*
the energy expectation. A JSON interface cannot carry a callable, and it does
not need to: the objective is the Hamiltonian expectation and nothing else.

**Cost is a different order of magnitude from `simulate`.**

| width | per step | 100 steps |
| --- | --- | --- |
| 8 qubits | ~20 ms | 2 s |
| 16 qubits | ~32 ms | 3.2 s |
| 20 qubits | ~300 ms | 30 s |
| 24 qubits | ~9.1 s | 15 min |

`simulate_circuit_tool`'s worst case at the existing bound is 2.4 s. Training
at the same width is 15 minutes and up. That difference is the whole reason
this design has a budget model.

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

- `parameter_names_in_order(ir)` — first appearance order, stable.
- `replay_builder(ir, names)` — returns the callable `Module` traces. For each
  instruction it calls `Circuit.gate(name, wires, params=..., matrix=...)`,
  substituting each parameter's tensor by name.
- `train_parameters(...)` — the tool body.

The replay carries no numerical logic. It translates a serialized circuit into
the callable shape the SDK asks for, and nothing else.

## Budget model

A prediction decides before any work starts, because the alternative is a
stdio session that hangs for fifteen minutes with the client blocked.

```
per_step        ≈ max(0.02, 2 ** n_wires × 1e-6)   seconds
predicted_total ≈ steps × per_step
```

Two terms, because cost has two regimes. The `2 ** n_wires` term is the state
the SDK holds, and the constant is calibrated from the measurements above,
where `0.29–0.54 µs` per state per step was observed — rounded **up** to `1e-6`
so it over-predicts. The `0.02` floor is the fixed per-step overhead that
dominates at small widths, where the observed ~20 ms/step was independent of
width between 8 and 12 qubits.

Checked against every measurement in the table:

| width | steps | predicted | measured | direction |
| --- | --- | --- | --- | --- |
| 8 | 50 | 1.0 s | ~1 s | close |
| 16 | 100 | 6.6 s | 3.2 s | over |
| 20 | 100 | 105 s | 30 s | over |
| 24 | 100 | 1677 s | 915 s | over |

It over-predicts everywhere, which is the direction that matters: a refusal
that sometimes declines work that would have fit is a better failure than a
call that blocks for an hour. A first draft of this model had only the
`2 ** n_wires` term, which under-predicted at 8 qubits by two orders of
magnitude — harmless for the decision at a 60 s budget, but wrong as a stated
model, and this design would have called it conservative when it was not.

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

- the replay round-trips a circuit through `Module` unchanged, matrix gate included
- training reduces the loss on a circuit whose optimum is known
- the returned `parameters` fed back as `values` continues rather than restarts
- each refusal above, asserting on the message and not only the code
- the budget refusal fires **before** the run: assert the wall clock stays small
- the bound is read per call, so a deployment can raise it
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
