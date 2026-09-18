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
closure works: rebuilding a circuit from the instructions in its own IR through
`Circuit.gate(name, wires, params=..., matrix=...)` reproduces **its instruction
list** byte for byte, matrix gates included.

**Its instruction list, not its whole envelope**, and the difference matters
because the wider claim is the one that sounds better. The replay reproduces the
instructions and the width; every other envelope field — `dtype`, `shape`,
`metadata` — comes from a freshly default-constructed `Circuit`. A
`dtype="complex128"` source is equal in instructions and unequal in everything
else, so "the replay rebuilds the payload byte for byte" is **false in general
and true only of a source that was itself default-constructed**. An earlier
version of this design stated the wider claim, and it passed the test written
against it for exactly that reason.

A parameterized circuit replayed with tensors serializes those tensors as
`$tensor` rather than as the `$parameter` symbol it came from, which is the
difference between a symbol and a value rather than a difference in the
circuit.

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

**A `hamiltonian` argument is ignored unless the policy asks for it.** This is
the finding that decides whether the tool is worth building at all.
`Module(hamiltonian=H)` stores `H` and then does not use it: the value comes from
`RuntimePolicy.observable`, whose default is `"z"` with `observable_wires=(0,)`.
Measured, with a `ry(θ)` circuit and θ = 0.7:

| Hamiltonian | policy | value | correct? |
| --- | --- | --- | --- |
| `1.0·Z` | default | 0.7648 | yes, by accident |
| `1.0·X` | default | 0.7648 | **no** — that is ⟨Z⟩, not ⟨X⟩ = 0.6442 |
| `1.0·Z + 2.0·Z` | default | 0.7648 | **no** — that is ⟨Z⟩, not 3⟨Z⟩ = 2.2945 |
| no Hamiltonian | default | 0.7648 | — |
| `1.0·X` | `observable="hamiltonian"` | 0.6442 | yes |
| `1.0·Z + 0.5·X` | `observable="hamiltonian"` | 1.0870 | yes |
| `2.0·Z − 1.0·X` | `observable="hamiltonian"` | 0.8855 | yes |

So the policy is part of the request, and a design that omitted it would train
⟨Z₀⟩ to convergence and report it as an energy. The earlier version of this
document did exactly that: its four-qubit "TFIM" run reached −1.0000, which is
the minimum of ⟨Z₀⟩ and not a property of the Hamiltonian it named. With the
policy set, the same ansatz reaches −4.7562 against an exact ground energy of
−4.7588.

**It converges, against the objective it was given.** A four-qubit layered
ansatz against a transverse-field Ising Hamiltonian
(`−Σ ZᵢZᵢ₊₁ + Σ Xᵢ`): loss `−2.8185 → −4.7562` in 300 steps, 0.71 s, against an
exact ground energy of `−4.7588`.

**`Module` requires static circuit topology** — it traces the builder with
`make_fx` and refuses a builder that branches on tensor values. A serialized
ansatz satisfies this by construction.

**The objective is fixed.** With `hamiltonian=H` and
`policy=RuntimePolicy(observable="hamiltonian")`, `ExecutionResult.value` *is*
the energy expectation. A JSON interface cannot carry a callable, and it does
not need to: the objective is the Hamiltonian expectation and nothing else.

**Cost is a different order of magnitude from `simulate`.**

Per step, warm, on an `ry`/`cx`/`rz` ansatz against a transverse-field Ising
Hamiltonian (`−Σ ZᵢZᵢ₊₁ + Σ Xᵢ`, one term per wire and one per bond). One layer
of that ansatz has `3n - 1` instructions; `n_instructions` below is the real
count, which for the four-layer rows is `12n - 4`.

| width | layers | instructions | per step | 100 steps |
| --- | --- | --- | --- | --- |
| 4 | 1 | 11 | 1.8 ms | 0.2 s |
| 8 | 1 | 23 | 4.1 ms | 0.4 s |
| 12 | 1 | 35 | 9.0 ms | 0.9 s |
| 16 | 1 | 47 | 85 ms | 8.5 s |
| 16 | 4 | 188 | 147 ms | 15 s |
| 20 | 1 | 59 | 1.29 s | 2.2 min |
| 22 | 1 | 65 | 8.6 s | 14 min |
| 24 | 1 | 71 | 68 s | 1.9 h |

Three costs, and all three are visible in that table. At small widths the per-step
cost is nearly flat and dominated by dispatch. Past 12 qubits the state the SDK
carries takes over and grows as `2 ** n_wires`. Gate count matters too, in both
regimes: at 16 qubits, four layers cost 147 ms/step against one layer's 85 ms.

A first call in a fresh process additionally pays ~0.5 s once, tracing the
builder with `make_fx`. That is why an unpractised measurement looks like a flat
~20 ms/step at small widths: the startup landed on 20 steps.

`simulate_circuit_tool`'s worst case at the existing bound is 2.4 s. Training
at 24 qubits is nearly two hours for a hundred steps. That difference is the
whole reason this design has a budget model.

## Interface

One new tool, one new module.

```
train_parameters_tool(
    circuit: str,
    hamiltonian: Sequence[Mapping[str, Any]],        # required, so no default
    circuit_format: CircuitFormat = "ir",
    values: Mapping[str, float] | None = None,       # initial, defaults to zeros
    steps: int = 50,
    learning_rate: float = 0.1,
) -> dict
```

`hamiltonian` comes before the defaulted arguments because it has no default.
That is a deliberate break with the other tools' `circuit, circuit_format`
opening: putting it after `circuit_format` would give it a default and make it
optional in the schema, and the whole point is that a caller cannot call this
without an objective.

`hamiltonian` takes the **same `terms` shape** `outputs` already uses —
`[{"pauli": "ZZZZ", "coefficient": 1.0}, ...]`, one letter per wire — so the
two are one concept with one spelling, and the validation is shared with
`planning.py` rather than reimplemented. The same shape means the same refusals,
including the one for an all-identity term.

`hamiltonian` is required, and required *in the schema*: it is declared without
a default, which puts it in FastMCP's `required` list, and it is also refused at
run time for callers that reach the module directly. There is no sensible
default — without one the value is not an energy and training it would minimise
an unnamed quantity.

The `RuntimePolicy` is not a caller-facing argument. It is set to
`observable="hamiltonian"` by the tool, always, because that is the only setting
under which the `hamiltonian` argument is read at all. Exposing it would offer a
caller a way to disable the objective they just supplied.

`values` defaults to zeros, and **the result reports the values it started
from**, so a starting point is never implicit.

Returns, in the shape the session that needed it would want:

```json
{
  "status": "success",
  "initial_loss": -2.81850,
  "final_loss": -4.75616,
  "losses": [-2.81850, -3.10044, "..."],
  "completed_steps": 300,
  "parameters": {"t0_0": 1.5733, "u0_3": -0.0945},
  "initial_parameters": {"t0_0": 0.05, "u0_3": 0.05},
  "circuit": {"n_qubits": 4, "n_instructions": 20, "content_hash": "..."},
  "execution": {"execution_path": "local_statevector", "...": "..."}
}
```

`parameters` is written in the `bind_parameters_tool`/`values` spelling, so the
continuation loop is mechanical: call again with `values=parameters`.

**`losses[k]` is the loss the parameters entered step `k+1` with**, because the
SDK evaluates before it updates. So `losses[0]` is the loss at the submitted
`values`, and `losses[-1]` is *not* the loss of the parameters being returned —
it is the loss one update earlier. `final_loss` therefore comes from one extra
forward evaluation at the returned parameters, which is what makes the
continuation exact: feed `parameters` back as `values` and the next call's
`initial_loss` is this call's `final_loss`. Reporting `losses[-1]` as
`final_loss` would cost nothing and quietly break that promise, and a caller
watching a curve would see it step backwards on resume.

## The replay layer (new module `training.py`)

- `parameter_names(ir)` — the circuit's own names, from
  `circuit_from_ir(ir).parameter_names`. Sorted by the SDK, each name once.
- `replay_builder(ir)` — returns the callable `Module` traces. For each
  instruction it calls `Circuit.gate(name, wires, params=..., matrix=...)`,
  substituting `parameters[name][0]` for each symbolic argument and passing
  every other value through unchanged.
- `train_parameters(...)` — the tool body.

The replay carries no numerical logic. It translates a serialized circuit into
the callable shape the SDK asks for, and nothing else.

**It reads the IR's live `instructions`, not `ir.to_dict()`**, and this is the
one place in this design where the intuitive choice is the broken one.
`to_dict()` runs every value through the SDK's encoder: a `Parameter` becomes
`{"$parameter": ...}`, a complex becomes `{"$complex": [...]}`, and a matrix's
entries the same way. Handing those encodings back to `Circuit.gate` builds a
circuit that **serializes byte-for-byte like the original and cannot execute** —
`fq.run` refuses it with `planned execution failed` while the source runs
normally. An earlier version of this design specified `to_dict()` and the replay
it produced was byte-equal to its source, which is what made the defect hard to
see: everything a test might compare said the two circuits were identical.

The live objects are `Parameter`, `ParameterExpression` and real matrices, which
is what the builder needs. A symbolic argument is resolved through the SDK's
public `ParameterExpression.bind`, so the arithmetic stays the SDK's rather than
being reimplemented here.

## Budget model

A prediction decides before any work starts, because the alternative is a
stdio session that hangs for an hour with the client blocked.

```
startup         ≈ 0.5                                          seconds, once
per_step        ≈ max(0.02,
                      2 ** n_wires × 1.5e-6,                  holding the state
                      n_instructions × 2 ** n_wires × 1e-7)   applying gates
predicted_total ≈ startup + steps × per_step
```

Three terms, because cost has three regimes, and each term is a cost that was
measured rather than a fitted fudge. `startup` is the `make_fx` trace, paid once
per process. The other two are paid on every step and are separate terms because
they were measured to be separate: **holding** a state of `2 ** n_wires`
amplitudes costs `1.5e-6` per amplitude, and **applying** gates against it costs
`1e-7` per amplitude per instruction. The `0.02` floor is dispatch, measured at
1.8 ms at 4 qubits and 9 ms at 12, rounded up.

The state term is the one this model first got wrong, and how it got it wrong is
worth recording, because the same mistake is available to anyone re-deriving the
model from this section. The version before this one had only the floor and the
gate term. Every row of the table below over-predicted, so it looked sound — but
every row was one of two kinds, and neither kind could see the missing cost.
Where the width was small (4, 8, 12) the floor swamped both terms, so the row
said nothing about either. Where the width was large (16 and up) the circuit
carried 47 instructions or more, so the gate term dominated and the row said
nothing about the state. No row was large-width *and* few-instruction, which is
exactly where the two-term model broke: at 24 wires with a single gate, one step
costs 12.5 s, 7.5× what one instruction explains, with the other 87% being the
cost of holding the state at all. The model admitted that call for 35 steps at a
predicted 59.2 s, and it took 438 s. **A calibration table that omits an axis
cannot constrain the model along it.** The shipped test's table is sixteen
points rather than ten, and the six it gained — `(16, 3)`, `(20, 1)`, `(20, 3)`,
`(22, 1)`, `(24, 1)`, `(24, 4)` — are the ones the state term is calibrated
against.

Checked against every measurement in the table above, plus the cold first call.
Every row is unchanged from the two-term model's predictions, which is the point
worth noticing: this table could not tell the two models apart either, because
no row of it is large-width *and* few-instruction. It is kept as the check that
neither model under-predicts the circuits this design measured.

| width | layers | steps | predicted | measured | over by |
| --- | --- | --- | --- | --- | --- |
| 4 | 1 | 20, cold | 0.90 s | 0.50 s | 1.8× |
| 4 | 1 | 20 | 0.90 s | 0.036 s | 25× |
| 8 | 1 | 20 | 0.90 s | 0.082 s | 11× |
| 12 | 1 | 20 | 0.90 s | 0.18 s | 5.0× |
| 12 | 4 | 20 | 1.65 s | 0.31 s | 5.3× |
| 16 | 1 | 20 | 6.66 s | 1.70 s | 3.9× |
| 16 | 4 | 20 | 25.1 s | 2.93 s | 8.6× |
| 20 | 1 | 20 | 124 s | 25.8 s | 4.8× |
| 22 | 1 | 2 | 55.1 s | 17.1 s | 3.2× |
| 24 | 1 | 1 | 119.6 s | 68.5 s | 1.7× |

Every row over-predicts, which is the direction that matters: a refusal that
sometimes declines work that would have fitted is a better failure than a call
that blocks for an hour. The margins run from 1.7× to 25×, and the shape of that
spread is honest — the model is a straight line through a curve, it is closest
at the top where the state cost dominates and the constants were calibrated, and
farthest at the bottom where the floor it uses (20 ms) is ten times the dispatch
it actually measured (1.8 ms). A tighter floor would fit the small widths better
and buy nothing: at a 60 s budget the floor is what lets a 4-qubit caller ask for
three thousand steps instead of thirty thousand, and three thousand is already
past the number anyone will read.

The same straight line is why the last row's margin is the thinnest. The
per-state cost is not constant across the whole range — it falls from 1.0e-5 per
instruction-state at 4 qubits to 2.1e-8 at 20, then rises again to 5.7e-8 at 24
as the state (134 MB of `complex64`) stops fitting where it used to. A single
coefficient cannot follow that, so `1e-7` is chosen to clear the highest point
rather than the average one: over the thirteen calibration points the floor does
not govern, it over-predicts between 1.9× and 8.4×. An earlier draft of this
model used `3e-8`, which was calibrated at 20 qubits and under-predicted 24 by
2.4×.

A still earlier draft had only a `2 ** n_wires` term and a `0.02` floor,
calibrated from cold measurements where the one-time trace had been divided
across the steps. It under-predicted at 16 qubits, and its stated table did not
reproduce: re-measuring warm gave 4 ms/step at 8 qubits where the draft said 20.

If `predicted_total > FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS` (default 60), the tool
refuses and names the prediction, the width, the steps, and the variable that
raises the bound.

At the top of the width range the state term alone is 25.2 s per step, so a
24-wire call is admitted for at most two steps; the one-layer ansatz at that
width, 71 instructions, predicts 119.1 s per step and is admitted for none. That
is worth stating plainly rather than leaving a caller to discover it from a wall
of arithmetic — but it is a statement about the *ansatz*, not about the width. A
one-gate circuit at 24 wires is under the bound and runs. An earlier draft of
this paragraph claimed the refusal at 24 qubits was total and that twenty-two
was the widest width admitting a run, both of which were false by arithmetic
alone: the 68 s and 119.6 s figures are properties of the 71-instruction circuit
the design was calibrated on, not of the width.

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
- **the Hamiltonian reaches the objective**: on a circuit whose ⟨Z₀⟩ optimum and
  whose Hamiltonian optimum differ, training moves the energy toward the
  Hamiltonian's. This is the test that would have caught the policy trap, and it
  is the one test here that a plausible implementation fails.
- training reduces the loss on a circuit whose optimum is known
- the returned `parameters` fed back as `values` continues rather than restarts
- each refusal above, asserting on the message and not only the code
- the budget refusal fires **before** the run: assert the wall clock stays small
- every new assertion mutation-tested, per AGENTS.md

## The dependencies this needs, and why they are allowed

Two SDK objects are outside the frozen snapshot.

`Module(hamiltonian=...)` takes a `flagquantum.algorithms.Hamiltonian`. That
class:

- lives in `flagquantum.algorithms`, which declares it in `__all__`;
- is **not** named in any capability's `public_apis`;
- is **not** reachable as `flagquantum.Hamiltonian`.

`RuntimePolicy` is reachable as `flagquantum.RuntimePolicy` but is likewise not
in `stable_exports` and not named in any capability's `public_apis`. Without it
the Hamiltonian is ignored, so it is not optional.

Both are tier 3: public, not frozen. AGENTS.md rule 3 admits tier 3 when there
is a reason and the names are pinned by a test, which is the situation here —
there is no other way to express the objective, and no other way to make the SDK
read it. The names go into the tier-3 table in `tests/test_api_contract.py` in
the same change.

`torch` is reached through the SDK rather than imported directly — the server
declares two dependencies and `torch` is not one of them — even though
`fq.train` takes a `torch.optim.Optimizer` and there is no way to call it
without one.

**This is an upstream gap worth recording.** `Module`'s own signature references
a type from a module the capability manifest does not declare, so a caller
following the manifest cannot construct the objective for the capability the
manifest advertises. `run_vqe`, `hardware_efficient_ansatz` and
`transverse_field_ising` are in the same position.

The larger gap is the policy. `Module(hamiltonian=H)` accepts `H`, stores it,
exposes it as `.hamiltonian`, and then evaluates something else, because which
observable a `Module` measures is governed by a separate object whose default is
`⟨Z₀⟩`. Nothing warns. A caller who reads `train`'s docstring — whose example
passes no `hamiltonian` at all — builds a variational loop that converges
cleanly on the wrong quantity. This server cannot fix that; it can decline to
reproduce it, and the test that the Hamiltonian reaches the objective is what
makes that a property rather than an intention.

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
