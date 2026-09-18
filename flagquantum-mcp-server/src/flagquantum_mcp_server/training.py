"""Train a circuit's parameters against a Pauli-sum energy, in one call.

``fq.Module`` and ``fq.train`` want a Python callable rather than JSON, which
looks like a dead end for a server whose every input is JSON. It is not: a
serialized circuit carries every instruction it was built from, and
``Circuit.gate`` is the single primitive that rebuilds them — built-in gates and
arbitrary matrices both go through it. So the replay below is a loop and not a
gate table, and it carries no numerical logic. It translates a serialized circuit
into the callable shape the SDK asks for, and nothing else.

A channel is the one instruction it does not rebuild faithfully. Flagging an
instruction as a channel is ``instruction.metadata["is_channel"]``, and
``Circuit.gate`` takes no ``metadata``, so the flag cannot be carried. Measured,
the consequence is smaller than it sounds: a channel circuit fails ``fq.run`` on
this path both before and after the replay, so no number a caller sees changes —
but the replay is not faithful there and this docstring will not claim it is.

Four facts about the SDK shape this module, all measured rather than read:

* ``Module(hamiltonian=H)`` stores ``H`` and does not evaluate it. Which
  observable a module measures comes from its ``RuntimePolicy``, whose default is
  ``⟨Z₀⟩``. Every ``Module`` built here passes
  ``policy=RuntimePolicy(observable="hamiltonian")``, and
  ``test_the_hamiltonian_reaches_the_objective`` is what holds that.
* A ``{"$parameter": ...}`` marker is a *serialization* encoding. Handed to the
  Python API it is stored as an opaque dict and the circuit reports itself as
  unparameterized, which is the silent failure ``circuits.py`` already refuses at
  the JSON boundary. Names are therefore read through ``circuit_from_ir``, which
  decodes the markers properly. The same trap covers the instruction list itself:
  see ``replay_builder``, which reads the IR's live objects and never
  ``to_dict()``.
* ``Circuit.gate`` takes its arguments as a ``params=`` mapping, not
  positionally, and the argument names are the SDK's own gate manifest.
* A parameter can also sit inside a ``ParameterExpression`` — ``2.0 * t0``.
  ``parameter_names`` reports the name either way, so a replay that substitutes
  only a top-level ``Parameter`` advertises a trainable group it never applies,
  and stays byte-equal to its source while doing it.

Training is the only operation here whose cost is not bounded by the size of its
input, so a prediction decides before any work starts. See ``predict_seconds``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from flagquantum_mcp_server._bridge import load_sdk
from flagquantum_mcp_server.circuits import circuit_from_ir


def parameter_names(ir: Any) -> tuple[str, ...]:
    """Return the circuit's parameter names, in the SDK's order.

    Read through ``circuit_from_ir`` rather than parsed from the instruction
    list, because that is the function that knows a ``$parameter`` marker is a
    symbol. The order is the SDK's — sorted, each name once however many gates
    carry it — and nothing here depends on it, because ``Module`` binds by name.

    Args:
        ir: A validated ``CircuitIR``.

    Returns:
        The parameter names.
    """
    return tuple(str(name) for name in circuit_from_ir(ir).parameter_names)


def replay_builder(ir: Any) -> Callable[[Mapping[str, Any]], Any]:
    """Return the callable ``fq.Module`` traces, rebuilding this circuit.

    The returned builder takes the parameter mapping ``Module`` hands it — one
    one-element tensor per name — and returns a live ``Circuit``.

    Instructions come from the IR's own ``instructions``, not from
    ``to_dict()``. The two differ in a way that is easy to miss and was
    measured: ``to_dict()`` runs every value through the SDK's encoder, turning
    a parameter into a ``{"$parameter": ...}`` marker, a complex into
    ``{"$complex": [...]}`` and a matrix's entries the same way. Handing those
    encodings back to ``Circuit.gate`` builds a circuit that serializes
    byte-for-byte like the original and cannot execute — ``fq.run`` refuses it
    with ``planned execution failed`` while the source runs normally. The live
    objects are ``Parameter``, ``ParameterExpression`` and real matrices, which
    is what the builder needs and what a JSON tool has no reason to reassemble
    from markers.

    Args:
        ir: A validated ``CircuitIR``.

    Returns:
        A callable taking ``{name: tensor}`` and returning a ``Circuit``.
    """
    instructions = tuple(ir.instructions)
    n_wires = int(ir.n_wires)

    def build(parameters: Mapping[str, Any]) -> Any:
        sdk = load_sdk()
        circuit = sdk.Circuit(n_wires)
        for instruction in instructions:
            arguments = {
                str(key): _argument(value, parameters, sdk)
                for key, value in instruction.params.items()
            }
            circuit.gate(
                str(instruction.name),
                [int(wire) for wire in instruction.wires],
                params=arguments or None,
                matrix=instruction.matrix,
            )
        return circuit

    return build


# The budget model's constants. Calibrated from warm per-step measurements at 4,
# 8, 12, 13, 14, 15, 16, 20, 22 and 24 qubits, over circuits from 1 to 188
# instructions; the table is in the design document and every point is pinned by
# a test. The model over-predicts at all sixteen points, by 1.9x at the worst.
#
# Three costs, each of them one that was measured rather than assumed:
STARTUP_SECONDS = 0.5
MIN_STEP_SECONDS = 0.02  # dispatch, for circuits too small for either term below
STATE_COST = 1.5e-6  # per step per amplitude: holding the state
GATE_COST = 1e-7  # per step per amplitude per instruction: applying gates


def predict_seconds(ir: Any, steps: int) -> float:
    """Predict how long ``steps`` updates will take, in seconds.

    Three terms, because cost has three regimes. ``STARTUP_SECONDS`` is the
    one-time ``make_fx`` trace ``Module`` performs when it first compiles the
    builder. The other two are paid on every step: ``STATE_COST`` per amplitude
    for holding a state of ``2 ** n_wires``, and ``GATE_COST`` per amplitude per
    instruction for applying gates against it. They are separate terms because
    they were measured to be: at twenty-four wires a single gate costs 12.5 s per
    step, which is 7.5x what one instruction explains, and a single
    width-independent floor could not see that cost at all.

    The floor sits under both. It is generous on purpose: 20 ms where 1.8 ms was
    measured at four qubits, which at the default 60-second budget is the
    difference between a caller being allowed three thousand steps and thirty
    thousand. Three thousand is already more than anyone reads, and a model that
    is an upper bound everywhere is worth more than one that is tight at the
    bottom.

    A single coefficient cannot follow the true curve either, which falls from
    1.0e-5 per instruction-state at four qubits to 2.1e-8 at twenty and rises
    again to 5.7e-8 at twenty-four as the state stops fitting where it used to.
    The first two are measured points in the sixteen-row calibration table; the
    third is the design document's own 24-wire, 71-instruction measurement,
    68.5 s per step, and is not one of the sixteen. ``GATE_COST`` clears the
    highest point rather than the average one.

    Where that leaves the margin, over the sixteen points it was calibrated at:
    11x at four qubits, 4.9x at eight and 2.2x at twelve, the three widths where
    the floor is doing all the work; between 1.9x and 8.4x at the other thirteen,
    thinnest at twenty-four wires with four instructions and thickest at sixteen
    wires with four layers. The thinnest matters least, because the budget admits
    almost nothing there: the state term alone is 25.2 s per step at twenty-four
    wires, so no circuit at that width gets more than two steps, and the full
    ansatz there, 119.1 s per step, gets none at all. Twenty-two wires costs
    6.3 s per step, and a one-gate circuit at that width gets nine.

    Args:
        ir: A validated ``CircuitIR``.
        steps: The number of updates requested.

    Returns:
        A prediction in seconds. An estimate rather than a measurement of the
        caller's machine.
    """
    per_step = max(
        MIN_STEP_SECONDS,
        2.0 ** int(ir.n_wires) * STATE_COST,  # holding the state
        len(ir.instructions) * 2.0 ** int(ir.n_wires) * GATE_COST,  # applying gates
    )
    return STARTUP_SECONDS + steps * per_step


def _argument(value: Any, parameters: Mapping[str, Any], sdk: Any) -> Any:
    """Resolve one live gate argument against the parameter mapping.

    A ``Parameter`` becomes the tensor ``Module`` supplied for that name. A
    ``ParameterExpression`` — ``2.0 * t0`` — is resolved through the SDK's own
    public ``bind``, so the arithmetic stays the SDK's rather than being
    reimplemented here; this module carries no numerical logic and should not
    acquire any.

    Every other value is already a number, a complex or an object the SDK built,
    and passes through untouched.

    ``[0]`` indexes the one-element group ``Module`` creates per name. A group of
    any other size would be a name bound to a vector, which this tool never asks
    for; the assertion that it is one element belongs at the construction site.

    Args:
        value: The live argument from an instruction.
        parameters: The mapping ``Module`` handed the builder.
        sdk: The loaded ``flagquantum`` module, for the two type checks.

    Returns:
        The argument to pass to ``Circuit.gate``.
    """
    if isinstance(value, sdk.Parameter):
        return parameters[value.name][0]
    if isinstance(value, sdk.ParameterExpression):
        return value.bind({name: group[0] for name, group in parameters.items()})
    return value
