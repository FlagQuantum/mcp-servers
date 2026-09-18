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


# The budget model's three constants. Calibrated from warm per-step measurements
# at 4, 8, 12, 13, 14, 15, 16, 20, 22 and 24 qubits; the table is in the design
# document and every point is pinned by a test. The model over-predicts at all
# of them, by 1.7x at the worst.
STARTUP_SECONDS = 0.5
MIN_STEP_SECONDS = 0.02
STEP_COST_COEFFICIENT = 1e-7


def predict_seconds(ir: Any, steps: int) -> float:
    """Predict how long ``steps`` updates will take, in seconds.

    Two terms, because cost has two regimes. ``STARTUP_SECONDS`` is the one-time
    ``make_fx`` trace ``Module`` performs when it first compiles the builder.
    The rest is work: gates applied against a state of ``2 ** n_wires``
    amplitudes, so it is the product of the two, floored at the dispatch cost
    that dominates at small widths.

    The floor is generous on purpose. It is 20 ms where 1.8 ms was measured at
    four qubits, which at the default 60-second budget is the difference between
    a caller being allowed three thousand steps and thirty thousand. Three
    thousand is already more than anyone reads, and a model that is an upper
    bound everywhere is worth more than one that is tight at the bottom.

    A single coefficient cannot follow the true curve, which falls from 1.0e-5
    per instruction-state at four qubits to 2.1e-8 at twenty and rises again to
    5.7e-8 at twenty-four as the state stops fitting where it used to. The
    coefficient clears the highest point rather than the average one, which
    makes this 3-5x pessimistic through the middle.

    Args:
        ir: A validated ``CircuitIR``.
        steps: The number of updates requested.

    Returns:
        A prediction in seconds. An estimate rather than a measurement of the
        caller's machine.
    """
    per_step = max(
        MIN_STEP_SECONDS,
        len(ir.instructions) * 2.0 ** int(ir.n_wires) * STEP_COST_COEFFICIENT,
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
