"""Train a circuit's parameters against a Pauli-sum energy, in one call.

``fq.Module`` and ``fq.train`` want a Python callable rather than JSON, which
looks like a dead end for a server whose every input is JSON. It is not: a
serialized circuit carries every instruction it was built from, and
``Circuit.gate`` is the single primitive that rebuilds them — built-in gates,
arbitrary matrices and channels all go through it. So the replay below is a loop
and not a gate table, and it carries no numerical logic. It translates a
serialized circuit into the callable shape the SDK asks for, and nothing else.

Three facts about the SDK shape this module, all measured rather than read:

* ``Module(hamiltonian=H)`` stores ``H`` and does not evaluate it. Which
  observable a module measures comes from its ``RuntimePolicy``, whose default is
  ``⟨Z₀⟩``. Every ``Module`` built here passes
  ``policy=RuntimePolicy(observable="hamiltonian")``, and
  ``test_the_hamiltonian_reaches_the_objective`` is what holds that.
* A ``{"$parameter": ...}`` marker is a *serialization* encoding. Handed to the
  Python API it is stored as an opaque dict and the circuit reports itself as
  unparameterized, which is the silent failure ``circuits.py`` already refuses at
  the JSON boundary. Names are therefore read through ``circuit_from_ir``, which
  decodes the markers properly.
* ``Circuit.gate`` takes its arguments as a ``params=`` mapping, not
  positionally, and the argument names are the SDK's own gate manifest.

Training is the only operation here whose cost is not bounded by the size of its
input, so a prediction decides before any work starts. See ``predict_seconds``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from flagquantum_mcp_server._bridge import load_sdk
from flagquantum_mcp_server.circuits import circuit_from_ir

# The one Python name a parameter value may travel under when it is a symbol
# rather than a number. Its presence is what makes the argument a tensor.
PARAMETER_MARKER = "$parameter"


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
    one-element tensor per name — and returns a live ``Circuit``. Instructions
    are read from ``ir.to_dict()``, where a ``Parameter`` has become the marker a
    caller sees over the wire, rather than from the live objects, whose ``params``
    hold SDK types a JSON tool has no business inspecting.

    Args:
        ir: A validated ``CircuitIR``.

    Returns:
        A callable taking ``{name: tensor}`` and returning a ``Circuit``.
    """
    instructions = tuple(ir.to_dict()["instructions"])
    n_wires = int(ir.n_wires)

    def build(parameters: Mapping[str, Any]) -> Any:
        circuit = load_sdk().Circuit(n_wires)
        for instruction in instructions:
            arguments = {
                str(key): _argument(value, parameters)
                for key, value in (instruction.get("params") or {}).items()
            }
            circuit.gate(
                str(instruction["opcode"]),
                [int(wire) for wire in instruction["wires"]],
                params=arguments or None,
                matrix=instruction.get("matrix"),
            )
        return circuit

    return build


def _argument(value: Any, parameters: Mapping[str, Any]) -> Any:
    """Resolve one serialized gate argument against the parameter mapping.

    A symbol becomes the tensor ``Module`` supplied for that name. Every other
    value — a bound number, a ``$tensor``, a ``$complex``, a ``$expression`` —
    passes through untouched, because the SDK's own decoder built it and this
    module has no business reinterpreting it.

    ``[0]`` indexes the one-element group ``Module`` creates per name. A group of
    any other size would be a name bound to a vector, which this tool never asks
    for.

    Args:
        value: The serialized argument.
        parameters: The mapping ``Module`` handed the builder.

    Returns:
        The argument to pass to ``Circuit.gate``.
    """
    symbol = value.get(PARAMETER_MARKER) if isinstance(value, Mapping) else None
    if isinstance(symbol, str):
        return parameters[symbol][0]
    return value
