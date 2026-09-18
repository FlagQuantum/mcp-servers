"""Inspect and bind circuit parameters.

A variational circuit carries symbols rather than numbers. A symbol is written
``{"theta": {"$parameter": "theta"}}`` in either input format; a plain number in
that slot is a bound angle rather than a symbol, and a bare string is neither,
so it is rejected at the input boundary (see
:func:`flagquantum_mcp_server.circuits._check_parameter_values`).

Binding is what turns symbols into an executable circuit, and it is the step
between "here is my ansatz" and "here is the circuit I planned".
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from flagquantum_mcp_server.analysis import summarize_ir
from flagquantum_mcp_server.circuits import (
    CircuitFormat,
    CircuitPayload,
    circuit_from_ir,
    ir_to_json,
    resolve_ir,
)
from flagquantum_mcp_server.errors import ToolInputError


def inspect_parameters(
    circuit: CircuitPayload, circuit_format: CircuitFormat = "ir"
) -> dict[str, Any]:
    """Report whether a circuit is parameterized, and where each symbol sits.

    Args:
        circuit: Serialized circuit, in either format. A symbol is written
            ``{"theta": {"$parameter": "theta"}}``.
        circuit_format: Either ``"ir"`` or ``"qir"``.

    Returns:
        A payload with the parameter names, the gates that carry them, and the
        circuit identity. ``is_parameterized`` is false for a fully bound
        circuit, which is still a valid answer rather than an error.
    """
    ir = resolve_ir(circuit, circuit_format)
    live = circuit_from_ir(ir)
    names = tuple(str(name) for name in live.parameter_names)

    return {
        "status": "success",
        "is_parameterized": bool(live.is_parameterized()),
        "parameter_names": list(names),
        "n_parameters": len(names),
        "occurrences": _occurrences(ir, names),
        "circuit": summarize_ir(ir)["circuit"],
        "note": (
            "Bind with bind_parameters_tool to obtain a concrete circuit. "
            "A parameterized circuit cannot be planned or exported as-is."
        ),
    }


def bind_parameters(
    circuit: CircuitPayload,
    values: Mapping[str, Any],
    circuit_format: CircuitFormat = "ir",
) -> dict[str, Any]:
    """Substitute values for a circuit's parameters.

    Args:
        circuit: Serialized circuit carrying parameters.
        values: One number per parameter name.
        circuit_format: Either ``"ir"`` or ``"qir"``.

    Returns:
        A payload with the bound circuit as canonical IR, plus its analysis.

    Raises:
        ToolInputError: If the circuit has no parameters, if a value is missing
            or unknown, or if the SDK rejects a value.
    """
    ir = resolve_ir(circuit, circuit_format)
    live = circuit_from_ir(ir)
    expected = tuple(str(name) for name in live.parameter_names)

    if not expected:
        raise ToolInputError(
            "This circuit has no parameters, so there is nothing to bind. "
            "Call inspect_parameters_tool to confirm."
        )
    _check_names(expected, values)

    try:
        bound = live.bind_parameters(dict(values))
    except (ValueError, TypeError, KeyError) as exc:
        raise ToolInputError(f"Parameters could not be bound: {exc}") from exc

    bound_ir = bound.to_ir()
    return {
        "status": "success",
        "bound_parameters": {str(k): _plain(v) for k, v in values.items()},
        "is_parameterized": bool(bound.is_parameterized()),
        "before": summarize_ir(ir)["circuit"],
        "after": summarize_ir(bound_ir)["circuit"],
        "analysis": summarize_ir(bound_ir)["analysis"],
        "ir_json": ir_to_json(bound_ir),
        "content_hash": str(bound_ir.content_hash),
    }


def _check_names(expected: tuple[str, ...], values: Mapping[str, Any]) -> None:
    """Reject a binding whose names do not match the circuit's parameters."""
    missing = sorted(set(expected) - set(values))
    if missing:
        raise ToolInputError(
            f"Missing a value for {missing}. This circuit has parameters {list(expected)}."
        )
    unknown = sorted(set(values) - set(expected))
    if unknown:
        raise ToolInputError(
            f"Unknown parameter(s) {unknown}. This circuit has parameters {list(expected)}."
        )


def _occurrences(ir: Any, names: tuple[str, ...]) -> list[dict[str, Any]]:
    """Locate each parameter in the instruction list.

    Reads the serialized form: a live ``Instruction.params`` holds the SDK's own
    ``Parameter`` objects, while only ``to_dict()`` turns them into the
    ``{"$parameter": ...}`` marker that a caller sees over the wire.
    """
    if not names:
        return []
    wanted = set(names)
    found: list[dict[str, Any]] = []
    for position, instruction in enumerate(ir.to_dict()["instructions"]):
        for key, value in instruction.get("params", {}).items():
            symbol = _symbol(value)
            if symbol is not None and symbol in wanted:
                found.append(
                    {
                        "parameter": symbol,
                        "gate": str(instruction.get("opcode")),
                        "wires": [int(wire) for wire in instruction.get("wires", ())],
                        "argument": str(key),
                        "instruction_index": position,
                    }
                )
    return found


def _symbol(value: Any) -> str | None:
    """Return the parameter name inside a serialized parameter value."""
    if isinstance(value, Mapping):
        marker = value.get("$parameter")
        if isinstance(marker, str):
            return marker
    return None


def _plain(value: Any) -> Any:
    """Convert a value into something JSON can carry."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
