"""Resolve caller input into a validated FlagQuantum circuit IR.

Two input formats are accepted, both of them the SDK's own serialization:

``ir``
    FlagQuantum's versioned circuit IR (``flagquantum.circuit_ir``), as
    produced by ``CircuitIR.to_json()``. This is the canonical format: it
    carries a version, a content hash, and strict unknown-field rejection.
``qir``
    The compact gate list produced by ``Circuit.to_qir()``. More ergonomic to
    write by hand, at the cost of the IR's version and hash guarantees.

Every tool funnels through :func:`resolve_ir`, so the size, width and gate
bounds in :mod:`flagquantum_mcp_server.limits` apply uniformly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from flagquantum_mcp_server import limits
from flagquantum_mcp_server._bridge import load_sdk
from flagquantum_mcp_server.errors import (
    ToolInputError,
    ToolLimitError,
    UnsupportedFormatError,
)

IR_FORMAT = "ir"
QIR_FORMAT = "qir"
SUPPORTED_FORMATS: tuple[str, ...] = (IR_FORMAT, QIR_FORMAT)
IR_KIND = "flagquantum.circuit_ir"

CircuitPayload = str | Mapping[str, Any] | Sequence[Any]


def resolve_ir(circuit: CircuitPayload, circuit_format: str = IR_FORMAT) -> Any:
    """Parse caller input and return a validated ``CircuitIR``.

    Args:
        circuit: Serialized circuit, either as JSON text or an already-decoded
            mapping / sequence.
        circuit_format: Either ``"ir"`` or ``"qir"``.

    Returns:
        A ``flagquantum.CircuitIR``. The concrete type is not annotated because
        importing FlagQuantum at module scope would defeat the lazy binding in
        :mod:`flagquantum_mcp_server._bridge`.

    Raises:
        UnsupportedFormatError: If ``circuit_format`` is not recognized.
        ToolInputError: If the payload cannot be parsed or fails IR validation.
        ToolLimitError: If the circuit exceeds a configured bound.
    """
    if circuit_format not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"circuit_format must be one of {list(SUPPORTED_FORMATS)}; received {circuit_format!r}."
        )

    decoded = _decode(circuit)
    ir = _ir_from_decoded(decoded) if circuit_format == IR_FORMAT else _ir_from_qir(decoded)
    return enforce_limits(ir)


def _decode(circuit: CircuitPayload) -> Any:
    """Turn JSON text into objects, passing decoded payloads through."""
    if not isinstance(circuit, str):
        return circuit
    encoded = circuit.encode("utf-8")
    if len(encoded) > limits.max_ir_bytes():
        raise ToolLimitError(
            f"Circuit payload is {len(encoded)} bytes, above the "
            f"{limits.max_ir_bytes()}-byte limit. "
            "Raise FLAGQUANTUM_MCP_MAX_IR_BYTES or send a smaller circuit."
        )
    try:
        return json.loads(circuit)
    except json.JSONDecodeError as exc:
        raise ToolInputError(
            f"Circuit payload is not valid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}."
        ) from exc


def _ir_from_decoded(decoded: Any) -> Any:
    """Build a CircuitIR from a decoded ``ir``-format payload."""
    if not isinstance(decoded, Mapping):
        raise ToolInputError(
            "circuit_format='ir' expects a JSON object with the "
            "flagquantum.circuit_ir fields; received "
            f"{type(decoded).__name__}."
        )
    kind = decoded.get("kind")
    if kind != IR_KIND:
        raise ToolInputError(
            f"Expected a circuit IR with kind={IR_KIND!r}; received kind={kind!r}. "
            "Use serialize_circuit_tool to produce a valid payload, or pass "
            "circuit_format='qir' for a gate list."
        )
    sdk = load_sdk()
    try:
        return sdk.CircuitIR.from_dict(decoded)
    except (ValueError, TypeError) as exc:
        raise ToolInputError(f"Circuit IR failed validation: {exc}") from exc


def _ir_from_qir(decoded: Any) -> Any:
    """Build a CircuitIR from a decoded ``qir``-format gate list."""
    if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes)):
        raise ToolInputError(
            "circuit_format='qir' expects a JSON array of gate objects such as "
            '[{"name": "h", "index": [0]}, {"name": "cx", "index": [0, 1]}]; '
            f"received {type(decoded).__name__}."
        )
    sdk = load_sdk()
    try:
        return sdk.Circuit.from_qir(list(decoded)).to_ir()
    except (ValueError, TypeError) as exc:
        raise ToolInputError(f"Gate list could not be built into a circuit: {exc}") from exc


def enforce_limits(ir: Any) -> Any:
    """Reject a circuit that exceeds the configured width or gate bound.

    Args:
        ir: A ``flagquantum.CircuitIR``.

    Returns:
        The same IR, when it is within bounds.

    Raises:
        ToolLimitError: If the circuit is wider or larger than allowed.
    """
    n_wires = int(ir.n_wires)
    if n_wires > limits.max_qubits():
        raise ToolLimitError(
            f"Circuit has {n_wires} qubits, above the {limits.max_qubits()}-qubit "
            "limit. Raise FLAGQUANTUM_MCP_MAX_QUBITS to allow it."
        )
    n_gates = len(ir.instructions)
    if n_gates > limits.max_gates():
        raise ToolLimitError(
            f"Circuit has {n_gates} instructions, above the "
            f"{limits.max_gates()}-instruction limit. "
            "Raise FLAGQUANTUM_MCP_MAX_GATES to allow it."
        )
    return ir


def ir_to_json(ir: Any, *, indent: int | None = None) -> str:
    """Serialize a circuit IR to JSON text.

    Args:
        ir: A ``flagquantum.CircuitIR``.
        indent: Optional indentation width for readability.

    Returns:
        The serialized IR.

    Raises:
        ToolLimitError: If the serialized form exceeds the payload bound.
    """
    text = str(ir.to_json(indent=indent))
    size = len(text.encode("utf-8"))
    if size > limits.max_ir_bytes():
        raise ToolLimitError(
            f"Serialized circuit is {size} bytes, above the "
            f"{limits.max_ir_bytes()}-byte limit. "
            "Raise FLAGQUANTUM_MCP_MAX_IR_BYTES or send a smaller circuit."
        )
    return text


def circuit_from_ir(ir: Any) -> Any:
    """Rebuild a live ``Circuit`` from a validated IR.

    Args:
        ir: A ``flagquantum.CircuitIR``.

    Returns:
        A ``flagquantum.Circuit``.
    """
    return load_sdk().Circuit.from_ir(ir)


def serialize(
    circuit: CircuitPayload, circuit_format: str = QIR_FORMAT, *, indent: int | None = None
) -> dict[str, Any]:
    """Canonicalize a circuit into FlagQuantum IR JSON.

    Accepts either input format and always answers with the canonical ``ir``
    form, so an agent that built a circuit as a gate list gets back the
    versioned, hashable representation.

    Args:
        circuit: Serialized circuit, as JSON text or a decoded payload.
        circuit_format: Either ``"ir"`` or ``"qir"``.
        indent: Optional indentation width for readability.

    Returns:
        A payload carrying the IR JSON, its content hash and its version.

    Raises:
        ToolError: If the payload is invalid or exceeds a configured bound.
    """
    ir = resolve_ir(circuit, circuit_format)
    return {
        "status": "success",
        "format": IR_FORMAT,
        "ir_json": ir_to_json(ir, indent=indent),
        "n_qubits": int(ir.n_wires),
        "n_instructions": len(ir.instructions),
        "ir_version": str(ir.version),
        "content_hash": str(ir.content_hash),
    }


def deserialize(ir_json: str, *, indent: int | None = None) -> dict[str, Any]:
    """Validate IR JSON and report whether it round-trips unchanged.

    Args:
        ir_json: FlagQuantum IR JSON text.
        indent: Optional indentation width for the canonical re-serialization.

    Returns:
        A payload with the circuit identity, the canonical IR JSON, and
        ``round_trip_stable`` recording whether re-serializing reproduced the
        same content hash.

    Raises:
        ToolError: If the payload is invalid or exceeds a configured bound.
    """
    ir = resolve_ir(ir_json, IR_FORMAT)
    canonical = ir_to_json(ir, indent=indent)
    return {
        "status": "success",
        "format": IR_FORMAT,
        "ir_json": canonical,
        "n_qubits": int(ir.n_wires),
        "n_instructions": len(ir.instructions),
        "ir_version": str(ir.version),
        "content_hash": str(ir.content_hash),
        "round_trip_stable": canonical == ir_json,
    }


def known_gate_names() -> list[str]:
    """Return the gate vocabulary of the installed FlagQuantum.

    Derived at runtime rather than hard-coded, so the answer cannot drift from
    the SDK it describes. The SDK installs every gate method twice — once
    lowercase and once uppercase — so a name that exists in both cases is a
    gate, while ``run``/``draw``/``analysis`` and friends exist only lowercase.

    Returns:
        Sorted opcode names.
    """
    sdk = load_sdk()
    attributes = set(dir(sdk.Circuit))
    return sorted(
        name
        for name in attributes
        if name.islower() and name.upper() in attributes and not name.startswith("_")
    )
