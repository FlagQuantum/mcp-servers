"""Structural analysis of a circuit, delegated to the SDK's own analyzer.

FlagQuantum already computes depth, gate counts and wire usage in
``Circuit.analysis()``. This module does not recompute any of it: it calls that
method and re-shapes the result for a model. Reimplementing the counting here
would create a second source of truth that drifts from the SDK the moment
either changes.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from flagquantum_mcp_server._bridge import load_sdk
from flagquantum_mcp_server.circuits import CircuitFormat, CircuitPayload, resolve_ir


def analyze(circuit: CircuitPayload, circuit_format: CircuitFormat = "ir") -> dict[str, Any]:
    """Describe a circuit's structure without executing it.

    Args:
        circuit: Serialized circuit, as JSON text or a decoded payload.
        circuit_format: Either ``"ir"`` or ``"qir"``.

    Returns:
        A payload containing the circuit identity (qubit count, IR version,
        content hash) and the SDK's structural analysis.

    Raises:
        ToolError: If the payload is invalid or exceeds a configured bound.
    """
    ir = resolve_ir(circuit, circuit_format)
    sdk = load_sdk()
    analysis = sdk.Circuit.from_ir(ir).analysis()
    return {
        "status": "success",
        "circuit": _identity(ir),
        "analysis": _fields(analysis),
    }


def summarize_ir(ir: Any) -> dict[str, Any]:
    """Return the identity-plus-analysis block for an already-validated IR.

    Shared by every tool that transforms a circuit and must report what the
    transformation did, so the shape of that report is defined once.

    Args:
        ir: A ``flagquantum.CircuitIR``.

    Returns:
        A mapping with ``circuit`` and ``analysis`` keys.
    """
    sdk = load_sdk()
    return {
        "circuit": _identity(ir),
        "analysis": _fields(sdk.Circuit.from_ir(ir).analysis()),
    }


def _identity(ir: Any) -> dict[str, Any]:
    """Return the small identity block shared by every tool payload."""
    return {
        "n_qubits": int(ir.n_wires),
        "n_instructions": len(ir.instructions),
        "ir_version": str(ir.version),
        "content_hash": str(ir.content_hash),
    }


def _fields(analysis: Any) -> dict[str, Any]:
    """Flatten the SDK analysis object into plain JSON types.

    Reads the object's fields rather than importing its class, so the analysis
    type stays an SDK implementation detail. Any field the SDK adds later flows
    through unchanged; a field it removes simply stops appearing.

    Args:
        analysis: The value returned by ``Circuit.analysis()``.

    Returns:
        A JSON-serializable mapping of the analysis fields.
    """
    if dataclasses.is_dataclass(analysis) and not isinstance(analysis, type):
        raw: dict[str, Any] = dict(dataclasses.asdict(analysis))
    else:  # pragma: no cover - defensive; the SDK returns a dataclass today
        raw = dict(vars(analysis))
    return {key: _plain(value) for key, value in raw.items()}


def _plain(value: Any) -> Any:
    """Convert tuples and other containers into JSON-native types."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
