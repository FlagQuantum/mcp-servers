"""Emit a circuit as OpenQASM or QCIS text.

Both emitters live in public submodules that FlagQuantum does not re-export
from ``flagquantum.compiler``. That makes them the weakest contract this server
rests on, so they are resolved through
:func:`flagquantum_mcp_server._bridge.load_attribute`, which turns a relocation
into a named error instead of an AttributeError inside a tool call. A test
pins both import paths.
"""

from __future__ import annotations

from typing import Any

from flagquantum_mcp_server import limits
from flagquantum_mcp_server._bridge import load_attribute
from flagquantum_mcp_server.circuits import CircuitFormat, CircuitPayload, resolve_ir
from flagquantum_mcp_server.errors import ToolLimitError, UnsupportedFormatError

OPENQASM_MODULE = "flagquantum.compiler.openqasm"
OPENQASM_FUNCTION = "emit_openqasm"
QCIS_MODULE = "flagquantum.compiler.qcis"
QCIS_FUNCTION = "emit_qcis"

SUPPORTED_QASM_VERSIONS: tuple[float, ...] = (2.0, 3.0)


def emit_openqasm(
    circuit: CircuitPayload,
    circuit_format: CircuitFormat = "ir",
    *,
    version: float = 3.0,
    result_wires: list[int] | None = None,
) -> dict[str, Any]:
    """Render a circuit as OpenQASM 2.0 or 3.0 text.

    Args:
        circuit: Serialized circuit, as JSON text or a decoded payload.
        circuit_format: Either ``"ir"`` or ``"qir"``.
        version: OpenQASM version, ``2.0`` or ``3.0``.
        result_wires: Wires to measure. ``None`` measures every wire.

    Returns:
        A payload carrying the emitted program and its metadata.

    Raises:
        ToolError: If the payload is invalid, the version is unsupported, or the
            emitted text exceeds the configured bound.
    """
    if version not in SUPPORTED_QASM_VERSIONS:
        raise UnsupportedFormatError(
            f"OpenQASM version must be one of {list(SUPPORTED_QASM_VERSIONS)}; "
            f"received {version!r}."
        )
    ir = resolve_ir(circuit, circuit_format)
    emit = load_attribute(OPENQASM_MODULE, OPENQASM_FUNCTION)
    wires = tuple(result_wires) if result_wires is not None else None
    text = _emit(emit, ir, "OpenQASM", version=version, result_wires=wires)
    return _payload(text, "openqasm", f"{version:g}", ir)


def emit_qcis(circuit: CircuitPayload, circuit_format: CircuitFormat = "ir") -> dict[str, Any]:
    """Render a circuit as QCIS text.

    Args:
        circuit: Serialized circuit, as JSON text or a decoded payload.
        circuit_format: Either ``"ir"`` or ``"qir"``.

    Returns:
        A payload carrying the emitted program and its metadata.

    Raises:
        ToolError: If the payload is invalid, or the circuit contains a gate
            QCIS cannot express (an arbitrary matrix, for instance).
    """
    ir = resolve_ir(circuit, circuit_format)
    emit = load_attribute(QCIS_MODULE, QCIS_FUNCTION)
    text = _emit(emit, ir, "QCIS")
    return _payload(text, "qcis", None, ir)


def _emit(emit: Any, ir: Any, target: str, **kwargs: Any) -> str:
    """Call an emitter, turning a lowering refusal into a caller-facing error.

    A gate the target cannot express is a property of the request, not a bug in
    this server, so the SDK's ``ValueError`` is re-raised as an input error with
    the gate named. Left alone it would reach the client as an internal error.

    Args:
        emit: The emitter function to call.
        ir: The circuit IR to emit.
        target: Format name, used in the error message.
        **kwargs: Extra keyword arguments forwarded to the emitter.

    Returns:
        The emitted program text.

    Raises:
        UnsupportedFormatError: If the emitter refuses to lower this circuit.
    """
    try:
        return str(emit(ir, **kwargs))
    except NotImplementedError as exc:
        raise UnsupportedFormatError(f"{target} cannot express this circuit: {exc}") from exc
    except ValueError as exc:
        raise UnsupportedFormatError(f"{target} cannot express this circuit: {exc}") from exc


def _payload(text: str, fmt: str, version: str | None, ir: Any) -> dict[str, Any]:
    """Assemble the emission payload, enforcing the output-size bound."""
    if len(text) > limits.max_qasm_chars():
        raise ToolLimitError(
            f"Emitted {fmt} text is {len(text)} characters, above the "
            f"{limits.max_qasm_chars()}-character limit. "
            "Raise FLAGQUANTUM_MCP_MAX_QASM_CHARS or send a smaller circuit."
        )
    payload: dict[str, Any] = {
        "status": "success",
        "format": fmt,
        "text": text,
        "n_lines": text.count("\n") + 1,
        "n_qubits": int(ir.n_wires),
        "content_hash": str(ir.content_hash),
    }
    if version is not None:
        payload["version"] = version
    return payload
