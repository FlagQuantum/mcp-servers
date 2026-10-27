"""Import OpenQASM text as FlagQuantum IR.

This is the inverse of :mod:`flagquantum_mcp_server.emission`, and it closes a
gap that was visible from the outside: ``emit_openqasm_tool`` writes OpenQASM,
``compile_circuit_tool`` and ``route_circuit_tool`` read FlagQuantum IR or a
gate list, and nothing in the served surface converted one back into the other.
An agent that exported a circuit could not feed the text to the compiler.

The importer is **not** a general OpenQASM front end and this module does not
pretend otherwise. FlagQuantum's importer reads the canonical subset its own
emitters write and refuses the rest with an issue code, so a round trip through
it is exact rather than approximate. Text written by another toolchain may be
refused; that refusal is reported as ``UNSUPPORTED_FORMAT`` with the importer's
own message, which names the statement and line, instead of being retried or
silently converted.

The import path is tier 3 in ``tests/test_api_contract.py``: a public
submodule with its own ``__all__`` that is not part of the frozen
``stable_exports`` snapshot, so a test here pins every name this module reads.
"""

from __future__ import annotations

from typing import Any

from flagquantum_mcp_server import limits
from flagquantum_mcp_server._bridge import load_attribute
from flagquantum_mcp_server.circuits import IR_FORMAT, enforce_limits, ir_to_json
from flagquantum_mcp_server.errors import ToolInputError, ToolLimitError, UnsupportedFormatError

IMPORT_MODULE = "flagquantum.compiler.openqasm_import"
IMPORT_FUNCTION = "import_openqasm"
IMPORT_ERROR = "OpenQASMImportError"


def import_openqasm(source: str, *, indent: int | None = None) -> dict[str, Any]:
    """Import OpenQASM 2 or 3 text and return canonical FlagQuantum IR.

    Args:
        source: The complete text of one OpenQASM program.
        indent: Optional indentation width for the returned IR JSON.

    Returns:
        A payload carrying the IR JSON, its content hash and version, the
        qubit count, and the imported opcodes. The ``ir_json`` value can be
        passed to any tool that accepts ``circuit_format="ir"``.

    Raises:
        ToolInputError: If the source is not text.
        ToolLimitError: If the source or the resulting circuit exceeds a
            configured bound.
        UnsupportedFormatError: If the importer refuses the text as outside
            the canonical subset. The message carries the importer's issue code.
    """
    if not isinstance(source, str):
        raise ToolInputError(
            f"OpenQASM source must be a string; received {type(source).__name__}."
        )
    if len(source) > limits.max_qasm_chars():
        raise ToolLimitError(
            f"OpenQASM source is {len(source)} characters, above the "
            f"{limits.max_qasm_chars()}-character limit. "
            "Raise FLAGQUANTUM_MCP_MAX_QASM_CHARS or send a smaller program."
        )

    importer = load_attribute(IMPORT_MODULE, IMPORT_FUNCTION)
    refusal = load_attribute(IMPORT_MODULE, IMPORT_ERROR)
    try:
        program = importer(source)
    except refusal as exc:
        # A program outside the canonical subset is a property of the request,
        # not a defect in this server, and the importer's message already names
        # the offending statement and line. Passing it through keeps that
        # detail; wrapping it as an internal error would throw it away.
        raise UnsupportedFormatError(f"OpenQASM import refused: {exc}") from exc

    ir = enforce_limits(program.to_ir())
    return {
        "status": "success",
        "format": IR_FORMAT,
        "ir_json": ir_to_json(ir, indent=indent),
        "n_qubits": int(ir.n_wires),
        "n_instructions": len(ir.instructions),
        "ir_version": str(ir.version),
        "content_hash": str(ir.content_hash),
        "qasm_version": f"{float(program.version):g}",
        "opcodes": [str(instruction.name) for instruction in program.instructions],
        "measurement_qubits": [int(qubit) for qubit in program.measurement_qubits],
        "whole_register_measurement": bool(program.whole_register_measurement),
    }
