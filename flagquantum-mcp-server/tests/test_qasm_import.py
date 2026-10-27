"""OpenQASM import, and the round trip that makes it useful."""

from __future__ import annotations

import json

import pytest

from flagquantum_mcp_server.circuits import serialize
from flagquantum_mcp_server.compilation import compile_circuit, route_circuit
from flagquantum_mcp_server.emission import emit_openqasm
from flagquantum_mcp_server.errors import ToolInputError, ToolLimitError, UnsupportedFormatError
from flagquantum_mcp_server.qasm_import import import_openqasm

pytestmark = pytest.mark.unit

BELL_QASM3 = (
    'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[2] q;\nbit[2] c;\n'
    "h q[0];\ncx q[0], q[1];\nc = measure q;"
)

BELL_QASM2 = (
    'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
    "h q[0];\ncx q[0], q[1];\nmeasure q[0] -> c[0];\nmeasure q[1] -> c[1];"
)


def test_imports_openqasm3_as_ir() -> None:
    result = import_openqasm(BELL_QASM3)

    assert result["status"] == "success"
    assert result["format"] == "ir"
    assert result["qasm_version"] == "3"
    assert result["opcodes"] == ["h", "cx"]
    assert result["n_qubits"] == 2
    assert result["n_instructions"] == 2
    assert result["whole_register_measurement"] is True
    assert result["measurement_qubits"] == [0, 1]


def test_import_carries_the_measurement_the_text_declares() -> None:
    """A program that measures must import as a circuit that measures.

    Dropping the measurement would make an imported circuit look like a pure
    unitary it is not, and the hash would then describe a different circuit
    from the text.
    """
    result = import_openqasm(BELL_QASM3)
    ir = json.loads(result["ir_json"])

    assert ir["measurements"] == [
        {"kind": "counts", "metadata": {}, "shots": None, "wires": [0, 1]}
    ]


def test_import_is_idempotent() -> None:
    once = import_openqasm(BELL_QASM3)
    twice = import_openqasm(BELL_QASM3)

    assert once["content_hash"] == twice["content_hash"]
    assert once["ir_json"] == twice["ir_json"]


def test_imports_openqasm2() -> None:
    result = import_openqasm(BELL_QASM2)

    assert result["qasm_version"] == "2"
    assert result["opcodes"] == ["h", "cx"]
    assert result["whole_register_measurement"] is False


def test_returned_ir_is_accepted_by_the_ir_reader() -> None:
    """The payload is only useful if the other tools can consume it."""
    imported = import_openqasm(BELL_QASM3)

    reserialized = serialize(imported["ir_json"], "ir")

    assert reserialized["content_hash"] == imported["content_hash"]


def test_emit_then_import_preserves_the_gates(bell_qir: str) -> None:
    """Exported text imports back to the same gates on the same wires.

    The hash deliberately does **not** match: ``emit_openqasm`` measures every
    wire unless told otherwise, so the text carries a measurement the source
    circuit never had. That makes the round trip a faithful import of the text,
    not an identity on the source, and a test asserting hash equality would be
    asserting something the emitter never promised.
    """
    for version in (2.0, 3.0):
        emitted = emit_openqasm(bell_qir, "qir", version=version)
        imported = import_openqasm(emitted["text"])

        assert imported["opcodes"] == ["h", "cx"], f"OpenQASM {version:g}"
        assert imported["n_qubits"] == emitted["n_qubits"], f"OpenQASM {version:g}"
        assert imported["measurement_qubits"] == [0, 1], f"OpenQASM {version:g}"


def test_imported_ir_can_be_compiled_and_routed(bell_qir: str) -> None:
    """The whole point of the tool: exported text goes back into the pipeline."""
    emitted = emit_openqasm(bell_qir, "qir", version=3.0)
    imported = import_openqasm(emitted["text"])

    compiled = compile_circuit(imported["ir_json"], "ir", topology="line")
    assert compiled["status"] == "success"

    routed = route_circuit(imported["ir_json"], "ir", topology="line")
    assert routed["status"] == "success"
    assert routed["instruction_delta"] == 0


def test_indent_only_changes_the_text() -> None:
    compact = import_openqasm(BELL_QASM3)
    indented = import_openqasm(BELL_QASM3, indent=2)

    assert indented["ir_json"] != compact["ir_json"]
    assert json.loads(indented["ir_json"]) == json.loads(compact["ir_json"])
    assert indented["content_hash"] == compact["content_hash"]


def test_non_text_source_is_rejected() -> None:
    with pytest.raises(ToolInputError, match="must be a string"):
        import_openqasm(b"OPENQASM 3.0;")  # type: ignore[arg-type]


def test_oversized_source_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_QASM_CHARS", "10")

    with pytest.raises(ToolLimitError, match="character limit"):
        import_openqasm(BELL_QASM3)


def test_foreign_openqasm_is_refused_rather_than_rewritten() -> None:
    """A file another toolchain wrote is refused, and the refusal is readable.

    Barrier is legal OpenQASM that FlagQuantum's canonical subset does not
    carry. The tool must report that, not drop the statement and return a
    circuit that no longer matches the text it was given.
    """
    source = (
        'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\ncreg c[2];\n'
        "h q[0];\nbarrier q;\ncx q[0], q[1];\nmeasure q[0] -> c[0];\n"
    )

    with pytest.raises(UnsupportedFormatError, match="OpenQASM import refused"):
        import_openqasm(source)


def test_refusal_names_the_offending_line() -> None:
    source = 'OPENQASM 3.0;\nqubit[1] q;\nh q[0];'

    with pytest.raises(UnsupportedFormatError) as excinfo:
        import_openqasm(source)

    assert "line" in str(excinfo.value)
