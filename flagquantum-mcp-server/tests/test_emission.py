"""OpenQASM and QCIS emission."""

from __future__ import annotations

import json

import pytest

from flagquantum_mcp_server.emission import emit_openqasm, emit_qcis
from flagquantum_mcp_server.errors import ToolLimitError, UnsupportedFormatError

pytestmark = pytest.mark.unit


def test_openqasm3_emission(bell_qir: str) -> None:
    result = emit_openqasm(bell_qir, "qir", version=3.0)

    assert result["status"] == "success"
    assert result["format"] == "openqasm"
    assert result["version"] == "3"
    assert result["text"].startswith("OPENQASM 3.0;")
    assert "h q[0];" in result["text"]
    assert "cx q[0], q[1];" in result["text"]
    assert result["n_qubits"] == 2


def test_openqasm2_emission(bell_qir: str) -> None:
    result = emit_openqasm(bell_qir, "qir", version=2.0)

    assert result["text"].startswith("OPENQASM 2.0;")
    assert "qreg q[2];" in result["text"]


def test_openqasm_defaults_to_version_3(bell_qir: str) -> None:
    assert emit_openqasm(bell_qir, "qir")["version"] == "3"


def test_openqasm_reports_line_count(bell_qir: str) -> None:
    result = emit_openqasm(bell_qir, "qir")

    assert result["n_lines"] == result["text"].count("\n") + 1


def test_result_wires_narrow_the_measurement(bell_qir: str) -> None:
    every_wire = emit_openqasm(bell_qir, "qir", version=3.0)
    one_wire = emit_openqasm(bell_qir, "qir", version=3.0, result_wires=[0])

    assert one_wire["text"] != every_wire["text"]
    assert one_wire["n_qubits"] == 2


def test_unsupported_openqasm_version_is_rejected(bell_qir: str) -> None:
    with pytest.raises(UnsupportedFormatError, match="OpenQASM version must be one of"):
        emit_openqasm(bell_qir, "qir", version=4.0)


def test_qcis_emission(ghz3_qir: str) -> None:
    result = emit_qcis(ghz3_qir, "qir")

    assert result["status"] == "success"
    assert result["format"] == "qcis"
    assert result["n_qubits"] == 3
    assert result["text"].strip() != ""
    assert "version" not in result


def test_qcis_reports_the_source_hash(ghz3_qir: str) -> None:
    from flagquantum_mcp_server.circuits import serialize

    result = emit_qcis(ghz3_qir, "qir")

    assert result["content_hash"] == serialize(ghz3_qir, "qir")["content_hash"]


def test_arbitrary_matrix_gate_cannot_be_emitted_as_qcis(bell_qir: str) -> None:
    import flagquantum as fq

    from flagquantum_mcp_server.circuits import serialize

    circuit = fq.Circuit(1)
    circuit.any(0, unitary=[[1.0, 0.0], [0.0, 1.0]])
    matrix_gate = serialize(json.dumps(circuit.to_ir().to_dict()), "ir")["ir_json"]

    with pytest.raises(UnsupportedFormatError, match="QCIS cannot express"):
        emit_qcis(matrix_gate, "ir")


def test_oversized_emission_is_rejected(monkeypatch: pytest.MonkeyPatch, ghz3_qir: str) -> None:
    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_QASM_CHARS", "10")

    with pytest.raises(ToolLimitError, match="above the 10-character limit"):
        emit_openqasm(ghz3_qir, "qir")


# --- a value the target cannot express ---
#
# A complex angle is a legal IR value: the SDK encodes and decodes ``$complex``
# itself. Neither text target can express it, and both emitters say so by
# raising TypeError rather than ValueError. Left uncaught that reaches the
# client as an unhandled exception instead of a structured error.


def _complex_angled_ir() -> str:
    """A one-gate IR whose angle is complex rather than real."""
    return json.dumps(
        {
            "kind": "flagquantum.circuit_ir",
            "version": "1.0",
            "n_wires": 1,
            "dtype": "complex64",
            "shape": [2],
            "instructions": [
                {
                    "opcode": "ry",
                    "wires": [0],
                    "params": {"theta": {"$complex": [1.0, 0.0]}},
                    "matrix": None,
                    "metadata": {},
                }
            ],
            "observables": [],
            "measurements": [],
            "metadata": {},
        }
    )


def test_openqasm_reports_an_inexpressible_value_as_an_input_error() -> None:
    with pytest.raises(UnsupportedFormatError, match="OpenQASM cannot express"):
        emit_openqasm(_complex_angled_ir(), "ir")


def test_qcis_reports_an_inexpressible_value_as_an_input_error() -> None:
    with pytest.raises(UnsupportedFormatError, match="QCIS cannot express"):
        emit_qcis(_complex_angled_ir(), "ir")


def test_the_refusal_names_the_value_rather_than_raising_bare() -> None:
    with pytest.raises(UnsupportedFormatError, match="must be real"):
        emit_openqasm(_complex_angled_ir(), "ir")


# --- what the payload's content_hash identifies ---


def test_openqasm_measures_every_wire_unless_told_otherwise(bell_qir: str) -> None:
    """The default appends measurements the source circuit does not have.

    Pinned because the result's ``content_hash`` is the *source* circuit's, so a
    caller who reads the hash as identifying the emitted text is wrong. The
    README and the tool description say so.
    """
    result = emit_openqasm(bell_qir, "qir")

    assert "measure" in result["text"]
    result_wires = emit_openqasm(bell_qir, "qir", result_wires=[0])
    assert "measure q[1]" not in result_wires["text"]


def test_the_emission_hash_identifies_the_source_circuit(bell_qir: str) -> None:
    from flagquantum_mcp_server.analysis import analyze

    emitted = emit_openqasm(bell_qir, "qir")

    assert emitted["content_hash"] == analyze(bell_qir, "qir")["circuit"]["content_hash"]


def test_qcis_appends_no_measurement(bell_qir: str) -> None:
    """The asymmetry is deliberate: only the QASM emitters measure by default."""
    text = emit_qcis(bell_qir, "qir")["text"]

    assert text.strip()
    assert "measure" not in text.lower()
