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
