"""Structural analysis, delegated to the SDK's own analyzer."""

from __future__ import annotations

import json

import pytest

from flagquantum_mcp_server.analysis import analyze

pytestmark = pytest.mark.unit


def test_bell_circuit_analysis(bell_qir: str) -> None:
    result = analyze(bell_qir, "qir")

    assert result["status"] == "success"
    assert result["circuit"]["n_qubits"] == 2
    assert result["circuit"]["n_instructions"] == 2
    assert result["circuit"]["ir_version"] == "1.0"
    assert result["analysis"]["depth"] == 2
    assert result["analysis"]["gate_counts"] == {"h": 1, "cx": 1}
    assert result["analysis"]["two_qubit_gates"] == 1


def test_ghz_circuit_analysis(ghz3_qir: str) -> None:
    result = analyze(ghz3_qir, "qir")

    assert result["circuit"]["n_qubits"] == 3
    assert result["analysis"]["gate_counts"] == {"h": 1, "cx": 2}
    assert result["analysis"]["multi_qubit_gates"] == 0


def test_depth_grows_with_a_chain_that_cannot_be_scheduled_in_parallel() -> None:
    chain = json.dumps(
        [
            {"name": "h", "index": [0]},
            {"name": "cx", "index": [0, 1]},
            {"name": "cx", "index": [1, 2]},
            {"name": "rz", "index": [2], "parameters": {"theta": 0.3}},
        ]
    )

    assert analyze(chain, "qir")["analysis"]["depth"] == 4


def test_independent_gates_share_a_layer() -> None:
    parallel = json.dumps([{"name": "h", "index": [0]}, {"name": "h", "index": [1]}])

    assert analyze(parallel, "qir")["analysis"]["depth"] == 1


def test_content_hash_identifies_the_circuit(bell_qir: str) -> None:
    first = analyze(bell_qir, "qir")
    second = analyze(bell_qir, "qir")

    assert len(first["circuit"]["content_hash"]) == 64
    assert first["circuit"]["content_hash"] == second["circuit"]["content_hash"]


def test_analysis_values_are_json_native(bell_qir: str) -> None:
    result = analyze(bell_qir, "qir")

    assert json.loads(json.dumps(result)) == result


def test_ir_input_analyzes_the_same_as_qir_input(bell_qir: str) -> None:
    from flagquantum_mcp_server.circuits import serialize

    canonical = serialize(bell_qir, "qir")["ir_json"]

    from_qir = analyze(bell_qir, "qir")
    from_ir = analyze(canonical, "ir")

    assert from_qir["circuit"] == from_ir["circuit"]
    assert from_qir["analysis"] == from_ir["analysis"]
