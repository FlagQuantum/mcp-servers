"""Execution planning, options and output requests."""

from __future__ import annotations

import json

import pytest

from flagquantum_mcp_server.circuits import serialize
from flagquantum_mcp_server.errors import (
    ToolInputError,
    UnsupportedFormatError,
)
from flagquantum_mcp_server.planning import plan_execution

pytestmark = pytest.mark.unit


def test_plan_reports_a_local_cpu_contract(ghz3_qir: str) -> None:
    result = plan_execution(ghz3_qir, "qir")

    assert result["status"] == "success"
    assert result["plan"]["mode"] == "statevector"
    assert result["plan"]["device"] == "cpu"
    assert result["plan"]["backend"] == "pytorch"
    assert result["plan"]["is_distributed"] is False
    assert result["plan"]["schema_version"] == "1.0"


def test_plan_summary_carries_the_structural_numbers(ghz3_qir: str) -> None:
    summary = plan_execution(ghz3_qir, "qir")["summary"]

    assert summary["depth"] == 3
    assert summary["n_wires"] == 3
    assert summary["n_instructions"] == 3


def test_plan_json_round_trips(ghz3_qir: str) -> None:
    import flagquantum as fq

    result = plan_execution(ghz3_qir, "qir")

    assert fq.ExecutionPlan.from_json(result["plan_json"]).identity == result["plan"]["identity"]


def test_options_are_applied(ghz3_qir: str) -> None:
    result = plan_execution(
        ghz3_qir, "qir", options={"precision": "complex128", "mode": "statevector"}
    )

    assert result["plan"]["precision"] == "complex128"


def test_unknown_option_is_rejected(ghz3_qir: str) -> None:
    with pytest.raises(UnsupportedFormatError, match="Unknown execution option"):
        plan_execution(ghz3_qir, "qir", options={"teleport": True})


def test_invalid_option_value_surfaces_as_input_error(ghz3_qir: str) -> None:
    with pytest.raises(ToolInputError, match="Invalid execution option"):
        plan_execution(ghz3_qir, "qir", options={"precision": "float16"})


def test_expectation_output_accepts_a_pauli_string(ghz3_qir: str) -> None:
    result = plan_execution(ghz3_qir, "qir", outputs=[{"kind": "expectation", "pauli": "ZZI"}])

    assert result["status"] == "success"


def test_expectation_requires_a_pauli_string(ghz3_qir: str) -> None:
    with pytest.raises(ToolInputError, match="requires a 'pauli' string"):
        plan_execution(ghz3_qir, "qir", outputs=[{"kind": "expectation"}])


def test_pauli_string_must_match_the_circuit_width(ghz3_qir: str) -> None:
    with pytest.raises(ToolInputError, match="covers 2 wires, but the circuit has 3"):
        plan_execution(ghz3_qir, "qir", outputs=[{"kind": "expectation", "pauli": "ZZ"}])


def test_pauli_string_rejects_unknown_letters(ghz3_qir: str) -> None:
    with pytest.raises(ToolInputError, match="unsupported letters"):
        plan_execution(ghz3_qir, "qir", outputs=[{"kind": "expectation", "pauli": "ZZA"}])


def test_all_identity_pauli_string_is_rejected(ghz3_qir: str) -> None:
    with pytest.raises(ToolInputError, match="entirely identity"):
        plan_execution(ghz3_qir, "qir", outputs=[{"kind": "expectation", "pauli": "III"}])


def test_counts_output_requires_shots(ghz3_qir: str) -> None:
    with pytest.raises(ToolInputError, match="require a shot count"):
        plan_execution(ghz3_qir, "qir", outputs=[{"kind": "counts", "wires": [0]}])


def test_counts_output_is_accepted_with_shots(ghz3_qir: str) -> None:
    result = plan_execution(
        ghz3_qir,
        "qir",
        options={"shots": 1024},
        outputs=[{"kind": "counts", "wires": [0, 1]}],
    )

    assert result["status"] == "success"


def test_probabilities_output_needs_no_shots(ghz3_qir: str) -> None:
    result = plan_execution(ghz3_qir, "qir", outputs=[{"kind": "probabilities", "wires": [0]}])

    assert result["status"] == "success"


def test_output_defaults_to_every_wire(ghz3_qir: str) -> None:
    result = plan_execution(ghz3_qir, "qir", options={"shots": 512}, outputs=[{"kind": "samples"}])

    assert result["status"] == "success"


def test_unknown_output_kind_is_rejected(ghz3_qir: str) -> None:
    with pytest.raises(UnsupportedFormatError, match="Output kind must be one of"):
        plan_execution(ghz3_qir, "qir", outputs=[{"kind": "tomography"}])


def test_out_of_range_wire_is_rejected(ghz3_qir: str) -> None:
    with pytest.raises(ToolInputError, match=r"outside 0\.\.2"):
        plan_execution(
            ghz3_qir, "qir", options={"shots": 100}, outputs=[{"kind": "counts", "wires": [7]}]
        )


def test_non_integer_wire_is_rejected(ghz3_qir: str) -> None:
    with pytest.raises(ToolInputError, match="must be integers"):
        plan_execution(
            ghz3_qir,
            "qir",
            options={"shots": 100},
            outputs=[{"kind": "counts", "wires": ["zero"]}],
        )


def _with_measurements(shots: int = 64) -> str:
    """A canonical IR payload that declares its own measurement."""
    envelope = json.loads(serialize(json.dumps([{"name": "h", "index": [0]}]), "qir")["ir_json"])
    envelope["measurements"] = [{"kind": "counts", "wires": [0], "shots": shots}]
    return json.dumps(envelope)


def test_outputs_alongside_declared_measurements_are_refused() -> None:
    """The SDK's refusal names neither field, so this tool has to.

    A circuit may carry its own measurements, and the SDK reads them; it also
    refuses output requests on top of them, saying only that measurements
    cannot be supplied when the program already contains measurement requests.
    A caller is told a conflict exists and not which two things are in it.
    """
    with pytest.raises(ToolInputError) as caught:
        plan_execution(
            _with_measurements(), "ir", outputs=[{"kind": "probabilities", "wires": [0]}]
        )

    message = str(caught.value)
    assert "measurements" in message
    assert "outputs" in message


def test_a_circuit_may_declare_measurements_without_outputs() -> None:
    """The field is legal on its own; only the combination is refused."""
    assert plan_execution(_with_measurements(), "ir")["status"] == "success"
