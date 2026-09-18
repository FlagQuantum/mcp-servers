"""Parameter inspection and binding, and the structure/drawing tools."""

from __future__ import annotations

import json

import flagquantum as fq
import pytest

from flagquantum_mcp_server.drawing import draw_circuit
from flagquantum_mcp_server.errors import ToolInputError, UnsupportedFormatError
from flagquantum_mcp_server.parameters import bind_parameters, inspect_parameters
from flagquantum_mcp_server.structure import describe_layers, describe_topology

pytestmark = pytest.mark.unit


def _parametric_ir() -> str:
    """A two-wire ansatz carrying one symbol, serialized as IR."""
    circuit = fq.Circuit(2).ry(0, fq.Parameter("theta")).cx(0, 1)
    return str(circuit.to_ir().to_json())


# --- parameters ---


def test_a_parameterized_circuit_reports_its_symbols() -> None:
    result = inspect_parameters(_parametric_ir(), "ir")

    assert result["status"] == "success"
    assert result["is_parameterized"] is True
    assert result["parameter_names"] == ["theta"]
    assert result["n_parameters"] == 1


def test_each_occurrence_names_its_gate_and_wire() -> None:
    result = inspect_parameters(_parametric_ir(), "ir")

    assert result["occurrences"] == [
        {
            "parameter": "theta",
            "gate": "ry",
            "wires": [0],
            "argument": "theta",
            "instruction_index": 0,
        }
    ]


def test_a_concrete_circuit_reports_no_parameters(bell_qir: str) -> None:
    result = inspect_parameters(bell_qir, "qir")

    assert result["is_parameterized"] is False
    assert result["parameter_names"] == []
    assert result["occurrences"] == []


def test_binding_produces_a_concrete_circuit() -> None:
    result = bind_parameters(_parametric_ir(), {"theta": 0.7}, "ir")

    assert result["status"] == "success"
    assert result["is_parameterized"] is False
    assert result["bound_parameters"] == {"theta": 0.7}
    assert result["content_hash"] == result["after"]["content_hash"]


def test_binding_is_deterministic() -> None:
    first = bind_parameters(_parametric_ir(), {"theta": 0.7}, "ir")
    second = bind_parameters(_parametric_ir(), {"theta": 0.7}, "ir")

    assert first["content_hash"] == second["content_hash"]


def test_a_different_value_gives_a_different_circuit() -> None:
    low = bind_parameters(_parametric_ir(), {"theta": 0.1}, "ir")
    high = bind_parameters(_parametric_ir(), {"theta": 0.9}, "ir")

    assert low["content_hash"] != high["content_hash"]


def test_the_bound_circuit_survives_a_round_trip() -> None:
    from flagquantum_mcp_server.circuits import serialize

    bound = bind_parameters(_parametric_ir(), {"theta": 0.7}, "ir")

    assert serialize(bound["ir_json"], "ir")["content_hash"] == bound["content_hash"]


def test_binding_a_circuit_without_parameters_is_rejected(bell_qir: str) -> None:
    with pytest.raises(ToolInputError, match="no parameters"):
        bind_parameters(bell_qir, {"theta": 1.0}, "qir")


def test_a_missing_value_is_named() -> None:
    with pytest.raises(ToolInputError, match=r"Missing a value for \['theta'\]"):
        bind_parameters(_parametric_ir(), {}, "ir")


def test_an_unknown_value_is_named() -> None:
    with pytest.raises(ToolInputError, match=r"Unknown parameter\(s\) \['phi'\]"):
        bind_parameters(_parametric_ir(), {"theta": 0.5, "phi": 0.5}, "ir")


# --- layers ---


def test_layers_decompose_a_ghz_circuit(ghz3_qir: str) -> None:
    result = describe_layers(ghz3_qir, "qir")

    assert result["status"] == "success"
    assert result["n_layers"] == 3


def test_independent_gates_land_in_one_layer() -> None:
    parallel = json.dumps(
        [{"name": "h", "index": [0]}, {"name": "h", "index": [1]}, {"name": "h", "index": [2]}]
    )

    result = describe_layers(parallel, "qir")

    assert result["n_layers"] == 1
    assert result["layers"][0]["wires"] == [0, 1, 2]


def test_layer_count_matches_the_reported_depth(ghz3_qir: str) -> None:
    from flagquantum_mcp_server.analysis import analyze

    assert (
        describe_layers(ghz3_qir, "qir")["n_layers"]
        == analyze(ghz3_qir, "qir")["analysis"]["depth"]
    )


def test_each_layer_reports_its_gates(ghz3_qir: str) -> None:
    layers = describe_layers(ghz3_qir, "qir")["layers"]

    assert layers[0]["gates"] == [{"name": "h", "wires": [0]}]
    assert [entry["index"] for entry in layers] == [0, 1, 2]


# --- topology ---


def test_a_line_topology_is_a_chain() -> None:
    result = describe_topology(n_qubits=4, topology="line")

    assert result["topology"]["n_edges"] == 3
    assert [entry["degree"] for entry in result["adjacency"]] == [1, 2, 2, 1]


def test_a_ring_closes_the_chain() -> None:
    result = describe_topology(n_qubits=4, topology="ring")

    assert result["topology"]["n_edges"] == 4
    assert [entry["degree"] for entry in result["adjacency"]] == [2, 2, 2, 2]


def test_grid_dimensions_are_derived_when_omitted() -> None:
    result = describe_topology(n_qubits=6, topology="grid")

    assert result["topology"]["n_edges"] == 7


def test_adjacent_pairs_are_measured_by_default() -> None:
    result = describe_topology(n_qubits=4, topology="line")

    assert [entry["distance"] for entry in result["pairs"]] == [1, 1, 1]


def test_a_distant_pair_reports_its_distance_and_path() -> None:
    result = describe_topology(n_qubits=6, topology="grid", pairs=[[0, 5]])

    assert result["pairs"] == [{"from": 0, "to": 5, "distance": 3, "path": [0, 1, 2, 5]}]


def test_a_custom_topology_is_described_from_its_edges() -> None:
    result = describe_topology(n_qubits=3, topology="custom", edges=[[0, 1], [0, 2]])

    assert result["topology"]["edges"] == [[0, 1], [0, 2]]
    assert result["adjacency"][0]["neighbours"] == [1, 2]


def test_a_pair_outside_the_topology_is_rejected() -> None:
    with pytest.raises(ToolInputError, match=r"outside 0\.\.2"):
        describe_topology(n_qubits=3, topology="line", pairs=[[0, 9]])


def test_a_non_positive_width_is_rejected() -> None:
    with pytest.raises(ToolInputError, match="must be positive"):
        describe_topology(n_qubits=0)


def test_too_many_pairs_are_rejected() -> None:
    with pytest.raises(ToolInputError, match="At most 16 pairs"):
        describe_topology(n_qubits=40, topology="line", pairs=[[i, i + 1] for i in range(17)])


# --- drawing ---


def test_a_diagram_names_every_wire(ghz3_qir: str) -> None:
    result = draw_circuit(ghz3_qir, "qir")

    assert result["status"] == "success"
    assert result["n_qubits"] == 3
    assert len(result["diagram"].splitlines()) == 3
    assert result["diagram"].splitlines()[0].startswith("0:")


def test_the_diagram_shows_the_gates(ghz3_qir: str) -> None:
    diagram = draw_circuit(ghz3_qir, "qir")["diagram"]

    assert "H" in diagram
    assert "X" in diagram


def test_the_diagram_reports_the_source_hash(bell_qir: str) -> None:
    from flagquantum_mcp_server.circuits import serialize

    assert (
        draw_circuit(bell_qir, "qir")["content_hash"] == serialize(bell_qir, "qir")["content_hash"]
    )


def test_decimals_control_parameter_precision() -> None:
    circuit = fq.Circuit(1).rz(0, 0.123456789)
    payload = json.dumps(circuit.to_ir().to_dict())

    assert "0.12" in draw_circuit(payload, "ir", decimals=2)["diagram"]
    assert "0.123457" in draw_circuit(payload, "ir", decimals=6)["diagram"]


def test_negative_decimals_are_rejected(ghz3_qir: str) -> None:
    with pytest.raises(UnsupportedFormatError, match="decimals must not be negative"):
        draw_circuit(ghz3_qir, "qir", decimals=-1)


def test_a_non_positive_line_width_is_rejected(ghz3_qir: str) -> None:
    with pytest.raises(UnsupportedFormatError, match="line_width must be positive"):
        draw_circuit(ghz3_qir, "qir", line_width=0)
