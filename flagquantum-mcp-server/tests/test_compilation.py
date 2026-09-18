"""Optimization, routing and topology comparison."""

from __future__ import annotations

import json

import pytest

from flagquantum_mcp_server.circuits import serialize
from flagquantum_mcp_server.compilation import (
    build_coupling_map,
    compare_topologies,
    grid_shape,
    optimize_circuit,
    route_circuit,
)
from flagquantum_mcp_server.errors import (
    ToolInputError,
    ToolLimitError,
    UnsupportedFormatError,
)

pytestmark = pytest.mark.unit


def _non_adjacent_ghz() -> str:
    """A circuit whose two-qubit gates skip a wire, forcing SWAP insertion."""
    return json.dumps(
        [
            {"name": "h", "index": [0]},
            {"name": "cx", "index": [0, 2]},
        ]
    )


def test_optimize_returns_the_ir_and_a_before_after(bell_qir: str) -> None:
    result = optimize_circuit(bell_qir, "qir")

    assert result["status"] == "success"
    assert result["transformation"] == "optimize"
    assert result["before"]["analysis"]["n_instructions"] == 2
    assert result["after"]["analysis"]["n_instructions"] == 2
    assert result["instruction_delta"] == 0
    assert json.loads(result["ir_json"])["kind"] == "flagquantum.circuit_ir"


def test_optimize_removes_redundant_gates() -> None:
    redundant = json.dumps(
        [
            {"name": "h", "index": [0]},
            {"name": "h", "index": [0]},
            {"name": "x", "index": [1]},
            {"name": "x", "index": [1]},
        ]
    )

    result = optimize_circuit(redundant, "qir")

    assert result["after"]["analysis"]["n_instructions"] < 4
    assert result["instruction_delta"] < 0
    assert result["content_hash_changed"] is True


def test_route_on_a_line_topology_of_the_same_width_is_free(bell_qir: str) -> None:
    result = route_circuit(bell_qir, "qir", topology="line")

    assert result["status"] == "success"
    assert result["topology"]["n_wires"] == 2
    assert result["instruction_delta"] == 0


def test_routing_inserts_swaps_when_the_topology_forbids_a_gate() -> None:
    result = route_circuit(_non_adjacent_ghz(), "qir", topology="line")

    assert result["before"]["analysis"]["two_qubit_gates"] == 1
    assert result["after"]["analysis"]["two_qubit_gates"] > 1
    assert result["instruction_delta"] > 0
    assert result["after"]["analysis"]["gate_counts"].get("swap", 0) > 0


def test_routing_can_skip_optimization() -> None:
    result = route_circuit(_non_adjacent_ghz(), "qir", topology="line", optimize_first=False)

    assert result["optimize_first"] is False


def test_custom_topology_accepts_an_explicit_edge_list(bell_qir: str) -> None:
    result = route_circuit(bell_qir, "qir", topology="custom", edges=[[0, 1]])

    assert result["topology"]["edges"] == [[0, 1]]
    assert result["topology"]["n_edges"] == 1


def test_grid_dimensions_are_derived_when_omitted(ghz3_qir: str) -> None:
    result = route_circuit(ghz3_qir, "qir", topology="grid")

    assert result["topology"]["n_wires"] == 3
    assert result["status"] == "success"


def test_grid_shape_prefers_the_most_square_factorization() -> None:
    assert grid_shape(4) == (2, 2)
    assert grid_shape(6) == (2, 3)
    assert grid_shape(12) == (3, 4)
    assert grid_shape(3) == (1, 3)
    assert grid_shape(7) == (1, 7)


def test_explicit_grid_must_match_the_circuit_width(ghz3_qir: str) -> None:
    with pytest.raises(ToolInputError, match="holds 4 wires, but the circuit has 3"):
        route_circuit(ghz3_qir, "qir", topology="grid", rows=2, cols=2)


def test_half_specified_grid_is_rejected(ghz3_qir: str) -> None:
    with pytest.raises(ToolInputError, match="needs both rows and cols, or neither"):
        route_circuit(ghz3_qir, "qir", topology="grid", rows=1)


def test_unknown_strategy_is_rejected(bell_qir: str) -> None:
    with pytest.raises(UnsupportedFormatError, match="strategy must be one of"):
        route_circuit(bell_qir, "qir", strategy="teleport")


def test_persistent_layout_is_accepted(bell_qir: str) -> None:
    result = route_circuit(bell_qir, "qir", strategy="persistent_layout")

    assert result["strategy"] == "persistent_layout"


def test_unknown_topology_is_rejected(bell_qir: str) -> None:
    with pytest.raises(UnsupportedFormatError, match="topology must be one of"):
        build_coupling_map("torus", n_wires=2)


def test_custom_topology_requires_edges(bell_qir: str) -> None:
    with pytest.raises(ToolInputError, match="requires a non-empty edges list"):
        route_circuit(bell_qir, "qir", topology="custom")


def test_edge_outside_the_circuit_is_rejected(bell_qir: str) -> None:
    with pytest.raises(ToolInputError, match=r"outside 0\.\.1"):
        route_circuit(bell_qir, "qir", topology="custom", edges=[[0, 9]])


def test_self_loop_edge_is_rejected(bell_qir: str) -> None:
    with pytest.raises(ToolInputError, match="cannot join a wire to itself"):
        route_circuit(bell_qir, "qir", topology="custom", edges=[[1, 1]])


def test_malformed_edge_is_rejected(bell_qir: str) -> None:
    with pytest.raises(ToolInputError, match="exactly two wires"):
        route_circuit(bell_qir, "qir", topology="custom", edges=[[0, 1, 2]])


def test_compare_topologies_covers_line_ring_and_grid_by_default(ghz3_qir: str) -> None:
    result = compare_topologies(ghz3_qir, "qir")

    assert result["status"] == "success"
    assert [entry["topology"]["n_wires"] for entry in result["topologies"]] == [3, 3, 3]
    assert len(result["topologies"]) == 3
    assert result["cheapest"] in [entry["topology"] for entry in result["topologies"]]


def test_compare_topologies_accepts_a_subset(bell_qir: str) -> None:
    result = compare_topologies(bell_qir, "qir", topologies=["line", "ring"])

    assert len(result["topologies"]) == 2


def test_compare_topologies_rejects_an_empty_selection(bell_qir: str) -> None:
    with pytest.raises(ToolInputError, match="at least one topology"):
        compare_topologies(bell_qir, "qir", topologies=[])


def test_compare_topologies_rejects_custom(bell_qir: str) -> None:
    with pytest.raises(UnsupportedFormatError, match="cannot be compared by name"):
        compare_topologies(bell_qir, "qir", topologies=["custom"])


def test_compare_topologies_enforces_its_budget(
    monkeypatch: pytest.MonkeyPatch, bell_qir: str
) -> None:
    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_COMPARE_TOPOLOGIES", "2")

    with pytest.raises(ToolLimitError, match="at most 2 topologies"):
        compare_topologies(bell_qir, "qir", topologies=["line", "ring", "grid"])


def test_routed_ir_is_accepted_back_as_input(ghz3_qir: str) -> None:
    routed = route_circuit(ghz3_qir, "qir", topology="line")

    reanalyzed = serialize(routed["ir_json"], "ir")

    assert reanalyzed["content_hash"] == routed["content_hash"]
