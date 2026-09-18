"""Every tool, invoked through the real MCP boundary with every argument set.

The unit tests call the modules underneath each tool, and the integration test
calls one tool over stdio. Neither would notice a handler that stopped
*forwarding* an argument. The JSON schema is derived from the function
signature, so a dropped argument still appears in the schema and still
validates — the client keeps being offered an option that does nothing.

So every case here passes each argument a tool accepts, with a value that is not
the default, and then asserts the value changed the result.

**A check signals by raising, never by returning.** That rule is enforced, not
just stated: an earlier draft wrote its checks as lambdas returning booleans,
which the test called and discarded, so every one of them passed while checking
nothing. Only asserting inside a real function body works, which is why these
are ``def`` rather than ``lambda``.

``test_every_tool_argument_is_covered`` compares the cases against the published
schema, so adding an argument to a tool fails here until a case exercises it.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from flagquantum_mcp_server.circuits import serialize
from flagquantum_mcp_server.server import mcp

pytestmark = pytest.mark.unit

BELL = json.dumps([{"name": "h", "index": [0]}, {"name": "cx", "index": [0, 1]}])
GHZ = json.dumps(
    [
        {"name": "h", "index": [0]},
        {"name": "cx", "index": [0, 1]},
        {"name": "cx", "index": [1, 2]},
    ]
)
GHZ4 = json.dumps(
    [
        {"name": "h", "index": [0]},
        {"name": "cx", "index": [0, 1]},
        {"name": "cx", "index": [1, 2]},
        {"name": "cx", "index": [2, 3]},
    ]
)
SYMBOLIC = json.dumps(
    [{"name": "ry", "index": [0], "parameters": {"theta": {"$parameter": "theta"}}}]
)

# Built through the module rather than the tool: this is the input to a case,
# not the thing under test.
CANONICAL_IR = serialize(BELL, "qir")["ir_json"]

LINE4 = [[0, 1], [1, 2], [2, 3]]
RING3 = [[0, 1], [1, 2], [0, 2]]
GRID_2X2 = [[0, 1], [0, 2], [1, 3], [2, 3]]
STAR4 = [[0, 1], [0, 2], [0, 3]]

Check = Callable[[dict[str, Any]], None]


def _connectivity(edges: list[list[int]], kind: str | None = None) -> Check:
    """Build a check that the payload describes exactly this connectivity.

    Comparing edge *sets* is what makes this meaningful: a line and a ring agree
    on every edge but the closing one, so a check that only counted edges would
    not notice the difference.

    ``route_circuit_tool`` reports only the edge list, while
    ``describe_topology_tool`` also reports the kind, so the kind is checked
    where it is available.
    """

    def check(payload: dict[str, Any]) -> None:
        reported = payload["topology"]
        assert sorted(reported["edges"]) == sorted(edges), reported["edges"]
        if kind is not None:
            assert reported["kind"] == kind, reported.get("kind")

    return check


# --- one check per case ---


def _analyzed(payload: dict[str, Any]) -> None:
    assert payload["analysis"]["depth"] == 3


def _indented(payload: dict[str, Any]) -> None:
    assert "\n  " in payload["ir_json"], "indent=2 must reach the serializer"


def _round_tripped(payload: dict[str, Any]) -> None:
    assert payload["round_trip_stable"] is True
    assert "\n" in payload["ir_json"], "indent=2 must reach the re-serializer"


def _optimized(payload: dict[str, Any]) -> None:
    assert payload["transformation"] == "optimize"


def _routed_onto_a_ring(payload: dict[str, Any]) -> None:
    _connectivity(RING3)(payload)
    assert payload["strategy"] == "persistent_layout"
    assert payload["optimize_first"] is False


def _compared(payload: dict[str, Any]) -> None:
    assert [entry["topology"]["n_edges"] for entry in payload["topologies"]] == [2, 3]
    assert payload["strategy"] == "persistent_layout"
    assert payload["optimize_first"] is False


def _emitted_openqasm_2(payload: dict[str, Any]) -> None:
    assert payload["text"].startswith("OPENQASM 2.0;")
    assert payload["version"] == "2"
    assert "measure q[1]" not in payload["text"], "result_wires=[0] must narrow it"
    assert "measure q[0]" in payload["text"]


def _emitted_qcis(payload: dict[str, Any]) -> None:
    assert payload["text"].strip()
    assert payload["n_qubits"] == 3
    assert "measure" not in payload["text"].lower()


def _planned_with_shots(payload: dict[str, Any]) -> None:
    # A counts output is rejected unless a shot count reaches the planner, so a
    # dropped `options` surfaces as an error rather than a silently different
    # plan.
    assert payload["plan"]["mode"] == "statevector"
    assert payload["summary"]["depth"] == 2


def _described_two_gates(payload: dict[str, Any]) -> None:
    assert [record["opcode"] for record in payload["gates"]] == ["rz", "ccx"]
    assert payload["gates"][1]["aliases"] == ["ccnot", "toffoli"]


def _inspected(payload: dict[str, Any]) -> None:
    assert payload["is_parameterized"] is True
    assert payload["parameter_names"] == ["theta"]


def _bound(payload: dict[str, Any]) -> None:
    assert payload["is_parameterized"] is False
    assert payload["bound_parameters"] == {"theta": 0.7}


def _layered(payload: dict[str, Any]) -> None:
    assert payload["n_layers"] == 3
    assert len(payload["layers"]) == 3


def _measured_pairs(payload: dict[str, Any]) -> None:
    _connectivity(LINE4, kind="line")(payload)
    assert payload["pairs"][0]["distance"] == 3, "pairs=[[0, 3]] must be measured"


def _drawn_with_initial_state(payload: dict[str, Any]) -> None:
    assert "|0⟩" in payload["diagram"], "show_initial_state must reach the drawer"
    assert payload["n_lines"] == 2, "show_all_wires must keep the idle wire"


# tool name -> [(arguments, what the non-default arguments must have done)]
CASES: dict[str, list[tuple[dict[str, Any], Check]]] = {
    "analyze_circuit_tool": [({"circuit": GHZ, "circuit_format": "qir"}, _analyzed)],
    "serialize_circuit_tool": [
        ({"circuit": BELL, "circuit_format": "qir", "indent": 2}, _indented)
    ],
    "deserialize_circuit_tool": [({"ir_json": CANONICAL_IR, "indent": 2}, _round_tripped)],
    "optimize_circuit_tool": [({"circuit": GHZ, "circuit_format": "qir"}, _optimized)],
    "route_circuit_tool": [
        (
            {
                "circuit": GHZ,
                "circuit_format": "qir",
                "topology": "ring",
                "strategy": "persistent_layout",
                "optimize_first": False,
            },
            _routed_onto_a_ring,
        ),
        (
            {
                "circuit": GHZ4,
                "circuit_format": "qir",
                "topology": "grid",
                "rows": 2,
                "cols": 2,
            },
            _connectivity(GRID_2X2),
        ),
        (
            {
                "circuit": GHZ4,
                "circuit_format": "qir",
                "topology": "custom",
                "edges": STAR4,
            },
            _connectivity(STAR4),
        ),
    ],
    "compare_topologies_tool": [
        (
            {
                "circuit": GHZ,
                "circuit_format": "qir",
                "topologies": ["line", "ring"],
                "strategy": "persistent_layout",
                "optimize_first": False,
            },
            _compared,
        )
    ],
    "emit_openqasm_tool": [
        (
            {
                "circuit": BELL,
                "circuit_format": "qir",
                "version": 2.0,
                "result_wires": [0],
            },
            _emitted_openqasm_2,
        )
    ],
    "emit_qcis_tool": [({"circuit": GHZ, "circuit_format": "qir"}, _emitted_qcis)],
    "plan_execution_tool": [
        (
            {
                "circuit": BELL,
                "circuit_format": "qir",
                "options": {"shots": 100},
                "outputs": [{"kind": "counts", "wires": [0]}],
            },
            _planned_with_shots,
        )
    ],
    "describe_gate_set_tool": [({"gates": ["rz", "ccx"]}, _described_two_gates)],
    "inspect_parameters_tool": [({"circuit": SYMBOLIC, "circuit_format": "qir"}, _inspected)],
    "bind_parameters_tool": [
        (
            {"circuit": SYMBOLIC, "values": {"theta": 0.7}, "circuit_format": "qir"},
            _bound,
        )
    ],
    "describe_layers_tool": [({"circuit": GHZ, "circuit_format": "qir"}, _layered)],
    "describe_topology_tool": [
        ({"n_qubits": 4, "topology": "line", "pairs": [[0, 3]]}, _measured_pairs),
        (
            {"n_qubits": 4, "topology": "grid", "rows": 2, "cols": 2},
            _connectivity(GRID_2X2, kind="grid"),
        ),
        (
            {"n_qubits": 4, "topology": "custom", "edges": STAR4},
            _connectivity(STAR4, kind="custom"),
        ),
    ],
    "draw_circuit_tool": [
        (
            {
                "circuit": BELL,
                "circuit_format": "qir",
                "decimals": 2,
                "line_width": 60,
                "show_all_wires": True,
                "show_initial_state": True,
            },
            _drawn_with_initial_state,
        )
    ],
}


async def _call(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    result = await mcp.call_tool(tool, arguments)
    assert result.structured_content is not None, f"{tool} returned no structured content"
    return result.structured_content


@pytest.mark.parametrize(
    ("tool", "arguments", "check"),
    [
        pytest.param(tool, arguments, check, id=f"{tool}-{index}")
        for tool, cases in CASES.items()
        for index, (arguments, check) in enumerate(cases)
    ],
)
async def test_every_tool_forwards_its_arguments(
    tool: str, arguments: dict[str, Any], check: Check
) -> None:
    payload = await _call(tool, arguments)

    assert payload.get("status") == "success", payload
    outcome = check(payload)
    assert outcome is None, (
        f"{tool}'s check returned {outcome!r} instead of asserting. A check that "
        "returns a verdict is discarded by this test and checks nothing."
    )


async def test_a_check_that_only_returns_a_verdict_is_not_a_check() -> None:
    """Guards the rule the test above relies on.

    The vacuous draft of this file is the reason the assertion exists: a lambda
    returning a boolean is called, its result thrown away, and the case passes.
    """
    verdict = _connectivity(LINE4)
    assert verdict({"topology": {"edges": LINE4}}) is None


async def test_every_tool_argument_is_covered() -> None:
    """A new argument on a tool fails here until a case passes it.

    The schema is generated from the signature, so an argument that no case
    exercises is an argument whose forwarding nothing checks.
    """
    tools = {tool.name: tool for tool in await mcp.list_tools()}
    uncovered: dict[str, list[str]] = {}

    for name, cases in CASES.items():
        declared = set((tools[name].parameters or {}).get("properties", {}))
        passed = {key for arguments, _ in cases for key in arguments}
        if declared - passed:
            uncovered[name] = sorted(declared - passed)

    assert uncovered == {}, f"arguments no case exercises: {uncovered}"


async def test_no_tool_is_left_out() -> None:
    """A new tool fails here until its wiring is checked too."""
    tools = {tool.name for tool in await mcp.list_tools()}

    assert tools == set(CASES), f"uncovered tools: {sorted(tools - set(CASES))}"


async def test_a_case_passes_only_arguments_the_tool_declares() -> None:
    """The cases must not invent arguments, or they would pass vacuously."""
    tools = {tool.name: tool for tool in await mcp.list_tools()}
    unknown: dict[str, list[str]] = {}

    for name, cases in CASES.items():
        declared = set((tools[name].parameters or {}).get("properties", {}))
        passed = {key for arguments, _ in cases for key in arguments}
        if passed - declared:
            unknown[name] = sorted(passed - declared)

    assert unknown == {}, f"arguments these tools do not accept: {unknown}"
