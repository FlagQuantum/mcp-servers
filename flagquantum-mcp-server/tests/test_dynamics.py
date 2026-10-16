from __future__ import annotations

import math

import pytest

from flagquantum_mcp_server.dynamics import evolve_dynamics, plan_evolution
from flagquantum_mcp_server.server import mcp

pytestmark = pytest.mark.unit


def _request() -> dict[str, object]:
    return {
        "hamiltonian": [
            {"pauli": "X", "coefficient": 0.5, "wires": [0]},
            {"pauli": "Z", "coefficient": 0.1, "wires": [0]},
        ],
        "initial_state": "1",
        "n_wires": 1,
        "times": [index * 0.05 for index in range(161)],
        "collapse_operators": [{"operator": "amplitude_damping", "rate": 0.1, "wire": 0}],
        "observables": [{"name": "z", "pauli": "Z", "wires": [0]}],
    }


def test_plan_and_evolution_share_the_explicit_time_grid() -> None:
    request = _request()

    plan = plan_evolution(**request)
    result = evolve_dynamics(**request)

    assert plan["plan"]["n_times"] == 161
    assert len(result["times"]) == 161
    assert len(result["populations"]) == 161
    assert result["population_bounded"] is True
    assert result["maximum_trace_drift"] <= result["numerics"]["trace_tolerance"]
    assert result["execution"] == plan["execution"]


def test_zero_hamiltonian_reports_exponential_decay() -> None:
    times = [index * 0.05 for index in range(161)]
    result = evolve_dynamics(
        [{"pauli": "I", "coefficient": 0.0, "wires": [0]}],
        "1",
        1,
        times,
        [{"operator": "amplitude_damping", "rate": 0.1, "wire": 0}],
    )

    for time, row in zip(times, result["populations"], strict=True):
        assert row[1] == pytest.approx(math.exp(-0.1 * time), abs=1e-8)


def test_negative_rate_is_a_structured_tool_error() -> None:
    from flagquantum_mcp_server.errors import ToolError

    request = _request()
    request["collapse_operators"] = [{"operator": "amplitude_damping", "rate": -0.1, "wire": 0}]

    with pytest.raises(ToolError) as caught:
        plan_evolution(**request)

    assert caught.value.code == "negative_rate"


async def test_negative_rate_crosses_mcp_as_a_structured_error() -> None:
    request = _request()
    request["collapse_operators"] = [{"operator": "amplitude_damping", "rate": -0.1, "wire": 0}]

    result = await mcp.call_tool("plan_evolution_tool", request)

    assert result.structured_content is not None
    assert result.structured_content["status"] == "error"
    assert result.structured_content["error"]["code"] == "negative_rate"
