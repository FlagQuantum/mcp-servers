"""Registered research tools stay thin, lossless, and scientifically equivalent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from flagquantum import services

from flagquantum_mcp_server.preconditions import plain
from flagquantum_mcp_server.server import mcp

pytestmark = pytest.mark.unit
_ROOT = Path(__file__).resolve().parents[1]


def _candidate() -> dict[str, Any]:
    return json.loads(
        (_ROOT / "contracts/research-capability-adapter-v1-candidate.json").read_text(
            encoding="utf-8"
        )
    )


def _request() -> dict[str, Any]:
    return services.GroundStateVQERequest(
        services.PauliSum(
            2,
            (
                services.PauliSumTerm(-1.052373245772859, "II"),
                services.PauliSumTerm(0.39793742484318045, "ZI"),
                services.PauliSumTerm(-0.39793742484318045, "IZ"),
                services.PauliSumTerm(-0.01128010425623538, "ZZ"),
                services.PauliSumTerm(0.18093119978423156, "XX"),
            ),
        ),
        fixed_hamming_weight=1,
    ).to_dict()


async def _result(tool: str, arguments: dict[str, Any]) -> Any:
    call = await mcp.call_tool(tool, arguments)
    assert call.structured_content is not None
    assert call.structured_content["status"] == "success"
    return call.structured_content["result"]


async def test_registered_tool_schemas_match_the_approved_candidate() -> None:
    registered = {tool.name: tool for tool in await mcp.list_tools()}

    for expected in _candidate()["proposed_tools"]:
        actual = registered[expected["name"]].parameters
        assert actual is not None
        assert actual["additionalProperties"] is False
        assert set(actual["properties"]) == set(expected["input_schema"]["properties"])
        assert actual.get("required", []) == expected["input_schema"]["required"]


async def test_six_tool_workflow_preserves_the_upstream_results() -> None:
    request = _request()
    candidates = await _result("search_capabilities_tool", {"query": {}})
    manifest = await _result(
        "describe_capability_tool",
        {"capability_id": "ground_state.solve.vqe", "version": "1.0.0"},
    )
    estimate = await _result(
        "estimate_capability_tool",
        {
            "capability_id": "ground_state.solve.vqe",
            "request": request,
            "target": "local-cpu",
        },
    )
    validation = await _result(
        "validate_capability_input_tool",
        {"capability_id": "ground_state.solve.vqe", "request": request},
    )
    evidence = await _result(
        "execute_local_capability_tool",
        {
            "capability_id": "ground_state.solve.vqe",
            "request": request,
            "options": {
                "provider_order": ["flagquantum-native"],
                "allow_provider_fallback": False,
            },
        },
    )
    evidence_validation = await _result(
        "validate_evidence_tool",
        {"validator_id": "validate.energy_variance", "evidence": evidence},
    )

    assert candidates == plain(services.search_research_capabilities({}))
    assert manifest == plain(
        services.describe_research_capability("ground_state.solve.vqe", "1.0.0")
    )
    assert estimate == plain(
        services.estimate_research_capability("ground_state.solve.vqe", request, target="local-cpu")
    )
    assert validation == plain(
        services.validate_research_capability_input("ground_state.solve.vqe", request)
    )
    assert evidence["status"] == "succeeded"
    assert evidence["capability"]["provider"] == "flagquantum-native"
    assert evidence_validation == plain(
        services.validate_research_evidence("validate.energy_variance", evidence)
    )
    assert evidence_validation["passed"] is True


async def test_mcp_and_direct_execution_have_the_same_scientific_projection() -> None:
    request = _request()
    mcp_evidence = await _result(
        "execute_local_capability_tool",
        {"capability_id": "ground_state.solve.vqe", "request": request},
    )
    direct = services.execute_research_capability("ground_state.solve.vqe", request).to_dict()

    for key in ("capability", "status", "result", "metrics", "validation", "warnings", "error"):
        assert mcp_evidence[key] == plain(direct[key])
    assert mcp_evidence["execution"]["provider_order"] == direct["execution"]["provider_order"]
    assert mcp_evidence["execution"]["fallback_used"] is False


async def test_pre_result_failure_uses_the_closed_protocol_error() -> None:
    call = await mcp.call_tool(
        "describe_capability_tool",
        {"capability_id": "missing"},
    )

    assert call.structured_content is not None
    assert call.structured_content["status"] == "error"
    assert call.structured_content["error"]["code"] == "INVALID_INPUT"
    assert set(call.structured_content) == {"status", "error"}


async def test_research_resources_come_from_the_installed_sdk() -> None:
    index = await mcp.read_resource("flagquantum://capabilities/index")
    manifest = await mcp.read_resource("flagquantum://capabilities/ground_state.solve.vqe/1.0.0")
    schema = await mcp.read_resource(
        "flagquantum://schemas/flagquantum.ground_state_vqe_request.v1"
    )

    assert json.loads(index.contents[0].content) == plain(services.search_research_capabilities())
    assert json.loads(manifest.contents[0].content) == plain(
        services.describe_research_capability("ground_state.solve.vqe", "1.0.0")
    )
    assert json.loads(schema.contents[0].content) == plain(
        services.describe_research_schema("flagquantum.ground_state_vqe_request.v1")
    )
