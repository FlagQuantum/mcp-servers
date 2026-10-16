"""Contract for the blocked, thin research-capability adapter design."""

from __future__ import annotations

import json
import os
import runpy
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest

from tests.surface import EXPECTED_RESOURCES, EXPECTED_TOOLS

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parent.parent
CONTRACT = ROOT / "contracts" / "research-capability-adapter-v1-candidate.json"
READINESS_CHECKER = ROOT / "tools" / "check_research_adapter_prerequisites.py"


def _candidate() -> dict[str, object]:
    payload = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_candidate_records_the_approved_and_activated_upstream_boundary() -> None:
    candidate = _candidate()
    upstream = candidate["upstream_contract"]

    assert candidate["status"] == "activated_local_research_surface"
    assert upstream == {
        "proposal": "API_CHANGE_PROPOSAL_070_RESEARCH_CAPABILITY_SERVICES.md",
        "machine_readable_contract": ("contracts/research-capability-services-v1-candidate.json"),
        "namespace": "flagquantum.services",
        "current_status": "approved_for_external_adapter",
        "current_maturity": "production_supported",
        "required_maturity_for_tool_registration": "production_supported",
        "upstream_approval_token": ("approve API_CHANGE_PROPOSAL_070_RESEARCH_CAPABILITY_SERVICES"),
        "upstream_approval_record_target": (
            "docs/api-changes/FQ-RESEARCH-CAPABILITY-SERVICES-20261013.md"
        ),
        "api_owner_approval_required": True,
        "services_domain_owner_approval_required": True,
    }


def test_candidate_maps_exactly_six_tools_to_six_upstream_operations() -> None:
    candidate = _candidate()
    tools = candidate["proposed_tools"]
    operations = candidate["upstream_operations"]

    assert len(tools) == len(operations) == 6
    assert {item["upstream_operation"] for item in tools} == set(operations)
    assert [item["name"] for item in tools] == [
        "search_capabilities_tool",
        "describe_capability_tool",
        "estimate_capability_tool",
        "validate_capability_input_tool",
        "execute_local_capability_tool",
        "validate_evidence_tool",
    ]
    assert {item["name"]: item["result_json_type"] for item in tools} == {
        "search_capabilities_tool": "array",
        "describe_capability_tool": "object",
        "estimate_capability_tool": "object",
        "validate_capability_input_tool": "object",
        "execute_local_capability_tool": "object",
        "validate_evidence_tool": "object",
    }


def test_candidate_owns_closed_protocol_shapes_but_not_nested_domain_schemas() -> None:
    tools = {item["name"]: item for item in _candidate()["proposed_tools"]}

    assert tools["execute_local_capability_tool"]["input_schema"] == {
        "type": "object",
        "properties": {
            "capability_id": {"type": "string"},
            "request": {"type": "object"},
            "options": {"type": "object"},
        },
        "required": ["capability_id", "request"],
        "additionalProperties": False,
    }
    for tool in tools.values():
        schema = tool["input_schema"]
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert set(schema["required"]).issubset(schema["properties"])
        for nested_name in {"query", "request", "options", "evidence"} & set(schema["properties"]):
            assert schema["properties"][nested_name] == {"type": "object"}


def test_candidate_envelope_preserves_upstream_results_and_errors_losslessly() -> None:
    envelope = _candidate()["tool_response_envelope"]

    assert envelope["success_schema"] == {
        "type": "object",
        "properties": {
            "status": {"const": "success"},
            "result": {"type": ["object", "array"]},
        },
        "required": ["status", "result"],
        "additionalProperties": False,
    }
    assert envelope["error_schema"] == {
        "type": "object",
        "properties": {
            "status": {"const": "error"},
            "error": {"type": "object"},
        },
        "required": ["status", "error"],
        "additionalProperties": False,
    }
    assert "without field omission or renaming" in envelope["result_policy"]
    assert "pre-result public API exceptions" in envelope["error_policy"]
    assert "transport-success" in envelope["domain_failure_policy"]


def test_protocol_exception_mapping_is_bounded_and_process_safe() -> None:
    mapping = _candidate()["protocol_exception_mapping"]

    assert mapping["rules"] == [
        {
            "python_exceptions": ["KeyError", "TypeError", "ValueError"],
            "code": "INVALID_INPUT",
            "message_policy": "use the public API exception message",
        },
        {
            "python_exceptions": ["ImportError", "ModuleNotFoundError"],
            "code": "SDK_UNAVAILABLE",
            "message_policy": "report that the installed FlagQuantum SDK is unavailable",
        },
    ]
    assert mapping["unexpected_exception_code"] == "INTERNAL_ERROR"
    assert mapping["unexpected_exception_message"] == (
        "FlagQuantum research capability call failed"
    )
    assert mapping["never_catch"] == [
        "KeyboardInterrupt",
        "SystemExit",
        "GeneratorExit",
    ]


def test_agent_workflow_is_complete_and_constraints_are_not_prompt_only() -> None:
    candidate = _candidate()
    tools = candidate["proposed_tools"]
    workflow = candidate["agent_workflow"]

    assert workflow["ordered_tools"] == [item["name"] for item in tools]
    assert workflow["prompt_is_not_authority"] is True
    descriptions = " ".join(item["description"] for item in tools)
    for required_phrase in (
        "Start here",
        "exact installed manifest",
        "before execution",
        "Continue only when",
        "local CPU path only",
        "provider_order",
        "transport success is not scientific success",
        "independent client or researcher acceptance",
    ):
        assert required_phrase in descriptions
    decisions = " ".join(workflow["required_decisions"])
    assert "credential-free" in decisions
    assert "fallback explicitly" in decisions
    assert "scientific gates" in decisions


def test_approved_candidate_is_registered_on_the_server_surface() -> None:
    proposed_tools = {item["name"] for item in _candidate()["proposed_tools"]}
    proposed_resources = {item["uri_template"] for item in _candidate()["proposed_resources"]}

    assert proposed_tools <= EXPECTED_TOOLS
    assert "flagquantum://capabilities/index" in EXPECTED_RESOURCES
    assert proposed_resources - {"flagquantum://capabilities/index"} == {
        "flagquantum://capabilities/{capability_id}/{version}",
        "flagquantum://schemas/{schema_id}",
    }


def test_adapter_invariants_keep_domain_logic_upstream() -> None:
    invariants = _candidate()["adapter_invariants"]

    assert invariants == {
        "structured_queries_only": True,
        "preserve_upstream_defaults": True,
        "preserve_all_evidence_fields": True,
        "preserve_upstream_domain_error_fields_inside_results": True,
        "protocol_exceptions_use_repository_error_codes": True,
        "schema_documents_come_from_installed_flagquantum": True,
        "no_schema_copy_or_redefinition": True,
        "no_domain_validation": True,
        "no_resource_estimation": True,
        "no_scientific_validation": True,
        "no_provider_routing": True,
        "no_candidate_scoring": True,
        "no_provider_dependency_probe": True,
        "no_provider_sdk_dependency": True,
        "forward_provider_order_verbatim": True,
        "forward_fallback_authorization_verbatim": True,
        "preserve_attempt_order_and_provider_errors": True,
        "preserve_provider_route_explanation": True,
        "closed_status_result_or_error_envelope": True,
        "no_adapter_retry_or_fallback": True,
        "no_network_egress": True,
        "no_credentials": True,
        "no_remote_or_paid_execution": True,
    }


def test_compatibility_range_matches_the_approved_upstream_release() -> None:
    compatibility = _candidate()["compatibility"]

    assert compatibility["current_package_range"] == "flagquantum>=0.2,<0.3"
    assert compatibility["released_flagquantum_version"] is None
    assert compatibility["candidate_range"] is None
    assert compatibility["version_mismatch_behavior"] == "fail_closed"


def test_activation_readiness_contract_is_fail_closed_and_non_activating() -> None:
    readiness = _candidate()["activation_readiness_contract"]

    assert readiness == {
        "approved_upstream_status": "approved_for_external_adapter",
        "approved_owner_value": "approved",
        "approval_record_marker": "Decision: approved",
        "allowed_maturity": ["production_supported", "release_certified"],
        "checker": "tools/check_research_adapter_prerequisites.py",
        "scope": (
            "read-only external prerequisites; passing does not register tools or prove "
            "adapter equivalence"
        ),
    }


def test_activation_checker_accepts_only_a_complete_external_ready_state() -> None:
    assess = runpy.run_path(str(READINESS_CHECKER))["assess_prerequisites"]
    adapter = deepcopy(_candidate())
    adapter["upstream_contract"]["current_status"] = "approved_for_external_adapter"
    adapter["upstream_contract"]["current_maturity"] = "production_supported"
    adapter["compatibility"]["released_flagquantum_version"] = "0.3.0"
    adapter["compatibility"]["candidate_range"] = "flagquantum>=0.3,<0.4"
    upstream = {
        "status": "approved_for_external_adapter",
        "approval": {
            "api_owner": "approved",
            "services_domain_owner": "approved",
            "external_execution_adapter_authorized": True,
            "approval_token": adapter["upstream_contract"]["upstream_approval_token"],
            "approval_record_target": adapter["upstream_contract"][
                "upstream_approval_record_target"
            ],
        },
    }
    maturity = {"capabilities": {"research_capability_services": {"level": "production_supported"}}}

    assert assess(adapter, upstream, maturity, "0.3.0", "Decision: approved") == []


def test_candidate_forwards_but_never_owns_provider_fallback() -> None:
    routing = _candidate()["required_provider_routing"]

    assert routing == {
        "capability_scope": ["ground_state.solve.vqe"],
        "default_provider_order": ["flagquantum-native"],
        "multiple_providers_require_authorization": True,
        "fallback_trigger": "provider_unavailable_only",
        "ordinary_execution_failure_triggers_fallback": False,
        "scientific_validation_failure_triggers_fallback": False,
        "installed_external_providers": ["qiskit-aer", "pennylane-lightning"],
        "candidate_scoring": "caller_order_rank_without_reordering",
        "route_explanation_is_content_hashed": True,
        "provider_backend_execution_estimate": True,
    }


def test_existing_overlap_is_quarantined_from_capability_adapters() -> None:
    overlap = _candidate()["existing_overlap"]

    assert {item["path"] for item in overlap} == {
        "src/flagquantum_mcp_server/planning.py",
        "src/flagquantum_mcp_server/training.py",
        "src/flagquantum_mcp_server/limits.py",
    }
    assert all("retain" in item["disposition"] for item in overlap)


def test_candidate_maps_resources_without_owning_schema_documents() -> None:
    candidate = _candidate()
    resources = candidate["proposed_resources"]

    assert candidate["upstream_schema_resource_operations"] == [
        "list_research_schemas",
        "describe_research_schema",
    ]
    assert candidate["required_schema_identities"] == [
        "flagquantum.pauli_sum.v1",
        "flagquantum.ground_state_vqe_request.v1",
        "flagquantum.ground_state_vqe_result.v1",
        "flagquantum.exact_ground_state_request.v1",
        "flagquantum.exact_sector_reference.v1",
        "flagquantum.energy_variance_request.v1",
        "flagquantum.energy_variance_reference.v1",
        "flagquantum.research_capability_manifest.v1",
        "flagquantum.research_validation_report.v1",
        "flagquantum.research_capability_error.v1",
        "flagquantum.research_capability_evidence_bundle.v1",
    ]
    assert resources == [
        {
            "uri_template": "flagquantum://capabilities/index",
            "upstream_operation": "search_research_capabilities",
            "adapter_scope": "serialize the installed structured capability index",
        },
        {
            "uri_template": ("flagquantum://capabilities/{capability_id}/{version}"),
            "upstream_operation": "describe_research_capability",
            "adapter_scope": "serialize the exact installed manifest",
        },
        {
            "uri_template": "flagquantum://schemas/{schema_id}",
            "upstream_operation": "describe_research_schema",
            "adapter_scope": (
                "serve the exact installed JSON Schema document without redefinition"
            ),
        },
    ]
    assert not any(
        path.suffix == ".json" and "schema" in path.name
        for path in (ROOT / "src" / "flagquantum_mcp_server").rglob("*")
    )


def test_optional_upstream_candidate_contract_has_exact_parity() -> None:
    source_root = os.environ.get("FLAGQUANTUM_SOURCE_ROOT")
    if source_root is None:
        pytest.skip("set FLAGQUANTUM_SOURCE_ROOT for cross-repository parity")

    candidate = _candidate()
    contract_path = Path(source_root) / candidate["upstream_contract"]["machine_readable_contract"]
    upstream = json.loads(contract_path.read_text(encoding="utf-8"))

    assert candidate["upstream_operations"] == upstream["operations"]
    assert set(candidate["upstream_schema_resource_operations"]) == set(
        upstream["schema_resource_operations"]
    )
    assert set(candidate["required_schema_identities"]) == set(
        upstream["schema_identities"].values()
    )
    assert candidate["required_provider_routing"] == {
        key: upstream["provider_routing"][key] for key in candidate["required_provider_routing"]
    }
    assert (
        candidate["upstream_contract"]["upstream_approval_token"]
        == upstream["approval"]["approval_token"]
    )
    assert (
        candidate["upstream_contract"]["upstream_approval_record_target"]
        == upstream["approval"]["approval_record_target"]
    )


def test_optional_tool_fields_and_required_sets_match_public_api_signatures() -> None:
    source_root = os.environ.get("FLAGQUANTUM_SOURCE_ROOT")
    if source_root is None:
        pytest.skip("set FLAGQUANTUM_SOURCE_ROOT for public API signature parity")

    operations = [item["upstream_operation"] for item in _candidate()["proposed_tools"]]
    probe = """
import inspect
import json
import sys

from flagquantum import services

operations = json.loads(sys.argv[1])
result = {}
for name in operations:
    signature = inspect.signature(getattr(services, name))
    result[name] = {
        "parameters": list(signature.parameters),
        "required": [
            key
            for key, value in signature.parameters.items()
            if value.default is inspect.Parameter.empty
        ],
    }
print(json.dumps(result, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [source_root, environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-c", probe, json.dumps(operations)],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    signatures = json.loads(completed.stdout)

    for tool in _candidate()["proposed_tools"]:
        schema = tool["input_schema"]
        signature = signatures[tool["upstream_operation"]]
        assert list(schema["properties"]) == signature["parameters"]
        assert schema["required"] == signature["required"]


def test_optional_six_operation_results_fit_lossless_candidate_envelopes() -> None:
    source_root = os.environ.get("FLAGQUANTUM_SOURCE_ROOT")
    if source_root is None:
        pytest.skip("set FLAGQUANTUM_SOURCE_ROOT for six-operation data parity")

    probe = """
import json

from flagquantum import services


def plain(value):
    if isinstance(value, tuple):
        return [plain(item) for item in value]
    if isinstance(value, list):
        return [plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if hasattr(value, "to_dict"):
        return plain(value.to_dict())
    return value


hamiltonian = services.PauliSum(
    2,
    (
        services.PauliSumTerm(-1.052373245772859, "II"),
        services.PauliSumTerm(0.39793742484318045, "ZI"),
        services.PauliSumTerm(-0.39793742484318045, "IZ"),
        services.PauliSumTerm(-0.01128010425623538, "ZZ"),
        services.PauliSumTerm(0.18093119978423156, "XX"),
    ),
)
request = services.GroundStateVQERequest(hamiltonian, fixed_hamming_weight=1)
evidence = services.execute_research_capability("ground_state.solve.vqe", request)
results = {
    "search_research_capabilities": services.search_research_capabilities(),
    "describe_research_capability": services.describe_research_capability(
        "ground_state.solve.vqe"
    ),
    "estimate_research_capability": services.estimate_research_capability(
        "ground_state.solve.vqe", request
    ),
    "validate_research_capability_input": services.validate_research_capability_input(
        "ground_state.solve.vqe", request
    ),
    "execute_research_capability": evidence,
    "validate_research_evidence": services.validate_research_evidence(
        "validate.energy_variance", evidence
    ),
}
print(json.dumps({key: plain(value) for key, value in results.items()}, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [source_root, environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    results = json.loads(completed.stdout)

    for tool in _candidate()["proposed_tools"]:
        result = results[tool["upstream_operation"]]
        expected_type = list if tool["result_json_type"] == "array" else dict
        assert isinstance(result, expected_type)
        envelope = {"status": "success", "result": result}
        assert json.loads(json.dumps(envelope, sort_keys=True)) == envelope
    assert results["execute_research_capability"]["schema"] == (
        "flagquantum.research_capability_evidence_bundle.v1"
    )
    assert results["validate_research_evidence"]["passed"] is True


def test_optional_pre_result_failures_map_without_swallowing_domain_reports() -> None:
    source_root = os.environ.get("FLAGQUANTUM_SOURCE_ROOT")
    if source_root is None:
        pytest.skip("set FLAGQUANTUM_SOURCE_ROOT for failure data parity")

    probe = """
import json

from flagquantum import services


def capture(call):
    try:
        value = call()
    except Exception as exc:
        return {"kind": "exception", "type": type(exc).__name__, "message": str(exc)}
    return {
        "kind": "result",
        "value": value.to_dict() if hasattr(value, "to_dict") else value,
    }


results = {
    "search": capture(lambda: services.search_research_capabilities({"prompt": "vqe"})),
    "describe": capture(lambda: services.describe_research_capability("missing")),
    "estimate": capture(
        lambda: services.estimate_research_capability("missing", {})
    ),
    "validate_input": capture(
        lambda: services.validate_research_capability_input("missing", {})
    ),
    "execute": capture(lambda: services.execute_research_capability("missing", {})),
    "validate_evidence": capture(
        lambda: services.validate_research_evidence("validate.energy_variance", {})
    ),
}
print(json.dumps(results, sort_keys=True))
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        [source_root, environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    results = json.loads(completed.stdout)
    mapping = {
        exception: rule["code"]
        for rule in _candidate()["protocol_exception_mapping"]["rules"]
        for exception in rule["python_exceptions"]
    }

    for operation in ("search", "describe", "estimate", "validate_input", "execute"):
        failure = results[operation]
        assert failure["kind"] == "exception"
        assert mapping[failure["type"]] == "INVALID_INPUT"
        envelope = {
            "status": "error",
            "error": {"code": mapping[failure["type"]], "message": failure["message"]},
        }
        assert set(envelope) == {"status", "error"}
        assert set(envelope["error"]) == {"code", "message"}

    evidence_validation = results["validate_evidence"]
    assert evidence_validation["kind"] == "result"
    assert evidence_validation["value"]["passed"] is False
    assert evidence_validation["value"]["blockers"][0]["code"] == ("INVALID_EVIDENCE_FIELDS")


def test_optional_activation_checker_reports_current_external_blockers() -> None:
    source_root = os.environ.get("FLAGQUANTUM_SOURCE_ROOT")
    if source_root is None:
        pytest.skip("set FLAGQUANTUM_SOURCE_ROOT for activation readiness")

    completed = subprocess.run(
        [
            sys.executable,
            str(READINESS_CHECKER),
            "--flagquantum-source-root",
            source_root,
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    result = json.loads(completed.stdout)

    assert completed.returncode == 0
    assert result["ready"] is True
    assert result["blockers"] == []
