#!/usr/bin/env python3
"""Fail-closed check for the external prerequisites to activate research tools."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.version import Version

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10 CI
    import tomli as tomllib


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTER_CONTRACT = ROOT / "contracts/research-capability-adapter-v1-candidate.json"


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _toml_object(path: Path) -> dict[str, Any]:
    with path.open("rb") as stream:
        value = tomllib.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a TOML table")
    return value


def _project_version(source_root: Path, project_document: dict[str, Any]) -> str:
    project = project_document["project"]
    static_version = project.get("version")
    if static_version is not None:
        return str(static_version)

    attr = project_document["tool"]["setuptools"]["dynamic"]["version"]["attr"]
    *module_parts, attribute = str(attr).split(".")
    module_path = source_root.joinpath(*module_parts).with_suffix(".py")
    module = ast.parse(module_path.read_text(encoding="utf-8"), filename=str(module_path))
    for statement in module.body:
        if not isinstance(statement, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == attribute for target in statement.targets
        ):
            value = ast.literal_eval(statement.value)
            if isinstance(value, str):
                return value
    raise ValueError(f"could not resolve dynamic project version {attr!r}")


def assess_prerequisites(
    adapter: dict[str, Any],
    upstream: dict[str, Any],
    maturity: dict[str, Any],
    upstream_version: str,
    approval_record: str,
) -> list[str]:
    """Return every external blocker; an empty list means activation may begin."""

    expected = adapter["activation_readiness_contract"]
    adapter_upstream = adapter["upstream_contract"]
    blockers = _approval_blockers(
        expected,
        adapter_upstream,
        upstream,
        approval_record,
    )
    blockers.extend(_maturity_blockers(expected, adapter_upstream, maturity))
    blockers.extend(_compatibility_blockers(adapter["compatibility"], upstream_version))
    return blockers


def _approval_blockers(
    expected: dict[str, Any],
    adapter_upstream: dict[str, Any],
    upstream: dict[str, Any],
    approval_record: str,
) -> list[str]:
    blockers: list[str] = []
    approval = upstream["approval"]

    if upstream["status"] != adapter_upstream["current_status"]:
        blockers.append("adapter current_status does not match the upstream contract")
    if upstream["status"] != expected["approved_upstream_status"]:
        blockers.append("upstream API contract is not approved for an external adapter")

    approved_owner_value = expected["approved_owner_value"]
    blockers.extend(
        f"upstream approval.{owner_field} is not approved"
        for owner_field in ("api_owner", "services_domain_owner")
        if approval[owner_field] != approved_owner_value
    )
    if approval["external_execution_adapter_authorized"] is not True:
        blockers.append("upstream external execution adapter is not authorized")

    if approval["approval_token"] != adapter_upstream["upstream_approval_token"]:
        blockers.append("approval token differs between adapter and upstream contracts")
    if approval["approval_record_target"] != adapter_upstream["upstream_approval_record_target"]:
        blockers.append("approval record target differs between adapter and upstream contracts")
    if expected["approval_record_marker"] not in approval_record:
        blockers.append("approval record does not contain the pinned approved marker")
    return blockers


def _maturity_blockers(
    expected: dict[str, Any],
    adapter_upstream: dict[str, Any],
    maturity: dict[str, Any],
) -> list[str]:
    blockers: list[str] = []
    capability = maturity["capabilities"]["research_capability_services"]
    level = capability["level"]
    if level != adapter_upstream["current_maturity"]:
        blockers.append("adapter current_maturity does not match capability-maturity.toml")
    if level not in expected["allowed_maturity"]:
        blockers.append(f"upstream maturity {level!r} does not permit tool registration")
    return blockers


def _compatibility_blockers(
    compatibility: dict[str, Any],
    upstream_version: str,
) -> list[str]:
    blockers: list[str] = []
    released_version = compatibility["released_flagquantum_version"]
    candidate_range = compatibility["candidate_range"]
    if released_version is None:
        blockers.append("released FlagQuantum version is not recorded")
    elif Version(released_version) != Version(upstream_version):
        blockers.append("recorded released version does not match upstream pyproject version")

    if candidate_range is None:
        blockers.append("candidate compatible FlagQuantum range is not recorded")
    else:
        requirement = Requirement(candidate_range)
        if requirement.name != "flagquantum":
            blockers.append("candidate compatibility requirement is not for flagquantum")
        elif (
            released_version is not None and Version(released_version) not in requirement.specifier
        ):
            blockers.append("released FlagQuantum version is outside candidate compatibility range")

    return blockers


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flagquantum-source-root", type=Path, required=True)
    parser.add_argument(
        "--adapter-contract",
        type=Path,
        default=DEFAULT_ADAPTER_CONTRACT,
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    adapter = _json_object(args.adapter_contract)
    upstream_root = args.flagquantum_source_root.resolve()
    upstream_contract_path = (
        upstream_root / adapter["upstream_contract"]["machine_readable_contract"]
    )
    upstream = _json_object(upstream_contract_path)
    maturity = _toml_object(upstream_root / "capability-maturity.toml")
    project = _toml_object(upstream_root / "pyproject.toml")
    approval_record_path = upstream_root / upstream["approval"]["approval_record_target"]
    approval_record = approval_record_path.read_text(encoding="utf-8")

    blockers = assess_prerequisites(
        adapter,
        upstream,
        maturity,
        _project_version(upstream_root, project),
        approval_record,
    )
    result = {
        "schema": "flagquantum_mcp.research_adapter_activation_readiness.v1",
        "ready": not blockers,
        "upstream_source_root": str(upstream_root),
        "blockers": blockers,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not blockers else 1


if __name__ == "__main__":
    raise SystemExit(main())
