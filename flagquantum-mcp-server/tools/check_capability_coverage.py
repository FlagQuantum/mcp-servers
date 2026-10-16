"""Check explicit MCP coverage decisions against a pinned FlagQuantum release."""

from __future__ import annotations

import argparse
import json
import runpy
import subprocess
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib


SERVER_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = SERVER_ROOT / "contracts" / "sdk-capability-coverage-v1.json"
SURFACE_PATH = SERVER_ROOT / "tests" / "surface.py"
ELIGIBLE_LEVELS = frozenset({"production_supported", "release_certified"})
LEGACY_EXCEPTIONS = {"circuit_drawing": "draw_circuit_tool"}


def _decision_errors(name: str, decision: dict[str, Any], eligible: set[str]) -> list[str]:
    disposition = decision.get("disposition")
    tools = decision.get("tools", [])
    if not isinstance(tools, list) or any(not isinstance(tool, str) for tool in tools):
        return [f"{name}: tools must be a list of names"]
    if disposition == "mapped" and name in eligible:
        return (
            []
            if tools and decision.get("scope")
            else [f"{name}: mapped capability needs tools and bounded scope"]
        )
    if disposition == "deferred" and name in eligible:
        return (
            []
            if not tools and decision.get("reason")
            else [f"{name}: deferred capability needs a reason and no tools"]
        )
    if disposition == "legacy_experimental" and name in LEGACY_EXCEPTIONS:
        return (
            []
            if tools == [LEGACY_EXCEPTIONS[name]] and decision.get("reason")
            else [f"{name}: legacy exception must name its existing tool and reason"]
        )
    return [f"{name}: disposition {disposition!r} is not allowed"]


def assess_coverage(
    upstream_capabilities: dict[str, Any],
    decisions: dict[str, Any],
    expected_tools: frozenset[str],
) -> list[str]:
    """Find missing decisions and inconsistent tool mappings without importing the SDK."""
    eligible = {
        name
        for name, capability in upstream_capabilities.items()
        if capability["level"] in ELIGIBLE_LEVELS
    }
    expected_ids = eligible | LEGACY_EXCEPTIONS.keys()
    errors = [
        f"missing coverage decision: {name}" for name in sorted(expected_ids - decisions.keys())
    ]
    errors.extend(
        f"unapproved coverage decision: {name}" for name in sorted(decisions.keys() - expected_ids)
    )
    for name in LEGACY_EXCEPTIONS:
        if name not in upstream_capabilities:
            errors.append(f"legacy exception absent from upstream: {name}")
        elif upstream_capabilities[name]["level"] in ELIGIBLE_LEVELS:
            errors.append(f"legacy exception is now eligible and needs a normal decision: {name}")

    mapped_tools: list[str] = []
    for name, decision in sorted(decisions.items()):
        if name not in upstream_capabilities:
            continue
        tools = decision.get("tools", [])
        if not isinstance(tools, list) or any(not isinstance(tool, str) for tool in tools):
            errors.append(f"{name}: tools must be a list of names")
            continue
        errors.extend(_decision_errors(name, decision, eligible))
        mapped_tools.extend(tools)

    duplicates = {tool for tool in mapped_tools if mapped_tools.count(tool) > 1}
    if duplicates:
        errors.append(f"tools mapped more than once: {', '.join(sorted(duplicates))}")
    missing_tools = expected_tools - set(mapped_tools)
    unknown_tools = set(mapped_tools) - expected_tools
    if missing_tools:
        errors.append(f"MCP tools without coverage: {', '.join(sorted(missing_tools))}")
    if unknown_tools:
        errors.append(f"coverage names unknown MCP tools: {', '.join(sorted(unknown_tools))}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-root", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    upstream = contract["upstream"]
    source_root = args.upstream_root.resolve()
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True
    ).strip()
    errors: list[str] = []
    if commit != upstream["commit"]:
        errors.append(f"upstream source is {commit}, expected {upstream['commit']}")
    with (source_root / upstream["manifest"]).open("rb") as source:
        capabilities = tomllib.load(source)["capabilities"]
    surface = runpy.run_path(str(SURFACE_PATH))["EXPECTED_TOOLS"]
    errors.extend(assess_coverage(capabilities, contract["capabilities"], surface))
    print(
        json.dumps(
            {
                "schema": "flagquantum_mcp.sdk_capability_coverage_check.v1",
                "coverage_complete": not errors,
                "upstream_commit": commit,
                "eligible_capabilities": sum(
                    capability["level"] in ELIGIBLE_LEVELS for capability in capabilities.values()
                ),
                "mapped_tools": len(surface),
                "errors": errors,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
