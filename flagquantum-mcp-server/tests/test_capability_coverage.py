"""Coverage decisions must fail when a supported capability or tool is omitted."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ASSESS = runpy.run_path(
    str(Path(__file__).resolve().parents[1] / "tools" / "check_capability_coverage.py")
)["assess_coverage"]


def _baseline() -> tuple[dict[str, dict[str, str]], dict[str, dict[str, object]]]:
    capabilities = {
        "ir": {"level": "release_certified"},
        "circuit_drawing": {"level": "experimental"},
    }
    decisions = {
        "ir": {"disposition": "mapped", "tools": ["analyze_circuit_tool"], "scope": "IR analysis"},
        "circuit_drawing": {
            "disposition": "legacy_experimental",
            "tools": ["draw_circuit_tool"],
            "reason": "Existing tool only",
        },
    }
    return capabilities, decisions


def test_complete_bounded_coverage_passes() -> None:
    capabilities, decisions = _baseline()
    assert (
        ASSESS(capabilities, decisions, frozenset({"analyze_circuit_tool", "draw_circuit_tool"}))
        == []
    )


def test_new_production_capability_requires_an_explicit_decision() -> None:
    capabilities, decisions = _baseline()
    capabilities["new_operation"] = {"level": "production_supported"}
    errors = ASSESS(
        capabilities, decisions, frozenset({"analyze_circuit_tool", "draw_circuit_tool"})
    )
    assert "missing coverage decision: new_operation" in errors


def test_new_mcp_tool_requires_a_coverage_mapping() -> None:
    capabilities, decisions = _baseline()
    errors = ASSESS(
        capabilities,
        decisions,
        frozenset({"analyze_circuit_tool", "draw_circuit_tool", "new_tool"}),
    )
    assert "MCP tools without coverage: new_tool" in errors


def test_new_experimental_mapping_cannot_be_declared_as_covered() -> None:
    capabilities, decisions = _baseline()
    capabilities["experimental_new"] = {"level": "experimental"}
    decisions["experimental_new"] = {
        "disposition": "mapped",
        "tools": ["new_tool"],
        "scope": "Too early",
    }
    errors = ASSESS(
        capabilities,
        decisions,
        frozenset({"analyze_circuit_tool", "draw_circuit_tool", "new_tool"}),
    )
    assert "unapproved coverage decision: experimental_new" in errors
    assert "experimental_new: disposition 'mapped' is not allowed" in errors
