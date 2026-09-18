"""The server surface this repository declares, and nothing else.

Kept apart from ``conftest.py`` because two different readers need it and only
one of them has pytest. The CI job that smoke-tests the built wheel installs
the wheel into a fresh environment with no development dependencies, then
asserts that the surface it serves is the surface declared here — so importing
this module must not drag in a test framework. It did, via ``conftest``, and
that job went red on a release commit for a reason that had nothing to do with
the wheel.

No imports at all, then, and no logic. This is a declaration.
"""

from __future__ import annotations

# The tools, resources and prompts this server publishes.
#
# Declared once, on purpose. It used to exist in three unlinked places — an
# in-process contract test, an over-stdio process test, and a ``len(tools)``
# assertion in the CI workflow — so adding a tool turned two of them red and
# left the third to fail a release later, which is what happened. Anything that
# needs to know the surface reads it from here.
EXPECTED_TOOLS = frozenset(
    {
        "analyze_circuit_tool",
        "serialize_circuit_tool",
        "deserialize_circuit_tool",
        "optimize_circuit_tool",
        "route_circuit_tool",
        "compare_topologies_tool",
        "emit_openqasm_tool",
        "emit_qcis_tool",
        "plan_execution_tool",
        "simulate_circuit_tool",
        "describe_gate_set_tool",
        "inspect_parameters_tool",
        "bind_parameters_tool",
        "describe_layers_tool",
        "describe_topology_tool",
        "draw_circuit_tool",
    }
)

EXPECTED_RESOURCES = frozenset(
    {
        "flagquantum://version",
        "flagquantum://gate-set",
        "flagquantum://ir-schema",
    }
)

EXPECTED_PROMPTS = frozenset(
    {
        "build_and_analyze_circuit",
        "compile_for_topology",
        "export_circuit",
    }
)
