"""Shared test configuration.

Pins BLAS and OpenMP thread counts before torch is imported, matching what the
FlagQuantum repository does in its own root conftest. Without it, a reduction
inside torch can pick a different summation order per run and make a numerical
assertion flap.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json

import pytest

BELL_QIR = [{"name": "h", "index": [0]}, {"name": "cx", "index": [0, 1]}]
GHZ3_QIR = [
    {"name": "h", "index": [0]},
    {"name": "cx", "index": [0, 1]},
    {"name": "cx", "index": [1, 2]},
]
ANGLED_QIR = [
    {"name": "ry", "index": [0], "parameters": {"theta": {"$parameter": "theta"}}},
    {"name": "cx", "index": [0, 1]},
]

# The server's published surface, declared once.
#
# It used to exist in three unlinked places — an in-process contract test, an
# over-stdio process test, and a ``len(tools) == N`` assertion in the CI
# workflow — so adding a tool turned two of them red and left the third to fail
# a release later. The CI script now imports this module rather than counting
# for itself, and both tests read it, so there is one thing to update.
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


@pytest.fixture
def bell_qir() -> str:
    """A two-qubit Bell circuit as a gate-list JSON string."""
    return json.dumps(BELL_QIR)


@pytest.fixture
def ghz3_qir() -> str:
    """A three-qubit GHZ circuit as a gate-list JSON string."""
    return json.dumps(GHZ3_QIR)


@pytest.fixture
def angled_qir() -> str:
    """A gate list whose angle is a symbol, not a number."""
    return json.dumps(ANGLED_QIR)
