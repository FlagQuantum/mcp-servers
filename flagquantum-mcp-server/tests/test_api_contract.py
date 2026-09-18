"""Pin the FlagQuantum contracts this server depends on.

Every import path and name below is a promise this repository makes to its
users. If FlagQuantum moves one, this module fails and the build stops —
rather than a tool failing at run time in someone's agent session.

The three tiers matter. The frozen ``stable_exports`` snapshot is the strongest
contract; ``flagquantum.compiler`` is a documented public module that is not in
that snapshot; the two emitters are public submodule functions with no
``__all__`` at all. This file records which tier each tool rests on, so a
deliberate promotion or demotion in FlagQuantum shows up as a test change.
"""

from __future__ import annotations

import dataclasses
import importlib

import flagquantum as fq
import pytest

pytestmark = pytest.mark.unit

# Tier 1: names in docs/public_api_v1.json's stable_exports.
TIER1_FROZEN = (
    "Circuit",
    "CircuitIR",
    "ExecutionOptions",
    "ExecutionPlan",
    "Instruction",
    "IR_VERSION",
    "plan",
    # Named by the training tool through a helper rather than at the call site,
    # and in the README's tier-1 row, so it belongs here where a reader looks to
    # ask which promise the SDK has made. The snapshot fixes the name only, which
    # is why the tool's own test pins the default observable too.
    "RuntimePolicy",
)

# Tier 2: public module with an explicit __all__, not in the frozen snapshot.
TIER2_COMPILER = ("CouplingMap", "optimize", "route_to_topology")

# Tier 3: public submodule functions, no __all__.
TIER3_EMITTERS = (
    ("flagquantum.compiler.openqasm", "emit_openqasm"),
    ("flagquantum.compiler.qcis", "emit_qcis"),
)

# Tier 3: public names a package declares in its own __all__, not in the frozen
# snapshot. The training objective needs both, and there is no other way to say
# what to minimise — see docs/superpowers/specs/2026-09-18-parameter-training-design.md.
TIER3_TRAINING = (
    ("flagquantum.algorithms", "Hamiltonian"),
    ("flagquantum.algorithms", "pauli_term"),
)

# The analysis fields this server reports. Extra fields are fine; these are the
# ones a prompt or an example may name.
REQUIRED_ANALYSIS_FIELDS = frozenset(
    {
        "n_wires",
        "n_instructions",
        "depth",
        "gate_counts",
        "two_qubit_gates",
        "multi_qubit_gates",
    }
)


@pytest.mark.parametrize("name", TIER1_FROZEN)
def test_frozen_exports_still_exist(name: str) -> None:
    assert hasattr(fq, name), f"stable export {name} disappeared"


def test_ir_version_is_the_one_we_report() -> None:
    assert fq.IR_VERSION == "1.0"


def test_circuit_methods_this_server_calls_exist() -> None:
    for method in ("to_ir", "from_ir", "to_qir", "from_qir", "analysis"):
        assert hasattr(fq.Circuit, method), f"Circuit.{method} disappeared"


def test_ir_serialization_methods_exist() -> None:
    for method in ("to_dict", "to_json", "from_dict", "from_json"):
        assert hasattr(fq.CircuitIR, method), f"CircuitIR.{method} disappeared"

    assert isinstance(fq.CircuitIR.content_hash, property)


def test_ir_envelope_keys_are_what_we_document() -> None:
    payload = fq.Circuit(2).h(0).cx(0, 1).to_ir().to_dict()

    assert payload["kind"] == "flagquantum.circuit_ir"
    assert set(payload) == {
        "kind",
        "version",
        "n_wires",
        "dtype",
        "shape",
        "instructions",
        "observables",
        "measurements",
        "metadata",
    }
    # Instructions encode the gate name under 'opcode', not 'name'.
    assert set(payload["instructions"][0]) == {
        "opcode",
        "wires",
        "params",
        "matrix",
        "metadata",
    }


def test_ir_rejects_unknown_top_level_keys() -> None:
    payload = fq.Circuit(1).h(0).to_ir().to_dict()
    payload["surprise"] = True

    with pytest.raises(fq.IRSerializationError):
        fq.CircuitIR.from_dict(payload)


@pytest.mark.parametrize("name", TIER2_COMPILER)
def test_compiler_interface_still_exports(name: str) -> None:
    compiler = importlib.import_module("flagquantum.compiler")

    assert name in compiler.__all__, f"flagquantum.compiler no longer exports {name}"
    assert hasattr(compiler, name)


@pytest.mark.parametrize(("module_path", "attribute"), TIER3_EMITTERS)
def test_emitter_submodules_still_expose_their_functions(module_path: str, attribute: str) -> None:
    module = importlib.import_module(module_path)

    assert hasattr(module, attribute), f"{module_path}.{attribute} disappeared"


@pytest.mark.parametrize(("module_path", "attribute"), TIER3_TRAINING)
def test_training_dependencies_are_still_declared_public(module_path: str, attribute: str) -> None:
    module = importlib.import_module(module_path)

    assert attribute in module.__all__, f"{module_path} no longer declares {attribute}"
    assert hasattr(module, attribute)


def test_the_policy_that_makes_a_hamiltonian_reach_the_objective_still_exists() -> None:
    """Without this object the SDK evaluates ⟨Z₀⟩ and ignores the Hamiltonian.

    Pinned by name and by the one attribute that matters, because a rename here
    would not fail anything else: the tool would keep training, keep converging
    and keep reporting an energy it never measured.
    """
    assert hasattr(fq, "RuntimePolicy")
    assert fq.RuntimePolicy().observable == "z"
    assert fq.RuntimePolicy(observable="hamiltonian").observable == "hamiltonian"


def test_the_module_members_the_training_tool_reads_still_exist() -> None:
    """The three class-level members. The fourth is pinned elsewhere, on purpose.

    ``named_parameter_groups`` is deliberately absent from this tuple. It is
    assigned in ``Module.__init__`` — ``self.named_parameter_groups:
    ParameterDict | None = None`` — so it does not exist on the class, and
    ``hasattr(fq.Module, "named_parameter_groups")`` is **False**. Measured, and
    a test that asserted it would fail on the very SDK it is meant to pin, which
    is the failure mode this file exists to prevent. It is covered instead by the
    training tests, which read it off a real module the tool built, at the one
    place ``training.py`` needs a named group.
    """
    for attribute in ("execute", "parameters", "named_parameters"):
        assert hasattr(fq.Module, attribute), f"Module.{attribute} disappeared"


def test_analysis_object_still_carries_the_fields_we_report() -> None:
    analysis = fq.Circuit(2).h(0).cx(0, 1).analysis()

    assert dataclasses.is_dataclass(analysis)
    names = {field.name for field in dataclasses.fields(analysis)}
    assert names >= REQUIRED_ANALYSIS_FIELDS


def test_analysis_is_reachable_from_a_circuit_built_from_ir() -> None:
    ir = fq.Circuit(2).h(0).cx(0, 1).to_ir()

    assert fq.Circuit.from_ir(ir).analysis().depth == 2


def test_plan_exposes_the_fields_this_server_reports() -> None:
    plan = fq.plan(fq.Circuit(2).h(0).cx(0, 1))

    for attribute in (
        "schema_version",
        "identity",
        "mode",
        "backend",
        "device",
        "precision",
        "is_distributed",
        "program_fingerprint",
        "summary",
        "to_json",
    ):
        assert hasattr(plan, attribute), f"ExecutionPlan.{attribute} disappeared"


def test_plan_summary_carries_the_keys_we_report() -> None:
    summary = fq.plan(fq.Circuit(2).h(0).cx(0, 1)).summary()

    assert {"depth", "n_wires", "n_instructions"} <= set(summary)


def test_execution_options_accepts_every_option_we_advertise() -> None:
    from flagquantum_mcp_server.planning import OPTION_FIELDS

    accepted = {field.name for field in dataclasses.fields(fq.ExecutionOptions)}

    assert set(OPTION_FIELDS) <= accepted


def test_shots_are_required_for_sampling_outputs() -> None:
    circuit = fq.Circuit(2).h(0).cx(0, 1)

    with pytest.raises(ValueError, match="requires shots"):
        fq.plan(circuit, outputs=fq.counts([0, 1]))


def test_the_installed_sdk_has_no_mcp_dependency() -> None:
    """The coupling must point one way: we depend on FlagQuantum, not the reverse.

    FlagQuantum's long-horizon architecture contract names "the main repository
    has no production MCP transport dependency" as a retirement condition, and
    its own `tests/team/services/test_service_boundaries.py` fails if `mcp` or
    `fastmcp` becomes importable on the core path. Asserting it from the
    installed distribution metadata checks the same thing from this side, and
    fails loudly if the SDK ever acquires one.
    """
    from importlib.metadata import requires

    declared = " ".join(requires("flagquantum") or ()).lower()

    assert "mcp" not in declared.replace("flagquantum", "")
    assert "fastmcp" not in declared


def test_the_training_budget_is_read_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from flagquantum_mcp_server import limits

    monkeypatch.delenv("FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS", raising=False)
    assert limits.max_train_seconds() == 60
    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS", "5")

    assert limits.max_train_seconds() == 5


def test_a_malformed_training_budget_falls_back_rather_than_disabling_the_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bound that reads as zero would refuse every call, which is a different bug."""
    from flagquantum_mcp_server import limits

    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS", "0")

    assert limits.max_train_seconds() == 60


def test_torch_is_reached_lazily_and_is_not_a_declared_dependency() -> None:
    """The optimizer type is torch's, but torch is not this package's to declare."""
    from importlib.metadata import requires

    from flagquantum_mcp_server._bridge import load_torch

    assert load_torch().optim.Adam is not None
    declared = " ".join(requires("flagquantum-mcp-server") or ()).lower()

    assert "torch" not in declared


def test_importing_the_server_does_not_pull_in_torch() -> None:
    """The laziness is a startup cost, not a style preference.

    ``_bridge`` binds both the SDK and torch on first use because the server
    must not be expensive to import. Measured, importing torch costs 0.66 s of
    cumulative import time, and every MCP client pays a server's import cost
    before it can list a tool.

    Asserted in a subprocess because this test process has torch loaded by
    then: the modules other tests import would mask an eager import here, and
    a check that cannot fail is not a check.
    """
    import subprocess
    import sys

    probe = (
        "import sys; import flagquantum_mcp_server.server; "
        "assert 'torch' not in sys.modules, sorted("
        "m for m in sys.modules if m.startswith('torch'))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, timeout=120, check=False
    )

    assert completed.returncode == 0, completed.stderr
