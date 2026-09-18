"""Training a circuit's parameters, and the four ways that can go quietly wrong.

The SDK does the numerics. What these tests pin is the boundary around them,
because every way this tool can mislead a caller is a way of accepting
something:

- a ``hamiltonian`` that the SDK stores and does not read, so training converges
  cleanly on a different objective and reports it as an energy;
- a replay that loses or reorders a gate, so the circuit trained is not the
  circuit sent;
- ``values`` that name a parameter the circuit does not have, so a typo trains
  from zero without saying so;
- a run whose predicted cost is past the budget, which must be refused before
  the client is left waiting rather than after.

The first is the one that shapes this module. It is not hypothetical: an earlier
version of this server's design trained ``⟨Z₀⟩`` for a week of afternoons
because ``Module(hamiltonian=H)`` accepts ``H`` and ignores it unless the
module's policy asks for it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from flagquantum_mcp_server.errors import (
    ToolInputError,
    ToolLimitError,
    UnsupportedFormatError,
)
from flagquantum_mcp_server.training import parameter_names, replay_builder

pytestmark = pytest.mark.unit

ANGLED = json.dumps(
    [
        {"name": "ry", "index": [0], "parameters": {"theta": {"$parameter": "t0"}}},
        {"name": "ry", "index": [1], "parameters": {"theta": {"$parameter": "t1"}}},
        {"name": "cx", "index": [0, 1]},
    ]
)


def _ir(qir: str):
    from flagquantum_mcp_server.circuits import resolve_ir

    return resolve_ir(qir, "qir")


def _numeric(value: Any) -> float:
    """Read a serialized gate argument as a number.

    A replayed symbol is a tensor, so the SDK serializes it as ``$tensor`` with
    the value under ``data`` — a scalar for the one-element group the builder
    indexes, a one-element list if the SDK ever encodes the group whole. A bound
    angle is a plain number. Reading either keeps these assertions about the
    value bound, rather than about the encoding the SDK chose for it.
    """
    if isinstance(value, Mapping) and "$tensor" in value:
        data = value["$tensor"]["data"]
        return float(data[0] if isinstance(data, list) else data)
    return float(value)


# --- the replay reproduces the circuit it was given ---


def test_a_replay_of_a_numeric_circuit_reproduces_its_instructions() -> None:
    """Byte-equality of the instruction list, which is what the replay rebuilds.

    Not byte-equality of the whole envelope: the replay reproduces the
    instructions and the width, and every other envelope field — ``dtype``,
    ``shape``, ``metadata`` — comes from a freshly default-constructed
    ``Circuit``. A ``dtype="complex128"`` source is equal in instructions and
    unequal in envelope, so the wider assertion would only pass on a source that
    happens to be default-constructed too.
    """
    import flagquantum as fq

    source = fq.Circuit(2).h(0).ry(1, theta=0.7).cx(0, 1)
    source.gate("any", [1], matrix=[[1, 0], [0, 1j]])
    source.rz(0, theta=-0.25)
    ir = source.to_ir()

    rebuilt = replay_builder(ir)({})

    assert rebuilt.to_ir().to_dict()["instructions"] == ir.to_dict()["instructions"]


def test_a_replayed_circuit_executes_and_agrees_with_its_source() -> None:
    """The claim byte-equality cannot make: the rebuild runs, and runs the same.

    Measured before this test existed: a replay built from ``to_dict()`` was
    byte-for-byte equal to its source on this same circuit and failed to
    execute. Serializing correctly is not the property that matters — producing
    the same numbers is, and only running it shows that.
    """
    import flagquantum as fq

    source = fq.Circuit(2).h(0).ry(1, theta=0.7).cx(0, 1)
    source.gate("any", [1], matrix=[[1, 0], [0, 1j]])
    ir = source.to_ir()

    rebuilt = replay_builder(ir)({})

    want = fq.run(source, outputs=[fq.probabilities([0, 1])]).measurements[0].value
    got = fq.run(rebuilt, outputs=[fq.probabilities([0, 1])]).measurements[0].value
    assert got == pytest.approx(want, abs=1e-9)


def test_a_parameter_inside_an_expression_is_substituted() -> None:
    """A name that appears only inside an expression is still a name to train.

    ``parameter_names`` reports it either way, so a caller cannot tell from the
    parameter list whether the substitution happened; only running the circuit
    shows it.
    """
    import torch

    qir = json.dumps(
        [
            {"name": "ry", "index": [0], "parameters": {"theta": {"$parameter": "t0"}}},
            {
                "name": "rz",
                "index": [1],
                "parameters": {
                    "theta": {"$expression": {"op": "mul", "args": [2.0, {"$parameter": "t1"}]}}
                },
            },
        ]
    )
    ir = _ir(qir)
    assert parameter_names(ir) == ("t0", "t1")

    built = replay_builder(ir)({"t0": torch.tensor([0.3]), "t1": torch.tensor([0.4])})

    assert built.is_parameterized() is False, "the rebuild still carries symbols"
    assert _numeric(built.to_ir().to_dict()["instructions"][1]["params"]["theta"]) == pytest.approx(
        0.8
    )


def test_a_gradient_flows_through_an_expression_to_the_name_inside_it() -> None:
    """The expression path is optimized, not merely evaluated once.

    A ``ParameterExpression`` resolves through the SDK's own ``bind``, so the
    question a test has to answer is whether the tensor that comes back is
    still attached to the graph ``Module`` optimizes. Measuring the gradient
    against its analytic value answers it: for ``ry(1, t0)`` then
    ``rz(1, 2*t1)`` against ``Y`` on wire 1, ``<Y> = sin(t0) sin(2 t1)``, so
    ``d/dt1`` is ``2 sin(t0) cos(2 t1)``.

    Eager binding does not reach the comparison: converting the traced tensor to
    a Python float makes ``make_fx`` refuse the builder at trace time ("Module
    builder compilation requires static circuit topology"), so that variant fails
    loudly rather than training nothing quietly. What the gradient assertion is
    for is the variant that does trace — a binding that detaches the group from
    the graph, which produces these same numbers with no gradient for the name
    inside the expression. This is the only test whose builder sees an
    expression, so it is the only one that can see either.
    """
    import math

    import flagquantum as fq
    from flagquantum.algorithms import Hamiltonian, pauli_term

    qir = json.dumps(
        [
            {"name": "ry", "index": [1], "parameters": {"theta": {"$parameter": "t0"}}},
            {
                "name": "rz",
                "index": [1],
                "parameters": {
                    "theta": {"$expression": {"op": "mul", "args": [2.0, {"$parameter": "t1"}]}}
                },
            },
        ]
    )
    ir = _ir(qir)
    module = fq.Module(
        replay_builder(ir),
        parameters=dict.fromkeys(parameter_names(ir), 1),
        init={"t0": 0.8, "t1": 0.3},
        hamiltonian=Hamiltonian([pauli_term(1.0, {1: "Y"})]),
        policy=fq.RuntimePolicy(observable="hamiltonian"),
    )

    loss = module.execute().require_value().mean()
    loss.backward()

    groups = module.named_parameter_groups
    assert float(loss.detach()) == pytest.approx(math.sin(0.8) * math.sin(0.6), abs=1e-6)
    assert float(groups["t1"].grad) == pytest.approx(2 * math.sin(0.8) * math.cos(0.6), abs=1e-5)


def test_a_matrix_gate_survives_the_replay() -> None:
    """A matrix is the one argument the replay must not drop or normalize."""
    import flagquantum as fq

    source = fq.Circuit(1)
    source.gate("any", [0], matrix=[[0, 1], [1, 0]])
    ir = source.to_ir()

    rebuilt = replay_builder(ir)({})

    payload = rebuilt.to_ir().to_dict()["instructions"]
    assert payload[0]["matrix"] == ir.to_dict()["instructions"][0]["matrix"]


def test_the_replay_binds_each_symbol_by_name_not_by_position() -> None:
    """Names are the contract; a positional replay would still pass a 2-param test."""
    import torch

    builder = replay_builder(_ir(ANGLED))

    # Deliberately the reverse of the order the parameters appear in.
    built = builder({"t1": torch.tensor([0.25]), "t0": torch.tensor([0.75])})
    payload = built.to_ir().to_dict()["instructions"]

    assert _numeric(payload[0]["params"]["theta"]) == pytest.approx(0.75)
    assert _numeric(payload[1]["params"]["theta"]) == pytest.approx(0.25)


def test_the_replay_carries_non_parameter_arguments_through_unchanged() -> None:
    """A bound angle mixed in with a symbol is not a symbol, and stays bound."""
    import torch

    qir = json.dumps(
        [
            {"name": "rx", "index": [0], "parameters": {"theta": 0.5}},
            {"name": "ry", "index": [0], "parameters": {"theta": {"$parameter": "t0"}}},
        ]
    )

    built = replay_builder(_ir(qir))({"t0": torch.tensor([0.25])})
    payload = built.to_ir().to_dict()["instructions"]

    assert payload[0]["params"]["theta"] == 0.5
    # The symbol became a tensor: a value rather than a name, which is what the
    # builder needs and is not the same thing as the marker it came from.
    symbol = payload[1]["params"]["theta"]
    assert isinstance(symbol, Mapping) and "$tensor" in symbol
    assert _numeric(symbol) == pytest.approx(0.25)


# --- the names come from the SDK ---


def test_parameter_names_are_the_circuits_own_and_are_sorted() -> None:
    """Sorted is the SDK's order, not first-appearance; the test says which."""
    names = parameter_names(_ir(ANGLED))

    assert names == ("t0", "t1")


def test_a_repeated_symbol_is_named_once() -> None:
    qir = json.dumps(
        [
            {"name": "ry", "index": [0], "parameters": {"theta": {"$parameter": "p"}}},
            {"name": "rz", "index": [0], "parameters": {"theta": {"$parameter": "p"}}},
        ]
    )

    assert parameter_names(_ir(qir)) == ("p",)


def test_a_circuit_without_parameters_has_no_names() -> None:
    qir = json.dumps([{"name": "h", "index": [0]}])

    assert parameter_names(_ir(qir)) == ()


def test_a_replayed_parameterized_circuit_evaluates_the_same_as_a_bound_one() -> None:
    """The end the replay exists for: same circuit, same numbers, same energy."""
    import flagquantum as fq
    import torch

    from flagquantum_mcp_server.circuits import circuit_from_ir

    ir = _ir(ANGLED)
    bound = circuit_from_ir(ir).bind_parameters({"t0": 0.3, "t1": 0.4})
    replayed = replay_builder(ir)({"t0": torch.tensor([0.3]), "t1": torch.tensor([0.4])})

    want = fq.run(bound, outputs=[fq.probabilities([0, 1])]).measurements[0].value
    got = fq.run(replayed, outputs=[fq.probabilities([0, 1])]).measurements[0].value

    assert got == pytest.approx(want, abs=1e-6)


# --- the budget model ---


def test_the_prediction_grows_with_steps_and_with_width() -> None:
    from flagquantum_mcp_server.training import predict_seconds

    narrow = _ir(json.dumps([{"name": "h", "index": [0]}]))
    wide = _ir(json.dumps(_layered_qir(16)))

    assert predict_seconds(wide, 10) > predict_seconds(wide, 5)
    assert predict_seconds(wide, 5) > predict_seconds(narrow, 5)


def test_the_gate_term_is_the_coefficient_times_instructions_times_state() -> None:
    """The slope is pinned to a value, not just a direction.

    The calibration test below only shows that the prediction exceeds the
    measurement, and the floor hides the slope entirely below twelve wires — so
    a coefficient wrong by orders of magnitude in the *safe* direction passes
    every other assertion here. This one reads the constant back.
    """
    from flagquantum_mcp_server.training import GATE_COST, STARTUP_SECONDS, predict_seconds

    ir = _ir(json.dumps(_layered_qir(24)))  # 71 instructions, far above the floor

    assert predict_seconds(ir, 1) - STARTUP_SECONDS == pytest.approx(
        len(ir.instructions) * 2.0**24 * GATE_COST
    )


def test_the_prediction_over_predicts_every_point_it_was_calibrated_from() -> None:
    """The model's only job is to be an upper bound, so this is the whole test.

    Measured per-step costs, warm, from the design's table. A model that
    under-predicts anywhere here declines to protect the caller exactly where
    the caller most needs it.

    The table runs from 1 instruction to 188, not just over the one-layer
    ansatz, and the low rows are the ones that carry the fixed cost: a flat
    per-step floor covers them by accident at four wires and by 105x too little
    at twenty-two, so a table of only the 35-to-71-instruction circuits cannot
    tell a state term that is present from one that is missing. The failure this
    test was written for — a 24-wire, one-gate circuit admitted for 35 steps and
    taking 438 s — lived entirely in the rows that were absent.
    """
    from flagquantum_mcp_server.training import STARTUP_SECONDS, predict_seconds

    measured = [
        (4, 11, 0.0018),
        (8, 23, 0.0041),
        (12, 35, 0.0090),
        (13, 38, 0.0155),
        (14, 41, 0.0207),
        (15, 44, 0.0421),
        (16, 47, 0.0852),
        (16, 188, 0.1470),
        (16, 3, 0.0280),
        (20, 1, 0.3019),
        (20, 3, 0.308),
        (20, 59, 1.2900),
        (22, 1, 2.1049),
        (22, 65, 8.5600),
        (24, 1, 12.5252),
        (24, 4, 13.1363),
    ]
    for n_wires, instructions, per_step in measured:
        ir = _ir(json.dumps(_ansatz_tail_qir(n_wires, instructions)))
        # The table names a width and an instruction count and the model reads
        # both, so both are checked: a circuit that quietly came out three wires
        # wide would measure the state term at the wrong power of two and pass
        # while doing it.
        assert len(ir.instructions) == instructions, (
            f"{n_wires} qubits: built {len(ir.instructions)} instructions, not {instructions}"
        )
        assert int(ir.n_wires) == n_wires, f"{n_wires} qubits: built width {int(ir.n_wires)}"
        # Strip the one-time startup, which belongs to no single step. Read from
        # the module rather than written as 0.5, so that changing the constant
        # changes what this compares instead of quietly comparing nothing.
        predicted = predict_seconds(ir, 1) - STARTUP_SECONDS

        assert predicted > per_step, (
            f"{n_wires} qubits, {instructions} instructions: predicted "
            f"{predicted:.4f}s per step against {per_step:.4f}s measured"
        )


def test_the_floor_is_what_covers_the_smallest_width() -> None:
    """At four qubits the state term is negligible and dispatch is the whole cost."""
    from flagquantum_mcp_server.training import (
        MIN_STEP_SECONDS,
        STARTUP_SECONDS,
        predict_seconds,
    )

    ir = _ir(json.dumps(_layered_qir(4)))

    assert predict_seconds(ir, 1) - STARTUP_SECONDS == pytest.approx(MIN_STEP_SECONDS)


def test_the_prediction_is_computed_without_running_anything() -> None:
    """It is arithmetic on the instruction count, so it costs nothing to ask."""
    import time

    from flagquantum_mcp_server.training import predict_seconds

    ir = _ir(json.dumps(_layered_qir(24)))
    started = time.perf_counter()
    predict_seconds(ir, 100_000)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05


def _layered_qir(n_wires: int, layers: int = 1) -> list[dict[str, object]]:
    """An ry/cx/rz ansatz as a gate list, one ``$parameter`` per rotation."""
    gates: list[dict[str, object]] = []
    for layer in range(layers):
        gates.extend(
            {
                "name": "ry",
                "index": [wire],
                "parameters": {"theta": {"$parameter": f"t{layer}_{wire}"}},
            }
            for wire in range(n_wires)
        )
        gates.extend({"name": "cx", "index": [wire, wire + 1]} for wire in range(n_wires - 1))
        gates.extend(
            {
                "name": "rz",
                "index": [wire],
                "parameters": {"theta": {"$parameter": f"u{layer}_{wire}"}},
            }
            for wire in range(n_wires)
        )
    return gates


def _ansatz_tail_qir(n_wires: int, instructions: int) -> list[dict[str, object]]:
    """The last ``instructions`` gates of the ``_layered_qir`` ansatz.

    Taken from the end of the list rather than the start, because the width is
    inferred from the wires the gates touch: the first gates of a layer are the
    ``ry`` block, so a three-gate *prefix* of a sixteen-wire ansatz is a
    three-wire circuit and the fixed per-step cost this table is mostly about
    would never be reached. The last gate of a layer is ``rz(n_wires - 1)``, so
    any non-empty tail spans the full width. A count that is a whole number of
    layers gives the whole ansatz either way.
    """
    per_layer = 3 * n_wires - 1
    layers = -(-instructions // per_layer)
    return _layered_qir(n_wires, layers)[-instructions:]


# --- the objective ---


def test_a_hamiltonian_is_built_from_the_same_terms_shape_the_outputs_use() -> None:
    from flagquantum_mcp_server.training import hamiltonian_from_terms

    built = hamiltonian_from_terms(
        [{"pauli": "ZZ", "coefficient": 1.0}, {"pauli": "XI", "coefficient": -0.5}],
        n_wires=2,
    )

    assert len(built.terms) == 2
    assert [float(term.coefficient) for term in built.terms] == [1.0, -0.5]


def test_a_term_that_the_expectation_builder_refuses_is_refused_here_too() -> None:
    """One shape, one validator, one set of refusals — that is the point of sharing."""
    from flagquantum_mcp_server.training import hamiltonian_from_terms

    with pytest.raises(ToolInputError) as caught:
        hamiltonian_from_terms([{"pauli": "ZZ"}, {"pauli": "II"}], n_wires=2)

    assert "terms[1]" in str(caught.value)
    assert "identity" in str(caught.value)


def test_a_term_that_does_not_match_the_circuit_width_is_refused_here_too() -> None:
    from flagquantum_mcp_server.training import hamiltonian_from_terms

    with pytest.raises(ToolInputError) as caught:
        hamiltonian_from_terms([{"pauli": "ZZZ"}], n_wires=2)

    assert "covers 3 wires" in str(caught.value)


def test_a_hamiltonian_with_a_single_identity_free_term_is_accepted() -> None:
    """The smallest legal objective, so the refusal above is not refusing everything."""
    from flagquantum_mcp_server.training import hamiltonian_from_terms

    assert len(hamiltonian_from_terms([{"pauli": "XI"}], n_wires=2).terms) == 1


def test_a_pauli_letter_lands_on_the_wire_its_position_names() -> None:
    """What "IX" versus "XI" actually decides, which the identity filter does not.

    The ``if letter != "I"`` is not what keeps the numbering right — the SDK's
    own normalizer drops identities from ``ops`` either way, measured — so a test
    of the filter would pass for the wrong reason. The position of the letter is
    what the caller reads as the wire, and getting it backwards is silent: every
    energy stays plausible, and reversing the enumeration leaves every other
    assertion in this module green.
    """
    from flagquantum_mcp_server.training import hamiltonian_from_terms

    assert hamiltonian_from_terms([{"pauli": "IX"}], n_wires=2).terms[0].ops == ((1, "x"),)
    assert hamiltonian_from_terms([{"pauli": "XI"}], n_wires=2).terms[0].ops == ((0, "x"),)


# --- the run ---

# -Z0Z1 + X0 + X1, whose ground energy is -2.2360679... (=-sqrt(5)).
TFIM2 = [
    {"pauli": "ZZ", "coefficient": -1.0},
    {"pauli": "XI", "coefficient": 1.0},
    {"pauli": "IX", "coefficient": 1.0},
]


def test_training_returns_a_trajectory_and_the_parameters_it_ended_on() -> None:
    from flagquantum_mcp_server.training import train_parameters

    payload = train_parameters(ANGLED, TFIM2, "qir", steps=20, learning_rate=0.3)

    assert payload["status"] == "success"
    assert payload["completed_steps"] == 20
    assert len(payload["losses"]) == 20
    assert set(payload["parameters"]) == {"t0", "t1"}
    assert payload["circuit"]["n_qubits"] == 2


def test_the_hamiltonian_reaches_the_objective() -> None:
    """The test this whole module exists for.

    With the default policy the SDK evaluates ⟨Z₀⟩ and ignores the Hamiltonian,
    so a run against ``-Z0Z1 + X0 + X1`` would converge to -1 while claiming to
    have found the ground state. The energy here has to approach -sqrt(5).
    """
    from flagquantum_mcp_server.training import train_parameters

    payload = train_parameters(ANGLED, TFIM2, "qir", steps=200, learning_rate=0.2)

    assert payload["final_loss"] == pytest.approx(-(5**0.5), abs=0.05)


def test_the_objective_is_the_hamiltonian_and_not_the_first_wire_s_z() -> None:
    """One step in, the two objectives are already different numbers.

    At the default all-zero starting angles the circuit is ``|00⟩``, so
    ``-Z₀Z₁ + X₀ + X₁`` evaluates to ``-1`` while ``⟨Z₀⟩`` evaluates to ``+1``.
    A run that reports ``+1`` is measuring the wrong operator, which is the trap
    this design was rewritten around, and the sign is what makes it visible in a
    single call rather than after two hundred.
    """
    from flagquantum_mcp_server.training import train_parameters

    payload = train_parameters(ANGLED, TFIM2, "qir", steps=1)

    assert payload["initial_loss"] == pytest.approx(-1.0, abs=1e-5)


def test_final_loss_is_the_loss_of_the_parameters_returned_not_the_one_before() -> None:
    """The continuation promise: resume and the curve joins without a step back."""
    from flagquantum_mcp_server.training import train_parameters

    first = train_parameters(ANGLED, TFIM2, "qir", steps=5, learning_rate=0.3)
    resumed = train_parameters(
        ANGLED, TFIM2, "qir", values=first["parameters"], steps=5, learning_rate=0.3
    )

    assert resumed["initial_loss"] == pytest.approx(first["final_loss"], abs=1e-6)
    assert resumed["initial_parameters"] == pytest.approx(first["parameters"])
    # The guard that keeps the assertion above from passing vacuously: if the
    # optimizer did nothing, initial and final would be the same number and the
    # equality would hold no matter what `final_loss` were computed from. This
    # is NOT the continuation claim — the equality above is — so it asserts only
    # that the resumed run improved.
    assert resumed["final_loss"] < resumed["initial_loss"]


def test_the_reported_parameters_are_what_the_next_call_starts_from() -> None:
    from flagquantum_mcp_server.training import train_parameters

    payload = train_parameters(ANGLED, TFIM2, "qir", steps=3, learning_rate=0.3)
    again = train_parameters(ANGLED, TFIM2, "qir", values=payload["parameters"], steps=1)

    assert again["initial_parameters"] == pytest.approx(payload["parameters"])


def test_the_execution_block_says_where_the_sdk_ran_this() -> None:
    """A Module run reports its own runtime shape, not a fq.run one.

    Measured: ``Module.execute()`` returns a runtime block holding ``executor``,
    ``backend``, ``mode``, ``world_size`` and ``node_count``, and no
    ``execution_path`` or ``platform_provider`` at all. So this tool restates
    what its own SDK call reported rather than reusing the simulation's field
    names, which would come back empty.
    """
    from flagquantum_mcp_server.training import train_parameters

    payload = train_parameters(ANGLED, TFIM2, "qir", steps=1)

    assert payload["execution"]["backend"] == "pytorch"
    assert payload["execution"]["mode"] == "statevector"
    assert payload["execution"]["node_count"] == 1
    assert payload["execution"]["world_size"] == 1
    assert payload["execution"]["accuracy"]["metric"] == "not_measured"


def test_values_defaults_to_zero_and_says_so() -> None:
    from flagquantum_mcp_server.training import train_parameters

    payload = train_parameters(ANGLED, TFIM2, "qir", steps=1)

    assert payload["initial_parameters"] == {"t0": 0.0, "t1": 0.0}


def test_a_different_learning_rate_changes_the_run() -> None:
    from flagquantum_mcp_server.training import train_parameters

    slow = train_parameters(ANGLED, TFIM2, "qir", steps=5, learning_rate=0.01)
    fast = train_parameters(ANGLED, TFIM2, "qir", steps=5, learning_rate=0.5)

    assert slow["parameters"] != fast["parameters"]


def test_a_different_step_count_changes_the_run() -> None:
    from flagquantum_mcp_server.training import train_parameters

    short = train_parameters(ANGLED, TFIM2, "qir", steps=1, learning_rate=0.3)
    longer = train_parameters(ANGLED, TFIM2, "qir", steps=4, learning_rate=0.3)

    assert short["completed_steps"] == 1
    assert longer["completed_steps"] == 4


def test_the_parameters_argument_is_not_mutated_by_the_call() -> None:
    """Rule 5: read-only over the caller's inputs."""
    from flagquantum_mcp_server.training import train_parameters

    values = {"t0": 0.25, "t1": -0.5}
    train_parameters(ANGLED, TFIM2, "qir", values=values, steps=2)

    assert values == {"t0": 0.25, "t1": -0.5}


# --- the budget ---


def test_a_run_past_the_budget_is_refused_before_any_work_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refused before, not after: the client is never left waiting."""
    import time

    from flagquantum_mcp_server.training import train_parameters

    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS", "1")
    wide = json.dumps(_layered_qir(16))
    started = time.perf_counter()

    with pytest.raises(ToolLimitError) as caught:
        train_parameters(wide, [{"pauli": "I" * 16, "coefficient": 1.0}], "qir", steps=5000)

    assert time.perf_counter() - started < 1.0, "the refusal did no work"
    message = str(caught.value)
    # "16-qubit", not "16": the predicted seconds and the instruction count are
    # both in this message, and either could contain "16" by coincidence.
    assert "16-qubit" in message
    assert "5000" in message
    assert "FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS" in message


def test_a_run_inside_the_budget_is_not_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from flagquantum_mcp_server.training import train_parameters

    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS", "1")

    assert train_parameters(ANGLED, TFIM2, "qir", steps=1)["status"] == "success"


# --- the refusals that matter ---
#
# These two are Task 8's tests, delivered early by this round's fix so that each
# of the two `math.isfinite` guards in `training.py` has a test that can red on
# its own. Task 8's step-1 block lists both verbatim; when that task runs it must
# not append a second copy of either — two `def` of one name in a module shadow,
# the later wins, and pytest would then collect only one of them.


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 10**400])
def test_a_non_finite_starting_value_is_refused(bad: object) -> None:
    """A starting value that is not a finite real reaches the optimizer otherwise.

    Measured: without this refusal, ``values={"t0": nan}`` returns
    ``status: "success"`` with every loss and every returned parameter NaN, and
    ``values={"t0": 10**400}`` returns an internal error. ``json.loads`` accepts
    the ``NaN`` and ``Infinity`` tokens and turns an integer literal into a
    Python ``int``, so no payload carrying one is stopped on the way in.
    """
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(ANGLED, TFIM2, "qir", values={"t0": bad, "t1": 0.1})

    assert "t0" in str(caught.value)


@pytest.mark.parametrize(
    "rate",
    [0, 0.0, -0.1, True, "0.1", float("nan"), float("inf"), 10**400],
)
def test_a_learning_rate_adam_cannot_use_is_refused(rate: object) -> None:
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(ANGLED, TFIM2, "qir", learning_rate=rate)

    assert "learning_rate" in str(caught.value)


# --- the refusals that matter ---


def test_a_circuit_with_no_parameters_is_refused_by_name() -> None:
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(json.dumps([{"name": "h", "index": [0]}]), [{"pauli": "Z"}], "qir")

    message = str(caught.value)
    assert "no parameters" in message
    assert "inspect_parameters_tool" in message


def test_a_missing_hamiltonian_is_refused() -> None:
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(ANGLED, None, "qir")

    assert "hamiltonian" in str(caught.value)


def test_an_empty_hamiltonian_is_refused() -> None:
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(ANGLED, [], "qir")

    message = str(caught.value)
    assert "empty" in message
    # Both words, because "empty" alone is already true of the shared
    # validator's message before this task renames anything: it reads
    # "'terms' is empty". Asserting only that would pass before this task
    # does its one job, and would keep passing if it never did.
    assert "hamiltonian" in message
    assert "'terms'" not in message


def test_a_value_for_a_parameter_the_circuit_does_not_have_is_refused() -> None:
    """The typo case: silently ignored, the parameter trains from zero."""
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(ANGLED, TFIM2, "qir", values={"t0": 0.1, "t1": 0.1, "t2": 0.1})

    message = str(caught.value)
    assert "t2" in message
    assert "t0" in message and "t1" in message


def test_a_missing_value_is_refused() -> None:
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(ANGLED, TFIM2, "qir", values={"t0": 0.1})

    assert "t1" in str(caught.value)


def test_a_starting_value_that_is_not_a_number_is_refused() -> None:
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(ANGLED, TFIM2, "qir", values={"t0": "0.1", "t1": 0.1})

    assert "t0" in str(caught.value)


@pytest.mark.parametrize("steps", [0, -1, 1.5, True, "10"])
def test_a_step_count_the_loop_cannot_use_is_refused(steps: object) -> None:
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(ANGLED, TFIM2, "qir", steps=steps)

    assert "steps" in str(caught.value)


def test_a_circuit_carrying_observables_is_refused_and_points_at_hamiltonian() -> None:
    """The field is never read, so a caller who put the objective there gets nothing."""
    from flagquantum_mcp_server.circuits import serialize
    from flagquantum_mcp_server.training import train_parameters

    envelope = json.loads(serialize(ANGLED, "qir")["ir_json"])
    envelope["observables"] = [{"name": "ZZ", "wires": [0, 1], "coefficient": 1.0}]

    with pytest.raises(ToolInputError) as caught:
        train_parameters(json.dumps(envelope), TFIM2, "ir")

    message = str(caught.value)
    assert "observables" in message
    assert "hamiltonian" in message


def test_no_refusal_from_this_tool_says_a_noun_the_caller_did_not_write() -> None:
    """Every refusal, in the caller's vocabulary — checked as a property.

    Every message below comes from the shared validator, which speaks the
    vocabulary of an ``expectation`` output. A caller who wrote ``hamiltonian``
    must never be told about ``terms``, ``terms[0]`` or an ``expectation``:
    those name something they never typed, and a message that has to be
    translated is one that cannot be acted on.

    Checked over every malformed shape at once rather than phrase by phrase,
    because phrase-by-phrase is exactly what let the first version miss the
    term-level refusals — it replaced the quoted field name, and ``terms[0]``
    is unquoted.
    """
    from flagquantum_mcp_server.training import train_parameters

    malformed = [
        None,
        [],
        [{"coefficient": 1.0}],
        [{"pauli": "ZZ", "coefficient": 1.0, "extra": 1}],
        [{"pauli": "ZZZ", "coefficient": 1.0}],
        [{"pauli": "II", "coefficient": 1.0}],
        [{"pauli": "ZZ", "coefficient": "x"}],
        "ZZ",
    ]

    for objective in malformed:
        with pytest.raises(ToolInputError) as caught:
            train_parameters(ANGLED, objective, "qir", steps=1)
        message = str(caught.value)
        assert "hamiltonian" in message, (objective, message)
        assert "terms" not in message, (objective, message)
        assert "expectation" not in message, (objective, message)


def test_the_term_bound_still_reports_as_a_limit_and_not_as_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rename must not swallow the bound's own error code.

    ``ToolLimitError`` subclasses ``ToolInputError``, so a bare
    ``except ToolInputError`` around the builder catches it and re-raises it as
    the base class. Measured: the caller then sees ``INVALID_INPUT`` where the
    bound had raised ``LIMIT_EXCEEDED``. The errors module exists so an agent can
    branch on the code rather than parse prose, and this collapses two branches
    into one.
    """
    from flagquantum_mcp_server.training import train_parameters

    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_HAMILTONIAN_TERMS", "2")

    with pytest.raises(ToolLimitError) as caught:
        train_parameters(ANGLED, [{"pauli": "Z", "coefficient": 1.0}] * 5, "qir", steps=1)

    assert caught.value.code == "LIMIT_EXCEEDED"
    assert "expectation" not in str(caught.value)
    assert "objective" in str(caught.value)


def test_an_unsupported_key_still_reports_as_a_format_problem() -> None:
    """The same collapse that hit the bound hits this code, and nothing noticed.

    ``UnsupportedFormatError`` subclasses ``ToolInputError``, so a bare
    ``except ToolInputError`` catches it and re-raises the base class. Measured
    before the fix: an unknown key in a term came back as ``INVALID_INPUT``
    where the validator had raised ``UNSUPPORTED_FORMAT``. The property test
    above passes either way, because it asserts prose — which is exactly why
    this test asserts the code.
    """
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(UnsupportedFormatError) as caught:
        train_parameters(ANGLED, [{"pauli": "ZZ", "coefficient": 1.0, "extra": 1}], "qir", steps=1)

    assert caught.value.code == "UNSUPPORTED_FORMAT"


def test_a_caller_s_literal_is_echoed_back_unchanged() -> None:
    """The rename touches the validator's nouns, never the caller's value.

    A refusal quotes the offending value back: ``has a coefficient that is a
    string ('terms')``. Renaming the field name ``terms`` therefore has a message
    in which the SDK's word and a caller's literal are the same six characters,
    and an unanchored replace answers ``('hamiltonian')`` — the caller's own
    value rewritten, in the message whose job is to explain it. Measured.

    A literal containing ``terms[`` is still echoed renamed; that is the residual
    this test records rather than claims away.
    """
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(ANGLED, [{"pauli": "ZZ", "coefficient": "terms"}], "qir", steps=1)

    message = str(caught.value)
    assert "('terms')" in message, message
    assert "('hamiltonian')" not in message, message
    assert "hamiltonian[0]" in message, message
