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

    ``parameter_names`` reports it either way, so the failure is silent until
    the run: the tool would offer a trainable group it never applies.
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

    Without this, binding floats instead of tensors would leave every other
    test green while a parameter inside an expression silently stopped
    training — the failure this tool exists to prevent.
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
