"""Executing a circuit locally, and the four things that must not be silent.

This is the only tool in the server that runs anything. The SDK does the
numerics; what these tests pin is the boundary around them, because every way
this tool can mislead a caller is a way of *accepting* something:

- an unbound parameter, which the SDK reports as ``planned execution failed``
  and nothing else;
- an ``observables`` entry on the circuit, which the default statevector path
  never reads — a Hamiltonian that serializes, hashes, round-trips and is then
  dropped without a word;
- an output request *and* circuit-declared measurements, which the SDK refuses
  in its own vocabulary;
- an output larger than a response can carry, which is a property of the
  request rather than of the run.

Each rejection test asserts on the message as well as the code: a refusal that
does not say which field to change is a refusal a caller has to guess at, which
is the failure mode this whole module exists to avoid.
"""

from __future__ import annotations

import json

import pytest

from flagquantum_mcp_server.circuits import serialize
from flagquantum_mcp_server.errors import ToolInputError
from flagquantum_mcp_server.simulation import simulate_execution

pytestmark = pytest.mark.unit

BELL = json.dumps([{"name": "h", "index": [0]}, {"name": "cx", "index": [0, 1]}])
SYMBOLIC = json.dumps(
    [{"name": "ry", "index": [0], "parameters": {"theta": {"$parameter": "theta"}}}]
)
# 17 wires is past the SDK's marginal-probability limit, which is 8.
WIDE = json.dumps(
    [{"name": "h", "index": [0]}, {"name": "cx", "index": [0, 1]}]
    + [{"name": "x", "index": [wire]} for wire in range(2, 17)]
)


def _envelope(qir: str, **fields: object) -> str:
    """Return a canonical IR payload carrying extra envelope fields."""
    envelope = json.loads(serialize(qir, "qir")["ir_json"])
    envelope.update(fields)
    return json.dumps(envelope)


# --- the two paths that need no invention ---


def test_a_requested_output_comes_back_with_its_value() -> None:
    result = simulate_execution(
        BELL, "qir", outputs=[{"kind": "counts", "wires": [0, 1]}], options={"shots": 512}
    )

    assert result["status"] == "success"
    (counts,) = result["outputs"]
    assert counts["kind"] == "counts"
    assert counts["shots"] == 512
    assert sum(counts["value"][0].values()) == 512


def test_an_expectation_returns_the_number_a_vqe_loop_wants() -> None:
    result = simulate_execution(BELL, "qir", outputs=[{"kind": "expectation", "pauli": "ZZ"}])

    (expectation,) = result["outputs"]
    assert expectation["kind"] == "expectation_ps"
    assert expectation["value"][0] == pytest.approx(1.0)


def test_without_outputs_the_circuit_speaks_for_itself() -> None:
    """A declared measurement is the circuit's own output request.

    The tool runs the circuit as it stands rather than inventing a default: the
    shot count in the envelope is the one used, so nothing here is the server's
    choice.
    """
    envelope = _envelope(BELL, measurements=[{"kind": "counts", "wires": [0, 1], "shots": 384}])

    result = simulate_execution(envelope, "ir")

    (counts,) = result["outputs"]
    assert counts["kind"] == "counts"
    assert counts["shots"] == 384
    assert sum(counts["value"][0].values()) == 384


def test_without_outputs_and_without_measurements_it_reports_probabilities() -> None:
    result = simulate_execution(BELL, "qir")

    (probabilities,) = result["outputs"]
    assert probabilities["kind"] == "probabilities"
    assert probabilities["wires"] == [0, 1]
    assert probabilities["value"] == pytest.approx([0.5, 0.0, 0.0, 0.5], abs=1e-6)


def test_the_result_says_where_it_ran() -> None:
    """Provenance is the SDK's, restated rather than summarised.

    The SDK marks a local run ``not_measured`` on accuracy and refuses to
    license a scalability claim. Those fields travel with the result so an
    agent reads them where it reads the numbers.
    """
    result = simulate_execution(
        BELL, "qir", outputs=[{"kind": "counts", "wires": [0]}], options={"shots": 16}
    )

    execution = result["execution"]
    assert execution["execution_path"] == "local_statevector"
    assert execution["platform_provider"] == "pytorch_cpu"
    assert execution["device"] == "cpu"
    assert execution["accuracy"]["metric"] == "not_measured"


def test_the_same_seed_gives_the_same_draws() -> None:
    """``seed`` reaches the sampler rather than being accepted and dropped."""
    call = lambda: simulate_execution(  # noqa: E731
        BELL, "qir", outputs=[{"kind": "counts", "wires": [0]}], options={"shots": 200, "seed": 7}
    )["outputs"][0]["value"]

    assert call() == call()


def test_a_different_seed_gives_different_draws() -> None:
    draws = [
        simulate_execution(
            BELL,
            "qir",
            outputs=[{"kind": "counts", "wires": [0]}],
            options={"shots": 200, "seed": seed},
        )["outputs"][0]["value"]
        for seed in (7, 8)
    ]

    assert draws[0] != draws[1]


# --- what must be refused rather than reinterpreted ---


def test_an_unbound_parameter_is_refused_by_name() -> None:
    """The SDK says only ``planned execution failed``, so the check is ours."""
    with pytest.raises(ToolInputError) as caught:
        simulate_execution(SYMBOLIC, "qir")

    message = str(caught.value)
    assert "theta" in message
    assert "bind_parameters_tool" in message


def test_an_observable_on_the_circuit_is_refused_rather_than_dropped() -> None:
    """The default local path never reads ``observables``.

    It serializes, it hashes, it round-trips, and it changes what the circuit
    reports as parameterized — because a coefficient may be a symbol. Then the
    run ignores it. Silence here would be the exact failure this server exists
    to prevent, so the field is refused and the working spelling is named.
    """
    envelope = _envelope(BELL, observables=[{"name": "zz", "wires": [0, 1]}])

    with pytest.raises(ToolInputError) as caught:
        simulate_execution(envelope, "ir")

    message = str(caught.value)
    assert "observables" in message
    assert "expectation" in message


def test_an_output_request_alongside_declared_measurements_is_refused() -> None:
    """The SDK's own error names neither field, so the conflict is caught here."""
    envelope = _envelope(BELL, measurements=[{"kind": "counts", "wires": [0, 1], "shots": 64}])

    with pytest.raises(ToolInputError) as caught:
        simulate_execution(envelope, "ir", outputs=[{"kind": "probabilities"}])

    message = str(caught.value)
    assert "measurements" in message
    assert "outputs" in message


def test_a_sampling_output_without_shots_is_refused() -> None:
    with pytest.raises(ToolInputError) as caught:
        simulate_execution(BELL, "qir", outputs=[{"kind": "counts", "wires": [0]}])

    assert "shots" in str(caught.value)


def test_probabilities_past_the_sdk_marginal_limit_name_a_way_forward() -> None:
    """The SDK's limit is real and it stays in its own words.

    The SDK computes a marginal distribution by contraction and refuses past a
    few wires by default, pointing at a setting it wants raised. That setting is
    not reachable through this server — raising it means accepting the cost the
    message names — so the refusal has to offer what the caller can actually do
    instead of relaying an instruction they cannot follow.
    """
    with pytest.raises(ToolInputError) as caught:
        simulate_execution(WIDE, "qir")

    message = str(caught.value)
    assert "max_marginal_wires" in message, "the SDK's own text is left intact"
    assert "does not expose" in message, "and named as unreachable from here"
    assert "counts" in message and "wires" in message, "the alternatives are named"


def test_the_same_wide_circuit_is_allowed_when_the_output_is_sparse() -> None:
    """Counts carry only the outcomes that occurred, so no marginal is computed."""
    result = simulate_execution(
        WIDE, "qir", outputs=[{"kind": "counts", "wires": [0, 1]}], options={"shots": 64}
    )

    assert result["status"] == "success"
    assert sum(result["outputs"][0]["value"][0].values()) == 64


def test_a_named_subset_of_wires_stays_under_the_marginal_limit() -> None:
    result = simulate_execution(WIDE, "qir", outputs=[{"kind": "probabilities", "wires": [0, 1]}])

    assert result["outputs"][0]["value"] == pytest.approx([0.5, 0.0, 0.0, 0.5], abs=1e-6)


def test_a_result_too_large_to_return_is_refused_with_its_size() -> None:
    """Sampling is bounded by ``shots``, which the caller chooses.

    This is the bound that grows without a circuit bound behind it: 200,000
    draws is a legal request by every other limit here and far more than a
    response can carry.
    """
    with pytest.raises(ToolInputError) as caught:
        simulate_execution(
            BELL,
            "qir",
            outputs=[{"kind": "samples", "wires": [0, 1]}],
            options={"shots": 200_000},
        )

    message = str(caught.value)
    assert "400000" in message, "the message names the size it refused"
    assert "FLAGQUANTUM_MCP_MAX_RESPONSE_VALUES" in message


def test_the_response_cap_is_read_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """A deployment can tighten the bound without a code change."""
    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_RESPONSE_VALUES", "2")

    with pytest.raises(ToolInputError) as caught:
        simulate_execution(BELL, "qir", outputs=[{"kind": "probabilities", "wires": [0, 1]}])

    assert "4" in str(caught.value)
