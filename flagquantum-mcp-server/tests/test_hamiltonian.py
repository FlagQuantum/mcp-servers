"""Named Pauli sums: the Hamiltonian a VQE run actually measures.

``{"kind": "expectation", "pauli": "ZZ"}`` evaluates one term. That is enough
to measure a correlation and not enough to measure an energy, so a caller
wanting ⟨H⟩ had to issue one call per term and add the results itself — seven
round trips for a four-qubit Ising chain.

The SDK already evaluates a whole sum: ``Observable`` accepts several weighted
terms, and a run returns one row per term with ``fq_coefficient`` in each row's
metadata. What was missing was a way to say those terms through JSON, which is
what ``terms`` is. The row-per-term shape is deliberate and preserved here: the
server reports what the SDK computed and does not add the rows up, because a
total this module calculated would be a number the SDK never produced.

Only public names are used. The term type underneath ``Observable`` is private,
and the operators build it without this server ever spelling it — so nothing
here breaks if upstream renames it.
"""

from __future__ import annotations

import json

import pytest

from flagquantum_mcp_server.errors import ToolInputError, ToolLimitError, UnsupportedFormatError
from flagquantum_mcp_server.simulation import simulate_execution
from tests.conftest import BELL_QIR, GHZ3_QIR

pytestmark = pytest.mark.unit

BELL = json.dumps(BELL_QIR)
MINUS = -0.5


def _zz_and_x(**extra: object) -> list[dict[str, object]]:
    """The Hamiltonian this module's tests measure: ZZ - 0.5 X on wire 0."""
    return [
        {"pauli": "ZZ", "coefficient": 1.0, **extra},
        {"pauli": "XI", "coefficient": MINUS, **extra},
    ]


def _run(terms: object, qir: str = BELL) -> dict[str, object]:
    return simulate_execution(qir, "qir", outputs=[{"kind": "expectation", "terms": terms}])


# --- the case that did not exist ---


def test_a_weighted_sum_returns_one_row_per_term() -> None:
    """A Hamiltonian is measured in one call, and it still reports term by term."""
    result = _run(_zz_and_x())

    rows = result["outputs"]
    assert [row["wires"] for row in rows] == [[0, 1], [0]]
    assert [row["coefficient"] for row in rows] == [1.0, MINUS]
    assert [row["value"][0] for row in rows] == pytest.approx([1.0, 0.0], abs=1e-6)


def test_the_rows_carry_the_coefficients_the_caller_wrote() -> None:
    """Without the coefficient the caller cannot reconstruct the energy.

    The sum is the caller's to make, so both numbers it needs — the coefficient
    and the per-term value — have to survive the trip.
    """
    rows = _run(_zz_and_x())["outputs"]

    energy = sum(row["coefficient"] * row["value"][0] for row in rows)

    assert energy == pytest.approx(1.0, abs=1e-6)


def test_a_term_may_omit_its_coefficient() -> None:
    """Unweighted is the common case, so it defaults rather than being required."""
    rows = _run([{"pauli": "ZZ"}, {"pauli": "ZZ"}])["outputs"]

    assert [row["coefficient"] for row in rows] == [1.0, 1.0]


def test_a_single_term_sum_agrees_with_the_pauli_shorthand() -> None:
    """The two spellings are one code path, so they cannot drift apart."""
    by_terms = _run([{"pauli": "ZZ", "coefficient": 1.0}])["outputs"]
    by_pauli = simulate_execution(BELL, "qir", outputs=[{"kind": "expectation", "pauli": "ZZ"}])[
        "outputs"
    ]

    assert by_terms == by_pauli


def test_a_negative_and_a_fractional_coefficient_survive() -> None:
    rows = _run([{"pauli": "IZ", "coefficient": -2}, {"pauli": "ZI", "coefficient": 0.25}])[
        "outputs"
    ]

    assert [row["coefficient"] for row in rows] == [-2.0, 0.25]


def test_each_term_matches_the_circuit_width() -> None:
    """A three-wire circuit refuses a four-letter term, naming which one.

    The first term is a legal three-wire string, so the refusal can only be
    about the second — which is the point: a message that named the whole
    request instead of the term would leave the caller searching.
    """
    with pytest.raises(ToolInputError) as caught:
        _run([{"pauli": "ZZZ"}, {"pauli": "ZZZZ"}], qir=json.dumps(GHZ3_QIR))

    message = str(caught.value)
    assert "terms[1]" in message, "the refusal has to name the offending term"
    assert "3" in message


# --- what must be refused rather than reinterpreted ---


def test_supplying_both_pauli_and_terms_is_refused() -> None:
    with pytest.raises(ToolInputError) as caught:
        simulate_execution(
            BELL,
            "qir",
            outputs=[{"kind": "expectation", "pauli": "ZZ", "terms": [{"pauli": "ZZ"}]}],
        )

    message = str(caught.value)
    assert "pauli" in message and "terms" in message


def test_an_empty_terms_list_is_refused() -> None:
    with pytest.raises(ToolInputError) as caught:
        _run([])

    assert "empty" in str(caught.value)


def test_terms_that_are_not_a_list_are_refused() -> None:
    with pytest.raises(ToolInputError) as caught:
        _run({"pauli": "ZZ"})

    assert "list of objects" in str(caught.value)


@pytest.mark.parametrize(
    "term",
    [
        pytest.param("ZZ", id="a-bare-string"),
        pytest.param(["ZZ", 1.0], id="a-list"),
        pytest.param(3, id="a-number"),
        pytest.param(None, id="null"),
    ],
)
def test_a_term_that_is_not_an_object_is_refused(term: object) -> None:
    with pytest.raises(ToolInputError) as caught:
        _run([term])

    assert "terms[0]" in str(caught.value)


def test_a_term_without_a_pauli_string_is_refused() -> None:
    with pytest.raises(ToolInputError) as caught:
        _run([{"coefficient": 1.0}])

    assert "terms[0]" in str(caught.value)
    assert "pauli" in str(caught.value)


@pytest.mark.parametrize(
    "coefficient",
    [
        pytest.param("1.0", id="a-string"),
        pytest.param(True, id="a-boolean"),
        pytest.param(None, id="null"),
        pytest.param([1.0], id="a-list"),
        pytest.param({"$parameter": "w"}, id="a-symbol"),
    ],
)
def test_a_coefficient_that_is_not_a_number_is_refused(coefficient: object) -> None:
    """A symbol is in this list for a reason worth stating.

    ``Z(0) * Parameter("w")`` builds a parameter expression rather than an
    observable, and the SDK then refuses the whole request as a type error.
    Checking the type here turns that into a message about the coefficient.
    """
    with pytest.raises(ToolInputError) as caught:
        _run([{"pauli": "ZZ", "coefficient": coefficient}])

    message = str(caught.value)
    assert "terms[0]" in message
    assert "coefficient" in message


def test_a_coefficient_no_float_can_hold_is_refused() -> None:
    """The same guard the training tool has, reached from the expectation path.

    Measured through this path before the fix: a coefficient of ``10**400`` came
    back as ``INTERNAL_ERROR: OverflowError: int too large to convert to float``
    from the ``float(coefficient)`` at the end of the shared validator. The guard
    belongs in that validator rather than in either caller, because both reach
    it, and this asserts on this side of the sharing.
    """
    with pytest.raises(ToolInputError) as caught:
        _run([{"pauli": "ZZ", "coefficient": 10**400}])

    message = str(caught.value)
    assert caught.value.code == "INVALID_INPUT", message
    assert "terms[0]" in message
    assert "coefficient" in message


def test_an_unknown_key_in_a_term_is_refused() -> None:
    """A misspelled key is a caller who meant something, not noise to drop."""
    with pytest.raises(UnsupportedFormatError) as caught:
        _run([{"pauli": "ZZ", "coeff": 1.0}])

    assert "coeff" in str(caught.value)


def test_an_unknown_key_in_the_request_is_refused() -> None:
    with pytest.raises(UnsupportedFormatError) as caught:
        simulate_execution(
            BELL, "qir", outputs=[{"kind": "expectation", "hamiltonian": [{"pauli": "ZZ"}]}]
        )

    assert "hamiltonian" in str(caught.value)


def test_an_all_identity_term_is_refused_by_its_position() -> None:
    with pytest.raises(ToolInputError) as caught:
        _run([{"pauli": "ZZ"}, {"pauli": "II"}])

    message = str(caught.value)
    assert "terms[1]" in message
    assert "identity" in message


def test_too_many_terms_are_refused_with_the_count(monkeypatch: pytest.MonkeyPatch) -> None:
    bound = 3
    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_HAMILTONIAN_TERMS", str(bound))

    with pytest.raises(ToolLimitError) as caught:
        _run([{"pauli": "ZZ"}] * (bound + 1))

    message = str(caught.value)
    assert str(bound + 1) in message
    assert "FLAGQUANTUM_MCP_MAX_HAMILTONIAN_TERMS" in message


def test_the_term_bound_is_read_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_HAMILTONIAN_TERMS", "1")

    with pytest.raises(ToolLimitError):
        _run([{"pauli": "ZZ"}, {"pauli": "XI"}])

    assert _run([{"pauli": "ZZ"}])["status"] == "success"


def test_a_single_pauli_request_still_refuses_a_missing_string() -> None:
    """The pre-existing shorthand keeps its own refusal."""
    with pytest.raises(ToolInputError) as caught:
        simulate_execution(BELL, "qir", outputs=[{"kind": "expectation"}])

    assert "terms" in str(caught.value)
