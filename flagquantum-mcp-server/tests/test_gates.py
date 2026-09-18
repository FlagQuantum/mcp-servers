"""Gate metadata and gate-list signature validation."""

from __future__ import annotations

import json

import pytest

from flagquantum_mcp_server.circuits import resolve_ir
from flagquantum_mcp_server.errors import ToolInputError
from flagquantum_mcp_server.gates import (
    closest_names,
    gate_records,
    known_opcodes,
    signature_or_none,
)

pytestmark = pytest.mark.unit


def _gate(name: str, index: list[int], **rest: object) -> str:
    return json.dumps([{"name": name, "index": index, **rest}])


# --- the manifest itself ---


def test_every_opcode_and_alias_is_known() -> None:
    names = known_opcodes()

    assert len(names) == 45
    assert {"h", "x", "y", "z", "cx", "cnot", "ccx", "ccnot", "rz", "swap"} <= names


def test_aliases_share_their_opcode_record() -> None:
    assert signature_or_none("cnot") == signature_or_none("cx")
    assert signature_or_none("hadamard") == signature_or_none("h")
    assert signature_or_none("ccnot") == signature_or_none("ccx")


def test_signatures_carry_arity_and_parameter_names() -> None:
    assert signature_or_none("h") == {"opcode": "h", "arity": 1, "parameters": ()}
    assert signature_or_none("rz") == {"opcode": "rz", "arity": 1, "parameters": ("theta",)}
    assert signature_or_none("u3") == {
        "opcode": "u3",
        "arity": 1,
        "parameters": ("theta", "phi", "lbd"),
    }
    assert signature_or_none("ccx") == {"opcode": "ccx", "arity": 3, "parameters": ()}


def test_an_opcode_outside_the_manifest_has_no_signature() -> None:
    """``any`` carries a matrix, so it is legal without being a built-in."""
    assert signature_or_none("any") is None
    assert signature_or_none("definitely-not-a-gate") is None


def test_gate_records_describes_the_named_gates() -> None:
    result = gate_records(["rz", "ccx"])

    assert result["status"] == "success"
    assert [record["opcode"] for record in result["gates"]] == ["rz", "ccx"]
    assert result["gates"][0]["parameters"] == ["theta"]
    assert result["gates"][1]["aliases"] == ["ccnot", "toffoli"]


def test_gate_records_describes_everything_by_default() -> None:
    result = gate_records()

    assert result["n_gates"] == 35
    assert result["n_names"] == 45
    assert len(result["gates"]) == 35
    opcodes = [record["opcode"] for record in result["gates"]]
    assert opcodes == sorted(opcodes)


def test_gate_records_rejects_an_unknown_name_with_suggestions() -> None:
    with pytest.raises(ToolInputError, match="Closest names"):
        gate_records(["rzzz"])


def test_the_note_names_the_key_a_gate_keeps_its_arguments_under() -> None:
    """The gate table knows the argument *names*; the note has to place them.

    It used to say "one entry per parameter" without naming the enclosing key,
    so a caller who had only this tool to go on had to try "params" first — the
    spelling the IR format uses — and be corrected by an error message.
    """
    note = gate_records(["rz"])["note"]

    assert "'parameters'" in note
    assert '"parameters": {"theta": 0.5}' in note
    assert "radians" in note
    assert "$parameter" in note


def test_closest_names_ranks_by_shared_prefix() -> None:
    """The best match comes first; ties between equally close names are free.

    Two names sharing the same prefix length and length (``rxx``, ``ryy``) are
    equally good suggestions, so the test does not pin their order.
    """
    suggestions = closest_names("rzzz", 4)

    assert suggestions[0] == "rzz"
    assert suggestions[1] == "rz"
    assert {"rxx", "ryy"} <= set(suggestions)


def test_closest_names_finds_the_gate_behind_a_typo() -> None:
    """Suggestions are canonical opcodes, never aliases.

    Suggesting an alias would be odd: an alias the caller typed is already
    valid, so this function would not be reached for it.
    """
    assert closest_names("hadmard", 3)[0] == "h"
    assert closest_names("rxx", 3)[0] == "rxx"
    assert "cnot" not in closest_names("cx", 5)


# --- signature validation on the qir path ---


def test_a_correct_gate_list_is_accepted() -> None:
    ir = resolve_ir(_gate("rz", [0], parameters={"theta": 0.5}), "qir")

    assert ir.n_wires == 1
    assert ir.instructions[0].name == "rz"


def test_a_wrong_wire_count_names_the_gate_and_its_arity() -> None:
    with pytest.raises(ToolInputError, match="'cx' takes 2 wire"):
        resolve_ir(_gate("cx", [0]), "qir")


def test_a_missing_parameter_names_what_the_gate_takes() -> None:
    with pytest.raises(ToolInputError, match=r"missing parameter\(s\) \['theta'\]"):
        resolve_ir(_gate("rz", [0]), "qir")


def test_a_misspelled_parameter_is_rejected_rather_than_ignored() -> None:
    """The failure mode this guards against: a silently dropped argument."""
    with pytest.raises(ToolInputError, match=r"unknown parameter\(s\) \['phi'\]"):
        resolve_ir(_gate("rz", [0], parameters={"phi": 1.0}), "qir")


def test_parameters_given_to_a_gate_that_takes_none_are_rejected() -> None:
    with pytest.raises(ToolInputError, match="takes no parameters"):
        resolve_ir(_gate("h", [0], parameters={"theta": 1.0}), "qir")


def test_an_alias_is_accepted_and_reported_by_its_canonical_name() -> None:
    ir = resolve_ir(_gate("cnot", [0, 1]), "qir")

    assert ir.instructions[0].name == "cx"


def test_an_alias_with_the_wrong_wire_count_says_which_opcode_it_is() -> None:
    with pytest.raises(ToolInputError, match="alias of 'cx'"):
        resolve_ir(_gate("cnot", [0]), "qir")


def test_the_previously_unhelpful_key_mismatch_is_named() -> None:
    with pytest.raises(ToolInputError, match="carries its arguments under 'parameters'"):
        resolve_ir(_gate("rz", [0], params={"theta": 0.5}), "qir")


def test_a_typo_suggests_real_gate_names() -> None:
    with pytest.raises(ToolInputError, match="is not a gate in the installed FlagQuantum"):
        resolve_ir(_gate("hadmard", [0]), "qir")


def test_a_custom_opcode_with_a_matrix_is_still_allowed() -> None:
    """``any`` is how a caller supplies an arbitrary unitary."""
    ir = resolve_ir(
        json.dumps([{"name": "mygate", "index": [0], "gate": [[1.0, 0.0], [0.0, 1.0]]}]),
        "qir",
    )

    assert ir.n_wires == 1


# --- parameter values ---
#
# A parameter value is either a number or one of the encodings the SDK's own
# decoder understands. Anything else is accepted by the SDK and stored as an
# opaque value that no longer reads as a parameter, which is how a circuit
# silently loses its symbol.


def test_a_number_is_accepted_as_a_parameter_value() -> None:
    ir = resolve_ir(_gate("rz", [0], parameters={"theta": 0.5}), "qir")

    assert ir.instructions[0].params == {"theta": 0.5}


def test_a_parameter_marker_is_accepted_and_decoded() -> None:
    ir = resolve_ir(_gate("ry", [0], parameters={"theta": {"$parameter": "theta"}}), "qir")

    assert ir.instructions[0].params["theta"].name == "theta"


def test_an_expression_marker_is_accepted_and_decoded() -> None:
    marker = {"$expression": {"op": "mul", "args": [2, {"$parameter": "theta"}]}}

    ir = resolve_ir(_gate("ry", [0], parameters={"theta": marker}), "qir")

    assert ir.instructions[0].params["theta"].op == "mul"


def test_a_bare_string_angle_is_rejected() -> None:
    """The failure this guards: a string that draws as if it were a symbol.

    Left alone the SDK stores it, ``draw_circuit_tool`` renders ``RY(theta)``
    exactly as it would for a real symbol, and the circuit is planned as though
    it were executable.
    """
    with pytest.raises(ToolInputError, match="is a string"):
        resolve_ir(_gate("ry", [0], parameters={"theta": "theta"}), "qir")


def test_a_misspelled_marker_is_rejected_rather_than_ignored() -> None:
    with pytest.raises(ToolInputError, match=r"\$parameter"):
        resolve_ir(_gate("ry", [0], parameters={"theta": {"$unknown": "theta"}}), "qir")


def test_a_non_string_parameter_name_is_rejected() -> None:
    """``{"$parameter": 3}`` decodes to ``Parameter("3")``: the name changes."""
    with pytest.raises(ToolInputError, match=r"\$parameter"):
        resolve_ir(_gate("ry", [0], parameters={"theta": {"$parameter": 3}}), "qir")


def test_an_empty_parameter_name_is_rejected() -> None:
    with pytest.raises(ToolInputError, match=r"\$parameter"):
        resolve_ir(_gate("ry", [0], parameters={"theta": {"$parameter": ""}}), "qir")


def test_a_null_angle_is_rejected() -> None:
    with pytest.raises(ToolInputError, match="null"):
        resolve_ir(_gate("ry", [0], parameters={"theta": None}), "qir")
