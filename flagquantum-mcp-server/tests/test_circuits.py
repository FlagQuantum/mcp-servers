"""Circuit payload resolution: formats, limits and round-tripping."""

from __future__ import annotations

import json
import re
from collections.abc import Callable

import pytest

from flagquantum_mcp_server.analysis import analyze
from flagquantum_mcp_server.circuits import (
    IR_FORMAT,
    QIR_FORMAT,
    deserialize,
    known_gate_names,
    resolve_ir,
    serialize,
)
from flagquantum_mcp_server.errors import (
    ToolInputError,
    ToolLimitError,
    UnsupportedFormatError,
)

pytestmark = pytest.mark.unit


def test_qir_payload_becomes_canonical_ir(bell_qir: str) -> None:
    ir = resolve_ir(bell_qir, QIR_FORMAT)

    assert ir.n_wires == 2
    assert [instruction.name for instruction in ir.instructions] == ["h", "cx"]
    assert ir.to_dict()["kind"] == "flagquantum.circuit_ir"


def test_ir_payload_is_accepted_directly(bell_qir: str) -> None:
    canonical = serialize(bell_qir, QIR_FORMAT)["ir_json"]

    ir = resolve_ir(canonical, IR_FORMAT)

    assert ir.n_wires == 2
    assert len(ir.instructions) == 2


def test_decoded_payload_is_accepted_as_well_as_text(bell_qir: str) -> None:
    from_text = resolve_ir(bell_qir, QIR_FORMAT)
    from_objects = resolve_ir(json.loads(bell_qir), QIR_FORMAT)

    assert from_text.content_hash == from_objects.content_hash


def test_serialize_reports_a_stable_content_hash(bell_qir: str) -> None:
    first = serialize(bell_qir, QIR_FORMAT)
    second = serialize(bell_qir, QIR_FORMAT)

    assert first["content_hash"] == second["content_hash"]
    assert first["ir_version"] == "1.0"
    assert first["n_instructions"] == 2


def test_round_trip_reports_stability(bell_qir: str) -> None:
    canonical = serialize(bell_qir, QIR_FORMAT)["ir_json"]

    result = deserialize(canonical)

    assert result["round_trip_stable"] is True
    assert result["content_hash"] == serialize(bell_qir, QIR_FORMAT)["content_hash"]


def test_round_trip_detects_non_canonical_input(bell_qir: str) -> None:
    canonical = serialize(bell_qir, QIR_FORMAT)["ir_json"]
    respaced = json.dumps(json.loads(canonical), indent=2)

    assert deserialize(respaced)["round_trip_stable"] is False


def test_round_trip_reports_the_input_not_the_requested_indent(bell_qir: str) -> None:
    """The flag is about the caller's text, not about how it is displayed.

    It used to compare against the re-serialization returned in the payload, so
    asking for ``indent=2`` flipped it to False for a payload that was already
    canonical — reporting a display preference as a stability problem.
    """
    canonical = serialize(bell_qir, QIR_FORMAT)["ir_json"]

    assert deserialize(canonical, indent=2)["round_trip_stable"] is True
    assert deserialize(canonical)["round_trip_stable"] is True

    respaced = json.dumps(json.loads(canonical), indent=2)
    assert deserialize(respaced, indent=2)["round_trip_stable"] is False


def test_unknown_format_is_rejected(bell_qir: str) -> None:
    with pytest.raises(UnsupportedFormatError, match="circuit_format must be one of"):
        resolve_ir(bell_qir, "qasm3")


def test_malformed_json_is_rejected_with_a_position() -> None:
    with pytest.raises(ToolInputError, match="not valid JSON"):
        resolve_ir("{not json", IR_FORMAT)


def test_payload_without_the_ir_discriminator_is_rejected(bell_qir: str) -> None:
    with pytest.raises(ToolInputError, match=re.escape("kind='flagquantum.circuit_ir'")):
        resolve_ir('{"n_wires": 2}', IR_FORMAT)


def test_ir_format_rejects_a_gate_list(bell_qir: str) -> None:
    with pytest.raises(ToolInputError, match="expects a JSON object"):
        resolve_ir(bell_qir, IR_FORMAT)


def test_qir_format_rejects_an_object(bell_qir: str) -> None:
    canonical = serialize(bell_qir, QIR_FORMAT)["ir_json"]

    with pytest.raises(ToolInputError, match="expects a JSON array"):
        resolve_ir(canonical, QIR_FORMAT)


def test_oversized_payload_is_rejected(monkeypatch: pytest.MonkeyPatch, bell_qir: str) -> None:
    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_IR_BYTES", "5")

    with pytest.raises(ToolLimitError, match="above the 5-byte limit"):
        resolve_ir(bell_qir, QIR_FORMAT)


def test_too_many_qubits_is_rejected(monkeypatch: pytest.MonkeyPatch, ghz3_qir: str) -> None:
    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_QUBITS", "2")

    with pytest.raises(ToolLimitError, match="3 qubits, above the 2-qubit limit"):
        resolve_ir(ghz3_qir, QIR_FORMAT)


def test_too_many_gates_is_rejected(monkeypatch: pytest.MonkeyPatch, ghz3_qir: str) -> None:
    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_GATES", "1")

    with pytest.raises(ToolLimitError, match="3 instructions, above the 1-instruction"):
        resolve_ir(ghz3_qir, QIR_FORMAT)


def test_unusable_limit_values_fall_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch, bell_qir: str
) -> None:
    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_QUBITS", "not-a-number")

    assert resolve_ir(bell_qir, QIR_FORMAT).n_wires == 2


def test_gate_set_matches_the_sdk_alias_table() -> None:
    names = known_gate_names()

    assert len(names) == 45
    assert {"h", "x", "cx", "cz", "rx", "rz", "ccx", "swap"} <= set(names)
    assert names == sorted(names)
    assert "run" not in names
    assert "analysis" not in names


# Gate-list (qir) shape validation.
#
# The SDK reports every malformation of a gate list as "max() iterable argument
# is empty", because it infers the wire count from the highest index. These
# tests pin the messages a caller actually needs; the first case is the one a
# model hits after reading the IR schema, since it reuses the IR key names.


def test_gate_using_ir_key_names_is_named_as_such() -> None:
    with pytest.raises(ToolInputError, match="uses the IR spelling 'opcode'"):
        resolve_ir(json.dumps([{"opcode": "h", "wires": [0]}]), QIR_FORMAT)


def test_gate_using_the_ir_wire_key_is_named_as_such() -> None:
    with pytest.raises(ToolInputError, match="IR spelling 'wires'"):
        resolve_ir(json.dumps([{"name": "h", "wires": [0]}]), QIR_FORMAT)


def test_gate_without_an_index_is_rejected_by_position() -> None:
    with pytest.raises(ToolInputError, match=r"Gate 1 \('cx'\) needs an 'index'"):
        resolve_ir(json.dumps([{"name": "h", "index": [0]}, {"name": "cx"}]), QIR_FORMAT)


def test_empty_gate_list_is_rejected() -> None:
    with pytest.raises(ToolInputError, match="cannot be empty"):
        resolve_ir("[]", QIR_FORMAT)


def test_gate_that_is_not_an_object_is_rejected() -> None:
    with pytest.raises(ToolInputError, match="must be a JSON object"):
        resolve_ir(json.dumps(["h"]), QIR_FORMAT)


def test_gate_without_a_name_is_rejected() -> None:
    with pytest.raises(ToolInputError, match="non-empty string 'name'"):
        resolve_ir(json.dumps([{"name": "", "index": [0]}]), QIR_FORMAT)


def test_gate_with_an_empty_index_is_rejected() -> None:
    with pytest.raises(ToolInputError, match="empty index"):
        resolve_ir(json.dumps([{"name": "h", "index": []}]), QIR_FORMAT)


def test_gate_with_a_non_integer_wire_is_rejected() -> None:
    with pytest.raises(ToolInputError, match="non-integer wire"):
        resolve_ir(json.dumps([{"name": "h", "index": ["0"]}]), QIR_FORMAT)


def test_gate_with_a_negative_wire_is_rejected() -> None:
    with pytest.raises(ToolInputError, match="negative wire -1"):
        resolve_ir(json.dumps([{"name": "h", "index": [-1]}]), QIR_FORMAT)


def test_gate_with_a_non_list_index_is_rejected() -> None:
    with pytest.raises(ToolInputError, match="expected a list of wire numbers"):
        resolve_ir(json.dumps([{"name": "h", "index": {"0": 1}}]), QIR_FORMAT)


def test_a_bare_integer_index_is_accepted_as_a_one_wire_target() -> None:
    """Writing "index": 0 is unambiguous, so it is accepted rather than refused."""
    ir = resolve_ir(json.dumps([{"name": "h", "index": 0}]), QIR_FORMAT)

    assert ir.n_wires == 1
    assert ir.instructions[0].wires == (0,)


# --- parameter values ---
#
# The IR path decodes ``$parameter`` into a live Parameter, but it does not
# reject a value it cannot decode — an unrecognized mapping stays an opaque
# dict. These pin the same boundary the qir path enforces, so a circuit cannot
# lose its symbol by arriving in one format rather than the other.


def _ir_with_params(params: dict[str, object]) -> str:
    """A one-gate IR payload whose instruction carries ``params``."""
    return json.dumps(
        {
            "kind": "flagquantum.circuit_ir",
            "version": "1.0",
            "n_wires": 1,
            "dtype": "complex64",
            "shape": [2],
            "instructions": [
                {"opcode": "ry", "wires": [0], "params": params, "matrix": None, "metadata": {}}
            ],
            "observables": [],
            "measurements": [],
            "metadata": {},
        }
    )


def test_ir_accepts_a_number_as_a_parameter_value() -> None:
    ir = resolve_ir(_ir_with_params({"theta": 0.5}), IR_FORMAT)

    assert ir.instructions[0].params == {"theta": 0.5}


def test_ir_accepts_a_parameter_marker() -> None:
    ir = resolve_ir(_ir_with_params({"theta": {"$parameter": "theta"}}), IR_FORMAT)

    assert ir.instructions[0].params["theta"].name == "theta"


def test_ir_rejects_a_bare_string_angle() -> None:
    with pytest.raises(ToolInputError, match="is a string"):
        resolve_ir(_ir_with_params({"theta": "theta"}), IR_FORMAT)


def test_ir_rejects_a_misspelled_marker() -> None:
    with pytest.raises(ToolInputError, match=r"\$parameter"):
        resolve_ir(_ir_with_params({"theta": {"$unknown": "theta"}}), IR_FORMAT)


def test_ir_rejects_a_non_string_parameter_name() -> None:
    with pytest.raises(ToolInputError, match=r"\$parameter"):
        resolve_ir(_ir_with_params({"theta": {"$parameter": 3}}), IR_FORMAT)


def test_ir_rejects_a_null_angle() -> None:
    with pytest.raises(ToolInputError, match="null"):
        resolve_ir(_ir_with_params({"theta": None}), IR_FORMAT)


def test_ir_names_the_offending_instruction() -> None:
    with pytest.raises(ToolInputError, match="Instruction 0"):
        resolve_ir(_ir_with_params({"theta": "theta"}), IR_FORMAT)


def test_the_payload_version_key_is_not_ir_version(bell_qir: str) -> None:
    """The payload says ``version``; tool results report ``ir_version``.

    Nothing in the payload's own example shows the field name, and a model that
    sends ``ir_version`` gets an unknown-key rejection, so this pins the
    asymmetry that the README and the server instructions both state.
    """
    canonical = json.loads(serialize(bell_qir, QIR_FORMAT)["ir_json"])
    assert "version" in canonical
    assert "ir_version" not in canonical

    renamed = {key: value for key, value in canonical.items() if key != "version"}
    renamed["ir_version"] = canonical["version"]
    with pytest.raises(ToolInputError, match="unknown field"):
        resolve_ir(json.dumps(renamed), IR_FORMAT)


def test_the_shape_field_is_carried_but_not_validated(bell_qir: str) -> None:
    """Pins what the ir-schema note claims: a wrong shape is accepted silently.

    It changes the payload's content hash and nothing else, which is why the
    note calls it part of the payload's identity rather than the circuit's.
    """
    canonical = json.loads(serialize(bell_qir, QIR_FORMAT)["ir_json"])
    canonical["shape"] = [7]

    wrong = analyze(json.dumps(canonical), IR_FORMAT)

    assert wrong["circuit"]["n_qubits"] == 2
    assert (
        wrong["circuit"]["content_hash"] != analyze(bell_qir, QIR_FORMAT)["circuit"]["content_hash"]
    )


# --- what the SDK coerces, and what it ignores ---
#
# The SDK validates the instruction list but does not validate wire *types*: it
# calls int() on each wire, so "1", true, 1.7 and 0.9 all become a wire index. A
# value that means nothing turns into a circuit that looks fine. The qir reader
# refuses all four, so the ir reader does too — the two formats must not
# disagree about the same circuit, which is the rule the whole module follows.


def _ir_payload(**overrides: object) -> str:
    """A one-gate IR envelope with individual fields overridden."""
    payload: dict[str, object] = {
        "kind": "flagquantum.circuit_ir",
        "version": "1.0",
        "n_wires": 2,
        "dtype": "complex64",
        "shape": [4],
        "instructions": [
            {"opcode": "h", "wires": [0], "params": {}, "matrix": None, "metadata": {}}
        ],
        "observables": [],
        "measurements": [],
        "metadata": {},
    }
    payload.update(overrides)
    return json.dumps(payload)


def _with_instruction(instruction: dict[str, object]) -> str:
    return _ir_payload(instructions=[instruction])


@pytest.mark.parametrize("wire", ["0", True, 1.7, 0.9, None, []])
def test_ir_refuses_a_wire_the_sdk_would_coerce(wire: object) -> None:
    with pytest.raises(ToolInputError, match="non-integer wire"):
        resolve_ir(_with_instruction({"opcode": "h", "wires": [wire], "params": {}}), IR_FORMAT)


def test_ir_still_accepts_integer_wires() -> None:
    ir = resolve_ir(_with_instruction({"opcode": "h", "wires": [0], "params": {}}), IR_FORMAT)

    assert ir.instructions[0].wires == (0,)


def test_qir_refuses_the_same_wire_values() -> None:
    """The control for the test above: the two formats must agree."""
    for wire in ["0", True, 1.7]:
        with pytest.raises(ToolInputError, match="non-integer wire"):
            resolve_ir(json.dumps([{"name": "h", "index": [wire]}]), QIR_FORMAT)


# --- observables and measurements ---
#
# The SDK calls .get() on each entry without checking it is an object, so a
# string or a bare object raises AttributeError inside the SDK. That used to
# reach the client as INTERNAL_ERROR — this server reporting a bug when the
# payload is simply the wrong shape.


@pytest.mark.parametrize("field", ["observables", "measurements"])
@pytest.mark.parametrize("value", ["ZZ", {"name": "ZZ"}, 1, None])
def test_ir_names_a_malformed_observables_or_measurements_field(field: str, value: object) -> None:
    with pytest.raises(ToolInputError, match=field):
        resolve_ir(_ir_payload(**{field: value}), IR_FORMAT)


@pytest.mark.parametrize("field", ["observables", "measurements"])
def test_ir_names_a_malformed_entry_inside_those_fields(field: str) -> None:
    with pytest.raises(ToolInputError, match=rf"{field}'\[0\]"):
        resolve_ir(_ir_payload(**{field: ["ZZ"]}), IR_FORMAT)


def test_a_real_observable_and_measurement_are_still_accepted() -> None:
    """The control: the shapes FlagQuantum itself serializes must load."""
    payload = _ir_payload(
        observables=[{"name": "ZZ", "wires": [0, 1], "coefficient": 1.0, "metadata": {}}],
        measurements=[{"kind": "counts", "wires": [0, 1], "shots": 100, "metadata": {}}],
    )

    ir = resolve_ir(payload, IR_FORMAT)

    assert ir.to_dict()["observables"][0]["name"] == "zz"
    assert ir.to_dict()["measurements"][0]["kind"] == "counts"


# --- a matrix attached to a gate that already has a definition ---
#
# A matrix belongs to the opcode the SDK reserves for it, ``any``. Attached to a
# built-in name the SDK keeps the matrix in the payload but lets the built-in's
# own definition win, so analyze reports the gate under the built-in's name
# while the emitters refuse to lower it. Two tools describing the same circuit
# as different things.


IDENTITY = [[1.0, 0.0], [0.0, 1.0]]


def test_ir_refuses_a_matrix_on_a_builtin_opcode() -> None:
    instruction = {"opcode": "h", "wires": [0], "params": {}, "matrix": IDENTITY}

    with pytest.raises(ToolInputError, match="built-in gate"):
        resolve_ir(_with_instruction(instruction), IR_FORMAT)


def test_qir_refuses_a_matrix_on_a_builtin_name() -> None:
    gate = json.dumps([{"name": "h", "index": [0], "gate": IDENTITY}])

    with pytest.raises(ToolInputError, match="built-in gate"):
        resolve_ir(gate, QIR_FORMAT)


def test_the_refusal_points_at_the_opcode_reserved_for_a_matrix() -> None:
    instruction = {"opcode": "cx", "wires": [0, 1], "params": {}, "matrix": IDENTITY}

    with pytest.raises(ToolInputError, match='"any"'):
        resolve_ir(_with_instruction(instruction), IR_FORMAT)


def test_the_reserved_opcode_still_takes_a_matrix() -> None:
    gate = json.dumps([{"name": "any", "index": [0], "gate": IDENTITY}])

    ir = resolve_ir(gate, QIR_FORMAT)

    assert ir.to_dict()["instructions"][0]["matrix"] is not None


def test_an_unknown_name_may_still_carry_a_matrix() -> None:
    """A custom opcode is how a caller names their own unitary."""
    gate = json.dumps([{"name": "mygate", "index": [0], "gate": IDENTITY}])

    ir = resolve_ir(gate, QIR_FORMAT)

    assert ir.to_dict()["instructions"][0]["opcode"] == "mygate"


# --- the matrix key differs by format, and the qir reader ignores the IR one ---


def test_qir_names_the_ir_spelling_of_the_matrix_key() -> None:
    """``Circuit.from_qir`` reads ``gate`` and ignores ``matrix`` entirely.

    So a caller who spells it the IR way, as this server's own IR schema shows
    it, gets a gate built without the matrix — silently, and for a built-in name
    that means a different circuit than they asked for.
    """
    gate = json.dumps([{"name": "h", "index": [0], "matrix": IDENTITY}])

    with pytest.raises(ToolInputError, match="IR spelling 'matrix'"):
        resolve_ir(gate, QIR_FORMAT)


def test_qir_names_the_ir_spelling_before_complaining_about_the_name() -> None:
    """A custom name defined only by ``matrix`` must not be reported as unknown.

    The name check used to accept any gate carrying a ``matrix`` key, because it
    could not tell the two spellings apart; the circuit then failed later with
    "unknown opcode", which names the wrong problem.
    """
    gate = json.dumps([{"name": "mygate", "index": [0], "matrix": IDENTITY}])

    with pytest.raises(ToolInputError, match="IR spelling 'matrix'"):
        resolve_ir(gate, QIR_FORMAT)


def test_the_reserved_matrix_opcode_is_the_one_the_sdk_emits() -> None:
    """Pins ``MATRIX_OPCODE`` against the SDK rather than trusting the name.

    ``_check_matrix`` allows a matrix only when the opcode has no signature, so
    it depends on the reserved opcode being outside the built-in manifest. If
    FlagQuantum ever renamed it, that check would start refusing every explicit
    unitary instead of only the ambiguous ones — a false positive on legal
    input, which is the failure mode worth a test.
    """
    import flagquantum as fq

    from flagquantum_mcp_server.circuits import MATRIX_OPCODE
    from flagquantum_mcp_server.gates import signature_or_none

    emitted = fq.Circuit(1).any(0, unitary=IDENTITY).to_ir().to_dict()["instructions"][0]

    assert emitted["opcode"] == MATRIX_OPCODE
    assert emitted["matrix"] is not None
    assert signature_or_none(MATRIX_OPCODE) is None, (
        f"{MATRIX_OPCODE!r} is in the built-in manifest, so _check_matrix would "
        "refuse every explicit unitary"
    )


# --- the wire list itself, not just its entries ---
#
# The check above refuses a wire whose *value* is not an integer. This is the
# other half: `wires` that is not a list at all. The SDK iterates whatever it is
# handed and int()s each element, so "01" becomes wires 0 and 1 and {"0": 1}
# becomes wire 0 — a payload the caller meant one way read another way.


@pytest.mark.parametrize("wires", ["0", "01", {"0": 1}, 1, None, 2.5])
def test_ir_refuses_a_wires_field_that_is_not_a_list(wires: object) -> None:
    with pytest.raises(ToolInputError, match="wires that are"):
        resolve_ir(_with_instruction({"opcode": "h", "wires": wires, "params": {}}), IR_FORMAT)


def test_ir_still_accepts_a_list_of_wires() -> None:
    ir = resolve_ir(_with_instruction({"opcode": "cx", "wires": [0, 1], "params": {}}), IR_FORMAT)

    assert ir.instructions[0].wires == (0, 1)


def test_an_absent_wires_key_is_left_to_the_sdk() -> None:
    """Absent is not the same as null, and the SDK's message for absent is good."""
    with pytest.raises(ToolInputError, match="requires at least one wire"):
        resolve_ir(_with_instruction({"opcode": "h", "params": {}}), IR_FORMAT)


@pytest.mark.parametrize("wires", ["01", 0, None, {"0": 1}])
def test_the_same_rule_applies_to_observables_and_measurements(wires: object) -> None:
    payload = _ir_payload(observables=[{"name": "zz", "wires": wires}])

    with pytest.raises(ToolInputError, match=r"observables'\[0\]"):
        resolve_ir(payload, IR_FORMAT)


# --- envelope fields the SDK coerces ---


@pytest.mark.parametrize("width", ["2", 2.7, True, [2], None])
def test_ir_refuses_a_coerced_n_wires(width: object) -> None:
    with pytest.raises(ToolInputError, match="'n_wires'"):
        resolve_ir(_ir_payload(n_wires=width), IR_FORMAT)


@pytest.mark.parametrize("shape", ["4", 4, [2.5], [True], [None]])
def test_ir_refuses_a_coerced_shape(shape: object) -> None:
    with pytest.raises(ToolInputError, match="'shape'"):
        resolve_ir(_ir_payload(shape=shape), IR_FORMAT)


@pytest.mark.parametrize("dtype", [5, None, [], ["complex64"], 1.5])
def test_ir_refuses_a_non_string_dtype(dtype: object) -> None:
    """A non-string dtype reaches a torch attribute lookup inside the SDK.

    That raises AttributeError from a library, which used to reach the client as
    INTERNAL_ERROR — this server reporting a bug over a payload that is simply
    wrong. The value check stays with the SDK, whose message names the rule
    ("complex_dtype must be complex64 or complex128").
    """
    with pytest.raises(ToolInputError, match="'dtype'"):
        resolve_ir(_ir_payload(dtype=dtype), IR_FORMAT)


@pytest.mark.parametrize("dtype", ["complex64", "complex128"])
def test_ir_still_accepts_the_dtypes_the_sdk_writes(dtype: str) -> None:
    assert resolve_ir(_ir_payload(dtype=dtype), IR_FORMAT).dtype == dtype


@pytest.mark.parametrize("dtype", ["float32", "nonsense", "complex32"])
def test_a_dtype_the_sdk_cannot_build_with_is_an_input_error(dtype: str) -> None:
    """Resolving succeeds; rebuilding the Circuit is where the SDK objects.

    ``float32`` is a real torch dtype but not a legal complex width, ``complex32``
    is the reverse, and ``nonsense`` is neither. All three reach the same
    attribute lookup or rule check inside the SDK, and all three used to arrive
    at the client as INTERNAL_ERROR.
    """
    with pytest.raises(ToolInputError):
        analyze(_ir_payload(dtype=dtype), IR_FORMAT)


def test_a_dtype_torch_does_not_have_names_the_field() -> None:
    with pytest.raises(ToolInputError, match="'dtype' field is the most likely cause"):
        analyze(_ir_payload(dtype="nonsense"), IR_FORMAT)


# --- error message grammar ---
#
# `_describe` used to include its own verb, so every caller that added one
# produced "is is a string" — three separate occurrences. It now returns a noun
# phrase, and this reads every message the boundary can produce to be sure.


def _every_rejection_message() -> list[str]:
    """Collect the message from a wide sweep of malformed payloads."""
    payloads = (
        [
            _with_instruction({"opcode": "h", "wires": w, "params": {}})
            for w in ["0", None, 1, {"a": 1}, [True]]
        ]
        + [
            _ir_payload(**kw)
            for kw in [
                {"n_wires": "2"},
                {"n_wires": 2.7},
                {"shape": "4"},
                {"shape": [2.5]},
                {"dtype": 5},
                {"dtype": []},
                {"observables": "ZZ"},
                {"observables": [1]},
                {"observables": None},
                {"observables": [{"name": "zz", "wires": "01"}]},
                {"measurements": {"kind": "counts"}},
            ]
        ]
        + [
            _with_instruction({"opcode": "ry", "wires": [0], "params": {"theta": v}})
            for v in ["theta", None, [1], {}, {"$bogus": 1}, {"$parameter": 3}]
        ]
        + [
            _with_instruction(
                {"opcode": "ry", "wires": [0], "params": {"theta": {"$expression": e}}}
            )
            for e in ["mul", {"op": "mul"}, {"op": "m", "args": 2}]
        ]
        + [
            _with_instruction({"opcode": "h", "wires": [0], "params": {}, "matrix": IDENTITY}),
            _with_instruction({"opcode": "cx", "wires": [0], "params": {}}),
            _with_instruction({"opcode": "nope", "wires": [0], "params": {}}),
        ]
    )

    messages: list[str] = []
    for payload in payloads:
        with pytest.raises(ToolInputError) as caught:
            resolve_ir(payload, IR_FORMAT)
        messages.append(str(caught.value))
    return messages


def test_no_rejection_message_says_is_is_or_is_has() -> None:
    """Guards the grammar of every message, since they are read by a model."""
    joined = "\n".join(_every_rejection_message())

    assert " is is " not in joined
    assert " is has " not in joined
    assert " has wires has " not in joined


def test_every_rejection_message_is_substantial() -> None:
    """A refusal a caller cannot act on is barely better than a silent one."""
    for message in _every_rejection_message():
        assert len(message) > 60, message
        assert message[0].isupper(), message


# --- the systematic sweep ---
#
# Everything above was found by hand, one probe at a time, and each pass found
# only what the probe happened to aim at. This is the same search written down:
# every field of the envelope, and of an instruction, crossed with every JSON
# value kind. A payload may be accepted (some are legal) or refused — what it
# may never do is leave as an unhandled exception, because the server promises
# a structured envelope and a raw `AttributeError` from inside a library breaks
# that promise and reads as a bug here rather than a mistake by the caller.


JSON_VALUES: dict[str, object] = {
    "string": "x",
    "integer": 1,
    "float": 1.5,
    "boolean": True,
    "null": None,
    "empty-list": [],
    "empty-object": {},
    "list-of-strings": ["x"],
    "object-of-strings": {"a": 1},
}

ENVELOPE_FIELDS = [
    "kind",
    "version",
    "n_wires",
    "dtype",
    "shape",
    "instructions",
    "observables",
    "measurements",
    "metadata",
]

INSTRUCTION_FIELDS = ["opcode", "wires", "params", "matrix", "metadata"]

# The calls the sweep makes: a circuit payload reaching each kind of reader.
READERS = [
    ("analyze", analyze, lambda payload: {"circuit": payload, "circuit_format": "ir"}),
    (
        "serialize",
        serialize,
        lambda payload: {"circuit": payload, "circuit_format": "ir"},
    ),
    (
        "deserialize",
        deserialize,
        lambda payload: {"ir_json": payload},
    ),
]


def _envelope(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": "flagquantum.circuit_ir",
        "version": "1.0",
        "n_wires": 2,
        "dtype": "complex64",
        "shape": [4],
        "instructions": [
            {"opcode": "h", "wires": [0], "params": {}, "matrix": None, "metadata": {}}
        ],
        "observables": [],
        "measurements": [],
        "metadata": {},
    }
    payload.update(overrides)
    return payload


def _swept_payloads() -> list[tuple[str, dict[str, object]]]:
    """Every field crossed with every JSON value kind."""
    payloads: list[tuple[str, dict[str, object]]] = []
    for field in ENVELOPE_FIELDS:
        for kind, value in JSON_VALUES.items():
            payloads.append((f"{field}={kind}", _envelope(**{field: value})))
    for field in INSTRUCTION_FIELDS:
        for kind, value in JSON_VALUES.items():
            instruction = {
                "opcode": "h",
                "wires": [0],
                "params": {},
                "matrix": None,
                "metadata": {},
            }
            instruction[field] = value
            payloads.append(
                (f"instructions[0].{field}={kind}", _envelope(instructions=[instruction]))
            )
    return payloads


def _assert_structured(
    reader_name: str,
    reader: Callable[..., object],
    arguments: Callable[[str], dict[str, object]],
    payload: dict[str, object],
) -> None:
    """Refuse anything but the two structured error types.

    Pulled out of the loop so the ``except Exception`` is not written inside one
    — catching everything is the whole point of the test, not an oversight, and
    keeping it in a helper keeps that legible.
    """
    try:
        reader(**arguments(json.dumps(payload)))
    except (ToolInputError, ToolLimitError):
        return
    except Exception as exc:
        raise AssertionError(
            f"{reader_name} raised {type(exc).__name__} instead of a structured error: {exc}"
        ) from exc


@pytest.mark.parametrize(("label", "payload"), _swept_payloads(), ids=lambda v: str(v)[:60])
def test_no_payload_escapes_as_an_unhandled_exception(
    label: str, payload: dict[str, object]
) -> None:
    del label  # carried by the parametrize id, which is what a failure prints

    for reader_name, reader, arguments in READERS:
        _assert_structured(reader_name, reader, arguments, payload)
