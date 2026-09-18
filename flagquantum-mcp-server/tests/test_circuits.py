"""Circuit payload resolution: formats, limits and round-tripping."""

from __future__ import annotations

import json
import re

import pytest

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
