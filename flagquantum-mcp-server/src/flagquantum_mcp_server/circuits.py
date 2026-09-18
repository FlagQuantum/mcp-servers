"""Resolve caller input into a validated FlagQuantum circuit IR.

Two input formats are accepted, both of them the SDK's own serialization:

``ir``
    FlagQuantum's versioned circuit IR (``flagquantum.circuit_ir``), as
    produced by ``CircuitIR.to_json()``. This is the canonical format: it
    carries a version, a content hash, and strict unknown-field rejection.
``qir``
    The compact gate list produced by ``Circuit.to_qir()``. More ergonomic to
    write by hand, at the cost of the IR's version and hash guarantees.

Every tool funnels through :func:`resolve_ir`, so the size, width and gate
bounds in :mod:`flagquantum_mcp_server.limits` apply uniformly.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Literal

from flagquantum_mcp_server import limits
from flagquantum_mcp_server._bridge import load_sdk
from flagquantum_mcp_server.errors import (
    ToolInputError,
    ToolLimitError,
    UnsupportedFormatError,
)
from flagquantum_mcp_server.gates import closest_names, known_opcodes, signature_or_none

# The accepted input formats. Declared as a closed set rather than plain
# ``str`` so that FastMCP turns it into a JSON-schema ``enum``: a model that
# sees the enum does not try "openqasm", "IR" or "qasm3", while a model that
# only sees ``string`` does. It also lets the runtime check below act as a
# second line of defence for callers that bypass the MCP layer.
CircuitFormat = Literal["ir", "qir"]

IR_FORMAT: CircuitFormat = "ir"
QIR_FORMAT: CircuitFormat = "qir"
SUPPORTED_FORMATS: tuple[CircuitFormat, ...] = (IR_FORMAT, QIR_FORMAT)
IR_KIND = "flagquantum.circuit_ir"

CircuitPayload = str | Mapping[str, Any] | Sequence[Any]

# The parameter encodings the SDK's own decoder understands. A value mapping
# that carries one of these keys becomes a live ``Parameter``,
# ``ParameterExpression``, ``complex`` or tensor; a mapping that carries none of
# them is kept as an opaque dict instead. That silent fall-through is how a
# misspelled marker turns a parameterized circuit into one that reports "no
# parameters", so the input boundary rejects it rather than passing it on.
VALUE_MARKERS: tuple[str, ...] = ("$parameter", "$expression", "$complex", "$tensor")


def resolve_ir(circuit: CircuitPayload, circuit_format: CircuitFormat = IR_FORMAT) -> Any:
    """Parse caller input and return a validated ``CircuitIR``.

    Args:
        circuit: Serialized circuit, either as JSON text or an already-decoded
            mapping / sequence.
        circuit_format: Either ``"ir"`` or ``"qir"``.

    Returns:
        A ``flagquantum.CircuitIR``. The concrete type is not annotated because
        importing FlagQuantum at module scope would defeat the lazy binding in
        :mod:`flagquantum_mcp_server._bridge`.

    Raises:
        UnsupportedFormatError: If ``circuit_format`` is not recognized.
        ToolInputError: If the payload cannot be parsed or fails IR validation.
        ToolLimitError: If the circuit exceeds a configured bound.
    """
    if circuit_format not in SUPPORTED_FORMATS:
        raise UnsupportedFormatError(
            f"circuit_format must be one of {list(SUPPORTED_FORMATS)}; received {circuit_format!r}."
        )

    decoded = _decode(circuit)
    ir = _ir_from_decoded(decoded) if circuit_format == IR_FORMAT else _ir_from_qir(decoded)
    return enforce_limits(ir)


def _decode(circuit: CircuitPayload) -> Any:
    """Turn JSON text into objects, passing decoded payloads through."""
    if not isinstance(circuit, str):
        return circuit
    encoded = circuit.encode("utf-8")
    if len(encoded) > limits.max_ir_bytes():
        raise ToolLimitError(
            f"Circuit payload is {len(encoded)} bytes, above the "
            f"{limits.max_ir_bytes()}-byte limit. "
            "Raise FLAGQUANTUM_MCP_MAX_IR_BYTES or send a smaller circuit."
        )
    try:
        return json.loads(circuit)
    except json.JSONDecodeError as exc:
        raise ToolInputError(
            f"Circuit payload is not valid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}."
        ) from exc


def _ir_from_decoded(decoded: Any) -> Any:
    """Build a CircuitIR from a decoded ``ir``-format payload."""
    if not isinstance(decoded, Mapping):
        raise ToolInputError(
            "circuit_format='ir' expects a JSON object with the "
            "flagquantum.circuit_ir fields; received "
            f"{type(decoded).__name__}."
        )
    kind = decoded.get("kind")
    if kind != IR_KIND:
        raise ToolInputError(
            f"Expected a circuit IR with kind={IR_KIND!r}; received kind={kind!r}. "
            "Use serialize_circuit_tool to produce a valid payload, or pass "
            "circuit_format='qir' for a gate list."
        )
    _check_ir_parameter_values(decoded)
    sdk = load_sdk()
    try:
        return sdk.CircuitIR.from_dict(decoded)
    except (ValueError, TypeError, KeyError) as exc:
        raise ToolInputError(f"Circuit IR failed validation: {exc}") from exc


def _check_ir_parameter_values(decoded: Mapping[str, Any]) -> None:
    """Check every instruction's parameter values in a decoded IR payload.

    The envelope is validated by the SDK, but a parameter *value* is not: a
    bare string or a misspelled marker is stored as an opaque value, and the
    circuit then reports itself as unparameterized. Checking here keeps the two
    input formats from disagreeing about the same circuit.
    """
    instructions = decoded.get("instructions")
    if not isinstance(instructions, Sequence) or isinstance(instructions, (str, bytes)):
        return
    for position, instruction in enumerate(instructions):
        if isinstance(instruction, Mapping):
            _check_parameter_values(
                instruction.get("params"),
                str(instruction.get("opcode")),
                f"Instruction {position}",
            )


def _ir_from_qir(decoded: Any) -> Any:
    """Build a CircuitIR from a decoded ``qir``-format gate list.

    The gate shape is checked here rather than left to the SDK. The SDK infers
    the wire count from the highest index and reports every malformation as
    ``max() iterable argument is empty``, which tells a caller nothing — and the
    most likely malformation is a caller reusing the key names from our own IR
    schema (``opcode``/``wires`` instead of ``name``/``index``).

    The finished circuit is round-tripped through ``CircuitIR.from_dict``
    because ``Circuit.from_qir`` takes parameter values literally: a
    ``{"$parameter": ...}`` marker handed to it is stored as a plain dict rather
    than decoded into a ``Parameter``. The circuit then reports no parameters
    while serialization echoes the marker back, so ``inspect_parameters_tool``
    and ``serialize_circuit_tool`` contradict each other. Re-decoding is what
    its own serializer produced restores the symbol, and costs nothing for a
    circuit that carries no marker.
    """
    if not isinstance(decoded, Sequence) or isinstance(decoded, (str, bytes)):
        raise ToolInputError(
            "circuit_format='qir' expects a JSON array of gate objects such as "
            '[{"name": "h", "index": [0]}, {"name": "cx", "index": [0, 1]}]; '
            f"received {type(decoded).__name__}."
        )
    if not decoded:
        raise ToolInputError(
            "A gate list cannot be empty: the wire count is inferred from the "
            "highest index in the list, so an empty list describes no circuit."
        )
    gates = [_validate_gate(item, position) for position, item in enumerate(decoded)]
    sdk = load_sdk()
    try:
        built = sdk.Circuit.from_qir(gates).to_ir()
        return sdk.CircuitIR.from_dict(built.to_dict())
    except (ValueError, TypeError, KeyError) as exc:
        raise ToolInputError(f"Gate list could not be built into a circuit: {exc}") from exc


def _validate_gate(gate: Any, position: int) -> dict[str, Any]:
    """Check one gate object and normalize its wire indices.

    Args:
        gate: The decoded gate object.
        position: Zero-based position, used to name the offending entry.

    Returns:
        The gate with ``index`` normalized to a list of ints.

    Raises:
        ToolInputError: If the gate is not an object, is missing ``name`` or
            ``index``, or carries indices that are not in-range integers.
    """
    prefix = f"Gate {position}"
    if not isinstance(gate, Mapping):
        raise ToolInputError(
            f"{prefix} must be a JSON object with 'name' and 'index'; "
            f"received {type(gate).__name__}."
        )
    if "name" not in gate and "opcode" in gate:
        raise ToolInputError(
            f"{prefix} uses the IR spelling 'opcode'. The qir format expects "
            "'name' and 'index', not the IR keys "
            f"({sorted(gate)}). Convert with circuit_format='ir', or rename the "
            'keys: {"name": "h", "index": [0]}.'
        )
    name = gate.get("name")
    if not isinstance(name, str) or not name:
        raise ToolInputError(f"{prefix} needs a non-empty string 'name'; received {name!r}.")
    if "index" not in gate and "wires" in gate:
        raise ToolInputError(
            f"{prefix} uses the IR spelling 'wires'. The qir format calls the same field 'index'."
        )
    index = gate.get("index")
    if index is None:
        raise ToolInputError(f"{prefix} ({name!r}) needs an 'index' naming its wires.")
    if isinstance(index, int) and not isinstance(index, bool):
        index = [index]
    if not isinstance(index, (list, tuple)):
        raise ToolInputError(
            f"{prefix} ({name!r}) has index {index!r}; expected a list of wire "
            "numbers such as [0] or [0, 1]."
        )
    wires: list[int] = []
    for wire in index:
        if isinstance(wire, bool) or not isinstance(wire, int):
            raise ToolInputError(
                f"{prefix} ({name!r}) has non-integer wire {wire!r}; wire numbers must be integers."
            )
        if wire < 0:
            raise ToolInputError(f"{prefix} ({name!r}) has negative wire {wire}.")
        wires.append(wire)
    if not wires:
        raise ToolInputError(f"{prefix} ({name!r}) has an empty index; name at least one wire.")
    if "params" in gate and "parameters" not in gate:
        raise ToolInputError(
            f"{prefix} ({name!r}) uses the key 'params'. A qir gate carries its "
            "arguments under 'parameters': "
            '{"name": "rz", "index": [0], "parameters": {"theta": 0.5}}.'
        )
    _check_known_name(name, gate, prefix)
    _check_signature(name, wires, gate.get("parameters"), prefix)
    _check_parameter_values(gate.get("parameters"), name, prefix)
    return {**gate, "index": wires}


def _check_known_name(name: str, gate: Mapping[str, Any], prefix: str) -> None:
    """Name a gate the SDK does not have, and suggest what was meant.

    A custom opcode is legal — ``any`` carries a matrix under the ``gate`` key —
    so an unknown name is only an error when the gate carries no matrix to
    define it. The SDK's own message for this is terse and does not guess.
    """
    if signature_or_none(name) is not None:
        return
    if gate.get("gate") is not None or gate.get("matrix") is not None:
        return
    raise ToolInputError(
        f"{prefix} ({name!r}) is not a gate in the installed FlagQuantum, and it "
        "carries no matrix. Closest names: "
        f"{closest_names(name)}. Call describe_gate_set_tool for the full list."
    )


def _check_signature(name: str, wires: list[int], params: Any, prefix: str) -> None:
    """Check a gate against the SDK's own arity and parameter names.

    A gate list is written by hand, so a wrong wire count or a misspelled
    parameter is a normal mistake rather than an exotic one. The SDK reports
    both as a bare wire-count complaint, which does not say what the gate
    actually expects — and a misspelled parameter is worse, because it can be
    accepted and then silently ignored.

    Gates outside the built-in manifest (``any``, which carries a matrix) are
    skipped rather than rejected.

    Args:
        name: The gate name as written, possibly an alias.
        wires: The wires the caller gave.
        params: The caller's ``params`` mapping, if any.
        prefix: Position label used to name the offending gate.

    Raises:
        ToolInputError: If the wire count or a parameter name is wrong.
    """
    expected = signature_or_none(name)
    if expected is None:
        return
    arity = int(expected["arity"])
    if len(wires) != arity:
        spelling = "" if expected["opcode"] == name else f" (alias of {expected['opcode']!r})"
        raise ToolInputError(
            f"{prefix} {name!r}{spelling} takes {arity} wire(s), but {len(wires)} "
            f"were given. Wires for this gate: {wires}."
        )
    accepted = tuple(expected["parameters"])
    if not isinstance(params, Mapping):
        if params is not None:
            raise ToolInputError(
                f"{prefix} ({name!r}) has params {params!r}; expected an object "
                "mapping parameter names to values."
            )
        params = {}
    unknown = sorted(set(params) - set(accepted))
    if unknown:
        raise ToolInputError(
            f"{prefix} ({name!r}) has unknown parameter(s) {unknown}. "
            + (
                f"This gate takes {list(accepted)}, in that order."
                if accepted
                else "This gate takes no parameters."
            )
        )
    missing = sorted(set(accepted) - set(params))
    if missing:
        raise ToolInputError(
            f"{prefix} ({name!r}) is missing parameter(s) {missing}. "
            f"This gate takes {list(accepted)}, in that order."
        )


def _check_parameter_values(params: Any, gate: str, prefix: str) -> None:
    """Reject a parameter value the SDK would store without understanding it.

    The SDK accepts any JSON value in a parameter slot. A number is a bound
    angle and a :data:`VALUE_MARKERS` mapping is a symbol or an expression, but
    a bare string or a misspelled marker is neither: it is stored as an opaque
    value that no longer reads as a parameter. The circuit then reports
    ``is_parameterized: false`` while a diagram renders the string exactly as it
    renders a real symbol and execution is planned without complaint — so the
    mistake surfaces at the far end of the pipeline, if at all.

    Gates outside the built-in manifest are checked too: their parameter names
    are unknown, but a value's shape is not.

    Args:
        params: The gate's ``parameters`` mapping, if any.
        gate: Gate name as written, used in the message.
        prefix: Position label used to name the offending instruction.

    Raises:
        ToolInputError: If a value is neither a number nor a known encoding.
    """
    if not isinstance(params, Mapping):
        return
    for name, value in params.items():
        _check_parameter_value(value, str(name), gate, prefix)


def _check_parameter_value(value: Any, name: str, gate: str, prefix: str) -> None:
    """Check one parameter value against the shapes the SDK can decode."""
    if isinstance(value, (int, float)):
        return
    if isinstance(value, Mapping):
        _check_marker(value, name, gate, prefix)
        return
    reason = (
        "a string is stored as an ordinary value rather than a symbol"
        if isinstance(value, str)
        else f"a parameter value is a number or one of {list(VALUE_MARKERS)}"
    )
    raise ToolInputError(
        f"{prefix} ({gate!r}) parameter {name!r} {_describe(value)}. A gate "
        "argument is either a number or a symbol written as "
        f'{{"$parameter": "{name}"}}; {reason}, so the circuit would report '
        "itself as unparameterized."
    )


def _check_marker(value: Mapping[str, Any], name: str, gate: str, prefix: str) -> None:
    """Check a mapping parameter value against the SDK's marker vocabulary."""
    if not any(marker in value for marker in VALUE_MARKERS):
        raise ToolInputError(
            f"{prefix} ({gate!r}) parameter {name!r} has the value {dict(value)!r}, "
            f"which carries none of the encodings {list(VALUE_MARKERS)}. A symbol "
            f'is written as {{"$parameter": "{name}"}}. The SDK keeps an '
            "unrecognized marker as an opaque value rather than a symbol, so the "
            "circuit would report itself as unparameterized."
        )
    if "$parameter" in value:
        symbol = value["$parameter"]
        if not isinstance(symbol, str) or not symbol:
            raise ToolInputError(
                f"{prefix} ({gate!r}) parameter {name!r} has the '$parameter' "
                f"marker {symbol!r}; it must be a non-empty string naming the "
                f'symbol, as in {{"$parameter": "{name}"}}.'
            )


def _describe(value: Any) -> str:
    """Name a rejected value in the terms the caller wrote it in."""
    if isinstance(value, str):
        return f"is a string ({value!r})"
    if value is None:
        return "is null"
    if isinstance(value, (list, tuple)):
        return f"is a list ({value!r})"
    return f"has unsupported value {value!r}"


def enforce_limits(ir: Any) -> Any:
    """Reject a circuit that exceeds the configured width or gate bound.

    Args:
        ir: A ``flagquantum.CircuitIR``.

    Returns:
        The same IR, when it is within bounds.

    Raises:
        ToolLimitError: If the circuit is wider or larger than allowed.
    """
    n_wires = int(ir.n_wires)
    if n_wires > limits.max_qubits():
        raise ToolLimitError(
            f"Circuit has {n_wires} qubits, above the {limits.max_qubits()}-qubit "
            "limit. Raise FLAGQUANTUM_MCP_MAX_QUBITS to allow it."
        )
    n_gates = len(ir.instructions)
    if n_gates > limits.max_gates():
        raise ToolLimitError(
            f"Circuit has {n_gates} instructions, above the "
            f"{limits.max_gates()}-instruction limit. "
            "Raise FLAGQUANTUM_MCP_MAX_GATES to allow it."
        )
    return ir


def ir_to_json(ir: Any, *, indent: int | None = None) -> str:
    """Serialize a circuit IR to JSON text.

    Args:
        ir: A ``flagquantum.CircuitIR``.
        indent: Optional indentation width for readability.

    Returns:
        The serialized IR.

    Raises:
        ToolLimitError: If the serialized form exceeds the payload bound.
    """
    text = str(ir.to_json(indent=indent))
    size = len(text.encode("utf-8"))
    if size > limits.max_ir_bytes():
        raise ToolLimitError(
            f"Serialized circuit is {size} bytes, above the "
            f"{limits.max_ir_bytes()}-byte limit. "
            "Raise FLAGQUANTUM_MCP_MAX_IR_BYTES or send a smaller circuit."
        )
    return text


def circuit_from_ir(ir: Any) -> Any:
    """Rebuild a live ``Circuit`` from a validated IR.

    Args:
        ir: A ``flagquantum.CircuitIR``.

    Returns:
        A ``flagquantum.Circuit``.
    """
    return load_sdk().Circuit.from_ir(ir)


def serialize(
    circuit: CircuitPayload,
    circuit_format: CircuitFormat = QIR_FORMAT,
    *,
    indent: int | None = None,
) -> dict[str, Any]:
    """Canonicalize a circuit into FlagQuantum IR JSON.

    Accepts either input format and always answers with the canonical ``ir``
    form, so an agent that built a circuit as a gate list gets back the
    versioned, hashable representation.

    Args:
        circuit: Serialized circuit, as JSON text or a decoded payload.
        circuit_format: Either ``"ir"`` or ``"qir"``.
        indent: Optional indentation width for readability.

    Returns:
        A payload carrying the IR JSON, its content hash and its version.

    Raises:
        ToolError: If the payload is invalid or exceeds a configured bound.
    """
    ir = resolve_ir(circuit, circuit_format)
    return {
        "status": "success",
        "format": IR_FORMAT,
        "ir_json": ir_to_json(ir, indent=indent),
        "n_qubits": int(ir.n_wires),
        "n_instructions": len(ir.instructions),
        "ir_version": str(ir.version),
        "content_hash": str(ir.content_hash),
    }


def deserialize(ir_json: str, *, indent: int | None = None) -> dict[str, Any]:
    """Validate IR JSON and report whether it round-trips unchanged.

    Args:
        ir_json: FlagQuantum IR JSON text.
        indent: Optional indentation width for the canonical re-serialization.

    Returns:
        A payload with the circuit identity, the canonical IR JSON, and
        ``round_trip_stable`` recording whether re-serializing reproduced the
        same content hash.

    Raises:
        ToolError: If the payload is invalid or exceeds a configured bound.
    """
    ir = resolve_ir(ir_json, IR_FORMAT)
    canonical = ir_to_json(ir, indent=indent)
    return {
        "status": "success",
        "format": IR_FORMAT,
        "ir_json": canonical,
        "n_qubits": int(ir.n_wires),
        "n_instructions": len(ir.instructions),
        "ir_version": str(ir.version),
        "content_hash": str(ir.content_hash),
        "round_trip_stable": canonical == ir_json,
    }


def known_gate_names() -> list[str]:
    """Return every gate name and alias the installed FlagQuantum accepts.

    Taken from the SDK's operator manifest rather than inferred from its method
    table, so aliases (``cx``/``cnot``, ``h``/``hadamard``) are included and the
    list matches what a gate list may legally contain.

    Returns:
        Sorted names.
    """
    return sorted(known_opcodes())
