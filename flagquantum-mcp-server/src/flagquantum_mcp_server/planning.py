"""Build an execution plan without running anything.

``fq.plan`` answers "what would this cost, on which device, in which mode, and
does it fit in memory" without executing a program. That makes it the right
planning tool for an agent that must decide before it commits, and it keeps
this server free of any execution path.

Noise models are deliberately not exposed: a ``NoiseModel`` is a live SDK
object rather than a serializable value, and every tool in this server takes
and returns JSON.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from flagquantum_mcp_server._bridge import load_sdk
from flagquantum_mcp_server.circuits import CircuitFormat, CircuitPayload, resolve_ir
from flagquantum_mcp_server.errors import ToolInputError, UnsupportedFormatError

OUTPUT_KINDS: tuple[str, ...] = ("counts", "expectation", "probabilities", "samples")
PAULI_LETTERS = frozenset("IXYZ")

OPTION_FIELDS: tuple[str, ...] = (
    "mode",
    "backend",
    "device",
    "target",
    "batch_size",
    "precision",
    "shots",
    "seed",
    "memory_limit_bytes",
    "require_gradients",
    "allow_approximate",
    "allow_backend_fallback",
)


def plan_execution(
    circuit: CircuitPayload,
    circuit_format: CircuitFormat = "ir",
    *,
    options: Mapping[str, Any] | None = None,
    outputs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Plan execution of a circuit and report the resolved execution contract.

    Args:
        circuit: Serialized circuit, as JSON text or a decoded payload.
        circuit_format: Either ``"ir"`` or ``"qir"``.
        options: Partial ``ExecutionOptions`` fields; unset fields stay at the
            SDK's defaults.
        outputs: Requested outputs, each a mapping with a ``kind`` of
            ``"counts"``, ``"expectation"``, ``"probabilities"`` or
            ``"samples"``, plus either ``wires`` or a ``pauli`` string.

    Returns:
        A payload with the plan's identity, its summary, and the full plan
        contract as JSON.

    Raises:
        ToolError: If the payload, an option, or an output request is invalid.
    """
    ir = resolve_ir(circuit, circuit_format)
    sdk = load_sdk()
    reject_outputs_with_declared_measurements(outputs, ir)
    resolved_outputs = _build_outputs(outputs, n_wires=int(ir.n_wires))
    _require_shots_for_sampling(outputs, options)
    try:
        plan = sdk.plan(
            ir,
            options=_build_options(options),
            outputs=resolved_outputs,
        )
    except ValueError as exc:
        raise ToolInputError(f"The SDK rejected this plan request: {exc}") from exc
    return {
        "status": "success",
        "plan": {
            "schema_version": str(plan.schema_version),
            "identity": str(plan.identity),
            "mode": str(plan.mode),
            "backend": str(plan.backend),
            "device": str(plan.device),
            "precision": str(plan.precision),
            "is_distributed": bool(plan.is_distributed),
            "program_fingerprint": str(plan.program_fingerprint),
        },
        "summary": _plain(plan.summary()),
        "plan_json": str(plan.to_json()),
    }


def reject_outputs_with_declared_measurements(outputs: Any, ir: Any) -> None:
    """Refuse an output request that collides with the circuit's own measurements.

    A circuit may declare its measurements in the envelope, and the SDK reads
    that field. It also refuses to accept output requests on top of it, in
    vocabulary that names neither ``measurements`` nor ``outputs``
    ("measurements cannot be supplied when the program already contains
    measurement requests"), so a caller is told a conflict exists without being
    told which two things are in conflict.

    Shared by the planning and simulation tools because it is a fact about the
    payload rather than about either one: both pass their outputs to the same
    SDK entry point, and both would otherwise relay the same puzzle.

    Args:
        outputs: The caller's output specifications, if any.
        ir: The validated ``CircuitIR``.

    Raises:
        ToolInputError: If both an output request and circuit-declared
            measurements are present.
    """
    if not outputs:
        return
    declared = tuple(getattr(ir, "measurements", ()) or ())
    if not declared:
        return
    raise ToolInputError(
        f"This circuit already declares {len(declared)} measurement(s), and "
        "outputs were passed as well. The SDK accepts one or the other, not "
        'both ("measurements cannot be supplied when the program already '
        'contains measurement requests"). Drop the "measurements" field from '
        "the circuit envelope, or call without the outputs argument and use "
        "what the circuit declares."
    )


def _require_shots_for_sampling(
    outputs: Sequence[Mapping[str, Any]] | None,
    options: Mapping[str, Any] | None,
) -> None:
    """Reject sampling outputs that were requested without a shot count.

    The SDK refuses ``counts`` and ``samples`` without ``shots``. Catching it
    here turns a generic SDK validation failure into a message that names the
    missing field.

    Args:
        outputs: The caller's output specifications.
        options: The caller's execution options.

    Raises:
        ToolInputError: If a sampling output was requested with no ``shots``.
    """
    needs_shots = sorted(
        {
            str(item.get("kind"))
            for item in outputs or ()
            if item.get("kind") in ("counts", "samples")
        }
    )
    if not needs_shots:
        return
    if (options or {}).get("shots"):
        return
    raise ToolInputError(
        f"Output kind(s) {needs_shots} require a shot count. "
        'Add {"shots": <positive int>} to options.'
    )


def _build_options(options: Mapping[str, Any] | None) -> Any:
    """Construct ``ExecutionOptions`` from a partial mapping."""
    sdk = load_sdk()
    if not options:
        return None
    unknown = sorted(set(options) - set(OPTION_FIELDS))
    if unknown:
        raise UnsupportedFormatError(
            f"Unknown execution option(s): {unknown}. Supported options are {list(OPTION_FIELDS)}."
        )
    try:
        return sdk.ExecutionOptions(**dict(options))
    except (ValueError, TypeError) as exc:
        raise ToolInputError(f"Invalid execution option: {exc}") from exc


def _build_outputs(
    outputs: Sequence[Mapping[str, Any]] | None, *, n_wires: int
) -> list[Any] | None:
    """Translate output specifications into SDK ``OutputRequest`` objects."""
    if not outputs:
        return None
    return [_build_output(item, n_wires=n_wires) for item in outputs]


def _build_output(spec: Mapping[str, Any], *, n_wires: int) -> Any:
    """Translate one output specification into an ``OutputRequest``."""
    sdk = load_sdk()
    kind = spec.get("kind")
    if kind not in OUTPUT_KINDS:
        raise UnsupportedFormatError(
            f"Output kind must be one of {list(OUTPUT_KINDS)}; received {kind!r}."
        )
    name = spec.get("name")
    pauli = spec.get("pauli")

    if kind == "expectation":
        if not isinstance(pauli, str):
            raise ToolInputError(
                "An 'expectation' output requires a 'pauli' string, for example "
                "'ZZI' to measure Z on wires 0 and 1."
            )
        return sdk.expectation(_pauli_observable(pauli, n_wires=n_wires), name=name)

    wires = _wires(spec.get("wires"), n_wires=n_wires)
    if kind == "counts":
        return sdk.counts(wires, name=name)
    if kind == "probabilities":
        return sdk.probabilities(wires, name=name)
    return sdk.samples(wires, name=name)


def _pauli_observable(pauli: str, *, n_wires: int) -> Any:
    """Build an ``Observable`` from a Pauli string such as ``"ZZI"``.

    Args:
        pauli: One letter per wire, drawn from ``I``, ``X``, ``Y``, ``Z``.
        n_wires: Circuit width the string must match.

    Returns:
        A ``flagquantum.Observable``.

    Raises:
        ToolInputError: If the string is malformed, is entirely identity, or
            does not match the circuit width.
    """
    upper = pauli.upper()
    if len(upper) != n_wires:
        raise ToolInputError(
            f"Pauli string {pauli!r} covers {len(upper)} wires, but the circuit has "
            f"{n_wires}. One letter per wire is required."
        )
    bad = sorted(set(upper) - PAULI_LETTERS)
    if bad:
        raise ToolInputError(
            f"Pauli string {pauli!r} contains unsupported letters {bad}; "
            f"use only {sorted(PAULI_LETTERS)}."
        )
    sdk = load_sdk()
    factors = [getattr(sdk, letter)(wire) for wire, letter in enumerate(upper) if letter != "I"]
    if not factors:
        raise ToolInputError(
            f"Pauli string {pauli!r} is entirely identity, which is not a measurable observable."
        )
    observable = factors[0]
    for factor in factors[1:]:
        observable = observable @ factor
    return observable


def _wires(value: Any, *, n_wires: int) -> tuple[int, ...]:
    """Validate an optional wire list.

    Args:
        value: Caller-supplied wires, or ``None`` for every wire.
        n_wires: Circuit width.

    Returns:
        A tuple of wire indices.

    Raises:
        ToolInputError: If a wire is out of range or not an integer.
    """
    if value is None:
        return tuple(range(n_wires))
    if isinstance(value, int):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ToolInputError(f"wires must be a list of integers; received {value!r}.")
    resolved: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ToolInputError(f"wires must be integers; received {item!r}.")
        if not 0 <= item < n_wires:
            raise ToolInputError(f"Wire {item} is outside 0..{n_wires - 1} for this circuit.")
        resolved.append(item)
    return tuple(resolved)


def _plain(value: Any) -> Any:
    """Convert SDK containers into JSON-native types."""
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
