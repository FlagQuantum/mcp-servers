"""Execution planning, and the output requests planning and execution share.

``fq.plan`` answers "what would this cost, on which device, in which mode, and
does it fit in memory" without executing a program, which is what an agent needs
before it commits.

This module also owns the translation from a caller's JSON output request into
the SDK's own ``OutputRequest``, because planning and execution must read that
JSON the same way. ``simulation`` calls the same builders rather than keeping a
second opinion about what ``{"kind": "counts", "wires": [0]}`` means.

Noise models are deliberately not exposed: a ``NoiseModel`` is a live SDK
object rather than a serializable value, and every tool in this server takes
and returns JSON.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from flagquantum_mcp_server import limits
from flagquantum_mcp_server._bridge import load_sdk
from flagquantum_mcp_server.circuits import CircuitFormat, CircuitPayload, resolve_ir
from flagquantum_mcp_server.errors import (
    ToolInputError,
    ToolLimitError,
    UnsupportedFormatError,
)
from flagquantum_mcp_server.preconditions import (
    reject_outputs_with_declared_measurements,
)

OUTPUT_KINDS: tuple[str, ...] = ("counts", "expectation", "probabilities", "samples")
PAULI_LETTERS = frozenset("IXYZ")

# Keys an expectation request accepts, and nothing else. An unknown key is a
# caller who meant something the request cannot express, which is worth saying
# rather than ignoring.
EXPECTATION_KEYS = frozenset({"kind", "pauli", "terms", "name"})
TERM_KEYS = frozenset({"pauli", "coefficient"})

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
    terms = spec.get("terms")

    if kind == "expectation":
        unknown = sorted(set(spec) - EXPECTATION_KEYS)
        if unknown:
            raise UnsupportedFormatError(
                f"An 'expectation' output does not take {unknown}. It takes "
                "'pauli' for one term, or 'terms' for several, plus an optional "
                "'name'."
            )
        if pauli is not None and terms is not None:
            raise ToolInputError(
                "An 'expectation' output takes either 'pauli' (one term) or "
                "'terms' (several), not both. A Hamiltonian is a 'terms' list: "
                '{"kind": "expectation", "terms": [{"pauli": "ZZ", '
                '"coefficient": 1.0}, {"pauli": "XI", "coefficient": -0.5}]}.'
            )
        if terms is not None:
            observable = _hamiltonian_observable(terms, n_wires=n_wires)
        elif isinstance(pauli, str):
            observable = _pauli_observable(pauli, n_wires=n_wires)
        else:
            raise ToolInputError(
                "An 'expectation' output requires a 'pauli' string such as "
                "'ZZI' to measure Z on wires 0 and 1, or a 'terms' list for a "
                "weighted sum."
            )
        return sdk.expectation(observable, name=name)

    wires = _wires(spec.get("wires"), n_wires=n_wires)
    if kind == "counts":
        return sdk.counts(wires, name=name)
    if kind == "probabilities":
        return sdk.probabilities(wires, name=name)
    return sdk.samples(wires, name=name)


def validate_pauli_terms(terms: Any, *, n_wires: int) -> list[tuple[str, float]]:
    """Validate a ``terms`` list and return its Pauli strings and coefficients.

    The vocabulary ``{"pauli": "ZZ", "coefficient": 1.0}`` is one concept, so it
    has one validator: the expectation builder in this module and the training
    objective in ``training.py`` both call this rather than each deciding for
    themselves what a term may say. One concept with two validators drifts into
    two concepts.

    Args:
        terms: A non-empty list of ``{"pauli": ..., "coefficient": ...}``
            mappings. ``coefficient`` defaults to 1.0.
        n_wires: Circuit width every term's Pauli string must match.

    Returns:
        One ``(pauli, coefficient)`` pair per term, in the caller's order, with
        the Pauli string upper-cased.

    Raises:
        ToolInputError: If the list, a term, a Pauli string, or a coefficient is
            unusable.
        ToolLimitError: If the list carries more terms than the bound allows.
    """
    if not isinstance(terms, Sequence) or isinstance(terms, (str, bytes, Mapping)):
        raise ToolInputError(
            f"'terms' is {type(terms).__name__}; it must be a list of objects "
            'such as [{"pauli": "ZZ", "coefficient": 1.0}].'
        )
    if not terms:
        raise ToolInputError(
            "'terms' is empty. An expectation needs at least one term; for a "
            "single unweighted term pass 'pauli' instead."
        )
    bound = limits.max_hamiltonian_terms()
    if len(terms) > bound:
        raise ToolLimitError(
            f"This expectation carries {len(terms)} terms, past the {bound} one "
            "request may ask for. FLAGQUANTUM_MCP_MAX_HAMILTONIAN_TERMS raises "
            "the bound for a deployment that needs a larger operator."
        )
    return [_weighted_term(term, position, n_wires=n_wires) for position, term in enumerate(terms)]


def _weighted_term(term: Any, position: int, *, n_wires: int) -> tuple[str, float]:
    """Validate one weighted Pauli term, naming its position in every refusal.

    Returns:
        The upper-cased Pauli string and its coefficient.
    """
    where = f"terms[{position}]"
    if not isinstance(term, Mapping):
        raise ToolInputError(
            f"{where} is {type(term).__name__}; every term must be an object "
            'such as {"pauli": "ZZ", "coefficient": 1.0}.'
        )
    unknown = sorted(set(term) - TERM_KEYS)
    if unknown:
        raise UnsupportedFormatError(
            f"{where} does not take {unknown}. A term is "
            '{"pauli": "ZZ", "coefficient": 1.0}, with the coefficient '
            "optional."
        )
    pauli = term.get("pauli")
    if not isinstance(pauli, str):
        raise ToolInputError(
            f"{where} has no 'pauli' string; every term needs one, such as "
            '"ZZ" to measure Z on wires 0 and 1.'
        )
    coefficient = term.get("coefficient", 1.0)
    if isinstance(coefficient, bool) or not isinstance(coefficient, (int, float)):
        raise ToolInputError(
            f"{where} has a coefficient that is {_name_of(coefficient)}; it "
            "must be a real number. A symbol is not accepted here — the SDK "
            "builds a parameter expression from one rather than an observable, "
            "so it cannot be evaluated."
        )
    _check_pauli_string(pauli, n_wires=n_wires, where=where)
    return pauli.upper(), float(coefficient)


def _hamiltonian_observable(terms: Any, *, n_wires: int) -> Any:
    """Build a weighted sum of Pauli strings from a ``terms`` list.

    The SDK already evaluates a multi-term observable, returning one row per
    term with its coefficient in the row's metadata, so this builder is the
    whole of what was missing: a way to say "these terms" through JSON. It
    deliberately does not add the rows up — that sum is the caller's, from
    numbers the SDK produced, and inventing it here would be this server
    reporting a value the SDK never computed.

    Only ``+`` and scalar ``*`` on the SDK's public ``Z``/``X``/``Y`` builders
    are used. The term type underneath is private, and naming it here would
    make a private class part of this server's contract; the operators build it
    without ever spelling it.

    Args:
        terms: A non-empty list of ``{"pauli": ..., "coefficient": ...}``
            mappings.
        n_wires: Circuit width every term's Pauli string must match.

    Returns:
        A ``flagquantum.Observable``.

    Raises:
        ToolInputError: If the list, a term, a Pauli string, or a coefficient is
            unusable.
        ToolLimitError: If the list carries more terms than the bound allows.
    """
    total: Any = None
    for pauli, coefficient in validate_pauli_terms(terms, n_wires=n_wires):
        observable = _pauli_observable(pauli, n_wires=n_wires) * coefficient
        total = observable if total is None else total + observable
    return total


def _name_of(value: Any) -> str:
    """Name a rejected value the way the caller wrote it."""
    if value is None:
        return "null"
    if isinstance(value, str):
        return f"a string ({value!r})"
    if isinstance(value, bool):
        return f"a boolean ({value!r})"
    if isinstance(value, Mapping):
        return f"an object ({dict(value)!r})"
    if isinstance(value, (list, tuple)):
        return f"a list ({list(value)!r})"
    return f"the unsupported value {value!r}"


def _check_pauli_string(pauli: str, *, n_wires: int, where: str | None = None) -> None:
    """Check a Pauli string's length, alphabet and content.

    Args:
        pauli: One letter per wire, drawn from ``I``, ``X``, ``Y``, ``Z``.
        n_wires: Circuit width the string must match.
        where: How to name the caller's location in a refusal, when this string
            is one term of several rather than the whole request.

    Raises:
        ToolInputError: If the string is malformed, is entirely identity, or
            does not match the circuit width.
    """
    subject = "Pauli string" if where is None else f"{where} 'pauli'"
    upper = pauli.upper()
    if len(upper) != n_wires:
        raise ToolInputError(
            f"{subject} {pauli!r} covers {len(upper)} wires, but the circuit has "
            f"{n_wires}. One letter per wire is required."
        )
    bad = sorted(set(upper) - PAULI_LETTERS)
    if bad:
        raise ToolInputError(
            f"{subject} {pauli!r} contains unsupported letters {bad}; "
            f"use only {sorted(PAULI_LETTERS)}."
        )
    if not any(letter != "I" for letter in upper):
        raise ToolInputError(
            f"{subject} {pauli!r} is entirely identity, which is not a measurable "
            "observable. Drop the term, or name a wire it acts on."
        )


def _pauli_observable(pauli: str, *, n_wires: int, where: str | None = None) -> Any:
    """Build an ``Observable`` from a Pauli string such as ``"ZZI"``.

    Args:
        pauli: One letter per wire, drawn from ``I``, ``X``, ``Y``, ``Z``.
        n_wires: Circuit width the string must match.
        where: How to name the caller's location in a refusal, when this string
            is one term of several rather than the whole request.

    Returns:
        A ``flagquantum.Observable``.

    Raises:
        ToolInputError: If the string is malformed, is entirely identity, or
            does not match the circuit width.
    """
    subject = "Pauli string" if where is None else f"{where} 'pauli'"
    upper = pauli.upper()
    if len(upper) != n_wires:
        raise ToolInputError(
            f"{subject} {pauli!r} covers {len(upper)} wires, but the circuit has "
            f"{n_wires}. One letter per wire is required."
        )
    bad = sorted(set(upper) - PAULI_LETTERS)
    if bad:
        raise ToolInputError(
            f"{subject} {pauli!r} contains unsupported letters {bad}; "
            f"use only {sorted(PAULI_LETTERS)}."
        )
    sdk = load_sdk()
    factors = [getattr(sdk, letter)(wire) for wire, letter in enumerate(upper) if letter != "I"]
    if not factors:
        raise ToolInputError(
            f"{subject} {pauli!r} is entirely identity, which is not a measurable "
            "observable. Drop the term, or name a wire it acts on."
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
