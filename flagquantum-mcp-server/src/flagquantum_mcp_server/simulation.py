"""Run a circuit locally, and be honest about what that does and does not prove.

This is the one module in the server that executes anything. Everything it
executes is a statevector simulation inside the current process: no network, no
credentials, no provider, no hardware. The SDK reaches that path by default and
reports it in its own result — ``execution_path: local_statevector``,
``platform_provider: pytorch_cpu`` — and the tool restates those fields rather
than summarising them, so a caller can see which path produced a number instead
of being told one was taken.

The SDK's own claim fields travel with the result too. A local run comes back
with ``accuracy.metric == "not_measured"`` and ``release_gate_allowed: False``:
the SDK does not present a simulation as evidence about hardware, and neither
does this server.

Four things are refused rather than reinterpreted, and they are the reason this
module is more than a call to ``fq.run``:

* an unbound parameter, which the SDK reports as ``planned execution failed``;
* a circuit carrying ``observables``, which the default statevector path never
  reads;
* an output request alongside circuit-declared measurements, which the SDK
  refuses in vocabulary that names neither field;
* an output too large to return, which is a property of the request.

Each one is a case where accepting would have produced a plausible-looking
answer.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

from flagquantum_mcp_server import limits
from flagquantum_mcp_server._bridge import load_sdk
from flagquantum_mcp_server.circuits import CircuitFormat, CircuitPayload, resolve_ir
from flagquantum_mcp_server.errors import ToolInputError, ToolLimitError
from flagquantum_mcp_server.planning import (
    _build_options,
    _build_outputs,
    _require_shots_for_sampling,
    reject_outputs_with_declared_measurements,
)

OUTPUT_KINDS: tuple[str, ...] = ("counts", "expectation", "probabilities", "samples")


def simulate_execution(
    circuit: CircuitPayload,
    circuit_format: CircuitFormat = "ir",
    *,
    options: Mapping[str, Any] | None = None,
    outputs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute a circuit locally and return the outputs it was asked for.

    Args:
        circuit: Serialized circuit, as JSON text or a decoded payload.
        circuit_format: Either ``"ir"`` or ``"qir"``.
        options: Partial ``ExecutionOptions`` fields, including ``shots`` and
            ``seed``.
        outputs: Requested outputs, each a mapping with a ``kind`` of
            ``"counts"``, ``"expectation"``, ``"probabilities"`` or
            ``"samples"``, plus either ``wires`` or a ``pauli`` string. When
            omitted, the circuit's own ``measurements`` decide; when it declares
            none either, probabilities over every wire are returned.

    Returns:
        A payload carrying the resolved outputs and the execution provenance
        the SDK reported.

    Raises:
        ToolError: If the payload, an option, an output request, or the
            interaction between an output request and the circuit is invalid.
    """
    ir = resolve_ir(circuit, circuit_format)
    _reject_circuit_observables(ir)
    _reject_unbound_parameters(ir)

    requested = _resolve_request(outputs, ir, n_wires=int(ir.n_wires))
    _require_shots_for_sampling(requested, options)
    sdk = load_sdk()
    built = _build_outputs(requested, n_wires=int(ir.n_wires)) if requested else None

    try:
        result = sdk.run(ir, options=_build_options(options), outputs=built)
    except (ValueError, RuntimeError, NotImplementedError) as exc:
        # The three SDK exception bases reachable from here: ValidationError is
        # a ValueError, ExecutionError a RuntimeError, CapabilityError a
        # NotImplementedError. The try block holds one SDK call and nothing
        # else, and the caller mistakes this module knows about are refused
        # above, so what arrives here is the SDK describing the request.
        raise ToolInputError(_with_guidance(str(exc))) from exc

    resolved = [_measurement(item) for item in result.measurements]
    _enforce_response_cap(resolved)
    return {
        "status": "success",
        "outputs": resolved,
        "execution": _execution_provenance(result),
    }


def _reject_circuit_observables(ir: Any) -> None:
    """Refuse a circuit that carries observables, and name the spelling that works.

    ``CircuitIR.observables`` is a real field: it serializes, it is covered by
    ``content_hash``, it is validated on the way in, and a coefficient may be a
    symbol, so it even shows up in the parameter list. The default statevector
    path does not read it. Only the hybrid operator path does, and that path
    demands a sum of unit single-wire Z terms and says so.

    An agent that followed this server's own ``flagquantum://ir-schema``
    resource would put its Hamiltonian exactly here and get a result with no
    expectation value in it, and nothing anywhere would have said so. Rather
    than reimplement the SDK's private lowering — the observable's name is
    lowercased but never checked against I/X/Y/Z, so it can legally read
    ``"heisenberg"`` — the field is refused and the working output spelling is
    given.

    Args:
        ir: The validated ``CircuitIR``.

    Raises:
        ToolInputError: If the circuit carries any observable.
    """
    observables = tuple(getattr(ir, "observables", ()) or ())
    if not observables:
        return
    raise ToolInputError(
        f"This circuit carries {len(observables)} observable(s), and a local "
        "simulation does not evaluate them: the default statevector path never "
        "reads the field, so the result would carry no expectation value and no "
        "error either. Ask for the expectation directly instead — pass "
        'outputs=[{"kind": "expectation", "pauli": "ZZ"}] with one letter per '
        "wire, or omit the observables from the circuit."
    )


MARGINAL_LIMIT_MARKER = "max_marginal_wires"

MARGINAL_GUIDANCE = (
    " This server does not expose that setting: raising it means accepting the "
    "contraction cost it names. Narrow the request instead — name a few wires "
    '({"kind": "probabilities", "wires": [0, 1]}), or ask for "counts" or '
    '"expectation", neither of which computes a marginal distribution.'
)


def _with_guidance(message: str) -> str:
    """Add what a caller can do about an SDK refusal, without restating it.

    The SDK owns its own limits and says so in its own words, which this
    function leaves intact. What it cannot know is which of those limits this
    server chose not to expose, so a message pointing at a setting the caller
    has no way to reach gets the alternative appended. The check is on a stable
    fragment of the SDK's own remedy text: if that wording changes the guidance
    is simply not added, which degrades to the SDK's message rather than to a
    wrong one.

    Args:
        message: The SDK's exception text.

    Returns:
        The message, with guidance appended when it applies.
    """
    if MARGINAL_LIMIT_MARKER in message:
        return f"{message}.{MARGINAL_GUIDANCE}"
    return message


def _reject_unbound_parameters(ir: Any) -> None:
    """Refuse a circuit with unbound symbols, naming them.

    Without this the refusal is the SDK's ``planned execution failed``, which
    names neither the circuit's problem nor the tool that fixes it. Reading the
    names goes through the same public path ``inspect_parameters_tool`` uses
    rather than inspecting the IR's value encoding here.

    Args:
        ir: The validated ``CircuitIR``.

    Raises:
        ToolInputError: If the circuit is still parameterized.
    """
    from flagquantum_mcp_server.circuits import circuit_from_ir

    names = tuple(str(name) for name in circuit_from_ir(ir).parameter_names)
    if not names:
        return
    raise ToolInputError(
        f"This circuit still has unbound parameter(s): {list(names)}. A "
        "simulation needs numbers, so bind them first with "
        "bind_parameters_tool and run the result it returns. (The SDK's own "
        'refusal for this case is "planned execution failed", which is why '
        "the check is here.)"
    )


def _resolve_request(
    outputs: Sequence[Mapping[str, Any]] | None,
    ir: Any,
    *,
    n_wires: int,
) -> Sequence[Mapping[str, Any]] | None:
    """Decide which output request governs the run.

    Three cases, and the middle one is the point: when the circuit declares its
    own measurements, they are what runs, with the shot count the circuit
    carries. That is the SDK's own reading of the field, and inventing a default
    on top of it would answer a question the circuit had already answered.

    Args:
        outputs: The caller's output specifications, if any.
        ir: The validated ``CircuitIR``.
        n_wires: Circuit width, for the default.

    Returns:
        The caller's outputs, ``None`` to let the circuit's own measurements
        govern, or a default probability request.

    Raises:
        ToolInputError: If an output request and circuit-declared measurements
            are supplied together.
    """
    if outputs:
        reject_outputs_with_declared_measurements(outputs, ir)
        return outputs
    if tuple(getattr(ir, "measurements", ()) or ()):
        return None
    return [{"kind": "probabilities", "wires": list(range(n_wires))}]


def _measurement(item: Any) -> dict[str, Any]:
    """Render one SDK measurement result as JSON."""
    return {
        "kind": str(item.kind),
        "wires": [int(wire) for wire in item.wires],
        "shots": None if item.shots is None else int(item.shots),
        "value": _plain(item.value),
    }


def _enforce_response_cap(resolved: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a result too large to carry, naming the size and the way out.

    Checked after the run rather than before it because only the SDK knows how
    many outcomes a sampled result actually produced. The cost of running first
    is bounded by ``max_qubits`` and is small next to sending a caller a result
    they cannot use.

    Args:
        resolved: The rendered outputs.

    Raises:
        ToolLimitError: If the outputs carry more values than the bound allows.
    """
    total = sum(_value_count(item) for item in resolved)
    bound = limits.max_response_values()
    if total <= bound:
        return
    raise ToolLimitError(
        f"This result carries {total} values, past the {bound} one response may "
        "return. Narrow it: name the wires you need "
        '({"kind": "probabilities", "wires": [0, 1]}), or ask for "counts" or '
        '"expectation", which report only what occurred or one number per term. '
        "FLAGQUANTUM_MCP_MAX_RESPONSE_VALUES raises the bound for a deployment "
        "that wants the whole distribution."
    )


def _value_count(item: Mapping[str, Any]) -> int:
    """Count the scalar values one output would put on the wire."""
    value = item["value"]
    if item["kind"] == "counts":
        return sum(len(entry) for entry in value)
    if item["kind"] == "samples":
        return sum(len(draw) for draw in value)
    return _scalars(value)


def _scalars(value: Any) -> int:
    """Count scalars in a nested list."""
    if isinstance(value, list):
        return sum(_scalars(item) for item in value) or 1
    return 1


def _execution_provenance(result: Any) -> dict[str, Any]:
    """Restate where the SDK ran this, and what it does not claim about it."""
    return {
        "execution_path": str(result.runtime.get("execution_path", "")),
        "platform_provider": str(result.runtime.get("platform_provider", "")),
        "simulation_engine": str(result.runtime.get("simulation_engine", "")),
        "mode": str(result.runtime.get("mode", "")),
        "device": str(result.runtime.get("device", "")),
        "accuracy": _plain(result.accuracy),
    }


def _plain(value: Any) -> Any:
    """Convert SDK containers and dataclasses into JSON-native types.

    Tensors arrive with a leading batch axis because the SDK carries a batch
    dimension through every result. The axis is dropped when it is exactly one,
    which loses nothing: a single-batch result is the same data without it.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _plain(
            {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
        )
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _plain(_without_batch_axis(value.tolist()))
    return str(value)


def _without_batch_axis(converted: Any) -> Any:
    """Drop a leading axis of length one, which carries no information."""
    if isinstance(converted, list) and len(converted) == 1 and isinstance(converted[0], list):
        return converted[0]
    return converted
