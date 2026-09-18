"""Run a circuit locally, and be honest about what that does and does not prove.

This is the one module in the server that executes anything. Everything it
executes is a statevector simulation inside the current process: no network, no
credentials, no provider, no hardware. The SDK reaches that path by default and
reports it in its own result — ``execution_path: local_statevector``,
``platform_provider: pytorch_cpu`` — and the tool restates those fields rather
than summarising them, so a caller can see which path produced a number instead
of being told one was taken.

The SDK's own claim fields travel with the result too. A local run comes back
with ``accuracy.metric == "not_measured"``: the SDK does not present a
simulation as evidence about hardware, and neither does this server. That is the
field this result actually carries — measured, it is ``{status, outputs,
execution}``, and there is no ``release_gate_allowed`` on it.
``release_gate_allowed`` belongs to ``plan_execution_tool``'s summary, which is
where a caller should look for it.

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
)
from flagquantum_mcp_server.preconditions import (
    SDK_FAILURE_BASES,
    plain,
    reject_circuit_observables,
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
    reject_circuit_observables(
        ir,
        remedy=(
            "Ask for the expectation directly instead — pass "
            'outputs=[{"kind": "expectation", "pauli": "ZZ"}] for a single '
            'term, or a \'terms\' list of {"pauli": ..., "coefficient": ...} '
            "objects to measure a weighted sum in one call, or drop the field "
            "from the circuit."
        ),
    )
    _reject_unbound_parameters(ir)

    requested = _resolve_request(outputs, ir, n_wires=int(ir.n_wires))
    _require_shots_for_sampling(requested, options)
    sdk = load_sdk()
    built = _build_outputs(requested, n_wires=int(ir.n_wires)) if requested else None

    try:
        result = sdk.run(ir, options=_build_options(options), outputs=built)
    except SDK_FAILURE_BASES as exc:
        raise ToolInputError(_with_guidance(str(exc))) from exc

    resolved = [_measurement(item) for item in result.measurements]
    _enforce_response_cap(resolved)
    return {
        "status": "success",
        "outputs": resolved,
        "execution": _execution_provenance(result),
    }


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
    """Render one SDK measurement result as JSON.

    ``coefficient`` appears only on rows the SDK put one on, which means the
    terms of a weighted expectation. It is carried through because it is half
    of what a caller needs to reconstruct an energy — the SDK reports one row
    per term and does not total them — and dropping it here would leave the
    other half unusable.
    """
    rendered = {
        "kind": str(item.kind),
        "wires": [int(wire) for wire in item.wires],
        "shots": None if item.shots is None else int(item.shots),
        "value": plain(item.value),
    }
    coefficient = dict(item.metadata or {}).get("fq_coefficient")
    if coefficient is not None:
        rendered["coefficient"] = float(coefficient)
    return rendered


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
        "accuracy": plain(result.accuracy),
    }
