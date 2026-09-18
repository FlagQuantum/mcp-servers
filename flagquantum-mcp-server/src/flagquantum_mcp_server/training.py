"""Train a circuit's parameters against a Pauli-sum energy, in one call.

``fq.Module`` and ``fq.train`` want a Python callable rather than JSON, which
looks like a dead end for a server whose every input is JSON. It is not: a
serialized circuit carries every instruction it was built from, and
``Circuit.gate`` is the single primitive that rebuilds them — built-in gates and
arbitrary matrices both go through it. So the replay below is a loop and not a
gate table, and it carries no numerical logic. It translates a serialized circuit
into the callable shape the SDK asks for, and nothing else.

A channel is the one instruction it does not rebuild faithfully. Flagging an
instruction as a channel is ``instruction.metadata["is_channel"]``, and
``Circuit.gate`` takes no ``metadata``, so the flag cannot be carried. Measured,
the consequence is smaller than it sounds: a channel circuit fails ``fq.run`` on
this path both before and after the replay, so no number a caller sees changes —
but the replay is not faithful there and this docstring will not claim it is.

Four facts about the SDK shape this module, all measured rather than read:

* ``Module(hamiltonian=H)`` stores ``H`` and does not evaluate it. Which
  observable a module measures comes from its ``RuntimePolicy``, whose default is
  ``⟨Z₀⟩``. Every ``Module`` built here passes
  ``policy=RuntimePolicy(observable="hamiltonian")``, and
  ``test_the_hamiltonian_reaches_the_objective`` is what holds that.
* A ``{"$parameter": ...}`` marker is a *serialization* encoding. Handed to the
  Python API it is stored as an opaque dict and the circuit reports itself as
  unparameterized, which is the silent failure ``circuits.py`` already refuses at
  the JSON boundary. Names are therefore read through ``circuit_from_ir``, which
  decodes the markers properly. The same trap covers the instruction list itself:
  see ``replay_builder``, which reads the IR's live objects and never
  ``to_dict()``.
* ``Circuit.gate`` takes its arguments as a ``params=`` mapping, not
  positionally, and the argument names are the SDK's own gate manifest.
* A parameter can also sit inside a ``ParameterExpression`` — ``2.0 * t0``.
  ``parameter_names`` reports the name either way, so a replay that substitutes
  only a top-level ``Parameter`` advertises a trainable group it never applies,
  and stays byte-equal to its source while doing it.

Training is the only operation here whose cost is not bounded by the size of its
input, so a prediction decides before any work starts. See ``predict_seconds``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from flagquantum_mcp_server import limits
from flagquantum_mcp_server._bridge import load_module, load_sdk, load_torch
from flagquantum_mcp_server.circuits import (
    CircuitFormat,
    CircuitPayload,
    circuit_from_ir,
    resolve_ir,
)
from flagquantum_mcp_server.errors import ToolInputError, ToolLimitError
from flagquantum_mcp_server.planning import validate_pauli_terms
from flagquantum_mcp_server.preconditions import (
    SDK_FAILURE_BASES,
    plain,
    reject_circuit_observables,
)


def parameter_names(ir: Any) -> tuple[str, ...]:
    """Return the circuit's parameter names, in the SDK's order.

    Read through ``circuit_from_ir`` rather than parsed from the instruction
    list, because that is the function that knows a ``$parameter`` marker is a
    symbol. The order is the SDK's — sorted, each name once however many gates
    carry it — and nothing here depends on it, because ``Module`` binds by name.

    Args:
        ir: A validated ``CircuitIR``.

    Returns:
        The parameter names.
    """
    return tuple(str(name) for name in circuit_from_ir(ir).parameter_names)


def replay_builder(ir: Any) -> Callable[[Mapping[str, Any]], Any]:
    """Return the callable ``fq.Module`` traces, rebuilding this circuit.

    The returned builder takes the parameter mapping ``Module`` hands it — one
    one-element tensor per name — and returns a live ``Circuit``.

    Instructions come from the IR's own ``instructions``, not from
    ``to_dict()``. The two differ in a way that is easy to miss and was
    measured: ``to_dict()`` runs every value through the SDK's encoder, turning
    a parameter into a ``{"$parameter": ...}`` marker, a complex into
    ``{"$complex": [...]}`` and a matrix's entries the same way. Handing those
    encodings back to ``Circuit.gate`` builds a circuit that serializes
    byte-for-byte like the original and cannot execute — ``fq.run`` refuses it
    with ``planned execution failed`` while the source runs normally. The live
    objects are ``Parameter``, ``ParameterExpression`` and real matrices, which
    is what the builder needs and what a JSON tool has no reason to reassemble
    from markers.

    Args:
        ir: A validated ``CircuitIR``.

    Returns:
        A callable taking ``{name: tensor}`` and returning a ``Circuit``.
    """
    instructions = tuple(ir.instructions)
    n_wires = int(ir.n_wires)

    def build(parameters: Mapping[str, Any]) -> Any:
        sdk = load_sdk()
        circuit = sdk.Circuit(n_wires)
        for instruction in instructions:
            arguments = {
                str(key): _argument(value, parameters, sdk)
                for key, value in instruction.params.items()
            }
            circuit.gate(
                str(instruction.name),
                [int(wire) for wire in instruction.wires],
                params=arguments or None,
                matrix=instruction.matrix,
            )
        return circuit

    return build


def hamiltonian_from_terms(terms: Any, *, n_wires: int) -> Any:
    """Build the SDK's ``Hamiltonian`` from a ``terms`` list.

    The same JSON shape ``outputs`` accepts, validated by the same function, so
    that "a weighted sum of Pauli strings" has one spelling and one set of
    refusals across the server. What differs is only what is built: an
    ``Observable`` for a measurement, a ``Hamiltonian`` for an objective.

    Reached through ``flagquantum.algorithms``, which declares both names in its
    ``__all__`` and is not in any capability's ``public_apis``. That is a tier-3
    dependency: admitted because there is no other way to say what to minimise,
    and pinned by ``tests/test_api_contract.py``.

    Args:
        terms: A non-empty list of ``{"pauli": ..., "coefficient": ...}``
            mappings, one letter per wire.
        n_wires: Circuit width every term's Pauli string must match.

    Returns:
        A ``flagquantum.algorithms.Hamiltonian``.

    Raises:
        ToolInputError: If the list, a term, a Pauli string, or a coefficient is
            unusable.
        ToolLimitError: If the list carries more terms than the bound allows.
    """
    algorithms = load_algorithms()
    return algorithms.Hamiltonian(
        [
            algorithms.pauli_term(
                coefficient,
                {wire: letter for wire, letter in enumerate(pauli) if letter != "I"},
            )
            for pauli, coefficient in validate_pauli_terms(terms, n_wires=n_wires)
        ]
    )


def load_algorithms() -> Any:
    """Return ``flagquantum.algorithms``, where the objective's types live.

    A module rather than one attribute, because two names come from it:
    ``Hamiltonian`` and ``pauli_term``. Importing it once and reading both
    through it is clearer than two ``load_attribute`` calls that each re-import
    the same module, and it makes the dependency's shape visible at the call
    site.

    Returns:
        The ``flagquantum.algorithms`` module.

    Raises:
        FlagQuantumUnavailableError: If the module is not importable.
    """
    return load_module("flagquantum.algorithms")


# The budget model's constants. Calibrated from warm per-step measurements at 4,
# 8, 12, 13, 14, 15, 16, 20, 22 and 24 qubits, over circuits from 1 to 188
# instructions; the table is in the design document and every point is pinned by
# a test. The model over-predicts at all sixteen points, by 1.9x at the worst.
#
# Three costs, each of them one that was measured rather than assumed:
STARTUP_SECONDS = 0.5
MIN_STEP_SECONDS = 0.02  # dispatch, for circuits too small for either term below
STATE_COST = 1.5e-6  # per step per amplitude: holding the state
GATE_COST = 1e-7  # per step per amplitude per instruction: applying gates


def predict_seconds(ir: Any, steps: int) -> float:
    """Predict how long ``steps`` updates will take, in seconds.

    Three terms, because cost has three regimes. ``STARTUP_SECONDS`` is the
    one-time ``make_fx`` trace ``Module`` performs when it first compiles the
    builder. The other two are paid on every step: ``STATE_COST`` per amplitude
    for holding a state of ``2 ** n_wires``, and ``GATE_COST`` per amplitude per
    instruction for applying gates against it. They are separate terms because
    they were measured to be: at twenty-four wires a single gate costs 12.5 s per
    step, which is 7.5x what one instruction explains, and a single
    width-independent floor could not see that cost at all.

    The floor sits under both. It is generous on purpose: 20 ms where 1.8 ms was
    measured at four qubits, which at the default 60-second budget is the
    difference between a caller being allowed three thousand steps and thirty
    thousand. Three thousand is already more than anyone reads, and a model that
    is an upper bound everywhere is worth more than one that is tight at the
    bottom.

    A single coefficient cannot follow the true curve either, which falls from
    1.0e-5 per instruction-state at four qubits to 2.1e-8 at twenty and rises
    again to 5.7e-8 at twenty-four as the state stops fitting where it used to.
    The first two are measured points in the sixteen-row calibration table; the
    third is the design document's own 24-wire, 71-instruction measurement,
    68.5 s per step, and is not one of the sixteen. ``GATE_COST`` clears the
    highest point rather than the average one.

    Where that leaves the margin, over the sixteen points it was calibrated at:
    11x at four qubits, 4.9x at eight and 2.2x at twelve, the three widths where
    the floor is doing all the work; between 1.9x and 8.4x at the other thirteen,
    thinnest at twenty-four wires with four instructions and thickest at sixteen
    wires with four layers. The thinnest matters least, because the budget admits
    almost nothing there: the state term alone is 25.2 s per step at twenty-four
    wires, so no circuit at that width gets more than two steps, and the full
    ansatz there, 119.1 s per step, gets none at all. Twenty-two wires costs
    6.3 s per step, and a one-gate circuit at that width gets nine.

    Args:
        ir: A validated ``CircuitIR``.
        steps: The number of updates requested.

    Returns:
        A prediction in seconds. An estimate rather than a measurement of the
        caller's machine.
    """
    per_step = max(
        MIN_STEP_SECONDS,
        2.0 ** int(ir.n_wires) * STATE_COST,  # holding the state
        len(ir.instructions) * 2.0 ** int(ir.n_wires) * GATE_COST,  # applying gates
    )
    return STARTUP_SECONDS + steps * per_step


def train_parameters(
    circuit: CircuitPayload,
    hamiltonian: Sequence[Mapping[str, Any]] | None,
    circuit_format: CircuitFormat = "ir",
    *,
    values: Mapping[str, float] | None = None,
    steps: int = 50,
    learning_rate: float = 0.1,
) -> dict[str, Any]:
    """Train a circuit's parameters against a Pauli-sum energy.

    Args:
        circuit: Serialized circuit, as JSON text or a decoded payload.
        hamiltonian: The objective, as a list of
            ``{"pauli": ..., "coefficient": ...}`` mappings — the same shape an
            ``expectation`` output takes. One letter per wire.
        circuit_format: Either ``"ir"`` or ``"qir"``.
        values: Starting value for each parameter, by name. Defaults to zeros.
        steps: Number of optimizer updates.
        learning_rate: Adam's learning rate.

    Returns:
        A payload with the loss trajectory, the parameters the run ended on, the
        parameters it started from, and the execution provenance.

    Raises:
        ToolInputError: If the payload, the objective, the values or the
            optimizer settings are unusable.
        ToolLimitError: If the predicted cost is past the configured budget.
    """
    ir = resolve_ir(circuit, circuit_format)
    reject_circuit_observables(
        ir,
        remedy=(
            "Pass the Hamiltonian as this tool's 'hamiltonian' argument "
            "instead, which is where the objective belongs."
        ),
    )
    _check_steps(steps)
    _check_learning_rate(learning_rate)
    _check_budget(ir, steps)

    names = parameter_names(ir)
    _check_names(names)
    starting = _resolve_values(names, values)
    objective = hamiltonian_from_terms(hamiltonian, n_wires=int(ir.n_wires))

    sdk = load_sdk()
    torch = load_torch()
    # One element per name, and the replay's `parameters[name][0]` depends on
    # it: a group of any other size would be a name bound to a vector, and a
    # zero-element group raises IndexError inside the builder rather than here.
    module = sdk.Module(
        replay_builder(ir),
        parameters=dict.fromkeys(names, 1),
        init=dict(starting),
        hamiltonian=objective,
        policy=sdk.RuntimePolicy(observable="hamiltonian"),
    )
    optimizer = torch.optim.Adam(module.parameters(), lr=float(learning_rate))

    try:
        result = sdk.train(
            module,
            optimizer=optimizer,
            objective=lambda value: value.mean(),
            steps=steps,
        )
        final_loss, execution = _final_loss(module, torch)
    except SDK_FAILURE_BASES as exc:
        raise ToolInputError(
            f"The SDK could not train this circuit: {exc}. The most common "
            "cause is a circuit whose builder cannot be traced — Module "
            "requires static topology, so a gate list or an IR payload built "
            "from one always satisfies it."
        ) from exc

    return {
        "status": "success",
        "initial_loss": float(result.losses[0]),
        "final_loss": final_loss,
        "losses": [float(loss) for loss in result.losses],
        "completed_steps": int(result.completed_steps),
        "parameters": _read_back(module, names),
        "initial_parameters": dict(starting),
        "circuit": {
            "n_qubits": int(ir.n_wires),
            "n_instructions": len(ir.instructions),
            "content_hash": str(ir.content_hash),
        },
        "execution": execution,
    }


def _is_finite_number(value: int | float) -> bool:
    """Is this a finite real, without raising on one too large to be a float?

    ``math.isfinite`` raises ``OverflowError`` on an ``int`` bigger than a float
    can hold, which a JSON integer literal can be — ``json.loads("1" + "0"*400)``
    is a perfectly ordinary Python int. Measured: without this, such a value
    escapes the guard as an ``OverflowError`` and reaches the client as an
    internal error instead of a refusal naming the argument. Pinned by the
    ``10**400`` case in `test_a_learning_rate_adam_cannot_use_is_refused`.

    Args:
        value: A number already known to be an ``int`` or a ``float``.

    Returns:
        True if it is finite.
    """
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _check_steps(steps: Any) -> None:
    """Reject a step count the loop cannot use.

    The SDK refuses ``steps < 1`` too, with ``fq.train() steps must be a positive
    integer``. Caught here so the message names the argument the caller wrote.

    Args:
        steps: The caller's step count.

    Raises:
        ToolInputError: If it is not a positive integer.
    """
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
        raise ToolInputError(
            f"steps must be a positive integer; received {steps!r}. A loop that "
            "runs zero times changes nothing and would still report success."
        )


def _check_learning_rate(learning_rate: Any) -> None:
    """Reject a learning rate Adam cannot use.

    Args:
        learning_rate: The caller's learning rate.

    Raises:
        ToolInputError: If it is not a positive real number.
    """
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        # A JSON integer can be larger than a float holds, and math.isfinite
        # raises OverflowError on one rather than returning False.
        or not _is_finite_number(learning_rate)
        or learning_rate <= 0
    ):
        raise ToolInputError(
            f"learning_rate must be a positive number; received "
            f"{learning_rate!r}. Adam refuses a non-positive rate, and a rate of "
            "zero would run the whole loop without moving anything."
        )


def _check_budget(ir: Any, steps: int) -> None:
    """Refuse a run whose predicted cost is past the configured budget.

    Checked before the objective is built and before anything is traced, so a
    refusal costs nothing and a caller who asked for an hour of work is told so
    immediately rather than by a client that stops responding.

    The prediction is an estimate on the caller's machine rather than a
    measurement of it, which the refusal says, because a bound that reads as a
    promise is a bound someone will plan around.

    Args:
        ir: The validated ``CircuitIR``.
        steps: The requested step count.

    Raises:
        ToolLimitError: If the prediction exceeds the budget.
    """
    budget = limits.max_train_seconds()
    predicted = predict_seconds(ir, steps)
    if predicted <= budget:
        return
    raise ToolLimitError(
        f"This run is predicted to take {predicted:.1f} s: {steps} steps on a "
        f"{int(ir.n_wires)}-qubit circuit with {len(ir.instructions)} "
        f"instructions, past the {budget} s budget. The prediction is an "
        "estimate, not a measurement of your machine. Reduce steps, or use a "
        "narrower or shallower circuit. FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS raises "
        "the budget for a deployment that wants to wait."
    )


def _check_names(names: tuple[str, ...]) -> None:
    """Reject a circuit with nothing to train.

    Args:
        names: The circuit's parameter names.

    Raises:
        ToolInputError: If the circuit has none.
    """
    if names:
        return
    raise ToolInputError(
        "This circuit has no parameters, so there is nothing to train: every "
        "step would evaluate the same circuit and report the same loss. Write "
        'an angle as a symbol — {"theta": {"$parameter": "theta"}} — or '
        "call inspect_parameters_tool to confirm."
    )


def _resolve_values(names: tuple[str, ...], values: Mapping[str, float] | None) -> dict[str, float]:
    """Resolve the caller's starting point, refusing anything ambiguous.

    Both directions are refused. An unknown name is a typo that would otherwise
    be ignored while the parameter it meant trains from zero; a missing name
    would start the rest from zero without saying which.

    Args:
        names: The circuit's parameter names.
        values: The caller's values, if any.

    Returns:
        One float per name, in ``names`` order.

    Raises:
        ToolInputError: If a name is unknown, missing, or not a real number.
    """
    if values is None:
        return dict.fromkeys(names, 0.0)
    if not isinstance(values, Mapping):
        raise ToolInputError(
            f"values must be an object mapping parameter names to numbers; "
            f"received {type(values).__name__}."
        )
    unknown = sorted(set(values) - set(names))
    if unknown:
        raise ToolInputError(
            f"values names {unknown}, which this circuit does not have. Its "
            f"parameters are {list(names)}. A misspelled name would otherwise be "
            "ignored and the parameter it meant would train from zero."
        )
    missing = sorted(set(names) - set(values))
    if missing:
        raise ToolInputError(
            f"values is missing {missing}. This circuit has parameters "
            f"{list(names)}, and every one needs a starting value; the rest "
            "would otherwise train from zero without saying so."
        )
    resolved: dict[str, float] = {}
    for name in names:
        value = values[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not _is_finite_number(value)
        ):
            raise ToolInputError(
                f"values[{name!r}] is {value!r}; every starting value must be a finite real number."
            )
        resolved[name] = float(value)
    return resolved


def _final_loss(module: Any, torch: Any) -> tuple[float, dict[str, Any]]:
    """Evaluate the loss at the parameters the run ended on.

    ``result.losses[-1]`` is the loss *entering* the last update, not the loss of
    the parameters being returned, because the SDK evaluates before it steps. One
    more forward evaluation is what makes the continuation exact: feed these
    parameters back as ``values`` and the next call's ``initial_loss`` equals
    this call's ``final_loss``.

    Args:
        module: The trained ``Module``.
        torch: The torch module, for ``no_grad``.

    Returns:
        The final loss, and the execution provenance of the evaluation.
    """
    with torch.no_grad():
        execution = module.execute()
        loss = float(execution.require_value().mean())
    return loss, _provenance(execution)


def _provenance(result: Any) -> dict[str, Any]:
    """Restate where the SDK ran this training run, and what it does not claim.

    A ``Module`` execution's runtime block is not ``fq.run``'s: it carries
    ``executor``, ``backend``, ``mode``, ``world_size`` and ``node_count``, and
    has no ``execution_path`` or ``platform_provider``. Reading the simulation's
    field names here would report two empty strings and look like a summary that
    dropped them.

    The fields restated are the ones that answer "where did this run", and the
    accuracy contract travels whole because it is the SDK's statement that an
    ideal simulation is not evidence about hardware.

    Args:
        result: A ``flagquantum.ExecutionResult`` from ``Module.execute``.

    Returns:
        The runtime facts, as JSON-native values.
    """
    runtime = dict(result.runtime or {})
    return {
        "executor": str(runtime.get("executor", "")),
        "backend": str(runtime.get("backend", "")),
        "mode": str(runtime.get("mode", "")),
        "world_size": int(runtime.get("world_size") or 1),
        "node_count": int(runtime.get("node_count") or 1),
        "accuracy": plain(result.accuracy),
    }


def _read_back(module: Any, names: tuple[str, ...]) -> dict[str, float]:
    """Read the trained value of each named parameter group.

    Addressed by name rather than by position: the groups are a
    ``torch.nn.ParameterDict`` keyed by the names this tool supplied, and reading
    them in ``names`` order would make the result depend on an ordering the SDK
    does not promise.

    Args:
        module: The trained ``Module``.
        names: The parameter names, in the order they are reported.

    Returns:
        One float per name.
    """
    groups = module.named_parameter_groups
    return {name: float(groups[name].detach()) for name in names}


def _argument(value: Any, parameters: Mapping[str, Any], sdk: Any) -> Any:
    """Resolve one live gate argument against the parameter mapping.

    A ``Parameter`` becomes the tensor ``Module`` supplied for that name. A
    ``ParameterExpression`` — ``2.0 * t0`` — is resolved through the SDK's own
    public ``bind``, so the arithmetic stays the SDK's rather than being
    reimplemented here; this module carries no numerical logic and should not
    acquire any.

    Every other value is already a number, a complex or an object the SDK built,
    and passes through untouched.

    ``[0]`` indexes the one-element group ``Module`` creates per name. A group of
    any other size would be a name bound to a vector, which this tool never asks
    for; the assertion that it is one element belongs at the construction site.

    Args:
        value: The live argument from an instruction.
        parameters: The mapping ``Module`` handed the builder.
        sdk: The loaded ``flagquantum`` module, for the two type checks.

    Returns:
        The argument to pass to ``Circuit.gate``.
    """
    if isinstance(value, sdk.Parameter):
        return parameters[value.name][0]
    if isinstance(value, sdk.ParameterExpression):
        return value.bind({name: group[0] for name, group in parameters.items()})
    return value
