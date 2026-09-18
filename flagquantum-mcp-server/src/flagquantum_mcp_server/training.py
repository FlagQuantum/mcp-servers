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
from flagquantum_mcp_server.errors import (
    ToolInputError,
    ToolLimitError,
    UnsupportedFormatError,
)
from flagquantum_mcp_server.planning import validate_pauli_terms
from flagquantum_mcp_server.preconditions import (
    SDK_FAILURE_BASES,
    is_finite_number,
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


def _objective(hamiltonian: Any, *, n_wires: int) -> Any:
    """Build the objective, restating the shared refusals in this tool's vocabulary.

    ``validate_pauli_terms`` says "terms" because that is what the list is called
    inside an ``expectation`` output. Here the caller wrote ``hamiltonian``, so a
    message about a field they never typed is a message they have to translate.
    The refusal is the same one and the validation is the same function; only the
    noun changes.

    Args:
        hamiltonian: The caller's term list, or ``None``.
        n_wires: Circuit width every term's Pauli string must match.

    Returns:
        A ``flagquantum.algorithms.Hamiltonian``.

    Raises:
        ToolInputError: If the objective is missing or unusable.
        ToolLimitError: If it carries more terms than the bound allows.
    """
    try:
        return hamiltonian_from_terms(hamiltonian, n_wires=n_wires)
    except UnsupportedFormatError as exc:
        raise UnsupportedFormatError(_in_this_tools_vocabulary(str(exc))) from exc
    except ToolLimitError as exc:
        raise ToolLimitError(_in_this_tools_vocabulary(str(exc))) from exc
    except ToolInputError as exc:
        raise ToolInputError(_in_this_tools_vocabulary(str(exc))) from exc


def _in_this_tools_vocabulary(message: str) -> str:
    """Restate a shared refusal in the nouns the training caller wrote.

    ``validate_pauli_terms`` speaks the vocabulary of an ``expectation`` output:
    the list is ``terms``, an entry is ``terms[0]``, and the whole thing is an
    "expectation". Reached from ``train_parameters`` the caller wrote
    ``hamiltonian`` and an objective, so every one of those nouns names something
    they never typed. The refusals themselves are unchanged; only the nouns move.

    **Every message this reaches was counted, not estimated.** Nine
    malformed-input refusals plus the bound's. Two open with ``'terms' is``,
    seven with ``terms[0]``, and the bound's with ``This expectation carries``;
    a fourth phrase, ``An expectation needs``, sits mid-sentence inside the
    second. An earlier version of this docstring said "eight ... (twice) ...
    (five times)" and every one of those numbers was wrong.

    **Three of the four patterns are anchored to the start of the message, and
    that is not cosmetic.** The validator quotes its own nouns — ``'terms' is
    NoneType``, ``terms[0] has no 'pauli' string`` — and it also quotes the
    caller's offending value back at them: ``has a coefficient that is a string
    ('terms')``. A caller who passes the literal string ``terms`` produces a
    message containing *both*, and an unanchored replace cannot tell them apart.
    Measured, the unanchored version answered ``('hamiltonian')`` — the caller's
    own value rewritten, in the message whose entire job is to explain it. All
    three of the validator's nouns are message prefixes, so all three can be
    anchored, and after anchoring the caller's literal survives intact in every
    position it can occupy.

    **The rename's boundary, measured rather than asserted.** An earlier draft of
    this paragraph opened "No residual remains", and that absolute was the same
    species of claim as the two before it: written rather than measured. It is
    false at the edge. ``An expectation needs`` has no message prefix of its own
    — it sits mid-sentence — so the replace keys on the context it does have,
    ``empty. An expectation needs``. That context is a position, and the previous
    paragraph's "no position to anchor on" was wrong for exactly that reason; but
    a caller whose value *contains the anchor itself* still has it rewritten.
    Measured: a coefficient of ``empty. An expectation needs a term`` is echoed as
    ``empty. An objective needs a term``. That is the whole remaining boundary,
    and reaching it takes a literal that is itself a fragment of the validator's
    message. It is pinned by an assertion in
    ``test_the_phrase_shaped_message_is_renamed_without_touching_a_caller_literal``
    rather than left as prose. The lesson this cost two rounds: "there is nothing
    to anchor on" and "nothing is left" are both claims about the messages, and
    both need the messages read rather than recalled.

    The two phrases are named rather than replaced wholesale: a blanket
    ``expectation`` -> ``objective`` would also rewrite the message for an
    ``expectation`` *output*, which is the one place the word is correct.

    Args:
        message: A refusal's text, from the shared validator.

    Returns:
        The same refusal, in this tool's nouns.
    """
    if message.startswith("'terms'"):
        message = "'hamiltonian'" + message[len("'terms'") :]
    if message.startswith("terms["):
        close = message.index("]")
        message = "hamiltonian" + message[len("terms") : close + 1] + message[close + 1 :]
    if message.startswith("This expectation carries"):
        message = "This objective carries" + message[len("This expectation carries") :]
    return message.replace(
        # This repository's validator writes this phrase, in exactly one message
        # (planning.py, the empty-terms refusal), always directly after
        # "empty. ": "'terms' is empty. An expectation needs at least one term".
        # A caller's copy of those words arrives inside a quoted value — "is a
        # string ('An expectation needs a term')" — so a context longer than the
        # bare phrase tells the two apart where the phrase alone cannot.
        "empty. An expectation needs",
        "empty. An objective needs",
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


# The budget model's constants. Calibrated from warm per-step measurements at 2,
# 4, 6, 8, 12, 13, 14, 15, 16, 20, 22 and 24 qubits, over circuits from 1 to
# 4000 instructions; the tables are in the design document, and every calibration
# point is pinned by a test. The model over-predicts at all twenty-one points, by
# 1.9x at the worst.
#
# Four costs, each of them one that was measured rather than assumed:
STARTUP_SECONDS = 0.5
MIN_STEP_SECONDS = 0.02  # dispatch, for circuits too small for any term below
STATE_COST = 1.5e-6  # per step per amplitude: holding the state
GATE_COST = 1e-7  # per step per amplitude per instruction: applying gates
# Per step per instruction: one gate's Python dispatch, paid whatever the width
# and therefore invisible to both terms above. Measured warm as the slope of wall
# time against step count, min of three runs. Two circuit shapes were measured,
# so that the term does not follow one generator: the calibration test's ansatz
# tail (3.0e-5 to 5.2e-5 s per instruction over its 4-to-8-wire rows) and its
# wide rows (5.0e-5 to 5.6e-5 at 2 wires with 4000 instructions, 4 with 2500, 6
# with 1500, 8 with 2500 and 8 with 4000 — the five points this term governs).
# The figure is flat in width across both, which is why this is a constant and
# not a power of two, and flat in circuit shape to within the noise.
#
# That noise is the reason for the coefficient's size rather than the mean of
# those numbers. Repeated runs of one circuit on one machine differed by up to
# 1.9x — 0.1260 s against 0.2420 s per step at 4 wires with 2500 instructions,
# i.e. 5.0e-5 against 9.7e-5 s per instruction — so the *range* the term has to
# cover is 5.0e-5 to 9.7e-5, not the minimum. This value clears the largest
# single observation anywhere on the axis by 2.1x and the smallest by 4x, and it
# is the only term covering this axis at all: under-predicting here is what let
# a 4000-instruction two-wire circuit be admitted for 2975 steps, 664 s of work
# against a 60 s budget.
DISPATCH_COST = 2e-4


def predict_seconds(ir: Any, steps: int) -> float:
    """Predict how long ``steps`` updates will take, in seconds.

    Four terms, because cost has four regimes. ``STARTUP_SECONDS`` is the
    one-time ``make_fx`` trace ``Module`` performs when it first compiles the
    builder. The other three are paid on every step: ``STATE_COST`` per amplitude
    for holding a state of ``2 ** n_wires``, ``GATE_COST`` per amplitude per
    instruction for applying gates against it, and ``DISPATCH_COST`` per
    instruction for the Python cost of building the gate at all. They are
    separate terms because they were measured to be: at twenty-four wires a
    single gate costs 12.5 s per step, which is 7.5x what one instruction
    explains, and a single width-independent floor could not see that cost at
    all.

    The floor sits under all three. It is generous on purpose: 20 ms where
    1.8 ms was measured at four qubits, which at the default 60-second budget is
    the difference between a caller being allowed three thousand steps and thirty
    thousand. Three thousand is already more than anyone reads, and a model that
    is an upper bound at every point it was calibrated from is worth more than
    one that is tight at the bottom.

    ``DISPATCH_COST`` is the term that was missing until this round, and it was
    missing along an axis none of the first sixteen calibration points could see:
    each of them was either small-width with few instructions, where the floor
    governs, or large-width, where the state or gate term governs. **None was
    small-width with many instructions.** Measured there, the extremes are worse
    than the shape suggested: a 2500-instruction four-wire circuit costs 0.126 s
    per step where the three-term model predicted the 0.02 s floor, and a
    4000-instruction two-wire circuit costs 0.223 s per step, also against the
    floor. The budget admitted 2975 steps of each — 375 s and 664 s of work
    against a 60 s budget. The design document states the rule this broke in its
    own words: a calibration table that omits an axis cannot constrain the model
    along it.

    The dispatch term governs over roughly two to eleven wires: above that the
    gate term is larger, so this cannot turn an under-prediction into an
    over-refusal at the width of any point the other sixteen rows cover. Those
    sixteen predictions are unchanged by it — recomputed, not argued: the term
    is smaller than the winning term at every one, so none of them moves. No
    test asserts that directly; the calibration loop only asserts the model is
    an upper bound, so a later change to DISPATCH_COST could move them without a
    red. The value pin in test_api_contract.py is what would fail.

    A single coefficient cannot follow the true curve either, which falls from
    1.0e-5 per instruction-state at four qubits to 2.1e-8 at twenty and rises
    again to 5.7e-8 at twenty-four as the state stops fitting where it used to.
    The first two are measured points in the twenty-one-row calibration table;
    the third is the design document's own 24-wire, 71-instruction measurement,
    68.5 s per step, and is not one of the twenty-one. ``GATE_COST`` clears the
    highest point rather than the average one, and this round does not change it.

    Where that leaves the margin, over the twenty-one points it is calibrated at:
    11x at four qubits with eleven instructions, 4.9x at eight and 2.2x at
    twelve, the three widths where the floor is doing all the work; between
    1.9x and 8.4x at the thirteen the state or gate term governs, thinnest at
    twenty-four wires with four instructions and thickest at sixteen wires with
    four layers; and 2.1x to 4.0x at the five the dispatch term governs, the ones
    this round added, measured for the margin against the noisiest single run of
    each rather than its best. The thinnest matters least, because the budget
    admits almost nothing there: the state term alone is 25.2 s per step at
    twenty-four wires, so no circuit at that width gets more than two steps, and
    the full ansatz there, 119.1 s per step, gets none at all. Twenty-two wires
    costs 6.3 s per step, and a one-gate circuit at that width gets nine.

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
        len(ir.instructions) * DISPATCH_COST,  # building each gate
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
    objective = _objective(hamiltonian, n_wires=int(ir.n_wires))

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
            f"The SDK could not complete this training run: {exc}. The SDK "
            "raised this inside the run rather than a check here refusing it "
            "before the run: the circuit, the objective, the starting values "
            "and the step count had all been accepted. 'learning_rate' is the "
            "argument to look at — measured on this SDK, a four-wire circuit "
            "trains normally at 1e37 and reaches this line at 4e37, and the "
            "boundary is float32's maximum divided by ten whether the circuit "
            "is two wires wide or six, so it is the rate being converted "
            "rather than the state being held. The circuit is not the "
            "candidate: what this tool hands the SDK is a single loop over the "
            "instruction list you sent, which always traces."
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


def _check_steps(steps: Any) -> None:
    """Reject a step count the loop cannot use.

    The SDK refuses ``steps < 1`` too, with ``fq.train() steps must be a positive
    integer``. Caught here so the message names the argument the caller wrote.

    Two things are wrong with a count that is merely positive, and this is the
    only one of the tool's three numeric guards a client can reach with a value
    no float holds: ``steps`` is typed ``integer``, so pydantic hands it a
    ``10**400`` unchanged, while ``values`` and ``learning_rate`` are typed
    ``number`` and never get that far. Measured through the tool: without this
    check the count escapes as ``OverflowError: int too large to convert to
    float``, and the client sees an ``INTERNAL_ERROR`` that names no argument
    instead of a refusal naming ``steps``.

    Args:
        steps: The caller's step count.

    Raises:
        ToolInputError: If it is not a positive integer.
    """
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or not is_finite_number(steps)
        or steps < 1
    ):
        raise ToolInputError(
            f"steps must be a positive integer; received {steps!r}. A loop that "
            "runs zero times changes nothing and would still report success, and "
            "a count too large to be a float overflows before the first step."
        )


def _check_learning_rate(learning_rate: Any) -> None:
    """Reject a learning rate Adam cannot use.

    Finiteness is part of the rule rather than an extra: ``nan <= 0`` is false,
    so the positivity test alone lets ``nan`` through, and Adam then refuses it
    with a ``ValueError`` that is not one of ``SDK_FAILURE_BASES``. The name in
    this docstring is the one the message below uses, because a rule with two
    names is a rule a reader has to reconcile.

    Args:
        learning_rate: The caller's learning rate.

    Raises:
        ToolInputError: If it is not a positive finite number.
    """
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, (int, float))
        # A JSON integer can be larger than a float holds; see is_finite_number.
        or not is_finite_number(learning_rate)
        or learning_rate <= 0
    ):
        raise ToolInputError(
            f"learning_rate must be a positive finite number; received "
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
        ToolInputError: If a name is unknown, missing, or not a finite real
            number.
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
            or not is_finite_number(value)
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
