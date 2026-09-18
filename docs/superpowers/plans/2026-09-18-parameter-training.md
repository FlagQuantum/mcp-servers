# Training a circuit's parameters — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `train_parameters_tool`, which optimizes a parameterized circuit's angles against a Pauli-sum energy in one call, replacing the fifty-call manual scan four recorded agent sessions converged on.

**Architecture:** A new `training.py` holds a replay layer that turns a serialized circuit back into the Python callable `fq.Module` traces, a budget model that decides before any work starts, and the tool body. Two existing modules are refactored first so that shared refusals and the shared `terms` vocabulary have one home each rather than two.

**Tech Stack:** Python 3.10–3.12, FastMCP 3.x, FlagQuantum 0.2.x, torch (reached through the SDK), pytest.

**Spec:** `docs/superpowers/specs/2026-09-18-parameter-training-design.md`

## Global Constraints

- **Which directory a command runs in.** Every task assumes the shell starts
  each command in the **repository root**. `git add flagquantum-mcp-server/...`
  and `git checkout flagquantum-mcp-server/...` are written for that directory.
  The mutation and verification blocks below open with
  `cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"` rather than
  `cd flagquantum-mcp-server`, so they work from either; use those forms, and
  never leave a mutation applied because a restore path did not resolve.
- Dependency range is `flagquantum>=0.2,<0.3` and `fastmcp>=3.2.0,<4`. Do not add a third declared dependency; `torch` is reached through the SDK.
- No network egress, no credentials, no hardware. Every tool runs locally and deterministically.
- Tools never raise across the MCP boundary; a failure returns `{"status": "error", "error": {"code": ..., "message": ...}}`.
- Tool names are `<verb>_<noun>_tool` and return a `dict` with `"status"`.
- Verify with these exact commands from `flagquantum-mcp-server/`, never with equivalents:
  ```bash
  ../.venv/bin/ruff check .
  ../.venv/bin/ruff format --check .
  ../.venv/bin/mypy --config-file ../mypy.ini src
  ../.venv/bin/pytest -m "not integration"
  ```
  The `pytest` console script is the one CI uses. `python -m pytest` resolves imports CI cannot.
- Every new assertion must be mutation-tested: break the thing it checks, prove the mutation applied (`assert after != before`), watch it go red.
- **The `RuntimePolicy` is not optional.** `Module(hamiltonian=H)` ignores `H` unless the module's policy sets `observable="hamiltonian"`; the default is `⟨Z₀⟩`. Every construction of a `Module` in this plan passes the policy.

---

### Task 1: Extract the shared Pauli-term validation

`planning.py` owns the `terms` vocabulary. `training.py` needs the same validation, so it is extracted into a named function that both call, rather than duplicated or reached as a private.

**Files:**
- Modify: `flagquantum-mcp-server/src/flagquantum_mcp_server/planning.py`
- Test: `flagquantum-mcp-server/tests/test_hamiltonian.py` (existing, must stay green)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `planning.validate_pauli_terms(terms: Any, *, n_wires: int) -> list[tuple[str, float]]` — one `(pauli_upper, coefficient)` pair per term, in the caller's order.
  - `planning._check_pauli_string(pauli: str, *, n_wires: int, where: str | None = None) -> None` — length, alphabet and all-identity checks, with the caller's location in every message.
  - `planning._pauli_observable(pauli: str, *, n_wires: int, where: str | None = None)` — **unchanged**, including its validation. See Step 1.
  - `planning._hamiltonian_observable(terms: Any, *, n_wires: int) -> Any` — unchanged signature, now looping over `validate_pauli_terms`.

- [ ] **Step 1: Rewrite `_weighted_term` and `_hamiltonian_observable`, add `validate_pauli_terms` and `_check_pauli_string`**

In `planning.py`, replace `_hamiltonian_observable` (currently lines 254-304) and
`_weighted_term` (currently lines 307-336) with the four functions below, and
insert `_check_pauli_string` before `_pauli_observable`. **`_pauli_observable`
itself is not edited** — leave lines 354-393 exactly as they are, `where`
parameter and all, and read the closing note below for why.

Every message below is copied verbatim from the code as it stands today.

```python
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
    return [
        _weighted_term(term, position, n_wires=n_wires)
        for position, term in enumerate(terms)
    ]


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

    Unchanged by this task, and deliberately so. It validates as it builds,
    which means a term reaching it through :func:`_hamiltonian_observable` is
    checked twice — once here and once by :func:`_weighted_term`. The second
    check cannot fire, and the cost is one length comparison.

    The alternative is stripping the validation out and adding a
    ``_check_pauli_string`` call at the single-pauli path in ``_build_output``.
    That is a smaller function and a larger change: it moves a refusal out of
    the function that owns the string and relies on every future caller
    remembering to validate first. Two cheap checks in the wrong order is the
    better trade here, and the mutation in Step 4 is what proves the outer one
    is the one naming ``terms[i]``.

    Args:
        pauli: One letter per wire, drawn from ``I``, ``X``, ``Y``, ``Z``.
        n_wires: Circuit width the string must match.
        where: How to name the caller's location in a refusal, when this string
            is one term of several. ``_weighted_term`` names it there instead, so
            this is only reached from the single-pauli path, where it is ``None``.

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
```

**Leave `_build_output` alone.** Its single-pauli path calls
`_pauli_observable(pauli, n_wires=n_wires)` and keeps every refusal it has today,
because `_pauli_observable` is unchanged. Only the `terms` path gains the
position-naming check, through `_weighted_term`.

Every message string in this task is copied from the code as it stands, so diff
the three functions against `git diff` before running anything: a reworded
refusal here is a failure the tests will report as a behaviour change rather
than as the typo it is.

- [ ] **Step 2: Run the existing Hamiltonian tests**

Run: `../.venv/bin/pytest tests/test_hamiltonian.py tests/test_simulation.py -q`
Expected: PASS, unchanged count. Every refusal message in this refactor is copied verbatim, so any failure here is a message that moved rather than moved intact.

- [ ] **Step 3: Run the whole suite**

Run: `../.venv/bin/pytest -m "not integration" -q`
Expected: PASS, same count as before the change.

- [ ] **Step 4: Mutation-test the refactor's one new claim**

The claim is that `validate_pauli_terms` is now the only thing that validates. Break it and confirm the tests notice.

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path
p = Path("src/flagquantum_mcp_server/planning.py")
before = p.read_text()
after = before.replace(
    "    _check_pauli_string(pauli, n_wires=n_wires, where=where)\n"
    "    return pauli.upper(), float(coefficient)",
    "    return pauli.upper(), float(coefficient)",
)
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_hamiltonian.py -q
```

Expected: FAIL — `test_an_all_identity_term_is_refused_by_its_position` reports that no error was raised, or a length test does. Then restore:

```bash
cd "$(git rev-parse --show-toplevel)" && git checkout flagquantum-mcp-server/src/flagquantum_mcp_server/planning.py
```

If the suite passes with the validation removed, the extraction is decorative and the terms are being checked somewhere else — find out where before continuing.

- [ ] **Step 5: Commit**

```bash
git add flagquantum-mcp-server/src/flagquantum_mcp_server/planning.py
git commit -m "refactor: name the Pauli-term validator so training can call it"
```

---

### Task 2: Move the payload refusals into `preconditions.py`

Three things are shared by tools that run a circuit locally but live in `simulation.py`, which `training.py` must not import from. They move to a module named for what they are.

**Files:**
- Create: `flagquantum-mcp-server/src/flagquantum_mcp_server/preconditions.py`
- Modify: `flagquantum-mcp-server/src/flagquantum_mcp_server/planning.py` (remove `reject_outputs_with_declared_measurements`, import it instead)
- Modify: `flagquantum-mcp-server/src/flagquantum_mcp_server/simulation.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `preconditions.reject_outputs_with_declared_measurements(outputs: Any, ir: Any) -> None`
  - `preconditions.reject_circuit_observables(ir: Any, *, remedy: str) -> None`
  - `preconditions.plain(value: Any) -> Any`
  - `preconditions.SDK_FAILURE_BASES: tuple[type[BaseException], ...]`

`execution_provenance` is deliberately **not** moved. A `Module` execution's
`result.runtime` is a different mapping from `fq.run`'s — it holds `executor`,
`backend`, `mode`, `world_size` and no `execution_path` or `platform_provider` at
all — so one function cannot honestly describe both. The two tools each restate
their own SDK call. What they share is the conversion, which is `plain`.

- [ ] **Step 1: Write the new module**

```python
"""What this server refuses about a circuit before anything acts on it.

Three tools — planning, simulation and training — hand a circuit to an SDK entry
point that reads some of the envelope and ignores the rest. The ignored parts do
not error: ``observables`` is a real field that serializes, hashes and validates
before the default statevector path drops it; a circuit's own ``measurements``
and a caller's ``outputs`` cannot both be honoured and the SDK says so without
naming either field.

Each of those is a place where accepting the input would produce a
plausible-looking answer, so each is refused here rather than found later. The
refusals live in this module rather than in the tool that first needed one, so
that the second tool to need it shares the message instead of writing a second
opinion about the same fact.

The remedies differ by tool — a simulation points at ``outputs``, a training run
at ``hamiltonian`` — so the caller supplies the sentence that says what to do
instead. The fact is shared; the advice is not.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from flagquantum_mcp_server.errors import ToolInputError

# The exception bases reachable from a FlagQuantum execution call:
# ValidationError is a ValueError, ExecutionError a RuntimeError, CapabilityError
# a NotImplementedError. A tool wraps its one SDK call in these and nothing else,
# so what arrives is the SDK describing the request.
SDK_FAILURE_BASES: tuple[type[BaseException], ...] = (
    ValueError,
    RuntimeError,
    NotImplementedError,
)


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


def reject_circuit_observables(ir: Any, *, remedy: str) -> None:
    """Refuse a circuit that carries observables, naming the spelling that works.

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
    ``"heisenberg"`` — the field is refused and the working spelling is given.

    Args:
        ir: The validated ``CircuitIR``.
        remedy: What this caller should do instead. The sentence after the
            count, ending in a period.

    Raises:
        ToolInputError: If the circuit carries any observable.
    """
    observables = tuple(getattr(ir, "observables", ()) or ())
    if not observables:
        return
    raise ToolInputError(
        f"This circuit's 'observables' field carries {len(observables)} "
        "entr(ies), and it is not read: the default statevector path never "
        "evaluates the field, so the result would carry no expectation value "
        f"and no error either. {remedy}"
    )


def plain(value: Any) -> Any:
    """Convert SDK containers and dataclasses into JSON-native types.

    Shared by the simulation and training tools, which report the same kinds of
    value, so that a float reported by one and the same float reported by the
    other have been through the same conversion. Tensors arrive with a leading
    batch axis because the SDK carries a batch dimension through every result;
    the axis is dropped when it is exactly one, which loses nothing.

    Args:
        value: Any value an SDK result may hold.

    Returns:
        A value ``json`` can carry.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return plain(
            {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
        )
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return plain(_without_batch_axis(value.tolist()))
    return str(value)


def _without_batch_axis(converted: Any) -> Any:
    """Drop a leading axis of length one, which carries no information."""
    if isinstance(converted, list) and len(converted) == 1 and isinstance(converted[0], list):
        return converted[0]
    return converted
```

- [ ] **Step 2: Delete the moved code from `simulation.py`**

Remove `_reject_circuit_observables`, `_plain` and `_without_batch_axis` from `simulation.py`, and remove `import dataclasses`. Keep `_execution_provenance`, which is this tool's own shape. Replace the import block:

```python
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
```

In `simulate_execution`, change `_reject_circuit_observables(ir)` to:

```python
    reject_circuit_observables(
        ir,
        remedy=(
            "Ask for the expectation directly instead — pass "
            'outputs=[{"kind": "expectation", "pauli": "ZZ"}] for a single '
            "term, or a 'terms' list of {\"pauli\": ..., \"coefficient\": ...} "
            "objects to measure a weighted sum in one call, or drop the field "
            "from the circuit."
        ),
    )
```

Change the exception clause to use the shared tuple:

```python
    try:
        result = sdk.run(ir, options=_build_options(options), outputs=built)
    except SDK_FAILURE_BASES as exc:
        raise ToolInputError(_with_guidance(str(exc))) from exc
```

`_measurement` calls `_plain` — rename that call to `plain`:

```python
        "value": plain(item.value),
```

and `_execution_provenance` calls `_plain(result.accuracy)` — rename that too.

- [ ] **Step 3: Delete `reject_outputs_with_declared_measurements` from `planning.py`**

It now lives in `preconditions.py`. Replace its definition (currently lines 111-145) with an import at the top of the module, so `planning.plan_execution` keeps calling the same name:

```python
from flagquantum_mcp_server.preconditions import (
    reject_outputs_with_declared_measurements,
)
```

Because the name is imported into `planning`'s namespace, `planning.reject_outputs_with_declared_measurements` still resolves and nothing that calls it changes. `simulation.py` imports it from `preconditions` directly, as its own import block above shows, so there is one definition and no re-export shim.

`planning.py`'s own `_plain` stays: it converts plan summaries, a different input type from the execution results `preconditions.plain` is written for.

- [ ] **Step 4: Run the suite**

Run: `../.venv/bin/pytest -m "not integration" -q`
Expected: PASS, same count. Then the lint and type gates, which will catch anything left unimported:

```bash
../.venv/bin/ruff check .
../.venv/bin/mypy --config-file ../mypy.ini src
```

- [ ] **Step 5: Mutation-test the shared import**

The claim is that `simulation.py` now uses the shared refusal rather than its own. Confirm the shared one is load-bearing:

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path
p = Path("src/flagquantum_mcp_server/preconditions.py")
before = p.read_text()
after = before.replace(
    "    if not observables:\n        return\n",
    "    if not observables:\n        return\n    return\n",
)
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_simulation.py -q
cd "$(git rev-parse --show-toplevel)" && git checkout flagquantum-mcp-server/src/flagquantum_mcp_server/preconditions.py
```

Expected: FAIL on the observable-refusal test before the restore. If it passes, `simulation.py` is still refusing on its own and the move is incomplete.

- [ ] **Step 6: Commit**

```bash
git add flagquantum-mcp-server/src/flagquantum_mcp_server/preconditions.py \
        flagquantum-mcp-server/src/flagquantum_mcp_server/planning.py \
        flagquantum-mcp-server/src/flagquantum_mcp_server/simulation.py
git commit -m "refactor: give the payload refusals one home, and the result shape one too"
```

---

### Task 3: Lazily reach torch, and bound training time

Two small additions. `training.py` needs a `torch.optim.Adam` and a `RuntimePolicy`, and the server declares neither `torch` nor a training time budget.

**Files:**
- Modify: `flagquantum-mcp-server/src/flagquantum_mcp_server/_bridge.py`
- Modify: `flagquantum-mcp-server/src/flagquantum_mcp_server/limits.py`
- Test: `flagquantum-mcp-server/tests/test_api_contract.py`

**Interfaces:**
- Consumes: nothing new.
- Produces:
  - `_bridge.load_torch() -> ModuleType`
  - `limits.max_train_seconds() -> int`, default `60`, variable `FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS`

- [ ] **Step 1: Write the failing test for the bound**

Add to `flagquantum-mcp-server/tests/test_api_contract.py`:

```python
def test_the_training_budget_is_read_per_call(monkeypatch: pytest.MonkeyPatch) -> None:
    from flagquantum_mcp_server import limits

    assert limits.max_train_seconds() == 60
    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS", "5")

    assert limits.max_train_seconds() == 5


def test_a_malformed_training_budget_falls_back_rather_than_disabling_the_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bound that reads as zero would refuse every call, which is a different bug."""
    from flagquantum_mcp_server import limits

    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS", "0")

    assert limits.max_train_seconds() == 60
```

- [ ] **Step 2: Run it to verify it fails**

Run: `../.venv/bin/pytest tests/test_api_contract.py -k training_budget -q`
Expected: FAIL with `AttributeError: module 'flagquantum_mcp_server.limits' has no attribute 'max_train_seconds'`

- [ ] **Step 3: Add the bound**

In `limits.py`, add to the defaults block:

```python
DEFAULT_MAX_TRAIN_SECONDS = 60
```

and after `max_hamiltonian_terms`:

```python
def max_train_seconds() -> int:
    """Return how long one training call may be predicted to take.

    Every other bound here limits what a single call may *consume* — memory,
    width, response size. This one limits time, because training is the only
    operation in this server whose cost is unbounded by the size of its input:
    the same four-qubit circuit costs 0.2 s for ten steps and three minutes for
    ten thousand, and ``steps`` is the caller's number.

    The prediction that reads this bound is an estimate rather than a
    measurement of the caller's machine, so the bound is a policy about how long
    an agent should be made to wait, not a guarantee about the clock.
    """
    return _positive_int("FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS", DEFAULT_MAX_TRAIN_SECONDS)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../.venv/bin/pytest tests/test_api_contract.py -k training_budget -q`
Expected: PASS, 2 tests.

- [ ] **Step 5: Add `load_torch` and its test**

Add to `_bridge.py`, after `load_sdk`:

```python
TORCH_NOT_INSTALLED = (
    "torch is not importable in this environment. It is not a declared "
    "dependency of this server, but the FlagQuantum SDK requires it, so this "
    "means the SDK is installed without its own dependency."
)


def load_torch() -> ModuleType:
    """Import and return ``torch``, which the SDK's training API takes.

    Not a declared dependency of this package: the server declares FlagQuantum
    and FastMCP, and FlagQuantum declares torch. Reaching it through the same
    lazy path as the SDK keeps that true — a direct import here would make torch
    a third dependency the packaging does not state, and would make importing
    this server expensive.

    Returns:
        The imported ``torch`` module.

    Raises:
        FlagQuantumUnavailableError: If torch is not importable.
    """
    return _load_torch()


@lru_cache(maxsize=1)
def _load_torch() -> ModuleType:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise FlagQuantumUnavailableError(TORCH_NOT_INSTALLED) from exc
```

Add to `tests/test_api_contract.py`:

```python
def test_torch_is_reached_lazily_and_is_not_a_declared_dependency() -> None:
    """The optimizer type is torch's, but torch is not this package's to declare."""
    from importlib.metadata import requires

    from flagquantum_mcp_server._bridge import load_torch

    assert load_torch().optim.Adam is not None
    declared = " ".join(requires("flagquantum-mcp-server") or ()).lower()

    assert "torch" not in declared
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `../.venv/bin/pytest tests/test_api_contract.py -q`
Expected: PASS.

- [ ] **Step 7: Mutation-test the bound's fallback**

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path
p = Path("src/flagquantum_mcp_server/limits.py")
before = p.read_text()
after = before.replace(
    'return _positive_int("FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS", DEFAULT_MAX_TRAIN_SECONDS)',
    'return int(os.environ.get("FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS", DEFAULT_MAX_TRAIN_SECONDS))',
)
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_api_contract.py -k training_budget -q
cd "$(git rev-parse --show-toplevel)" && git checkout flagquantum-mcp-server/src/flagquantum_mcp_server/limits.py
```

Expected: FAIL on the malformed-value test. A bound that reads `"0"` as zero would refuse every call for a deployment that meant "unlimited", which is the failure the fallback exists to prevent.

- [ ] **Step 8: Commit**

```bash
git add flagquantum-mcp-server/src/flagquantum_mcp_server/_bridge.py \
        flagquantum-mcp-server/src/flagquantum_mcp_server/limits.py \
        flagquantum-mcp-server/tests/test_api_contract.py
git commit -m "feat: reach torch lazily, and bound a training call's predicted cost"
```

---

### Task 4: The replay layer

A serialized circuit has to become the Python callable `fq.Module` traces. This task builds that and proves it faithful, with no tool attached yet.

**Files:**
- Create: `flagquantum-mcp-server/src/flagquantum_mcp_server/training.py`
- Test: `flagquantum-mcp-server/tests/test_training.py`

**Interfaces:**
- Consumes: `circuits.circuit_from_ir`, `circuits.resolve_ir`, `_bridge.load_sdk`.
- Produces:
  - `training.parameter_names(ir: Any) -> tuple[str, ...]`
  - `training.replay_builder(ir: Any) -> Callable[[Mapping[str, Any]], Any]`

- [ ] **Step 1: Write the failing tests**

Create `flagquantum-mcp-server/tests/test_training.py`:

```python
"""Training a circuit's parameters, and the four ways that can go quietly wrong.

The SDK does the numerics. What these tests pin is the boundary around them,
because every way this tool can mislead a caller is a way of accepting
something:

- a ``hamiltonian`` that the SDK stores and does not read, so training converges
  cleanly on a different objective and reports it as an energy;
- a replay that loses or reorders a gate, so the circuit trained is not the
  circuit sent;
- ``values`` that name a parameter the circuit does not have, so a typo trains
  from zero without saying so;
- a run whose predicted cost is past the budget, which must be refused before
  the client is left waiting rather than after.

The first is the one that shapes this module. It is not hypothetical: an earlier
version of this server's design trained ``⟨Z₀⟩`` for a week of afternoons
because ``Module(hamiltonian=H)`` accepts ``H`` and ignores it unless the
module's policy asks for it.
"""

from __future__ import annotations

import json

import pytest

from flagquantum_mcp_server.circuits import serialize
from flagquantum_mcp_server.training import parameter_names, replay_builder

pytestmark = pytest.mark.unit

ANGLED = json.dumps(
    [
        {"name": "ry", "index": [0], "parameters": {"theta": {"$parameter": "t0"}}},
        {"name": "ry", "index": [1], "parameters": {"theta": {"$parameter": "t1"}}},
        {"name": "cx", "index": [0, 1]},
    ]
)


def _ir(qir: str):
    from flagquantum_mcp_server.circuits import resolve_ir

    return resolve_ir(qir, "qir")


# --- the replay reproduces the circuit it was given ---


def test_a_replay_of_a_numeric_circuit_is_byte_for_byte_the_same_circuit() -> None:
    """The strongest statement available: serialize the rebuild, compare."""
    import flagquantum as fq

    source = fq.Circuit(2).h(0).ry(1, theta=0.7).cx(0, 1)
    source.gate("any", [1], matrix=[[1, 0], [0, 1j]])
    source.rz(0, theta=-0.25)
    ir = source.to_ir()

    rebuilt = replay_builder(ir)({})

    assert rebuilt.to_ir().to_dict() == ir.to_dict()


def test_a_matrix_gate_survives_the_replay() -> None:
    """A matrix is the one argument the replay must not drop or normalize."""
    import flagquantum as fq

    source = fq.Circuit(1)
    source.gate("any", [0], matrix=[[0, 1], [1, 0]])
    ir = source.to_ir()

    rebuilt = replay_builder(ir)({})

    payload = rebuilt.to_ir().to_dict()["instructions"]
    assert payload[0]["matrix"] == ir.to_dict()["instructions"][0]["matrix"]


def test_the_replay_binds_each_symbol_by_name_not_by_position() -> None:
    """Names are the contract; a positional replay would still pass a 2-param test."""
    import torch

    builder = replay_builder(_ir(ANGLED))

    # Deliberately the reverse of the order the parameters appear in.
    built = builder({"t1": torch.tensor([0.25]), "t0": torch.tensor([0.75])})
    payload = built.to_ir().to_dict()["instructions"]

    assert payload[0]["params"]["theta"] == pytest.approx(0.75)
    assert payload[1]["params"]["theta"] == pytest.approx(0.25)


def test_the_replay_carries_non_parameter_arguments_through_unchanged() -> None:
    """A bound angle mixed in with a symbol is not a symbol, and stays bound."""
    import torch

    qir = json.dumps(
        [
            {"name": "rx", "index": [0], "parameters": {"theta": 0.5}},
            {"name": "ry", "index": [0], "parameters": {"theta": {"$parameter": "t0"}}},
        ]
    )

    built = replay_builder(_ir(qir))({"t0": torch.tensor([0.25])})
    payload = built.to_ir().to_dict()["instructions"]

    assert payload[0]["params"]["theta"] == 0.5
    # The symbol became a tensor: a value rather than a name, which is what the
    # builder needs and is not the same thing as the marker it came from.
    assert payload[1]["params"]["theta"]["$tensor"]["data"] == [0.25]


# --- the names come from the SDK ---


def test_parameter_names_are_the_circuits_own_and_are_sorted() -> None:
    """Sorted is the SDK's order, not first-appearance; the test says which."""
    names = parameter_names(_ir(ANGLED))

    assert names == ("t0", "t1")


def test_a_repeated_symbol_is_named_once() -> None:
    qir = json.dumps(
        [
            {"name": "ry", "index": [0], "parameters": {"theta": {"$parameter": "p"}}},
            {"name": "rz", "index": [0], "parameters": {"theta": {"$parameter": "p"}}},
        ]
    )

    assert parameter_names(_ir(qir)) == ("p",)


def test_a_circuit_without_parameters_has_no_names() -> None:
    qir = json.dumps([{"name": "h", "index": [0]}])

    assert parameter_names(_ir(qir)) == ()


def test_a_replayed_parameterized_circuit_evaluates_the_same_as_a_bound_one() -> None:
    """The end the replay exists for: same circuit, same numbers, same energy."""
    import torch
    import flagquantum as fq
    from flagquantum_mcp_server.circuits import circuit_from_ir

    ir = _ir(ANGLED)
    bound = circuit_from_ir(ir).bind_parameters({"t0": 0.3, "t1": 0.4})
    replayed = replay_builder(ir)({"t0": torch.tensor([0.3]), "t1": torch.tensor([0.4])})

    want = fq.run(bound, outputs=[fq.probabilities([0, 1])]).measurements[0].value
    got = fq.run(replayed, outputs=[fq.probabilities([0, 1])]).measurements[0].value

    assert got == pytest.approx(want, abs=1e-6)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `../.venv/bin/pytest tests/test_training.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'flagquantum_mcp_server.training'`

- [ ] **Step 3: Write the replay layer**

Create `flagquantum-mcp-server/src/flagquantum_mcp_server/training.py`:

**The import block below is deliberately minimal.** It carries what *this
task's* code uses and nothing else, because `ruff` selects `F` and an unused
import fails `ruff check .`. Later tasks in this plan extend it as they add the
code that needs each name — Task 5 adds nothing, Task 6 adds `load_module` and
`validate_pauli_terms`, Task 7 adds the rest. Do not import ahead of the code.

```python
"""Train a circuit's parameters against a Pauli-sum energy, in one call.

``fq.Module`` and ``fq.train`` want a Python callable rather than JSON, which
looks like a dead end for a server whose every input is JSON. It is not: a
serialized circuit carries every instruction it was built from, and
``Circuit.gate`` is the single primitive that rebuilds them — built-in gates,
arbitrary matrices and channels all go through it. So the replay below is a loop
and not a gate table, and it carries no numerical logic. It translates a
serialized circuit into the callable shape the SDK asks for, and nothing else.

Three facts about the SDK shape this module, all measured rather than read:

* ``Module(hamiltonian=H)`` stores ``H`` and does not evaluate it. Which
  observable a module measures comes from its ``RuntimePolicy``, whose default is
  ``⟨Z₀⟩``. Every ``Module`` built here passes
  ``policy=RuntimePolicy(observable="hamiltonian")``, and
  ``test_the_hamiltonian_reaches_the_objective`` is what holds that.
* A ``{"$parameter": ...}`` marker is a *serialization* encoding. Handed to the
  Python API it is stored as an opaque dict and the circuit reports itself as
  unparameterized, which is the silent failure ``circuits.py`` already refuses at
  the JSON boundary. Names are therefore read through ``circuit_from_ir``, which
  decodes the markers properly.
* ``Circuit.gate`` takes its arguments as a ``params=`` mapping, not
  positionally, and the argument names are the SDK's own gate manifest.

Training is the only operation here whose cost is not bounded by the size of its
input, so a prediction decides before any work starts. See ``predict_seconds``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from flagquantum_mcp_server._bridge import load_sdk
from flagquantum_mcp_server.circuits import circuit_from_ir

# The one Python name a parameter value may travel under when it is a symbol
# rather than a number. Its presence is what makes the argument a tensor.
PARAMETER_MARKER = "$parameter"


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
    one-element tensor per name — and returns a live ``Circuit``. Instructions
    are read from ``ir.to_dict()``, where a ``Parameter`` has become the marker a
    caller sees over the wire, rather than from the live objects, whose ``params``
    hold SDK types a JSON tool has no business inspecting.

    Args:
        ir: A validated ``CircuitIR``.

    Returns:
        A callable taking ``{name: tensor}`` and returning a ``Circuit``.
    """
    instructions = tuple(ir.to_dict()["instructions"])
    n_wires = int(ir.n_wires)

    def build(parameters: Mapping[str, Any]) -> Any:
        circuit = load_sdk().Circuit(n_wires)
        for instruction in instructions:
            arguments = {
                str(key): _argument(value, parameters)
                for key, value in (instruction.get("params") or {}).items()
            }
            circuit.gate(
                str(instruction["opcode"]),
                [int(wire) for wire in instruction["wires"]],
                params=arguments or None,
                matrix=instruction.get("matrix"),
            )
        return circuit

    return build


def _argument(value: Any, parameters: Mapping[str, Any]) -> Any:
    """Resolve one serialized gate argument against the parameter mapping.

    A symbol becomes the tensor ``Module`` supplied for that name. Every other
    value — a bound number, a ``$tensor``, a ``$complex``, a ``$expression`` —
    passes through untouched, because the SDK's own decoder built it and this
    module has no business reinterpreting it.

    ``[0]`` indexes the one-element group ``Module`` creates per name. A group of
    any other size would be a name bound to a vector, which this tool never asks
    for.

    Args:
        value: The serialized argument.
        parameters: The mapping ``Module`` handed the builder.

    Returns:
        The argument to pass to ``Circuit.gate``.
    """
    symbol = value.get(PARAMETER_MARKER) if isinstance(value, Mapping) else None
    if isinstance(symbol, str):
        return parameters[symbol][0]
    return value
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../.venv/bin/pytest tests/test_training.py -q`
Expected: PASS, 8 tests.

- [ ] **Step 5: Mutation-test each new claim**

Run each mutation, watch it go red, restore. The `assert after != before` line is not decoration: an earlier session shipped a mutation whose replacement target spanned two string literals, so nothing changed and the suite stayed green for a reason that had nothing to do with the test.

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path
p = Path("src/flagquantum_mcp_server/training.py")
before = p.read_text()
after = before.replace(
    "                params=arguments or None,\n                matrix=instruction.get(\"matrix\"),\n",
    "                params=arguments or None,\n",
)
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_training.py -q
cd "$(git rev-parse --show-toplevel)" && git checkout flagquantum-mcp-server/src/flagquantum_mcp_server/training.py
```

Expected: FAIL on `test_a_replay_of_a_numeric_circuit_is_byte_for_byte_the_same_circuit` and `test_a_matrix_gate_survives_the_replay`.

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path
p = Path("src/flagquantum_mcp_server/training.py")
before = p.read_text()
after = before.replace(
    "        return parameters[symbol][0]",
    "        return next(iter(parameters.values()))[0]",
)
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_training.py -q
cd "$(git rev-parse --show-toplevel)" && git checkout flagquantum-mcp-server/src/flagquantum_mcp_server/training.py
```

Expected: FAIL on `test_the_replay_binds_each_symbol_by_name_not_by_position`. That test uses two parameters and binds them in reverse order precisely so that a positional replay cannot pass.

- [ ] **Step 6: Commit**

```bash
git add flagquantum-mcp-server/src/flagquantum_mcp_server/training.py \
        flagquantum-mcp-server/tests/test_training.py
git commit -m "feat: replay a serialized circuit into the callable fq.Module traces"
```

---

### Task 5: The budget model

Cost is the one thing here that a caller cannot predict from the size of their input, so it is predicted for them and refused before any work starts.

**Files:**
- Modify: `flagquantum-mcp-server/src/flagquantum_mcp_server/training.py`
- Test: `flagquantum-mcp-server/tests/test_training.py`

**Interfaces:**
- Consumes: `limits.max_train_seconds()`.
- Produces:
  - `training.predict_seconds(ir: Any, steps: int) -> float`
  - `training.STARTUP_SECONDS: float` = 0.5
  - `training.MIN_STEP_SECONDS: float` = 0.02
  - `training.STEP_COST_COEFFICIENT: float` = 1e-7

- [ ] **Step 1: Write the failing tests**

Append to `flagquantum-mcp-server/tests/test_training.py`:

```python
# --- the budget model ---


def test_the_prediction_grows_with_steps_and_with_width() -> None:
    from flagquantum_mcp_server.training import predict_seconds

    narrow = _ir(json.dumps([{"name": "h", "index": [0]}]))
    wide = _ir(json.dumps(_layered_qir(16)))

    assert predict_seconds(wide, 10) > predict_seconds(wide, 5)
    assert predict_seconds(wide, 5) > predict_seconds(narrow, 5)


def test_the_prediction_over_predicts_every_point_it_was_calibrated_from() -> None:
    """The model's only job is to be an upper bound, so this is the whole test.

    Measured per-step costs, warm, from the design's table. A model that
    under-predicts anywhere here declines to protect the caller exactly where
    the caller most needs it.
    """
    from flagquantum_mcp_server.training import STARTUP_SECONDS, predict_seconds

    measured = [
        (4, 1, 0.0018),
        (8, 1, 0.0041),
        (12, 1, 0.0090),
        (13, 1, 0.0155),
        (14, 1, 0.0207),
        (15, 1, 0.0421),
        (16, 1, 0.0852),
        (20, 1, 1.290),
        (22, 1, 8.56),
        (24, 1, 68.5),
    ]
    for n_wires, layers, per_step in measured:
        ir = _ir(json.dumps(_layered_qir(n_wires, layers)))
        # Strip the one-time startup, which belongs to no single step. Read from
        # the module rather than written as 0.5, so that changing the constant
        # changes what this compares instead of quietly comparing nothing.
        predicted = predict_seconds(ir, 1) - STARTUP_SECONDS

        assert predicted > per_step, (
            f"{n_wires} qubits: predicted {predicted:.4f}s per step against "
            f"{per_step:.4f}s measured"
        )


def test_the_floor_is_what_covers_the_smallest_width() -> None:
    """At four qubits the state term is negligible and dispatch is the whole cost."""
    from flagquantum_mcp_server.training import (
        MIN_STEP_SECONDS,
        STARTUP_SECONDS,
        predict_seconds,
    )

    ir = _ir(json.dumps(_layered_qir(4)))

    assert predict_seconds(ir, 1) - STARTUP_SECONDS == pytest.approx(MIN_STEP_SECONDS)


def test_the_prediction_is_computed_without_running_anything() -> None:
    """It is arithmetic on the instruction count, so it costs nothing to ask."""
    import time

    from flagquantum_mcp_server.training import predict_seconds

    ir = _ir(json.dumps(_layered_qir(24)))
    started = time.perf_counter()
    predict_seconds(ir, 100_000)
    elapsed = time.perf_counter() - started

    assert elapsed < 0.05


def _layered_qir(n_wires: int, layers: int = 1) -> list[dict[str, object]]:
    """An ry/cx/rz ansatz as a gate list, one ``$parameter`` per rotation."""
    gates: list[dict[str, object]] = []
    for layer in range(layers):
        for wire in range(n_wires):
            gates.append(
                {
                    "name": "ry",
                    "index": [wire],
                    "parameters": {"theta": {"$parameter": f"t{layer}_{wire}"}},
                }
            )
        for wire in range(n_wires - 1):
            gates.append({"name": "cx", "index": [wire, wire + 1]})
        for wire in range(n_wires):
            gates.append(
                {
                    "name": "rz",
                    "index": [wire],
                    "parameters": {"theta": {"$parameter": f"u{layer}_{wire}"}},
                }
            )
    return gates
```

- [ ] **Step 2: Run them to verify they fail**

Run: `../.venv/bin/pytest tests/test_training.py -k prediction -q`
Expected: FAIL with `ImportError: cannot import name 'predict_seconds'`

- [ ] **Step 3: Write the model**

Add to `training.py`. No new imports: this is arithmetic on `ir` and `steps`,
and the three constants it defines are the module's own.

```python
# The budget model's three constants. Calibrated from warm per-step measurements
# at 4, 8, 12, 13, 14, 15, 16, 20, 22 and 24 qubits; the table is in the design
# document and every point is pinned by a test. The model over-predicts at all
# of them, by 1.7x at the worst.
STARTUP_SECONDS = 0.5
MIN_STEP_SECONDS = 0.02
STEP_COST_COEFFICIENT = 1e-7


def predict_seconds(ir: Any, steps: int) -> float:
    """Predict how long ``steps`` updates will take, in seconds.

    Two terms, because cost has two regimes. ``STARTUP_SECONDS`` is the one-time
    ``make_fx`` trace ``Module`` performs when it first compiles the builder.
    The rest is work: gates applied against a state of ``2 ** n_wires``
    amplitudes, so it is the product of the two, floored at the dispatch cost
    that dominates at small widths.

    The floor is generous on purpose. It is 20 ms where 1.8 ms was measured at
    four qubits, which at the default 60-second budget is the difference between
    a caller being allowed three thousand steps and thirty thousand. Three
    thousand is already more than anyone reads, and a model that is an upper
    bound everywhere is worth more than one that is tight at the bottom.

    A single coefficient cannot follow the true curve, which falls from 1.0e-5
    per instruction-state at four qubits to 2.1e-8 at twenty and rises again to
    5.7e-8 at twenty-four as the state stops fitting where it used to. The
    coefficient clears the highest point rather than the average one, which
    makes this 3-5x pessimistic through the middle.

    Args:
        ir: A validated ``CircuitIR``.
        steps: The number of updates requested.

    Returns:
        A prediction in seconds. An estimate rather than a measurement of the
        caller's machine.
    """
    per_step = max(
        MIN_STEP_SECONDS,
        len(ir.instructions) * 2 ** int(ir.n_wires) * STEP_COST_COEFFICIENT,
    )
    return STARTUP_SECONDS + steps * per_step
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../.venv/bin/pytest tests/test_training.py -q`
Expected: PASS, 13 tests.

- [ ] **Step 5: Mutation-test the model's direction**

The claim is that the prediction is an upper bound. Break it downward and confirm the calibration test notices.

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path
p = Path("src/flagquantum_mcp_server/training.py")
before = p.read_text()
after = before.replace("STEP_COST_COEFFICIENT = 1e-7", "STEP_COST_COEFFICIENT = 1e-9")
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_training.py -k over_predicts -q
cd "$(git rev-parse --show-toplevel)" && git checkout flagquantum-mcp-server/src/flagquantum_mcp_server/training.py
```

Expected: FAIL, naming the width whose prediction fell below its measurement. Then confirm the floor is load-bearing the same way:

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path
p = Path("src/flagquantum_mcp_server/training.py")
before = p.read_text()
after = before.replace("MIN_STEP_SECONDS = 0.02", "MIN_STEP_SECONDS = 0.0001")
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_training.py -k over_predicts -q
cd "$(git rev-parse --show-toplevel)" && git checkout flagquantum-mcp-server/src/flagquantum_mcp_server/training.py
```

Expected: FAIL at 4 and 8 qubits.

- [ ] **Step 6: Commit**

```bash
git add flagquantum-mcp-server/src/flagquantum_mcp_server/training.py \
        flagquantum-mcp-server/tests/test_training.py
git commit -m "feat: predict a training run's cost before starting it"
```

---

### Task 6: The objective

`Module(hamiltonian=...)` takes a `flagquantum.algorithms.Hamiltonian`, and there is no way to build one from the `terms` JSON the rest of the server speaks. This task adds that, sharing Task 1's validator.

**Files:**
- Modify: `flagquantum-mcp-server/src/flagquantum_mcp_server/training.py`
- Modify: `flagquantum-mcp-server/src/flagquantum_mcp_server/_bridge.py`
- Test: `flagquantum-mcp-server/tests/test_training.py`

**Interfaces:**
- Consumes: `planning.validate_pauli_terms`.
- Produces:
  - `training.hamiltonian_from_terms(terms: Any, *, n_wires: int) -> Any`
  - `training.load_algorithms() -> Any`
  - `_bridge.load_module(module_path: str) -> ModuleType`

- [ ] **Step 1: Write the failing tests**

Append to `flagquantum-mcp-server/tests/test_training.py`:

```python
# --- the objective ---


def test_a_hamiltonian_is_built_from_the_same_terms_shape_the_outputs_use() -> None:
    from flagquantum_mcp_server.training import hamiltonian_from_terms

    built = hamiltonian_from_terms(
        [{"pauli": "ZZ", "coefficient": 1.0}, {"pauli": "XI", "coefficient": -0.5}],
        n_wires=2,
    )

    assert len(built.terms) == 2
    assert [float(term.coefficient) for term in built.terms] == [1.0, -0.5]


def test_a_term_that_the_expectation_builder_refuses_is_refused_here_too() -> None:
    """One shape, one validator, one set of refusals — that is the point of sharing."""
    from flagquantum_mcp_server.training import hamiltonian_from_terms

    with pytest.raises(ToolInputError) as caught:
        hamiltonian_from_terms([{"pauli": "ZZ"}, {"pauli": "II"}], n_wires=2)

    assert "terms[1]" in str(caught.value)
    assert "identity" in str(caught.value)


def test_a_term_that_does_not_match_the_circuit_width_is_refused_here_too() -> None:
    from flagquantum_mcp_server.training import hamiltonian_from_terms

    with pytest.raises(ToolInputError) as caught:
        hamiltonian_from_terms([{"pauli": "ZZZ"}], n_wires=2)

    assert "covers 3 wires" in str(caught.value)


def test_a_hamiltonian_with_a_single_identity_free_term_is_accepted() -> None:
    """The smallest legal objective, so the refusal above is not refusing everything."""
    from flagquantum_mcp_server.training import hamiltonian_from_terms

    assert len(hamiltonian_from_terms([{"pauli": "XI"}], n_wires=2).terms) == 1
```

Add `from flagquantum_mcp_server.errors import ToolInputError` to the test module's imports.

- [ ] **Step 2: Run them to verify they fail**

Run: `../.venv/bin/pytest tests/test_training.py -k hamiltonian -q`
Expected: FAIL with `ImportError: cannot import name 'hamiltonian_from_terms'`

- [ ] **Step 3: Write the builder**

Add to `training.py`, and extend its import block with the two names this code
uses:

```python
from flagquantum_mcp_server._bridge import load_module, load_sdk
from flagquantum_mcp_server.planning import validate_pauli_terms
```

```python
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
```

That needs `load_module` beside `load_attribute` in `_bridge.py`. Add it in this task:

```python
def load_module(module_path: str) -> ModuleType:
    """Import and return a submodule of the SDK by path.

    A companion to :func:`load_attribute` for the case where several names come
    from one module: importing it once and reading the attributes is clearer than
    three calls that each re-import it.

    Args:
        module_path: Dotted path of the module to import.

    Returns:
        The imported module.

    Raises:
        FlagQuantumUnavailableError: If the module is not importable.
    """
    try:
        return importlib.import_module(module_path)
    except ImportError as exc:
        raise FlagQuantumUnavailableError(
            f"{module_path} is not importable in the installed flagquantum. "
            "This server requires the module that provides the training "
            "objective."
        ) from exc
```

Change `training.py`'s bridge import to carry both new names:

```python
from flagquantum_mcp_server._bridge import load_module, load_sdk, load_torch
```

`load_algorithms` calls `load_module`, so without this the module raises
`NameError` on the first import rather than on the first call — which is the
failure a collection error reports anyway, but fix it here rather than
discovering it in Step 4.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../.venv/bin/pytest tests/test_training.py -q`
Expected: PASS, 17 tests.

- [ ] **Step 5: Mutation-test the identity filter**

The comprehension drops each `I` before it becomes a wire, which is what makes
the letter's *position* its wire number. Break it and check what the SDK's own
normalizer does with the result — this is worth knowing, not just worth
asserting:

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path
p = Path("src/flagquantum_mcp_server/training.py")
before = p.read_text()
after = before.replace(
    '{wire: letter for wire, letter in enumerate(pauli) if letter != "I"}',
    "{wire: letter for wire, letter in enumerate(pauli)}",
)
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/python -c "
from flagquantum_mcp_server.training import hamiltonian_from_terms
term = hamiltonian_from_terms([{'pauli': 'IX'}], n_wires=2).terms[0]
print('ops after the SDK normalized it:', term.ops)
"
cd "$(git rev-parse --show-toplevel)" && git checkout flagquantum-mcp-server/src/flagquantum_mcp_server/training.py
```

`_normalize_pauli` in the SDK filters identities out of its `ops` tuple, so the
printed ops may be identical either way. That is the finding: the `if letter !=
"I"` is **not** what keeps the wire numbering right, and a test asserting it
would be a test that passes for the wrong reason. Confirm the numbering a
different way — that `"IX"` puts its `X` on wire 1 and not wire 0:

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from flagquantum_mcp_server.training import hamiltonian_from_terms
assert hamiltonian_from_terms([{"pauli": "IX"}], n_wires=2).terms[0].ops == ((1, "x"),), "IX is on the wrong wire"
assert hamiltonian_from_terms([{"pauli": "XI"}], n_wires=2).terms[0].ops == ((0, "x"),), "XI is on the wrong wire"
print("the letter's position is its wire")
PY
```

If either assertion fails, add it to the test module as
`test_a_pauli_letter_lands_on_the_wire_its_position_names` before continuing —
a mismatch here would train a different Hamiltonian than the caller wrote, with
every energy plausible.

- [ ] **Step 6: Commit**

```bash
git add flagquantum-mcp-server/src/flagquantum_mcp_server/_bridge.py \
        flagquantum-mcp-server/src/flagquantum_mcp_server/training.py \
        flagquantum-mcp-server/tests/test_training.py
git commit -m "feat: build a training objective from the terms shape outputs already use"
```

---

### Task 7: The tool body

This is the task that makes the tool real: validate, predict, refuse if the prediction is past the budget, run, report.

**Files:**
- Modify: `flagquantum-mcp-server/src/flagquantum_mcp_server/training.py`
- Test: `flagquantum-mcp-server/tests/test_training.py`

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: `training.train_parameters(circuit, hamiltonian, circuit_format="ir", *, values=None, steps=50, learning_rate=0.1) -> dict[str, Any]`

- [ ] **Step 1: Write the failing tests**

Append to `flagquantum-mcp-server/tests/test_training.py`:

```python
# --- the run ---

# -Z0Z1 + X0 + X1, whose ground energy is -2.2360679... (=-sqrt(5)).
TFIM2 = [{"pauli": "ZZ", "coefficient": -1.0}, {"pauli": "XI", "coefficient": 1.0},
         {"pauli": "IX", "coefficient": 1.0}]


def test_training_returns_a_trajectory_and_the_parameters_it_ended_on() -> None:
    from flagquantum_mcp_server.training import train_parameters

    payload = train_parameters(ANGLED, TFIM2, "qir", steps=20, learning_rate=0.3)

    assert payload["status"] == "success"
    assert payload["completed_steps"] == 20
    assert len(payload["losses"]) == 20
    assert set(payload["parameters"]) == {"t0", "t1"}
    assert payload["circuit"]["n_qubits"] == 2


def test_the_hamiltonian_reaches_the_objective() -> None:
    """The test this whole module exists for.

    With the default policy the SDK evaluates ⟨Z₀⟩ and ignores the Hamiltonian,
    so a run against ``-Z0Z1 + X0 + X1`` would converge to -1 while claiming to
    have found the ground state. The energy here has to approach -sqrt(5).
    """
    from flagquantum_mcp_server.training import train_parameters

    payload = train_parameters(ANGLED, TFIM2, "qir", steps=200, learning_rate=0.2)

    assert payload["final_loss"] == pytest.approx(-5 ** 0.5, abs=0.05)


def test_the_objective_is_the_hamiltonian_and_not_the_first_wire_s_z() -> None:
    """One step in, the two objectives are already different numbers.

    At the default all-zero starting angles the circuit is ``|00⟩``, so
    ``-Z₀Z₁ + X₀ + X₁`` evaluates to ``-1`` while ``⟨Z₀⟩`` evaluates to ``+1``.
    A run that reports ``+1`` is measuring the wrong operator, which is the trap
    this design was rewritten around, and the sign is what makes it visible in a
    single call rather than after two hundred.
    """
    from flagquantum_mcp_server.training import train_parameters

    payload = train_parameters(ANGLED, TFIM2, "qir", steps=1)

    assert payload["initial_loss"] == pytest.approx(-1.0, abs=1e-5)


def test_final_loss_is_the_loss_of_the_parameters_returned_not_the_one_before() -> None:
    """The continuation promise: resume and the curve joins without a step back."""
    from flagquantum_mcp_server.training import train_parameters

    first = train_parameters(ANGLED, TFIM2, "qir", steps=5, learning_rate=0.3)
    resumed = train_parameters(
        ANGLED, TFIM2, "qir", values=first["parameters"], steps=5, learning_rate=0.3
    )

    assert resumed["initial_loss"] == pytest.approx(first["final_loss"], abs=1e-6)
    assert resumed["initial_parameters"] == pytest.approx(first["parameters"])
    assert resumed["initial_loss"] < resumed["final_loss"]


def test_the_reported_parameters_are_what_the_next_call_starts_from() -> None:
    from flagquantum_mcp_server.training import train_parameters

    payload = train_parameters(ANGLED, TFIM2, "qir", steps=3, learning_rate=0.3)
    again = train_parameters(
        ANGLED, TFIM2, "qir", values=payload["parameters"], steps=1
    )

    assert again["initial_parameters"] == pytest.approx(payload["parameters"])


def test_the_execution_block_says_where_the_sdk_ran_this() -> None:
    """A Module run reports its own runtime shape, not a fq.run one.

    Measured: ``Module.execute()`` returns a runtime block holding ``executor``,
    ``backend``, ``mode``, ``world_size`` and ``node_count``, and no
    ``execution_path`` or ``platform_provider`` at all. So this tool restates
    what its own SDK call reported rather than reusing the simulation's field
    names, which would come back empty.
    """
    from flagquantum_mcp_server.training import train_parameters

    payload = train_parameters(ANGLED, TFIM2, "qir", steps=1)

    assert payload["execution"]["backend"] == "pytorch"
    assert payload["execution"]["mode"] == "statevector"
    assert payload["execution"]["node_count"] == 1
    assert payload["execution"]["world_size"] == 1
    assert payload["execution"]["accuracy"]["metric"] == "not_measured"


def test_values_defaults_to_zero_and_says_so() -> None:
    from flagquantum_mcp_server.training import train_parameters

    payload = train_parameters(ANGLED, TFIM2, "qir", steps=1)

    assert payload["initial_parameters"] == {"t0": 0.0, "t1": 0.0}


def test_a_different_learning_rate_changes_the_run() -> None:
    from flagquantum_mcp_server.training import train_parameters

    slow = train_parameters(ANGLED, TFIM2, "qir", steps=5, learning_rate=0.01)
    fast = train_parameters(ANGLED, TFIM2, "qir", steps=5, learning_rate=0.5)

    assert slow["parameters"] != fast["parameters"]


def test_a_different_step_count_changes_the_run() -> None:
    from flagquantum_mcp_server.training import train_parameters

    short = train_parameters(ANGLED, TFIM2, "qir", steps=1, learning_rate=0.3)
    longer = train_parameters(ANGLED, TFIM2, "qir", steps=4, learning_rate=0.3)

    assert short["completed_steps"] == 1
    assert longer["completed_steps"] == 4


def test_the_parameters_argument_is_not_mutated_by_the_call() -> None:
    """Rule 5: read-only over the caller's inputs."""
    from flagquantum_mcp_server.training import train_parameters

    values = {"t0": 0.25, "t1": -0.5}
    train_parameters(ANGLED, TFIM2, "qir", values=values, steps=2)

    assert values == {"t0": 0.25, "t1": -0.5}


# --- the budget ---


def test_a_run_past_the_budget_is_refused_before_any_work_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refused before, not after: the client is never left waiting."""
    import time

    from flagquantum_mcp_server.training import train_parameters

    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS", "1")
    wide = json.dumps(_layered_qir(16))
    started = time.perf_counter()

    with pytest.raises(ToolLimitError) as caught:
        train_parameters(wide, [{"pauli": "I" * 16, "coefficient": 1.0}], "qir", steps=5000)

    assert time.perf_counter() - started < 1.0, "the refusal did no work"
    message = str(caught.value)
    assert "16" in message
    assert "5000" in message
    assert "FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS" in message


def test_a_run_inside_the_budget_is_not_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from flagquantum_mcp_server.training import train_parameters

    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS", "1")

    assert train_parameters(ANGLED, TFIM2, "qir", steps=1)["status"] == "success"
```

Add `ToolLimitError` to the test module's error import.

- [ ] **Step 2: Run them to verify they fail**

Run: `../.venv/bin/pytest tests/test_training.py -k "trajectory or ham_ or reaches" -q`
Expected: FAIL with `ImportError: cannot import name 'train_parameters'`

Note: the refusal test at 16 qubits uses `[{"pauli": "I"*16, ...}]`, which the shared validator refuses as all-identity — so the budget check must run before the objective is built, or this test fails for the wrong reason. Put the budget check first and let this test be the one that proves the ordering.

- [ ] **Step 3: Write the tool body**

Add to `training.py`, and extend its import block to this complete set — every
name here is used by this task's code, and `ruff`'s `F401` will say so if one is
not:

```python
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
```

```python
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
    module = sdk.Module(
        replay_builder(ir),
        parameters={name: 1 for name in names},
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
        "an angle as a symbol — {\"theta\": {\"$parameter\": \"theta\"}} — or "
        "call inspect_parameters_tool to confirm."
    )


def _resolve_values(
    names: tuple[str, ...], values: Mapping[str, float] | None
) -> dict[str, float]:
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
        return {name: 0.0 for name in names}
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
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ToolInputError(
                f"values[{name!r}] is {value!r}; every starting value must be a "
                "real number."
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../.venv/bin/pytest tests/test_training.py -q`
Expected: PASS, 31 tests.

If `test_the_hamiltonian_reaches_the_objective` fails by converging to -1, the `policy=` argument is not reaching the SDK. That is the exact failure this task exists to prevent; do not weaken the assertion.

- [ ] **Step 5: Mutation-test the policy, which is the point of the module**

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path
p = Path("src/flagquantum_mcp_server/training.py")
before = p.read_text()
after = before.replace(
    '        policy=sdk.RuntimePolicy(observable="hamiltonian"),\n', ""
)
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_training.py -q
cd "$(git rev-parse --show-toplevel)" && git checkout flagquantum-mcp-server/src/flagquantum_mcp_server/training.py
```

Expected: FAIL on `test_the_hamiltonian_reaches_the_objective` — the run converges to `-1.0` instead of `-2.236`. If it passes, the policy is being set somewhere else and this tool is not the thing under test.

Then the continuation promise:

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path
p = Path("src/flagquantum_mcp_server/training.py")
before = p.read_text()
after = before.replace(
    '        "final_loss": final_loss,',
    '        "final_loss": float(result.losses[-1]),',
)
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_training.py -k final_loss -q
cd "$(git rev-parse --show-toplevel)" && git checkout flagquantum-mcp-server/src/flagquantum_mcp_server/training.py
```

Expected: FAIL on `test_final_loss_is_the_loss_of_the_parameters_returned_not_the_one_before`.

Then the budget check's *ordering*, which is the claim that a refusal costs
nothing and a caller is never left waiting:

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path
p = Path("src/flagquantum_mcp_server/training.py")
before = p.read_text()
after = before.replace(
    "    _check_budget(ir, steps)\n\n    names = parameter_names(ir)",
    "\n    names = parameter_names(ir)",
).replace(
    "    objective = hamiltonian_from_terms(hamiltonian, n_wires=int(ir.n_wires))",
    "    _check_budget(ir, steps)\n    objective = hamiltonian_from_terms(hamiltonian, n_wires=int(ir.n_wires))",
)
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_training.py -k budget -q
cd "$(git rev-parse --show-toplevel)" && git checkout flagquantum-mcp-server/src/flagquantum_mcp_server/training.py
```

Expected: FAIL on `test_a_run_past_the_budget_is_refused_before_any_work_starts`.
That test hands in an all-identity objective, which the shared validator
refuses, so with the budget check moved below it the raise is a `ToolInputError`
about `terms[0]` rather than a `ToolLimitError` about the budget — and
`pytest.raises(ToolLimitError)` does not catch it. The all-identity objective is
there on purpose: it is what makes the ordering observable rather than merely
asserted.

- [ ] **Step 6: Run every gate**

```bash
../.venv/bin/ruff check .
../.venv/bin/ruff format --check .
../.venv/bin/mypy --config-file ../mypy.ini src
../.venv/bin/pytest -m "not integration"
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add flagquantum-mcp-server/src/flagquantum_mcp_server/training.py \
        flagquantum-mcp-server/tests/test_training.py
git commit -m "feat: train a circuit's parameters against a Pauli-sum energy"
```

---

### Task 8: The input refusals

Task 7 refuses what would break the run. This task refuses what would make it *succeed at the wrong thing*, which is the harder class.

**Files:**
- Modify: `flagquantum-mcp-server/src/flagquantum_mcp_server/training.py`
- Test: `flagquantum-mcp-server/tests/test_training.py`

**Interfaces:**
- Consumes: `training.train_parameters`.
- Produces: no new names. The refusals are inside `train_parameters` and its helpers.

- [ ] **Step 1: Write the failing tests**

Append to `flagquantum-mcp-server/tests/test_training.py`:

```python
# --- the refusals that matter ---


def test_a_circuit_with_no_parameters_is_refused_by_name() -> None:
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(json.dumps([{"name": "h", "index": [0]}]), [{"pauli": "Z"}], "qir")

    message = str(caught.value)
    assert "no parameters" in message
    assert "inspect_parameters_tool" in message


def test_a_missing_hamiltonian_is_refused() -> None:
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(ANGLED, None, "qir")

    assert "hamiltonian" in str(caught.value)


def test_an_empty_hamiltonian_is_refused() -> None:
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(ANGLED, [], "qir")

    assert "empty" in str(caught.value)


def test_a_value_for_a_parameter_the_circuit_does_not_have_is_refused() -> None:
    """The typo case: silently ignored, the parameter trains from zero."""
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(
            ANGLED, TFIM2, "qir", values={"t0": 0.1, "t1": 0.1, "t2": 0.1}
        )

    message = str(caught.value)
    assert "t2" in message
    assert "t0" in message and "t1" in message


def test_a_missing_value_is_refused() -> None:
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(ANGLED, TFIM2, "qir", values={"t0": 0.1})

    assert "t1" in str(caught.value)


def test_a_starting_value_that_is_not_a_number_is_refused() -> None:
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(ANGLED, TFIM2, "qir", values={"t0": "0.1", "t1": 0.1})

    assert "t0" in str(caught.value)


@pytest.mark.parametrize("steps", [0, -1, 1.5, True, "10"])
def test_a_step_count_the_loop_cannot_use_is_refused(steps: object) -> None:
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(ANGLED, TFIM2, "qir", steps=steps)

    assert "steps" in str(caught.value)


@pytest.mark.parametrize("rate", [0, 0.0, -0.1, True, "0.1"])
def test_a_learning_rate_adam_cannot_use_is_refused(rate: object) -> None:
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(ANGLED, TFIM2, "qir", learning_rate=rate)

    assert "learning_rate" in str(caught.value)


def test_a_circuit_carrying_observables_is_refused_and_points_at_hamiltonian() -> None:
    """The field is never read, so a caller who put the objective there gets nothing."""
    from flagquantum_mcp_server.circuits import serialize
    from flagquantum_mcp_server.training import train_parameters

    envelope = json.loads(serialize(ANGLED, "qir")["ir_json"])
    envelope["observables"] = [{"name": "ZZ", "wires": [0, 1], "coefficient": 1.0}]

    with pytest.raises(ToolInputError) as caught:
        train_parameters(json.dumps(envelope), TFIM2, "ir")

    message = str(caught.value)
    assert "observables" in message
    assert "hamiltonian" in message
```

- [ ] **Step 2: Run them to verify they fail**

Run: `../.venv/bin/pytest tests/test_training.py -k "refused or no_parameters" -q`
Expected: some FAIL. Which ones fail depends on what Task 7 left implicit — the missing-`hamiltonian` and empty-`hamiltonian` cases in particular currently reach `validate_pauli_terms` and produce a message about `'terms'` rather than about `hamiltonian`. Note the actual messages before changing anything; a refusal that names the wrong field is the failure this task is for.

- [ ] **Step 3: Name the `hamiltonian` argument in its own refusals**

`validate_pauli_terms` writes its messages in the vocabulary of the `outputs` argument, where the list is called `terms`. Reached from `train_parameters` the caller wrote `hamiltonian`, so the message names a field they never typed. Wrap the call:

```python
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
    if hamiltonian is None:
        raise ToolInputError(
            "hamiltonian is required. Without an objective the value the SDK "
            "reports is not an energy, and training it would minimise an "
            "unnamed quantity to convergence and hand back a plausible number. "
            'Give the Pauli sum to minimise: [{"pauli": "ZZ", "coefficient": '
            '1.0}, {"pauli": "XI", "coefficient": -0.5}].'
        )
    if isinstance(hamiltonian, (list, tuple)) and not hamiltonian:
        raise ToolInputError(
            "hamiltonian is empty. A training objective needs at least one "
            'term, such as [{"pauli": "ZZ", "coefficient": 1.0}].'
        )
    try:
        return hamiltonian_from_terms(hamiltonian, n_wires=n_wires)
    except ToolInputError as exc:
        raise ToolInputError(str(exc).replace("'terms'", "'hamiltonian'")) from exc
```

Change `train_parameters` to call `_objective(hamiltonian, n_wires=int(ir.n_wires))`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../.venv/bin/pytest tests/test_training.py -q`
Expected: PASS, 44 tests.

- [ ] **Step 5: Mutation-test each refusal**

Each refusal is a guard clause. The mutation is to delete it and watch exactly one test go red. Run this loop and read the output rather than trusting it:

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
import re, subprocess, sys
from pathlib import Path

path = Path("src/flagquantum_mcp_server/training.py")
original = path.read_text()

# (label, what to delete, the test that must notice)
mutations = [
    ("steps guard", "    _check_steps(steps)\n",
     "test_a_step_count_the_loop_cannot_use_is_refused"),
    ("learning-rate guard", "    _check_learning_rate(learning_rate)\n",
     "test_a_learning_rate_adam_cannot_use_is_refused"),
    ("name guard", "    _check_names(names)\n",
     "test_a_circuit_with_no_parameters_is_refused_by_name"),
    ("observables guard", "    reject_circuit_observables(\n        ir,\n",
     "test_a_circuit_carrying_observables_is_refused_and_points_at_hamiltonian"),
]

for label, target, test in mutations:
    if target not in original:
        print(f"{label}: TARGET NOT FOUND — fix the mutation script")
        continue
    path.write_text(original.replace(target, "", 1))
    result = subprocess.run(
        ["../.venv/bin/pytest", "tests/test_training.py", "-k", test, "-q"],
        capture_output=True, text=True,
    )
    print(f"{label}: {'CAUGHT' if result.returncode != 0 else 'MISSED'}")
    path.write_text(original)

assert path.read_text() == original, "restore failed"
PY
```

Expected: every line reads `CAUGHT`. A `MISSED` means that guard is not what the test is checking. A `TARGET NOT FOUND` means the mutation did not apply and the result would have been meaningless — the same failure that hid a broken test in an earlier session.

- [ ] **Step 6: Run every gate**

```bash
../.venv/bin/ruff check .
../.venv/bin/ruff format --check .
../.venv/bin/mypy --config-file ../mypy.ini src
../.venv/bin/pytest -m "not integration"
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add flagquantum-mcp-server/src/flagquantum_mcp_server/training.py \
        flagquantum-mcp-server/tests/test_training.py
git commit -m "feat: refuse the seven inputs that would train the wrong thing quietly"
```

---

### Task 9: Publish the tool

Registering the tool is what makes it real: a module nobody can call is not a feature, and this repository has shipped one before.

**Files:**
- Modify: `flagquantum-mcp-server/src/flagquantum_mcp_server/server.py`
- Modify: `flagquantum-mcp-server/tests/surface.py`
- Modify: `flagquantum-mcp-server/tests/test_tool_wiring.py`
- Test: `flagquantum-mcp-server/tests/test_server_contract.py`, `flagquantum-mcp-server/tests/test_server_process.py`

**Interfaces:**
- Consumes: `training.train_parameters`.
- Produces: `server.train_parameters_tool`, and `train_parameters_tool` in `tests/surface.EXPECTED_TOOLS`.

- [ ] **Step 1: Write the failing test**

Add to `flagquantum-mcp-server/tests/surface.py`, in `EXPECTED_TOOLS`, keeping the alphabetical order the set already has:

```python
        "train_parameters_tool",
```

Run: `../.venv/bin/pytest tests/test_server_contract.py tests/test_server_process.py -q`
Expected: FAIL — the tools the server serves are now one short of the declared surface.

- [ ] **Step 2: Register the tool**

In `server.py`, add `train_parameters` to the training import and the tool below `simulate_circuit_tool`:

```python
from flagquantum_mcp_server.training import train_parameters
```

```python
@mcp.tool(annotations=READ_ONLY)
@_structured_errors
def train_parameters_tool(
    circuit: str,
    hamiltonian: Sequence[Mapping[str, Any]],
    circuit_format: CircuitFormat = "ir",
    values: Mapping[str, float] | None = None,
    steps: int = 50,
    learning_rate: float = 0.1,
) -> dict[str, Any]:
    """Optimize a parameterized circuit's angles against a Pauli-sum energy.

    This is the gradient step that simulate_circuit_tool cannot take. That tool
    evaluates an energy; this one improves it. Fifty calls to simulate, each one
    a guess, is a manual scan; one call here is a descent, because the SDK
    reports exact gradients for a statevector simulation and this tool uses
    them.

    The objective is the Hamiltonian expectation, and it has to be given: the
    "hamiltonian" argument takes the same term list an "expectation" output
    takes, one letter per wire. There is no default, because without an
    objective the number being minimised is not an energy.

    Runs in this process: a statevector simulation on CPU, no network, no
    credentials, no provider. The result carries the SDK's provenance and its
    accuracy contract, which reports metric "not_measured" — an ideal
    simulation is not evidence about hardware.

    Refused before starting: a circuit with no parameters, a missing or
    malformed objective, a starting value that names a parameter the circuit
    does not have or omits one it does, a step count or learning rate the
    optimizer cannot use, a circuit carrying "observables" (never read — put the
    objective in "hamiltonian"), and any run whose predicted cost is past the
    budget. The prediction is an estimate, not a measurement of your machine.

    Adam at the learning rate you set. Losses are reported one per step, so a
    caller can see whether the run is still moving; whether it has converged is
    your reading, not this tool's claim.

    Args:
        circuit: Circuit payload carrying parameters. With circuit_format="qir"
            pass a gate list such as '[{"name": "ry", "index": [0],
            "parameters": {"theta": {"$parameter": "t0"}}}]'. With
            circuit_format="ir" pass FlagQuantum IR JSON. OpenQASM is not
            accepted.
        hamiltonian: The objective, as a list of {"pauli": ..., "coefficient":
            ...} objects, one letter per wire: [{"pauli": "ZZ", "coefficient":
            1.0}, {"pauli": "XI", "coefficient": -0.5}]. Required.
        circuit_format: "ir" for FlagQuantum IR JSON, "qir" for a gate list.
        values: Starting value for each parameter, by name: {"t0": 0.1}. Every
            parameter needs one, and no others are accepted. Defaults to zeros.
        steps: Number of optimizer updates. Default 50.
        learning_rate: Adam's learning rate. Default 0.1.

    Returns:
        The loss trajectory, the parameters the run ended on, the parameters it
        started from, the circuit identity and the execution provenance. Feed
        "parameters" back as "values" to continue a run rather than restart it.
    """
    return train_parameters(
        circuit,
        hamiltonian,
        circuit_format,
        values=values,
        steps=steps,
        learning_rate=learning_rate,
    )
```

- [ ] **Step 3: Run the surface tests to verify they pass**

Run: `../.venv/bin/pytest tests/test_server_contract.py tests/test_server_process.py -q`
Expected: PASS.

- [ ] **Step 4: Extend the wiring test**

`tests/test_tool_wiring.py` exists so that a handler which stops forwarding an argument is caught: the JSON schema still advertises the argument and still validates, so the client keeps being offered an option that does nothing. Add a check and a case.

Add near the other checks:

```python
def _trained(payload: dict[str, Any]) -> None:
    assert payload["completed_steps"] == 7, "steps must reach the loop"
    assert payload["circuit"]["n_qubits"] == 2
    assert payload["initial_parameters"] == {"t0": 0.25, "t1": -0.25}, (
        "values must reach the optimizer"
    )
    assert payload["final_loss"] < payload["initial_loss"], (
        "learning_rate must reach the optimizer"
    )
```

Add a case to `CASES`, using a two-qubit parameterized gate list and a two-term objective:

```python
    "train_parameters_tool": [
        (
            {
                "circuit": TRAINABLE,
                "hamiltonian": [
                    {"pauli": "ZZ", "coefficient": -1.0},
                    {"pauli": "XI", "coefficient": 1.0},
                    {"pauli": "IX", "coefficient": 1.0},
                ],
                "circuit_format": "qir",
                "values": {"t0": 0.25, "t1": -0.25},
                "steps": 7,
                "learning_rate": 0.4,
            },
            _trained,
        )
    ],
```

Add the circuit with the other module-level constants near `SYMBOLIC`:

```python
TRAINABLE = json.dumps(
    [
        {"name": "ry", "index": [0], "parameters": {"theta": {"$parameter": "t0"}}},
        {"name": "ry", "index": [1], "parameters": {"theta": {"$parameter": "t1"}}},
        {"name": "cx", "index": [0, 1]},
    ]
)
```

- [ ] **Step 5: Run the wiring test**

Run: `../.venv/bin/pytest tests/test_tool_wiring.py -q`
Expected: PASS. `test_every_tool_argument_is_covered` compares the cases against the published schema, so a missed argument fails here.

- [ ] **Step 6: Mutation-test the wiring**

The wiring test's whole purpose is to notice a dropped argument. Confirm it does:

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path
p = Path("src/flagquantum_mcp_server/server.py")
before = p.read_text()
after = before.replace(
    "        values=values,\n        steps=steps,\n        learning_rate=learning_rate,\n",
    "        steps=steps,\n        learning_rate=learning_rate,\n",
)
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_tool_wiring.py -q
cd "$(git rev-parse --show-toplevel)" && git checkout flagquantum-mcp-server/src/flagquantum_mcp_server/server.py
```

Expected: FAIL on the `train_parameters_tool` case, because the starting values no longer reach the optimizer.

- [ ] **Step 7: Run every gate, including the two CI jobs the four commands do not cover**

```bash
../.venv/bin/ruff check .
../.venv/bin/ruff format --check .
../.venv/bin/mypy --config-file ../mypy.ini src
../.venv/bin/pytest -m "not integration"
../.venv/bin/pytest -m integration
../.venv/bin/python examples/stdio_client.py
```

Expected: all pass.

- [ ] **Step 8: Rehearse the wheel job, in a clean environment**

```bash
../.venv/bin/python -m build
python3 -m venv /tmp/wheelcheck
/tmp/wheelcheck/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/wheelcheck/bin/python -m pip install dist/*.whl
/tmp/wheelcheck/bin/python -c "
import asyncio
from fastmcp import Client
from flagquantum_mcp_server.server import mcp

async def main():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print(sorted(t.name for t in tools))

asyncio.run(main())
"
/tmp/wheelcheck/bin/python -c "import pytest"   # must FAIL
```

Expected: the tool list includes `train_parameters_tool`, and the `import pytest` line fails — that is the check that the artifact does not depend on the test framework, which is the one the 0.2.0 release got wrong.

- [ ] **Step 9: Commit**

```bash
git add flagquantum-mcp-server/src/flagquantum_mcp_server/server.py \
        flagquantum-mcp-server/tests/surface.py \
        flagquantum-mcp-server/tests/test_tool_wiring.py
git commit -m "feat: publish train_parameters_tool"
```

---

### Task 10: Pin the contract and document it

Two jobs that are cheap together and expensive apart: a tier-3 dependency nothing pins, and a README that describes a tool no test exercises.

**Files:**
- Modify: `flagquantum-mcp-server/tests/test_api_contract.py`
- Modify: `flagquantum-mcp-server/README.md`
- Modify: `AGENTS.md`

**Interfaces:**
- Consumes: the tier-3 names `flagquantum.algorithms.Hamiltonian`, `flagquantum.algorithms.pauli_term` and `flagquantum.RuntimePolicy`.
- Produces: no runtime names.

- [ ] **Step 1: Write the failing contract test**

Add to `flagquantum-mcp-server/tests/test_api_contract.py`:

```python
# Tier 3: public names a package declares in its own __all__, not in the frozen
# snapshot. The training objective needs both, and there is no other way to say
# what to minimise — see docs/superpowers/specs/2026-09-18-parameter-training-design.md.
TIER3_TRAINING = (
    ("flagquantum.algorithms", "Hamiltonian"),
    ("flagquantum.algorithms", "pauli_term"),
)


@pytest.mark.parametrize(("module_path", "attribute"), TIER3_TRAINING)
def test_training_dependencies_are_still_declared_public(
    module_path: str, attribute: str
) -> None:
    module = importlib.import_module(module_path)

    assert attribute in module.__all__, f"{module_path} no longer declares {attribute}"
    assert hasattr(module, attribute)


def test_the_policy_that_makes_a_hamiltonian_reach_the_objective_still_exists() -> None:
    """Without this object the SDK evaluates ⟨Z₀⟩ and ignores the Hamiltonian.

    Pinned by name and by the one attribute that matters, because a rename here
    would not fail anything else: the tool would keep training, keep converging
    and keep reporting an energy it never measured.
    """
    assert hasattr(fq, "RuntimePolicy")
    assert fq.RuntimePolicy().observable == "z"
    assert fq.RuntimePolicy(observable="hamiltonian").observable == "hamiltonian"


def test_the_module_members_the_training_tool_reads_still_exist() -> None:
    for attribute in ("named_parameter_groups", "execute", "parameters", "named_parameters"):
        assert hasattr(fq.Module, attribute), f"Module.{attribute} disappeared"
```

- [ ] **Step 2: Run it to verify it passes on the current SDK and would fail on a moved one**

Run: `../.venv/bin/pytest tests/test_api_contract.py -q`
Expected: PASS. Then confirm the pin bites:

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path
p = Path("tests/test_api_contract.py")
before = p.read_text()
after = before.replace('("flagquantum.algorithms", "Hamiltonian"),', '("flagquantum.algorithms", "HamiltonianX"),')
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_api_contract.py -k training_dependencies -q
git checkout tests/test_api_contract.py
```

Expected: FAIL before the restore.

- [ ] **Step 3: Update the README**

Four edits. First, the opening sentence. The server publishes sixteen tools
before this change and seventeen after, of which two execute rather than one.
Confirm the count before writing the number:

```bash
../.venv/bin/python -c "
import asyncio
from flagquantum_mcp_server.server import mcp
print(len(asyncio.run(mcp.list_tools())))
"
```

Then write:

```
Seventeen tools over stdio. Fifteen of them only read the circuit they are
given; the other two run it, one to measure it and one to improve it.
```

Second, a row in the execution table, after the `simulate_circuit_tool` row:

```markdown
| `train_parameters_tool` | The angles that lower a circuit's energy against a Pauli-sum Hamiltonian |
```

Third, a section after "Measuring an energy":

```markdown
### Lowering an energy

`simulate_circuit_tool` evaluates an energy. `train_parameters_tool` improves
one, using the exact gradients a statevector simulation reports:

```json
{
  "circuit": "[{\"name\": \"ry\", \"index\": [0], \"parameters\": {\"theta\": {\"$parameter\": \"t0\"}}}]",
  "circuit_format": "qir",
  "hamiltonian": [{"pauli": "ZZ", "coefficient": -1.0}, {"pauli": "XI", "coefficient": 1.0}],
  "steps": 100,
  "learning_rate": 0.1
}
```

It returns the loss after each step, the parameters it ended on, and the
parameters it started from. Call it again with `values` set to the `parameters`
it returned to continue the run rather than restart it.

The `hamiltonian` argument is the same term list an `expectation` output takes,
and it is required: without an objective the number being minimised is not an
energy.

Two limits are worth knowing before you call it. Training is far more expensive
than simulating — measured, a 16-qubit step costs 85 ms against a simulation's
few milliseconds — so a run is refused up front when its predicted cost exceeds
`FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS`, with the prediction, the width and the step
count in the message. At 24 qubits a single step is 68 seconds, which means the
budget refuses every run at that width; train narrower circuits and simulate
wide ones.

And the result is what it is: a loss curve and a set of angles. Whether the run
converged is your reading, not this tool's claim, and the SDK's
`accuracy.metric == "not_measured"` travels with the execution exactly as it
does for a simulation.
```

Fourth, two table rows. In the limits table:

```markdown
| `FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS` | 60 | Predicted cost of one training call |
```

And in the tier table's third row, extending the existing list:

```markdown
| 3. Public but not frozen | …, `flagquantum.algorithms.{Hamiltonian, pauli_term}`, `flagquantum.RuntimePolicy` | train_parameters |
```

- [ ] **Step 4: Update `AGENTS.md`**

Extend rule 4's in-process paragraph so it covers the tool that trains as well as the one that simulates, and add the training budget to the Verification section's environment list. Specifically, after the existing `simulate_circuit_tool` paragraph, add:

```markdown
   `train_parameters_tool` is inside the same line. The gradients it uses are
   the SDK's exact gradients for a statevector simulation, computed in this
   process from a circuit the caller supplied — no target, no token, no
   hardware. What it adds is a way to *lower* an energy rather than read one,
   which is why it carries the same provenance block and the same
   `accuracy.metric == "not_measured"` as a simulation: a converged loss curve
   is not evidence about any machine either.
```

And in the Conventions section, after the prompt-channel paragraph, add:

```markdown
- **A silent default is a wrong answer waiting.** `Module(hamiltonian=H)`
  accepts a Hamiltonian, stores it, exposes it as `.hamiltonian`, and evaluates
  `⟨Z₀⟩` instead — the observable is chosen by `RuntimePolicy.observable`, whose
  default is `z`. Nothing warns. The training tool sets the policy explicitly
  and a test asserts the Hamiltonian reaches the objective. When a public API
  takes an argument it does not act on, assume the same shape elsewhere: find
  the second object that decides, and set it.
```

- [ ] **Step 5: Check the documented commands still all resolve**

Run: `../.venv/bin/pytest tests/test_documented_commands.py -q`
Expected: PASS. This test parses the bash blocks under `## Verification` in `AGENTS.md` and asserts every relative path resolves, so an edit that adds an unrunnable command fails here.

- [ ] **Step 6: Run every gate**

```bash
../.venv/bin/ruff check .
../.venv/bin/ruff format --check .
../.venv/bin/mypy --config-file ../mypy.ini src
../.venv/bin/pytest -m "not integration"
../.venv/bin/pytest -m integration
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add flagquantum-mcp-server/tests/test_api_contract.py \
        flagquantum-mcp-server/README.md \
        AGENTS.md
git commit -m "docs: pin the training dependencies and describe the tool"
```

Do **not** bump the version and do not release. AGENTS.md rule 8: accumulate on `main` and release when someone outside the repository would benefit — that decision is separate and needs explicit authorization.

---

## What this plan does not cover

- **A real agent session against the new tool.** Four sessions drove the design; a fifth should drive the finished tool, with no prior FlagQuantum knowledge and no filesystem, and its transcript is the evidence that the docstring is usable. That is a follow-up, not a task here, because it needs the tool merged to be worth running.
- **Upstream.** The `RuntimePolicy` behaviour — an accepted argument that is silently ignored — is a FlagQuantum issue, not an MCP-server one. Rule 1 forbids adding an MCP dependency to that repository; reporting the behaviour is not the same thing and is worth doing separately.
- **Noise models, distributed training, optimizer choice, convergence claims.** Each is argued in the design document's "What this deliberately does not do".
