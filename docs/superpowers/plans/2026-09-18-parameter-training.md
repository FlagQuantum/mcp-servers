# Training a circuit's parameters — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `train_parameters_tool`, which optimizes a parameterized circuit's angles against a Pauli-sum energy in one call, replacing the fifty-call manual scan four recorded agent sessions converged on.

**Architecture:** A new `training.py` holds a replay layer that turns a serialized circuit back into the Python callable `fq.Module` traces, a budget model that decides before any work starts, and the tool body. Two existing modules are refactored first so that shared refusals and the shared `terms` vocabulary have one home each rather than two.

**Tech Stack:** Python 3.10–3.12, FastMCP 3.x, FlagQuantum 0.2.x, torch (reached through the SDK), pytest.

**Spec:** `docs/superpowers/specs/2026-09-18-parameter-training-design.md`

## Global Constraints

- **Which directory a command runs in.** Every task assumes the shell starts
  each command in the **repository root**. `git add flagquantum-mcp-server/...`
  is written for that directory. The mutation and verification blocks below open
  with `cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"` rather
  than `cd flagquantum-mcp-server`, so they work from either.
- **Regenerate the brief before writing any dispatch that names one.** The brief
  files under `.superpowers/sdd/<plan>/` are extracts, not sources: they are
  produced by `scripts/task-brief PLAN_FILE N` and they go stale the moment the
  plan is edited. A dispatch that says "the amended brief" while pointing at an
  extract taken before the amendment hands over exactly the text the amendment
  removed — and the implementer, seeing plan and brief disagree, has to guess
  which is authoritative. Round 3 of Task 7 lost time to this twice.
  Run the extractor immediately before composing the dispatch, every time.
- **Mutations are reverted by moving a sidecar back, never by `git checkout`.**
  Each mutation script writes `Path(str(p) + ".mutbak").write_text(before)`
  before it breaks anything, and the restore replaces the file from that
  sidecar: `pathlib.Path(...).replace(...)`, relative to the directory the
  mutation block cd'd into. Not `mv`, which this sandbox
  refuses when compounded after a `cd`; and not `git checkout <path>`, which
  reverts the file to HEAD. Every mutation step runs *before* its task's commit,
  so a checkout would discard the task's own work along with the mutation,
  leaving a tree that is neither mutated nor implemented. If you see a `.mutbak` file still on disk at
  the end of a task, the restore did not run.
- **A restore is not complete until `__pycache__` is cleared, and this was
  measured rather than reasoned.** `Path.replace` gives the restored file the
  *sidecar's* mtime, and a `.pyc` records its source's mtime truncated to whole
  seconds. A length-preserving mutation restored inside that same second
  therefore matches the mutant `.pyc` and is a cache hit on it: the source on
  disk reads correctly while the interpreter keeps running the mutant. Probed
  directly — source `VALUE = 111`, mutant `999`, restore, `reload` → `999`.
  So run `find . -name __pycache__ -type d -exec rm -rf {} +` immediately after
  every `.replace(...)`, and before you believe any red or green that follows a
  restore. This is the fourth distinct way a mutation has been left applied
  while appearing reverted; the other three are above.
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
```

**`_pauli_observable` is not reproduced here and is not edited.** Its
signature keeps `where: str | None = None` and its body keeps every check it
has today, including the `subject` line that reads it. Open the file and leave
the function as you find it. Two reasons, and the second is the one that
matters:

1. It is already correct. The `where` parameter arrived with `terms` in 0.2.0,
   and the single-pauli path through `_build_output` depends on it.
2. A `terms` entry now passes two checks: `_weighted_term`'s, which names
   `terms[i]`, and this one, which fires only if the first did not. The second
   cannot fire for a `terms` call. It costs one length comparison and it keeps
   the refusal inside the function that owns the string, which is worth more
   than the branch it saves. The Step 4 mutation is what shows the outer check
   is the one carrying the location, and that is the only observable
   difference between them.

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
Path(str(p) + ".mutbak").write_text(before)
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
python3 -c "import pathlib; pathlib.Path('src/flagquantum_mcp_server/planning.py.mutbak').replace(pathlib.Path('src/flagquantum_mcp_server/planning.py'))"
find . -name __pycache__ -type d -exec rm -rf {} +
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
Path(str(p) + ".mutbak").write_text(before)
after = before.replace(
    "    if not observables:\n        return\n",
    "    if not observables:\n        return\n    return\n",
)
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_simulation.py -q
python3 -c "import pathlib; pathlib.Path('src/flagquantum_mcp_server/preconditions.py.mutbak').replace(pathlib.Path('src/flagquantum_mcp_server/preconditions.py'))"
find . -name __pycache__ -type d -exec rm -rf {} +
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
Path(str(p) + ".mutbak").write_text(before)
after = before.replace(
    'return _positive_int("FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS", DEFAULT_MAX_TRAIN_SECONDS)',
    'return int(os.environ.get("FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS", DEFAULT_MAX_TRAIN_SECONDS))',
)
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_api_contract.py -k training_budget -q
python3 -c "import pathlib; pathlib.Path('src/flagquantum_mcp_server/limits.py.mutbak').replace(pathlib.Path('src/flagquantum_mcp_server/limits.py'))"
find . -name __pycache__ -type d -exec rm -rf {} +
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


def _numeric(value: Any) -> float:
    """Read a serialized gate argument as a number.

    A replayed symbol is a tensor, so the SDK serializes it as ``$tensor``
    with the value under ``data`` — a scalar for the one-element group the
    builder indexes, a one-element list if the SDK ever encodes the group
    whole. A bound angle is a plain number. Reading either keeps these
    assertions about the value bound, rather than about the encoding the
    SDK chose for it.
    """
    if isinstance(value, Mapping) and "$tensor" in value:
        data = value["$tensor"]["data"]
        return float(data[0] if isinstance(data, list) else data)
    return float(value)


# --- the replay reproduces the circuit it was given ---


def test_a_replay_of_a_numeric_circuit_reproduces_its_instructions() -> None:
    """Scoped to the instructions on purpose.

    The replay reproduces the instruction list and the width; it does not
    reproduce dtype, shape or metadata, which come from a freshly default
    constructed ``Circuit``. Measured: a ``complex128`` source gives equal
    instructions and an unequal envelope. Byte-equality of the whole payload
    would pass only because this source is default-constructed.

    And it is not the property that matters. A replay built from
    ``to_dict()`` was byte-for-byte equal on this very circuit and could not
    execute; the test below is the one that runs it.
    """
    import flagquantum as fq

    source = fq.Circuit(2).h(0).ry(1, theta=0.7).cx(0, 1)
    source.gate("any", [1], matrix=[[1, 0], [0, 1j]])
    source.rz(0, theta=-0.25)
    ir = source.to_ir()

    rebuilt = replay_builder(ir)({})

    assert rebuilt.to_ir().to_dict()["instructions"] == ir.to_dict()["instructions"]


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

    assert _numeric(payload[0]["params"]["theta"]) == pytest.approx(0.75)
    assert _numeric(payload[1]["params"]["theta"]) == pytest.approx(0.25)


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
    symbol = payload[1]["params"]["theta"]
    assert isinstance(symbol, Mapping) and "$tensor" in symbol
    assert _numeric(symbol) == pytest.approx(0.25)


def test_a_replayed_circuit_executes_and_agrees_with_its_source() -> None:
    """The claim byte-equality cannot make: the rebuild runs, and runs the same.

    Measured before this test existed: a replay built from ``to_dict()`` was
    byte-for-byte equal to its source on this same circuit and failed to
    execute. Serializing correctly is not the property that matters — producing
    the same numbers is, and only running it shows that.
    """
    import flagquantum as fq

    source = fq.Circuit(2).h(0).ry(1, theta=0.7).cx(0, 1)
    source.gate("any", [1], matrix=[[1, 0], [0, 1j]])
    ir = source.to_ir()

    rebuilt = replay_builder(ir)({})

    want = fq.run(source, outputs=[fq.probabilities([0, 1])]).measurements[0].value
    got = fq.run(rebuilt, outputs=[fq.probabilities([0, 1])]).measurements[0].value

    assert got == pytest.approx(want, abs=1e-9)


def test_a_parameter_inside_an_expression_is_substituted() -> None:
    """A name that appears only inside an expression is still a name to train.

    ``parameter_names`` reports it either way, so the failure is silent until
    the run: the tool would offer a trainable group it never applies.
    """
    import torch

    qir = json.dumps(
        [
            {"name": "ry", "index": [0], "parameters": {"theta": {"$parameter": "t0"}}},
            {
                "name": "rz",
                "index": [1],
                "parameters": {
                    "theta": {"$expression": {"op": "mul", "args": [2.0, {"$parameter": "t1"}]}}
                },
            },
        ]
    )
    ir = _ir(qir)

    assert parameter_names(ir) == ("t0", "t1")

    built = replay_builder(ir)({"t0": torch.tensor([0.3]), "t1": torch.tensor([0.4])})

    assert built.is_parameterized() is False, "the rebuild still carries symbols"
    assert _numeric(built.to_ir().to_dict()["instructions"][1]["params"]["theta"]) == pytest.approx(
        0.8
    )


def test_a_gradient_flows_through_an_expression_to_the_name_inside_it() -> None:
    """The expression path is optimized, not merely evaluated once.

    A ``ParameterExpression`` resolves through the SDK's own ``bind``, so the
    question a test has to answer is whether the tensor that comes back is still
    attached to the graph ``Module`` optimizes. Measuring the gradient against
    its analytic value answers it: for ``ry(1, t0)`` then ``rz(1, 2*t1)``
    against ``Y`` on wire 1, ``<Y> = sin(t0) sin(2 t1)``, so ``d/dt1`` is
    ``2 sin(t0) cos(2 t1)``.
    """
    import math

    import flagquantum as fq
    from flagquantum.algorithms import Hamiltonian, pauli_term

    qir = json.dumps(
        [
            {"name": "ry", "index": [1], "parameters": {"theta": {"$parameter": "t0"}}},
            {
                "name": "rz",
                "index": [1],
                "parameters": {
                    "theta": {"$expression": {"op": "mul", "args": [2.0, {"$parameter": "t1"}]}}
                },
            },
        ]
    )
    ir = _ir(qir)
    module = fq.Module(
        replay_builder(ir),
        parameters=dict.fromkeys(parameter_names(ir), 1),
        init={"t0": 0.8, "t1": 0.3},
        hamiltonian=Hamiltonian([pauli_term(1.0, {1: "Y"})]),
        policy=fq.RuntimePolicy(observable="hamiltonian"),
    )

    loss = module.execute().require_value().mean()
    loss.backward()

    groups = module.named_parameter_groups
    assert float(loss.detach()) == pytest.approx(math.sin(0.8) * math.sin(0.6), abs=1e-6)
    assert float(groups["t1"].grad) == pytest.approx(2 * math.sin(0.8) * math.cos(0.6), abs=1e-5)


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

from collections.abc import Callable, Mapping
from typing import Any

from flagquantum_mcp_server._bridge import load_sdk
from flagquantum_mcp_server.circuits import circuit_from_ir


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

    A channel instruction's ``is_channel`` flag lives in its ``metadata``, and
    ``Circuit.gate`` takes no metadata, so the replay does not carry it. Measured,
    a channel circuit fails to execute on this path either way, so no number a
    caller sees changes — but the replay is not faithful for one and does not
    claim to be.

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


def _argument(value: Any, parameters: Mapping[str, Any], sdk: Any) -> Any:
    """Resolve one live gate argument against the parameter mapping.

    A ``Parameter`` becomes the tensor ``Module`` supplied for that name. A
    ``ParameterExpression`` — ``2.0 * t0`` — is resolved through the SDK's own
    public ``bind``, so the arithmetic stays the SDK's rather than being
    reimplemented here; this module carries no numerical logic and should not
    acquire any.

    Every other value is already a number, a complex, or an object the SDK
    built, and passes through untouched.

    ``[0]`` indexes the one-element group ``Module`` creates per name. A group
    of any other size would be a name bound to a vector, which this tool never
    asks for; the construction site is where that shape is asserted.

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../.venv/bin/pytest tests/test_training.py -q`
Expected: PASS, 11 tests in this file. (The task's block grew from 8 to 11
during its review — see the execution and expression tests below — so a count
taken from an earlier draft of this plan will be three short.)

- [ ] **Step 5: Mutation-test each new claim**

Run each mutation, watch it go red, restore from the sidecar. The
`assert after != before` line is not decoration: a mutation whose search text
does not match changes nothing, and that is indistinguishable from a test that
did not notice.

**Mutation 1 — put the encoded read back.** This is the one that justifies the
whole design, and its result is the argument: it must turn the *execution* test
red while the byte-equality assertion stays green under it.

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path

p = Path("src/flagquantum_mcp_server/training.py")
before = p.read_text()
Path(str(p) + ".mutbak").write_text(before)
after = (
    before.replace(
        "    instructions = tuple(ir.instructions)",
        '    instructions = tuple(ir.to_dict()["instructions"])',
    )
    .replace(
        "                str(instruction.name),",
        '                str(instruction["opcode"]),',
    )
    .replace(
        "                [int(wire) for wire in instruction.wires],",
        '                [int(wire) for wire in instruction["wires"]],',
    )
    .replace(
        "                matrix=instruction.matrix,",
        '                matrix=instruction.get("matrix"),',
    )
    .replace(
        "                for key, value in instruction.params.items()",
        '                for key, value in (instruction.get("params") or {}).items()',
    )
    .replace(
        "                str(key): _argument(value, parameters, sdk)",
        "                str(key): value",
    )
)
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_training.py -q
```

Expected, and measured: **5 failed, 5 passed**.
`test_a_replayed_circuit_executes_and_agrees_with_its_source` is the first red,
with `ExecutionError: planned execution failed` — and
`test_a_replay_of_a_numeric_circuit_reproduces_its_instructions` stays **green**
under it. That contrast is the finding the review produced: byte-equality was
satisfied by a circuit that cannot run, which is why the execution test exists.

Restore, then the second mutation:

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 -c "import pathlib; pathlib.Path('src/flagquantum_mcp_server/training.py.mutbak').replace(pathlib.Path('src/flagquantum_mcp_server/training.py'))"
find . -name __pycache__ -type d -exec rm -rf {} +
```

**Mutation 2 — drop the expression branch.**

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path

p = Path("src/flagquantum_mcp_server/training.py")
before = p.read_text()
Path(str(p) + ".mutbak").write_text(before)
after = before.replace(
    "    if isinstance(value, sdk.ParameterExpression):\n"
    "        return value.bind({name: group[0] for name, group in parameters.items()})\n",
    "",
)
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_training.py -q
python3 -c "import pathlib; pathlib.Path('src/flagquantum_mcp_server/training.py.mutbak').replace(pathlib.Path('src/flagquantum_mcp_server/training.py'))"
find . -name __pycache__ -type d -exec rm -rf {} +
```

Expected: FAIL on `test_a_parameter_inside_an_expression_is_substituted` and
`test_a_gradient_flows_through_an_expression_to_the_name_inside_it` — measured
as 1 failed, 9 passed before the gradient test was added, 2 failed after. A name
appearing only inside an expression stops being substituted, and
`parameter_names` never reports it: the tool would advertise a trainable group it
never applies.

**Mutation 3 — bind the expression eagerly instead of through the SDK.**

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path

p = Path("src/flagquantum_mcp_server/training.py")
before = p.read_text()
Path(str(p) + ".mutbak").write_text(before)
after = before.replace(
    "        return value.bind({name: group[0] for name, group in parameters.items()})",
    "        return value.bind({name: float(group) for name, group in parameters.items()})",
)
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_training.py -q
python3 -c "import pathlib; pathlib.Path('src/flagquantum_mcp_server/training.py.mutbak').replace(pathlib.Path('src/flagquantum_mcp_server/training.py'))"
find . -name __pycache__ -type d -exec rm -rf {} +
```

Expected, and measured: **1 failed, 10 passed** — only the gradient test, while
`test_a_parameter_inside_an_expression_is_substituted` stays green in isolation.
That is what proves the gradient test is not redundant with it. Note *how* it
goes red: `make_fx` refuses the builder at trace time ("Module builder
compilation requires static circuit topology"), not on the gradient comparison.
The failure is loud, so the test's own docstring says so rather than claiming a
silent one — see the paragraph it carries.

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
  - `training.STATE_COST: float` = 1.5e-6
  - `training.GATE_COST: float` = 1e-7

**This task's code blocks below are superseded.** The first version of the model
was one width-independent floor plus one coefficient, and review measured it
under-predicting by up to 105x in the low-instruction region — a 24-wire,
one-gate call it admitted for 35 steps took 438 s. The shipped shape has three
terms and the constants above; **the shipped source is the authority for this
task, not the code quoted here.** The steps are kept because the calibration
table and the mutation targets they describe are still the ones in force.

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
    """An ry/cx/rz ansatz as a gate list, one ``$parameter`` per rotation.

    Written with ``extend`` rather than a loop of ``append`` because ruff's
    PERF401 says so, and ``tests/**`` in the shared config ignores only S101,
    PLR2004 and SLF001.
    """
    gates: list[dict[str, object]] = []
    for layer in range(layers):
        gates.extend(
            {
                "name": "ry",
                "index": [wire],
                "parameters": {"theta": {"$parameter": f"t{layer}_{wire}"}},
            }
            for wire in range(n_wires)
        )
        gates.extend({"name": "cx", "index": [wire, wire + 1]} for wire in range(n_wires - 1))
        gates.extend(
            {
                "name": "rz",
                "index": [wire],
                "parameters": {"theta": {"$parameter": f"u{layer}_{wire}"}},
            }
            for wire in range(n_wires)
        )
    return gates
```

- [ ] **Step 2: Run them to verify they fail**

Run: `../.venv/bin/pytest tests/test_training.py -k prediction -q`
Expected: FAIL with `ImportError: cannot import name 'predict_seconds'`

- [ ] **Step 3: Write the model**

Do not write it from this plan. **The shipped `predict_seconds` is the authority
for this task.** The first version of the model — one width-independent floor plus
one coefficient — was measured under-predicting by up to 105x in the
low-instruction region, and every code block that described it has been removed
from this document rather than left where a reader could copy it. Read the
function and its constants in `training.py` before changing either.

What this step still owns is the shape, because it is what the tests below assert:

```
per_step = max(MIN_STEP_SECONDS,
               2 ** n_wires * STATE_COST,                   # holding the state
               n_instructions * 2 ** n_wires * GATE_COST)   # applying gates
```

Three terms. The floor is dispatch. The state term is the one the first model
lacked: at 24 wires, holding the state costs 12.5 s per step even with a single
gate, which is 7.5x what one instruction explains. The gate term is the marginal
cost of applying gates against that state.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../.venv/bin/pytest tests/test_training.py -q`
Expected: PASS. This step adds four tests; read the count off the run and check
it rose by four.

- [ ] **Step 5: Mutation-test the model's direction**

The claim is that the prediction is an upper bound everywhere it was calibrated.
Break each term downward and confirm the calibration test notices — one mutation
per term, each with its own guard, because a single mutation that drops two
constants at once proves neither.

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path
p = Path("src/flagquantum_mcp_server/training.py")
before = p.read_text()
Path(str(p) + ".mutbak").write_text(before)
after = before.replace("GATE_COST = 1e-7", "GATE_COST = 1e-9")
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_training.py -k over_predicts -q
python3 -c "import pathlib; pathlib.Path('src/flagquantum_mcp_server/training.py.mutbak').replace(pathlib.Path('src/flagquantum_mcp_server/training.py'))"
find . -name __pycache__ -type d -exec rm -rf {} +
```

Expected: FAIL, naming the rows whose prediction fell below their measurement.
Then the state term:

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path
p = Path("src/flagquantum_mcp_server/training.py")
before = p.read_text()
Path(str(p) + ".mutbak").write_text(before)
after = before.replace("STATE_COST = 1.5e-6", "STATE_COST = 1e-9")
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_training.py -k over_predicts -q
python3 -c "import pathlib; pathlib.Path('src/flagquantum_mcp_server/training.py.mutbak').replace(pathlib.Path('src/flagquantum_mcp_server/training.py'))"
find . -name __pycache__ -type d -exec rm -rf {} +
```

Expected: FAIL on the low-instruction rows — `(16, 3)`, `(20, 1)`, `(22, 1)`,
`(24, 1)`, `(24, 4)` — which is the whole reason the term exists. Then the floor:

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path
p = Path("src/flagquantum_mcp_server/training.py")
before = p.read_text()
Path(str(p) + ".mutbak").write_text(before)
after = before.replace("MIN_STEP_SECONDS = 0.02", "MIN_STEP_SECONDS = 0.0001")
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_training.py -k over_predicts -q
python3 -c "import pathlib; pathlib.Path('src/flagquantum_mcp_server/training.py.mutbak').replace(pathlib.Path('src/flagquantum_mcp_server/training.py'))"
find . -name __pycache__ -type d -exec rm -rf {} +
```

Expected: FAIL at the three floor-bound rows, 4, 8 and 12 wires.

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

`load_algorithms` calls `load_module`, so without it the module raises `NameError`
on the first import rather than on the first call — which is the failure a
collection error reports anyway, but fix it here rather than discovering it in
Step 4. The import becomes exactly these two names:

```python
from flagquantum_mcp_server._bridge import load_module, load_sdk
```

`load_torch` is deliberately **not** added here: this task's code does not call
it, `ruff`'s `F` selection makes an unused import an error, and Task 7 adds it in
the same edit that first uses it. Do not write an import ahead of its use.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../.venv/bin/pytest tests/test_training.py -q`
Expected: PASS, 20 tests. (Sixteen were there before this task; Step 1 adds four.
Count them in the run's own output rather than trusting this number.)

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
Path(str(p) + ".mutbak").write_text(before)
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
python3 -c "import pathlib; pathlib.Path('src/flagquantum_mcp_server/training.py.mutbak').replace(pathlib.Path('src/flagquantum_mcp_server/training.py'))"
find . -name __pycache__ -type d -exec rm -rf {} +
```

`_normalize_pauli` in the SDK filters identities out of its `ops` tuple, so the
printed ops may be identical either way. That is the finding: the `if letter !=
"I"` is **not** what keeps the wire numbering right, and a test asserting it
would be a test that passes for the wrong reason. Confirm the numbering a
different way — that `"IX"` puts its `X` on wire 1 and not wire 0:

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
../.venv/bin/python - <<'PY'
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

**Add that test unconditionally, whether or not the probe above passes.** The
probe is a one-shot check against today's SDK; the test is what keeps it true
after the next upgrade. Measured, reversing the wire-map enumeration leaves every
other test in the module green while `"IX"` lands on wire 0 — so without this
test the failure mode is a silently different Hamiltonian, not a red suite. A
probe that passes is not a reason to skip writing the test down; two tasks in
this plan have already lost a fix round to a check that was run once and never
pinned.

Then the mutation that matters most in this task. On this path
`hamiltonian_from_terms` calls `validate_pauli_terms` and builds through
`algorithms.pauli_term`; `planning._pauli_observable` is **not** involved. So
`_check_pauli_string` is the only thing standing between the caller and a term
the SDK will quietly accept. Break it and see what that buys:

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path
p = Path("src/flagquantum_mcp_server/planning.py")
before = p.read_text()
Path(str(p) + ".mutbak").write_text(before)
after = before.replace(
    "    _check_pauli_string(pauli, n_wires=n_wires, where=where)\n"
    "    return pauli.upper(), float(coefficient)",
    "    return pauli.upper(), float(coefficient)",
)
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_training.py -k "expectation_builder_refuses" -q
../.venv/bin/python -c "
from flagquantum_mcp_server.training import hamiltonian_from_terms
H = hamiltonian_from_terms([{'pauli': 'ZZ', 'coefficient': -1.0}, {'pauli': 'II', 'coefficient': 99.0}], n_wires=2)
print('objective built with an all-identity term:', H)
print('its ops:', [t.ops for t in H.terms])
"
python3 -c "import pathlib; pathlib.Path('src/flagquantum_mcp_server/planning.py.mutbak').replace(pathlib.Path('src/flagquantum_mcp_server/planning.py'))"
find . -name __pycache__ -type d -exec rm -rf {} +
```

Expected, and measured before this plan was written: the pytest run FAILS on
`test_a_term_that_the_expectation_builder_refuses_is_refused_here_too`, and the
second command **succeeds**, building a `HamiltonianTerm` whose `ops` is `()`.

**The `-k` expression must select exactly one test.** `-k` matches substrings of
the test name, and the name is
`test_a_term_that_the_expectation_builder_refuses_is_refused_here_too` — so a
filter spelled `refuses_the_expectation_builder` selects **nothing** and exits 5
with `21 deselected`, which looks like a result and is not one. Read the
selector's own line (`1 failed, 20 deselected` / `21 deselected`) before reading
anything into the outcome; an earlier draft of this step carried the
non-matching spelling and would have reported a red that never happened.

The SDK does not refuse an all-identity term — it accepts it and contributes
`coefficient × ⟨I⟩` to every energy. So without this check the training tool
would minimize a Hamiltonian with a constant the caller never wrote, converge
cleanly, and report the result as an energy. That is the same shape of failure
as the `RuntimePolicy` trap this design was rewritten around, which is why the
refusal is pinned by a test rather than left to the SDK.

- [ ] **Step 6: Pin the tier-3 dependency this task introduces**

`hamiltonian_from_terms` reaches two names through `flagquantum.algorithms`,
which is tier 3: public, but in no capability's `public_apis` and not in the
frozen snapshot. AGENTS.md rule 3 admits a tier-3 name when there is a reason
and a test pins it — so the pin belongs in the same change as the dependency,
not four tasks later. Add to `flagquantum-mcp-server/tests/test_api_contract.py`,
beside `TIER3_EMITTERS`:

```python
# Tier 3: public names a package declares in its own __all__, not in the frozen
# snapshot. The training objective needs both, and there is no other way to say
# what to minimise — see docs/superpowers/specs/2026-09-18-parameter-training-design.md.
TIER3_TRAINING = (
    ("flagquantum.algorithms", "Hamiltonian"),
    ("flagquantum.algorithms", "pauli_term"),
)
```

and, beside `test_emitter_submodules_still_expose_their_functions`:

```python
@pytest.mark.parametrize(("module_path", "attribute"), TIER3_TRAINING)
def test_training_dependencies_are_still_declared_public(module_path: str, attribute: str) -> None:
    module = importlib.import_module(module_path)

    assert attribute in module.__all__, f"{module_path} no longer declares {attribute}"
    assert hasattr(module, attribute)
```

Run: `../.venv/bin/pytest tests/test_api_contract.py -q`
Expected: PASS, and the new test appears twice, once per tuple.

- [ ] **Step 7: Commit**

```bash
git add flagquantum-mcp-server/src/flagquantum_mcp_server/_bridge.py \
        flagquantum-mcp-server/src/flagquantum_mcp_server/training.py \
        flagquantum-mcp-server/tests/test_training.py \
        flagquantum-mcp-server/tests/test_api_contract.py
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
TFIM2 = [
    {"pauli": "ZZ", "coefficient": -1.0},
    {"pauli": "XI", "coefficient": 1.0},
    {"pauli": "IX", "coefficient": 1.0},
]


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

    assert payload["final_loss"] == pytest.approx(-(5**0.5), abs=0.05)


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
    # The guard that keeps the assertion above from passing vacuously: if the
    # optimizer did nothing, initial and final would be the same number and the
    # equality would hold no matter what `final_loss` were computed from. This
    # is NOT the continuation claim — the equality above is — so it asserts only
    # that the resumed run improved.
    assert resumed["final_loss"] < resumed["initial_loss"]


def test_the_reported_parameters_are_what_the_next_call_starts_from() -> None:
    from flagquantum_mcp_server.training import train_parameters

    payload = train_parameters(ANGLED, TFIM2, "qir", steps=3, learning_rate=0.3)
    again = train_parameters(ANGLED, TFIM2, "qir", values=payload["parameters"], steps=1)

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
    # "16-qubit", not "16": the predicted seconds and the instruction count are
    # both in this message, and either could contain "16" by coincidence.
    assert "16-qubit" in message
    assert "5000" in message
    assert "FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS" in message


def test_a_run_inside_the_budget_is_not_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    from flagquantum_mcp_server.training import train_parameters

    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS", "1")

    assert train_parameters(ANGLED, TFIM2, "qir", steps=1)["status"] == "success"
```

Add `ToolLimitError` to the test module's error import.

`tests/test_training.py` already contains `test_a_learning_rate_adam_cannot_use_is_refused`
and `test_a_non_finite_starting_value_is_refused`. **Do not re-add either.**
Step 1 above does not list them, and they are here because a review of Task 7
found that a non-finite learning rate or starting value returned a `success`
envelope full of NaN — so they shipped with the validators they pin rather than a
task later, which is where a test belongs. Two `def`s with one name in a module
means pytest collects one and silently never runs the other.

Both shipped tests are recorded here, because the plan assigned them to no task
and a test that exists only in the file is one the next reader cannot check:

```python
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 10**400])
def test_a_non_finite_starting_value_is_refused(bad: object) -> None:
    """A starting value that is not a finite real reaches the optimizer otherwise.

    Measured: without this refusal, ``values={"t0": nan}`` returns
    ``status: "success"`` with every loss and every returned parameter NaN, and
    ``values={"t0": 10**400}`` returns an internal error. ``json.loads`` accepts
    the ``NaN`` and ``Infinity`` tokens and turns an integer literal into a
    Python ``int``, so no payload carrying one is stopped on the way in.
    """
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(ANGLED, TFIM2, "qir", values={"t0": bad, "t1": 0.1})

    assert "t0" in str(caught.value)
```

``bad: object`` rather than ``bad: float``, because the third case is an ``int``
and the first two are floats; ``object`` is the honest annotation, and it matches
the learning-rate test. mypy runs on ``src`` only, so no gate would catch the
wrong one.

All three cases are here on purpose and none is redundant. ``nan`` and ``inf``
exercise the ``isfinite`` test; ``10**400`` exercises the ``OverflowError`` it
*raises*, because a value can be finite and still be too large to be a float —
only that case reaches the helper's ``except`` branch. A helper whose handler
never fires is decoration. The learning-rate test's list carries the same three
for the same reason.

That test's rate list has since grown a case the guard did not originally cover.
`json.loads` yields a Python `int` for a JSON integer literal, and one larger than
a float can hold makes `math.isfinite` raise `OverflowError` rather than return
`False` — so `learning_rate=10**400` escaped the guard as an internal error.
Measured, both before and after the `isfinite` fix. The list is now:

... the decorator on `test_a_learning_rate_adam_cannot_use_is_refused` becomes:

```python
@pytest.mark.parametrize(
    "rate",
    [0, 0.0, -0.1, True, "0.1", float("nan"), float("inf"), 10**400],
)
def test_a_learning_rate_adam_cannot_use_is_refused(rate: object) -> None: ...
```

- [ ] **Step 2: Run them to verify they fail**

Run: `../.venv/bin/pytest tests/test_training.py -k "trajectory or ham_ or reaches" -q`
Expected: FAIL with `ImportError: cannot import name 'train_parameters'`

Note: the refusal test at 16 qubits uses `[{"pauli": "I"*16, ...}]`, which the shared validator refuses as all-identity — so the budget check must run before the objective is built, or this test fails for the wrong reason. Put the budget check first and let this test be the one that proves the ordering.

- [ ] **Step 3: Write the tool body**

Add to `training.py`, and extend its import block to this complete set — every
name here is used by this task's code, and `ruff`'s `F401` will say so if one is
not:

```python
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
from flagquantum_mcp_server.errors import (
    ToolInputError,
    ToolLimitError,
    UnsupportedFormatError,
)
from flagquantum_mcp_server.planning import validate_pauli_terms
from flagquantum_mcp_server.preconditions import (
    SDK_FAILURE_BASES,
    plain,
    reject_circuit_observables,
)
```

This is the file's complete block, not this task's own additions. `load_module`
and `validate_pauli_terms` are not called by `train_parameters`, but they are
called by `load_algorithms` and `hamiltonian_from_terms`, which live in this
same file — so they stay. `F401` is scoped to the file, not to the task, and a
name used anywhere in the module is used. Do not prune this block by asking
which names this task's new code calls.

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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../.venv/bin/pytest tests/test_training.py -q`
Expected: PASS, 32 tests — twenty after Task 6, plus the twelve Step 1 adds.
Assert the *delta*, not the total: read the count off the run's own output and
check it went up by twelve. Every absolute figure in this plan's earlier tasks
was written before their fix rounds added tests, and every one of them came out
one or two low as a result.

If `test_the_hamiltonian_reaches_the_objective` fails by converging to -1, the `policy=` argument is not reaching the SDK. That is the exact failure this task exists to prevent; do not weaken the assertion.

- [ ] **Step 5: Mutation-test the policy, which is the point of the module**

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path
p = Path("src/flagquantum_mcp_server/training.py")
before = p.read_text()
Path(str(p) + ".mutbak").write_text(before)
after = before.replace(
    '        policy=sdk.RuntimePolicy(observable="hamiltonian"),\n', ""
)
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_training.py -q
python3 -c "import pathlib; pathlib.Path('src/flagquantum_mcp_server/training.py.mutbak').replace(pathlib.Path('src/flagquantum_mcp_server/training.py'))"
find . -name __pycache__ -type d -exec rm -rf {} +
```

Expected: FAIL on `test_the_hamiltonian_reaches_the_objective` — the run converges to `-1.0` instead of `-2.236`. If it passes, the policy is being set somewhere else and this tool is not the thing under test.

Then the continuation promise:

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
from pathlib import Path
p = Path("src/flagquantum_mcp_server/training.py")
before = p.read_text()
Path(str(p) + ".mutbak").write_text(before)
after = before.replace(
    '        "final_loss": final_loss,',
    '        "final_loss": float(result.losses[-1]),',
)
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_training.py -k final_loss -q
python3 -c "import pathlib; pathlib.Path('src/flagquantum_mcp_server/training.py.mutbak').replace(pathlib.Path('src/flagquantum_mcp_server/training.py'))"
find . -name __pycache__ -type d -exec rm -rf {} +
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
Path(str(p) + ".mutbak").write_text(before)
after = before.replace(
    "    _check_budget(ir, steps)\n\n    names = parameter_names(ir)",
    "\n    names = parameter_names(ir)",
)
assert after != before, "the first replace did not apply"
moved = after.replace(
    "    objective = hamiltonian_from_terms(hamiltonian, n_wires=int(ir.n_wires))",
    "    objective = hamiltonian_from_terms(hamiltonian, n_wires=int(ir.n_wires))\n    _check_budget(ir, steps)",
)
assert moved != after, "the second replace did not apply"
after = moved
p.write_text(after)
PY
../.venv/bin/pytest tests/test_training.py -k budget -q
python3 -c "import pathlib; pathlib.Path('src/flagquantum_mcp_server/training.py.mutbak').replace(pathlib.Path('src/flagquantum_mcp_server/training.py'))"
find . -name __pycache__ -type d -exec rm -rf {} +
```

**The budget check must be re-inserted *below* the objective line, not above
it.** An earlier draft of this step put it immediately *above* — which keeps
`_check_budget` running before `hamiltonian_from_terms`, so the all-identity
objective is still never built, the `ToolLimitError` still fires, and the test
stays **green**. Measured: the above-version gives `2 passed`, the below-version
gives `1 failed, 1 passed` with `ToolInputError: terms[0] 'IIII…' is entirely
identity`. A mutation that leaves the property it means to break intact is worse
than no mutation, because its green reads as "the ordering is unpinned".

**This mutation is two replacements, so it gets two guards.** A single
`assert after != before` after a chain proves only that *one* of them applied.
Each half-applied state is a different experiment from the one you mean to run,
and both look like a result:

- first replaced, second not — `_check_budget` is gone entirely, so the test
  goes red because no limit error is raised at all, which is not the ordering
  claim.
- second replaced, first not — there are now two `_check_budget` calls, the
  early one still runs first, and the test goes **green**. An implementer who saw
  that would conclude the ordering is unpinned and rewrite the test to match.

Guard each replacement against the text it consumed, not against the original.

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

**Three refusal tests shipped early, with Task 7, and are already in
`tests/test_training.py`:**
`test_a_learning_rate_adam_cannot_use_is_refused` (with `nan` and `inf` in its
parametrize list), `test_a_non_finite_starting_value_is_refused`, and the
`_check_steps` cases. All three pin validators that live in Task 7, and the
non-finite pair was found by review after Task 7 was already written — so they
shipped with the code they pin rather than a task later, which is where a test
belongs. **Do not re-add any of them here.** Two `def`s with one name in a module
means pytest collects one and silently never runs the other. Check the file
before appending: if a name below is already present, skip it and say so in your
report.


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

    message = str(caught.value)
    assert "empty" in message
    # Both words, because "empty" alone is already true of the shared
    # validator's message before this task renames anything: it reads
    # "'terms' is empty". Asserting only that would pass before this task
    # does its one job, and would keep passing if it never did.
    assert "hamiltonian" in message
    assert "'terms'" not in message


def test_a_value_for_a_parameter_the_circuit_does_not_have_is_refused() -> None:
    """The typo case: silently ignored, the parameter trains from zero."""
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(ANGLED, TFIM2, "qir", values={"t0": 0.1, "t1": 0.1, "t2": 0.1})

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

Run: `../.venv/bin/pytest tests/test_training.py -q`
Expected: **exactly two** of the eight fail, and I measured which:

- `test_a_missing_hamiltonian_is_refused`
- `test_an_empty_hamiltonian_is_refused`

Both for the same reason, which is this task's whole job: the call reaches
`validate_pauli_terms`, whose messages are written in the vocabulary of the
`outputs` argument, so they say `'terms'` — a field the caller never typed. Note
the second one carefully: the shared validator's message is `'terms' is empty`,
so a test asserting only the word `empty` would pass **before this task changes
anything**. That is why it asserts `hamiltonian` is present and `'terms'` is
absent.

**The other six already pass, and that is not a mistake in your run.** Task 7
ships those refusals in the same code path that must perform them: the
no-parameters case, the step count, the learning rate, the budget, and the
observables check — the last of which already names `hamiltonian` in its remedy.
Do not delete, move, or weaken any of them to make this task's tests red. Step 2
is a check on your reading, not a checklist you are required to turn red. Say in
your report which were already green.

An earlier draft of this step said "four of the nine"; both numbers were wrong,
and it named two tests as failing that pass. A wrong expectation is worse than no
expectation, because a run that disagrees with it looks like a defect in the
code.

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
```

Then change the call in `train_parameters`. Replace exactly this line:

```python
    objective = hamiltonian_from_terms(hamiltonian, n_wires=int(ir.n_wires))
```

with:

```python
    objective = _objective(hamiltonian, n_wires=int(ir.n_wires))
```

`hamiltonian_from_terms` stays imported — `_objective` calls it.

Then two more, which are what make the rename a property rather than a phrase
list. Append them to `flagquantum-mcp-server/tests/test_training.py`:

```python
def test_no_refusal_from_this_tool_says_a_noun_the_caller_did_not_write() -> None:
    """Every refusal, in the caller's vocabulary — checked as a property.

    Every message below comes from the shared validator, which speaks the
    vocabulary of an ``expectation`` output. A caller who wrote ``hamiltonian``
    must never be told about ``terms``, ``terms[0]`` or an ``expectation``:
    those name something they never typed, and a message that has to be
    translated is one that cannot be acted on.

    Checked over every malformed shape at once rather than phrase by phrase,
    because phrase-by-phrase is exactly what let the first version miss the
    term-level refusals — it replaced the quoted field name, and ``terms[0]``
    is unquoted.
    """
    from flagquantum_mcp_server.training import train_parameters

    malformed = [
        None,
        [],
        [{"coefficient": 1.0}],
        [{"pauli": "ZZ", "coefficient": 1.0, "extra": 1}],
        [{"pauli": "ZZZ", "coefficient": 1.0}],
        [{"pauli": "II", "coefficient": 1.0}],
        [{"pauli": "ZZ", "coefficient": "x"}],
        "ZZ",
    ]

    for objective in malformed:
        with pytest.raises(ToolInputError) as caught:
            train_parameters(ANGLED, objective, "qir", steps=1)
        message = str(caught.value)
        assert "hamiltonian" in message, (objective, message)
        assert "terms" not in message, (objective, message)
        assert "expectation" not in message, (objective, message)


def test_an_unsupported_key_still_reports_as_a_format_problem() -> None:
    """The same collapse that hit the bound hits this code, and nothing noticed.

    ``UnsupportedFormatError`` subclasses ``ToolInputError``, so a bare
    ``except ToolInputError`` catches it and re-raises the base class. Measured
    before the fix: an unknown key in a term came back as ``INVALID_INPUT``
    where the validator had raised ``UNSUPPORTED_FORMAT``. The property test
    above passes either way, because it asserts prose — which is exactly why
    this test asserts the code.
    """
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(UnsupportedFormatError) as caught:
        train_parameters(ANGLED, [{"pauli": "ZZ", "coefficient": 1.0, "extra": 1}], "qir", steps=1)

    assert caught.value.code == "UNSUPPORTED_FORMAT"


def test_a_caller_s_literal_is_echoed_back_unchanged() -> None:
    """The rename touches the validator's nouns, never the caller's value.

    A refusal quotes the offending value back: ``has a coefficient that is a
    string ('terms')``. Renaming the field name ``terms`` therefore has a message
    in which the SDK's word and a caller's literal are the same six characters,
    and an unanchored replace answers ``('hamiltonian')`` — the caller's own
    value rewritten, in the message whose job is to explain it. Measured.

    A literal containing ``terms[`` is still echoed renamed; that is the residual
    this test records rather than claims away.
    """
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as caught:
        train_parameters(ANGLED, [{"pauli": "ZZ", "coefficient": "terms"}], "qir", steps=1)

    message = str(caught.value)
    assert "('terms')" in message, message
    assert "('hamiltonian')" not in message, message
    assert "hamiltonian[0]" in message, message


def test_a_literal_is_echoed_unchanged_in_every_position_it_can_occupy() -> None:
    """A caller's value is never rewritten by a message about that value.

    The SDK's nouns and a caller's literals meet in the same message: the
    validator says ``terms[0] has no 'pauli' string`` and, for a caller who
    passed the literal string ``terms[``, it says ``terms[0] 'pauli' 'terms['
    covers 6 wires``. A blanket replace cannot tell those apart.

    **Each case carries the literal it sends and the form a blanket replace
    would have produced, and the two halves are asserted per case.** An earlier
    version of this test asserted a flat ``"'terms'" not in message`` for all
    six cases. Measured, that reds against correct code in every one of them:
    the caller's own quoted value *contains* the substring the assertion
    forbids — ``'terms['`` contains ``'terms'``. A single ``not in`` pair cannot
    express "the caller's text is intact and the SDK's noun is renamed" when the
    two are the same word with quotes around it.

    This test replaced one that asserted the *defect* in the bracket position.
    Both bracket and field-name forms are anchored now, so the defect is gone and
    the tripwire fired, which is what it was for. The one surviving residual is
    the phrase form, and it has its own test below.
    """
    from flagquantum_mcp_server.training import train_parameters

    cases = [
        # (objective, the caller's literal, what a blanket replace would say)
        ([{"pauli": "terms[", "coefficient": 1.0}], "terms[", "hamiltonian["),
        ([{"pauli": "ZZ", "coefficient": "terms["}], "terms[", "hamiltonian["),
        ([{"pauli": "ZZ", "coefficient": 1.0, "terms[": 1}], "terms[", "hamiltonian["),
        ([{"pauli": "terms", "coefficient": 1.0}], "terms", "hamiltonian"),
        ([{"pauli": "ZZ", "coefficient": "terms"}], "terms", "hamiltonian"),
        ([{"pauli": "ZZ", "coefficient": 1.0, "terms": 1}], "terms", "hamiltonian"),
    ]

    for objective, literal, rewritten in cases:
        with pytest.raises(ToolInputError) as caught:
            train_parameters(ANGLED, objective, "qir", steps=1)
        message = str(caught.value)
        assert "hamiltonian" in message, (objective, message)
        # The caller's own value is quoted back verbatim ...
        assert f"'{literal}'" in message, (objective, message)
        # ... in the same message whose SDK noun was renamed.
        assert f"'{rewritten}'" not in message, (objective, message)


def test_the_phrase_shaped_message_is_renamed_without_touching_a_caller_literal() -> None:
    """The last pattern with no message prefix, and the two messages it must tell apart.

    ``An expectation needs`` sits mid-sentence, so it cannot be anchored to the
    start of a message the way the other three patterns are. It appears in
    exactly one message the validator writes, always directly after ``empty. `` —
    and a caller who passes the same words as a value gets them inside
    ``is a string ('...')``. Anchoring on the longer context rewrites the
    validator's copy and leaves the caller's alone.

    This replaced a test that asserted the *defect* here, on the recorded belief
    that there was no position to anchor on. The belief was wrong and the test
    caught it: it went red the moment the anchor was added, which is what a
    tripwire is for.

    The last block pins the **boundary** of the fix rather than only its success,
    because the success was twice described as total and twice measured to be
    short of it. A literal that contains the anchor *itself* is still rewritten;
    reaching that takes passing a fragment of the validator's own sentence as a
    value. If the replace is ever made exact, this is the assertion to invert.
    """
    from flagquantum_mcp_server.training import train_parameters

    with pytest.raises(ToolInputError) as empty:
        train_parameters(ANGLED, [], "qir", steps=1)

    assert "An objective needs" in str(empty.value), str(empty.value)
    assert "An expectation needs" not in str(empty.value), str(empty.value)

    phrase = "An expectation needs a term"
    with pytest.raises(ToolInputError) as quoted:
        train_parameters(ANGLED, [{"pauli": "ZZ", "coefficient": phrase}], "qir", steps=1)

    message = str(quoted.value)
    assert f"('{phrase}')" in message, message
    assert "('An objective needs a term')" not in message, message

    anchored = "empty. An expectation needs a term"
    with pytest.raises(ToolInputError) as nested:
        train_parameters(ANGLED, [{"pauli": "ZZ", "coefficient": anchored}], "qir", steps=1)

    nested_message = str(nested.value)
    assert f"('{anchored}')" not in nested_message, nested_message
    assert "('empty. An objective needs a term')" in nested_message, nested_message


def test_the_term_bound_still_reports_as_a_limit_and_not_as_invalid_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The rename must not swallow the bound's own error code.

    ``ToolLimitError`` subclasses ``ToolInputError``, so a bare
    ``except ToolInputError`` around the builder catches it and re-raises it as
    the base class. Measured: the caller then sees ``INVALID_INPUT`` where the
    bound had raised ``LIMIT_EXCEEDED``. The errors module exists so an agent can
    branch on the code rather than parse prose, and this collapses two branches
    into one.
    """
    from flagquantum_mcp_server.training import train_parameters

    monkeypatch.setenv("FLAGQUANTUM_MCP_MAX_HAMILTONIAN_TERMS", "2")

    with pytest.raises(ToolLimitError) as caught:
        train_parameters(ANGLED, [{"pauli": "Z", "coefficient": 1.0}] * 5, "qir", steps=1)

    assert caught.value.code == "LIMIT_EXCEEDED"
    # Not `"hamiltonian" in ...`: the bound's own message says "This expectation
    # carries 5 terms", and the rename turns that phrase into "This objective
    # carries" — it names no field at all, so the word never appears. Asserting
    # for it here is a test that cannot pass, which is how this was found.
    # These two lines pin the fourth rename shape instead, which is worth more:
    # the phrase is the only one that reaches the bound's message.
    assert "expectation" not in str(caught.value)
    assert "objective" in str(caught.value)
```

`ToolLimitError` is already in this file's error import, from Task 7.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `../.venv/bin/pytest tests/test_training.py -q`
Expected: PASS. **The def count is unchanged at 49 and the collected count stays
at 62.** This round deletes one test and adds two, but one of the two is
parametrized over six cases, so the collected total moves by more than the def
total does. Read both numbers off the run rather than assuming either:

```bash
../.venv/bin/pytest tests/test_training.py --collect-only -q | tail -1
```

An earlier draft of this step said "adds fourteen tests" from a before-count of
60. The before-count was 61 — this plan's own round-3 step had added the residual
test this round deletes — and neither figure was checked against the file. (Three of the refusals it would otherwise add
are already in the file, shipped with Task 7 — see the note at the top of this
task. One of the ten is parametrized, so the collected count rises by more than
the def count; that is what Step 3 of Task 7 saw too.)

- [ ] **Step 5: Mutation-test each refusal**

Each refusal is a guard clause. The mutation is to delete it and watch exactly one test go red. Run this loop and read the output rather than trusting it:

```bash
cd "$(git rev-parse --show-toplevel)/flagquantum-mcp-server"
python3 - <<'PY'
import ast, pathlib, shutil, subprocess
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
    # The WHOLE call, not a prefix of it. An earlier draft deleted only its first
    # two lines, which leaves `remedy=(...)` and `)` dangling: an
    # IndentationError, a collection error, a non-zero exit, and this script
    # printing CAUGHT for a test that never ran. Measured.
    ("observables guard",
     "    reject_circuit_observables(\n        ir,\n        remedy=(\n"
     "            \"Pass the Hamiltonian as this tool's 'hamiltonian' argument \"\n"
     "            \"instead, which is where the objective belongs.\"\n"
     "        ),\n    )\n",
     "test_a_circuit_carrying_observables_is_refused_and_points_at_hamiltonian"),
]

import shutil

for label, target, test in mutations:
    if target not in original:
        print(f"{label}: TARGET NOT FOUND — fix the mutation script")
        continue
    mutant = original.replace(target, "", 1)
    assert mutant != original, f"{label}: mutation did not apply"
    try:
        ast.parse(mutant)
    except SyntaxError as exc:
        print(f"{label}: INVALID MUTANT — {exc}; this would have reported a false CAUGHT")
        continue
    path.write_text(mutant)
    result = subprocess.run(
        ["../.venv/bin/pytest", "tests/test_training.py", "-k", test, "-q"],
        capture_output=True, text=True,
    )
    # Read WHY it failed. A collection error is a non-zero exit and is not a catch.
    summary = next(
        (ln for ln in reversed(result.stdout.splitlines()) if ln.strip()), ""
    )
    print(f"{label}: {'CAUGHT' if result.returncode != 0 else 'MISSED'}  | {summary[:90]}")
    path.write_text(original)
    shutil.rmtree(".pytest_cache", ignore_errors=True)
    for cache in pathlib.Path(".").rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)

assert path.read_text() == original, "restore failed"
PY
```

Expected: every line reads `CAUGHT`. A `MISSED` means that guard is not what the test is checking. A `TARGET NOT FOUND` means the mutation did not apply and the result would have been meaningless — the same failure that hid a broken test in an earlier session.

**Read the summary the script prints beside each verdict, not only the verdict.**
`CAUGHT` means "pytest exited non-zero", and a module that does not parse also
exits non-zero. An earlier draft of this step deleted the first two lines of the
observables call, which leaves the rest of its arguments dangling: the run
printed `CAUGHT` with the summary `1 error in 0.06s` and the test never
executed. That is why the loop now parses each mutant before running it and
prints what the run actually said.

The cache clearing is not decoration either. A restored file whose mtime matches
the mutant's to the second, at the same byte length, is a cache hit on the
mutant's bytecode — the source reads correctly while the interpreter runs the
mutation. Each mutant here differs in length so that particular collision does
not fire, but the clear costs nothing and the failure it prevents is invisible.


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

Add to `flagquantum-mcp-server/tests/surface.py`, to `EXPECTED_TOOLS`:

```python
        "train_parameters_tool",
```

`EXPECTED_TOOLS` is a `frozenset`, so placement carries no meaning and the
existing entries are grouped by kind (circuit tools, then emitters, then
execution, then the describe/inspect family) rather than sorted. Put the new name
at the end and leave the rest alone — **do not re-sort the set.** Reordering it
would produce a large diff that changes nothing, in the one file whose whole
purpose is to be readable as the server's surface.

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
            pass a gate list such as '[{"name": "h", "index": [0]}, {"name":
            "ry", "index": [0], "parameters": {"theta": {"$parameter": "t0"}}}]'.
            A gate carries its arguments under "parameters", so a symbol is
            written {"theta": {"$parameter": "t0"}}. With circuit_format="ir"
            pass FlagQuantum IR JSON. OpenQASM is not accepted.
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
    # A bound, not just "it went down". `final < initial` is true at every rate
    # and every step count tried — measured, this circuit at default lr=0.1
    # reaches -2.0539 after seven steps, and at lr=0.4 reaches -2.1289 — so
    # "it went down" passes even when `learning_rate` never leaves the handler
    # and the default is used. Both runs are deterministic to six decimals.
    assert payload["final_loss"] < -2.10, "learning_rate must reach the optimizer"
```

Add a case to `CASES`, using a two-qubit parameterized gate list and a three-term objective:

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
Path(str(p) + ".mutbak").write_text(before)
after = before.replace(
    "        values=values,\n        steps=steps,\n        learning_rate=learning_rate,\n",
    "        steps=steps,\n        learning_rate=learning_rate,\n",
)
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_tool_wiring.py -q
python3 -c "import pathlib; pathlib.Path('src/flagquantum_mcp_server/server.py.mutbak').replace(pathlib.Path('src/flagquantum_mcp_server/server.py'))"
find . -name __pycache__ -type d -exec rm -rf {} +
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
WHEELCHECK="$(mktemp -d)"
python3 -m venv "$WHEELCHECK"
"$WHEELCHECK/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"$WHEELCHECK/bin/python" -m pip install dist/*.whl
"$WHEELCHECK/bin/python" -c "
import asyncio
from fastmcp import Client
from flagquantum_mcp_server.server import mcp

async def main():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print(sorted(t.name for t in tools))

asyncio.run(main())
"
"$WHEELCHECK/bin/python" -c "import pytest"   # must FAIL
```

**`mktemp -d`, not a fixed path.** The rule this step exists to satisfy is that a
check on the artifact must run in the environment the artifact will have — "when
a step needs an environment, build that environment rather than borrowing the one
that is already warm". A fixed `/tmp/wheelcheck` does not build one: measured on
this machine, that path already existed from an earlier rehearsal, holding
`flagquantum` 0.2.0. `python3 -m venv` on an existing directory reuses it rather
than refusing, so the install would be an upgrade over a warm tree and the
`import pytest` line below would be asserting about whatever the previous run
left behind. A fresh temporary directory makes the claim true by construction and
deletes nothing.

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

`TIER3_TRAINING` and `test_training_dependencies_are_still_declared_public`
already landed in Task 6, beside the code that creates the dependency. This step
adds the rest — the policy pin, and the `Module` members the tool reads:

```python
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
    """The three class-level members. The fourth is pinned elsewhere, on purpose.

    ``named_parameter_groups`` is deliberately absent from this tuple. It is
    assigned in ``Module.__init__`` — ``self.named_parameter_groups:
    ParameterDict | None = None`` — so it does not exist on the class, and
    ``hasattr(fq.Module, "named_parameter_groups")`` is **False**. Measured, and
    a test that asserted it would fail on the very SDK it is meant to pin, which
    is the failure mode this file exists to prevent. It is covered instead by the
    training tests, which read it off a real module the tool built, at the one
    place ``training.py`` needs a named group.
    """
    for attribute in ("execute", "parameters", "named_parameters"):
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
Path(str(p) + ".mutbak").write_text(before)
after = before.replace('("flagquantum.algorithms", "Hamiltonian"),', '("flagquantum.algorithms", "HamiltonianX"),')
assert after != before, "mutation did not apply"
p.write_text(after)
PY
../.venv/bin/pytest tests/test_api_contract.py -k training_dependencies -q
python3 -c "import pathlib; pathlib.Path('tests/test_api_contract.py.mutbak').replace(pathlib.Path('tests/test_api_contract.py'))"
find . -name __pycache__ -type d -exec rm -rf {} +
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
count in the message. The cost that dominates at the top of the width range is
holding the state: 25 s per step at 24 wires before a single gate is applied, so
no 24-wire circuit gets more than two steps, and the layered ansatz measured here
— 71 instructions, 119 s per step — gets none. That is a statement about the
circuit, not about the width: one gate at 24 wires is still under the budget.

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

Three edits to `AGENTS.md`.

**First, rule 4's in-process paragraph**, so it covers the tool that trains as well as the one that simulates. After the existing `simulate_circuit_tool` paragraph, add:

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
  default is `z`. Nothing warns. Measured: on a two-qubit circuit whose ⟨Z₀⟩
  gradient vanishes at the initial state, the run does not merely converge on the
  wrong quantity — it reports `status: "success"` with 200 completed steps and a
  loss that never moved. The training tool sets the policy explicitly and a test
  asserts the Hamiltonian reaches the objective. When a public API takes an
  argument it does not act on, assume the same shape elsewhere: find the second
  object that decides, and set it.
```

**Third, the wheel rehearsal's environment**, in the Verification section. Its
example uses a fixed `/tmp/wheelcheck`, which does not build an environment — it
reuses whatever is there. Measured on this machine: that path already existed
from an earlier rehearsal, holding `flagquantum` 0.2.0, and `python3 -m venv` on
an existing directory reuses it rather than refusing. So the `import pytest` line
underneath it asserts about the previous run's tree, in the section whose own
words are "when a step needs an environment, build that environment rather than
borrowing the one that is already warm". Replace the fixed path with a fresh
temporary directory:

```bash
WHEELCHECK="$(mktemp -d)"
python3 -m venv "$WHEELCHECK"
"$WHEELCHECK/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
"$WHEELCHECK/bin/python" -m pip install dist/*.whl
"$WHEELCHECK/bin/python" -c "import pytest"            # must FAIL: see below
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
