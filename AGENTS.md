# AGENTS.md

Guidance for AI coding agents working in this repository.

## Core purpose

This repository publishes MCP (Model Context Protocol) servers that give AI
assistants, agents and IDEs direct access to the FlagQuantum SDK through a
standardized protocol that works with any MCP-compatible client.

The reference design is [`Qiskit/mcp-servers`](https://github.com/Qiskit/mcp-servers).
We follow its packaging, testing and publishing conventions so that anyone who
has configured a Qiskit MCP server already knows how to configure ours.

## Non-negotiable rules

1. **This repository is an out-of-tree edge adapter. Nothing here may be
   imported by the FlagQuantum main repository.** FlagQuantum's long-horizon
   architecture contract names
   `the main repository has no production MCP transport dependency` as a
   retirement condition, and `tests/team/services/test_service_boundaries.py`
   in that repository fails if `mcp` or `fastmcp` becomes importable on the
   core path. Do not propose moving these servers in-tree; do not open pull
   requests against FlagQuantum that add an MCP dependency.

2. **Adapters stay thin.** A tool handler validates input, calls a public
   FlagQuantum API, and serializes the result. Domain logic, numerical
   kernels, scheduling, retries and persistence belong to FlagQuantum. If a
   tool needs behaviour FlagQuantum does not expose, the correct response is a
   public-API change in FlagQuantum, not a reimplementation here.

3. **Depend only on the public API, and record which tier you are using.**
   Three tiers are allowed, and a test in `tests/test_api_contract.py` pins the
   members of each:

   | Tier | Surface | Strength |
   | --- | --- | --- |
   | 1 | The frozen `stable_exports` snapshot | Strongest; each name has a verification test upstream |
   | 2 | `flagquantum.compiler` (documented `__all__`, not in the snapshot) | Public, but not frozen |
   | 3 | Public names a package declares in its own `__all__` (`flagquantum.core`, `flagquantum.drawer`, `flagquantum.algorithms.{Hamiltonian, pauli_term}`) and the two emitters | Weakest; must be pinned by a test here |

   Anything else — planner internals, executor internals, anything reachable
   only by a private path — is off limits. A private import in this repository
   is a bug even if it currently works.

   Tier 3 is admitted for one reason: a circuit is only usable if the caller
   knows each gate's wire count and parameter names, and the SDK's operator
   manifest is the only authority for that. Reimplementing it here would create
   a second source of truth that drifts. When you add a tier-3 dependency,
   extend the table in `tests/test_api_contract.py` in the same change.

4. **No network egress, no credentials, no hardware.** Every tool in this
   repository runs locally and deterministically. Tools must never submit
   remote jobs, read tokens, contact a provider, or consume paid resources.
   Docstring examples must run offline with no credentials — the same rule
   FlagQuantum sets for its own stable entry points.

   **In-process simulation is inside this line, not across it.**
   `simulate_circuit_tool` runs a statevector simulation on CPU in the same
   process: no target, no provider, no token, and the SDK reports
   `execution_path: local_statevector` / `platform_provider: pytorch_cpu` on
   the result. It is also the one tool that must not be read as evidence about
   hardware, so its result carries the SDK's `accuracy.metric == "not_measured"`
   rather than a summary that drops it. Measured, the result is
   `{execution, outputs, status}` and its `accuracy` block has five fields
   (`version, metric, value, tolerance, passed`) — no `provenance` key and no
   `release_gate_allowed` anywhere in it. That second name belongs to the
   *plan*: `plan_execution_tool`'s `summary` carries
   `release_gate_allowed: false` for the same circuit, and it is the plan's
   field, not the result's. FlagQuantum's ARCH-001 draws the same boundary from
   the other side: direct in-process resources are *Compute*/*Simulation*, and
   credentials and submission belong to *Remote*, which this server does not
   touch.

   `train_parameters_tool` is inside the same line. The gradients it uses are
   the SDK's exact gradients for a statevector simulation, computed in this
   process from a circuit the caller supplied — no target, no token, no
   hardware. What it adds is a way to *lower* an energy rather than read one,
   which is why it carries the same provenance block and the same
   `accuracy.metric == "not_measured"` as a simulation: a converged loss curve
   is not evidence about any machine either.

5. **Tools are read-only over the caller's inputs and side-effect free.** No
   tool may mutate files, environment variables, or global state.

6. **Pin the FlagQuantum contract.** The dependency is a version range
   (`flagquantum>=0.2,<0.3`), never a git URL or a revision. A dependency
   table containing a URL makes every job that installs a different upstream
   revision unresolvable.

7. **Adding a tool requires evidence that the underlying API is stable.** Do
   not add a tool to reach a capability that is still `experimental` or
   `development_evidence` in FlagQuantum's `capability-maturity.toml`.

8. **A PyPI release is a discovery event, not a way to ship a commit.** Do not
   release for a documentation-only, test-only or CI-only change, and do not
   bump the version in the same breath as making a change. Accumulate work on
   `main` and release when someone outside this repository would benefit from
   installing it: a new tool, a change that alters a tool's results, a contract
   change, a packaging fix. Anyone who wants `main` in the meantime installs
   from git — see CONTRIBUTING.md. Every published filename is permanent and
   public, so the release list is part of what this project looks like; a
   version per commit makes it look abandoned-and-thrashing at the same time.
   Never release without explicit authorization.

## Conventions

- Every server is a standalone PyPI package under its own top-level directory,
  with `src/` layout, hatchling, its own `pyproject.toml`, `README.md`,
  `server.json`, `tests/` and `examples/`.
- Package directory name equals the console-script name equals the PyPI name.
- Tools are named `<verb>_<noun>_tool` and return a `dict` with a
  `"status"` key of `"success"` or `"error"`.
- Shared code lives in the leaf server package and is imported by others via
  the package name, not by relative paths across directories.
- Lint and type configuration is shared at the repository root and extended
  per package. Do not fork the rule set inside a package.
- **A prompt is a recipe, not a channel.** Prompts are for the clients that
  present them to a person — Claude Desktop, the MCP Inspector, a slash
  command. An autonomous agent may receive none of them: four recorded sessions
  each hunted for a prompt, tried directory listing, and could not tell whether
  prompts existed at all. So a constraint an agent must obey cannot live only
  in a prompt. Each of the seven rules across this server's three prompts was
  checked against the channels an agent does read — the server instructions,
  the tool descriptions, the resources — and the two reachable nowhere else
  now also appear in the summary of the tool that acts on them.
  `PROMPT_CONSTRAINTS` in `tests/test_server_contract.py` holds that mapping,
  and a new prompt rule fails the build until it is either placed where an
  agent will read it or recorded there with a tool that carries it.
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

## Verification

Before claiming work is complete, run everything:

```bash
cd flagquantum-mcp-server
../.venv/bin/ruff check .
../.venv/bin/ruff format --check .
../.venv/bin/mypy --config-file ../mypy.ini src
../.venv/bin/pytest -m "not integration"
```

Run **these commands**, not equivalents. `pytest` as a console script inserts
only the test file's own directory into `sys.path`; `python -m pytest` inserts
the working directory as well, so it resolves imports that CI cannot and turns a
collection error into a green run. That difference put two red commits on `main`
and reached a release bump before anyone saw it: 512 tests passed locally under
the invocation CI does not use. If a check has two spellings, the one written
here is the one that counts.

**These four are not the whole of CI.** `.github/workflows/ci.yml` runs nine
steps across three jobs, and the ones the list above does not cover are exactly
the ones that broke next:

```bash
cd flagquantum-mcp-server
../.venv/bin/pytest -m integration
../.venv/bin/python examples/stdio_client.py          # the documented example

# The package job, rehearsed rather than approximated: a clean environment with
# the wheel and no development dependencies, which is the only one that proves
# the built artifact works.
../.venv/bin/python -m build
WHEELCHECK="$(mktemp -d)"
python3 -m venv "$WHEELCHECK"
"$WHEELCHECK/bin/python" -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python3 - <<'PY'
import glob
import pathlib
import zipfile

# The directory check comes first: run from the wrong place, every other
# assertion here passes on empty sets and reports a match that compared nothing.
source = pathlib.Path("src/flagquantum_mcp_server")
assert source.is_dir(), f"run this from flagquantum-mcp-server/: {source} is not a directory"

wheels = sorted(glob.glob("dist/*.whl"))
assert wheels, "no wheel in dist/: run `../.venv/bin/python -m build` first"
assert len(wheels) == 1, f"more than one wheel in dist/: {wheels}"
wheel = pathlib.Path(wheels[0])

here = {str(f.relative_to(source)) for f in source.rglob("*.py")}
assert here, f"no .py files under {source}"

with zipfile.ZipFile(wheel) as z:
    names = set(z.namelist())
    packaged, stale, absent = {}, [], []
    for rel in sorted(here):
        member = "flagquantum_mcp_server/" + rel
        if member not in names:
            absent.append(member)
        else:
            packaged[member] = z.read(member)
            if packaged[member] != (source / rel).read_bytes():
                stale.append(member)

# The other direction: a wheel carrying a .py this tree no longer has is a stale
# build too -- a rename, a deletion -- and only this half sees it. The expected
# set is built in the wheel's own member spelling and compared whole, so a
# top-level module is matched as itself rather than stripped to a basename that
# could collide with an in-package name. .dist-info/ is the wheel's own
# bookkeeping, not a module.
expected = {"flagquantum_mcp_server/" + rel for rel in here}
extra = sorted(
    n for n in names if n.endswith(".py") and ".dist-info/" not in n and n not in expected
)

# No anti-vacuity assert here: `assert here` above guarantees the loop ran, and
# an empty `packaged` can only mean every module was absent, which the next line
# reports accurately. An assert that can only fire with the wrong message is
# worse than none.
assert not absent, f"{wheel} is missing {absent}"
assert not stale, f"{wheel} was not built from this tree: {stale}"
assert not extra, f"{wheel} carries modules this tree does not have: {extra}"
print(f"wheel matches the source tree ({len(packaged)} modules, both directions)")
PY
"$WHEELCHECK/bin/python" -m pip install dist/*.whl
"$WHEELCHECK/bin/python" -c "import pytest"            # must FAIL: see below
```

**Check the wheel is the one you just built, before you install it.** The version
does not change between rehearsals, so a stale
`flagquantum_mcp_server-0.2.0-py3-none-any.whl` and a fresh one are the same file
path, and a build that failed or was skipped leaves the rehearsal green against
the previous run's code. That block compares every packaged module to the source
tree byte for byte, which names no feature and so cannot go stale when the next
change lands. It also refuses to pass vacuously: run from the wrong directory,
`rglob` finds nothing, every comparison is skipped, and "0 modules compared"
reads as success — measured, which is why it checks the source directory before it
looks at the wheel at all, and refuses to finish on zero modules compared.

**Every job in CI must be reproducible here, and the environment is part of the
job.** The wheel smoke test failed on a release commit with `No module named
'pytest'` because the declaration it read lived in `conftest.py`: a check on the
artifact, broken by a dependency of the check. A verification step that only
passes in the environment you happen to have is not verifying the thing CI
verifies. When you add a step to the workflow, run it here the same way — and
when a step needs an environment, build that environment rather than borrowing
the one that is already warm.

Report the actual output. A scaffold that has never been executed is not
finished, and this repository's README must not describe a capability that no
test exercises.

**A new test earns its place only if it fails when the code is wrong.** Break
the thing it claims to check and watch it go red before trusting it. A test
written against an implementation that is already correct will pass whether or
not it asserts anything — this repository has shipped exactly that mistake once,
with twenty-two cases that all passed while checking nothing, and only a
deliberate mutation revealed it.

**This applies to the release path too, where the failure is more expensive.**
The registry workflow's last step asserted that *some* version of this server
is listed. That is true from the first release onward and cannot fail
afterwards, so it never checked anything — and it was also written to ask once,
with a 30-second timeout, against an endpoint measured at 18–38 seconds. On the
0.2.0 release both faults landed at once: the publish succeeded, the check
timed out, and the run went red on a release that worked. A red publish is the
most expensive kind of false alarm, because the obvious response is to publish
again. A verification step on this path must name the version it expects and
must be able to fail.
