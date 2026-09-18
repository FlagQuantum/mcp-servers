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
   | 3 | Public names a package declares in its own `__all__` (`flagquantum.core`, `flagquantum.drawer`) and the two emitters | Weakest; must be pinned by a test here |

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
   and the run's `release_gate_allowed: false` rather than a summary that drops
   them. FlagQuantum's ARCH-001 draws the same boundary from the other side:
   direct in-process resources are *Compute*/*Simulation*, and credentials and
   submission belong to *Remote*, which this server does not touch.

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

## Verification

Before claiming work is complete, run everything:

```bash
cd flagquantum-mcp-server
../../.venv/bin/ruff check .
../../.venv/bin/ruff format --check .
../../.venv/bin/mypy --config-file ../mypy.ini src
../../.venv/bin/pytest -m "not integration"
```

Run **these commands**, not equivalents. `pytest` as a console script inserts
only the test file's own directory into `sys.path`; `python -m pytest` inserts
the working directory as well, so it resolves imports that CI cannot and turns a
collection error into a green run. That difference put two red commits on `main`
and reached a release bump before anyone saw it: 512 tests passed locally under
the invocation CI does not use. If a check has two spellings, the one written
here is the one that counts.

Report the actual output. A scaffold that has never been executed is not
finished, and this repository's README must not describe a capability that no
test exercises.

**A new test earns its place only if it fails when the code is wrong.** Break
the thing it claims to check and watch it go red before trusting it. A test
written against an implementation that is already correct will pass whether or
not it asserts anything — this repository has shipped exactly that mistake once,
with twenty-two cases that all passed while checking nothing, and only a
deliberate mutation revealed it.
