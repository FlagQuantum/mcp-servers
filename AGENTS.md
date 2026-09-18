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

3. **Depend only on the stable public API.** The allowed surface is the set of
   names in FlagQuantum's `docs/public_api_v1.json` (`stable_exports`), plus
   what a stable export returns. Internal modules (`flagquantum.core.*`,
   `flagquantum.runtime.*` internals, planner internals) are off limits. A
   private import in this repository is a bug even if it currently works.

4. **No network egress, no credentials, no hardware.** Every tool in this
   repository runs locally and deterministically. Tools must never submit
   remote jobs, read tokens, contact a provider, or consume paid resources.
   Docstring examples must run offline with no credentials — the same rule
   FlagQuantum sets for its own stable entry points.

5. **Tools are read-only over the caller's inputs and side-effect free.** No
   tool may mutate files, environment variables, or global state.

6. **Pin the FlagQuantum contract.** The dependency is a version range
   (`flagquantum>=0.2,<0.3`), never a git URL or a revision. A dependency
   table containing a URL makes every job that installs a different upstream
   revision unresolvable.

7. **Adding a tool requires evidence that the underlying API is stable.** Do
   not add a tool to reach a capability that is still `experimental` or
   `development_evidence` in FlagQuantum's `capability-maturity.toml`.

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

Report the actual output. A scaffold that has never been executed is not
finished, and this repository's README must not describe a capability that no
test exercises.
