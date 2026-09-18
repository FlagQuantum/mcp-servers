# Contributing

## Prerequisites

- Python 3.10 – 3.12 (this is FlagQuantum's supported range; it is narrower
  than what the MCP SDK alone would allow)
- Git

## Setting up

```bash
git clone https://github.com/FlagQuantum/mcp-servers.git
cd mcp-servers
python3 -m venv .venv
.venv/bin/python -m pip install -e "./flagquantum-mcp-server[dev]"
```

The first install pulls `torch`, because `flagquantum` depends on it. Nothing
in this repository imports torch directly.

## Layout

```
mcp-servers/
├── flagquantum-mcp-server/     # one standalone PyPI package per server
│   ├── src/flagquantum_mcp_server/
│   ├── tests/
│   ├── examples/
│   ├── pyproject.toml
│   ├── server.json             # MCP Registry manifest
│   └── README.md
├── ruff.toml                   # shared lint config, extended per package
├── mypy.ini                    # shared type config
└── .github/workflows/
```

There is deliberately no root meta-package yet. See the repository README for
why.

## MCP server patterns

### Tools

Register with the `@mcp.tool()` decorator. Name the function `<verb>_<noun>_tool`.
Return a `dict`; include `"status": "success"` or `"status": "error"`. Never
raise out of a tool — catch, and return a structured error, so the client sees
a useful message instead of a transport failure.

```python
@mcp.tool()
def analyze_circuit_tool(circuit: str, circuit_format: str = "qasm3") -> dict[str, Any]:
    """Report gate counts and depth for one circuit."""
    ...
```

### Resources

Use `@mcp.resource("flagquantum://...")` for read-only reference data that the
agent should be able to pull without a tool call — capability manifests, gate
tables, version contracts.

### Prompts

Use `@mcp.prompt()` for reusable multi-step workflows. Prompts return strings;
they never execute anything.

### Boundary rules

Read `AGENTS.md` before adding a tool. The short version: adapters stay thin,
only the stable public API is reachable, and nothing here talks to the network
or to hardware.

## Tests

- `pytest -m "unit"` runs by default; `pytest -m "integration"` is deselected
  in CI unless a `QISKIT`-style credential is present — and in this repository
  there is no such credential, because no tool needs one.
- Mock nothing about FlagQuantum. The package is a real dependency, so tests
  call the real library. Only mock the transport/process boundary.
- Every tool needs a test that asserts its exact output shape, not merely that
  it did not raise.
- Every tool that can reject input needs a test for the rejection path.

## Adding a new server

1. Create `<name>-mcp-server/` at the repository root with the standard layout.
2. Add `pyproject.toml` with `[tool.ruff] extend = "../ruff.toml"`.
3. Add the `server.json` manifest for MCP Registry discovery.
4. Add the package to CI: a lint entry and a test job in
   `.github/workflows/ci.yml`, and a job in `.github/workflows/publish-pypi.yml`.
5. Add a row to the repository README's server table.
6. Add an `examples/stdio_client.py` following the existing shape, and run it.
   An example that has never been executed is not an example.

## Publishing

Releases are tag-driven in the reference suite: `<package>-v<version>`.
**Here they are not.** `publish-pypi.yml` is manual-dispatch only until PyPI
trusted publishing is configured for this repository and a release has been
rehearsed — a tag that publishes is a tag that cannot be taken back. Publishing
uses OIDC; there are no long-lived API tokens in this repository. The version
in `server.json` must match `pyproject.toml`, and CI checks the two against
each other.

Publishing, and changing a package's visibility, are outward-facing actions.
Do not do either without explicit authorization.

## Publishing a second server

Three places enumerate the packages, and a new one must be added to all of
them or it cannot be released at all:

1. `.github/workflows/publish-pypi.yml` — `on.workflow_dispatch.inputs.package.options`
2. `.github/workflows/publish-mcp-registry.yml` — the same field
3. the server table in the repository `README.md`

Both workflows run with `working-directory: ${{ inputs.package }}`, so a
package that is not in the options list cannot be selected.

The MCP Registry entry also needs, in the new package's README, the ownership
marker the registry reads:

```markdown
<!-- mcp-name: io.github.<owner>/<server-name> -->
```

`tests/test_versions.py` in the first package shows all three checks — the
marker, the 100-character description limit, and the transport allow-list.

**Do not create a second GitHub repository for a second server.** PyPI trusted
publishing lets a project trust one repository and one workflow filename, so a
new repository means a new publisher configuration on PyPI, and a new
repository also cannot own the `io.github.FlagQuantum` namespace the registry
verifies. Adding a directory here is the cheap path; splitting is the expensive
one.
