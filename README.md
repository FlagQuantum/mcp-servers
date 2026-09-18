# FlagQuantum MCP Servers

A collection of [Model Context Protocol](https://modelcontextprotocol.io)
servers that give AI assistants, agents and IDEs direct access to the
[FlagQuantum](https://github.com/flagos-ai/FlagQuantum) SDK through a
standardized protocol that works with any MCP-compatible client.

The reference design is [`Qiskit/mcp-servers`](https://github.com/Qiskit/mcp-servers),
and we follow its packaging, testing and publishing conventions so that anyone
who has configured a Qiskit MCP server already knows how to configure ours.

## Servers

| Server | Package | What it does | Auth |
| --- | --- | --- | --- |
| [FlagQuantum](flagquantum-mcp-server/) | `flagquantum-mcp-server` | Build, compile, route, serialize and plan FlagQuantum circuits locally | None |

## Quick start

```bash
claude mcp add flagquantum -- uvx flagquantum-mcp-server
```

Any MCP-compatible client works the same way, because there is no credential to
configure:

```json
{
  "mcpServers": {
    "flagquantum": { "command": "uvx", "args": ["flagquantum-mcp-server"] }
  }
}
```

See [`flagquantum-mcp-server/README.md`](flagquantum-mcp-server/README.md) for
the full tool list and the limits each tool enforces, and
[`flagquantum-mcp-server/examples/`](flagquantum-mcp-server/examples/) for a
runnable end-to-end script.

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
├── AGENTS.md                   # rules for agents working in this repo
├── CONTRIBUTING.md
└── .github/workflows/
```

## Design constraints

These are deliberate, and each one is a departure from the reference suite that
a reader should understand rather than "fix".

**This repository is out of tree, permanently.** FlagQuantum's long-horizon
architecture contract names *"the main repository has no production MCP
transport dependency"* as a retirement condition, and its
`tests/team/services/test_service_boundaries.py` fails if `mcp` or `fastmcp`
becomes importable on the core path. Its `AGENTS.md` says MCP, REST, CLI and
notebook adapters "remain thin" while the domain stays vendor- and
SDK-neutral. Keeping protocol gateways at the system edge, in their own
repository, is what that contract asks for. A test in
`tests/test_api_contract.py` asserts the coupling points one way only.

**There is no root meta-package yet.** The reference suite ships
`qiskit-mcp-servers`, which installs a chosen subset through extras. That earns
its keep when the servers are genuinely separable, which for Qiskit they are: a
local circuit server needs no IBM account, a runtime server needs a token, a
gym server drags in torch through a reinforcement-learning stack. FlagQuantum
has one SDK, one IR and no heavyweight subsystems to split, so a meta-package
would aggregate nothing. It can be added when a second server exists.

**No server here submits hardware jobs.** FlagQuantum's released 0.2.0 ships no
remote-submission entry point, and QPU submission is a governed capability —
preflight, approval, budget, evidence — that belongs to a control plane rather
than to a local adapter any agent can call. Every tool in this repository runs
locally, deterministically, with no credentials.

**One dependency is unavoidable and it is heavy.** `flagquantum` requires
`torch`, so any package here pulls it. The reference suite's four core servers
are all light, and its `qiskit-docs-mcp-server` does not depend on `qiskit` at
all — there is no equivalent here, because every tool needs the SDK. On Linux
x86_64 a bare `pip install` resolves torch to the CUDA wheel set; CI pins the
CPU index URL. Expect the first `uvx` run to be slow.

## Which contracts a server may use

FlagQuantum publishes a frozen `stable_exports` snapshot (34 names, each with a
named verification test) and separately describes `flagquantum.compiler` as its
"stable expert compiler interface". Servers here may use both, in tiers, and
must say which tier they rest on:

| Tier | Surface | Strength |
| --- | --- | --- |
| 1 | The frozen `stable_exports` snapshot | Strongest; each name has a verification test upstream |
| 2 | `flagquantum.compiler` (`__all__` documented, not in the snapshot) | Public, but not frozen |
| 3 | Public submodule functions with no `__all__` (the emitters) | Weakest; must be pinned by a test here |

Anything else — `flagquantum.core.*`, planner internals, executor internals — is
off limits. See [`AGENTS.md`](AGENTS.md).

## Development

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e "./flagquantum-mcp-server[dev]"

cd flagquantum-mcp-server
../.venv/bin/ruff check .
../.venv/bin/ruff format --check .
../.venv/bin/mypy --config-file ../mypy.ini src
../.venv/bin/pytest -m "not integration"
../.venv/bin/pytest -m integration
```

The first install pulls `torch`. On Linux, prefer the CPU wheel:

```bash
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

## Adding a server

See [`CONTRIBUTING.md`](CONTRIBUTING.md). Read [`AGENTS.md`](AGENTS.md) first:
the boundary rules there are not stylistic.

## License

Apache-2.0. See [LICENSE](LICENSE).
