# HTTP transport for `flagquantum-mcp-server` — design

Status: proposed
Date: 2026-09-19
Author: FlagQAI integration work (B1 of the quantum chat slice)

## The problem this solves

The server publishes one entry point, and it is stdio:

```python
mcp.run(transport="stdio", show_banner=False)   # flagquantum_mcp_server/__init__.py:31
```

A stdio server can only be launched by the client that owns its process. FlagQAI's
MCP client gateway speaks **Streamable HTTP only** — its sole adapter is
`StreamableHttpMcpClient` (`FlagQAI-release-integration/backend/src/tovx_control/adapters/mcp/streamable_http.py:83`),
and the five services it already consumes are all HTTP apps. So the server is
unreachable from the one consumer that wants it, and the only way to change that
is an entry point that listens.

`AGENTS.md` rule 4 ("**No network egress**, no credentials, no hardware") is **not
crossed by this change**. The rule governs what the tools *do* — outbound calls,
tokens, provider submission, paid resources. A process that listens on a socket
and answers its caller makes no outbound call. The rule stays as written; the
public phrasing elsewhere does not (see *Documentation*).

## What was measured before designing

All against `main` (`c90b17c`) in this repository's `.venv` (fastmcp 3.4.7,
uvicorn 0.53.0, starlette 1.6.0). Every claim below was run, not read.

**The capability already exists.** `mcp.http_app()` returns a
`fastmcp.server.http.StarletteWithLifespan`; served under uvicorn, a real
`fastmcp` `Client(StreamableHttpTransport(...))` got **17 tools** over HTTP, and
`simulate_circuit_tool` returned real counts
(`[{"00": 119, "11": 137}]` for a 2-qubit Bell circuit, 256 shots, seed 7) with
`execution_path: local_statevector` / `platform_provider: pytorch_cpu`.

**`mcp.run` already accepts it.** `Transport` is
`Literal['stdio', 'http', 'sse', 'streamable-http']` and `run` forwards
`**transport_kwargs`. Measured:

```python
mcp.run(transport="http", host="127.0.0.1", port=61501, show_banner=False)
# → uvicorn running on http://127.0.0.1:61501 ; POST /mcp → 200, mcp-session-id set
```

**No new runtime dependency.** `uvicorn>=0.35` and `starlette>=1.0.1` already
arrive via `fastmcp-slim[server]`, which `fastmcp` (already a dependency) pulls in.

**Host validation is off by default, and the obvious way to turn it on does
nothing.** Measured with `httpx(trust_env=False)` (a host proxy otherwise
intercepts non-localhost `Host` values and returns a misleading 502):

| configuration | `Host: 127.0.0.1:<port>` | `Host: quantum-mcp:8105` | `Host: evil.example.com` |
|---|---|---|---|
| default (no kwargs) | 200 | 200 | **200** |
| `http_app(allowed_hosts=["quantum-mcp:8105"])` | 200 | 200 | **200** |
| `settings.http_host_origin_protection = True` + `settings.http_allowed_hosts = [...]` | 200 | 200 | **421** |

`http_app(allowed_hosts=...)` **adds to a list that is never consulted while
protection is off** — it reads as a security configuration and enforces nothing.
The working configuration is the settings route with protection explicitly
enabled; a rejected `Host` answers **421 Misdirected Request**.

## Design

**`main()` gains a transport choice; stdio stays the default.**

```
flagquantum-mcp-server                          # stdio, unchanged
flagquantum-mcp-server --transport http --host 0.0.0.0 --port 8105
```

- Parsed with `argparse`, defaults `--transport stdio`, `--host 127.0.0.1`,
  `--port 8105`. **The default host is loopback**; binding to `0.0.0.0` is
  explicit, so a package run by hand on a laptop does not become reachable.
- `http`/`streamable-http` map to `mcp.run(transport=..., host=..., port=..., show_banner=False)`.
- **Default behaviour is byte-identical to today.** No existing client, no CI
  job, and no stdio handshake test changes.

**Host/origin protection is configured explicitly, through `fastmcp.settings`:**

```python
fastmcp.settings.http_host_origin_protection = True
fastmcp.settings.http_allowed_hosts = [...]      # from --allowed-host, repeatable
fastmcp.settings.http_allowed_origins = [...]
```

Measured to be the only form that enforces. The allowlist is **required** when
binding a non-loopback host, and the server refuses to start otherwise with a
message naming `--allowed-host` — a listening socket with no host check on a
shared machine is the failure this guards, and the `allowed_hosts=` kwarg is a
trap that must not be the documented answer.

## Testing

1. **A process test over real HTTP**, modelled on `test_server_process.py` (which
   does the same over stdio): start the console script with `--transport http` on
   an ephemeral port, connect a real `Client(StreamableHttpTransport(...))`,
   assert the full `EXPECTED_TOOLS` surface, and round-trip one tool call.
2. **Host policy**: with `--allowed-host quantum-mcp:8105`, a request carrying
   `Host: evil.example.com` is rejected **421** and the allowed host is served.
   Without `--allowed-host` on a non-loopback bind, startup fails with a message
   naming the flag.
3. **Default preserved**: no `--transport` still speaks stdio (the existing
   handshake job already covers this; keep it).
4. **Mutation**: set `http_host_origin_protection = False` and confirm test 2
   goes red; drop `--allowed-host` from the wiring and confirm the startup
   refusal goes red. A test that cannot fail is not a test — this repository's
   `test_tool_wiring.py` learned that the hard way.

## Documentation

- `flagquantum-mcp-server/README.md:10` and `:348` and root `README.md:102` —
  "no network access" must become **no outbound network access**, and gain the
  one line that says how to listen. Leaving it is a false statement about a
  process that now accepts connections.
- `flagquantum-mcp-server/server.json:5` — `"Build, compile, run and export
  FlagQuantum circuits locally. No credentials, no network."` The Registry caps
  `description` at **100 characters** (currently 83) and
  `tests/test_versions.py` pins it; the replacement must be re-counted, and the
  `latest` Registry entry only changes when the new version is published.
- `AGENTS.md` rule 4 needs a sentence, in the same style as the existing
  in-process-simulation paragraph, stating that a listening transport is inside
  the line because it makes no egress. **The rule itself is not amended.**

## Release

One version bump (per the repo's own rule: the release commit changes the
version and nothing else), then

```bash
gh workflow run publish-pypi.yml         --repo FlagQuantum/mcp-servers -f package=flagquantum-mcp-server
gh workflow run publish-mcp-registry.yml --repo FlagQuantum/mcp-servers -f package=flagquantum-mcp-server
```

**`train_parameters_tool` is not part of this release.** It is complete on
`main` and unreleased; shipping it here would bundle two unrelated reasons into
one publication, against the policy this repo wrote for itself. It rides the
next release.

## Acceptance

- From a clean venv with the **published** wheel: `--transport http` serves the
  full surface over HTTP and a tool call round-trips.
- `--transport` omitted still speaks stdio.
- A bogus `Host` is rejected 421 when an allowlist is configured.
- `ruff`, `ruff format`, `mypy --strict`, unit + integration suites, and the
  wheel handshake CI job are green.

## Not doing

- **Not changing the stdio default**, and not touching the stdio handshake job.
- **Not adding a remote-submission tool.** `QuafuProvider` still needs
  credentials, spends paid quota, and is irreversible; the SDK's own
  `SERVICE_INTEGRATION.md` puts paid-resource submission on the consuming
  application. This design adds a transport, not a capability.
- **Not adding a `flagquantum` runtime dependency** or changing its range.
- **Not moving anything into the FlagQuantum repository.** The single-directional
  dependency is that project's retirement condition
  (`contracts/long-horizon-architecture-v1.json:124-125`).

## Unverified

- **The container never ran.** Loopback binding was exercised on macOS; the
  `0.0.0.0` bind, the compose service, and the gateway's actual discovery
  against it were not. That is B2's acceptance, not this design's.
- **`streamable-http` vs `http`** was not compared beyond both existing in the
  `Transport` literal; `run(transport="http")` was the one measured.
- **No load or concurrency testing.** One client, one call at a time.
- **`--port` reuse / launchd-style supervision** was not considered; the
  container is expected to own the process.
