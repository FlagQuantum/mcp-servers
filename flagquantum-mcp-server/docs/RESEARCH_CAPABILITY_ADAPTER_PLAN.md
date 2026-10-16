# Research capability MCP adapter plan

Status: design only. No research capability tool or resource is registered by
this change.

The corresponding FlagQuantum API is implemented on its development branch but
is still pending API-owner and Services-domain-owner approval and remains an
experimental `flagquantum.services` surface. This repository's policy forbids
exposing an experimental or development-evidence capability as an MCP tool.
Registration therefore remains blocked until an upstream release contains the
approved surface with at least `production_supported` maturity.

The upstream repository now names the exact pending authorization token,
`approve API_CHANGE_PROPOSAL_070_RESEARCH_CAPABILITY_SERVICES`, and the record
target `docs/api-changes/FQ-RESEARCH-CAPABILITY-SERVICES-20261013.md`. Neither is
an approval until supplied and recorded by the authorized owners.

The machine-readable companion is
[`research-capability-adapter-v1-candidate.json`](../contracts/research-capability-adapter-v1-candidate.json).

## Ownership boundary

FlagQuantum owns capability identity, typed scientific inputs, applicability,
resource estimates, execution, scientific validators, error semantics,
Evidence Bundle structure, provider selection, and fallback policy.

This MCP server will own only MCP registration, JSON request decoding, calls to
the released public API, lossless JSON response encoding, protocol request-size
and local-time safety limits, and protocol-level error mapping. An independent
client or another control plane owns natural-language interpretation, Research TaskSpec,
research planning, candidate selection, and final scientific acceptance.

No VQE, exact diagonalization, variance calculation, scientific threshold,
resource estimate, provider router, retry, or fallback implementation belongs
in this repository.

## Proposed surface after activation

| MCP tool | FlagQuantum operation | Adapter work only |
| --- | --- | --- |
| `search_capabilities_tool` | `search_research_capabilities` | Decode a structured query; serialize candidates |
| `describe_capability_tool` | `describe_research_capability` | Decode identity; serialize the Manifest |
| `estimate_capability_tool` | `estimate_research_capability` | Decode a typed request; serialize the estimate |
| `validate_capability_input_tool` | `validate_research_capability_input` | Decode a typed request; preserve the report |
| `execute_local_capability_tool` | `execute_research_capability` | Forward provider order and fallback authority; preserve the Evidence Bundle |
| `validate_evidence_tool` | `validate_research_evidence` | Decode evidence; preserve the validation report |

The candidate contract owns only each MCP tool's closed top-level input shape.
Nested `query`, `request`, `options`, and `evidence` objects remain opaque at
the adapter boundary and are passed to FlagQuantum for schema lookup, typed
decoding, applicability checks, estimation, and scientific validation. This
prevents a generated FastMCP schema from silently changing the consumer
contract without creating a second copy of the quantum-domain JSON Schemas.

Every research tool uses the repository's closed transport envelope. Success is
exactly `{"status":"success","result":<upstream JSON value>}` and adapter failure
is exactly `{"status":"error","error":<upstream structured error>}`. The
adapter may add neither sibling fields nor a reduced error vocabulary. An
Evidence Bundle whose domain status is `failed`, `validation_failed`, or
`unsupported` is still a successfully transported upstream result; MCP must not
turn it into a transport error or erase the generated evidence.

Pre-result exceptions are a separate protocol concern. Public `KeyError`,
`TypeError`, and `ValueError` inputs map to the repository's `INVALID_INPUT`;
SDK import failures map to `SDK_UNAVAILABLE`; any other ordinary exception maps
to a sanitized `INTERNAL_ERROR`. The adapter never catches process-control
exceptions. This mapping adds no quantum semantics: validation reports and
Evidence Bundle errors with `stage`, `retryable`, context, generated evidence,
and suggested actions stay untouched inside successful result transport.

The shared success envelope permits an object or array because capability
search serializes the upstream tuple as a JSON array. Each tool separately pins
its result kind in the candidate contract: search returns an array; describe,
estimate, input validation, execution, and evidence validation return objects.

The candidate also pins the model-visible descriptions and the ordered
search → describe → estimate → input validation → local execution → evidence
validation workflow. Safety and scientific constraints appear in the tool
descriptions an autonomous client reads, not only in an optional prompt:
execution is local and credential-free, multiple providers and fallback require
explicit authority, upstream preflight failures stop the workflow, and neither
an MCP success envelope nor an optimizer result is a scientific conclusion.

Planned resources are:

```text
flagquantum://capabilities/index
flagquantum://capabilities/{capability_id}/{version}
flagquantum://schemas/{schema_id}
```

The index maps to `search_research_capabilities`, a manifest maps to
`describe_research_capability`, and every schema URI maps directly to
`describe_research_schema`. Resources must be generated from the installed
FlagQuantum package. The MCP package must not carry a second editable copy of
Manifest or JSON Schema facts. The generic schema resource covers every request,
result, report, structured error, and Evidence Bundle identity published by the
installed package; the adapter does not curate a smaller independent list at
runtime.

## Provider routing and fallback

Provider selection remains a FlagQuantum decision. The future execution tool may
decode `provider_order` and `allow_provider_fallback`, but it must forward both
values verbatim. It may not reorder providers, probe dependencies independently,
retry a failed execution, skip a recorded attempt, or decide that a scientific
validation failure deserves fallback.

The current upstream candidate defaults to `flagquantum-native`, requires
explicit authorization for more than one provider, and advances only past an
unknown or unavailable adapter. The MCP response must preserve
`execution.provider_order`, `fallback_used`, every ordered `attempt`, and every
provider error. These fields are part of the upstream content digest.

FlagQuantum also returns a content-hashed `metrics.provider_route` assessment
containing caller-order rank, availability, selection reason, adapter version,
and estimated backend executions. MCP must serialize this object losslessly. It
must not recompute scores, reorder candidates, probe dependencies, or treat the
estimate as an MCP-owned budget decision.

The upstream development candidate now installs `qiskit-aer` and
`pennylane-lightning` as optional local providers while retaining the native
default. This MCP package neither depends on either vendor SDK nor probes it: it
forwards requested provider IDs, and FlagQuantum alone reports whether optional
dependencies are available, executes the adapter, and records
`backend_executions`. The approved local MCP surface is now registered; provider
availability remains an upstream FlagQuantum decision recorded in the Evidence
Bundle.

## Upstream dependency inventory

The adapter depends on exactly six scientific-operation names and two
read-only schema-resource names from
`flagquantum.services`: search, describe, estimate, input validation, execution,
evidence validation, schema listing, and exact schema lookup.
`tests/test_api_contract.py` records the reviewed public tier and pins
signatures and schema identities. The package dependency remains a released
version range; a Git URL or commit pin is not acceptable.

The current package range is `flagquantum>=0.3,<0.4`. The published
`0.3.0rc3` and `0.3.0` wheels do not contain the approved research service API,
so no compatible range is approved yet. Adapter activation remains blocked
until a future installed wheel passes the API and process checks.

## Existing overlap audit

The current MCP server contains Hamiltonian term validation and a calibrated
training budget for its pre-existing generic planning and training tools:

- `planning.validate_pauli_terms` validates the JSON vocabulary those tools
  already accept;
- `training.hamiltonian_from_terms` constructs their SDK objective;
- `training._check_budget` and limits in `limits.py` protect an MCP request from
  monopolizing the local process.

Those functions are not the research capability contract. They remain for the
existing tools, but a future capability adapter must neither reuse nor extend
them to decide scientific validity, applicability, convergence, or resource
cost. It passes the request to FlagQuantum and preserves FlagQuantum's result.
Protocol safety limits may reject a request earlier, but must be labelled as MCP
limits and must not be reported as FlagQuantum resource estimates.

## Compatibility and activation gates

Activation requires all of these to remain true:

1. the upstream API proposal is approved by its owner;
2. the maturity entry is `production_supported` or `release_certified`;
3. an upstream release and compatible version range are known;
4. the six operation names, two schema-resource names, and schema identities are
   pinned in the API contract test;
5. unknown fields and incompatible schema versions fail closed;
6. defaults and optional arguments are forwarded without reinterpretation;
7. every Evidence Bundle field survives direct Python → MCP → JSON comparison;
8. `code`, `stage`, `retryable`, context, generated evidence, and suggested
   actions survive error mapping;
9. no tool performs network egress, reads credentials, or submits hardware;
10. the registered tool and resource surface changes only in the activation
    commit, with unit, process, example, and wheel verification.
11. cross-repository parity confirms that every required schema identity is
    present in the installed FlagQuantum catalog and no adapter-owned schema
    document exists.
12. provider order, fallback authorization, attempt order, selected provider,
    and fallback history match the direct Python result byte-for-byte after JSON
    normalization.

Until then, the existing `EXPECTED_TOOLS` and `EXPECTED_RESOURCES` declarations
must remain unchanged. This is intentional evidence that the design has not
silently exposed an unapproved scientific API.

The first three gates are executable rather than inferred from prose:

```bash
python tools/check_research_adapter_prerequisites.py \
  --flagquantum-source-root /path/to/FlagQuantum/source \
  --flagquantum-wheel /path/to/published/flagquantum.whl
```

The checker reads the upstream candidate contract, approval record,
`capability-maturity.toml`, and package version together with this adapter
contract. It exits non-zero and emits a JSON list of blockers unless the owner
decisions use the pinned approved state, the record contains `Decision:
approved`, the maturity is allowed, and the explicitly recorded release and
compatible range agree, and the wheel contains the research service modules
and public operation exports. A passing result only permits the activation
change to begin; it does not register any tool/resource or prove the remaining
forwarding, losslessness, client-discovery, and direct-Python/MCP equivalence
gates. The supplied wheel must be verified against the actual published file,
not merely a local build with the same version number.
