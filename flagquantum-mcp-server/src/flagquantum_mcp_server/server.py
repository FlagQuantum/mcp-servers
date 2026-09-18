"""Tool, resource and prompt registration for the FlagQuantum MCP server.

Every tool follows the same three-step shape: validate the caller's input,
delegate to one public FlagQuantum API, and re-shape the result for a model.
Nothing here computes quantum semantics, and nothing reaches the network — the
whole server is a local, deterministic, read-only adapter over the SDK.

Tools never raise across the MCP boundary. A failure comes back as
``{"status": "error", "error": {"code": ..., "message": ...}}`` so a client can
branch on the code instead of parsing prose.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from fastmcp import FastMCP
from fastmcp.tools.tool import ToolAnnotations

from flagquantum_mcp_server._bridge import FlagQuantumUnavailableError, load_sdk
from flagquantum_mcp_server._version import __version__
from flagquantum_mcp_server.analysis import analyze
from flagquantum_mcp_server.circuits import (
    deserialize,
    known_gate_names,
    serialize,
)
from flagquantum_mcp_server.compilation import (
    compare_topologies,
    optimize_circuit,
    route_circuit,
)
from flagquantum_mcp_server.emission import emit_openqasm, emit_qcis
from flagquantum_mcp_server.errors import (
    INTERNAL_ERROR,
    SDK_UNAVAILABLE,
    ToolError,
    error_payload,
)
from flagquantum_mcp_server.planning import plan_execution

INSTRUCTIONS = """\
FlagQuantum is a quantum AI framework whose circuits are built, compiled and
planned locally on CPU, with no credentials and no network access.

A typical session:
1. Build a circuit as a gate list and canonicalize it with
   serialize_circuit_tool (circuit_format="qir"), or bring an existing
   circuit in FlagQuantum IR JSON (circuit_format="ir").
2. Call analyze_circuit_tool to see gate counts and depth.
3. Call optimize_circuit_tool, then route_circuit_tool or
   compare_topologies_tool if the circuit must respect a connectivity limit.
4. Call emit_openqasm_tool or emit_qcis_tool to export the result.
5. Call plan_execution_tool to see how the SDK would execute it.

Read the flagquantum://gate-set resource for the available gate names and
flagquantum://ir-schema for the IR envelope. IR JSON is the canonical format;
every IR payload carries an ir_version and a content_hash that identifies it.
"""

READ_ONLY = ToolAnnotations(readOnlyHint=True, openWorldHint=False)

mcp: FastMCP = FastMCP(
    "FlagQuantum",
    instructions=INSTRUCTIONS,
    version=__version__,
)


def _structured_errors(fn: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
    """Turn any exception into the structured error envelope.

    Keeps the MCP transport healthy when a tool fails: the client receives a
    payload it can reason about rather than a protocol-level fault.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
        try:
            return fn(*args, **kwargs)
        except ToolError as exc:
            return error_payload(exc.code, exc.message)
        except FlagQuantumUnavailableError as exc:
            return error_payload(SDK_UNAVAILABLE, str(exc))
        except Exception as exc:
            return error_payload(INTERNAL_ERROR, f"{type(exc).__name__}: {exc}")

    return wrapper


@mcp.tool(annotations=READ_ONLY)
@_structured_errors
def analyze_circuit_tool(circuit: str, circuit_format: str = "ir") -> dict[str, Any]:
    """Report gate counts, depth and wire usage for one circuit.

    Args:
        circuit: Circuit payload. IR JSON when circuit_format is "ir", or a
            gate-list JSON array when it is "qir".
        circuit_format: "ir" for FlagQuantum IR JSON, "qir" for a gate list.

    Returns:
        The circuit identity (qubit count, IR version, content hash) and the
        SDK's structural analysis: gate_counts, depth, n_instructions,
        wire_usage, two_qubit_gates and related fields.
    """
    return analyze(circuit, circuit_format)


@mcp.tool(annotations=READ_ONLY)
@_structured_errors
def serialize_circuit_tool(
    circuit: str, circuit_format: str = "qir", indent: int | None = None
) -> dict[str, Any]:
    """Canonicalize a circuit into FlagQuantum IR JSON.

    Accepts either input format and always answers with the canonical "ir"
    form, so a circuit written as a gate list gains a version and a content
    hash.

    Args:
        circuit: Circuit payload, in either supported format.
        circuit_format: Format of the input, "ir" or "qir".
        indent: Optional indentation width for the returned JSON.

    Returns:
        The IR JSON plus its ir_version, content_hash, qubit count and
        instruction count.
    """
    return serialize(circuit, circuit_format, indent=indent)


@mcp.tool(annotations=READ_ONLY)
@_structured_errors
def deserialize_circuit_tool(ir_json: str, indent: int | None = None) -> dict[str, Any]:
    """Validate IR JSON and report whether it round-trips unchanged.

    Args:
        ir_json: FlagQuantum IR JSON text.
        indent: Optional indentation width for the canonical re-serialization.

    Returns:
        The circuit identity, the canonical IR JSON, and round_trip_stable —
        whether re-serializing reproduced the same content hash.
    """
    return deserialize(ir_json, indent=indent)


@mcp.tool(annotations=READ_ONLY)
@_structured_errors
def optimize_circuit_tool(circuit: str, circuit_format: str = "ir") -> dict[str, Any]:
    """Apply target-independent optimizations and report what changed.

    Args:
        circuit: Circuit payload, in either supported format.
        circuit_format: Format of the input, "ir" or "qir".

    Returns:
        The optimized IR, a before/after analysis, and instruction_delta.
    """
    return optimize_circuit(circuit, circuit_format)


@mcp.tool(annotations=READ_ONLY)
@_structured_errors
def route_circuit_tool(
    circuit: str,
    circuit_format: str = "ir",
    topology: str = "line",
    rows: int | None = None,
    cols: int | None = None,
    edges: Sequence[Sequence[int]] | None = None,
    strategy: str = "restore_after_each_gate",
    optimize_first: bool = True,
) -> dict[str, Any]:
    """Route a circuit onto a coupling map, inserting SWAPs where needed.

    Args:
        circuit: Circuit payload, in either supported format.
        circuit_format: Format of the input, "ir" or "qir".
        topology: "line", "ring", "grid" or "custom".
        rows: Grid rows; required when topology is "grid".
        cols: Grid columns; required when topology is "grid".
        edges: Explicit wire pairs; required when topology is "custom".
        strategy: "restore_after_each_gate" or "persistent_layout".
        optimize_first: Optimize before routing.

    Returns:
        The routed IR, the coupling map used, a before/after analysis, and the
        SWAP overhead as instruction_delta.
    """
    return route_circuit(
        circuit,
        circuit_format,
        topology=topology,
        rows=rows,
        cols=cols,
        edges=edges,
        strategy=strategy,
        optimize_first=optimize_first,
    )


@mcp.tool(annotations=READ_ONLY)
@_structured_errors
def compare_topologies_tool(
    circuit: str,
    circuit_format: str = "ir",
    topologies: Sequence[str] | None = None,
    strategy: str = "restore_after_each_gate",
    optimize_first: bool = True,
) -> dict[str, Any]:
    """Route one circuit onto several topologies and compare the cost.

    Args:
        circuit: Circuit payload, in either supported format.
        circuit_format: Format of the input, "ir" or "qir".
        topologies: Topology kinds to compare; defaults to line, ring and grid.
        strategy: "restore_after_each_gate" or "persistent_layout".
        optimize_first: Optimize before routing.

    Returns:
        One entry per topology with its routed gate count, depth and SWAP
        overhead, plus the cheapest topology by those measures.
    """
    return compare_topologies(
        circuit,
        circuit_format,
        topologies=topologies,
        strategy=strategy,
        optimize_first=optimize_first,
    )


@mcp.tool(annotations=READ_ONLY)
@_structured_errors
def emit_openqasm_tool(
    circuit: str,
    circuit_format: str = "ir",
    version: float = 3.0,
    result_wires: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Render a circuit as OpenQASM 2.0 or 3.0 text.

    Args:
        circuit: Circuit payload, in either supported format.
        circuit_format: Format of the input, "ir" or "qir".
        version: OpenQASM version, 2.0 or 3.0.
        result_wires: Wires to measure; omit to measure every wire.

    Returns:
        The emitted program as text, with its line count and the source
        circuit's content hash.
    """
    return emit_openqasm(
        circuit,
        circuit_format,
        version=version,
        result_wires=list(result_wires) if result_wires is not None else None,
    )


@mcp.tool(annotations=READ_ONLY)
@_structured_errors
def emit_qcis_tool(circuit: str, circuit_format: str = "ir") -> dict[str, Any]:
    """Render a circuit as QCIS text.

    Args:
        circuit: Circuit payload, in either supported format.
        circuit_format: Format of the input, "ir" or "qir".

    Returns:
        The emitted QCIS program as text. A circuit containing an arbitrary
        matrix gate returns an error, because QCIS cannot express one.
    """
    return emit_qcis(circuit, circuit_format)


@mcp.tool(annotations=READ_ONLY)
@_structured_errors
def plan_execution_tool(
    circuit: str,
    circuit_format: str = "ir",
    options: Mapping[str, Any] | None = None,
    outputs: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Plan execution without running anything.

    Args:
        circuit: Circuit payload, in either supported format.
        circuit_format: Format of the input, "ir" or "qir".
        options: Partial execution options. Supported keys: mode, backend,
            device, target, batch_size, precision, shots, seed,
            memory_limit_bytes, require_gradients, allow_approximate,
            allow_backend_fallback.
        outputs: Requested outputs; each has a "kind" of "counts",
            "expectation", "probabilities" or "samples", plus either a "wires"
            list or, for "expectation", a "pauli" string such as "ZZI".

    Returns:
        The resolved execution contract (mode, backend, device, precision) and
        the plan's summary, which reports depth, state size and whether the
        program is distributed.
    """
    return plan_execution(circuit, circuit_format, options=options, outputs=outputs)


@mcp.resource("flagquantum://version", mime_type="application/json")
def version_resource() -> dict[str, Any]:
    """Versions of this server, the SDK it wraps, and the IR contract."""
    sdk = load_sdk()
    return {
        "server": "flagquantum-mcp-server",
        "server_version": __version__,
        "flagquantum_version": str(sdk.__version__),
        "ir_version": str(sdk.IR_VERSION),
        "python_requires": ">=3.10,<3.13",
    }


@mcp.resource("flagquantum://gate-set", mime_type="application/json")
def gate_set_resource() -> dict[str, Any]:
    """Gate names accepted by the installed FlagQuantum, derived at runtime."""
    names = known_gate_names()
    sdk = load_sdk()
    return {
        "n_gates": len(names),
        "gates": names,
        "flagquantum_version": str(sdk.__version__),
        "note": (
            "Derived from the installed SDK's public Circuit API rather than "
            "hard-coded, so this list always matches the version in use."
        ),
    }


@mcp.resource("flagquantum://ir-schema", mime_type="application/json")
def ir_schema_resource() -> dict[str, Any]:
    """The circuit IR envelope, shown by example from a real serialization."""
    sdk = load_sdk()
    sample = sdk.Circuit(2).h(0).cx(0, 1).to_ir()
    return {
        "kind": "flagquantum.circuit_ir",
        "ir_version": str(sdk.IR_VERSION),
        "example": sample.to_dict(),
        "canonical_json": str(sample.to_json()),
        "content_hash": str(sample.content_hash),
        "notes": [
            "Instructions encode the gate name under the 'opcode' key.",
            "Every IR payload carries 'kind': 'flagquantum.circuit_ir'.",
            "Unknown top-level keys are rejected.",
            "content_hash is the SHA-256 of the canonical JSON.",
        ],
    }


@mcp.prompt()
def build_and_analyze_circuit(circuit_description: str) -> str:
    """Plan a circuit build followed by a structural analysis.

    Args:
        circuit_description: What the circuit should compute.
    """
    return (
        f"Build a FlagQuantum circuit for: {circuit_description}\n\n"
        "1. Read the flagquantum://gate-set resource to confirm the available "
        "gate names.\n"
        "2. Write the circuit as a gate list — a JSON array of objects with "
        '"name" and "index" keys, for example '
        '[{"name": "h", "index": [0]}, {"name": "cx", "index": [0, 1]}].\n'
        '3. Call serialize_circuit_tool with circuit_format="qir" to obtain '
        "the canonical IR and its content hash.\n"
        "4. Call analyze_circuit_tool on the result and report the gate counts "
        "and depth.\n"
        "5. If the circuit is parameterized, say which gates carry parameters "
        "and what values you assumed."
    )


@mcp.prompt()
def compile_for_topology(circuit: str, topology: str = "line") -> str:
    """Plan a compile-and-route pass for a circuit that must fit a topology.

    Args:
        circuit: The circuit, as IR or gate-list JSON.
        topology: Target connectivity, "line", "ring" or "grid".
    """
    return (
        f"Route this circuit onto a {topology} topology:\n\n{circuit}\n\n"
        "1. Call optimize_circuit_tool first and record the instruction delta.\n"
        "2. Call compare_topologies_tool to see the cost on line, ring and "
        "grid before committing to one.\n"
        "3. Call route_circuit_tool for the chosen topology and report the SWAP "
        "overhead as instruction_delta.\n"
        "4. Report the routed circuit's content_hash so the result is "
        "identifiable.\n"
        "5. State the depth before and after routing. Do not describe the "
        "routed circuit as equivalent to the source if the SWAP overhead is "
        "non-zero."
    )


@mcp.prompt()
def export_circuit(circuit: str, target_format: str = "openqasm3") -> str:
    """Plan an export of a circuit to a text interchange format.

    Args:
        circuit: The circuit, as IR or gate-list JSON.
        target_format: "openqasm3", "openqasm2" or "qcis".
    """
    return (
        f"Export this circuit as {target_format}:\n\n{circuit}\n\n"
        "1. Call serialize_circuit_tool first so you hold the canonical IR and "
        "its content hash.\n"
        "2. Call emit_openqasm_tool for openqasm3 or openqasm2, and "
        "emit_qcis_tool for qcis.\n"
        "3. If the circuit contains an arbitrary matrix gate, QCIS export "
        "fails by design — report that rather than silently substituting a "
        "different gate.\n"
        "4. Report the emitted text verbatim, plus the source content hash."
    )
