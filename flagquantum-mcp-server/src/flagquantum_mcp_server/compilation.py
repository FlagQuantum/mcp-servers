"""Optimize a circuit and route it onto a hardware topology.

The compiler surface used here is ``flagquantum.compiler``, which the SDK
describes as its "stable expert compiler interface" and exports through an
explicit ``__all__``. It is public but is not part of the frozen
``stable_exports`` snapshot, so this module is the second-weakest contract in
the server after the emitters. Tests pin every name it touches.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from flagquantum_mcp_server import limits
from flagquantum_mcp_server._bridge import load_attribute
from flagquantum_mcp_server.analysis import summarize_ir
from flagquantum_mcp_server.circuits import (
    CircuitFormat,
    CircuitPayload,
    ir_to_json,
    resolve_ir,
)
from flagquantum_mcp_server.errors import (
    ToolInputError,
    ToolLimitError,
    UnsupportedFormatError,
)

COMPILER_MODULE = "flagquantum.compiler"

ROUTING_STRATEGIES: tuple[str, ...] = ("restore_after_each_gate", "persistent_layout")
TOPOLOGY_KINDS: tuple[str, ...] = ("line", "ring", "grid", "custom")
TOPOLOGY_RENDERERS: tuple[str, ...] = ("line", "ring", "grid")

# Every coupling-map edge names exactly two wires.
EDGE_ARITY = 2


def optimize_circuit(
    circuit: CircuitPayload, circuit_format: CircuitFormat = "ir"
) -> dict[str, Any]:
    """Apply target-independent optimizations and report what changed.

    Args:
        circuit: Serialized circuit, as JSON text or a decoded payload.
        circuit_format: Either ``"ir"`` or ``"qir"``.

    Returns:
        A payload with the optimized IR and a before/after analysis.

    Raises:
        ToolError: If the payload is invalid or exceeds a configured bound.
    """
    before = resolve_ir(circuit, circuit_format)
    optimize = load_attribute(COMPILER_MODULE, "optimize")
    after = optimize(before)
    return {
        "status": "success",
        "transformation": "optimize",
        "before": summarize_ir(before),
        "after": summarize_ir(after),
        **_delta(before, after),
        **_ir_block(after),
    }


def route_circuit(
    circuit: CircuitPayload,
    circuit_format: CircuitFormat = "ir",
    *,
    topology: str = "line",
    rows: int | None = None,
    cols: int | None = None,
    edges: Sequence[Sequence[int]] | None = None,
    strategy: str = "restore_after_each_gate",
    optimize_first: bool = True,
) -> dict[str, Any]:
    """Route a circuit onto a coupling map, inserting SWAPs as needed.

    Args:
        circuit: Serialized circuit, as JSON text or a decoded payload.
        circuit_format: Either ``"ir"`` or ``"qir"``.
        topology: One of ``"line"``, ``"ring"``, ``"grid"`` or ``"custom"``.
        rows: Grid rows, required when ``topology`` is ``"grid"``.
        cols: Grid columns, required when ``topology`` is ``"grid"``.
        edges: Explicit wire pairs, required when ``topology`` is ``"custom"``.
        strategy: Routing strategy, ``"restore_after_each_gate"`` or
            ``"persistent_layout"``.
        optimize_first: Optimize before routing.

    Returns:
        A payload with the routed IR, the coupling map used, and a before/after
        analysis.

    Raises:
        ToolError: If the payload is invalid, the topology specification is
            incomplete, or a bound is exceeded.
    """
    before = resolve_ir(circuit, circuit_format)
    coupling_map = build_coupling_map(
        topology, n_wires=int(before.n_wires), rows=rows, cols=cols, edges=edges
    )
    route_to_topology = load_attribute(COMPILER_MODULE, "route_to_topology")
    after = route_to_topology(
        _prepare(before, optimize_first), coupling_map, strategy=_strategy(strategy)
    )
    return {
        "status": "success",
        "transformation": "route_to_topology",
        "topology": _describe_coupling_map(coupling_map),
        "strategy": strategy,
        "optimize_first": optimize_first,
        "before": summarize_ir(before),
        "after": summarize_ir(after),
        **_delta(before, after),
        **_ir_block(after),
    }


def compare_topologies(
    circuit: CircuitPayload,
    circuit_format: CircuitFormat = "ir",
    *,
    topologies: Sequence[str] | None = None,
    strategy: str = "restore_after_each_gate",
    optimize_first: bool = True,
) -> dict[str, Any]:
    """Route one circuit onto several topologies and compare the cost.

    Answers the question an agent actually has — which connectivity is cheapest
    for this circuit — without the caller issuing one call per topology.

    Args:
        circuit: Serialized circuit, as JSON text or a decoded payload.
        circuit_format: Either ``"ir"`` or ``"qir"``.
        topologies: Topology kinds to evaluate. Defaults to every renderer.
        strategy: Routing strategy.
        optimize_first: Optimize before routing.

    Returns:
        A payload listing each topology's routed cost, plus the cheapest one.

    Raises:
        ToolError: If the request is invalid or exceeds a configured bound.
    """
    requested = list(TOPOLOGY_RENDERERS) if topologies is None else list(topologies)
    if not requested:
        raise ToolInputError("topologies must contain at least one topology kind.")
    if len(requested) > limits.max_compare_topologies():
        raise ToolLimitError(
            f"compare_topologies accepts at most {limits.max_compare_topologies()} "
            f"topologies per call; received {len(requested)}. "
            "Raise FLAGQUANTUM_MCP_MAX_COMPARE_TOPOLOGIES to allow more."
        )
    for kind in requested:
        if kind not in TOPOLOGY_RENDERERS:
            raise UnsupportedFormatError(
                f"topology must be one of {list(TOPOLOGY_RENDERERS)}; "
                f"received {kind!r}. Custom edge lists cannot be compared by name."
            )

    before = resolve_ir(circuit, circuit_format)
    source = _prepare(before, optimize_first)
    route_to_topology = load_attribute(COMPILER_MODULE, "route_to_topology")
    resolved_strategy = _strategy(strategy)
    n_wires = int(before.n_wires)

    results: list[dict[str, Any]] = []
    for kind in requested:
        coupling_map = build_coupling_map(kind, n_wires=n_wires)
        after = route_to_topology(source, coupling_map, strategy=resolved_strategy)
        results.append(
            {
                "topology": _describe_coupling_map(coupling_map),
                **_delta(before, after),
                **summarize_ir(after)["analysis"],
            }
        )

    cheapest = min(results, key=lambda item: (item["n_instructions"], item["depth"]))
    return {
        "status": "success",
        "transformation": "compare_topologies",
        "strategy": strategy,
        "optimize_first": optimize_first,
        "source": summarize_ir(before),
        "topologies": results,
        "cheapest": cheapest["topology"],
    }


def build_coupling_map(
    topology: str,
    *,
    n_wires: int,
    rows: int | None = None,
    cols: int | None = None,
    edges: Sequence[Sequence[int]] | None = None,
) -> Any:
    """Build a ``CouplingMap`` from a caller-friendly specification.

    Args:
        topology: One of ``"line"``, ``"ring"``, ``"grid"`` or ``"custom"``.
        n_wires: Number of wires in the source circuit.
        rows: Grid rows, required for ``"grid"``.
        cols: Grid columns, required for ``"grid"``.
        edges: Explicit wire pairs, required for ``"custom"``.

    Returns:
        A ``flagquantum.compiler.CouplingMap``.

    Raises:
        UnsupportedFormatError: If ``topology`` is unknown.
        ToolInputError: If the specification is incomplete or inconsistent with
            the circuit width.
    """
    if topology not in TOPOLOGY_KINDS:
        raise UnsupportedFormatError(
            f"topology must be one of {list(TOPOLOGY_KINDS)}; received {topology!r}."
        )
    coupling_map_cls = load_attribute(COMPILER_MODULE, "CouplingMap")
    if topology == "line":
        return coupling_map_cls.line(n_wires)
    if topology == "ring":
        return coupling_map_cls.ring(n_wires)
    if topology == "grid":
        if (rows is None) != (cols is None):
            raise ToolInputError(
                "topology='grid' needs both rows and cols, or neither; "
                "supply both to pin the shape, or omit both to derive it."
            )
        if rows is None and cols is None:
            rows, cols = grid_shape(n_wires)
        assert rows is not None and cols is not None
        if rows < 1 or cols < 1:
            raise ToolInputError(f"grid dimensions must be positive; got {rows}x{cols}.")
        if rows * cols != n_wires:
            raise ToolInputError(
                f"A {rows}x{cols} grid holds {rows * cols} wires, but the circuit "
                f"has {n_wires}. Grid dimensions must multiply to the circuit width."
            )
        return coupling_map_cls.grid(rows, cols)
    return coupling_map_cls(n_wires, _normalize_edges(edges, n_wires))


def grid_shape(n_wires: int) -> tuple[int, int]:
    """Choose the most square grid that holds exactly ``n_wires`` wires.

    Derived rather than fixed, so a caller who does not care about the grid
    aspect ratio still gets a valid one. A prime width falls back to a single
    row, which is a line.

    Args:
        n_wires: Number of wires the grid must hold.

    Returns:
        A ``(rows, cols)`` pair whose product is ``n_wires``.
    """
    best = (1, n_wires)
    for rows in range(2, int(n_wires**0.5) + 1):
        if n_wires % rows == 0:
            best = (rows, n_wires // rows)
    return best


def _prepare(ir: Any, optimize_first: bool) -> Any:
    """Optionally optimize a circuit before routing it."""
    if not optimize_first:
        return ir
    optimize = load_attribute(COMPILER_MODULE, "optimize")
    return optimize(ir)


def _normalize_edges(edges: Sequence[Sequence[int]] | None, n_wires: int) -> list[tuple[int, int]]:
    """Validate an explicit edge list against the circuit width."""
    if not edges:
        raise ToolInputError("topology='custom' requires a non-empty edges list.")
    normalized: list[tuple[int, int]] = []
    for edge in edges:
        pair = list(edge)
        if len(pair) != EDGE_ARITY:
            raise ToolInputError(f"Each edge must have exactly two wires; received {pair!r}.")
        left, right = int(pair[0]), int(pair[1])
        if left == right:
            raise ToolInputError(f"An edge cannot join a wire to itself: {left}.")
        if not (0 <= left < n_wires and 0 <= right < n_wires):
            raise ToolInputError(
                f"Edge {(left, right)} references a wire outside 0..{n_wires - 1}."
            )
        normalized.append((left, right))
    return normalized


def _describe_coupling_map(coupling_map: Any) -> dict[str, Any]:
    """Render a coupling map as a plain mapping."""
    return {
        "n_wires": int(coupling_map.n_wires),
        "n_edges": len(coupling_map.edges),
        "edges": [list(edge) for edge in coupling_map.edges],
    }


def _strategy(strategy: str) -> str:
    """Validate a routing strategy name."""
    if strategy not in ROUTING_STRATEGIES:
        raise UnsupportedFormatError(
            f"strategy must be one of {list(ROUTING_STRATEGIES)}; received {strategy!r}."
        )
    return strategy


def _delta(before: Any, after: Any) -> dict[str, Any]:
    """Report the instruction-count change a transformation produced."""
    return {
        "instruction_delta": len(after.instructions) - len(before.instructions),
        "content_hash_changed": str(before.content_hash) != str(after.content_hash),
    }


def _ir_block(ir: Any) -> dict[str, Any]:
    """Return the canonical IR JSON for a transformed circuit."""
    return {
        "ir_json": ir_to_json(ir),
        "n_qubits": int(ir.n_wires),
        "n_instructions": len(ir.instructions),
        "ir_version": str(ir.version),
        "content_hash": str(ir.content_hash),
    }
