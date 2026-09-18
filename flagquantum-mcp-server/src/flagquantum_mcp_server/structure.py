"""Describe a circuit's parallelism and a topology's connectivity.

``describe_layers`` answers the question behind ``depth``: which gates can run
at the same time, and therefore what the critical path is. ``describe_topology``
gives the caller the connectivity facts it needs to reason *before* routing,
so it can pick a topology instead of trying one and reading the SWAP count.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from flagquantum_mcp_server._bridge import load_attribute
from flagquantum_mcp_server.analysis import summarize_ir
from flagquantum_mcp_server.circuits import (
    CircuitFormat,
    CircuitPayload,
    resolve_ir,
)
from flagquantum_mcp_server.compilation import (
    COMPILER_MODULE,
    build_coupling_map,
)
from flagquantum_mcp_server.errors import ToolInputError

# A wire pair has exactly two endpoints.
PAIR_ARITY = 2

# A topology query returns at most this many shortest paths.
MAX_PATHS = 16


def describe_layers(
    circuit: CircuitPayload, circuit_format: CircuitFormat = "ir"
) -> dict[str, Any]:
    """Decompose a circuit into the layers it can execute in.

    Args:
        circuit: Serialized circuit, in either supported format.
        circuit_format: Either ``"ir"`` or ``"qir"``.

    Returns:
        A payload listing each layer's gates and wires, plus the layer count,
        which is the circuit's depth.
    """
    ir = resolve_ir(circuit, circuit_format)
    schedule_layers = load_attribute(COMPILER_MODULE, "schedule_layers")
    layers = schedule_layers(ir)

    return {
        "status": "success",
        "n_layers": len(layers),
        "layers": [
            {
                "index": index,
                "gates": [
                    {
                        "name": str(instruction.name),
                        "wires": [int(wire) for wire in instruction.wires],
                    }
                    for instruction in layer
                ],
                "wires": sorted({int(wire) for instruction in layer for wire in instruction.wires}),
            }
            for index, layer in enumerate(layers)
        ],
        "circuit": summarize_ir(ir)["circuit"],
        "note": (
            "Gates inside one layer are independent and can run concurrently. "
            "n_layers equals the circuit depth; a wider layer means more "
            "parallelism, not more work."
        ),
    }


def describe_topology(
    *,
    n_qubits: int,
    topology: str = "line",
    rows: int | None = None,
    cols: int | None = None,
    edges: Sequence[Sequence[int]] | None = None,
    pairs: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    """Describe a coupling map: its edges, degrees and pairwise distances.

    Args:
        n_qubits: Number of wires the topology must hold.
        topology: ``"line"``, ``"ring"``, ``"grid"`` or ``"custom"``.
        rows: Grid rows; required for ``"grid"``, or derived when both are
            omitted.
        cols: Grid columns; see ``rows``.
        edges: Explicit wire pairs; required for ``"custom"``.
        pairs: Specific wire pairs to measure. Defaults to adjacent wires only,
            so the response stays bounded on a wide device.

    Returns:
        A payload with the edge list, each wire's neighbours, and the shortest
        path between the requested pairs.

    Raises:
        ToolError: If the topology specification is invalid or a pair is
            unreachable.
    """
    if n_qubits < 1:
        raise ToolInputError(f"n_qubits must be positive; received {n_qubits}.")

    coupling_map = build_coupling_map(topology, n_wires=n_qubits, rows=rows, cols=cols, edges=edges)
    requested = _pairs(pairs, n_qubits)

    adjacency: list[dict[str, Any]] = []
    for wire in range(n_qubits):
        neighbours = [int(item) for item in coupling_map.neighbors(wire)]
        adjacency.append({"wire": wire, "neighbours": neighbours, "degree": len(neighbours)})

    paths: list[dict[str, Any]] = []
    for left, right in requested:
        path = [int(item) for item in coupling_map.shortest_path(left, right)]
        paths.append({"from": left, "to": right, "distance": len(path) - 1, "path": path})

    edges_out = [[int(a), int(b)] for a, b in coupling_map.edges]
    return {
        "status": "success",
        "topology": {
            "kind": topology,
            "n_wires": int(coupling_map.n_wires),
            "n_edges": len(edges_out),
            "edges": edges_out,
        },
        "adjacency": adjacency,
        "pairs": paths,
        "note": (
            "A two-qubit gate whose wires are not an edge forces SWAP insertion; "
            "route_circuit_tool reports that cost as instruction_delta. "
            "'distance' is the number of SWAPs a naive route would need."
        ),
    }


def _pairs(pairs: Sequence[Sequence[int]] | None, n_qubits: int) -> list[tuple[int, int]]:
    """Validate the requested wire pairs, defaulting to adjacent wires."""
    if pairs is None:
        return [(wire, wire + 1) for wire in range(n_qubits - 1)]
    if len(pairs) > MAX_PATHS:
        raise ToolInputError(
            f"At most {MAX_PATHS} pairs can be measured per call; received {len(pairs)}."
        )
    resolved: list[tuple[int, int]] = []
    for pair in pairs:
        items = list(pair)
        if len(items) != PAIR_ARITY:
            raise ToolInputError(f"Each pair needs exactly two wires; received {items!r}.")
        left, right = int(items[0]), int(items[1])
        for wire in (left, right):
            if not 0 <= wire < n_qubits:
                raise ToolInputError(f"Wire {wire} is outside 0..{n_qubits - 1} for this topology.")
        resolved.append((left, right))
    return resolved
