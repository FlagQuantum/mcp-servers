"""Render a circuit as an ASCII diagram.

Text drawing is the one presentation the SDK offers that needs no optional
extra: ``Circuit.draw(format="mpl")`` requires matplotlib, which is not a
dependency of this server, so only ``format="text"`` is exposed.

The value for an agent is not decoration. A diagram shows gate order and wire
pairing at a glance, which is often what a user actually asked for, and it
costs one call instead of a chain of analyses.
"""

from __future__ import annotations

from typing import Any

from flagquantum_mcp_server import limits
from flagquantum_mcp_server._bridge import load_attribute
from flagquantum_mcp_server.circuits import (
    CircuitFormat,
    CircuitPayload,
    circuit_from_ir,
    resolve_ir,
)
from flagquantum_mcp_server.errors import ToolLimitError, UnsupportedFormatError

# ``flagquantum.drawer`` is not reachable as an attribute of the package root,
# so it is resolved by module path. Like the emitters, it is tier 3.
DRAWER_MODULE = "flagquantum.drawer"
DRAW_FUNCTION = "draw"

# The SDK's styles (``available_styles()``) only reach the matplotlib backend,
# which needs the ``viz`` extra and is therefore not available here. The text
# drawer's own knobs are what this tool exposes instead.
TEXT_KWARGS = ("decimals", "max_length", "show_all_wires", "show_initial_state")

DEFAULT_LINE_WIDTH = 100


def draw_circuit(
    circuit: CircuitPayload,
    circuit_format: CircuitFormat = "ir",
    *,
    decimals: int = 3,
    line_width: int = DEFAULT_LINE_WIDTH,
    show_all_wires: bool = False,
    show_initial_state: bool = False,
) -> dict[str, Any]:
    """Draw a circuit as text.

    Args:
        circuit: Serialized circuit, in either supported format.
        circuit_format: Either ``"ir"`` or ``"qir"``.
        decimals: Digits shown per parameter.
        line_width: Maximum characters per wire line before the drawer wraps.
        show_all_wires: Keep wires that carry no gate.
        show_initial_state: Annotate each wire's initial state.

    Returns:
        A payload carrying the diagram and its line count.

    Raises:
        UnsupportedFormatError: If the circuit cannot be drawn.
        ToolLimitError: If the diagram exceeds the output bound.
    """
    if decimals < 0:
        raise UnsupportedFormatError(f"decimals must not be negative; received {decimals}.")
    if line_width < 1:
        raise UnsupportedFormatError(f"line_width must be positive; received {line_width}.")

    ir = resolve_ir(circuit, circuit_format)
    live = circuit_from_ir(ir)
    draw = load_attribute(DRAWER_MODULE, DRAW_FUNCTION)

    try:
        diagram = str(
            draw(
                live,
                "text",
                decimals=decimals,
                max_length=line_width,
                show_all_wires=show_all_wires,
                show_initial_state=show_initial_state,
            )
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise UnsupportedFormatError(f"Could not draw this circuit: {exc}") from exc

    if len(diagram) > limits.max_qasm_chars():
        raise ToolLimitError(
            f"The diagram is {len(diagram)} characters, above the "
            f"{limits.max_qasm_chars()}-character limit. Draw a smaller circuit."
        )

    return {
        "status": "success",
        "format": "text",
        "diagram": diagram,
        "n_lines": diagram.count("\n") + 1,
        "n_qubits": int(ir.n_wires),
        "content_hash": str(ir.content_hash),
    }
