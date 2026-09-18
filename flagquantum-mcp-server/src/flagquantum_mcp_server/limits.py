"""Bound every input this server accepts.

Defaults are sized so that a single tool call cannot make the server allocate
unbounded memory or keep an agent waiting. Each limit is overridable through
an environment variable so a deployment can tighten or relax it without a code
change; malformed or non-positive values fall back to the default rather than
disabling the bound.
"""

from __future__ import annotations

import os

DEFAULT_MAX_QUBITS = 24
DEFAULT_MAX_GATES = 10_000
DEFAULT_MAX_IR_BYTES = 262_144
DEFAULT_MAX_QASM_CHARS = 1_000_000
DEFAULT_MAX_COMPARE_TOPOLOGIES = 4
DEFAULT_MAX_RESPONSE_VALUES = 65_536
DEFAULT_MAX_HAMILTONIAN_TERMS = 1024
DEFAULT_MAX_TRAIN_SECONDS = 60


def _positive_int(name: str, default: int) -> int:
    """Read a positive integer from the environment, falling back on default.

    Args:
        name: Environment variable name.
        default: Value to use when the variable is unset or unusable.

    Returns:
        The resolved positive integer.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


def max_qubits() -> int:
    """Return the largest circuit width this server will process."""
    return _positive_int("FLAGQUANTUM_MCP_MAX_QUBITS", DEFAULT_MAX_QUBITS)


def max_gates() -> int:
    """Return the largest instruction count this server will process."""
    return _positive_int("FLAGQUANTUM_MCP_MAX_GATES", DEFAULT_MAX_GATES)


def max_ir_bytes() -> int:
    """Return the largest serialized circuit payload this server will accept."""
    return _positive_int("FLAGQUANTUM_MCP_MAX_IR_BYTES", DEFAULT_MAX_IR_BYTES)


def max_qasm_chars() -> int:
    """Return the largest emitted program this server will return."""
    return _positive_int("FLAGQUANTUM_MCP_MAX_QASM_CHARS", DEFAULT_MAX_QASM_CHARS)


def max_compare_topologies() -> int:
    """Return how many topologies one comparison call may evaluate."""
    return _positive_int("FLAGQUANTUM_MCP_MAX_COMPARE_TOPOLOGIES", DEFAULT_MAX_COMPARE_TOPOLOGIES)


def max_response_values() -> int:
    """Return how many numbers one tool result may carry back to a client.

    The bounds above limit what this server *accepts*. This one limits what it
    *returns*, because a result travels to a model's context rather than into
    memory, and a full probability distribution grows as 2 ** n_wires — 4.2
    million floats at 22 wires, which is a legal circuit by every other bound
    here and an unusable answer. The count is of scalar values, so a sparse
    ``counts`` result over the same circuit is unaffected.
    """
    return _positive_int("FLAGQUANTUM_MCP_MAX_RESPONSE_VALUES", DEFAULT_MAX_RESPONSE_VALUES)


def max_hamiltonian_terms() -> int:
    """Return how many Pauli terms one expectation request may carry.

    An expectation over a weighted sum reports one row per term, so the terms a
    caller writes are the rows it gets back. Evaluating a few thousand terms is
    cheap — measured at under a third of a second for 2000 on a four-qubit
    circuit — so this bound is about the answer rather than the work: past it
    the response stops being something a model can read.
    """
    return _positive_int("FLAGQUANTUM_MCP_MAX_HAMILTONIAN_TERMS", DEFAULT_MAX_HAMILTONIAN_TERMS)


def max_train_seconds() -> int:
    """Return how long one training call may be predicted to take.

    Every other bound here limits what a single call may *consume* — memory,
    width, response size. This one limits time, because training is the only
    operation in this server whose cost is unbounded by the size of its input:
    the same four-qubit circuit costs 0.2 s for ten steps and three minutes for
    ten thousand, and ``steps`` is the caller's number.

    The prediction that reads this bound is an estimate rather than a
    measurement of the caller's machine, so the bound is a policy about how long
    an agent should be made to wait, not a guarantee about the clock.
    """
    return _positive_int("FLAGQUANTUM_MCP_MAX_TRAIN_SECONDS", DEFAULT_MAX_TRAIN_SECONDS)
