"""What this server refuses about a circuit before anything acts on it.

Three tools — planning, simulation and training — hand a circuit to an SDK entry
point that reads some of the envelope and ignores the rest. The ignored parts do
not error: ``observables`` is a real field that serializes, hashes and validates
before the default statevector path drops it; a circuit's own ``measurements``
and a caller's ``outputs`` cannot both be honoured and the SDK says so without
naming either field.

Each of those is a place where accepting the input would produce a
plausible-looking answer, so each is refused here rather than found later. The
refusals live in this module rather than in the tool that first needed one, so
that the second tool to need it shares the message instead of writing a second
opinion about the same fact.

The remedies differ by tool — a simulation points at ``outputs``, a training run
at ``hamiltonian`` — so the caller supplies the sentence that says what to do
instead. The fact is shared; the advice is not.

The module also carries the two things that are shared but are not refusals:
``plain``, the conversion from SDK containers to JSON-native values, and
``SDK_FAILURE_BASES``, the exception bases an execution call can raise. Keeping
them beside the refusals is deliberate — a caller reaching for one should find
the others, and all three exist because more than one tool needs them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from flagquantum_mcp_server.errors import ToolInputError

# The exception bases reachable from a FlagQuantum execution call:
# ValidationError is a ValueError, ExecutionError a RuntimeError, CapabilityError
# a NotImplementedError. This tool's one SDK call is wrapped in these and nothing
# else, so what arrives is the SDK describing the request.
SDK_FAILURE_BASES: tuple[type[BaseException], ...] = (
    ValueError,
    RuntimeError,
    NotImplementedError,
)


def reject_outputs_with_declared_measurements(outputs: Any, ir: Any) -> None:
    """Refuse an output request that collides with the circuit's own measurements.

    A circuit may declare its measurements in the envelope, and the SDK reads
    that field. It also refuses to accept output requests on top of it, in
    vocabulary that names neither ``measurements`` nor ``outputs``
    ("measurements cannot be supplied when the program already contains
    measurement requests"), so a caller is told a conflict exists without being
    told which two things are in conflict.

    Shared by the planning and simulation tools because it is a fact about the
    payload rather than about either one: both pass their outputs to the same
    SDK entry point, and both would otherwise relay the same puzzle.

    Args:
        outputs: The caller's output specifications, if any.
        ir: The validated ``CircuitIR``.

    Raises:
        ToolInputError: If both an output request and circuit-declared
            measurements are present.
    """
    if not outputs:
        return
    declared = tuple(getattr(ir, "measurements", ()) or ())
    if not declared:
        return
    raise ToolInputError(
        f"This circuit already declares {len(declared)} measurement(s), and "
        "outputs were passed as well. The SDK accepts one or the other, not "
        'both ("measurements cannot be supplied when the program already '
        'contains measurement requests"). Drop the "measurements" field from '
        "the circuit envelope, or call without the outputs argument and use "
        "what the circuit declares."
    )


def reject_circuit_observables(ir: Any, *, remedy: str) -> None:
    """Refuse a circuit that carries observables, naming the spelling that works.

    ``CircuitIR.observables`` is a real field: it serializes, it is covered by
    ``content_hash``, it is validated on the way in, and a coefficient may be a
    symbol, so it even shows up in the parameter list. The default statevector
    path does not read it. Only the hybrid operator path does, and that path
    demands a sum of unit single-wire Z terms and says so.

    An agent that followed this server's own ``flagquantum://ir-schema``
    resource would put its Hamiltonian exactly here and get a result with no
    expectation value in it, and nothing anywhere would have said so. Rather
    than reimplement the SDK's private lowering — the observable's name is
    lowercased but never checked against I/X/Y/Z, so it can legally read
    ``"heisenberg"`` — the field is refused and the working spelling is given.

    Args:
        ir: The validated ``CircuitIR``.
        remedy: What this caller should do instead, as one or more sentences.
            Appended after the shared explanation.

    Raises:
        ToolInputError: If the circuit carries any observable.
    """
    observables = tuple(getattr(ir, "observables", ()) or ())
    if not observables:
        return
    raise ToolInputError(
        f"This circuit's 'observables' field carries {len(observables)} "
        "entr(ies), and nothing this server runs evaluates it: the field is "
        "dropped without a word rather than refused, so a result would carry "
        f"neither the value it names nor an error. {remedy}"
    )


def plain(value: Any) -> Any:
    """Convert SDK containers and dataclasses into JSON-native types.

    Shared by the simulation and training tools, which report the same kinds of
    value, so that a float reported by one and the same float reported by the
    other have been through the same conversion. Tensors arrive with a leading
    batch axis because the SDK carries a batch dimension through every result;
    the axis is dropped when it is exactly one, which loses nothing.

    Args:
        value: Any value an SDK result may hold.

    Returns:
        A value ``json`` can carry.
    """
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return plain(
            {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
        )
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [plain(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return plain(_without_batch_axis(value.tolist()))
    return str(value)


def _without_batch_axis(converted: Any) -> Any:
    """Drop a leading axis of length one, which carries no information."""
    if isinstance(converted, list) and len(converted) == 1 and isinstance(converted[0], list):
        return converted[0]
    return converted
