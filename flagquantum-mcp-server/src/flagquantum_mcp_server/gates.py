"""Gate metadata, read from the SDK's own operator manifest.

An agent writing a gate list has to get three things right per gate: the name,
the number of wires, and the parameter names. Nothing in the circuit format
tells it any of those, so without this module a wrong signature either fails
with a message about wire counts or — worse — is silently accepted when the
parameter names happen to line up.

The metadata comes from ``flagquantum.core``, which is **tier 3**: public, but
not part of the frozen ``stable_exports`` snapshot, exactly like the two
emitters. ``flagquantum/core/__init__.py`` declares all four names in its own
``__all__``, and ``tests/test_api_contract.py`` pins them, so a relocation
fails the build rather than a user's session.
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any, cast

from flagquantum_mcp_server._bridge import load_attribute

CORE_MODULE = "flagquantum.core"
MANIFEST_FUNCTION = "operator_manifest"
GATE_INFO_FUNCTION = "gate_info"
CANONICAL_OPCODE_FUNCTION = "canonical_opcode"

# Fields worth showing an agent, in the order it needs them.
RECORD_FIELDS = (
    "opcode",
    "aliases",
    "arity",
    "parameters",
    "differentiable",
    "semantic_kind",
)


def _manifest() -> tuple[Mapping[str, Any], ...]:
    """Return the SDK's operator manifest with values readable.

    The manifest is typed ``tuple[dict[str, object], ...]`` upstream because it
    is built from heterogeneous literals. Casting once here keeps the rest of
    this module free of per-field ``type: ignore`` comments.
    """
    load = load_attribute(CORE_MODULE, MANIFEST_FUNCTION)
    return cast("tuple[Mapping[str, Any], ...]", load())


@lru_cache(maxsize=1)
def _signatures() -> dict[str, dict[str, Any]]:
    """Map every opcode and alias to its arity and parameter names.

    Cached because signature validation runs on every gate of every qir
    payload, and the manifest is immutable for the life of the process.
    """
    table: dict[str, dict[str, Any]] = {}
    for record in _manifest():
        entry: dict[str, Any] = {
            "opcode": str(record["opcode"]),
            "arity": int(record["arity"]),
            "parameters": tuple(str(name) for name in record["parameters"]),
        }
        table[entry["opcode"]] = entry
        for alias in record["aliases"]:
            table[str(alias)] = entry
    return table


def signature_or_none(name: str) -> dict[str, Any] | None:
    """Return a gate's signature, or ``None`` if the name is not a built-in.

    ``None`` is not an error here: the SDK accepts opcodes outside its manifest
    (``any`` carries a caller-supplied matrix), so signature checks are skipped
    for those rather than rejecting them.
    """
    return _signatures().get(name)


def closest_names(name: str, count: int = 8) -> list[str]:
    """Return the canonical opcodes whose spelling is nearest ``name``."""
    target = name.lower()
    scored = sorted(
        {entry["opcode"] for entry in _signatures().values()},
        key=lambda candidate: _distance(candidate, target),
    )
    return scored[:count]


def _distance(left: str, right: str) -> tuple[int, int]:
    """Rank candidates by shared prefix length, then by edit distance."""
    shared = 0
    for a, b in zip(left, right, strict=False):
        if a != b:
            break
        shared += 1
    return (-shared, abs(len(left) - len(right)))


def gate_records(names: list[str] | None = None) -> dict[str, Any]:
    """Describe gates: arity, parameter names, aliases and kind.

    Args:
        names: Gate names or aliases to describe. ``None`` describes every gate.

    Returns:
        A payload with one record per requested gate, or the whole manifest.

    Raises:
        ToolInputError: If a requested name is not a known gate.
    """
    by_opcode = {str(record["opcode"]): record for record in _manifest()}

    if names is None:
        records = [by_opcode[opcode] for opcode in sorted(by_opcode)]
    else:
        records = [_lookup(name, by_opcode) for name in names]

    return {
        "status": "success",
        # Distinct counts: 35 opcodes, and the aliases that name them.
        "n_gates": len(by_opcode),
        "n_names": len(known_opcodes()),
        "gates": [_shape(record) for record in records],
        "note": (
            "Pass these names in a qir gate list as {'name': ..., 'index': [...]}, "
            "with one index per wire. A gate's arguments go under the key "
            "'parameters', as in "
            '{"name": "rz", "index": [0], "parameters": {"theta": 0.5}}. The '
            "'parameters' field of each record lists the keyword names that gate "
            "accepts, in order. An angle is in radians, and a value may be a "
            "number or a symbol written {'$parameter': '<name>'}."
        ),
    }


def known_opcodes() -> frozenset[str]:
    """Return every gate name and alias the installed SDK accepts."""
    names: set[str] = set()
    for record in _manifest():
        names.add(str(record["opcode"]))
        names.update(str(alias) for alias in record["aliases"])
    return frozenset(names)


def signature(opcode: str) -> dict[str, Any]:
    """Return the arity and parameter names for one opcode.

    Args:
        opcode: A canonical opcode, or any alias of one.

    Returns:
        A mapping with ``opcode``, ``arity`` and ``parameters``.

    Raises:
        ToolInputError: If the name is not a known gate.
    """
    info = load_attribute(CORE_MODULE, GATE_INFO_FUNCTION)(opcode)
    return {
        "opcode": str(info.name),
        "arity": int(info.n_wires),
        "parameters": tuple(str(name) for name in info.parameters),
    }


def canonical(name: str) -> str:
    """Resolve an alias to its canonical opcode."""
    return str(load_attribute(CORE_MODULE, CANONICAL_OPCODE_FUNCTION)(name))


def _lookup(name: str, by_opcode: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    """Resolve one requested name, or explain that it is not a gate."""
    if name in by_opcode:
        return by_opcode[name]
    for record in by_opcode.values():
        if name in record["aliases"]:
            return record
    from flagquantum_mcp_server.errors import ToolInputError

    raise ToolInputError(
        f"{name!r} is not a gate in the installed FlagQuantum. Closest names: "
        f"{closest_names(name)}."
    )


def _shape(record: Mapping[str, Any]) -> dict[str, Any]:
    """Reduce a manifest record to the fields a caller needs."""
    shaped: dict[str, Any] = {field: record[field] for field in RECORD_FIELDS if field in record}
    # Tuples are not JSON; the caller reads these as lists.
    for field in ("aliases", "parameters"):
        if field in shaped:
            shaped[field] = [str(item) for item in shaped[field]]
    return shaped
