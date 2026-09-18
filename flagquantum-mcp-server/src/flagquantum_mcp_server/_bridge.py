"""Lazy access to the FlagQuantum SDK.

This repository is an out-of-tree edge adapter: it must not become a reason
for FlagQuantum to grow an MCP dependency, and it must not make importing the
MCP server expensive. So the SDK and torch are both bound on first use and
cached rather than imported at module load: a failed import surfaces as a
structured tool error rather than a traceback at start-up, and the one call
that needs the SDK pays for it instead of every client that only wants to list
a tool.

The weight being avoided is ``torch``'s, not the SDK's. Measured on the pair
installed here: ``import flagquantum`` costs 5 to 16 ms and leaves ``torch``
absent from ``sys.modules``, while ``import torch`` costs 0.53 s. An earlier
version of this paragraph gave "``flagquantum`` pulls ``torch`` at import time"
as the reason. That is false, and the true reason runs the other way: the SDK's
training API needs torch, and this package deliberately does not declare it.
"""

from __future__ import annotations

import importlib
from functools import lru_cache
from types import ModuleType
from typing import Any

SDK_NOT_INSTALLED = (
    "The flagquantum package is not importable in this environment. "
    "Install the server with its own dependency: "
    "pip install flagquantum-mcp-server"
)


class FlagQuantumUnavailableError(RuntimeError):
    """Raised when the FlagQuantum SDK cannot be imported."""


def load_sdk() -> ModuleType:
    """Import and return the top-level ``flagquantum`` module.

    Returns:
        The imported ``flagquantum`` module.

    Raises:
        FlagQuantumUnavailableError: If the SDK is not installed or fails to
            import. The original exception is chained for diagnosis.
    """
    return _load()


TORCH_NOT_INSTALLED = (
    "torch is not importable in this environment. It is not a declared "
    "dependency of this server, but the FlagQuantum SDK requires it, so this "
    "means the SDK is installed without its own dependency."
)


def load_torch() -> ModuleType:
    """Import and return ``torch``, which the SDK's training API takes.

    Not a declared dependency of this package: the server declares FlagQuantum
    and FastMCP, and FlagQuantum declares torch. Reaching it through the same
    lazy path as the SDK keeps that true — a direct import here would make torch
    a third dependency the packaging does not state, and would make importing
    this server expensive.

    Returns:
        The imported ``torch`` module.

    Raises:
        FlagQuantumUnavailableError: If torch is not importable.
    """
    return _load_torch()


@lru_cache(maxsize=1)
def _load_torch() -> ModuleType:
    try:
        return importlib.import_module("torch")
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise FlagQuantumUnavailableError(TORCH_NOT_INSTALLED) from exc


@lru_cache(maxsize=1)
def _load() -> ModuleType:
    try:
        return importlib.import_module("flagquantum")
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise FlagQuantumUnavailableError(SDK_NOT_INSTALLED) from exc


def load_module(module_path: str) -> ModuleType:
    """Import and return a submodule of the SDK by path.

    A companion to :func:`load_attribute` for the case where several names come
    from one module: importing it once and reading the attributes is clearer than
    a call per name that each re-imports it.

    Args:
        module_path: Dotted path of the module to import.

    Returns:
        The imported module.

    Raises:
        FlagQuantumUnavailableError: If the module is not importable.
    """
    try:
        return importlib.import_module(module_path)
    except ImportError as exc:
        raise FlagQuantumUnavailableError(
            f"{module_path} is not importable in the installed flagquantum. "
            "This server requires the module that provides the training "
            "objective."
        ) from exc


def load_attribute(module_path: str, attribute: str) -> Any:
    """Resolve one attribute, reporting a clear error when a contract moved.

    The emitters live in public-but-unsnapshotted submodules
    (``flagquantum.compiler.openqasm``). If FlagQuantum relocates one, this
    raises a message naming the exact path instead of an AttributeError deep
    inside a tool call.

    Args:
        module_path: Dotted path of the module to import.
        attribute: Name of the attribute to read from that module.

    Returns:
        The resolved attribute.

    Raises:
        FlagQuantumUnavailableError: If the module or attribute is missing.
    """
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise FlagQuantumUnavailableError(
            f"{module_path} is not importable in the installed flagquantum. "
            f"This server requires the module that provides {attribute!r}."
        ) from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        raise FlagQuantumUnavailableError(
            f"{module_path}.{attribute} no longer exists in the installed "
            f"flagquantum. The public contract this tool depends on has moved."
        ) from exc
