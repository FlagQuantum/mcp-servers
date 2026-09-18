"""Lazy access to the FlagQuantum SDK.

This repository is an out-of-tree edge adapter: it must not become a reason
for FlagQuantum to grow an MCP dependency, and it must not make importing the
MCP server expensive. ``flagquantum`` pulls ``torch`` at import time, so the
SDK is bound lazily and cached on first use. A failed import surfaces as a
structured tool error rather than a traceback at server start-up.
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


@lru_cache(maxsize=1)
def _load() -> ModuleType:
    try:
        return importlib.import_module("flagquantum")
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise FlagQuantumUnavailableError(SDK_NOT_INSTALLED) from exc


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
