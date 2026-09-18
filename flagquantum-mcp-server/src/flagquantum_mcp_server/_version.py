"""Single source of truth for this package's version.

Kept in its own module so that :mod:`flagquantum_mcp_server.server` can read the
version without importing the package root, which would be a circular import.
Keep in sync with ``pyproject.toml`` and ``server.json``; a test enforces it.
"""

from __future__ import annotations

__version__ = "0.1.0"
