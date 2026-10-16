"""Thin MCP adaptation of FlagQuantum's approved research capability service."""

from __future__ import annotations

from typing import Any

from flagquantum_mcp_server._bridge import load_module
from flagquantum_mcp_server.errors import INTERNAL_ERROR, INVALID_INPUT, error_payload
from flagquantum_mcp_server.preconditions import plain


def call_research_operation(operation: str, /, *args: Any, **kwargs: Any) -> dict[str, Any]:
    """Call one public service operation and preserve its result losslessly."""

    services = load_module("flagquantum.services")
    function = getattr(services, operation)
    value = function(*args, **kwargs)
    if hasattr(value, "to_dict"):
        value = value.to_dict()
    return {"status": "success", "result": plain(value)}


def research_invalid_input(exc: KeyError | TypeError | ValueError) -> dict[str, Any]:
    """Map a public pre-result input exception to the protocol envelope."""

    return error_payload(INVALID_INPUT, str(exc))


def research_internal_error() -> dict[str, Any]:
    """Return the deliberately sanitized unexpected-failure envelope."""

    return error_payload(INTERNAL_ERROR, "FlagQuantum research capability call failed")
