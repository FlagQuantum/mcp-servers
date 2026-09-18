"""Error types and the structured error envelope every tool returns.

Tools never raise out of the MCP boundary. A client sees either a successful
payload or ``{"status": "error", "error": {...}}`` with a stable ``code``, so
an agent can branch on the failure instead of parsing prose. Codes are
deliberately coarse: they describe what the caller did wrong, not where inside
FlagQuantum it surfaced.
"""

from __future__ import annotations

from typing import Any

INVALID_INPUT = "INVALID_INPUT"
LIMIT_EXCEEDED = "LIMIT_EXCEEDED"
UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
SDK_UNAVAILABLE = "SDK_UNAVAILABLE"
INTERNAL_ERROR = "INTERNAL_ERROR"


class ToolError(Exception):
    """Base class for failures that map onto a structured error response."""

    code = INTERNAL_ERROR

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ToolInputError(ToolError):
    """The caller supplied input this tool cannot interpret."""

    code = INVALID_INPUT


class ToolLimitError(ToolInputError):
    """The caller's input is well-formed but exceeds a configured bound."""

    code = LIMIT_EXCEEDED


class UnsupportedFormatError(ToolInputError):
    """The caller asked for a format or option this server does not support."""

    code = UNSUPPORTED_FORMAT


class SdkUnavailableError(ToolError):
    """The FlagQuantum SDK could not be loaded in this environment."""

    code = SDK_UNAVAILABLE


def error_payload(code: str, message: str) -> dict[str, Any]:
    """Build the error envelope returned from a failed tool call.

    Args:
        code: One of the module-level code constants.
        message: Human-readable explanation, safe to show to a model.

    Returns:
        A dictionary with ``status`` and ``error`` keys.
    """
    return {"status": "error", "error": {"code": code, "message": message}}
