from __future__ import annotations

from typing import Any


class TextFillAPIError(RuntimeError):
    """Raised when a structured Phase 2b text-fill call fails."""

    def __init__(self, message: str, *, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.diagnostics = diagnostics or {}
