"""Read-surface and trace URL matching helpers."""

from __future__ import annotations

from worldsim.outcome_taxonomy._impl import (
    _build_read_surface_patterns,
    _compile,
    _derive_read_surface_from_editor,
    _is_document_load,
    _match_task_url_to_trace_url,
    _normalize_url,
    _path_template_regex,
    scheme_host_fallback,
)

__all__ = [
    "_build_read_surface_patterns",
    "_compile",
    "_derive_read_surface_from_editor",
    "_is_document_load",
    "_match_task_url_to_trace_url",
    "_normalize_url",
    "_path_template_regex",
    "scheme_host_fallback",
]
