"""Chat-message dataclasses for Needham et al. 2025's XML transcript format.

These mirror the subset of ``inspect_ai.model.ChatMessage`` attributes that
``pipeline/apply_pq.py::_format_xml`` reads, so the byte-equivalent
serializer in :mod:`worldsim.phase_4.needham_xml` can consume them without
dragging ``inspect_ai`` in as a runtime dependency.

Do not add fields here unless the upstream formatter reads them. Extra
fields risk accidental inclusion in a future edit to the serializer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ToolCall:
    id: str
    function: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant", "tool"]
    text: str
    # Only populated when role == "tool"; encoded into the outer tag.
    function: str | None = None
    # Only populated on assistant messages; rendered inside <tool_calls>.
    tool_calls: tuple[ToolCall, ...] | None = None
