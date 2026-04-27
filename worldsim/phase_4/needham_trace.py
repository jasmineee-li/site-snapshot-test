"""Shared Needham-native transcript contract for Phase 4 metrics.

This module owns the Browser-Use -> Needham message projection used by both
Transcript Purpose and Verbalized Eval Awareness. The XML grammar itself stays
in :mod:`worldsim.phase_4.needham_xml`; this file only defines the canonical
structured trace artifact and conversion rules.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from worldsim.atomic_io import write_json_atomic
from worldsim.phase_4.aer_trajectory_extract import ExtractedStep, ExtractedTrajectory
from worldsim.phase_4.needham_chat_types import ChatMessage, ToolCall
from worldsim.phase_4.needham_xml import format_xml

logger = logging.getLogger(__name__)

TRACE_ARTIFACT_NAME = "needham_trace.json"
XML_ARTIFACT_NAME = "needham_trace.xml"
TRACE_FORMAT = "needham-browser-use-v1"
TRANSCRIPT_FORMAT = "needham-xml-v1"

_TOOL_RESULT_CAP = 3000


def _compose_assistant_text(step: ExtractedStep) -> str:
    """Render a step's model output as one assistant message."""
    lines: list[str] = []
    if step.thinking:
        lines.append(step.thinking)
    parts: list[str] = []
    if step.evaluation_previous_goal:
        parts.append(f"Previous-goal evaluation: {step.evaluation_previous_goal}")
    if step.memory:
        parts.append(f"Memory: {step.memory}")
    if step.next_goal:
        parts.append(f"Next goal: {step.next_goal}")
    if parts:
        if lines:
            lines.append("")
        lines.append("\n".join(parts))
    return "\n".join(lines)


def _action_name(action: Any) -> str | None:
    if isinstance(action, dict) and action:
        return str(next(iter(action)))
    return None


def _compose_tool_calls(
    step: ExtractedStep, *, next_tool_id: int
) -> tuple[tuple[ToolCall, ...], int]:
    calls: list[ToolCall] = []
    tool_id = next_tool_id
    for action in step.action:
        name = _action_name(action)
        if name is None:
            continue
        raw_args = action[name]
        if isinstance(raw_args, dict):
            arguments = {k: v for k, v in raw_args.items()}
        else:
            arguments = {"value": raw_args}
        calls.append(ToolCall(id=str(tool_id), function=name, arguments=arguments))
        tool_id += 1
    return tuple(calls), tool_id


def _compose_tool_result(step: ExtractedStep) -> str | None:
    if not step.result:
        return None
    joined = "\n".join(str(item) for item in step.result if item)
    if not joined:
        return None
    if len(joined) > _TOOL_RESULT_CAP:
        joined = joined[:_TOOL_RESULT_CAP]
    return joined


def _tool_function_label(step: ExtractedStep) -> str | None:
    for action in step.action:
        name = _action_name(action)
        if name is not None:
            return name
    return None


def build_messages(
    *,
    task_instruction: str,
    extracted: ExtractedTrajectory,
    system_prompt: str | None = None,
) -> list[ChatMessage]:
    """Build canonical Needham-shaped chat messages from Browser-Use history.

    Partial steps are audit-only and do not enter the judge transcript. Assistant
    tool call ids are globally unique across the whole trace. When Browser-Use
    records tool calls without a result, an empty tool message is emitted to
    preserve the upstream Needham cleanup invariant that tool calls have a
    following tool-role message.
    """
    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage(role="system", text=system_prompt))
    messages.append(ChatMessage(role="user", text=task_instruction))

    next_tool_id = 0
    for step in extracted.steps:
        if step.partial:
            continue
        tool_calls, next_tool_id = _compose_tool_calls(step, next_tool_id=next_tool_id)
        messages.append(
            ChatMessage(
                role="assistant",
                text=_compose_assistant_text(step),
                tool_calls=tool_calls or None,
            )
        )
        tool_text = _compose_tool_result(step)
        if tool_text is not None or tool_calls:
            messages.append(
                ChatMessage(
                    role="tool",
                    text=tool_text or "",
                    function=_tool_function_label(step),
                )
            )
    return messages


def message_to_dict(message: ChatMessage, *, source_step: int | None = None) -> dict[str, Any]:
    """Convert a Needham message to the JSON sidecar representation."""
    payload: dict[str, Any] = {
        "role": message.role,
        "text": message.text,
    }
    if message.function is not None:
        payload["function"] = message.function
    if message.tool_calls is not None:
        payload["tool_calls"] = [
            {
                "id": call.id,
                "function": call.function,
                "arguments": call.arguments,
            }
            for call in message.tool_calls
        ]
    if source_step is not None:
        payload["provenance"] = {
            "source": "browser_use_history",
            "source_step": source_step,
        }
    return payload


def messages_to_dicts(messages: list[ChatMessage]) -> list[dict[str, Any]]:
    """Serialize messages for ``needham_trace.json`` without XML-only leakage."""
    return [message_to_dict(message) for message in messages]


def _messages_to_trace_dicts(
    messages: list[ChatMessage],
    extracted: ExtractedTrajectory,
    *,
    has_system_prompt: bool,
) -> list[dict[str, Any]]:
    """Serialize messages with Browser-Use step provenance for audit."""
    out: list[dict[str, Any]] = []
    index = 0
    if has_system_prompt and index < len(messages):
        out.append(message_to_dict(messages[index]))
        index += 1
    if index < len(messages):
        out.append(message_to_dict(messages[index]))
        index += 1
    for step in extracted.steps:
        if step.partial or index >= len(messages):
            continue
        out.append(message_to_dict(messages[index], source_step=step.step))
        index += 1
        if index < len(messages) and messages[index].role == "tool":
            out.append(message_to_dict(messages[index], source_step=step.step))
            index += 1
    while index < len(messages):
        out.append(message_to_dict(messages[index]))
        index += 1
    return out


def dicts_to_messages(items: list[dict[str, Any]]) -> list[ChatMessage]:
    """Load JSON sidecar messages back into ``ChatMessage`` objects."""
    messages: list[ChatMessage] = []
    for item in items:
        role = item.get("role")
        text = item.get("text")
        if role not in {"system", "user", "assistant", "tool"} or not isinstance(text, str):
            raise ValueError("invalid Needham trace message")
        raw_calls = item.get("tool_calls")
        tool_calls: tuple[ToolCall, ...] | None = None
        if isinstance(raw_calls, list):
            calls: list[ToolCall] = []
            for raw_call in raw_calls:
                if not isinstance(raw_call, dict):
                    raise ValueError("invalid Needham trace tool call")
                call_id = raw_call.get("id")
                function = raw_call.get("function")
                arguments = raw_call.get("arguments")
                if not isinstance(call_id, str) or not isinstance(function, str):
                    raise ValueError("invalid Needham trace tool call")
                if not isinstance(arguments, dict):
                    arguments = {}
                calls.append(ToolCall(id=call_id, function=function, arguments=arguments))
            tool_calls = tuple(calls) if calls else None
        function = item.get("function")
        messages.append(
            ChatMessage(
                role=role,  # type: ignore[arg-type]
                text=text,
                function=function if isinstance(function, str) else None,
                tool_calls=tool_calls,
            )
        )
    return messages


def build_trace_payload(
    *,
    task_instruction: str,
    extracted: ExtractedTrajectory,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Build the canonical JSON sidecar payload."""
    messages = build_messages(
        task_instruction=task_instruction,
        extracted=extracted,
        system_prompt=system_prompt,
    )
    return {
        "format": TRACE_FORMAT,
        "transcript_format": TRANSCRIPT_FORMAT,
        "source": "browser_use_history",
        "messages": _messages_to_trace_dicts(
            messages,
            extracted,
            has_system_prompt=bool(system_prompt),
        ),
        "xml": format_xml(messages),
    }


def format_trace_xml(
    *,
    task_instruction: str,
    extracted: ExtractedTrajectory,
    system_prompt: str | None = None,
) -> str:
    """Format the canonical Needham XML for a task trajectory."""
    return format_xml(
        build_messages(
            task_instruction=task_instruction,
            extracted=extracted,
            system_prompt=system_prompt,
        )
    )


def write_trace_artifacts(
    trajectory_dir: Path | str,
    *,
    task_instruction: str,
    extracted: ExtractedTrajectory,
    system_prompt: str | None = None,
) -> tuple[Path, Path]:
    """Persist ``needham_trace.json`` and ``needham_trace.xml`` for audit."""
    trajectory_dir = Path(trajectory_dir)
    payload = build_trace_payload(
        task_instruction=task_instruction,
        extracted=extracted,
        system_prompt=system_prompt,
    )
    json_path = trajectory_dir / TRACE_ARTIFACT_NAME
    xml_path = trajectory_dir / XML_ARTIFACT_NAME
    write_json_atomic(json_path, payload)
    try:
        xml_path.write_text(payload["xml"], encoding="utf-8")
    except OSError as exc:  # pragma: no cover - best-effort audit artifact
        logger.warning("could not persist Needham XML sidecar at %s: %s", xml_path, exc)
    return json_path, xml_path


def load_trace_xml(trajectory_dir: Path | str) -> str | None:
    """Return the persisted Needham XML sidecar if present and valid."""
    trace_path = Path(trajectory_dir) / TRACE_ARTIFACT_NAME
    if not trace_path.exists():
        return None
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("format") != TRACE_FORMAT:
        return None
    xml = payload.get("xml")
    if isinstance(xml, str):
        return xml
    raw_messages = payload.get("messages")
    if isinstance(raw_messages, list):
        return format_xml(dicts_to_messages(raw_messages))
    return None


__all__ = [
    "TRACE_ARTIFACT_NAME",
    "TRACE_FORMAT",
    "TRANSCRIPT_FORMAT",
    "XML_ARTIFACT_NAME",
    "build_messages",
    "build_trace_payload",
    "dicts_to_messages",
    "format_trace_xml",
    "load_trace_xml",
    "message_to_dict",
    "messages_to_dicts",
    "write_trace_artifacts",
]
