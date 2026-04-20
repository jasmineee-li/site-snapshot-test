"""Project :class:`ExtractedTrajectory` into a Needham-shaped message list.

Needham et al. 2025 judges a ``list[ChatMessage]`` transcript. Browser-Use
emits structured per-step records (thinking, evaluation_previous_goal,
memory, next_goal, action, result) — this adapter folds them into the
shape ``needham_xml.format_xml`` expects without leaking any field the
transcript shouldn't contain (screenshot paths, state_message HTML).

Projection rules:

- The task intent becomes a single ``<message role="user">``.
- A system prompt becomes ``<message role="system">`` *only* when it's
  non-empty; we don't emit empty system blocks (they'd dilute the
  transcript and drift a judge's prior about what a normal trajectory
  looks like).
- Partial steps (``ExtractedStep.partial=True``) are skipped. They carry
  no agent-attributable content; including them would weaken the
  verbalized-awareness judge's ability to quote the agent's own words.
- Each non-partial step emits one assistant message. Its text folds
  thinking + evaluation-previous-goal + memory + next-goal into prose.
  Each ``action`` dict in the step becomes a ``<tool_call>`` with
  sequential ids starting at 0 within the step.
- A step's ``result`` (if any, truncated to 3000 chars) emits one
  ``<message role="tool">``. Multi-item results are joined on ``\\n``.
  Empty/None results produce no tool message.

The 3000-char truncation on tool results matches Jasmine's inspect_ai-
native serializer behavior (``eval_awareness_experiments/judges/
transcript_utils.py``) and stops one runaway page dump from dominating
the judge's token budget. Thinking/goal fields are never truncated —
they carry the signal we're judging.
"""

from __future__ import annotations

from worldsim.phase_4.aer_trajectory_extract import ExtractedStep, ExtractedTrajectory
from worldsim.phase_4.needham_chat_types import ChatMessage, ToolCall

_TOOL_RESULT_CAP = 3000


def _compose_assistant_text(step: ExtractedStep) -> str:
    """Render a step's CoT + goal fields as one prose assistant block.

    Absent fields are omitted from the rendering, not printed as empty
    lines. The ordering mirrors the AER formatter's per-step labels so a
    reviewer reading both transcripts for the same run sees the same
    shape.
    """
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


def _compose_tool_calls(step: ExtractedStep) -> tuple[ToolCall, ...]:
    calls: list[ToolCall] = []
    for index, action in enumerate(step.action):
        if not isinstance(action, dict) or not action:
            continue
        # Browser-Use actions are ``{name: args_dict}``. Pick the first
        # key; the format guarantees a single entry per action dict.
        name = next(iter(action))
        raw_args = action[name]
        if isinstance(raw_args, dict):
            arguments = {k: v for k, v in raw_args.items()}
        else:
            arguments = {"value": raw_args}
        calls.append(ToolCall(id=str(index), function=str(name), arguments=arguments))
    return tuple(calls)


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
    """Attribute on ``<message role="tool">`` — first action's name for this step.

    Returns ``None`` when the step produced no action, so the tool-role
    opening tag falls back to the role-only form.
    """
    for action in step.action:
        if isinstance(action, dict) and action:
            return str(next(iter(action)))
    return None


def build_messages(
    *,
    task_instruction: str,
    extracted: ExtractedTrajectory,
    system_prompt: str | None = None,
) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    if system_prompt:
        messages.append(ChatMessage(role="system", text=system_prompt))
    messages.append(ChatMessage(role="user", text=task_instruction))

    for step in extracted.steps:
        if step.partial:
            continue
        assistant_text = _compose_assistant_text(step)
        tool_calls = _compose_tool_calls(step) or None
        messages.append(
            ChatMessage(
                role="assistant",
                text=assistant_text,
                tool_calls=tool_calls,
            )
        )
        tool_text = _compose_tool_result(step)
        if tool_text is not None:
            messages.append(
                ChatMessage(
                    role="tool",
                    text=tool_text,
                    function=_tool_function_label(step),
                )
            )
    return messages
