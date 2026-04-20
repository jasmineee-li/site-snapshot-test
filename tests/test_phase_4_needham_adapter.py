"""Tests for :mod:`worldsim.phase_4.needham_adapter`."""

from __future__ import annotations

from pathlib import Path

import pytest

from worldsim.phase_4.aer_trajectory_extract import (
    ExtractedStep,
    ExtractedTrajectory,
    extract_trajectory,
)
from worldsim.phase_4.needham_adapter import build_messages
from worldsim.phase_4.needham_xml import format_xml

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "phase_4_aer"


def _extracted(*steps: ExtractedStep, partial: bool = False) -> ExtractedTrajectory:
    return ExtractedTrajectory(steps=list(steps), partial=partial, agent_errors=[])


def test_empty_trajectory_still_emits_user_intent() -> None:
    messages = build_messages(task_instruction="find flight", extracted=_extracted())
    assert [m.role for m in messages] == ["user"]
    assert messages[0].text == "find flight"


def test_system_prompt_omitted_when_blank() -> None:
    messages = build_messages(
        task_instruction="x",
        extracted=_extracted(),
        system_prompt=None,
    )
    assert "system" not in {m.role for m in messages}
    messages = build_messages(task_instruction="x", extracted=_extracted(), system_prompt="")
    assert "system" not in {m.role for m in messages}


def test_system_prompt_emitted_when_present() -> None:
    messages = build_messages(
        task_instruction="x",
        extracted=_extracted(),
        system_prompt="be a browser agent",
    )
    assert messages[0].role == "system"
    assert messages[0].text == "be a browser agent"
    assert messages[1].role == "user"


def test_partial_steps_skipped() -> None:
    step_ok = ExtractedStep(
        step=0,
        thinking="t",
        action=[{"click": {"index": 1}}],
        result=["clicked"],
    )
    step_partial = ExtractedStep(step=1, partial=True, partial_reason="parse_error")
    messages = build_messages(task_instruction="x", extracted=_extracted(step_ok, step_partial))
    # user + assistant + tool = 3, partial step contributes nothing
    assert len(messages) == 3
    assert [m.role for m in messages] == ["user", "assistant", "tool"]


def test_assistant_text_orders_thinking_then_goal_fields() -> None:
    step = ExtractedStep(
        step=0,
        thinking="thinking body",
        evaluation_previous_goal="e",
        memory="m",
        next_goal="g",
        action=[],
    )
    messages = build_messages(task_instruction="x", extracted=_extracted(step))
    assistant_text = messages[1].text
    assert assistant_text.startswith("thinking body")
    assert "Previous-goal evaluation: e" in assistant_text
    assert "Memory: m" in assistant_text
    assert "Next goal: g" in assistant_text
    # Thinking and structured block separated by a blank line
    assert "thinking body\n\nPrevious-goal evaluation: e" in assistant_text


def test_assistant_text_omits_absent_fields_without_placeholder() -> None:
    step = ExtractedStep(step=0, thinking="only thinking", action=[])
    messages = build_messages(task_instruction="x", extracted=_extracted(step))
    text = messages[1].text
    assert text == "only thinking"
    for label in ("Previous-goal evaluation:", "Memory:", "Next goal:"):
        assert label not in text


def test_assistant_tool_calls_from_action_list() -> None:
    step = ExtractedStep(
        step=0,
        thinking="t",
        action=[
            {"click": {"selector": "#ok"}},
            {"fill": {"value": "hello"}},
        ],
    )
    messages = build_messages(task_instruction="x", extracted=_extracted(step))
    assistant = messages[1]
    assert assistant.tool_calls is not None
    assert [tc.id for tc in assistant.tool_calls] == ["0", "1"]
    assert [tc.function for tc in assistant.tool_calls] == ["click", "fill"]
    assert assistant.tool_calls[0].arguments == {"selector": "#ok"}


def test_tool_message_emitted_when_result_non_empty() -> None:
    step = ExtractedStep(
        step=0,
        thinking="t",
        action=[{"click": {"selector": "#ok"}}],
        result=["clicked ok"],
    )
    messages = build_messages(task_instruction="x", extracted=_extracted(step))
    tool_msg = next(m for m in messages if m.role == "tool")
    assert tool_msg.text == "clicked ok"
    assert tool_msg.function == "click"


def test_tool_message_omitted_when_result_empty() -> None:
    step = ExtractedStep(step=0, thinking="t", action=[{"click": {"i": 1}}], result=[])
    messages = build_messages(task_instruction="x", extracted=_extracted(step))
    assert "tool" not in {m.role for m in messages}


def test_tool_message_omitted_when_result_none() -> None:
    step = ExtractedStep(step=0, thinking="t", action=[{"click": {"i": 1}}], result=None)
    messages = build_messages(task_instruction="x", extracted=_extracted(step))
    assert "tool" not in {m.role for m in messages}


def test_tool_result_truncated_at_3000_chars() -> None:
    big = "x" * 10000
    step = ExtractedStep(step=0, action=[{"click": {}}], result=[big])
    messages = build_messages(task_instruction="x", extracted=_extracted(step))
    tool_msg = next(m for m in messages if m.role == "tool")
    assert len(tool_msg.text) == 3000


def test_non_dict_action_entries_are_skipped_in_tool_calls() -> None:
    step = ExtractedStep(
        step=0,
        action=[{"click": {"i": 1}}, "junk", {}, {"fill": {"v": "x"}}],
    )
    messages = build_messages(task_instruction="x", extracted=_extracted(step))
    assistant = messages[1]
    # Only the two well-formed dict entries are kept; ids 0 and 3 because
    # enumerate counts across all entries (not just valid ones).
    assert assistant.tool_calls is not None
    assert [tc.id for tc in assistant.tool_calls] == ["0", "3"]
    assert [tc.function for tc in assistant.tool_calls] == ["click", "fill"]


def test_screenshot_paths_never_reach_messages() -> None:
    step = ExtractedStep(
        step=0,
        thinking="t",
        action=[{"click": {}}],
        screenshot_path="/abs/path/step_0.png",
    )
    messages = build_messages(task_instruction="x", extracted=_extracted(step))
    for m in messages:
        assert "/abs/path/step_0.png" not in m.text


def test_state_message_never_reaches_messages() -> None:
    step = ExtractedStep(
        step=0,
        thinking="t",
        action=[{"click": {}}],
        state_message="<html>giant html dump</html>" * 200,
    )
    messages = build_messages(task_instruction="x", extracted=_extracted(step))
    for m in messages:
        assert "giant html dump" not in m.text


def test_real_fixture_roundtrip_adv_002() -> None:
    traj = extract_trajectory(_FIXTURE_DIR / "ADV-002")
    messages = build_messages(task_instruction="fixture intent", extracted=traj)
    # Every step yields at least one message; some produce 2 (assistant +
    # tool). Adapter never raises on real data.
    assert len(messages) >= len(traj.steps) + 1  # user + per-step assistants
    xml = format_xml(messages)
    assert xml.startswith('<message role="user">')
    assert xml.endswith("</message>\n\n")
    assert "/abs" not in xml  # no screenshot paths


def test_xml_serialization_contains_message_role_tags_not_flat_tags() -> None:
    """Serialized output uses Needham tag shape, not Jasmine's flat tags."""
    step = ExtractedStep(
        step=0, thinking="t", action=[{"click": {"selector": "#ok"}}], result=["ok"]
    )
    messages = build_messages(task_instruction="intent", extracted=_extracted(step))
    xml = format_xml(messages)
    assert '<message role="user">' in xml
    assert '<message role="assistant">' in xml
    assert '<message role="tool", function="click">' in xml
    for forbidden in ("<user>", "<assistant>", "<reasoning>", "<parameter"):
        assert forbidden not in xml


@pytest.mark.parametrize("action_name", ["click", "fill", "go_to", "scroll"])
def test_arbitrary_action_names_preserved(action_name: str) -> None:
    step = ExtractedStep(step=0, action=[{action_name: {"arg": 1}}])
    messages = build_messages(task_instruction="x", extracted=_extracted(step))
    tool_calls = messages[1].tool_calls
    assert tool_calls is not None
    assert tool_calls[0].function == action_name
