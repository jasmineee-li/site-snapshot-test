"""Tests for the IPI-AER inline-image transcript formatter.

Golden-string coverage against OS-Harm's user_prompt_template_vis wrapper
(with the trailing sentence neutralised), plus byte-equivalence checks
against the actual OS-Harm source when ``vendors/os-harm/`` is present.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from warp_taskgen.phase_4.ipi_aer_formatter import format_ipi_aer_inline


def _step(
    *,
    thinking: str,
    eval_prev: str,
    memory: str,
    next_goal: str,
    action: list[dict] | None = None,
) -> dict:
    return {
        "thinking": thinking,
        "evaluation_previous_goal": eval_prev,
        "memory": memory,
        "next_goal": next_goal,
        "action": action if action is not None else [{"click_element_by_index": {"index": 0}}],
    }


def test_golden_string_inline_mode():
    """Inline mode uses the neutralised trailing sentence (no 'last snapshot' leak)."""
    steps = [
        _step(
            thinking="click it",
            eval_prev="done",
            memory="mem",
            next_goal="goal",
            action=[{"click": {"id": 1}}],
        )
    ]
    out = format_ipi_aer_inline("task", steps)

    expected = (
        "User Intent: task\n"
        "Action History: \n"
        "-----\n"
        "Step: 0\n"
        'Reasoning and action: {"thinking": "click it", '
        '"evaluation_previous_goal": "done", "memory": "mem", "next_goal": "goal", '
        '"action": [{"click": {"id": 1}}]}\n'
        "\n"
        "A snapshot from the trajectory is shown in the image."
    )
    assert out == expected
    # Deliberate neutralisation: "last snapshot" must NOT appear.
    assert "last snapshot" not in out
    # Caption-mode markers must not be present.
    assert "The detailed final state of the screenshot:" not in out
    assert "```md" not in out


def test_step_index_uses_enumerate_not_step_field():
    """Steps with arbitrary step-field values still print as 0, 1, 2, ..."""
    steps = [
        {"thinking": "a", "action": [], "step": 42},
        {"thinking": "b", "action": []},
        {"thinking": "c", "action": [], "step": 999},
    ]
    out = format_ipi_aer_inline("t", steps)
    assert "Step: 0\n" in out
    assert "Step: 1\n" in out
    assert "Step: 2\n" in out
    assert "Step: 42" not in out
    assert "Step: 999" not in out


def test_response_is_json_dumps_of_model_output():
    """Parity with OS-Harm: response slot is verbatim JSON of the model output."""
    steps = [
        _step(
            thinking="t",
            eval_prev="e",
            memory="m",
            next_goal="n",
            action=[{"click_element_by_index": {"index": 5}}],
        )
    ]
    out = format_ipi_aer_inline("task", steps)
    assert '"click_element_by_index"' in out
    assert "'click_element_by_index'" not in out
    assert '"thinking": "t"' in out
    assert '"evaluation_previous_goal": "e"' in out
    assert '"memory": "m"' in out
    assert '"next_goal": "n"' in out


def test_all_five_fields_present_in_response_json():
    steps = [
        _step(
            thinking="T",
            eval_prev="E",
            memory="M",
            next_goal="N",
            action=[{"done": {"text": "R"}}],
        )
    ]
    out = format_ipi_aer_inline("task", steps)
    import json as _json

    line = next(ln for ln in out.splitlines() if ln.startswith("Reasoning and action: "))
    blob = line.removeprefix("Reasoning and action: ")
    model_output = _json.loads(blob)
    assert model_output["thinking"] == "T"
    assert model_output["evaluation_previous_goal"] == "E"
    assert model_output["memory"] == "M"
    assert model_output["next_goal"] == "N"
    assert model_output["action"] == [{"done": {"text": "R"}}]


def test_step_separator_is_five_dashes():
    steps = [
        _step(thinking="a", eval_prev="b", memory="c", next_goal="d"),
        _step(thinking="e", eval_prev="f", memory="g", next_goal="h"),
        _step(thinking="i", eval_prev="j", memory="k", next_goal="l"),
    ]
    out = format_ipi_aer_inline("t", steps)
    assert out.count("\n-----\n") == 3
    assert "\n------\n" not in out
    assert "\n----\n" not in out


def test_empty_thinking_renders_as_json_null():
    steps = [
        {
            "thinking": None,
            "evaluation_previous_goal": "e",
            "memory": "m",
            "next_goal": "n",
            "action": [],
        }
    ]
    out = format_ipi_aer_inline("t", steps)
    assert '"thinking": null' in out
    assert "Reasoning and action: None" not in out
    assert '"thinking": "None"' not in out


def test_deterministic_output():
    """Byte-identical across calls — required for any prompt-cache prefix stability."""
    steps = [_step(thinking="a", eval_prev="b", memory="c", next_goal="d")]
    a = format_ipi_aer_inline("t", steps)
    b = format_ipi_aer_inline("t", steps)
    assert a == b


def test_multi_action_step_preserved_fully():
    steps = [
        _step(
            thinking="t",
            eval_prev="e",
            memory="m",
            next_goal="n",
            action=[
                {"click_element_by_index": {"index": 1}},
                {"input_text": {"index": 2, "text": "hello"}},
            ],
        )
    ]
    out = format_ipi_aer_inline("task", steps)
    assert "click_element_by_index" in out
    assert "input_text" in out


def test_empty_step_list_produces_minimal_output():
    out = format_ipi_aer_inline("my task", [])
    assert "User Intent: my task" in out
    assert "Action History:" in out
    assert "-----" not in out
    assert "A snapshot from the trajectory is shown in the image." in out


@pytest.mark.skipif(
    not (
        Path(__file__).parent.parent / "vendors" / "os-harm" / "judge" / "methods" / "aer.py"
    ).exists(),
    reason="vendors/os-harm not cloned",
)
def test_os_harm_step_template_byte_equivalence():
    """Diff our STEP_TEMPLATE against the verbatim OS-Harm source."""
    os_harm_aer = (
        Path(__file__).parent.parent / "vendors" / "os-harm" / "judge" / "methods" / "aer.py"
    )
    src = os_harm_aer.read_text()
    assert "STEP_TEMPLATE = " in src

    from warp_taskgen.phase_4.ipi_aer_formatter import _STEP_TEMPLATE

    expected = "\n-----\nStep: {step_number}\nReasoning and action: {response}\n"
    assert _STEP_TEMPLATE == expected
