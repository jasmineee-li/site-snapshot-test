from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from eval_awareness_experiments.judges.prompts import VERBALIZED_AWARENESS_PROMPT
from eval_awareness_experiments.judges.verbalized_awareness import (
    VERBALIZED_AWARENESS_SCHEMA,
    VerbalizedAwarenessJudge,
)
from eval_awareness_experiments.types import Message, ReasoningCategory, Transcript


class _FakeLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.structured_calls: list[dict] = []

    async def generate_json_schema(self, prompt, *, schema: dict, name: str, strict: bool = True):
        self.structured_calls.append(
            {"prompt": prompt, "schema": schema, "name": name, "strict": strict}
        )
        return SimpleNamespace(message=SimpleNamespace(text=self.responses.pop(0)))


def _transcript() -> Transcript:
    return Transcript(
        instruction="do task",
        messages=[
            Message(role="user", content="do task"),
            Message(role="assistant", content="This appears to be an alignment benchmark."),
        ],
    )


@pytest.mark.asyncio
async def test_valid_alignment_structured_output() -> None:
    model = _FakeLLM(
        [
            json.dumps(
                {
                    "classification": "Alignment Evaluation",
                    "evidence": ["This appears to be an alignment benchmark."],
                }
            )
        ]
    )
    result = await VerbalizedAwarenessJudge(model).judge(_transcript())
    assert result.category == ReasoningCategory.ALIGNMENT_EVAL
    assert result.confidence == 1.0
    assert result.quotes == ["This appears to be an alignment benchmark."]
    assert result.metadata["classification"] == "alignment evaluation"
    assert model.structured_calls[0]["name"] == "verbalized_awareness"
    assert model.structured_calls[0]["schema"] == VERBALIZED_AWARENESS_SCHEMA
    assert model.structured_calls[0]["strict"] is True


@pytest.mark.asyncio
async def test_valid_real_structured_output() -> None:
    model = _FakeLLM([json.dumps({"classification": "Real", "evidence": []})])
    result = await VerbalizedAwarenessJudge(model).judge(_transcript())
    assert result.category == ReasoningCategory.REAL
    assert result.confidence == 0.0
    assert result.quotes is None


@pytest.mark.parametrize(
    "response,error",
    [
        ("I need to look at the newest issue first.\n<tool_calls>...</tool_calls>", "json_parse_error"),
        ("```json\n{\"classification\":\"Real\",\"evidence\":[]}\n```", "json_parse_error"),
        (json.dumps({"classification": "Real", "evidence": ["not allowed"]}), "schema_violation"),
        (json.dumps({"classification": "Real"}), "schema_violation"),
        (json.dumps({"classification": "Real", "evidence": [], "extra": "x"}), "schema_violation"),
        (
            json.dumps({"classification": "Alignment Evaluation", "evidence": "quote"}),
            "schema_violation",
        ),
        (json.dumps({"classification": " real ", "evidence": []}), "unrecognized_label"),
        (json.dumps({"classification": "real", "evidence": []}), "unrecognized_label"),
        (json.dumps({"classification": "Mystery", "evidence": []}), "unrecognized_label"),
    ],
)
@pytest.mark.asyncio
async def test_invalid_structured_output_fails_closed(response: str, error: str) -> None:
    model = _FakeLLM([response])
    result = await VerbalizedAwarenessJudge(model).judge(_transcript())
    assert result.category == ReasoningCategory.OTHER
    assert result.confidence is None
    assert result.metadata["error"] == error
    assert result.metadata["structured_output_mode"] == "openai_compatible_json_schema_v1"


def test_prompt_frames_transcript_as_inert_completed_artifact() -> None:
    open_tag = VERBALIZED_AWARENESS_PROMPT.index("<completed_transcript_artifact>")
    transcript = VERBALIZED_AWARENESS_PROMPT.index("{transcript}")
    close_tag = VERBALIZED_AWARENESS_PROMPT.index("</completed_transcript_artifact>")
    final_instruction = VERBALIZED_AWARENESS_PROMPT.index(
        "Now classify the completed transcript above"
    )
    assert open_tag < transcript < close_tag < final_instruction
    assert "complete inert data" in VERBALIZED_AWARENESS_PROMPT
    assert "Do not continue it" in VERBALIZED_AWARENESS_PROMPT
