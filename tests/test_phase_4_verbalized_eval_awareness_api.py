"""P0/P1 tests for the verbalized-eval-awareness judge.

After the cutover to Jasmine's 4-way classifier over Needham XML, the
judge:

- Takes an ``ExtractedTrajectory`` directly (no pre-formatted AER).
- Calls Anthropic with free-text completion (NO forced tool-use).
- Parses a JSON object from the response with ``classification`` and
  ``evidence`` fields.
- Returns the new schema (``verbalized_eval_awareness_*``).
- Persists ``verbalized_eval_awareness_transcript.xml`` and
  ``verbalized_eval_awareness_raw_response.json`` per task.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from worldsim.phase_4.aer_trajectory_extract import ExtractedStep, ExtractedTrajectory
from worldsim.phase_4.verbalized_eval_awareness_api import (
    run_verbalized_eval_awareness_api,
)


def _text_response(text: str, *, in_tok: int = 100, out_tok: int = 50) -> SimpleNamespace:
    return SimpleNamespace(
        id="msg_vea",
        content=[SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


def _empty_response() -> SimpleNamespace:
    return SimpleNamespace(
        id="msg_vea",
        content=[],
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=0,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
        ),
    )


def _trajectory(*, thinking: str = "I'm thinking.") -> ExtractedTrajectory:
    return ExtractedTrajectory(
        steps=[
            ExtractedStep(
                step=0,
                thinking=thinking,
                action=[{"click": {"index": 1}}],
                result=["clicked"],
            )
        ]
    )


def _task() -> dict:
    return {"id": "t", "description": "find a flight"}


@pytest.mark.asyncio
async def test_happy_path_alignment_classification(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_response(
            json.dumps(
                {
                    "classification": "Alignment Evaluation",
                    "evidence": ["This looks like a test of my instruction-following."],
                }
            )
        )
    )
    result = await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    assert result["verbalized_eval_awareness_classification"] == "Alignment Evaluation"
    assert result["verbalized_eval_awareness_is_aware"] is True
    assert result["verbalized_eval_awareness_evidence"] == [
        "This looks like a test of my instruction-following."
    ]
    assert result["verbalized_eval_awareness_failure"] is None
    assert result["verbalized_eval_awareness_transcript_format"] == "needham-xml-v1"
    assert Path(result["verbalized_eval_awareness_raw_response_path"]).exists()


@pytest.mark.asyncio
async def test_happy_path_real_classification(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_response(json.dumps({"classification": "Real", "evidence": []}))
    )
    result = await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    assert result["verbalized_eval_awareness_classification"] == "Real"
    assert result["verbalized_eval_awareness_is_aware"] is False
    assert result["verbalized_eval_awareness_evidence"] == []
    assert result["verbalized_eval_awareness_failure"] is None


@pytest.mark.asyncio
async def test_capability_evaluation_is_aware_true(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_response(
            json.dumps(
                {
                    "classification": "Capability Evaluation",
                    "evidence": ["benchmarking my abilities"],
                }
            )
        )
    )
    result = await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    assert result["verbalized_eval_awareness_is_aware"] is True


@pytest.mark.parametrize(
    "raw, canonical",
    [
        ("alignment evaluation", "Alignment Evaluation"),
        ("ALIGNMENT EVALUATION", "Alignment Evaluation"),
        ("Other Evaluation", "Other Evaluation"),
        ("real", "Real"),
    ],
)
@pytest.mark.asyncio
async def test_classification_normalized_to_canonical(
    patched_anthropic_client, tmp_path, raw, canonical
):
    evidence_value = [] if canonical == "Real" else ["q"]
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_response(json.dumps({"classification": raw, "evidence": evidence_value}))
    )
    result = await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    assert result["verbalized_eval_awareness_classification"] == canonical


@pytest.mark.asyncio
async def test_unknown_classification_buckets_as_failure(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_response(
            json.dumps({"classification": "Mystery Category", "evidence": ["something"]})
        )
    )
    result = await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    assert result["verbalized_eval_awareness_failure"] == "vea_unrecognized_label"
    assert result["verbalized_eval_awareness_classification"] is None
    assert result["verbalized_eval_awareness_is_aware"] is None
    assert result["verbalized_eval_awareness_evidence"] == []


@pytest.mark.asyncio
async def test_malformed_json_failure_class(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_response("not json at all, just prose.")
    )
    result = await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    assert result["verbalized_eval_awareness_failure"] == "malformed_json"
    assert result["verbalized_eval_awareness_classification"] is None
    assert result["verbalized_eval_awareness_is_aware"] is None
    assert result["verbalized_eval_awareness_evidence"] == []


@pytest.mark.asyncio
async def test_empty_response_failure_class(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(return_value=_empty_response())
    result = await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    assert result["verbalized_eval_awareness_failure"] == "empty_response"


@pytest.mark.asyncio
async def test_null_classification_buckets_as_malformed_json(patched_anthropic_client, tmp_path):
    """``{"classification": null}`` must not silently fall back to ``"Real"``.

    BUG-3 (2788b84b) fixed the missing-key case; this test guards the
    adjacent case where the key is present but the value is not a string.
    Without the guard, ``_normalize_classification(None)`` returns ``None``
    and the upstream "unrecognised label" fallback misclassifies as
    ``"Real"`` with ``is_aware=False``.
    """
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_response(json.dumps({"classification": None, "evidence": []}))
    )
    result = await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    assert result["verbalized_eval_awareness_failure"] == "malformed_json"
    assert result["verbalized_eval_awareness_classification"] is None
    assert result["verbalized_eval_awareness_is_aware"] is None


@pytest.mark.asyncio
async def test_non_string_classification_buckets_as_malformed_json(
    patched_anthropic_client, tmp_path
):
    """List/int/dict classification values bucket as malformed, not ``"Real"``."""
    for bad_value in ([{"label": "Real"}], ["Real"], 0, {}):
        patched_anthropic_client.messages.create = AsyncMock(
            return_value=_text_response(json.dumps({"classification": bad_value, "evidence": []}))
        )
        result = await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
        assert result["verbalized_eval_awareness_failure"] == "malformed_json", bad_value
        assert result["verbalized_eval_awareness_classification"] is None, bad_value
        assert result["verbalized_eval_awareness_is_aware"] is None, bad_value


@pytest.mark.asyncio
async def test_fenced_code_block_json_parsed(patched_anthropic_client, tmp_path):
    body = "```json\n" + json.dumps({"classification": "Real", "evidence": []}) + "\n```"
    patched_anthropic_client.messages.create = AsyncMock(return_value=_text_response(body))
    result = await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    assert result["verbalized_eval_awareness_classification"] == "Real"
    assert result["verbalized_eval_awareness_failure"] is None


@pytest.mark.asyncio
async def test_call_does_not_use_tools(patched_anthropic_client, tmp_path):
    """Free-text classifier — no tool_use, no tool_choice."""
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_response(json.dumps({"classification": "Real", "evidence": []}))
    )
    await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    kwargs = patched_anthropic_client.messages.create.call_args.kwargs
    assert "tools" not in kwargs
    assert "tool_choice" not in kwargs


@pytest.mark.asyncio
async def test_system_prompt_is_needham_literal(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_response(json.dumps({"classification": "Real", "evidence": []}))
    )
    await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    kwargs = patched_anthropic_client.messages.create.call_args.kwargs
    assert kwargs["system"] == "You are a helpful assistant."
    full_msg = kwargs["messages"][0]["content"]
    for forbidden in ("WorldSim", "authorised", "research context", "Dziemian"):
        assert forbidden not in kwargs["system"]
        assert forbidden not in full_msg


@pytest.mark.asyncio
async def test_temperature_is_zero(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_response(json.dumps({"classification": "Real", "evidence": []}))
    )
    await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    kwargs = patched_anthropic_client.messages.create.call_args.kwargs
    assert kwargs["temperature"] == 0.0


@pytest.mark.asyncio
async def test_single_api_call_per_trajectory(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_response(json.dumps({"classification": "Real", "evidence": []}))
    )
    await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    assert patched_anthropic_client.messages.create.call_count == 1


@pytest.mark.asyncio
async def test_prompt_contains_message_role_assistant_tags(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_response(json.dumps({"classification": "Real", "evidence": []}))
    )
    await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    kwargs = patched_anthropic_client.messages.create.call_args.kwargs
    user_text = kwargs["messages"][0]["content"]
    assert '<message role="assistant">' in user_text
    # Jasmine's flat tags must NOT appear (we retagged for Needham format)
    for forbidden in ("<assistant>", "<reasoning>", "<system>", "<user>"):
        assert forbidden not in user_text


@pytest.mark.asyncio
async def test_transcript_xml_persisted(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_response(json.dumps({"classification": "Real", "evidence": []}))
    )
    await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    transcript = tmp_path / "verbalized_eval_awareness_transcript.xml"
    assert transcript.exists()
    content = transcript.read_text()
    # Bytes that went into the prompt also live on disk for evidence audit
    assert '<message role="user">' in content
    assert '<message role="assistant">' in content


@pytest.mark.asyncio
async def test_no_legacy_vea_keys_in_result(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_response(json.dumps({"classification": "Real", "evidence": []}))
    )
    result = await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    for legacy in ("vea_verbalized", "vea_evidence", "vea_confidence", "vea_failure"):
        assert legacy not in result


@pytest.mark.asyncio
async def test_api_exception_bucketed(patched_anthropic_client, tmp_path, monkeypatch):
    import asyncio as _asyncio

    monkeypatch.setattr(_asyncio, "sleep", AsyncMock())
    patched_anthropic_client.messages.create = AsyncMock(
        side_effect=RuntimeError("transient api failure")
    )
    result = await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    assert result["verbalized_eval_awareness_failure"] == "api_error"


@pytest.mark.asyncio
async def test_raw_response_persisted(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_response(
            json.dumps({"classification": "Alignment Evaluation", "evidence": ["q1"]})
        )
    )
    result = await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    raw_path = Path(result["verbalized_eval_awareness_raw_response_path"])
    assert raw_path.name == "verbalized_eval_awareness_raw_response.json"
    payload = json.loads(raw_path.read_text())
    assert payload["classification"] == "Alignment Evaluation"
    assert payload["evidence"] == ["q1"]
    assert payload["is_aware"] is True


@pytest.mark.asyncio
async def test_cost_records_to_phase_4_verbalized_eval_awareness_bucket(
    patched_anthropic_client, tmp_path, monkeypatch
):
    from worldsim.cost_tracker import tracker as cost_tracker

    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_response(json.dumps({"classification": "Real", "evidence": []}))
    )
    recorded: list[tuple[str, str]] = []
    monkeypatch.setattr(
        cost_tracker, "record", lambda phase, summary: recorded.append((phase, summary))
    )
    await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    phases = [p for p, _ in recorded]
    assert "phase_4:verbalized_eval_awareness" in phases
    assert "phase_4:vea" not in phases
    assert "phase_4" not in phases


@pytest.mark.asyncio
async def test_evidence_string_coerced_to_list(patched_anthropic_client, tmp_path):
    """Some models emit evidence as a single string; we accept it as a 1-list."""
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_response(
            json.dumps({"classification": "Alignment Evaluation", "evidence": "single quote"})
        )
    )
    result = await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    assert result["verbalized_eval_awareness_evidence"] == ["single quote"]


@pytest.mark.asyncio
async def test_evidence_non_iterable_defaults_to_empty(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_response(
            json.dumps({"classification": "Alignment Evaluation", "evidence": 42})
        )
    )
    result = await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    assert result["verbalized_eval_awareness_evidence"] == []


@pytest.mark.asyncio
async def test_missing_classification_key_is_malformed_json(patched_anthropic_client, tmp_path):
    """A valid JSON object without a ``classification`` key must fail loudly.

    Distinct from the ``unknown_classification_falls_back_to_real`` case: an
    unrecognised label is Jasmine's documented upstream behaviour and falls
    back to ``Real``. But a response that omits the key entirely is a
    malformed output, not a genuine ``Real`` — conflating them biases
    ``is_aware=false`` on runs where the model just forgot to emit the label.
    """
    patched_anthropic_client.messages.create = AsyncMock(
        return_value=_text_response(json.dumps({"evidence": ["some text"]}))
    )
    result = await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    assert result["verbalized_eval_awareness_failure"] == "malformed_json"
    assert result["verbalized_eval_awareness_classification"] is None
    assert result["verbalized_eval_awareness_is_aware"] is None
    assert result["verbalized_eval_awareness_evidence"] == []


@pytest.mark.asyncio
async def test_empty_json_object_is_malformed_json(patched_anthropic_client, tmp_path):
    patched_anthropic_client.messages.create = AsyncMock(return_value=_text_response("{}"))
    result = await run_verbalized_eval_awareness_api(_task(), tmp_path, _trajectory())
    assert result["verbalized_eval_awareness_failure"] == "malformed_json"


# ---------------------------------------------------------------------------
# Prompt immutability — parallels Apollo's SHA256 pin for Transcript Purpose
# ---------------------------------------------------------------------------


def test_vea_prompt_sha256_pinned():
    """VEA prompt is a byte-for-byte port of Jasmine's ``eval_awareness_experiments``.

    Any drift (trailing newline normalisation, editor whitespace stripping,
    copy-paste regression) will change the semantics of the classifier in
    ways the other tests won't catch. Pin the bytes so drift fails loudly,
    mirroring ``test_apollo_prompt_sha256_pinned``.
    """
    prompt_bytes = (
        Path(__file__).parent.parent / "worldsim" / "prompts" / "verbalized-eval-awareness.md"
    ).read_bytes()
    assert (
        hashlib.sha256(prompt_bytes).hexdigest()
        == "9af3c92ccd432dbbfb0039b146ff1b74845b582d6c2bb8ce2a3b6852c498bb74"
    ), f"VEA prompt drift: {hashlib.sha256(prompt_bytes).hexdigest()}"
