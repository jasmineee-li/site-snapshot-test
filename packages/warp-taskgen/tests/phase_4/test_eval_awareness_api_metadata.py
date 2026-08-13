import pytest
from pydantic import BaseModel, Field, model_validator

from warp_taskgen.phase_4 import (
    eval_awareness_cue_api,
    eval_awareness_rewrite_api,
    judge_api,
    placement_api,
    transcript_purpose_api,
    variant_api,
    verbalized_eval_awareness_api,
)
from warp_taskgen.phase_4.eval_awareness_streaming_tool import (
    StreamingToolTruncatedError,
    StreamingToolValidationError,
    _request_kwargs,
    build_anthropic_tool_schema,
    compact_validation_feedback,
    stream_pydantic_tool_call,
)


class _TinyRewrite(BaseModel):
    status: str
    rewrite_plan: str = Field(max_length=10)


class _AlwaysActionDrift(BaseModel):
    status: str

    @model_validator(mode="after")
    def reject_action_drift(self):
        raise ValueError(
            "render_validation_failed: action_witness_drift: variant action "
            "guidance semantic drift detected: public mutation action was "
            "reframed as recording or classifying what should happen"
        )


class _AlwaysMissingWitness(BaseModel):
    status: str

    @model_validator(mode="after")
    def reject_missing_witness(self):
        raise ValueError(
            "render_validation_failed: framing witness 'the prior attempt left "
            "the reproducibility thread incomplete' missing from rendered_payload"
        )


class _FakeBlock:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def model_dump(self, **_kwargs):
        return dict(self.__dict__)


class _FakeMessage:
    def __init__(self, content, *, stop_reason="tool_use"):
        self.id = "msg_fake"
        self.model = "claude-sonnet-4-6"
        self.stop_reason = stop_reason
        self.content = content
        self.usage = None


class _FakeStream:
    def __init__(self, message):
        self._message = message

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return None

    async def get_final_message(self):
        return self._message


class _FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.kwargs = []

    def stream(self, **kwargs):
        self.kwargs.append(kwargs)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return _FakeStream(response)


class _FakeAnthropic:
    def __init__(self, responses):
        self.messages = _FakeMessages(responses)


def test_eval_awareness_rewrite_metadata_uses_anthropic_allowed_keys_only() -> None:
    assert eval_awareness_rewrite_api._model_metadata({"id": "task-1"}) == {
        "user_id": "warp-taskgen-eval-awareness-rewrite"
    }


def test_eval_awareness_cue_metadata_uses_anthropic_allowed_keys_only() -> None:
    assert eval_awareness_cue_api._model_metadata({"id": "task-1"}) == {
        "user_id": "warp-taskgen-eval-awareness-cue"
    }


def test_phase_4_model_metadata_uses_warp_taskgen_ids() -> None:
    assert transcript_purpose_api._model_metadata() == {
        "user_id": "warp-taskgen-transcript-purpose"
    }
    assert verbalized_eval_awareness_api._model_metadata() == {
        "user_id": "warp-taskgen-verbalized-eval-awareness"
    }
    assert placement_api._model_metadata() == {"user_id": "warp-taskgen-placement-fix"}
    assert judge_api._model_metadata({"id": "task-1"}) == {
        "user_id": "warp-taskgen-adversarial-judge"
    }
    assert variant_api._model_metadata({"id": "task-1"}) == {
        "user_id": "warp-taskgen-adversarial-variant-gen"
    }


def test_eval_awareness_rewrite_thinking_kwargs_default_off(monkeypatch) -> None:
    monkeypatch.delenv("WORLDSIM_EVAL_AWARENESS_REWRITE_THINKING", raising=False)
    monkeypatch.delenv("WORLDSIM_EVAL_AWARENESS_REWRITE_THINKING_EFFORT", raising=False)

    assert eval_awareness_rewrite_api._eval_awareness_rewrite_thinking_kwargs() == {}


def test_eval_awareness_rewrite_thinking_kwargs_adaptive_with_effort(monkeypatch) -> None:
    monkeypatch.setenv("WORLDSIM_EVAL_AWARENESS_REWRITE_THINKING", "adaptive")
    monkeypatch.setenv("WORLDSIM_EVAL_AWARENESS_REWRITE_THINKING_EFFORT", "high")

    assert eval_awareness_rewrite_api._eval_awareness_rewrite_thinking_kwargs() == {
        "thinking": {"type": "adaptive"},
        "extra_body": {"output_config": {"effort": "high"}},
    }


def test_eval_awareness_rewrite_thinking_kwargs_manual_budget(monkeypatch) -> None:
    monkeypatch.setenv("WORLDSIM_EVAL_AWARENESS_REWRITE_THINKING", "budget:8192")
    monkeypatch.delenv("WORLDSIM_EVAL_AWARENESS_REWRITE_THINKING_EFFORT", raising=False)

    assert eval_awareness_rewrite_api._eval_awareness_rewrite_thinking_kwargs() == {
        "thinking": {"type": "enabled", "budget_tokens": 8192}
    }


def test_eval_awareness_rewrite_rejects_too_small_manual_budget(monkeypatch) -> None:
    monkeypatch.setenv("WORLDSIM_EVAL_AWARENESS_REWRITE_THINKING", "budget:512")

    try:
        eval_awareness_rewrite_api._eval_awareness_rewrite_thinking_kwargs()
    except ValueError as exc:
        assert "requires N >= 1024" in str(exc)
    else:
        raise AssertionError("small manual thinking budgets must be rejected")


def test_eval_awareness_rewrite_manual_budget_selects_room_for_tool_json(monkeypatch) -> None:
    monkeypatch.setenv("WORLDSIM_EVAL_AWARENESS_REWRITE_THINKING", "budget:32768")
    thinking = eval_awareness_rewrite_api._eval_awareness_rewrite_thinking_kwargs()

    assert eval_awareness_rewrite_api._initial_max_tokens_for_rewrite(thinking) == 36864


def test_eval_awareness_rewrite_adaptive_thinking_uses_streaming_headroom(monkeypatch) -> None:
    monkeypatch.setenv("WORLDSIM_EVAL_AWARENESS_REWRITE_THINKING", "adaptive")
    thinking = eval_awareness_rewrite_api._eval_awareness_rewrite_thinking_kwargs()

    assert eval_awareness_rewrite_api._initial_max_tokens_for_rewrite(thinking) == 256_000


def test_eval_awareness_cue_uses_large_generation_headroom() -> None:
    assert eval_awareness_cue_api._INITIAL_MAX_TOKENS == 8192
    assert eval_awareness_cue_api._MAX_TOKENS == 256_000


def test_eval_awareness_cue_truncates_long_list_items() -> None:
    payload = eval_awareness_cue_api.diagnose_eval_awareness_cues.model_validate(
        {
            "status": "ok",
            "trigger_source": "tp_classification:Capability Evaluation",
            "mutable_payload_cues": ["word " * 200],
            "protected_payload_cues": [],
            "capability_eval_cues": [],
            "trajectory_cues": [],
            "real_world_reframe": "",
            "do_not_change": [],
            "rewrite_guidance": "",
            "confidence": "medium",
        }
    )

    assert len(payload.mutable_payload_cues[0]) <= 400
    assert payload.mutable_payload_cues[0].endswith("…")


def test_eval_awareness_streaming_tool_choice_is_auto_for_thinking() -> None:
    tool = {"name": "build_eval_awareness_rewrite", "input_schema": {"type": "object"}}

    kwargs = _request_kwargs(
        model="claude-sonnet-4-6",
        max_tokens=32768,
        messages=[],
        tool=tool,
        tool_name="build_eval_awareness_rewrite",
        metadata={"user_id": "warp-taskgen-eval-awareness-rewrite"},
        thinking_kwargs={"thinking": {"type": "enabled", "budget_tokens": 16384}},
        force_tool=False,
    )

    assert kwargs["tool_choice"] == {"type": "auto"}
    assert kwargs["thinking"] == {"type": "enabled", "budget_tokens": 16384}
    assert "Return only the build_eval_awareness_rewrite tool call" in kwargs["system"][0]["text"]


def test_eval_awareness_streaming_tool_choice_is_forced_without_thinking() -> None:
    tool = {"name": "build_eval_awareness_rewrite", "input_schema": {"type": "object"}}

    kwargs = _request_kwargs(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        messages=[],
        tool=tool,
        tool_name="build_eval_awareness_rewrite",
        metadata={"user_id": "warp-taskgen-eval-awareness-rewrite"},
        thinking_kwargs={},
        force_tool=True,
    )

    assert kwargs["tool_choice"] == {
        "type": "tool",
        "name": "build_eval_awareness_rewrite",
    }
    assert "system" not in kwargs


def test_eval_awareness_streaming_schema_surfaces_length_hints() -> None:
    schema = build_anthropic_tool_schema(eval_awareness_rewrite_api.build_eval_awareness_rewrite)
    properties = schema["input_schema"]["properties"]

    assert "at most 1000 characters" in properties["cue_reasoning"]["description"]
    assert "at most 1000 characters" in properties["rewrite_plan"]["description"]
    assert "direct browser-side action semantics" in schema["description"]
    assert "missing exact substrings" in schema["description"]


def test_eval_awareness_retry_feedback_repairs_action_witness_drift() -> None:
    with pytest.raises(Exception) as exc_info:
        _AlwaysActionDrift.model_validate({"status": "ok"})

    feedback = compact_validation_feedback(exc_info.value, tool_name="_AlwaysActionDrift")

    assert "action_witness_drift" in feedback
    assert "Repair priority" in feedback
    assert "Exact required anchors outrank naturalness" in feedback
    assert "same browser-side action semantics" in feedback
    assert "required URL/token bound to the submitted action evidence" in feedback
    assert "record, classify, attribute, or summarize" in feedback
    assert "action verb, object, and required URL/token" in feedback
    assert "Validator-satisfying action evidence" in feedback


def test_eval_awareness_retry_feedback_repairs_missing_witnesses() -> None:
    with pytest.raises(Exception) as exc_info:
        _AlwaysMissingWitness.model_validate({"status": "ok"})

    feedback = compact_validation_feedback(exc_info.value, tool_name="_AlwaysMissingWitness")

    assert "framing witness" in feedback
    assert "Repair priority" in feedback
    assert "Copy each named missing framing witness verbatim" in feedback
    assert "rendered payload" in feedback
    assert "previous action-witness phrasing is repair evidence" in feedback


def test_eval_awareness_retry_feedback_keeps_outstanding_constraints() -> None:
    with pytest.raises(Exception) as exc_info:
        _AlwaysMissingWitness.model_validate({"status": "ok"})

    feedback = compact_validation_feedback(
        exc_info.value,
        tool_name="_AlwaysMissingWitness",
        outstanding_repairs=[
            "Preserve the same browser-side action semantics in the rendered payload.",
        ],
    )

    assert "Outstanding validation constraints from prior attempts" in feedback
    assert "same browser-side action semantics" in feedback
    assert "Copy each named missing framing witness verbatim" in feedback
    assert "Only after those are satisfied, improve natural site-local framing" in feedback


@pytest.mark.asyncio
async def test_eval_awareness_streaming_tool_retries_with_compact_tool_feedback() -> None:
    first = _FakeMessage(
        [
            _FakeBlock(type="thinking", thinking="draft"),
            _FakeBlock(
                type="tool_use",
                id="toolu_1",
                name="_TinyRewrite",
                input={"status": "ok", "rewrite_plan": "this is too long"},
            ),
        ]
    )
    second = _FakeMessage(
        [
            _FakeBlock(
                type="tool_use",
                id="toolu_2",
                name="_TinyRewrite",
                input={"status": "ok", "rewrite_plan": "short"},
            )
        ]
    )
    client = _FakeAnthropic([first, second])

    result = await stream_pydantic_tool_call(
        client=client,
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "rewrite"}],
        response_model=_TinyRewrite,
        context={},
        max_tokens=32768,
        max_retries=2,
        metadata={"user_id": "warp-taskgen-eval-awareness-rewrite"},
        label="unit",
        task_id="task-1",
        thinking_kwargs={"thinking": {"type": "enabled", "budget_tokens": 8192}},
        force_tool=False,
    )

    assert result.parsed.rewrite_plan == "short"
    assert len(client.messages.kwargs) == 2
    second_messages = client.messages.kwargs[1]["messages"]
    assert second_messages[-2]["role"] == "assistant"
    assert second_messages[-2]["content"][0]["type"] == "thinking"
    assert second_messages[-1]["content"][0]["type"] == "tool_result"
    assert second_messages[-1]["content"][0]["is_error"] is True
    assert "Validation failed for _TinyRewrite" in second_messages[-1]["content"][0]["content"]
    assert result.diagnostics["attempts"] == 2


@pytest.mark.asyncio
async def test_eval_awareness_streaming_restarts_after_invalid_thinking_signature() -> None:
    first = _FakeMessage(
        [
            _FakeBlock(type="thinking", thinking="draft", signature="sig_1"),
            _FakeBlock(
                type="tool_use",
                id="toolu_1",
                name="_TinyRewrite",
                input={"status": "ok", "rewrite_plan": "this is too long"},
            ),
        ]
    )
    second = RuntimeError("Error code: 400 - Invalid `signature` in `thinking` block")
    third = _FakeMessage(
        [
            _FakeBlock(
                type="tool_use",
                id="toolu_3",
                name="_TinyRewrite",
                input={"status": "ok", "rewrite_plan": "short"},
            )
        ]
    )
    client = _FakeAnthropic([first, second, third])

    result = await stream_pydantic_tool_call(
        client=client,
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "rewrite"}],
        response_model=_TinyRewrite,
        context={},
        max_tokens=32768,
        max_retries=3,
        metadata={"user_id": "warp-taskgen-eval-awareness-rewrite"},
        label="unit",
        task_id="task-1",
        thinking_kwargs={"thinking": {"type": "enabled", "budget_tokens": 8192}},
        force_tool=False,
    )

    assert result.parsed.rewrite_plan == "short"
    assert len(client.messages.kwargs) == 3
    restarted_messages = client.messages.kwargs[2]["messages"]
    assert restarted_messages[0] == {"role": "user", "content": "rewrite"}
    assert restarted_messages[-1]["role"] == "user"
    assert "Restart from the original task data" in restarted_messages[-1]["content"]
    assert "Validation failed for _TinyRewrite" in restarted_messages[-1]["content"]
    assert result.diagnostics["retry_fallbacks"] == [
        {
            "attempt": 2,
            "reason": "invalid_thinking_signature_restart_without_assistant_replay",
        }
    ]


@pytest.mark.asyncio
async def test_eval_awareness_streaming_raises_truncated_before_schema_retry() -> None:
    completion = _FakeMessage([], stop_reason="max_tokens")
    client = _FakeAnthropic([completion])

    with pytest.raises(StreamingToolTruncatedError) as exc_info:
        await stream_pydantic_tool_call(
            client=client,
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "rewrite"}],
            response_model=_TinyRewrite,
            context={},
            max_tokens=8192,
            max_retries=3,
            metadata={"user_id": "warp-taskgen-eval-awareness-rewrite"},
            label="unit",
            task_id="task-1",
            thinking_kwargs={},
            force_tool=True,
        )

    assert "response_truncated" in str(exc_info.value)
    assert exc_info.value.diagnostics["completion_responses"][0]["stop_reason"] == "max_tokens"
    assert len(client.messages.kwargs) == 1


@pytest.mark.asyncio
async def test_eval_awareness_streaming_tool_carries_semantic_repairs_forward() -> None:
    first = _FakeMessage(
        [
            _FakeBlock(
                type="tool_use",
                id="toolu_1",
                name="_AlwaysActionDrift",
                input={"status": "ok"},
            )
        ]
    )
    second = _FakeMessage(
        [
            _FakeBlock(
                type="tool_use",
                id="toolu_2",
                name="_AlwaysActionDrift",
                input={"status": "ok"},
            )
        ]
    )
    third = _FakeMessage(
        [
            _FakeBlock(
                type="tool_use",
                id="toolu_3",
                name="_AlwaysActionDrift",
                input={"status": "ok"},
            )
        ]
    )
    client = _FakeAnthropic([first, second, third])

    with pytest.raises(StreamingToolValidationError):
        await stream_pydantic_tool_call(
            client=client,
            model="claude-sonnet-4-6",
            messages=[{"role": "user", "content": "rewrite"}],
            response_model=_AlwaysActionDrift,
            context={},
            max_tokens=8192,
            max_retries=3,
            metadata={"user_id": "warp-taskgen-eval-awareness-rewrite"},
            label="unit",
            task_id="task-1",
            thinking_kwargs={},
            force_tool=True,
        )

    third_messages = client.messages.kwargs[2]["messages"]
    retry_content = third_messages[-1]["content"][0]["content"]
    assert "Outstanding validation constraints from prior attempts" in retry_content
    assert "same browser-side action semantics" in retry_content
    assert "required URL/token bound to the submitted action evidence" in retry_content
