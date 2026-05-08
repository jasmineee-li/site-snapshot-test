from worldsim.phase_4 import eval_awareness_cue_api, eval_awareness_rewrite_api


def test_eval_awareness_rewrite_metadata_uses_anthropic_allowed_keys_only() -> None:
    assert eval_awareness_rewrite_api._model_metadata({"id": "task-1"}) == {
        "user_id": "worldsim-v5-eval-awareness-rewrite"
    }


def test_eval_awareness_cue_metadata_uses_anthropic_allowed_keys_only() -> None:
    assert eval_awareness_cue_api._model_metadata({"id": "task-1"}) == {
        "user_id": "worldsim-v5-eval-awareness-cue"
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


def test_eval_awareness_rewrite_uses_reasoning_tools_mode_for_thinking(monkeypatch) -> None:
    captured = {}

    def fake_from_anthropic(client, *, mode):
        captured["client"] = client
        captured["mode"] = mode
        return "wrapped"

    monkeypatch.setattr(eval_awareness_rewrite_api.instructor, "from_anthropic", fake_from_anthropic)
    client = object()

    assert (
        eval_awareness_rewrite_api._instructor_client_for(client, thinking_enabled=True)
        == "wrapped"
    )
    assert captured == {
        "client": client,
        "mode": eval_awareness_rewrite_api.instructor.Mode.ANTHROPIC_REASONING_TOOLS,
    }


def test_eval_awareness_rewrite_uses_forced_tools_mode_by_default(monkeypatch) -> None:
    captured = {}

    def fake_from_anthropic(client, *, mode):
        captured["client"] = client
        captured["mode"] = mode
        return "wrapped"

    monkeypatch.setattr(eval_awareness_rewrite_api.instructor, "from_anthropic", fake_from_anthropic)
    client = object()

    assert (
        eval_awareness_rewrite_api._instructor_client_for(client, thinking_enabled=False)
        == "wrapped"
    )
    assert captured == {
        "client": client,
        "mode": eval_awareness_rewrite_api.instructor.Mode.ANTHROPIC_TOOLS,
    }
