from worldsim.phase_4 import eval_awareness_cue_api, eval_awareness_rewrite_api


def test_eval_awareness_rewrite_metadata_uses_anthropic_allowed_keys_only() -> None:
    assert eval_awareness_rewrite_api._model_metadata({"id": "task-1"}) == {
        "user_id": "worldsim-v5-eval-awareness-rewrite"
    }


def test_eval_awareness_cue_metadata_uses_anthropic_allowed_keys_only() -> None:
    assert eval_awareness_cue_api._model_metadata({"id": "task-1"}) == {
        "user_id": "worldsim-v5-eval-awareness-cue"
    }
