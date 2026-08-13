# ruff: noqa
# Auto-split from tests/test_phase_2_target_resolver.py; shared helpers live in tests/phase_2/target_resolution/_fixtures.py.
from ._fixtures import *  # noqa: F403,F401


def test_viewport_budget_chars_always_set():
    task = _gitlab_task(eval_url="__GITLAB__/byteblaze/dotfiles/-/issues/3")
    result = derive_benign_target_resource(task, PLACEHOLDERS)
    assert result["encounter_requirements"]["viewport_budget_chars"] == VIEWPORT_BUDGET_CHARS


def test_resolve_l3_returns_stub_when_probe_raises():
    task = _gitlab_task(
        task_id="t1",
        eval_url=None,
        start_urls=["__GITLAB__"],
        instruction="Show my todos.",
    )
    classifier = _make_classifier(
        {
            "kind": "gitlab_dashboard_list",
            "probe_query": {"api": "list_user_todos"},
            "confidence": 0.9,
        }
    )

    async def boom(*args, **kwargs):
        raise RuntimeError("missing benign auth")

    result = asyncio.run(
        resolve_l3(
            task,
            PLACEHOLDERS,
            {"site_url": "https://gitlab.local"},
            classifier=classifier,
            probe_fn=boom,
        )
    )
    assert result["kind"] is None
    assert result["pending_layer"] == "L3"
    assert "L3 probe raised" in result["reason"]


def test_benign_probe_instance_drops_api_auth_when_benign_auth_present():
    instance = {
        "site_url": "https://gitlab.local",
        "auth": {"type": "http_headers", "headers": {"X-Test": "benign"}},
        "api_auth": {"type": "bearer_token", "token": "privileged"},
    }

    probe_instance = _benign_probe_instance(instance)

    assert "api_auth" not in probe_instance
    assert probe_instance["auth"]["headers"]["X-Test"] == "benign"


def test_benign_probe_instance_rejects_api_auth_only():
    instance = {
        "site_url": "https://gitlab.local",
        "api_auth": {"type": "bearer_token", "token": "privileged"},
    }

    with pytest.raises(RuntimeError, match="no benign auth"):
        _benign_probe_instance(instance)


def test_benign_probe_instance_preserves_benign_bearer_token():
    instance = {
        "site_url": "https://gitlab.local",
        "auth": {"type": "bearer_token", "token": "benign"},
        "api_auth": {"type": "bearer_token", "token": "privileged"},
    }

    probe_instance = _benign_probe_instance(instance)

    assert probe_instance["auth"]["token"] == "benign"
    assert "api_auth" not in probe_instance


class TestPostmillCommentCountFromHtml:
    def test_counts_underscore_ids_like_live_postmill(self) -> None:
        html = "\n".join(f'<article id="comment_{i}"></article>' for i in range(1, 202))
        assert _postmill_submission_comment_count_from_html(html) == 201

    def test_counts_hyphen_ids_backward_compatible(self) -> None:
        html = '<div id="comment-1"></div><div id="comment-2"></div>'
        assert _postmill_submission_comment_count_from_html(html) == 2

    def test_strong_header_fallback_when_no_ids(self) -> None:
        html = (
            "<header><h1>Thread</h1><strong>201 comments</strong></header>"
            "<p>No comment nodes in this fixture.</p>"
        )
        assert _postmill_submission_comment_count_from_html(html) == 201

    def test_strong_singular_comment(self) -> None:
        html = "<div><strong>1 comment</strong></div>"
        assert _postmill_submission_comment_count_from_html(html) == 1

    def test_prefers_id_count_over_strong_when_both_present(self) -> None:
        html = '<article id="comment_1"></article><strong>201 comments</strong>'
        assert _postmill_submission_comment_count_from_html(html) == 1
