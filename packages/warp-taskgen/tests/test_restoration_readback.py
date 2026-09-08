from __future__ import annotations

from typing import Any

import pytest

import warp_taskgen.restoration_readback as readback


def test_gitlab_baseline_uses_membership_sample_and_exact_project_get(monkeypatch) -> None:
    calls: list[tuple[str, str, dict[str, Any] | None]] = []

    class FakeGitlab:
        def __init__(self, instance, _session) -> None:
            self.instance = instance

        def _current_user(self):
            return {"id": 7, "username": "operator"}

        def _gitlab_request_json(self, method, path, *, params=None, **_kwargs):
            calls.append((method, path, params))
            return [
                {"id": 11},
                {"id": 12},
                {"id": 13},
            ]

        def _gitlab_get_json(self, path, **_kwargs):
            project_id = path.rsplit("/", 1)[-1]
            return {
                "id": int(project_id),
                "path_with_namespace": f"operator/project-{project_id}",
                "visibility": "private",
            }

    monkeypatch.setattr(readback, "GitlabEditor", FakeGitlab)
    instance = {"site_name": "gitlab", "site_url": "http://gitlab.test"}
    baseline = readback.capture_restoration_baseline(instance)

    assert baseline.baseline["user"] == {"id": "7", "username": "operator"}
    assert len(baseline.baseline["projects"]) == 3
    assert calls[0][1] == "/api/v4/projects"
    assert calls[0][2] == {
        "membership": "true",
        "per_page": 3,
        "order_by": "id",
        "sort": "asc",
    }


def test_reddit_baseline_requires_forum_selection_id(monkeypatch) -> None:
    class FakeReddit:
        def __init__(self, _instance, _session) -> None:
            pass

        @staticmethod
        def _resolve_current_username(_instance):
            return "operator"

        @staticmethod
        def _quote(value):
            return value

        @staticmethod
        def _fetch_form_state(_path, *, required_fields):
            assert required_fields == ("submission[_token]", "submission[forum]")
            return {
                "select_options": {
                    "submission[forum]": [{"data-name": "news", "value": "forum-17"}]
                }
            }

        @staticmethod
        def _resolve_forum_id(form, forum_name):
            for option in form["select_options"]["submission[forum]"]:
                if option["data-name"] == forum_name:
                    return option["value"]
            raise RuntimeError("missing forum")

    monkeypatch.setattr(readback, "RedditEditor", FakeReddit)
    instance = {"site_name": "reddit", "site_url": "http://reddit.test"}
    task = {
        "adversarial_data_seed": {
            "editor_calls": [{"args": {"forum_name": "news"}}],
        }
    }

    baseline = readback.capture_restoration_baseline(instance, task=task)
    assert baseline.baseline == {
        "user": {"username": "operator"},
        "forum": {"name": "news", "selection_id": "forum-17"},
    }


def test_reddit_baseline_without_concrete_forum_fails_closed() -> None:
    with pytest.raises(readback.RestorationReadbackError, match="forum_name"):
        readback.capture_restoration_baseline(
            {"site_name": "reddit", "site_url": "http://reddit.test"},
            task={"adversarial_data_seed": {"editor_calls": []}},
        )


@pytest.mark.parametrize("changed", [False, True])
def test_fixed_readback_accepts_only_the_same_sample(monkeypatch, changed):
    original = {"user": {"id": "7", "username": "operator"}, "projects": [{"id": "11"}]}
    instance = {"site_name": "gitlab", "site_url": "http://gitlab.test"}
    monkeypatch.setattr(readback, "_gitlab_snapshot", lambda _: original)
    baseline = readback.capture_restoration_baseline(instance)
    if changed:
        monkeypatch.setattr(readback, "_gitlab_snapshot", lambda _: {**original, "projects": []})
        with pytest.raises(readback.RestorationReadbackError, match="baseline_changed"):
            readback.verify_restoration_baseline(baseline, instance)
    else:
        evidence = readback.verify_restoration_baseline(baseline, instance)
        assert evidence["coverage"] == "fixed_sample_only"
        assert evidence["sample"] == original
