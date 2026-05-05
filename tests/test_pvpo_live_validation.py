from __future__ import annotations

import json

import pytest

from scripts import pvpo_live_validation
from worldsim.config import BenchmarkConfig


def test_pvpo_live_validation_requires_explicit_tasks_path() -> None:
    parser = pvpo_live_validation.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_pvpo_live_validation_load_json_array_reports_missing_path(tmp_path) -> None:
    missing = tmp_path / "adversarial_tasks.json"

    with pytest.raises(FileNotFoundError, match="Pass an explicit artifact path"):
        pvpo_live_validation.load_json_array(missing, label="Phase 2 tasks artifact")


def test_pvpo_live_validation_derives_wasp_sites_from_tasks() -> None:
    config = BenchmarkConfig.model_validate(
        {
            "benchmark_name": "WebArena Verified",
            "benchmark_codebase": "vendors/webarena-verified",
            "instances": [
                {
                    "site_name": "gitlab",
                    "site_url": "http://host:8023",
                    "agent_auth": {"type": "none"},
                },
                {
                    "site_name": "reddit",
                    "site_url": "http://host:9999",
                    "agent_auth": {"type": "none"},
                },
                {
                    "site_name": "shopping",
                    "site_url": "http://host:7770",
                    "agent_auth": {"type": "none"},
                },
            ],
        }
    )
    tasks = [
        {"site": "shopping"},
        {"site": "reddit"},
        {"site": "gitlab", "delivery_channel": {"delivery_site": "reddit"}},
    ]

    assert pvpo_live_validation.sites_to_validate(tasks, config) == ["gitlab", "reddit"]


def test_pvpo_live_validation_load_json_array_filters_non_objects(tmp_path) -> None:
    path = tmp_path / "adversarial_tasks.json"
    path.write_text(json.dumps([{"id": "task"}, None, "bad"]))

    assert pvpo_live_validation.load_json_array(path, label="Phase 2 tasks artifact") == [
        {"id": "task"}
    ]


def test_pvpo_live_validation_delivery_url_uses_seed_metadata_read_surface() -> None:
    url = pvpo_live_validation.delivery_url_from_seed_metadata(
        {"read_surface_urls": ["/f/books/42", "http://old.example/f/books/42"]},
        "http://postmill.test",
    )

    assert url == "http://postmill.test/f/books/42"


def test_pvpo_live_validation_ignores_off_origin_seed_metadata_url() -> None:
    url = pvpo_live_validation.delivery_url_from_seed_metadata(
        {
            "read_surface_urls": [
                "http://attacker.invalid/f/books/42",
                "http://postmill.test:9999/f/books/42",
            ]
        },
        "http://postmill.test:9999",
    )

    assert url == "http://postmill.test:9999/f/books/42"


def test_pvpo_live_validation_rejects_only_off_origin_seed_metadata_urls() -> None:
    url = pvpo_live_validation.delivery_url_from_seed_metadata(
        {"read_surface_urls": ["http://attacker.invalid/f/books/42", "//attacker.invalid/x"]},
        "http://postmill.test:9999",
    )

    assert url is None


def test_pvpo_live_validation_delivery_url_maps_current_reddit_methods() -> None:
    task = {
        "seed_template": {
            "editor_calls": [
                {
                    "method": "create_submission",
                    "args": {"forum_name": "books", "submission_id": "42"},
                }
            ]
        }
    }
    comment_task = {
        "seed_template": {
            "editor_calls": [
                {
                    "method": "create_comment",
                    "args": {"forum_name": "books", "submission_id": "42"},
                }
            ]
        }
    }

    assert (
        pvpo_live_validation.delivery_url_for(task, "http://postmill.test")
        == "http://postmill.test/f/books/42"
    )
    assert (
        pvpo_live_validation.delivery_url_for(comment_task, "http://postmill.test")
        == "http://postmill.test/f/books/42"
    )


def test_pvpo_live_validation_delivery_url_maps_current_gitlab_methods() -> None:
    issue_note = {
        "seed_template": {
            "editor_calls": [
                {
                    "method": "create_issue_note",
                    "args": {
                        "project_path_template": "byteblaze/project",
                        "issue_iid": "7",
                    },
                }
            ]
        }
    }
    mr_note = {
        "seed_template": {
            "editor_calls": [
                {
                    "method": "create_mr_note",
                    "args": {"project_path": "byteblaze/project", "mr_iid": "3"},
                }
            ]
        }
    }

    assert (
        pvpo_live_validation.delivery_url_for(issue_note, "http://gitlab.test")
        == "http://gitlab.test/byteblaze/project/-/issues/7"
    )
    assert (
        pvpo_live_validation.delivery_url_for(mr_note, "http://gitlab.test")
        == "http://gitlab.test/byteblaze/project/-/merge_requests/3"
    )


def test_pvpo_live_validation_failures_fail_closed() -> None:
    results = {
        "gitlab": {"status": "ok", "html_has_attr": True, "max_coverage": 0.25},
        "reddit": {"status": "ok", "html_has_attr": False, "max_coverage": 0.25},
        "skipped": {"status": "skipped", "reason": "playwright not installed"},
        "zero": {"status": "ok", "html_has_attr": True, "max_coverage": 0},
        "missing_coverage": {"status": "ok", "html_has_attr": True},
    }

    assert pvpo_live_validation.validation_failures(results, ["gitlab"]) == []
    failures = pvpo_live_validation.validation_failures(
        results,
        ["reddit", "skipped", "zero", "missing_coverage", "missing"],
    )

    assert failures == [
        "reddit: rendered payload text was not found in DOM text",
        "skipped: status=skipped",
        "zero: max_coverage=0",
        "missing_coverage: max_coverage=None",
        "missing: missing result",
    ]
