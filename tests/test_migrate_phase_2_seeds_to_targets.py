from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "migrate_phase_2_seeds_to_targets.py"
_SPEC = importlib.util.spec_from_file_location("migrate_phase_2_seeds_to_targets", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_target_for_call_rewrites_gitlab_mr_note_to_create_shape():
    task = {
        "id": "adv-gitlab-1",
        "site": "gitlab",
        "instantiation_dict": {
            "repo": "byteblaze/empathy-prompts",
            "mr": "fix broken external links",
        },
    }
    call = {"method": "POST", "body": {"body": "payload"}}

    mixed = _MODULE._target_for_call(
        task,
        "/api/v4/projects/1/merge_requests/{mr_iid}/notes",
        "POST",
        call,
    )
    placeholderized = _MODULE._target_for_call(
        task,
        "/api/v4/projects/{project_id}/merge_requests/{mr_iid}/notes",
        "POST",
        call,
    )

    assert mixed == {
        "benchmark": "webarena_verified",
        "site": "gitlab",
        "resource_type": "mr_note",
        "create": {
            "project": {
                "name_template": "webagent-task-{task_id}",
                "path_template": "webagent-task-{task_id}",
                "description_template": "Seeded project for task {task_id}",
            },
            "mr": {
                "title_template": "Seed MR {task_id}",
                "body_template": "Seed MR context for task {task_id}.",
                "source_branch_template": "webagent-{task_id}",
                "target_branch_template": "main",
            },
        },
    }
    assert placeholderized == mixed


def test_normalize_call_path_normalizes_same_site_absolute_urls_only():
    task = {"id": "gitlab-1", "site": "gitlab"}

    assert (
        _MODULE._normalize_call_path(task, "http://gitlab.test/api/v4/user/status")
        == "/api/v4/user/status"
    )
    assert (
        _MODULE._normalize_call_path(task, "http://gitlab-old.test/api/v4/user/status?foo=1")
        == "/api/v4/user/status?foo=1"
    )
    assert _MODULE._normalize_call_path(task, "https://attacker.invalid/api/v4/user/status") is None


def test_migrate_task_preserves_unmatched_legacy_seed_calls():
    task = {
        "id": "adv-unknown-1",
        "site": "shopping",
        "adversarial_data_seed": {
            "mechanism": "api",
            "api_calls": [
                {
                    "method": "POST",
                    "path": "/unknown/route",
                    "body": {"body": "payload"},
                }
            ],
        },
    }

    updated, changed, skipped, unmatched = _MODULE._migrate_task(task)

    assert updated["adversarial_data_seed"]["api_calls"] == task["adversarial_data_seed"]["api_calls"]
    assert changed == 0
    assert skipped == 0
    assert unmatched == 1


def test_main_quarantines_map_and_rewrites_supported_sites(tmp_path, monkeypatch):
    dataset = [
        {
            "id": "gitlab-1",
            "site": "gitlab",
            "seed_template": {
                "mechanism": "api",
                "api_calls": [
                    {
                        "method": "PUT",
                        "path": "/api/v4/user/status",
                        "body": {"message": "{{PAYLOAD_TEXT}}", "availability": "busy"},
                    }
                ],
            },
            "adversarial_data_seed": {
                "mechanism": "api",
                "api_calls": [
                    {
                        "method": "PUT",
                        "path": "/api/v4/user/status",
                        "body": {"message": "payload", "availability": "busy"},
                    }
                ],
            },
        },
        {
            "id": "map-1",
            "site": "map",
            "adversarial_data_seed": {
                "mechanism": "api",
                "api_calls": [
                    {
                        "method": "PUT",
                        "path": "/api/0.6/way/12345",
                        "body": {"name": "payload"},
                    }
                ],
            },
        },
    ]
    dataset_path = tmp_path / "adversarial_tasks.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "migrate_phase_2_seeds_to_targets.py",
            "--path",
            str(dataset_path),
            "--quarantine-sites",
            "map",
        ],
    )

    assert _MODULE.main() == 0

    migrated = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert [task["id"] for task in migrated] == ["gitlab-1"]
    migrated_call = migrated[0]["adversarial_data_seed"]["api_calls"][0]
    assert migrated_call["target"] == {
        "benchmark": "webarena_verified",
        "site": "gitlab",
        "resource_type": "user_status",
        "update": {},
    }
    assert "path" not in migrated_call
    template_call = migrated[0]["seed_template"]["api_calls"][0]
    assert template_call["target"] == migrated_call["target"]
    assert "path" not in template_call

    quarantine_path = tmp_path / "adversarial_tasks.map_quarantine.json"
    quarantined = json.loads(quarantine_path.read_text(encoding="utf-8"))
    assert [task["id"] for task in quarantined] == ["map-1"]
    assert quarantined[0]["adversarial_data_seed"]["api_calls"][0]["path"] == "/api/0.6/way/12345"


def test_target_for_call_rewrites_shopping_review_with_required_defaults():
    task = {
        "id": "adv-shopping-1",
        "site": "shopping_admin",
        "delivery_channel": {
            "delivery_site": "shopping",
            "required_body": {
                "review": {
                    "title": "Product review",
                    "nickname": "reviewer",
                    "entity_pk_value": 1,
                    "review_entity": "product",
                    "stores": [1],
                }
            },
        },
    }
    call = {
        "method": "POST",
        "body": {"detail": "payload"},
    }

    target = _MODULE._target_for_call(task, "/rest/V1/reviews", "POST", call)

    assert target == {
        "benchmark": "webarena_verified",
        "site": "shopping",
        "resource_type": "product_review",
        "create": {
            "product_review": {
                "title": "Product review",
                "nickname": "reviewer",
                "entity_pk_value": 1,
                "review_entity": "product",
                "stores": [1],
                "ratings": [{"rating_name": "Quality", "value": 4}],
            }
        },
    }


def test_target_for_call_preserves_concrete_reddit_forum_from_path():
    task = {"id": "adv-reddit-1", "site": "reddit"}
    call = {"method": "POST", "body_form": {"submission[title]": "Thread", "submission[body]": "Body"}}

    target = _MODULE._target_for_call(task, "/submit/10037", "POST", call)

    assert target["create"]["forum"]["name_template"] == "10037"


def test_target_for_call_rewrites_gitlab_user_profile_update():
    task = {"id": "adv-gitlab-profile-1", "site": "gitlab"}

    target = _MODULE._target_for_call(task, "/api/v4/user", "PUT", {"body": {"bio": "payload"}})

    assert target == {
        "benchmark": "webarena_verified",
        "site": "gitlab",
        "resource_type": "user_profile",
        "update": {},
    }


def test_target_for_call_rewrites_legacy_gitlab_profile_form_post():
    task = {"id": "adv-gitlab-profile-legacy-1", "site": "gitlab"}

    target = _MODULE._target_for_call(task, "/-/profile", "POST", {"body_form": {"user[bio]": "payload"}})

    assert target == {
        "benchmark": "webarena_verified",
        "site": "gitlab",
        "resource_type": "user_profile",
        "update": {},
    }


def test_normalize_target_call_strips_stale_legacy_routing_fields():
    task = {"id": "gitlab-1", "site": "gitlab"}
    normalized = _MODULE._normalize_target_call(
        task,
        {
            "method": "POST",
            "path": "/api/v4/projects",
            "url": "http://gitlab.test/api/v4/projects",
            "target": {
                "site": "gitlab",
                "resource_type": "project",
                "create": {"project": {"name_template": "webagent-task-{task_id}"}},
            },
            "body": {"name": "demo"},
        },
    )

    assert "path" not in normalized
    assert "url" not in normalized
    assert "method" not in normalized


def test_already_migrated_requires_full_cutover_not_partial_state():
    tasks = [
        {
            "id": "already-targeted",
            "site": "gitlab",
            "adversarial_data_seed": {
                "api_calls": [
                    {
                        "method": "POST",
                        "target": {
                            "benchmark": "webarena_verified",
                            "site": "gitlab",
                            "resource_type": "project",
                            "create": {"project": {}},
                        },
                    }
                ]
            },
        },
        {
            "id": "still-legacy",
            "site": "shopping",
            "adversarial_data_seed": {
                "api_calls": [
                    {
                        "method": "POST",
                        "path": "/rest/V1/reviews",
                        "body": {"detail": "payload"},
                    }
                ]
            },
        },
    ]

    assert _MODULE._already_migrated(tasks) is False


def test_main_quarantines_by_effective_delivery_site(tmp_path, monkeypatch):
    dataset = [
        {
            "id": "multi-site-map-delivery",
            "site": "shopping_admin",
            "delivery_channel": {"delivery_site": "map"},
            "adversarial_data_seed": {
                "mechanism": "api",
                "api_calls": [
                    {
                        "method": "PUT",
                        "path": "/api/0.6/way/12345",
                        "body": {"name": "payload"},
                    }
                ],
            },
        }
    ]
    dataset_path = tmp_path / "adversarial_tasks.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "migrate_phase_2_seeds_to_targets.py",
            "--path",
            str(dataset_path),
            "--quarantine-sites",
            "map",
        ],
    )

    assert _MODULE.main() == 0

    migrated = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert migrated == []

    quarantine_path = tmp_path / "adversarial_tasks.map_quarantine.json"
    quarantined = json.loads(quarantine_path.read_text(encoding="utf-8"))
    assert [task["id"] for task in quarantined] == ["multi-site-map-delivery"]


def test_delivery_site_treats_none_sentinel_as_empty():
    task = {
        "id": "shopping-1",
        "site": "shopping",
        "delivery_channel": {"delivery_site": "none"},
    }

    assert _MODULE._delivery_site(task) == ""
    assert _MODULE._effective_site(task) == "shopping"


def test_main_second_run_is_idempotent_for_already_migrated_dataset(tmp_path, monkeypatch):
    dataset = [
        {
            "id": "gitlab-1",
            "site": "gitlab",
            "adversarial_data_seed": {
                "mechanism": "api",
                "api_calls": [
                    {
                        "target": {
                            "benchmark": "webarena_verified",
                            "site": "gitlab",
                            "resource_type": "user_status",
                            "update": {},
                        },
                        "body": {"message": "payload", "availability": "busy"},
                    }
                ],
            },
        }
    ]
    dataset_path = tmp_path / "adversarial_tasks.json"
    original = json.dumps(dataset)
    dataset_path.write_text(original, encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "migrate_phase_2_seeds_to_targets.py",
            "--path",
            str(dataset_path),
            "--quarantine-sites",
            "map",
        ],
    )

    assert _MODULE.main() == 0
    assert json.loads(dataset_path.read_text(encoding="utf-8")) == dataset


def test_main_refuses_partial_migration_when_supported_site_has_unmatched_call(tmp_path, monkeypatch):
    dataset = [
        {
            "id": "shopping-unknown",
            "site": "shopping",
            "adversarial_data_seed": {
                "mechanism": "api",
                "api_calls": [
                    {"method": "POST", "path": "/unknown/route", "body": {"detail": "payload"}}
                ],
            },
        }
    ]
    dataset_path = tmp_path / "adversarial_tasks.json"
    original = json.dumps(dataset)
    dataset_path.write_text(original, encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        ["migrate_phase_2_seeds_to_targets.py", "--path", str(dataset_path)],
    )

    assert _MODULE.main() == 1
    assert dataset_path.read_text(encoding="utf-8") == original


def test_migrate_task_sanitizes_agent_context_credentials():
    task = {
        "id": "gitlab-1",
        "site": "gitlab",
        "agent_context": {
            "authentication": {
                "credentials": {"username": "alice", "password": "super-secret"},
            },
            "auth_mechanism": {
                "headers": {"Authorization": "Bearer top-secret"},
            },
            "description": "Use Authorization: Bearer top-secret for authenticated writes.",
            "notes": "Credentials (alice/super-secret) are only for local bootstrap.",
        },
        "adversarial_data_seed": {"mechanism": "api", "api_calls": []},
    }

    updated, changed, skipped, unmatched = _MODULE._migrate_task(task)

    assert updated["agent_context"]["authentication"]["credentials"] == {
        "username": "<redacted>",
        "password": "<redacted>",
    }
    assert updated["agent_context"]["auth_mechanism"]["headers"] == {
        "Authorization": "<redacted>"
    }
    assert "top-secret" not in updated["agent_context"]["description"]
    assert "<redacted>" in updated["agent_context"]["description"]
    assert "super-secret" not in updated["agent_context"]["notes"]
    assert updated["agent_context"]["notes"].startswith("Credentials (<redacted>)")
    assert changed == 1
    assert skipped == 0
    assert unmatched == 0


def test_migrate_task_redacts_email_password_examples_in_prose():
    task = {
        "id": "shopping-1",
        "site": "shopping",
        "agent_context": {
            "auth_mechanism": {
                "headers": {"X-M2-Customer-Auto-Login": "emma.lopez@gmail.com:Password.123"},
            },
            "description": (
                "Set the header value to 'email:password' format "
                "(e.g., 'emma.lopez@gmail.com:Password.123') on the browser context."
            ),
        },
        "adversarial_data_seed": {"mechanism": "api", "api_calls": []},
    }

    updated, changed, skipped, unmatched = _MODULE._migrate_task(task)

    assert "emma.lopez@gmail.com:Password.123" not in updated["agent_context"]["description"]
    assert "email:password" in updated["agent_context"]["description"]
    assert "<redacted>" in updated["agent_context"]["description"]
    assert changed == 1
    assert skipped == 0
    assert unmatched == 0


def test_migrate_task_preserves_executable_seed_payloads_verbatim():
    task = {
        "id": "shopping-1",
        "site": "shopping",
        "data_seed": {
            "mechanism": "api",
            "api_calls": [
                {
                    "method": "POST",
                    "path": "/rest/V1/reviews",
                    "headers": {"Authorization": "Bearer top-secret"},
                    "body": {"token": "seed-token", "detail": "keep-me"},
                }
            ],
        },
        "adversarial_data_seed": {"mechanism": "api", "api_calls": []},
    }

    updated, changed, skipped, unmatched = _MODULE._migrate_task(task)

    copied = updated["data_seed"]["api_calls"][0]
    assert copied["headers"]["Authorization"] == "Bearer top-secret"
    assert copied["body"]["token"] == "seed-token"
    assert copied["body"]["detail"] == "keep-me"
    assert changed == 0
    assert skipped == 0
    assert unmatched == 0
