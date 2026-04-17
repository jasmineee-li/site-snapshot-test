from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "migrate_phase_2_seeds_to_targets.py"
)
_SPEC = importlib.util.spec_from_file_location("migrate_phase_2_seeds_to_targets", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_migrate_seed_rewrites_targeted_shopping_review_to_editor_call():
    seed = {
        "mechanism": "api",
        "api_calls": [
            {
                "target": {
                    "benchmark": "webarena_verified",
                    "site": "shopping",
                    "resource_type": "product_review",
                    "create": {
                        "product_review": {
                            "title": "Review title",
                            "nickname": "reviewer",
                            "entity_pk_value": 7,
                            "ratings": [{"rating_name": "Quality", "value": 5}],
                        }
                    },
                },
                "body": {"detail": "{{PAYLOAD_TEXT}}"},
            }
        ],
    }

    migrated = _MODULE.migrate_seed(seed)

    assert migrated == {
        "mechanism": "editor",
        "editor_calls": [
            {
                "benchmark": "webarena_verified",
                "site": "shopping",
                "method": "create_product_review",
                "args": {
                    "title": "Review title",
                    "detail": "{{PAYLOAD_TEXT}}",
                    "nickname": "reviewer",
                    "entity_pk_value": 7,
                    "rating": 5,
                },
            }
        ],
    }


def test_migrate_task_rewrites_seed_template_and_adversarial_seed():
    task = {
        "id": "gitlab-1",
        "site": "gitlab",
        "seed_template": {
            "mechanism": "api",
            "api_calls": [
                {
                    "target": {
                        "benchmark": "webarena_verified",
                        "site": "gitlab",
                        "resource_type": "user_status",
                        "update": {},
                    },
                    "body": {"message": "{{PAYLOAD_TEXT}}", "emoji": "speech_balloon"},
                }
            ],
        },
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
                    "body": {"message": "payload", "emoji": "speech_balloon"},
                }
            ],
        },
    }

    migrated = _MODULE.migrate_task(task)

    assert migrated["seed_template"] == {
        "mechanism": "editor",
        "editor_calls": [
            {
                "benchmark": "webarena_verified",
                "site": "gitlab",
                "method": "update_user_status",
                "args": {"message": "{{PAYLOAD_TEXT}}", "emoji": "speech_balloon"},
            }
        ],
    }
    assert migrated["adversarial_data_seed"] == {
        "mechanism": "editor",
        "editor_calls": [
            {
                "benchmark": "webarena_verified",
                "site": "gitlab",
                "method": "update_user_status",
                "args": {"message": "payload", "emoji": "speech_balloon"},
            }
        ],
    }


def test_migrate_task_rewrites_reddit_postcondition_body_field_to_editor_alias():
    task = {
        "id": "reddit-1",
        "site": "reddit",
        "delivery_channel": {
            "mechanism": "form",
            "body_field": "reply_to_submission_{submission_id}[comment]",
            "postcondition": {
                "type": "db_row_value",
                "table": "comment",
                "value_column": "body",
                "where": {
                    "body": {"body_field": "reply_to_submission_{submission_id}[comment]"}
                },
            },
        },
        "seed_template": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_comment",
                    "args": {
                        "forum_name": "books",
                        "submission_id": "{submission_id}",
                        "body": "{{PAYLOAD_TEXT}}",
                    },
                }
            ],
        },
        "adversarial_data_seed": {
            "mechanism": "editor",
            "editor_calls": [
                {
                    "benchmark": "webarena_verified",
                    "site": "reddit",
                    "method": "create_comment",
                    "args": {
                        "forum_name": "books",
                        "submission_id": "42",
                        "body": "payload",
                    },
                }
            ],
        },
    }

    migrated = _MODULE.migrate_task(task)

    assert migrated["delivery_channel"]["body_field"] == "body"
    assert migrated["delivery_channel"]["postcondition"]["where"]["body"] == {
        "body_field": "body"
    }


def test_migrate_seed_rejects_mixed_target_and_legacy_api_calls():
    seed = {
        "mechanism": "api",
        "api_calls": [
            {
                "target": {
                    "benchmark": "webarena_verified",
                    "site": "shopping",
                    "resource_type": "customer_profile",
                    "update": {},
                },
                "body": {"firstname": "Alice"},
            },
            {
                "method": "POST",
                "path": "/legacy/bootstrap",
                "body": {"detail": "keep-me"},
            },
        ],
    }

    with pytest.raises(ValueError, match="mixed legacy api_calls and target-based calls"):
        _MODULE.migrate_seed(seed)


def test_migrate_seed_appends_leftover_target_calls_to_existing_editor_calls():
    seed = {
        "mechanism": "editor",
        "editor_calls": [
            {
                "benchmark": "webarena_verified",
                "site": "reddit",
                "method": "create_forum",
                "args": {"name_template": "books", "description_template": "desc"},
            }
        ],
        "api_calls": [
            {
                "target": {
                    "benchmark": "webarena_verified",
                    "site": "shopping",
                    "resource_type": "customer_profile",
                    "update": {},
                },
                "body": {"firstname": "Alice"},
            }
        ],
    }

    migrated = _MODULE.migrate_seed(seed)

    assert migrated == {
        "mechanism": "editor",
        "editor_calls": [
            {
                "benchmark": "webarena_verified",
                "site": "reddit",
                "method": "create_forum",
                "args": {"name_template": "books", "description_template": "desc"},
            },
            {
                "benchmark": "webarena_verified",
                "site": "shopping",
                "method": "update_customer_profile",
                "args": {"field": "firstname", "value": "Alice"},
            },
        ],
    }


def test_migrate_seed_preserves_non_default_benchmark():
    seed = {
        "mechanism": "api",
        "api_calls": [
            {
                "target": {
                    "benchmark": "custom_benchmark",
                    "site": "gitlab",
                    "resource_type": "user_status",
                    "update": {},
                },
                "body": {"message": "payload"},
            }
        ],
    }

    migrated = _MODULE.migrate_seed(seed)

    assert migrated["editor_calls"] == [
        {
            "benchmark": "custom_benchmark",
            "site": "gitlab",
            "method": "update_user_status",
            "args": {"message": "payload", "emoji": None},
        }
    ]


def test_migrate_seed_maps_gitlab_profile_name_updates():
    seed = {
        "mechanism": "api",
        "api_calls": [
            {
                "target": {
                    "benchmark": "webarena_verified",
                    "site": "gitlab",
                    "resource_type": "user_profile",
                    "update": {},
                },
                "body_form": {"user[name]": "{{PAYLOAD_TEXT}}"},
            }
        ],
    }

    migrated = _MODULE.migrate_seed(seed)

    assert migrated["editor_calls"] == [
        {
            "benchmark": "webarena_verified",
            "site": "gitlab",
            "method": "update_user_profile",
            "args": {"name": "{{PAYLOAD_TEXT}}"},
        }
    ]


def test_migrate_seed_maps_gitlab_profile_name_and_bio_updates():
    seed = {
        "mechanism": "api",
        "api_calls": [
            {
                "target": {
                    "benchmark": "webarena_verified",
                    "site": "gitlab",
                    "resource_type": "user_profile",
                    "update": {},
                },
                "body_form": {
                    "_method": "put",
                    "user[name]": "Display Name",
                    "user[bio]": "{{PAYLOAD_TEXT}}",
                },
            }
        ],
    }

    migrated = _MODULE.migrate_seed(seed)

    assert migrated["editor_calls"] == [
        {
            "benchmark": "webarena_verified",
            "site": "gitlab",
            "method": "update_user_profile",
            "args": {"bio": "{{PAYLOAD_TEXT}}", "name": "Display Name"},
        }
    ]


def test_main_redacts_embedded_secret_strings(tmp_path, monkeypatch):
    dataset = [
        {
            "id": "gitlab-1",
            "site": "gitlab",
            "instruction": "Set to 'secret-token-value'",
            "adversarial_data_seed": {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "gitlab",
                        "method": "update_user_status",
                        "args": {"message": "Bearer abc123"},
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
        ["migrate_phase_2_seeds_to_targets.py", "--path", str(dataset_path)],
    )

    assert _MODULE.main() == 0

    migrated = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert migrated[0]["instruction"] == "Set to '<redacted>'"
    assert (
        migrated[0]["adversarial_data_seed"]["editor_calls"][0]["args"]["message"]
        == "Bearer <redacted>"
    )


def test_main_redacts_structured_agent_context_secrets(tmp_path, monkeypatch):
    dataset = [
        {
            "id": "shopping-1",
            "site": "shopping",
            "agent_context": {
                "authentication": {
                    "credentials": {"username": "emma@example.com", "password": "Password.123"}
                },
                "auth_mechanism": {
                    "headers": {
                        "X-M2-Customer-Auto-Login": "emma@example.com:Password.123",
                    }
                },
                "description": "Use emma@example.com:Password.123 in the auto-login header.",
            },
            "adversarial_data_seed": {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "shopping",
                        "method": "create_product_review",
                        "args": {
                            "entity_pk_value": 1,
                            "title": "payload",
                            "detail": "payload",
                            "nickname": "reviewer",
                        },
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
        ["migrate_phase_2_seeds_to_targets.py", "--path", str(dataset_path)],
    )

    assert _MODULE.main() == 0

    migrated = json.loads(dataset_path.read_text(encoding="utf-8"))
    assert migrated[0]["agent_context"]["authentication"]["credentials"] == {
        "username": "<redacted>",
        "password": "<redacted>",
    }
    assert migrated[0]["agent_context"]["auth_mechanism"]["headers"] == {
        "X-M2-Customer-Auto-Login": "<redacted>"
    }
    assert "emma@example.com:Password.123" not in migrated[0]["agent_context"]["description"]
    assert "<redacted>" in migrated[0]["agent_context"]["description"]


def test_main_quarantines_effective_map_delivery_site(tmp_path, monkeypatch):
    dataset = [
        {
            "id": "shopping-admin-map",
            "site": "shopping_admin",
            "delivery_channel": {"delivery_site": "map"},
            "adversarial_data_seed": {
                "mechanism": "api",
                "api_calls": [
                    {
                        "target": {
                            "benchmark": "webarena_verified",
                            "site": "map",
                            "resource_type": "way",
                            "update": {},
                        },
                        "body": {"name": "payload"},
                    }
                ],
            },
        },
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
                            "resource_type": "user_profile",
                            "update": {},
                        },
                        "body": {"bio": "payload"},
                    }
                ],
            },
        },
    ]
    dataset_path = tmp_path / "adversarial_tasks.json"
    quarantine_path = tmp_path / "adversarial_tasks.map_quarantine.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "migrate_phase_2_seeds_to_targets.py",
            "--path",
            str(dataset_path),
            "--quarantine-path",
            str(quarantine_path),
        ],
    )

    assert _MODULE.main() == 0

    migrated = json.loads(dataset_path.read_text(encoding="utf-8"))
    quarantined = json.loads(quarantine_path.read_text(encoding="utf-8"))
    assert [task["id"] for task in migrated] == ["gitlab-1"]
    assert [task["id"] for task in quarantined] == ["shopping-admin-map"]


def test_main_is_idempotent_for_already_migrated_dataset(tmp_path, monkeypatch):
    dataset = [
        {
            "id": "reddit-1",
            "site": "reddit",
            "adversarial_data_seed": {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "reddit",
                        "method": "create_comment",
                        "args": {"forum_name": "books", "submission_id": "1", "body": "payload"},
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
        ["migrate_phase_2_seeds_to_targets.py", "--path", str(dataset_path)],
    )

    assert _MODULE.main() == 0
    assert _MODULE.main() == 0

    assert json.loads(dataset_path.read_text(encoding="utf-8")) == dataset


def test_main_preserves_existing_quarantine_when_no_map_tasks_remain(tmp_path, monkeypatch):
    dataset = [
        {
            "id": "gitlab-1",
            "site": "gitlab",
            "adversarial_data_seed": {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "gitlab",
                        "method": "update_user_status",
                        "args": {"message": "payload"},
                    }
                ],
            },
        }
    ]
    dataset_path = tmp_path / "adversarial_tasks.json"
    quarantine_path = tmp_path / "adversarial_tasks.map_quarantine.json"
    dataset_path.write_text(json.dumps(dataset), encoding="utf-8")
    prior_quarantine = [{"id": "stale"}]
    quarantine_path.write_text(json.dumps(prior_quarantine), encoding="utf-8")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "migrate_phase_2_seeds_to_targets.py",
            "--path",
            str(dataset_path),
            "--quarantine-path",
            str(quarantine_path),
        ],
    )

    assert _MODULE.main() == 0
    assert quarantine_path.exists()
    assert json.loads(quarantine_path.read_text(encoding="utf-8")) == prior_quarantine
