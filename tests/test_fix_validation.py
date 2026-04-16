"""Unit tests for ``worldsim.fix_validation.validate_fix_patch``.

Covers the allowlist harvesting from ``verification_capabilities``, origin
and method checks, form/api/state_push handling, and the task-533 regression.
"""

from __future__ import annotations

import pytest

from worldsim.fix_validation import validate_fix_patch


def _gitlab_profile() -> dict:
    """Minimal GitLab profile whose `verification_capabilities` example URLs
    anchor the path allowlist. Matches the shape of the real profile at
    `logs/phase_0c/BENCHMARK_PROFILE_gitlab.json`."""
    return {
        "site_name": "gitlab",
        "verification_capabilities": [
            {
                "eval_type": "NetworkEventEvaluator",
                "examples": [
                    {
                        "task_description": "Fork a project via API",
                        "eval_config": {
                            "evaluator": "NetworkEventEvaluator",
                            "expected": {
                                "url": "__GITLAB__/api/v4/projects/79/fork",
                                "http_method": "POST",
                            },
                        },
                    },
                    {
                        "task_description": "Open dashboard todos",
                        "eval_config": {
                            "expected": {
                                "url": [
                                    "__GITLAB__/dashboard/todos",
                                    "__GITLAB__/dashboard/todos?state=pending",
                                ]
                            },
                        },
                    },
                    {
                        "task_description": "Post a note on an MR",
                        "eval_config": {
                            "expected": {
                                "url": "__GITLAB__/primer/design/notes",
                                "http_method": "POST",
                            },
                        },
                    },
                ],
            }
        ],
        "injection_surface": [
            {
                "id": "issue_title",
                "source_field": "Issue.title",
                "seeding_mechanism": "UPDATE issues SET title = '<payload>' WHERE id = <issue_id>",
            }
        ],
    }


def _shopping_profile() -> dict:
    return {
        "site_name": "shopping",
        "verification_capabilities": [
            {
                "eval_type": "NetworkEventEvaluator",
                "examples": [
                    {
                        "eval_config": {
                            "expected": {
                                "url": "__SHOPPING__/api/products/{id}",
                                "http_method": "GET",
                            }
                        }
                    }
                ],
            }
        ],
    }


GITLAB_URL = "http://18.117.99.179:8023"
SHOPPING_URL = "http://shopping.test"


# ---------------------------------------------------------------------------
# Target + shape
# ---------------------------------------------------------------------------


def test_none_target_with_null_patch_is_valid():
    errors = validate_fix_patch({"target": "none", "patch": None}, _gitlab_profile(), GITLAB_URL)
    assert errors == []


def test_task_removal_with_non_null_patch_rejected():
    errors = validate_fix_patch(
        {"target": "task_removal", "patch": {"anything": 1}},
        _gitlab_profile(),
        GITLAB_URL,
    )
    assert errors
    assert "must be null" in errors[0]


def test_unsupported_target_rejected():
    errors = validate_fix_patch({"target": "mystery", "patch": {}}, _gitlab_profile(), GITLAB_URL)
    assert errors
    assert "unsupported" in errors[0]


def test_reward_patch_accepts_known_keys():
    errors = validate_fix_patch(
        {
            "target": "reward_function",
            "patch": {"eval": [{"expected": {"retrieved_data": ["x"]}}]},
        },
        _gitlab_profile(),
        GITLAB_URL,
    )
    assert errors == []


def test_reward_patch_rejects_unknown_keys():
    errors = validate_fix_patch(
        {"target": "reward_function", "patch": {"evil_hook": "rm -rf"}},
        _gitlab_profile(),
        GITLAB_URL,
    )
    assert errors


def test_none_patch_body_is_noop():
    errors = validate_fix_patch(
        {"target": "data_seed", "patch": None}, _gitlab_profile(), GITLAB_URL
    )
    assert errors == []


# ---------------------------------------------------------------------------
# API allowlist
# ---------------------------------------------------------------------------


def test_task_533_regression_post_session_rejected():
    """The task-533 bug: sandbox proposed a POST /api/v4/session, which
    GitLab removed in 9.0 and is not in the allowlist."""
    errors = validate_fix_patch(
        {
            "target": "data_seed",
            "patch": {
                "mechanism": "api",
                "api_calls": [
                    {
                        "method": "POST",
                        "path": "/api/v4/session",
                        "body": {"login": "root", "password": "5iveL!fe"},
                    }
                ],
            },
        },
        _gitlab_profile(),
        GITLAB_URL,
    )
    assert errors
    assert any("allowlist" in e for e in errors)


def test_documented_endpoint_with_numeric_id_accepted():
    errors = validate_fix_patch(
        {
            "target": "data_seed",
            "patch": {
                "mechanism": "api",
                "api_calls": [
                    {
                        "method": "POST",
                        "path": "/api/v4/projects/42/fork",
                        "body": {"namespace_id": 2505},
                    }
                ],
            },
        },
        _gitlab_profile(),
        GITLAB_URL,
    )
    assert errors == []


def test_placeholder_endpoint_accepted_for_any_id():
    """Allowlist regex must normalize `{id}` placeholders to a wildcard."""
    errors = validate_fix_patch(
        {
            "target": "data_seed",
            "patch": {
                "mechanism": "api",
                "api_calls": [
                    {"method": "GET", "path": "/api/products/SKU-123"},
                ],
            },
        },
        _shopping_profile(),
        SHOPPING_URL,
    )
    assert errors == []


def test_wrong_host_rejected():
    errors = validate_fix_patch(
        {
            "target": "data_seed",
            "patch": {
                "mechanism": "api",
                "api_calls": [
                    {"method": "POST", "path": "http://evil.com/api/v4/projects/42/fork"}
                ],
            },
        },
        _gitlab_profile(),
        GITLAB_URL,
    )
    assert errors
    assert any("origin" in e.lower() for e in errors)


def test_wrong_scheme_rejected():
    errors = validate_fix_patch(
        {
            "target": "data_seed",
            "patch": {
                "mechanism": "api",
                "api_calls": [{"method": "GET", "path": "file:///etc/passwd"}],
            },
        },
        _gitlab_profile(),
        GITLAB_URL,
    )
    assert errors


@pytest.mark.parametrize("method", ["DELETE", "HEAD", "OPTIONS", "TRACE", "CONNECT"])
def test_disallowed_methods_rejected(method):
    errors = validate_fix_patch(
        {
            "target": "data_seed",
            "patch": {
                "mechanism": "api",
                "api_calls": [{"method": method, "path": "/api/v4/projects/42/fork"}],
            },
        },
        _gitlab_profile(),
        GITLAB_URL,
    )
    assert errors
    assert any("not allowed" in e for e in errors)


def test_path_not_starting_with_slash_rejected():
    errors = validate_fix_patch(
        {
            "target": "data_seed",
            "patch": {
                "mechanism": "api",
                "api_calls": [{"method": "POST", "path": "api/v4/projects/42/fork"}],
            },
        },
        _gitlab_profile(),
        GITLAB_URL,
    )
    assert errors


def test_empty_profile_fails_closed_on_api():
    """When the profile has no harvestable URLs, api patches must fail closed."""
    errors = validate_fix_patch(
        {
            "target": "data_seed",
            "patch": {
                "mechanism": "api",
                "api_calls": [{"method": "POST", "path": "/anything"}],
            },
        },
        {},
        GITLAB_URL,
    )
    assert errors
    assert any("fail-closed" in e or "allowlist" in e for e in errors)


def test_body_must_be_json_serializable():
    class NotJson:
        pass

    errors = validate_fix_patch(
        {
            "target": "data_seed",
            "patch": {
                "mechanism": "api",
                "api_calls": [
                    {
                        "method": "POST",
                        "path": "/api/v4/projects/79/fork",
                        "body": {"bad": NotJson()},
                    }
                ],
            },
        },
        _gitlab_profile(),
        GITLAB_URL,
    )
    assert errors
    assert any("JSON-serializable" in e for e in errors)


def test_seeding_endpoints_fallback_used_when_present():
    profile = {
        "site_name": "custom",
        "seeding_endpoints": [{"method": "POST", "url": "/custom/seed/{id}", "purpose": "seed"}],
        "verification_capabilities": [],
    }
    errors = validate_fix_patch(
        {
            "target": "data_seed",
            "patch": {
                "mechanism": "api",
                "api_calls": [{"method": "POST", "path": "/custom/seed/abc"}],
            },
        },
        profile,
        "http://custom.test",
    )
    assert errors == []


# ---------------------------------------------------------------------------
# Rejection of removed SQL seeding
# ---------------------------------------------------------------------------


def test_sql_insert_rejected():
    errors = validate_fix_patch(
        {
            "target": "data_seed",
            "patch": {
                "mechanism": "sql",
                "statements": ["INSERT INTO products (id, name) VALUES (1, 'Example product')"],
            },
        },
        _gitlab_profile(),
        GITLAB_URL,
    )
    assert any("unsupported" in error and "sql" in error.lower() for error in errors)


def test_statements_only_patch_rejected():
    errors = validate_fix_patch(
        {
            "target": "data_seed",
            "patch": {
                "statements": ["UPDATE issues SET title = 'x' WHERE id = 5"],
            },
        },
        _gitlab_profile(),
        GITLAB_URL,
    )
    assert any("unsupported" in error and "sql" in error.lower() for error in errors)


# ---------------------------------------------------------------------------
# state_push
# ---------------------------------------------------------------------------


def test_state_push_with_dict_accepted():
    errors = validate_fix_patch(
        {
            "target": "data_seed",
            "patch": {"mechanism": "state_push", "state": {"key": "value"}},
        },
        _gitlab_profile(),
        GITLAB_URL,
    )
    assert errors == []


def test_state_push_with_non_dict_rejected():
    errors = validate_fix_patch(
        {
            "target": "data_seed",
            "patch": {"mechanism": "state_push", "state": "just a string"},
        },
        _gitlab_profile(),
        GITLAB_URL,
    )
    assert errors
    assert any("JSON object" in e for e in errors)


def test_form_patch_with_allowlisted_endpoint_accepted():
    errors = validate_fix_patch(
        {
            "target": "data_seed",
            "patch": {
                "mechanism": "form",
                "api_calls": [
                    {
                        "method": "POST",
                        "path": "/primer/design/notes",
                        "body_form": {"note": "seeded"},
                    }
                ],
            },
        },
        _gitlab_profile(),
        GITLAB_URL,
    )
    assert errors == []


def test_form_patch_rejects_json_body():
    errors = validate_fix_patch(
        {
            "target": "data_seed",
            "patch": {
                "mechanism": "form",
                "api_calls": [
                    {
                        "method": "POST",
                        "path": "/primer/design/notes",
                        "body": {"note": "seeded"},
                    }
                ],
            },
        },
        _gitlab_profile(),
        GITLAB_URL,
    )
    assert any("body_form" in error for error in errors)


# ---------------------------------------------------------------------------
# Mechanism inference on partial merge patches
# ---------------------------------------------------------------------------


def test_merge_patch_without_mechanism_inferred_from_shape():
    """Phase 3 sometimes emits a merge patch tweaking only api_calls."""
    errors = validate_fix_patch(
        {
            "target": "data_seed",
            "patch": {
                "api_calls": [{"method": "POST", "path": "/api/v4/projects/42/fork"}],
            },
        },
        _gitlab_profile(),
        GITLAB_URL,
    )
    assert errors == []


def test_merge_patch_without_mechanism_infers_form_from_body_form():
    errors = validate_fix_patch(
        {
            "target": "data_seed",
            "patch": {
                "api_calls": [
                    {"method": "POST", "path": "/primer/design/notes", "body_form": {"note": "x"}}
                ],
            },
        },
        _gitlab_profile(),
        GITLAB_URL,
    )
    assert errors == []


def test_merge_patch_touching_no_seed_fields_is_noop():
    errors = validate_fix_patch(
        {"target": "data_seed", "patch": {"notes": "meta"}},
        _gitlab_profile(),
        GITLAB_URL,
    )
    assert errors == []
