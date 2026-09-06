# ruff: noqa: F403, F405
"""Sandbox validator finalized HTTP seed contract checks (``_validate_finalized_http_seed_contract``).

Split from tests/test_sandbox_validator.py; shared helpers live in
tests/sandbox_validator/_fixtures.py.
"""

from __future__ import annotations

from ._fixtures import *


class TestFinalizedHttpSeedContract:
    def test_extract_attack_write_deduplicates_editor_alias_placeholder_fields(self):
        attack_write = validator._extract_attack_write(
            {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "gitlab",
                        "method": "create_issue",
                        "args": {
                            "project_name_template": "webagent-task-{task_id}",
                            "title_template": "Read this carefully",
                            "body_template": "{{PAYLOAD_TEXT}}",
                        },
                    }
                ],
            }
        )

        assert attack_write is not None
        assert attack_write["placeholder_count"] == 1
        assert attack_write["placeholder_fields"] == {"body", "description"}

    def test_extract_attack_write_maps_gitlab_issue_description_body_alias(self):
        attack_write = validator._extract_attack_write(
            {
                "mechanism": "editor",
                "editor_calls": [
                    {
                        "benchmark": "webarena_verified",
                        "site": "gitlab",
                        "method": "create_issue_description",
                        "args": {
                            "project_path_template": "{benign_project_path}",
                            "body": "{{PAYLOAD_TEXT}}",
                        },
                    }
                ],
            }
        )

        assert attack_write is not None
        assert attack_write["placeholder_count"] == 1
        assert attack_write["placeholder_fields"] == {"description"}

    def test_explicit_non_webarena_editor_call_does_not_use_legacy_contract(self):
        call = {
            "benchmark": "stwebagentbench",
            "site": "gitlab",
            "method": "create_issue",
            "args": {"body_template": "{{PAYLOAD_TEXT}}"},
        }

        assert validator._editor_delivery_binding(call) is None
        assert validator._editor_arg_alias_pairs(call) == []
        assert validator._call_delivery_mechanism({"mechanism": "editor"}, call) is None

    def test_rejects_unresolved_http_path_placeholder(self):
        error = validator._validate_finalized_http_seed_contract(
            {
                "mechanism": "api",
                "api_calls": [
                    {
                        "method": "POST",
                        "path": "/{namespace}/{project}/-/merge_requests",
                        "body_form": {"merge_request[description]": "payload"},
                    }
                ],
            },
            {
                "mechanism": "form",
                "body_field": "merge_request[description]",
                "postcondition": _db_row_value(
                    table="merge_requests",
                    value_column="description",
                    where={"target_project_id": {"literal": 123}},
                ),
            },
            sites=["gitlab"],
        )

        assert error == "adversarial_data_seed api_calls[0].path contains unresolved placeholders"

    def test_accepts_cross_site_delivery_outside_task_sites(self):
        error = validator._validate_finalized_http_seed_contract(
            {
                "mechanism": "form",
                "api_calls": [
                    {
                        "method": "POST",
                        "path": "/review/product/post/id/123/",
                        "body_form": {"detail": "payload"},
                    }
                ],
            },
            {
                "mechanism": "form",
                "body_field": "detail",
                "delivery_site": "shopping",
                "postcondition": _db_row_value(
                    table="review_detail",
                    value_column="detail",
                    where={"product_id": {"literal": 123}},
                ),
            },
            sites=["shopping_admin"],
        )

        assert error is None

    def test_accepts_reddit_legacy_dynamic_comment_field_for_editor_calls(self):
        error = validator._validate_finalized_http_seed_contract(
            {
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
            {
                "mechanism": "form",
                "body_field": "reply_to_submission_{submission_id}[comment]",
                "postcondition": _db_row_value(
                    table="comment",
                    value_column="body",
                    where={"body": {"body_field": "reply_to_submission_{submission_id}[comment]"}},
                ),
            },
            sites=["reddit"],
        )

        assert error is None
