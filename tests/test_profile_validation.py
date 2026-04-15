from __future__ import annotations

import pytest

from worldsim.profile_validation import load_and_validate_profile, validate_profile


def _db_row_value(*, table: str, value_column: str, where: dict[str, object]) -> dict:
    return {
        "type": "db_row_value",
        "table": table,
        "value_column": value_column,
        "where": where,
    }


def test_load_and_validate_profile_rejects_invalid_json(tmp_path):
    profile_path = tmp_path / "BENCHMARK_PROFILE_shopping.json"
    profile_path.write_text("{not-json")

    with pytest.raises(ValueError, match="invalid profile JSON"):
        load_and_validate_profile("shopping", profile_path)


def test_validate_profile_rejects_eval_types_absent_from_manifest():
    profile = {
        "data_model": [],
        "injection_surface": [],
        "verification_capabilities": [{"eval_type": "db_query_match"}],
    }

    with pytest.raises(
        ValueError,
        match="verification capabilities reference eval types absent from manifest",
    ):
        validate_profile("shopping", profile, manifest_eval_types=["string_match"])


def test_validate_profile_rejects_field_from_wrong_entity():
    profile = {
        "site_name": "shopping",
        "data_model": [
            {"entity": "Product", "fields": [{"name": "title"}]},
            {"entity": "Review", "fields": [{"name": "body"}]},
        ],
        "injection_surface": [
            {
                "id": "surface-1",
                "source_field": "Product.body",
            }
        ],
        "verification_capabilities": [],
    }

    with pytest.raises(ValueError, match="unknown field 'Product.body'"):
        validate_profile("shopping", profile)


def test_validate_profile_rejects_invalid_injection_surface_labels():
    profile = {
        "site_name": "shopping",
        "data_model": [
            {"entity": "Review", "fields": [{"name": "body"}]},
        ],
        "injection_surface": [
            {
                "id": "surface-1",
                "source_field": "Review.body",
                "rendering_format": "markdown",
                "controllable_by_tier": "authed_user",
                "controllability_justification": "Users can post reviews via /review/product/post.",
                "delivery_channels": [
                    {
                        "mechanism": "form",
                        "privileged_seed": False,
                        "path_template": "/review/product/post/id/{product_id}/",
                        "method": "POST",
                        "body_field": "detail",
                        "table": None,
                        "column": None,
                        "postcondition": _db_row_value(
                            table="review_detail",
                            value_column="detail",
                            where={"product_id": {"path_param": "product_id"}},
                        ),
                    },
                    {
                        "mechanism": "bogus",
                        "privileged_seed": False,
                        "path_template": "/bogus",
                        "method": "POST",
                        "body_field": "detail",
                        "table": None,
                        "column": None,
                        "postcondition": _db_row_value(
                            table="bogus",
                            value_column="detail",
                            where={"product_id": {"literal": 1}},
                        ),
                    },
                ],
                "compatible_concealments": ["offscreen_css"],
            }
        ],
        "verification_capabilities": [],
    }

    with pytest.raises(ValueError, match="delivery_channels\\[1\\] mechanism must be one of"):
        validate_profile("shopping", profile)


def test_validate_profile_rejects_tier_that_conflicts_with_no_auth_context():
    profile = {
        "site_name": "wikipedia",
        "data_model": [
            {"entity": "WikiPage", "fields": [{"name": "body"}]},
        ],
        "injection_surface": [
            {
                "id": "surface-1",
                "source_field": "WikiPage.body",
                "rendering_format": "plaintext",
                "controllable_by_tier": "authed_user",
                "controllability_justification": "Only logged-in users can edit this page.",
                "delivery_channels": [
                    {
                        "mechanism": "api",
                        "privileged_seed": False,
                        "path_template": "/api/pages/{id}",
                        "method": "PATCH",
                        "body_field": "body",
                        "table": None,
                        "column": None,
                        "postcondition": _db_row_value(
                            table="wiki_pages",
                            value_column="body",
                            where={"id": {"path_param": "id"}},
                        ),
                    }
                ],
                "compatible_concealments": ["plaintext"],
            }
        ],
        "verification_capabilities": [],
        "agent_context": {
            "auth_mechanism": {
                "type": "none",
                "notes": "Site is public.",
            }
        },
    }

    with pytest.raises(
        ValueError,
        match="controllable_by_tier='authed_user' requires agent_context auth_mechanism.type != 'none'",
    ):
        validate_profile("wikipedia", profile)


def test_validate_profile_rejects_authed_user_surface_without_agent_context():
    profile = {
        "site_name": "shopping",
        "data_model": [
            {"entity": "Review", "fields": [{"name": "body"}]},
        ],
        "injection_surface": [
            {
                "id": "surface-1",
                "source_field": "Review.body",
                "rendering_format": "plaintext",
                "controllable_by_tier": "authed_user",
                "controllability_justification": "Only logged-in users can post here.",
                "delivery_channels": [
                    {
                        "mechanism": "form",
                        "privileged_seed": False,
                        "path_template": "/review/product/post/id/{product_id}/",
                        "method": "POST",
                        "body_field": "detail",
                        "table": None,
                        "column": None,
                        "postcondition": _db_row_value(
                            table="review_detail",
                            value_column="detail",
                            where={"product_id": {"path_param": "product_id"}},
                        ),
                    }
                ],
                "compatible_concealments": ["plaintext"],
            }
        ],
        "verification_capabilities": [],
    }

    with pytest.raises(
        ValueError,
        match="controllable_by_tier='authed_user' require agent_context",
    ):
        validate_profile("shopping", profile)


def test_validate_profile_rejects_http_delivery_channel_without_postcondition():
    profile = {
        "site_name": "shopping",
        "data_model": [{"entity": "Review", "fields": [{"name": "body"}]}],
        "injection_surface": [
            {
                "id": "surface-1",
                "source_field": "Review.body",
                "rendering_format": "plaintext",
                "controllable_by_tier": "any_user",
                "controllability_justification": "Users can post reviews here.",
                "delivery_channels": [
                    {
                        "mechanism": "form",
                        "privileged_seed": False,
                        "path_template": "/review/product/post/id/{product_id}/",
                        "method": "POST",
                        "body_field": "detail",
                        "table": None,
                        "column": None,
                    }
                ],
                "compatible_concealments": ["plaintext"],
            }
        ],
        "verification_capabilities": [],
    }

    with pytest.raises(ValueError, match="postcondition is required for non-sql delivery channels"):
        validate_profile("shopping", profile)


def test_validate_profile_accepts_delivery_site_on_http_channel():
    profile = {
        "site_name": "shopping_admin",
        "data_model": [{"entity": "Review", "fields": [{"name": "body"}]}],
        "injection_surface": [
            {
                "id": "surface-1",
                "source_field": "Review.body",
                "rendering_format": "plaintext",
                "controllable_by_tier": "any_user",
                "controllability_justification": "Storefront users create content shown in admin.",
                "delivery_channels": [
                    {
                        "mechanism": "form",
                        "privileged_seed": False,
                        "delivery_site": "shopping",
                        "path_template": "/review/product/post/id/{product_id}/",
                        "method": "POST",
                        "body_field": "detail",
                        "table": None,
                        "column": None,
                        "postcondition": _db_row_value(
                            table="review_detail",
                            value_column="detail",
                            where={"product_id": {"path_param": "product_id"}},
                        ),
                    }
                ],
                "compatible_concealments": ["plaintext"],
            }
        ],
        "verification_capabilities": [],
    }

    validate_profile("shopping_admin", profile)
