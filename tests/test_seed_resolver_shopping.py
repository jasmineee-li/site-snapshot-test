from __future__ import annotations

import pytest

from worldsim.seed_resolvers import shopping
from worldsim.seed_resolvers.types import ResolverError


def test_create_review_normalizes_payload():
    resolved = shopping.create(
        {
            "method": "POST",
            "resource_type": "product_review",
            "create": {
                "product_review": {
                    "title": "Fallback title",
                    "nickname": "fallback-user",
                    "entity_pk_value": 67,
                }
            },
            "body": {
                "detail": "Payload text",
            },
        },
        {"site_url": "http://shopping.test"},
        {},
    )

    assert resolved.url == "http://shopping.test/rest/V1/reviews"
    assert resolved.body == {
        "review": {
            "title": "Fallback title",
            "detail": "Payload text",
            "nickname": "fallback-user",
            "entity_pk_value": 67,
            "review_entity": "product",
            "stores": [1],
            "ratings": [{"rating_name": "Quality", "value": 4}],
        }
    }


def test_update_rejects_unknown_resource_type():
    with pytest.raises(ResolverError, match="unsupported_resource_type"):
        shopping.update({}, {"site_url": "http://shopping.test"}, {})


def test_update_customer_profile_routes_to_profile_endpoint():
    resolved = shopping.update(
        {"resource_type": "customer_profile"},
        {"site_url": "http://shopping.test"},
        {},
    )

    assert resolved.method == "PUT"
    assert resolved.url == "http://shopping.test/rest/V1/customers/me"


def test_create_review_requires_complete_review_shape():
    with pytest.raises(ResolverError, match="missing_required_fields"):
        shopping.create(
            {
                "method": "POST",
                "resource_type": "product_review",
                "create": {"product_review": {"entity_pk_value": 67}},
                "body": {"detail": "Payload text"},
            },
            {"site_url": "http://shopping.test"},
            {},
        )


def test_create_review_rejects_conflicting_nested_and_top_level_body_shapes():
    with pytest.raises(ResolverError, match="either top-level body fields or body.review"):
        shopping.create(
            {
                "method": "POST",
                "resource_type": "product_review",
                "create": {
                    "product_review": {
                        "title": "Fallback title",
                        "nickname": "fallback-user",
                        "entity_pk_value": 67,
                    }
                },
                "body": {
                    "detail": "outer payload",
                    "review": {"detail": "inner payload"},
                },
            },
            {"site_url": "http://shopping.test"},
            {},
        )


def test_create_review_noops_when_same_review_already_exists(monkeypatch):
    monkeypatch.setattr(shopping, "_existing_review_seeded", lambda instance, payload: True)

    resolved = shopping.create(
        {
            "method": "POST",
            "resource_type": "product_review",
            "create": {
                "product_review": {
                    "title": "Fallback title",
                    "nickname": "fallback-user",
                    "entity_pk_value": 67,
                }
            },
            "body": {
                "detail": "Payload text",
            },
        },
        {"site_url": "http://shopping.test"},
        {},
    )

    assert resolved.method == "GET"
    assert resolved.url == "http://shopping.test/"
    assert resolved.context_additions["review_already_seeded"] is True


def test_create_review_ignores_invalid_db_connection_lookup():
    resolved = shopping.create(
        {
            "method": "POST",
            "resource_type": "product_review",
            "create": {
                "product_review": {
                    "title": "Fallback title",
                    "nickname": "fallback-user",
                    "entity_pk_value": 67,
                }
            },
            "body": {
                "detail": "Payload text",
            },
        },
        {"site_url": "http://shopping.test", "db_connection": "sqlite:///tmp/demo.db"},
        {},
    )

    assert resolved.method == "POST"
    assert resolved.url == "http://shopping.test/rest/V1/reviews"
