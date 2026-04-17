from __future__ import annotations

from worldsim.seed_resolvers import shopping_admin


def test_create_review_uses_admin_site_origin():
    resolved = shopping_admin.create(
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
            "body": {"detail": "Payload text"},
        },
        {"site_url": "http://shopping-admin.test"},
        {},
    )

    assert resolved.url == "http://shopping-admin.test/rest/V1/reviews"


def test_update_admin_profile_routes_to_admin_profile_endpoint():
    resolved = shopping_admin.update(
        {"resource_type": "admin_profile"},
        {"site_url": "http://shopping-admin.test"},
        {},
    )

    assert resolved.method == "PUT"
    assert resolved.url == "http://shopping-admin.test/rest/V1/users/me"
