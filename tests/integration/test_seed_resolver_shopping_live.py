from __future__ import annotations

import pytest

from worldsim import seeding
from worldsim.auth_tokens import acquire_tokens_for_instances

pytestmark = pytest.mark.integration


def test_shopping_live_review_create_and_db_verify(live_instance, unique_suffix):
    instance = live_instance("shopping")
    errors = acquire_tokens_for_instances([instance])
    assert errors == []

    detail = f"Live integration review detail {unique_suffix}"
    title = f"Live integration review {unique_suffix}"
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
                            "title": title,
                            "nickname": "LiveIntegrationUser",
                            "entity_pk_value": 67,
                        }
                    },
                },
                "body": {
                    "title": title,
                    "detail": detail,
                    "nickname": "LiveIntegrationUser",
                    "entity_pk_value": 67,
                    "stores": [1],
                },
            }
        ],
    }
    instance["site_profile"] = {
        "injection_surface": [
            {
                "id": "review_detail_pdp",
                "delivery_channels": [
                    {
                        "mechanism": "api",
                        "privileged_seed": False,
                        "path_template": "/rest/V1/reviews",
                        "method": "POST",
                        "body_field": "detail",
                        "table": None,
                        "column": None,
                        "postcondition": {
                            "type": "db_row_value",
                            "table": "review_detail",
                            "value_column": "detail",
                            "where": {"detail": {"body_field": "detail"}},
                        },
                    }
                ],
            }
        ]
    }

    seeding.apply_data_seed(seed, instance)
    values = seeding._select_db_values(
        db_connection=instance["db_connection"],
        table="review_detail",
        value_column="detail",
        predicates=[("detail", detail)],
    )
    assert detail in [str(value) for value in values]
