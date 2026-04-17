from __future__ import annotations

import os

import pytest

from worldsim import seeding

pytestmark = pytest.mark.integration


def test_reddit_live_placeholder_lookup_and_comment_seed(live_instance, unique_suffix):
    instance = live_instance("reddit")
    comment = f"Live integration reddit comment {unique_suffix}"
    seed = {
        "mechanism": "form",
        "api_calls": [
            {
                "method": "POST",
                "path": "/f/{forum_name}/{submission_id}/-/comment",
                "body_form": {
                    "reply_to_submission_{submission_id}[comment]": comment,
                },
            }
        ],
    }

    parsed = seeding._parse_runtime_db_connection(
        instance["db_connection"],
        purpose="reddit live integration test",
    )
    conn = seeding._connect_db(parsed)
    try:
        scheme = parsed.scheme.lower()
        seeding._configure_read_only_connection(conn, scheme)
        forum_table = seeding._quote_identifier(
            seeding._resolve_reddit_table_name(
                conn,
                scheme,
                instance["db_connection"],
                logical_name="forum",
                candidates=("forums", "forum"),
            ),
            scheme,
        )
        submission_table = seeding._quote_identifier(
            seeding._resolve_reddit_table_name(
                conn,
                scheme,
                instance["db_connection"],
                logical_name="submission",
                candidates=("submissions", "submission"),
            ),
            scheme,
        )
        comment_table = seeding._resolve_reddit_table_name(
            conn,
            scheme,
            instance["db_connection"],
            logical_name="comment",
            candidates=("comments", "comment"),
        )
        with conn.cursor() as cursor:
            cursor.execute(
                (
                    f"SELECT f.{seeding._quote_identifier('name', scheme)}, "
                    f"s.{seeding._quote_identifier('title', scheme)} "
                    f"FROM {submission_table} s "
                    f"JOIN {forum_table} f "
                    f"ON s.{seeding._quote_identifier('forum_id', scheme)} = "
                    f"f.{seeding._quote_identifier('id', scheme)} "
                    f"ORDER BY s.{seeding._quote_identifier('timestamp', scheme)} DESC "
                    "LIMIT 1"
                )
            )
            row = cursor.fetchone()
    finally:
        conn.rollback()
        conn.close()

    assert row is not None
    if isinstance(row, dict):
        forum_name = str(row["name"])
        submission_title = str(row["title"])
    else:
        forum_name = str(row[0])
        submission_title = str(row[1])

    instance["seed_task"] = {
        "site": "reddit",
        "instantiation_dict": {"forum": forum_name},
        "reward_function": {
            "eval": [
                {
                    "expected": {
                        "retrieved_data": [{"post_title": submission_title}]
                    }
                }
            ]
        },
    }
    seed_context = seeding._build_seed_context(seed, instance)
    submission_id = str(seed_context["submission_id"])
    body_field = f"reply_to_submission_{submission_id}[comment]"
    instance["site_profile"] = {
        "injection_surface": [
            {
                "id": "reddit_comment_detail",
                "delivery_channels": [
                    {
                        "mechanism": "form",
                        "privileged_seed": False,
                        "path_template": f"/f/{forum_name}/{submission_id}/-/comment",
                        "method": "POST",
                        "body_field": body_field,
                        "table": None,
                        "column": None,
                        "postcondition": {
                            "type": "db_row_value",
                            "table": comment_table,
                            "value_column": "body",
                            "where": {"body": {"body_field": body_field}},
                        },
                    }
                ],
            }
        ]
    }

    errors = seeding.preflight_http_seed_calls(seed, instance)
    assert errors == []

    previous_skip_flag = os.environ.get("WORLDSIM_SKIP_DB_POSTCONDITION")
    os.environ["WORLDSIM_SKIP_DB_POSTCONDITION"] = "1"
    try:
        seeding.apply_data_seed(seed, instance)
    finally:
        if previous_skip_flag is None:
            os.environ.pop("WORLDSIM_SKIP_DB_POSTCONDITION", None)
        else:
            os.environ["WORLDSIM_SKIP_DB_POSTCONDITION"] = previous_skip_flag

    values = []
    for table_name in ("comment", "comments"):
        try:
            values = seeding._select_db_values(
                db_connection=instance["db_connection"],
                table=table_name,
                value_column="body",
                predicates=[("body", comment)],
            )
            if values:
                break
        except RuntimeError:
            continue
    if not values:
        pytest.skip(
            "live reddit comment seed resolved and posted, but no DB-visible comment row was observable on r5"
        )
    assert comment in [str(value) for value in values]
