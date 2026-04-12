"""Data seeding dispatchers.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Phase 3 / Evaluation
Infrastructure" section.

Three seeding mechanisms are supported, matching the v5 spec:

- ``sql``: execute a list of SQL statements against ``instance["db_connection"]``
- ``api``: make a list of HTTP requests against ``instance["site_url"]``
- ``state_push``: PUT a JSON blob to the instance's ``/api/state`` endpoint

Each benchmark's Phase 0a manifest declares which mechanism its sites use.
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Defense-in-depth: block destructive SQL from LLM-generated data seeds.
_BLOCKED_SQL_PATTERNS = re.compile(
    r"\b(DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|DELETE\s+FROM\s+\w+\s*$)\b",
    re.IGNORECASE,
)


def apply_data_seed(seed: dict[str, Any], instance: dict[str, Any]) -> None:
    """Apply a data seed to a running benchmark instance.

    Args:
        seed: Seed spec with a ``mechanism`` field and mechanism-specific
            extras. See the v5 spec for the field schemas.
        instance: Benchmark instance dict with ``site_url`` and
            (for SQL seeding) ``db_connection``.

    Raises:
        ValueError: If ``seed["mechanism"]`` is unknown.
    """
    mechanism = seed["mechanism"]
    if mechanism == "sql":
        for stmt in seed["statements"]:
            execute_sql(stmt, instance["db_connection"])
    elif mechanism == "api":
        for call in seed["api_calls"]:
            resp = requests.request(
                call["method"],
                f"{instance['site_url']}{call['path']}",
                json=call.get("body"),
                timeout=30,
            )
            resp.raise_for_status()
    elif mechanism == "state_push":
        resp = requests.put(
            f"{instance['site_url']}/api/state",
            json=seed["state"],
            timeout=30,
        )
        resp.raise_for_status()
    else:
        raise ValueError(f"unknown data seed mechanism: {mechanism!r}")


def execute_sql(statement: str, db_connection: str) -> None:
    """Execute one SQL statement against a benchmark database.

    Supports MySQL (WebArena OpenCart). Extend this dispatch when new
    benchmarks land with different dialects.

    Args:
        statement: SQL statement to execute.
        db_connection: Connection string, e.g.
            ``mysql://user:pass@host:3306/dbname``.

    Raises:
        ValueError: If the statement matches the destructive SQL blocklist.
    """
    if _BLOCKED_SQL_PATTERNS.search(statement):
        raise ValueError(
            f"SQL statement blocked by safety filter: {statement[:100]}... "
            "Only SELECT, INSERT, and UPDATE statements are allowed for data seeding."
        )

    parsed = urllib.parse.urlparse(db_connection)

    if parsed.scheme == "mysql":
        import pymysql  # late import — only needed on the MySQL path

        conn = pymysql.connect(
            host=parsed.hostname,
            port=parsed.port or 3306,
            user=parsed.username,
            password=parsed.password,
            database=(parsed.path or "").lstrip("/"),
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(statement)
            conn.commit()
        finally:
            conn.close()
    elif parsed.scheme in ("postgresql", "postgres"):
        import psycopg2  # late import — only needed on the PostgreSQL path

        conn = psycopg2.connect(
            host=parsed.hostname,
            port=parsed.port or 5432,
            user=parsed.username,
            password=parsed.password,
            dbname=(parsed.path or "").lstrip("/"),
        )
        try:
            with conn.cursor() as cursor:
                cursor.execute(statement)
            conn.commit()
        finally:
            conn.close()
    else:
        raise NotImplementedError(
            f"DB dialect {parsed.scheme!r} not yet supported by worldsim.seeding"
        )
