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

import asyncio
import logging
import re
import urllib.parse
from typing import Any

import requests

logger = logging.getLogger(__name__)

_MULTI_STATEMENT_PATTERN = re.compile(r";(?=(?:[^']|'[^']*')*$)")
_DISALLOWED_SQL_KEYWORDS = re.compile(
    r"\b("
    r"DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|DELETE|REPLACE|MERGE|CALL|DO|EXEC|EXECUTE|"
    r"BEGIN|COMMIT|ROLLBACK|SAVEPOINT|PREPARE|DEALLOCATE|COPY|LOAD|ATTACH|DETACH|VACUUM"
    r")\b",
    re.IGNORECASE,
)


def validate_data_seed(seed: dict[str, Any], *, allow_none: bool = False) -> None:
    """Validate a seed payload before it is persisted or executed."""
    if not isinstance(seed, dict):
        raise ValueError("data seed must be an object")

    mechanism = seed.get("mechanism")
    if mechanism in (None, "none"):
        if allow_none:
            return
        raise ValueError("data seed must declare a non-empty mechanism")

    if mechanism == "sql":
        statements = seed.get("statements")
        if not isinstance(statements, list) or not statements:
            raise ValueError("sql data seed must include a non-empty statements list")
        for statement in statements:
            if not isinstance(statement, str):
                raise ValueError("sql data seed statements must be strings")
            _validate_seed_sql(statement)
        return

    if mechanism == "api":
        api_calls = seed.get("api_calls")
        if not isinstance(api_calls, list) or not api_calls:
            raise ValueError("api data seed must include a non-empty api_calls list")
        for call in api_calls:
            if not isinstance(call, dict):
                raise ValueError("api data seed calls must be objects")
            method = call.get("method")
            path = call.get("path")
            if not isinstance(method, str) or not method.strip():
                raise ValueError("api data seed calls must include a method")
            if not isinstance(path, str) or not path.startswith("/"):
                raise ValueError("api data seed calls must include a path starting with '/'")
        return

    if mechanism == "state_push":
        if "state" not in seed:
            raise ValueError("state_push data seed must include a state payload")
        return

    raise ValueError(f"unknown data seed mechanism: {mechanism!r}")


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
    validate_data_seed(seed)
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


async def apply_data_seed_async(seed: dict[str, Any], instance: dict[str, Any]) -> None:
    """Apply a data seed without blocking the event loop."""
    await asyncio.to_thread(apply_data_seed, seed, instance)


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
    _validate_seed_sql(statement)

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


def _validate_seed_sql(statement: str) -> None:
    """Allow only single-statement INSERT/UPDATE seed mutations."""
    normalized = statement.strip()
    if not normalized:
        raise ValueError("SQL seed statement is empty")
    if _MULTI_STATEMENT_PATTERN.search(normalized.rstrip(";")):
        raise ValueError("SQL seed must be a single statement")
    if _DISALLOWED_SQL_KEYWORDS.search(normalized):
        raise ValueError(f"SQL seed contains a disallowed keyword: {normalized[:100]}...")

    first_token = normalized.split(None, 1)[0].upper()
    if first_token not in {"INSERT", "UPDATE"}:
        raise ValueError(f"SQL seed must start with INSERT or UPDATE, got {first_token!r}")
    if first_token == "UPDATE" and " WHERE " not in f" {normalized.upper()} ":
        raise ValueError("UPDATE seed statements must include a WHERE clause")
