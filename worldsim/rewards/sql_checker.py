from __future__ import annotations

import logging
import re
from typing import Any

from worldsim.db_urls import parse_supported_db_connection

logger = logging.getLogger(__name__)

_MULTI_STATEMENT_PATTERN = re.compile(r";(?=(?:[^']|'[^']*')*$)")
_READ_ONLY_QUERY_PREFIX = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_UNSAFE_QUERY_KEYWORDS = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|REPLACE|MERGE|CALL|"
    r"DO|EXEC|EXECUTE|COPY|LOAD|ATTACH|DETACH|VACUUM|ANALYZE|REFRESH|SET"
    r")\b",
    re.IGNORECASE,
)
_UNSAFE_QUERY_PATTERNS = (
    re.compile(r"\bINTO\b", re.IGNORECASE),
    re.compile(r"\bOUTFILE\b", re.IGNORECASE),
    re.compile(r"\bDUMPFILE\b", re.IGNORECASE),
    re.compile(r"\bLOCK\b", re.IGNORECASE),
    re.compile(r"\bFOR\s+(UPDATE|SHARE)\b", re.IGNORECASE),
)


def _db_query_match(reward: dict[str, Any], instance: dict[str, Any]) -> tuple[bool, str]:
    """Execute a SQL query and compare result to expected value.

    Used for injection verification in Phase 4 where we need to check
    database state directly.
    """
    query = reward.get("query", "")
    expected = reward.get("expected")
    db_conn = instance.get("db_connection")

    if not db_conn:
        return False, "no db_connection on instance"
    if not query:
        return False, "no query in reward spec"
    try:
        _validate_read_only_query(query)
    except ValueError as exc:
        return False, str(exc)

    try:
        parsed = parse_supported_db_connection(
            db_conn,
            purpose="db_query_match requires instance['db_connection']",
        )
        scheme = parsed.scheme.lower()

        if scheme == "mysql":
            import pymysql

            conn = pymysql.connect(
                host=parsed.hostname,
                port=parsed.port or 3306,
                user=parsed.username,
                password=parsed.password,
                database=(parsed.path or "").lstrip("/"),
            )
        elif scheme in ("postgresql", "postgres"):
            import psycopg2  # late import — only needed on the PostgreSQL path

            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                dbname=(parsed.path or "").lstrip("/"),
            )
        else:
            return False, f"unsupported DB dialect: {scheme}"

        try:
            _configure_read_only_connection(conn, scheme)
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
        finally:
            try:
                conn.rollback()
            except Exception:
                logger.debug("Failed to rollback read-only query connection", exc_info=True)
            conn.close()

        # Compare result
        if expected is None:
            passed = len(rows) > 0
            return passed, f"got {len(rows)} rows"

        passed = _rows_match_expected(rows, expected)
        return passed, f"expected {expected!r}; got rows {rows[:5]!r}"

    except Exception as e:
        return False, f"db_query_match error: {e}"


def _rows_match_expected(rows: Any, expected: Any) -> bool:
    """Compare SQL rows against an expected value without substring matching."""
    if expected is None:
        return bool(rows)
    if not isinstance(rows, (list, tuple)):
        return _values_match(rows, expected)

    if _values_match(rows, expected):
        return True

    for row in rows:
        if _values_match(row, expected):
            return True
        if isinstance(row, (list, tuple)):
            for cell in row:
                if _values_match(cell, expected):
                    return True
    return False


def _values_match(actual: Any, expected: Any) -> bool:
    """Return True when two SQL result values are materially equal."""
    if isinstance(actual, tuple):
        actual = list(actual)
    if isinstance(expected, tuple):
        expected = list(expected)

    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _values_match(left, right) for left, right in zip(actual, expected, strict=True)
        )

    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            return False
        return all(_values_match(actual[key], expected[key]) for key in actual)

    if type(actual) is type(expected):
        return actual == expected

    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return actual == expected

    if isinstance(actual, str) or isinstance(expected, str):
        return str(actual).strip().casefold() == str(expected).strip().casefold()

    return actual == expected


def _validate_read_only_query(query: str) -> None:
    """Restrict db_query_match to a single read-only query."""
    normalized = query.strip()
    if not normalized:
        raise ValueError("db_query_match query is empty")
    if _MULTI_STATEMENT_PATTERN.search(normalized.rstrip(";")):
        raise ValueError("db_query_match must be a single statement")
    if not _READ_ONLY_QUERY_PREFIX.match(normalized):
        raise ValueError("db_query_match only permits SELECT or WITH queries")
    if _UNSAFE_QUERY_KEYWORDS.search(normalized):
        raise ValueError("db_query_match query contains a write-capable SQL keyword")
    for pattern in _UNSAFE_QUERY_PATTERNS:
        if pattern.search(normalized):
            raise ValueError("db_query_match query contains a write-capable SQL clause")


def _configure_read_only_connection(conn: Any, scheme: str) -> None:
    """Enable a required read-only transaction guard for SQL reward checks."""
    try:
        if hasattr(conn, "autocommit"):
            conn.autocommit = False
        with conn.cursor() as cursor:
            if scheme == "mysql":
                cursor.execute("SET SESSION TRANSACTION READ ONLY")
                cursor.execute("START TRANSACTION READ ONLY")
            elif scheme in ("postgresql", "postgres"):
                cursor.execute("BEGIN")
                cursor.execute("SET TRANSACTION READ ONLY")
            else:
                raise ValueError(f"unsupported DB dialect: {scheme}")
    except Exception as exc:
        raise RuntimeError("could not enable read-only transaction guard") from exc
