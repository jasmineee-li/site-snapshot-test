"""Reddit/Postmill seed context resolution."""

from __future__ import annotations

import logging
from typing import Any

from warp_taskgen.seeding import db as _db
from warp_taskgen.seeding.db import (
    _configure_read_only_connection,
    _parse_runtime_db_connection,
    _quote_identifier,
)

logger = logging.getLogger(__name__)


_REDDIT_TABLE_NAME_CACHE: dict[tuple[str, str], str] = {}


def _derive_reddit_seed_context(
    task: dict[str, Any],
    instance: dict[str, Any],
    placeholders: set[str],
) -> dict[str, Any]:
    if "forum_name" not in placeholders and "submission_id" not in placeholders:
        return {}

    forum = _resolve_reddit_forum(task, instance)
    context: dict[str, Any] = {}
    if forum is not None:
        forum_name = forum.get("name")
        forum_id = forum.get("id")
        if isinstance(forum_name, str) and forum_name.strip():
            context["forum_name"] = forum_name.strip()
        if forum_id is not None:
            context["forum_id"] = forum_id

    if "submission_id" in placeholders and "forum_name" in context:
        submission_id = _resolve_reddit_submission_id(
            task, instance, forum_name=context["forum_name"]
        )
        if submission_id is not None:
            context["submission_id"] = submission_id
    return context


def _resolve_reddit_forum(task: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any] | None:
    instantiation = task.get("instantiation_dict")
    forum_hint = None
    if isinstance(instantiation, dict):
        raw_forum = instantiation.get("forum")
        if isinstance(raw_forum, str) and raw_forum.strip():
            forum_hint = raw_forum.strip()
    if forum_hint is None:
        return None

    db_connection = instance.get("db_connection")
    if not db_connection:
        return {"name": forum_hint}

    parsed = _parse_runtime_db_connection(
        db_connection,
        purpose="Reddit seed placeholder resolution requires instance['db_connection']",
    )
    conn = _db._connect_db(parsed)
    try:
        scheme = parsed.scheme.lower()
        _configure_read_only_connection(conn, scheme)
        forum_table = _quote_identifier(
            _resolve_reddit_table_name(
                conn,
                scheme,
                db_connection,
                logical_name="forum",
                candidates=("forums", "forum"),
            ),
            scheme,
        )
        name_col = _quote_identifier("name", scheme)
        title_col = _quote_identifier("title", scheme)
        id_col = _quote_identifier("id", scheme)
        query = (
            f"SELECT {id_col}, {name_col}, {title_col} "
            f"FROM {forum_table} "
            f"WHERE LOWER({name_col}) = LOWER(%s) OR LOWER({title_col}) = LOWER(%s) "
            f"ORDER BY CASE WHEN LOWER({name_col}) = LOWER(%s) THEN 0 ELSE 1 END "
            "LIMIT 1"
        )
        with conn.cursor() as cursor:
            cursor.execute(query, [forum_hint, forum_hint, forum_hint])
            row = cursor.fetchone()
    except Exception as exc:
        raise RuntimeError(
            f"failed to resolve reddit forum_name for {forum_hint!r}: {exc}"
        ) from exc
    finally:
        try:
            conn.rollback()
        except Exception:
            logger.debug("Failed to rollback reddit forum lookup", exc_info=True)
        conn.close()

    if not row:
        return {"name": forum_hint}
    if isinstance(row, dict):
        return {"id": row.get("id"), "name": row.get("name"), "title": row.get("title")}
    row_values = list(row)
    return {
        "id": row_values[0] if len(row_values) > 0 else None,
        "name": row_values[1] if len(row_values) > 1 else forum_hint,
        "title": row_values[2] if len(row_values) > 2 else None,
    }


def _resolve_reddit_submission_id(
    task: dict[str, Any],
    instance: dict[str, Any],
    *,
    forum_name: str,
) -> Any:
    benign_reward = task.get("reward_function", {})
    if isinstance(benign_reward, dict) and isinstance(benign_reward.get("benign_reward"), dict):
        reward = benign_reward["benign_reward"]
    else:
        reward = benign_reward

    expected_title = _expected_reddit_post_title(reward)
    if not expected_title:
        return None

    db_connection = instance.get("db_connection")
    if not db_connection:
        raise RuntimeError(
            "reddit seed placeholder resolution requires instance['db_connection'] "
            "to map the target post title to a submission_id"
        )

    parsed = _parse_runtime_db_connection(
        db_connection,
        purpose="Reddit seed placeholder resolution requires instance['db_connection']",
    )
    conn = _db._connect_db(parsed)
    try:
        scheme = parsed.scheme.lower()
        _configure_read_only_connection(conn, scheme)
        submission_table = _quote_identifier(
            _resolve_reddit_table_name(
                conn,
                scheme,
                db_connection,
                logical_name="submission",
                candidates=("submissions", "submission"),
            ),
            scheme,
        )
        forum_table = _quote_identifier(
            _resolve_reddit_table_name(
                conn,
                scheme,
                db_connection,
                logical_name="forum",
                candidates=("forums", "forum"),
            ),
            scheme,
        )
        submission_id_col = _quote_identifier("id", scheme)
        forum_id_col = _quote_identifier("forum_id", scheme)
        timestamp_col = _quote_identifier("timestamp", scheme)
        title_col = _quote_identifier("title", scheme)
        forum_name_col = _quote_identifier("name", scheme)
        forum_title_col = _quote_identifier("title", scheme)
        query = (
            f"SELECT s.{submission_id_col} "
            f"FROM {submission_table} s "
            f"JOIN {forum_table} f ON s.{forum_id_col} = f.{_quote_identifier('id', scheme)} "
            f"WHERE s.{title_col} = %s "
            f"AND (LOWER(f.{forum_name_col}) = LOWER(%s) OR LOWER(f.{forum_title_col}) = LOWER(%s)) "
            f"ORDER BY s.{timestamp_col} DESC "
            "LIMIT 1"
        )
        with conn.cursor() as cursor:
            cursor.execute(query, [expected_title, forum_name, forum_name])
            row = cursor.fetchone()
    except Exception as exc:
        raise RuntimeError(
            f"failed to resolve reddit submission_id for {expected_title!r} in forum {forum_name!r}: {exc}"
        ) from exc
    finally:
        try:
            conn.rollback()
        except Exception:
            logger.debug("Failed to rollback reddit submission lookup", exc_info=True)
        conn.close()

    if row is None:
        raise RuntimeError(
            f"could not resolve reddit submission_id for title {expected_title!r} in forum "
            f"{forum_name!r}"
        )
    if isinstance(row, dict):
        return row.get("id")
    if isinstance(row, (list, tuple)):
        return row[0] if row else None
    return row


def _resolve_reddit_table_name(
    conn: Any,
    scheme: str,
    db_connection: str,
    *,
    logical_name: str,
    candidates: tuple[str, ...],
) -> str:
    cache_key = (db_connection, logical_name)
    cached = _REDDIT_TABLE_NAME_CACHE.get(cache_key)
    if cached:
        return cached
    if scheme not in ("postgresql", "postgres"):
        resolved = candidates[0]
        _REDDIT_TABLE_NAME_CACHE[cache_key] = resolved
        return resolved

    with conn.cursor() as cursor:
        for candidate in candidates:
            cursor.execute("SELECT to_regclass(%s)", [candidate])
            row = cursor.fetchone()
            resolved = None
            if isinstance(row, dict):
                resolved = row.get("to_regclass")
            elif isinstance(row, (list, tuple)):
                resolved = row[0] if row else None
            else:
                resolved = row
            if resolved not in (None, ""):
                _REDDIT_TABLE_NAME_CACHE[cache_key] = candidate
                return candidate

    tried = ", ".join(candidates)
    raise RuntimeError(
        f"reddit schema table resolution failed for logical table {logical_name!r} (tried: {tried})"
    )


def _expected_reddit_post_title(reward_function: dict[str, Any]) -> str | None:
    eval_entries = reward_function.get("eval")
    if not isinstance(eval_entries, list):
        return None
    for entry in eval_entries:
        if not isinstance(entry, dict):
            continue
        expected = entry.get("expected")
        if not isinstance(expected, dict):
            continue
        retrieved = expected.get("retrieved_data")
        if not isinstance(retrieved, list):
            continue
        for item in retrieved:
            if not isinstance(item, dict):
                continue
            post_title = item.get("post_title")
            if isinstance(post_title, str) and post_title.strip():
                return post_title.strip()
    return None


__all__ = [
    "_REDDIT_TABLE_NAME_CACHE",
    "_derive_reddit_seed_context",
    "_expected_reddit_post_title",
    "_resolve_reddit_forum",
    "_resolve_reddit_submission_id",
    "_resolve_reddit_table_name",
]
