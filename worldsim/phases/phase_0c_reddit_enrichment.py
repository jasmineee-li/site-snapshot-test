"""Phase 0c host-side Reddit inventory enrichment.

The LLM-authored Phase 0c profile can contain stale Postmill forum samples from
task fixtures or database examples. Phase 1 route contracts need concrete
forum-list anchors that are reachable on the live instance, so enumerate the
forum table host-side and verify each ``/f/{name}`` page before exposing it as
``profile.available_entities.forums``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

import requests

from worldsim.db_urls import replace_db_host
from worldsim.seeding import (
    _configure_read_only_connection,
    _connect_db,
    _parse_runtime_db_connection,
    _quote_identifier,
    _resolve_reddit_table_name,
)

logger = logging.getLogger(__name__)

_MAX_FORUMS = 100
_MAX_EMPTY_SUBMISSIONS = 100
_TIMEOUT_SECONDS = 10


class RedditInventoryEnrichmentError(RuntimeError):
    """Raised when Reddit inventory enrichment cannot complete."""


def enrich_reddit_forums(
    site_url: str,
    db_connection: str | None,
    *,
    runtime_db_host: str | None = None,
    timeout: int = _TIMEOUT_SECONDS,
) -> dict[str, list[dict[str, str]]]:
    """Return live-reachable Reddit/Postmill forums from the benchmark instance."""
    if not site_url:
        raise RedditInventoryEnrichmentError("site_url is required for reddit forum enrichment")
    if not db_connection:
        raise RedditInventoryEnrichmentError(
            "db_connection is required for reddit forum enrichment"
        )

    errors: list[str] = []
    rows: list[dict[str, Any]] | None = None
    for candidate in _db_connection_candidates(db_connection, runtime_db_host):
        try:
            rows = _read_forum_rows(candidate)
            break
        except RedditInventoryEnrichmentError as exc:
            errors.append(str(exc))
    if rows is None:
        detail = "; ".join(errors) if errors else "no DB connection candidates"
        raise RedditInventoryEnrichmentError(detail)

    forums: list[dict[str, str]] = []
    seen: set[str] = set()
    base = site_url.rstrip("/")
    for row in rows:
        name = str(row.get("name") or "").strip()
        if not name or name in seen:
            continue
        if not _forum_page_reachable(base, name, timeout=timeout):
            continue
        seen.add(name)
        forum: dict[str, str] = {"name": name}
        for key in ("id", "title"):
            value = row.get(key)
            if value not in (None, ""):
                forum[key] = str(value).strip()
        forums.append(forum)
    try:
        submissions = _read_empty_submission_rows_from_candidates(
            db_connection,
            runtime_db_host=runtime_db_host,
        )
    except RedditInventoryEnrichmentError as exc:
        logger.warning("Reddit empty submission enrichment failed: %s", exc)
        submissions = []
    empty_submissions: list[dict[str, str]] = []
    for row in submissions:
        forum_name = str(row.get("forum_name") or "").strip()
        submission_id = str(row.get("id") or "").strip()
        if not forum_name or not submission_id:
            continue
        if not _submission_page_reachable(base, forum_name, submission_id, timeout=timeout):
            continue
        item: dict[str, str] = {
            "id": submission_id,
            "forum": forum_name,
            "existing_comment_count": "0",
            "max_existing_comments_for_comment_seed": "0",
            "seeded_comment_visibility_candidate": "true",
        }
        title = row.get("title")
        if title not in (None, ""):
            item["title"] = str(title).strip()
        empty_submissions.append(item)
    result: dict[str, list[dict[str, str]]] = {"forums": forums}
    if empty_submissions:
        result["submissions"] = empty_submissions
    return result


def _db_connection_candidates(
    db_connection: str,
    runtime_db_host: str | None,
) -> list[str]:
    """Return DB connection strings in preferred host-local order.

    Phase 0c's browser probes run in Modal and need externally reachable web
    URLs, but this DB inventory enrichment runs on the orchestrator host. On r5
    that host cannot hairpin to its public EC2 IP, so remote jobs export the
    host-local orchestrator address and we try that first.
    """

    candidates: list[str] = []
    host = str(runtime_db_host or "").strip()
    if host:
        rewritten = replace_db_host(db_connection, host)
        if rewritten != db_connection:
            candidates.append(rewritten)
    candidates.append(db_connection)
    deduped: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        deduped.append(candidate)
    return deduped


def merge_reddit_inventory_into_profile(
    profile: Mapping[str, Any],
    inventory: Mapping[str, list[dict[str, str]]],
) -> dict[str, Any]:
    """Return a profile copy with ``available_entities.forums`` populated."""
    merged = dict(profile)
    existing = merged.get("available_entities")
    available: dict[str, Any] = dict(existing) if isinstance(existing, Mapping) else {}
    forums = inventory.get("forums")
    if forums:
        available["forums"] = list(forums)
    submissions = inventory.get("submissions")
    if submissions:
        available["submissions"] = list(submissions)
    if available:
        merged["available_entities"] = available
    return merged


def common_reddit_forum_inventory(
    inventories: list[Mapping[str, list[dict[str, str]]]],
) -> dict[str, list[dict[str, str]]]:
    """Return forums present in every replica inventory, preserving first order."""
    if not inventories:
        return {"forums": []}
    forum_lists: list[list[dict[str, str]]] = []
    for inventory in inventories:
        forums = inventory.get("forums")
        if not forums:
            return {"forums": []}
        forum_lists.append([forum for forum in forums if isinstance(forum, Mapping)])
    name_sets = [
        {str(forum.get("name") or "").strip() for forum in forums if forum.get("name")}
        for forums in forum_lists
    ]
    common_names = set.intersection(*name_sets) if name_sets else set()
    if not common_names:
        return {"forums": []}
    out: list[dict[str, str]] = []
    seen: set[str] = set()
    for forum in forum_lists[0]:
        name = str(forum.get("name") or "").strip()
        if not name or name not in common_names or name in seen:
            continue
        seen.add(name)
        out.append(
            {str(key): str(value) for key, value in forum.items() if value not in (None, "")}
        )
    result: dict[str, list[dict[str, str]]] = {"forums": out}
    submissions = _common_reddit_submission_inventory(inventories)
    if submissions:
        result["submissions"] = submissions
    return result


def _common_reddit_submission_inventory(
    inventories: list[Mapping[str, list[dict[str, str]]]],
) -> list[dict[str, str]]:
    submission_lists: list[list[dict[str, str]]] = []
    for inventory in inventories:
        submissions = inventory.get("submissions")
        if not submissions:
            return []
        submission_lists.append(
            [submission for submission in submissions if isinstance(submission, Mapping)]
        )
    key_sets = [
        {
            (
                str(submission.get("forum") or submission.get("forum_name") or "").strip(),
                str(submission.get("id") or submission.get("submission_id") or "").strip(),
            )
            for submission in submissions
            if submission.get("id") or submission.get("submission_id")
        }
        for submissions in submission_lists
    ]
    common_keys = set.intersection(*key_sets) if key_sets else set()
    if not common_keys:
        return []
    out: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for submission in submission_lists[0]:
        key = (
            str(submission.get("forum") or submission.get("forum_name") or "").strip(),
            str(submission.get("id") or submission.get("submission_id") or "").strip(),
        )
        if not key[0] or not key[1] or key not in common_keys or key in seen:
            continue
        seen.add(key)
        out.append(
            {str(item_key): str(value) for item_key, value in submission.items() if value not in (None, "")}
        )
    return out


def _read_forum_rows(db_connection: str) -> list[dict[str, Any]]:
    parsed = _parse_runtime_db_connection(
        db_connection,
        purpose="Reddit Phase 0c forum enrichment requires instance['db_connection']",
    )
    conn = None
    try:
        conn = _connect_db(parsed)
        scheme = parsed.scheme.lower()
        _configure_read_only_connection(conn, scheme)
        table = _quote_identifier(
            _resolve_reddit_table_name(
                conn,
                scheme,
                db_connection,
                logical_name="forum",
                candidates=("forums", "forum"),
            ),
            scheme,
        )
        id_col = _quote_identifier("id", scheme)
        name_col = _quote_identifier("name", scheme)
        title_col = _quote_identifier("title", scheme)
        query = (
            f"SELECT {id_col}, {name_col}, {title_col} "
            f"FROM {table} "
            f"WHERE {name_col} IS NOT NULL AND {name_col} <> '' "
            f"ORDER BY {id_col} ASC "
            f"LIMIT {_MAX_FORUMS}"
        )
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    except Exception as exc:
        raise RedditInventoryEnrichmentError(f"failed to enumerate reddit forums: {exc}") from exc
    finally:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                logger.debug("Failed to rollback reddit forum enrichment lookup", exc_info=True)
            conn.close()

    result: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            result.append({"id": row.get("id"), "name": row.get("name"), "title": row.get("title")})
            continue
        values = list(row)
        result.append(
            {
                "id": values[0] if len(values) > 0 else None,
                "name": values[1] if len(values) > 1 else None,
                "title": values[2] if len(values) > 2 else None,
            }
        )
    return result


def _read_empty_submission_rows_from_candidates(
    db_connection: str,
    *,
    runtime_db_host: str | None,
) -> list[dict[str, Any]]:
    errors: list[str] = []
    for candidate in _db_connection_candidates(db_connection, runtime_db_host):
        try:
            return _read_empty_submission_rows(candidate)
        except RedditInventoryEnrichmentError as exc:
            errors.append(str(exc))
    detail = "; ".join(errors) if errors else "no DB connection candidates"
    raise RedditInventoryEnrichmentError(detail)


def _read_empty_submission_rows(db_connection: str) -> list[dict[str, Any]]:
    parsed = _parse_runtime_db_connection(
        db_connection,
        purpose="Reddit Phase 0c empty submission enrichment requires instance['db_connection']",
    )
    conn = None
    try:
        conn = _connect_db(parsed)
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
        comment_table = _quote_identifier(
            _resolve_reddit_table_name(
                conn,
                scheme,
                db_connection,
                logical_name="comment",
                candidates=("comments", "comment"),
            ),
            scheme,
        )
        submission_id_col = _quote_identifier("id", scheme)
        forum_id_col = _quote_identifier("forum_id", scheme)
        title_col = _quote_identifier("title", scheme)
        timestamp_col = _quote_identifier("timestamp", scheme)
        forum_name_col = _quote_identifier("name", scheme)
        comment_id_col = _quote_identifier("id", scheme)
        comment_submission_id_col = _quote_identifier("submission_id", scheme)
        query = (
            f"SELECT s.{submission_id_col} AS id, s.{title_col} AS title, "
            f"f.{forum_name_col} AS forum_name, COUNT(c.{comment_id_col}) AS comment_count "
            f"FROM {submission_table} s "
            f"JOIN {forum_table} f ON s.{forum_id_col} = f.{_quote_identifier('id', scheme)} "
            f"LEFT JOIN {comment_table} c ON c.{comment_submission_id_col} = s.{submission_id_col} "
            f"WHERE f.{forum_name_col} IS NOT NULL AND f.{forum_name_col} <> '' "
            f"AND LOWER(s.{title_col}) NOT LIKE '%worldsim%' "
            f"GROUP BY s.{submission_id_col}, s.{title_col}, f.{forum_name_col}, s.{timestamp_col} "
            f"HAVING COUNT(c.{comment_id_col}) = 0 "
            f"ORDER BY s.{timestamp_col} DESC "
            f"LIMIT {_MAX_EMPTY_SUBMISSIONS}"
        )
        with conn.cursor() as cursor:
            cursor.execute(query)
            rows = cursor.fetchall()
    except Exception as exc:
        raise RedditInventoryEnrichmentError(
            f"failed to enumerate empty reddit submissions: {exc}"
        ) from exc
    finally:
        if conn is not None:
            try:
                conn.rollback()
            except Exception:
                logger.debug("Failed to rollback reddit submission enrichment lookup", exc_info=True)
            conn.close()

    result: list[dict[str, Any]] = []
    for row in rows:
        if isinstance(row, Mapping):
            result.append(
                {
                    "id": row.get("id"),
                    "title": row.get("title"),
                    "forum_name": row.get("forum_name"),
                    "comment_count": row.get("comment_count"),
                }
            )
            continue
        values = list(row)
        result.append(
            {
                "id": values[0] if len(values) > 0 else None,
                "title": values[1] if len(values) > 1 else None,
                "forum_name": values[2] if len(values) > 2 else None,
                "comment_count": values[3] if len(values) > 3 else None,
            }
        )
    return result


def _forum_page_reachable(base_url: str, forum_name: str, *, timeout: int) -> bool:
    encoded = quote(forum_name.strip().strip("/"), safe="")
    if not encoded:
        return False
    url = f"{base_url}/f/{encoded}"
    try:
        response = requests.get(url, timeout=timeout, allow_redirects=False)
    except requests.RequestException as exc:
        logger.debug("Reddit forum reachability probe failed for %s: %s", url, exc)
        return False
    return response.status_code == 200


def _submission_page_reachable(
    base_url: str,
    forum_name: str,
    submission_id: str,
    *,
    timeout: int,
) -> bool:
    encoded_forum = quote(forum_name.strip().strip("/"), safe="")
    encoded_submission = quote(str(submission_id).strip().strip("/"), safe="")
    if not encoded_forum or not encoded_submission:
        return False
    url = f"{base_url}/f/{encoded_forum}/{encoded_submission}"
    try:
        response = requests.get(url, timeout=timeout, allow_redirects=False)
    except requests.RequestException as exc:
        logger.debug("Reddit submission reachability probe failed for %s: %s", url, exc)
        return False
    return response.status_code == 200


__all__ = [
    "RedditInventoryEnrichmentError",
    "common_reddit_forum_inventory",
    "enrich_reddit_forums",
    "merge_reddit_inventory_into_profile",
]
