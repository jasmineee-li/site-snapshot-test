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
    return {"forums": forums}


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
    if available:
        merged["available_entities"] = available
    return merged


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


__all__ = [
    "RedditInventoryEnrichmentError",
    "enrich_reddit_forums",
    "merge_reddit_inventory_into_profile",
]
