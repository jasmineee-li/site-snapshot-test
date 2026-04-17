from __future__ import annotations

import re
from typing import Any
from urllib.parse import quote

import requests

from .types import ResolvedCall, ResolverError

_SUBMISSION_ID_RE = re.compile(r"/f/[^/]+/(\d+)(?:/|$)")


def create(
    target: dict[str, Any],
    instance: dict[str, Any],
    seed_context: dict[str, Any],
) -> ResolvedCall:
    site_url = str(instance.get("site_url", "")).rstrip("/")
    if not site_url:
        raise ResolverError("missing_site_url", "reddit resolver requires instance['site_url']")

    resource_type = str(target.get("resource_type", "")).strip().lower()
    create_spec = target.get("create")
    if not isinstance(create_spec, dict):
        create_spec = {}

    forum_name = _forum_name(create_spec, seed_context)
    if resource_type == "forum":
        if not forum_name:
            raise ResolverError(
                "missing_forum_name", "reddit forum target requires create.forum.name_template"
            )
        return ResolvedCall(
            method=str(target.get("method") or "POST").strip().upper() or "POST",
            url=f"{site_url}/create_forum",
            headers={},
            context_additions={"forum_name": forum_name, "forum_created": True},
        )

    if resource_type == "submission":
        if not forum_name:
            raise ResolverError(
                "missing_forum_name",
                "reddit submission target requires create.forum.name_template",
            )
        _ensure_forum(site_url, forum_name, create_spec, instance, seed_context)
        return ResolvedCall(
            method=str(target.get("method") or "POST").strip().upper() or "POST",
            url=f"{site_url}/submit/{forum_name}",
            headers={},
            context_additions={"forum_name": forum_name},
        )

    if resource_type == "comment":
        if not forum_name:
            raise ResolverError(
                "missing_forum_name",
                "reddit comment target requires create.forum.name_template",
            )
        submission_id = seed_context.get("submission_id")
        if submission_id in (None, ""):
            submission_id = _ensure_submission(
                site_url, forum_name, create_spec, instance, seed_context
            )
        if submission_id in (None, ""):
            raise ResolverError(
                "missing_submission_id",
                "reddit comment target could not determine submission_id for the parent submission",
            )
        return ResolvedCall(
            method=str(target.get("method") or "POST").strip().upper() or "POST",
            url=f"{site_url}/f/{forum_name}/{submission_id}/-/comment",
            headers={},
            context_additions={"forum_name": forum_name, "submission_id": submission_id},
        )

    raise ResolverError(
        "unsupported_resource_type",
        f"reddit resolver does not support create resource_type={resource_type!r}",
    )


def update(
    target: dict[str, Any],
    instance: dict[str, Any],
    seed_context: dict[str, Any],
) -> ResolvedCall:
    resource_type = str(target.get("resource_type", "")).strip().lower()
    if resource_type != "user_bio":
        raise ResolverError(
            "unsupported_resource_type",
            f"reddit resolver does not support update resource_type={resource_type!r}",
        )
    username = _current_username(instance, seed_context)
    return ResolvedCall(
        method="POST",
        url=f"{_site_url(instance)}/user/{quote(username, safe='')}/edit_biography",
        headers={},
    )


def _ensure_forum(
    site_url: str,
    forum_name: str,
    create_spec: dict[str, Any],
    instance: dict[str, Any],
    seed_context: dict[str, Any],
) -> None:
    if seed_context.get("forum_created") and seed_context.get("forum_name") == forum_name:
        return
    if _is_dry_run(instance):
        return
    if _find_existing_forum_name(instance, forum_name) is not None:
        return
    forum_spec = create_spec.get("forum")
    if not isinstance(forum_spec, dict):
        forum_spec = {}
    description = str(forum_spec.get("description_template") or f"Forum for {forum_name}").strip()
    with requests.Session() as session:
        _perform_login(session, instance)
        _submit_form(
            session,
            instance,
            f"{site_url}/create_forum",
            {
                "forum[name]": forum_name,
                "forum[description]": description,
            },
        )


def _ensure_submission(
    site_url: str,
    forum_name: str,
    create_spec: dict[str, Any],
    instance: dict[str, Any],
    seed_context: dict[str, Any],
) -> str:
    submission_spec = create_spec.get("submission")
    if not isinstance(submission_spec, dict):
        submission_spec = {}
    existing_submission_id = submission_spec.get("submission_id")
    if existing_submission_id not in (None, ""):
        return str(existing_submission_id)
    existing_context_submission = seed_context.get("submission_id")
    if existing_context_submission not in (None, ""):
        return str(existing_context_submission)
    if _is_dry_run(instance):
        return "preview-submission"

    title = str(submission_spec.get("title_template") or f"Submission {forum_name}").strip()
    body = str(
        submission_spec.get("body_template") or f"Discussion thread for {forum_name}."
    ).strip()
    existing_submission_id = _find_existing_submission_id(instance, forum_name, title)
    if existing_submission_id not in (None, ""):
        return str(existing_submission_id)

    _ensure_forum(site_url, forum_name, create_spec, instance, seed_context)
    with requests.Session() as session:
        _perform_login(session, instance)
        response = _submit_form(
            session,
            instance,
            f"{site_url}/submit/{forum_name}",
            {
                "submission[title]": title,
                "submission[body]": body,
            },
        )
    location = response.headers.get("Location", "")
    match = _SUBMISSION_ID_RE.search(location)
    if match:
        return match.group(1)
    try:
        payload = response.json()
    except ValueError as exc:
        raise ResolverError(
            "submission_id_missing",
            "reddit submission create response did not include a parseable submission id",
        ) from exc
    submission_id = payload.get("id") or payload.get("submission_id")
    if submission_id in (None, ""):
        raise ResolverError(
            "submission_id_missing",
            "reddit submission create response did not include a submission id",
        )
    return str(submission_id)


def _submit_form(
    session: requests.Session,
    instance: dict[str, Any],
    url: str,
    body_form: dict[str, Any],
) -> requests.Response:
    from worldsim import seeding as seeding_module

    headers = seeding_module._build_request_headers(instance, {}, mechanism="form")
    form_body = seeding_module._prepare_form_body(
        "POST",
        url,
        headers,
        body_form,
        instance,
        session,
    )
    response = session.request(
        "POST",
        url,
        headers=headers,
        data=form_body,
        timeout=30,
        allow_redirects=False,
    )
    if response.status_code >= 400:
        response.raise_for_status()
    return response


def _perform_login(session: requests.Session, instance: dict[str, Any]) -> None:
    from worldsim import seeding as seeding_module

    seeding_module._perform_web_login_if_needed(session, instance, "form")


def _forum_name(create_spec: dict[str, Any], seed_context: dict[str, Any]) -> str:
    forum_spec = create_spec.get("forum")
    if not isinstance(forum_spec, dict):
        forum_spec = {}
    candidate = (
        forum_spec.get("name_template") or forum_spec.get("forum_name") or forum_spec.get("name")
    )
    if isinstance(candidate, str) and candidate.strip():
        return candidate.strip()
    if isinstance(seed_context.get("forum_name"), str) and seed_context["forum_name"].strip():
        return seed_context["forum_name"].strip()
    return ""


def _current_username(instance: dict[str, Any], seed_context: dict[str, Any]) -> str:
    auth_username = _auth_username(instance)
    if auth_username is None:
        raise ResolverError(
            "missing_username",
            "reddit user_bio target requires auth credentials with a username",
        )
    context_username = seed_context.get("username")
    if (
        isinstance(context_username, str)
        and context_username.strip()
        and context_username.strip() != auth_username
    ):
        raise ResolverError(
            "username_mismatch",
            "reddit user_bio target username must match the authenticated user",
        )
    return auth_username


def _auth_username(instance: dict[str, Any]) -> str | None:
    # Reddit credentials may live at any of several paths depending on which
    # auth block the site uses. The agent_auth.authentication.credentials.*
    # path is the one reddit's instances.smoke.json actually uses — don't
    # miss it.
    for source in (
        _nested_lookup(instance.get("auth"), ("credentials", "username")),
        _nested_lookup(instance.get("auth"), ("username",)),
        _nested_lookup(instance.get("api_auth"), ("credentials", "username")),
        _nested_lookup(instance.get("api_auth"), ("username",)),
        _nested_lookup(instance.get("agent_auth"), ("authentication", "credentials", "username")),
        _nested_lookup(instance.get("agent_auth"), ("credentials", "username")),
        _nested_lookup(instance.get("agent_auth"), ("username",)),
    ):
        if isinstance(source, str) and source.strip():
            return source.strip()
    return None


def _nested_lookup(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _site_url(instance: dict[str, Any]) -> str:
    site_url = str(instance.get("site_url", "")).rstrip("/")
    if not site_url:
        raise ResolverError("missing_site_url", "reddit resolver requires instance['site_url']")
    return site_url


def _is_dry_run(instance: dict[str, Any]) -> bool:
    return bool(instance.get("seed_resolver_dry_run"))


def _find_existing_forum_name(instance: dict[str, Any], forum_name: str) -> str | None:
    row = _query_one(
        instance,
        """
        SELECT name
        FROM forum
        WHERE LOWER(name) = LOWER(%s) OR LOWER(title) = LOWER(%s)
        LIMIT 1
        """,
        [forum_name, forum_name],
    )
    if isinstance(row, dict):
        value = row.get("name")
        return str(value).strip() if value not in (None, "") else None
    if isinstance(row, (list, tuple)) and row:
        return str(row[0]).strip()
    return None


def _find_existing_submission_id(
    instance: dict[str, Any], forum_name: str, title: str
) -> str | None:
    row = _query_one(
        instance,
        """
        SELECT s.id
        FROM submission s
        JOIN forum f ON s.forum_id = f.id
        WHERE s.title = %s
          AND (LOWER(f.name) = LOWER(%s) OR LOWER(f.title) = LOWER(%s))
        ORDER BY s.timestamp DESC
        LIMIT 1
        """,
        [title, forum_name, forum_name],
    )
    if isinstance(row, dict):
        value = row.get("id")
        return str(value).strip() if value not in (None, "") else None
    if isinstance(row, (list, tuple)) and row:
        return str(row[0]).strip()
    if row not in (None, ""):
        return str(row).strip()
    return None


def _query_one(instance: dict[str, Any], query: str, params: list[Any]) -> Any:
    db_connection = instance.get("db_connection")
    if not db_connection:
        return None

    from worldsim import seeding as seeding_module

    try:
        parsed = seeding_module._parse_runtime_db_connection(
            db_connection,
            purpose="reddit resolver lookup requires instance['db_connection']",
        )
        conn = seeding_module._connect_db(parsed)
    except Exception:
        return None
    try:
        try:
            seeding_module._configure_read_only_connection(conn, parsed.scheme.lower())
            with conn.cursor() as cursor:
                cursor.execute(query, params)
                return cursor.fetchone()
        except Exception:
            return None
    finally:
        try:
            conn.rollback()
        except Exception:
            pass
        conn.close()
