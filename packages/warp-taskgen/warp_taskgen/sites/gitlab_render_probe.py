"""GitLab-owned Phase 2c render probe.

Everything Phase 2c's render check knows about GitLab lives here: which
URLs are issuable surfaces, which selectors mean "the thread rendered",
and the two read-your-write fallbacks that read the authoritative
resource back through the page's request context when the DOM race
loses. ``phase_2_render_check`` reaches this behavior through a
Site-keyed lookup rather than a ``site_name == "gitlab"`` branch.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any
from urllib.parse import quote as urlquote
from urllib.parse import urlsplit

from warp_taskgen.sites.catalog import default_catalog
from warp_taskgen.sites.readback import ReadbackDecision, ReadbackObservation
from warp_taskgen.sites.render_probe import (
    RenderOutcome,
    _same_origin,
    normalize_for_text_match,
    wait_for_body_text,
)

logger = logging.getLogger(__name__)

_GITLAB_NOTE_SELECTOR = ".notes .note, .discussion-notes .note, ul.notes-list .note"
_GITLAB_ISSUABLE_LIST_SELECTOR = ".issuable-list, .issues-list, .merge-requests-list"


def _is_gitlab_issuable_surface(url: str) -> bool:
    return "/-/issues" in url or "/-/merge_requests" in url


def _is_gitlab_issuable_detail_surface(url: str) -> bool:
    return "/-/issues/" in url or "/-/merge_requests/" in url


def _is_gitlab_issue_detail_surface(url: str) -> bool:
    return "/-/issues/" in url


async def _gitlab_note_ryw_fastpath(
    *,
    page: Any,
    target_url: str,
    site_name: str,
    write_tokens: dict[str, Any] | None,
    timeout_ms: int,
    scoped_extra_http_headers: dict[str, str] | None = None,
    header_scope_url: str | None = None,
    diagnostics: dict[str, Any] | None = None,
    readback_site: Any | None = None,
) -> RenderOutcome | None:
    """Read-your-write fallback for GitLab issue / MR notes.

    When the body-text match misses on an issue or MR page, fetch the
    authoritative ``/discussions.json`` endpoint via the page's request
    context and look for the note_id the editor returned from its POST.
    If the id is in the JSON, the note is observably present on the
    server — downstream agents will see it the moment the Vue layer
    finishes hydrating, regardless of where the DOM-render race left
    ``text_content('body')``.

    Returns a passed ``RenderOutcome`` on match, ``None`` otherwise
    (so the caller falls through to the existing error classification).
    Skip conditions: non-gitlab site, URL not an issue/MR page,
    write_tokens missing note_id, or the JSON fetch fails.
    """
    if site_name != "gitlab" or write_tokens is None:
        logger.info(
            "phase 2c render RYW skip: site=%s write_tokens=%s", site_name, bool(write_tokens)
        )
        return None
    note_id = write_tokens.get("note_id")
    if note_id in (None, ""):
        logger.info("phase 2c render RYW skip: no note_id in write_tokens=%s", write_tokens)
        return None
    lower = target_url.lower()
    if "/-/issues/" not in lower and "/-/merge_requests/" not in lower:
        logger.info("phase 2c render RYW skip: URL not issue/MR: %s", target_url)
        return None
    # Strip an existing /discussions.json suffix so we don't build /discussions.json/discussions.json
    # when the target_url is already the JSON surface.
    base_url = target_url.split("?", 1)[0].rstrip("/")
    if base_url.endswith("/discussions.json"):
        base_url = base_url[: -len("/discussions.json")]
    json_url = f"{base_url}/discussions.json"
    logger.info(
        "phase 2c render RYW firing: note_id=%s json_url=%s (from target=%s)",
        note_id,
        json_url,
        target_url,
    )
    try:
        request_kwargs: dict[str, Any] = {
            "timeout": max(1000, int(timeout_ms)),
            "max_redirects": 0,
        }
        headers = _gitlab_same_origin_request_headers(
            url=json_url,
            scoped_extra_http_headers=scoped_extra_http_headers,
            header_scope_url=header_scope_url,
        )
        if headers:
            request_kwargs["headers"] = headers
        response = await page.request.get(json_url, **request_kwargs)
    except Exception as exc:
        logger.debug(
            "phase 2c render check: RYW fetch of %s raised %s: %s",
            json_url,
            exc.__class__.__name__,
            exc,
        )
        return None
    try:
        status = response.status
    except Exception:
        status = None
    if status is None or status < 200 or status >= 300:
        return None
    try:
        body = await response.text()
    except Exception:
        return None
    note_id_str = str(note_id)
    bound_readback = readback_site or default_catalog().bind(
        site="gitlab",
        origin=_gitlab_origin_from_target(target_url),
    )
    decision = bound_readback.interpret_readback(
        ReadbackObservation(
            kind="resource_identity",
            identity_tokens={"note_id": note_id},
            payload=body,
        )
    )
    matched = decision if isinstance(decision, ReadbackDecision) and decision.verified else None
    logger.info(
        "phase 2c render RYW fetched %s status=%s body_len=%d looking_for=%r match=%s",
        json_url,
        status,
        len(body or ""),
        f"note_id={note_id_str}",
        matched is not None,
    )
    if matched is not None:
        marker = next(
            (
                candidate
                for candidate in (
                    f'"id":{note_id_str}',
                    f'"id": {note_id_str}',
                    f'"id":"{note_id_str}"',
                    f'"id": "{note_id_str}"',
                )
                if candidate in body
            ),
            None,
        )
        pos = body.find(marker) if marker is not None else -1
        snippet = body[max(0, pos - 80) : pos + 200] if pos >= 0 else body[:200]
        return RenderOutcome.passed(
            url=json_url,
            signature=matched.matched_signature or f"note_id={note_id_str}",
            snippet=snippet,
            rendered_body_text=matched.rendered_text,
            diagnostics=diagnostics,
        )
    return None


def _gitlab_same_origin_request_headers(
    *,
    url: str,
    scoped_extra_http_headers: dict[str, str] | None,
    header_scope_url: str | None,
) -> dict[str, str] | None:
    if scoped_extra_http_headers and header_scope_url:
        if _same_origin(url, header_scope_url):
            return scoped_extra_http_headers
    return None


def _gitlab_origin_from_target(target_url: str) -> str | None:
    try:
        parsed = urlsplit(target_url)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return f"{parsed.scheme}://{parsed.netloc}"


def _gitlab_issue_description_ryw_urls(target_url: str, write_tokens: dict[str, Any]) -> list[str]:
    origin = _gitlab_origin_from_target(target_url)
    if origin is None:
        return []
    issue_iid = write_tokens.get("issue_iid")
    urls: list[str] = []
    try:
        path = urlsplit(target_url).path
    except ValueError:
        path = ""
    match = re.match(
        r"(?P<issue_base>.*?/-/issues/)(?P<path_iid>\d+)(?:\.json|/discussions\.json)?/?$",
        path,
    )
    if match and issue_iid not in (None, ""):
        urls.append(f"{origin}{match.group('issue_base')}{urlquote(str(issue_iid), safe='')}.json")
    project_id = write_tokens.get("project_id")
    if project_id not in (None, "") and issue_iid not in (None, ""):
        urls.append(
            f"{origin}/api/v4/projects/{urlquote(str(project_id), safe='')}"
            f"/issues/{urlquote(str(issue_iid), safe='')}"
        )
    return list(dict.fromkeys(urls))


def _append_gitlab_issue_description_ryw_diagnostic(
    diagnostics: dict[str, Any] | None,
    entry: dict[str, Any],
) -> None:
    if diagnostics is None:
        return
    attempts = diagnostics.setdefault("gitlab_issue_description_ryw_attempts", [])
    if not isinstance(attempts, list):
        attempts = []
        diagnostics["gitlab_issue_description_ryw_attempts"] = attempts
    if len(attempts) >= 8:
        return
    attempts.append(entry)


def _gitlab_issue_description_snapshot(description: str) -> dict[str, Any]:
    return {
        "description_len": len(description),
        "description_sha256": hashlib.sha256(description.encode("utf-8")).hexdigest()[:16],
        "description_prefix": description[:240],
    }


async def _gitlab_issue_description_ryw_fastpath(
    *,
    page: Any,
    target_url: str,
    site_name: str,
    signature: str,
    write_tokens: dict[str, Any] | None,
    timeout_ms: int,
    scoped_extra_http_headers: dict[str, str] | None = None,
    header_scope_url: str | None = None,
    diagnostics: dict[str, Any] | None = None,
    readback_site: Any | None = None,
) -> RenderOutcome | None:
    """Read-your-write fallback for GitLab created issue descriptions.

    This deliberately fires only from issue detail/read surfaces, never from
    project issue-list pages. A project-list title carrier must still prove the
    title is visible in the listing DOM; this fallback exists for created issue
    descriptions whose detail page/API visibility can race the first render
    probe. Phase 2c still runs the separate reachability graph after this, so a
    listing-to-detail carrier is admitted only if the benign route exposes the
    detail page witnesses.
    """
    if site_name != "gitlab" or write_tokens is None:
        return None
    if write_tokens.get("note_id") not in (None, ""):
        return None
    project_id = write_tokens.get("project_id")
    issue_iid = write_tokens.get("issue_iid")
    if project_id in (None, "") or issue_iid in (None, ""):
        return None
    if not _is_gitlab_issue_detail_surface(target_url):
        return None
    urls = _gitlab_issue_description_ryw_urls(target_url, write_tokens)
    if not urls:
        return None
    logger.info(
        "phase 2c render issue RYW firing: project_id=%s issue_iid=%s urls=%s",
        project_id,
        issue_iid,
        urls,
    )
    for read_url in urls:
        try:
            request_kwargs: dict[str, Any] = {
                "timeout": max(1000, int(timeout_ms)),
                "max_redirects": 0,
            }
            headers = _gitlab_same_origin_request_headers(
                url=read_url,
                scoped_extra_http_headers=scoped_extra_http_headers,
                header_scope_url=header_scope_url,
            )
            if headers:
                request_kwargs["headers"] = headers
            response = await page.request.get(read_url, **request_kwargs)
        except Exception as exc:
            logger.debug(
                "phase 2c render check: issue RYW fetch of %s raised %s: %s",
                read_url,
                exc.__class__.__name__,
                exc,
            )
            _append_gitlab_issue_description_ryw_diagnostic(
                diagnostics,
                {
                    "url": read_url,
                    "result": "request_error",
                    "error": f"{exc.__class__.__name__}: {exc}",
                },
            )
            continue
        try:
            status = response.status
        except Exception:
            status = None
        if status is None or status < 200 or status >= 300:
            _append_gitlab_issue_description_ryw_diagnostic(
                diagnostics,
                {"url": read_url, "status": status, "result": "http_status"},
            )
            continue
        try:
            payload = await response.json()
        except Exception:
            try:
                payload = json.loads(await response.text())
            except Exception:
                _append_gitlab_issue_description_ryw_diagnostic(
                    diagnostics,
                    {"url": read_url, "status": status, "result": "invalid_json"},
                )
                continue
        if not isinstance(payload, dict):
            _append_gitlab_issue_description_ryw_diagnostic(
                diagnostics,
                {"url": read_url, "status": status, "result": "non_object_json"},
            )
            continue
        description = payload.get("description")
        if not isinstance(description, str):
            _append_gitlab_issue_description_ryw_diagnostic(
                diagnostics,
                {"url": read_url, "status": status, "result": "missing_description"},
            )
            continue
        normalized_description = normalize_for_text_match(description)
        normalized_signature = normalize_for_text_match(signature)
        bound_readback = readback_site or default_catalog().bind(
            site="gitlab",
            origin=_gitlab_origin_from_target(target_url),
        )
        decision = bound_readback.interpret_readback(
            ReadbackObservation(
                kind="resource_signature",
                identity_tokens={"project_id": project_id, "issue_iid": issue_iid},
                payload={
                    "normalized_description": normalize_for_text_match(description),
                    "normalized_signature": normalize_for_text_match(signature),
                    "project_id": payload.get("project_id"),
                    "iid": payload.get("iid"),
                },
                signature=normalized_signature,
            )
        )
        if not isinstance(decision, ReadbackDecision) or not decision.verified:
            _append_gitlab_issue_description_ryw_diagnostic(
                diagnostics,
                {
                    "url": read_url,
                    "status": status,
                    "result": "description_signature_absent",
                    "signature": signature[:120],
                    **_gitlab_issue_description_snapshot(description),
                },
            )
            logger.info(
                "phase 2c render issue RYW fetched %s status=%s description_len=%d match=False",
                read_url,
                status,
                len(description),
            )
            continue
        pos = normalized_description.find(normalized_signature)
        snippet = (
            description[max(0, pos - 40) : pos + len(signature) + 120] if pos >= 0 else description
        )
        logger.info(
            "phase 2c render issue RYW fetched %s status=%s description_len=%d match=True",
            read_url,
            status,
            len(description),
        )
        _append_gitlab_issue_description_ryw_diagnostic(
            diagnostics,
            {
                "url": read_url,
                "status": status,
                "result": "description_signature_present",
                **_gitlab_issue_description_snapshot(description),
            },
        )
        return RenderOutcome.passed(
            url=read_url,
            signature=f"issue_iid={issue_iid}",
            snippet=snippet,
            rendered_body_text=description,
            diagnostics=diagnostics,
        )
    return None


class GitLabRenderProbe:
    """Site-owned render-probe behavior for GitLab."""

    site_name = "gitlab"

    async def wait_for_render(
        self,
        page: Any,
        *,
        target_url: str,
        signature: str,
        selector_timeout_ms: int,
        body_poll_timeout_ms: int,
    ) -> None:
        if not _is_gitlab_issuable_surface(target_url):
            return
        if _is_gitlab_issuable_detail_surface(target_url):
            selector = _GITLAB_NOTE_SELECTOR
        else:
            selector = _GITLAB_ISSUABLE_LIST_SELECTOR
        try:
            await page.wait_for_selector(selector, timeout=selector_timeout_ms)
        except Exception:
            pass
        # GitLab issue/MR detail threads and listing rows are both
        # populated after the initial response. Poll body text so
        # Phase 2c does not sample the SPA shell before the seeded
        # title/note has been inserted into the DOM.
        await wait_for_body_text(page, signature, body_poll_timeout_ms)

    def exact_visibility_comment_id(self, write_tokens: dict[str, Any] | None) -> str | None:
        """GitLab has no exact-comment visibility contract in Phase 2c."""

        return None

    async def read_your_write(
        self,
        *,
        page: Any,
        target_url: str,
        site_name: str,
        signature: str,
        write_tokens: dict[str, Any] | None,
        timeout_ms: int,
        scoped_extra_http_headers: dict[str, str] | None = None,
        header_scope_url: str | None = None,
        diagnostics: dict[str, Any] | None = None,
        readback_site: Any | None = None,
    ) -> RenderOutcome | None:
        note_hit = await _gitlab_note_ryw_fastpath(
            page=page,
            target_url=target_url,
            site_name=site_name,
            write_tokens=write_tokens,
            timeout_ms=timeout_ms,
            scoped_extra_http_headers=scoped_extra_http_headers,
            header_scope_url=header_scope_url,
            diagnostics=diagnostics,
            readback_site=readback_site,
        )
        if note_hit is not None:
            return note_hit
        return await _gitlab_issue_description_ryw_fastpath(
            page=page,
            target_url=target_url,
            site_name=site_name,
            signature=signature,
            write_tokens=write_tokens,
            timeout_ms=timeout_ms,
            scoped_extra_http_headers=scoped_extra_http_headers,
            header_scope_url=header_scope_url,
            diagnostics=diagnostics,
            readback_site=readback_site,
        )


__all__ = ["GitLabRenderProbe"]
