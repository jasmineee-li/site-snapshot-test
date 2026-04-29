"""Phase 2 target resolution listing probes."""
# ruff: noqa: F821

from __future__ import annotations

from worldsim.phase_2.target_resolution._context import install_context

install_context(globals())

async def _list_gitlab_search(
    resource: Mapping[str, Any],
    task: Mapping[str, Any],
    instance: Mapping[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    anchors = resource.get("anchors") or {}
    query = _literalize_regex_value(str(anchors.get("query") or "")) or ""
    scope = _literalize_regex_value(str(anchors.get("scope") or "")) or "issues"
    project_id = anchors.get("project_id")
    endpoint = (
        f"/api/v4/projects/{project_id}/issues"
        if project_id and scope == "issues"
        else (
            f"/api/v4/projects/{project_id}/merge_requests"
            if project_id and scope == "merge_requests"
            else ("/api/v4/issues" if scope == "issues" else "/api/v4/merge_requests")
        )
    )
    params: dict[str, Any] = {
        "order_by": "updated_at",
        "sort": "desc",
        "per_page": limit,
    }
    if query:
        params["search"] = query
    data = await _probe_http_json(instance, endpoint, params=params)
    if not isinstance(data, list):
        return []
    item_kind = "gitlab_mr" if scope == "merge_requests" else "gitlab_issue"
    return [{"_item_kind": item_kind, **item} for item in data if isinstance(item, dict)]

def _first_query_value(query: Mapping[str, list[str]], key: str) -> str | None:
    values = query.get(key)
    if not values:
        return None
    value = str(values[0]).strip()
    return value or None

def _dashboard_query(resource: Mapping[str, Any], task: Mapping[str, Any]) -> dict[str, str]:
    query: dict[str, list[str]] = {}
    for raw in [
        str(resource.get("benign_read_url") or resource.get("start_url_resolved") or ""),
        *_iter_eval_urls(task),
        *_iter_start_urls(task),
    ]:
        if not raw:
            continue
        parsed = urlsplit(_strip_regex_anchors(raw))
        if not parsed.query:
            continue
        for key, values in parse_qs(parsed.query, keep_blank_values=True).items():
            if key in {
                "assignee_username",
                "author_username",
                "state",
                "scope",
                "sort",
                "order_by",
            }:
                query[key] = values
    out: dict[str, str] = {}
    for key in query:
        raw_value = _first_query_value(query, key)
        literal = _literalize_regex_value(raw_value)
        if literal is None:
            if raw_value is not None:
                logger.debug(
                    "dropping regex-encoded dashboard query param %s=%r (no literal branch)",
                    key,
                    raw_value,
                )
            continue
        out[key] = literal
    return out

def _gitlab_item_url(item: Mapping[str, Any]) -> str | None:
    for key in ("web_url", "target_url", "url"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    target = item.get("target")
    if isinstance(target, Mapping):
        return _gitlab_item_url(target)
    return None

def _normalize_href_path(
    href: str, site_url: str, *, require_same_origin: bool = True
) -> str | None:
    value = html.unescape(href).strip()
    if not value or value.startswith("#"):
        return None
    site = urlsplit(site_url.rstrip("/"))
    parsed = urlsplit(value)
    if require_same_origin and (parsed.scheme or parsed.netloc):
        if (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
        ) != (
            site.scheme.lower(),
            site.netloc.lower(),
        ):
            return None
    path = parsed.path or "/"
    return path.rstrip("/") or "/"

async def _gitlab_visible_dashboard_hrefs(
    instance: Mapping[str, Any],
    entry_url: str,
) -> set[str] | None:
    import requests

    from worldsim.auth_tokens import build_auth_headers

    site_url = str(instance.get("site_url") or "").rstrip("/")
    if not site_url or not entry_url:
        return None
    parsed_entry = urlsplit(entry_url)
    path = parsed_entry.path or "/"
    query = f"?{parsed_entry.query}" if parsed_entry.query else ""
    url = f"{site_url}{path}{query}"
    headers = build_auth_headers(_benign_probe_instance(instance), {}, mechanism="form")

    def _send() -> str | None:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code >= 400:
            return None
        return response.text

    body = await asyncio.to_thread(_send)
    if not body:
        return None
    hrefs: set[str] = set()
    for match in re.finditer(r"""href=["'](?P<href>[^"']+)["']""", body):
        normalized = _normalize_href_path(match.group("href"), site_url)
        if normalized:
            hrefs.add(normalized)
    return hrefs

def _filter_visible_gitlab_dashboard_items(
    items: list[dict[str, Any]],
    *,
    visible_hrefs: set[str],
    instance: Mapping[str, Any],
) -> list[dict[str, Any]]:
    site_url = str(instance.get("site_url") or "").rstrip("/")
    visible: list[dict[str, Any]] = []
    for item in items:
        item_url = _gitlab_item_url(item)
        if not item_url:
            continue
        item_path = _normalize_href_path(item_url, site_url, require_same_origin=False)
        if item_path is None:
            continue
        if item_path in visible_hrefs:
            visible.append({**item, "_entry_visible_href": item_path})
    return visible

async def _list_gitlab_dashboard(
    resource: Mapping[str, Any],
    task: Mapping[str, Any],
    instance: Mapping[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    dashboard = str((resource.get("anchors") or {}).get("dashboard") or "")
    username = _benign_user_handle(task) or ""
    params: dict[str, Any] = {
        "order_by": "updated_at",
        "sort": "desc",
        "per_page": limit,
    }
    params.update(_dashboard_query(resource, task))
    if dashboard in ("todos", "merge_requests"):
        if dashboard == "merge_requests" and username:
            params.setdefault("assignee_username", username)
        elif username:
            params.setdefault("author_username", username)
        endpoint = "/api/v4/merge_requests" if dashboard == "merge_requests" else "/api/v4/todos"
    else:
        endpoint = "/api/v4/issues"
        if username:
            params.setdefault("assignee_username", username)
    data = await _probe_http_json(instance, endpoint, params=params)
    if not isinstance(data, list):
        return []
    item_kind = "gitlab_mr" if dashboard == "merge_requests" else "gitlab_issue"
    items = [{"_item_kind": item_kind, **item} for item in data if isinstance(item, dict)]
    entry_url = str(resource.get("benign_read_url") or resource.get("start_url_resolved") or "")
    visible_hrefs = await _gitlab_visible_dashboard_hrefs(instance, entry_url)
    if not visible_hrefs:
        logger.info(
            "L4 gitlab dashboard %r has no visible anchors at %s; excluding API-only candidates",
            dashboard,
            entry_url,
        )
        return []
    return _filter_visible_gitlab_dashboard_items(
        items,
        visible_hrefs=visible_hrefs,
        instance=instance,
    )

async def _list_reddit_forum(
    resource: Mapping[str, Any],
    task: Mapping[str, Any],
    instance: Mapping[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    forum_name = _literalize_regex_value(
        str((resource.get("anchors") or {}).get("forum_name") or "")
    )
    if not forum_name:
        return []
    submissions = await _fetch_forum_submissions(instance, forum_name, limit=limit)
    if not submissions:
        return []
    return [{"_item_kind": "reddit_submission", **entry} for entry in submissions]

async def _default_listing_probe(
    resource: Mapping[str, Any],
    task: Mapping[str, Any],
    instance: Mapping[str, Any],
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    kind = resource.get("kind")
    limit = limit if limit is not None else _l4_top_n_default()
    if kind == "gitlab_search_result":
        return await _list_gitlab_search(resource, task, instance, limit=limit)
    if kind == "gitlab_dashboard_list":
        return await _list_gitlab_dashboard(resource, task, instance, limit=limit)
    if kind == "reddit_forum":
        return await _list_reddit_forum(resource, task, instance, limit=limit)
    return []
