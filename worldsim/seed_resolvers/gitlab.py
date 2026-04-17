from __future__ import annotations

from typing import Any
from urllib.parse import quote

import requests

from .types import ResolvedCall, ResolverError


def create(
    target: dict[str, Any],
    instance: dict[str, Any],
    seed_context: dict[str, Any],
) -> ResolvedCall:
    resource_type = str(target.get("resource_type", "")).strip().lower()
    mechanism = str(target.get("mechanism") or "api").strip().lower()

    if resource_type == "project":
        preview = _preview_project_context(target, instance, seed_context)
        return ResolvedCall(
            method=str(target.get("method") or "POST").strip().upper() or "POST",
            url=_form_or_api_url(instance, mechanism, form_path="/projects", api_path="/api/v4/projects"),
            headers={},
            context_additions={"project_path": preview["project_path"]},
        )
    if resource_type == "group":
        return ResolvedCall(
            method=str(target.get("method") or "POST").strip().upper() or "POST",
            url=_form_or_api_url(instance, mechanism, form_path="/groups", api_path="/api/v4/groups"),
            headers={},
        )

    if resource_type in {"issue", "issue_note", "mr", "mr_note", "repo_file"}:
        project = _ensure_project(target, instance, seed_context)
        context_additions: dict[str, Any] = {
            "project_id": project["id"],
            "project_path": project["path_with_namespace"],
        }

        if resource_type == "issue":
            if _is_dry_run(instance):
                context_additions["issue_iid"] = _seed_context_iid(seed_context, key="issue_iid", default="1")
            return ResolvedCall(
                method="POST",
                url=_api_url(instance, f"/api/v4/projects/{project['id']}/issues"),
                headers={},
                context_additions=context_additions,
            )

        if resource_type == "issue_note":
            issue = _ensure_issue(project, target, instance, seed_context)
            context_additions["issue_iid"] = issue["iid"]
            return ResolvedCall(
                method="POST",
                url=_api_url(
                    instance,
                    f"/api/v4/projects/{project['id']}/issues/{issue['iid']}/notes",
                ),
                headers={},
                context_additions=context_additions,
            )

        if resource_type == "mr":
            mr_payload = _prepare_mr_payload(project, target, seed_context)
            if not _is_dry_run(instance):
                _ensure_branch_and_commit(project, mr_payload, instance)
            else:
                context_additions["mr_iid"] = _seed_context_iid(seed_context, key="mr_iid", default="1")
            return ResolvedCall(
                method="POST",
                url=_api_url(instance, f"/api/v4/projects/{project['id']}/merge_requests"),
                headers={},
                body=mr_payload,
                context_additions=context_additions,
            )

        if resource_type == "mr_note":
            merge_request = _ensure_merge_request(project, target, instance, seed_context)
            context_additions["mr_iid"] = merge_request["iid"]
            return ResolvedCall(
                method="POST",
                url=_api_url(
                    instance,
                    f"/api/v4/projects/{project['id']}/merge_requests/{merge_request['iid']}/notes",
                ),
                headers={},
                context_additions=context_additions,
            )

        if resource_type == "repo_file":
            return ResolvedCall(
                method="POST",
                url=_api_url(instance, f"/api/v4/projects/{project['id']}/repository/commits"),
                headers={},
                context_additions=context_additions,
            )

    raise ResolverError(
        "unsupported_resource_type",
        f"gitlab resolver does not support create resource_type={resource_type!r}",
    )


def update(
    target: dict[str, Any],
    instance: dict[str, Any],
    seed_context: dict[str, Any],
) -> ResolvedCall:
    resource_type = str(target.get("resource_type", "")).strip().lower()
    if resource_type == "user_status":
        return ResolvedCall(
            method="PUT",
            url=_api_url(instance, "/api/v4/user/status"),
            headers={},
        )
    if resource_type == "user_profile":
        return ResolvedCall(
            method="PUT",
            url=_api_url(instance, "/api/v4/user"),
            headers={},
        )
    raise ResolverError(
        "unsupported_resource_type",
        f"gitlab resolver does not support update resource_type={resource_type!r}",
    )


def _ensure_project(
    target: dict[str, Any],
    instance: dict[str, Any],
    seed_context: dict[str, Any],
) -> dict[str, Any]:
    preview = _preview_project_context(target, instance, seed_context)
    project_path = preview["project_path"]
    if _is_dry_run(instance):
        return {"id": 0, "path_with_namespace": project_path, "default_branch": "main"}

    project = _gitlab_get_json(
        instance,
        f"/api/v4/projects/{quote(project_path, safe='')}",
        allow_missing=True,
    )
    if isinstance(project, dict):
        return project

    current_user = _current_user(instance)
    leaf_name = project_path.split("/")[-1]
    project = _find_accessible_project(
        instance,
        current_user=current_user,
        leaf_name=leaf_name,
        expected_path=project_path,
    )
    if isinstance(project, dict):
        return project
    create_spec = _create_spec(target).get("project")
    description = ""
    if isinstance(create_spec, dict):
        description = str(create_spec.get("description_template") or "").strip()
    try:
        project = _gitlab_request_json(
            instance,
            "POST",
            "/api/v4/projects",
            json_body={
                "name": leaf_name,
                "path": leaf_name,
                "description": description,
                "initialize_with_readme": True,
                "visibility": "private",
            },
        )
    except requests.HTTPError as exc:
        raise _classify_project_create_error(exc, project_path) from exc
    if not isinstance(project, dict):
        raise ResolverError("invalid_project_create", "gitlab project create returned invalid payload")
    path_with_namespace = str(project.get("path_with_namespace") or "").strip()
    if path_with_namespace:
        project_path = path_with_namespace
    elif isinstance(current_user.get("username"), str) and current_user["username"].strip():
        project_path = f"{current_user['username'].strip()}/{leaf_name}"
        project["path_with_namespace"] = project_path
    return project


def _find_accessible_project(
    instance: dict[str, Any],
    *,
    current_user: dict[str, Any],
    leaf_name: str,
    expected_path: str,
) -> dict[str, Any] | None:
    user_id = current_user.get("id")
    search_paths: list[tuple[str, dict[str, Any]]] = []
    if user_id not in (None, ""):
        search_paths.append(
            (
                f"/api/v4/users/{quote(str(user_id), safe='')}/projects",
                {"search": leaf_name, "per_page": 100, "simple": True},
            )
        )
    search_paths.append(
        (
            "/api/v4/projects",
            {"search": leaf_name, "membership": True, "per_page": 100, "simple": True},
        )
    )
    for path, params in search_paths:
        projects = _gitlab_request_json(instance, "GET", path, params=params)
        match = _match_project_by_path(projects, expected_path)
        if match is not None:
            return match
    return None


def _match_project_by_path(projects: Any, expected_path: str) -> dict[str, Any] | None:
    normalized_expected = expected_path.strip().lower()
    if not isinstance(projects, list):
        return None
    for project in projects:
        if not isinstance(project, dict):
            continue
        path_with_namespace = _project_path_with_namespace(project)
        if path_with_namespace and path_with_namespace.lower() == normalized_expected:
            return project
    return None


def _project_path_with_namespace(project: dict[str, Any]) -> str:
    direct = str(project.get("path_with_namespace") or "").strip()
    if direct:
        return direct
    namespace = project.get("namespace")
    if isinstance(namespace, dict):
        namespace_path = str(namespace.get("full_path") or namespace.get("path") or "").strip()
        leaf = str(project.get("path") or "").strip()
        if namespace_path and leaf:
            return f"{namespace_path}/{leaf}"
    return ""


def _classify_project_create_error(exc: requests.HTTPError, project_path: str) -> ResolverError:
    response = exc.response
    if response is None:
        return ResolverError(
            "project_create_failed",
            f"gitlab rejected project create for {project_path!r}: {exc}",
        )

    message = _gitlab_error_message(response)
    kind = "project_create_failed"
    if response.status_code == 400:
        if _gitlab_error_contains(response, "has already been taken"):
            kind = "project_already_exists"
        elif _gitlab_error_contains(response, "can contain only") or _gitlab_error_contains(
            response, "is invalid"
        ):
            kind = "invalid_project_path"
    return ResolverError(
        kind,
        f"gitlab rejected project create for {project_path!r}: {message}",
    )


def _ensure_issue(
    project: dict[str, Any],
    target: dict[str, Any],
    instance: dict[str, Any],
    seed_context: dict[str, Any],
) -> dict[str, Any]:
    issue_spec = _create_spec(target).get("issue")
    title = _coerce_text(issue_spec, "title_template", f"Notes for {seed_context.get('task_id', 'task')}")
    description = _coerce_text(issue_spec, "body_template", "Context for seeded issue.")
    existing_issue_iid = _seed_context_iid(seed_context, key="issue_iid")
    if existing_issue_iid is not None:
        if _is_dry_run(instance):
            return {"iid": existing_issue_iid, "title": title}
        issue = _gitlab_get_json(
            instance,
            f"/api/v4/projects/{project['id']}/issues/{quote(str(existing_issue_iid), safe='')}",
            allow_missing=True,
        )
        if isinstance(issue, dict):
            return issue
    if _is_dry_run(instance):
        return {"iid": 1, "title": title}

    issues = _gitlab_request_json(
        instance,
        "GET",
        f"/api/v4/projects/{project['id']}/issues",
        params={"state": "opened", "search": title, "per_page": 20},
    )
    if isinstance(issues, list):
        for issue in issues:
            if (
                isinstance(issue, dict)
                and str(issue.get("title", "")).strip() == title
                and str(issue.get("description", "")).strip() == description
            ):
                return issue

    issue = _gitlab_request_json(
        instance,
        "POST",
        f"/api/v4/projects/{project['id']}/issues",
        json_body={"title": title, "description": description},
    )
    if not isinstance(issue, dict):
        raise ResolverError("invalid_issue_create", "gitlab issue create returned invalid payload")
    return issue


def _ensure_merge_request(
    project: dict[str, Any],
    target: dict[str, Any],
    instance: dict[str, Any],
    seed_context: dict[str, Any],
) -> dict[str, Any]:
    mr_payload = _prepare_mr_payload(project, target, seed_context)
    title = str(mr_payload.get("title") or "").strip()
    existing_mr_iid = _seed_context_iid(seed_context, key="mr_iid")
    if existing_mr_iid is not None:
        if _is_dry_run(instance):
            return {"iid": existing_mr_iid, "title": title}
        merge_request = _gitlab_get_json(
            instance,
            f"/api/v4/projects/{project['id']}/merge_requests/{quote(str(existing_mr_iid), safe='')}",
            allow_missing=True,
        )
        if isinstance(merge_request, dict):
            return merge_request
    if _is_dry_run(instance):
        return {"iid": 1, "title": title}

    merge_requests = _gitlab_request_json(
        instance,
        "GET",
        f"/api/v4/projects/{project['id']}/merge_requests",
        params={"state": "opened", "search": title, "per_page": 20},
    )
    if isinstance(merge_requests, list):
        for merge_request in merge_requests:
            if (
                isinstance(merge_request, dict)
                and str(merge_request.get("title", "")).strip() == title
                and str(merge_request.get("description", "")).strip()
                == str(mr_payload.get("description", "")).strip()
                and str(merge_request.get("source_branch", "")).strip()
                == str(mr_payload.get("source_branch", "")).strip()
                and str(merge_request.get("target_branch", "")).strip()
                == str(mr_payload.get("target_branch", "")).strip()
            ):
                return merge_request

    _ensure_branch_and_commit(project, mr_payload, instance)
    merge_request = _gitlab_request_json(
        instance,
        "POST",
        f"/api/v4/projects/{project['id']}/merge_requests",
        json_body=mr_payload,
    )
    if not isinstance(merge_request, dict):
        raise ResolverError(
            "invalid_merge_request_create",
            "gitlab merge request create returned invalid payload",
        )
    return merge_request


def _ensure_branch_and_commit(
    project: dict[str, Any],
    mr_payload: dict[str, Any],
    instance: dict[str, Any],
) -> None:
    source_branch = str(mr_payload.get("source_branch") or "").strip()
    target_branch = str(
        mr_payload.get("target_branch") or project.get("default_branch") or "main"
    ).strip()
    if not source_branch:
        raise ResolverError("missing_source_branch", "merge request payload is missing source_branch")
    if not target_branch:
        target_branch = "main"

    try:
        _gitlab_request_json(
            instance,
            "POST",
            f"/api/v4/projects/{project['id']}/repository/branches",
            json_body={"branch": source_branch, "ref": target_branch},
        )
    except requests.HTTPError as exc:
        response = exc.response
        if response is None or response.status_code != 400:
            raise
        if not _gitlab_error_contains(response, "already exists"):
            raise

    file_path = f"seed-{source_branch.replace('/', '-')}.md"
    content = f"Seed branch for {source_branch}\n"
    commit_message = f"Seed branch {source_branch}"
    existing_content = _gitlab_get_file_content(
        instance,
        project_id=project["id"],
        file_path=file_path,
        ref=source_branch,
    )
    if existing_content == content:
        return
    try:
        _gitlab_request_json(
            instance,
            "POST",
            f"/api/v4/projects/{project['id']}/repository/commits",
            json_body={
                "branch": source_branch,
                "commit_message": commit_message,
                "actions": [
                    {
                        "action": "create",
                        "file_path": file_path,
                        "content": content,
                    }
                ],
            },
        )
    except requests.HTTPError as exc:
        response = exc.response
        if response is None or response.status_code != 400:
            raise
        if not _gitlab_error_contains(response, "already exists"):
            raise
        _gitlab_request_json(
            instance,
            "POST",
            f"/api/v4/projects/{project['id']}/repository/commits",
            json_body={
                "branch": source_branch,
                "commit_message": commit_message,
                "actions": [
                    {
                        "action": "update",
                        "file_path": file_path,
                        "content": content,
                    }
                ],
            },
        )


def _prepare_mr_payload(
    project: dict[str, Any],
    target: dict[str, Any],
    seed_context: dict[str, Any],
) -> dict[str, Any]:
    create_spec = _create_spec(target)
    mr_spec = create_spec.get("mr")
    if not isinstance(mr_spec, dict):
        mr_spec = {}

    raw_body = target.get("body")
    payload = dict(raw_body) if isinstance(raw_body, dict) else {}
    title = _coerce_text(mr_spec, "title_template", payload.get("title") or f"Notes on {seed_context.get('topic', 'task')}")
    description = _coerce_text(
        mr_spec,
        "body_template",
        payload.get("description") or f"Seeded MR context for {seed_context.get('task_id', 'task')}.",
    )
    default_branch = str(project.get("default_branch") or "main").strip() or "main"
    source_branch = str(
        payload.get("source_branch")
        or mr_spec.get("source_branch_template")
        or f"webagent-{seed_context.get('task_id', 'task')}"
    ).strip()
    target_branch = str(
        payload.get("target_branch") or mr_spec.get("target_branch_template") or default_branch
    ).strip()
    payload.update(
        {
            "title": title,
            "description": description,
            "source_branch": source_branch,
            "target_branch": target_branch,
        }
    )
    return payload


def _preview_project_context(
    target: dict[str, Any],
    instance: dict[str, Any],
    seed_context: dict[str, Any],
) -> dict[str, Any]:
    create_spec = _create_spec(target)
    project_spec = create_spec.get("project")
    if not isinstance(project_spec, dict):
        project_spec = {}
    raw_path = (
        project_spec.get("path_template")
        or project_spec.get("name_template")
        or f"webagent-task-{seed_context.get('task_id', 'task')}"
    )
    leaf_path = _slugify(str(raw_path))
    username = str(seed_context.get("username") or _current_username_preview(instance)).strip()
    project_path = f"{username}/{leaf_path}" if username and "/" not in leaf_path else leaf_path
    return {"project_id": 0, "project_path": project_path}


def _create_spec(target: dict[str, Any]) -> dict[str, Any]:
    create_spec = target.get("create")
    return create_spec if isinstance(create_spec, dict) else {}


def _coerce_text(spec: Any, key: str, default: str) -> str:
    if isinstance(spec, dict):
        value = spec.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return str(default).strip()


def _slugify(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in value.strip())
    while "--" in slug:
        slug = slug.replace("--", "-")
    slug = slug.strip("-")
    return slug or "webagent-task"


def _current_username_preview(instance: dict[str, Any]) -> str:
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
    return "current-user"


def _nested_lookup(value: Any, path: tuple[str, ...]) -> Any:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _current_user(instance: dict[str, Any]) -> dict[str, Any]:
    user = _gitlab_request_json(instance, "GET", "/api/v4/user")
    if not isinstance(user, dict):
        raise ResolverError("invalid_current_user", "gitlab resolver received invalid current-user payload")
    return user


def _seed_context_iid(
    seed_context: dict[str, Any],
    *,
    key: str,
    default: str | None = None,
) -> str | int | None:
    for candidate in (seed_context.get(key), seed_context.get("iid"), default):
        if candidate not in (None, ""):
            return candidate
    return None


def _gitlab_get_json(
    instance: dict[str, Any],
    path: str,
    *,
    allow_missing: bool = False,
) -> Any:
    return _gitlab_request_json(instance, "GET", path, allow_missing=allow_missing)


def _gitlab_request_json(
    instance: dict[str, Any],
    method: str,
    path: str,
    *,
    json_body: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    allow_missing: bool = False,
) -> Any:
    from worldsim import seeding as seeding_module

    response = requests.request(
        method.strip().upper(),
        _api_url(instance, path),
        headers=seeding_module._build_request_headers(instance, {}, mechanism="api"),
        json=json_body,
        params=params,
        timeout=30,
        allow_redirects=False,
    )
    if allow_missing and response.status_code == 404:
        return None
    if 300 <= response.status_code < 400:
        raise ResolverError(
            "unexpected_redirect",
            f"gitlab resolver request for {path} returned redirect status={response.status_code}",
        )
    response.raise_for_status()
    if not response.text:
        return {}
    try:
        return response.json()
    except ValueError:
        return {}


def _gitlab_error_contains(response: requests.Response, needle: str) -> bool:
    needle = needle.lower()
    body = response.text or ""
    return needle in body.lower()


def _gitlab_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        message = _flatten_gitlab_error_payload(payload.get("message"))
        if message:
            return message
    body = (response.text or "").strip()
    if body:
        return body
    return f"HTTP {response.status_code}"


def _flatten_gitlab_error_payload(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_flatten_gitlab_error_payload(item) for item in value]
        return "; ".join(part for part in parts if part)
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            flattened = _flatten_gitlab_error_payload(item)
            if flattened:
                parts.append(f"{key}: {flattened}")
        return "; ".join(parts)
    return ""


def _gitlab_get_file_content(
    instance: dict[str, Any],
    *,
    project_id: Any,
    file_path: str,
    ref: str,
) -> str | None:
    from worldsim import seeding as seeding_module

    response = requests.request(
        "GET",
        _api_url(instance, f"/api/v4/projects/{project_id}/repository/files/{quote(file_path, safe='')}/raw"),
        headers=seeding_module._build_request_headers(instance, {}, mechanism="api"),
        params={"ref": ref},
        timeout=30,
        allow_redirects=False,
    )
    if response.status_code == 404:
        return None
    if 300 <= response.status_code < 400:
        raise ResolverError(
            "unexpected_redirect",
            f"gitlab resolver file probe for {file_path!r} returned redirect status={response.status_code}",
        )
    response.raise_for_status()
    return response.text


def _form_or_api_url(instance: dict[str, Any], mechanism: str, *, form_path: str, api_path: str) -> str:
    if mechanism == "form":
        return _api_url(instance, form_path)
    return _api_url(instance, api_path)


def _api_url(instance: dict[str, Any], path: str) -> str:
    site_url = str(instance.get("site_url", "")).rstrip("/")
    if not site_url:
        raise ResolverError("missing_site_url", "gitlab resolver requires instance['site_url']")
    return f"{site_url}{path}"


def _is_dry_run(instance: dict[str, Any]) -> bool:
    return bool(instance.get("seed_resolver_dry_run"))
