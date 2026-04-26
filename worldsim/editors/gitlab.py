from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

import requests

from ._method_spec import FreeText, SelectorGroup, Token, editor_method
from ._read_surface import collect_platform_urls, host_and_path_forms, normalize_surface_urls
from .base import BaseSiteEditor, EditorError

_GITLAB_WEB_URL_PATHS = ("web_url", "_links.self")


def _gitlab_read_surface(
    response: Any,
    constructed_paths: list[str],
    site_url: str,
) -> list[str]:
    """Combine API-returned web_urls + constructed paths into host+path forms.

    Each URL is emitted in BOTH host-qualified and path-only forms so the
    C1b matcher survives cross-host replay (see handoff §12.8).
    """
    urls: list[str] = list(collect_platform_urls(response, list(_GITLAB_WEB_URL_PATHS)))
    urls.extend(constructed_paths)
    expanded: list[str] = []
    for url in urls:
        expanded.extend(host_and_path_forms(site_url, url))
    return normalize_surface_urls(expanded)


_GITLAB_LENGTH_TOKENS: tuple[str, ...] = (
    "is too long",
    "too long (maximum is",
    "exceeds the limit",
    "value too long for type",
)
_GITLAB_REQUIRED_TOKENS: tuple[str, ...] = (
    "is required",
    "can't be blank",
    "must be filled",
    "is missing",
    "must exist",
    "must be provided",
)
_GITLAB_POLICY_TOKENS: tuple[str, ...] = (
    "blocked",
    "has been blocked",
    "spam",
    "abuse",
    "forbidden",
    "rejected",
    "policy violation",
    "not allowed",
    "prohibited",
)
_SAFE_RESEED_PROJECT_PREFIXES: tuple[str, ...] = (
    "webagent-task-",
    "webagent-verify-",
)


def _extract_gitlab_error_messages(response: requests.Response) -> list[str]:
    """Flatten GitLab error payloads into a list of human-readable strings.

    GitLab answers validation failures in several shapes:

    - ``{"message": "..."}``
    - ``{"message": {"description": ["is too long ..."], ...}}``
    - ``{"error": "has already been taken"}``
    - Free-text body for non-JSON errors.
    """
    body: Any
    try:
        body = response.json()
    except ValueError:
        text = (response.text or "").strip()
        return [text] if text else []

    collected: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, str):
            if node.strip():
                collected.append(node.strip())
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for value in node.values():
                walk(value)

    if isinstance(body, dict):
        for key in ("message", "error", "errors", "error_description"):
            if key in body:
                walk(body[key])
        if not collected:
            walk(body)
    else:
        walk(body)

    return collected


def _first_line(messages: list[str]) -> str:
    if not messages:
        return ""
    primary = messages[0]
    return primary.splitlines()[0] if primary else ""


class GitlabEditor(BaseSiteEditor):
    site_name = "gitlab"
    supported_methods = frozenset(
        {
            "create_project",
            "create_group",
            "create_issue",
            "create_issue_note",
            "create_mr",
            "create_mr_note",
            "create_repo_file",
            "update_user_status",
            "update_user_profile",
            "update_user_profile_by_id",
            "update_user_status_by_id",
            "update_snippet",
            "create_snippet",
            "update_milestone",
            "create_label",
            "update_group",
        }
    )

    def __init__(self, instance: dict[str, Any], session: requests.Session) -> None:
        super().__init__(instance, session)
        self._profile_form_session_prepared = False
        self._created_projects_by_path: dict[str, dict[str, Any]] = {}

    @classmethod
    def probe_base_state(cls, instance: dict[str, Any]) -> None:
        with requests.Session() as session:
            editor = cls(instance, session)
            editor._current_user()

    def validate_args(self, method_name: str, args: dict[str, Any]) -> None:
        validators = {
            "create_project": lambda: self._require_args(args, "name_template"),
            "create_group": lambda: self._require_args(args, "name_template"),
            "create_issue": lambda: self._validate_issue_args(args),
            "create_issue_note": lambda: self._validate_issue_note_args(args),
            "create_mr": lambda: self._validate_merge_request_args(args),
            "create_mr_note": lambda: self._validate_merge_request_note_args(args),
            "create_repo_file": lambda: self._validate_repo_file_args(args),
            "update_user_status": lambda: self._require_args(args, "message"),
            "update_user_profile": lambda: self._validate_user_profile_args(args),
            "update_user_profile_by_id": lambda: self._require_args(args, "username", "bio"),
            "update_user_status_by_id": lambda: self._require_args(args, "username", "message"),
            "update_snippet": lambda: self._require_args(args, "snippet_id", "content"),
            "create_snippet": lambda: self._require_args(args, "title", "content"),
            "update_milestone": lambda: self._validate_milestone_args(args),
            "create_label": lambda: self._validate_label_args(args),
            "update_group": lambda: self._require_args(args, "group_path", "description"),
        }
        validator = validators.get(method_name)
        if validator is None:
            raise EditorError(
                "unsupported_method", f"unsupported gitlab editor method {method_name!r}"
            )
        validator()
        self._current_user()

    def preview_context(self, method_name: str, args: dict[str, Any]) -> dict[str, Any]:
        if method_name == "create_project":
            name_template = str(args.get("name_template") or "").strip() or "webagent-task"
            preview = self._preview_project_context(
                name_template=name_template,
                path_template=args.get("path_template"),
            )
            return {"project_id": 0, **preview}
        if method_name == "create_group":
            group_path = self._slugify(
                str(args.get("path_template") or args.get("name_template") or "group")
            )
            return {"group_id": 0, "group_path": group_path}
        if method_name == "create_issue":
            return {
                **self.preview_context(
                    "create_project",
                    {
                        "name_template": args.get("project_name_template") or "webagent-task",
                        "path_template": args.get("project_path_template"),
                    },
                ),
                "issue_iid": 1,
            }
        if method_name == "create_issue_note":
            return {
                **self.preview_context(
                    "create_issue",
                    {
                        "project_name_template": args.get("project_name_template"),
                        "project_path_template": args.get("project_path_template"),
                    },
                ),
                "note_id": 1,
            }
        if method_name == "create_mr":
            return {
                **self.preview_context(
                    "create_project",
                    {
                        "name_template": args.get("project_name_template") or "webagent-task",
                        "path_template": args.get("project_path_template"),
                    },
                ),
                "mr_iid": 1,
            }
        if method_name == "create_mr_note":
            return {
                **self.preview_context(
                    "create_mr",
                    {
                        "project_name_template": args.get("project_name_template"),
                        "project_path_template": args.get("project_path_template"),
                    },
                ),
                "note_id": 1,
            }
        if method_name == "create_repo_file":
            return self.preview_context(
                "create_project",
                {
                    "name_template": args.get("project_name_template") or "webagent-task",
                    "path_template": args.get("project_path_template"),
                },
            )
        return {}

    @editor_method(
        kinds=frozenset(),  # dangling parent — never a valid Option A attach
        http=("POST", "/api/v4/projects"),
        bindings={
            "name_template": FreeText(),
            "description_template": FreeText(required=False),
            "path_template": FreeText(required=False),
        },
    )
    def create_project(
        self,
        *,
        name_template: str,
        description_template: str | None = None,
        path_template: str | None = None,
    ) -> dict[str, Any]:
        preview = self._preview_project_context(
            name_template=name_template, path_template=path_template
        )
        project_path = preview["project_path"]
        cached = self._created_projects_by_path.get(project_path.lower())
        if isinstance(cached, dict):
            return dict(cached)
        current_user = self._current_user()
        project = self._gitlab_get_json(
            f"/api/v4/projects/{self._quote(project_path)}",
            allow_missing=True,
        )
        if isinstance(project, dict):
            if self._should_reap_preexisting_project(
                project_path, project, current_user=current_user
            ):
                self._reap_preexisting_project(project_path, project)
            else:
                raise self._preexisting_project_error(project_path)
        leaf_name = project_path.split("/")[-1]
        project = self._find_accessible_project(
            current_user=current_user,
            leaf_name=leaf_name,
            expected_path=project_path,
        )
        if isinstance(project, dict):
            if self._should_reap_preexisting_project(
                project_path, project, current_user=current_user
            ):
                self._reap_preexisting_project(project_path, project)
            else:
                raise self._preexisting_project_error(project_path)
        try:
            project = self._gitlab_request_json(
                "POST",
                "/api/v4/projects",
                json_body={
                    "name": leaf_name,
                    "path": leaf_name,
                    "description": str(description_template or "").strip(),
                    "initialize_with_readme": True,
                    "visibility": "private",
                },
            )
        except EditorError as exc:
            if exc.kind == "request_failed":
                classified = self._classify_project_create_error(exc, project_path)
                if classified.kind == "project_already_exists":
                    existing = self._gitlab_get_json(
                        f"/api/v4/projects/{self._quote(project_path)}",
                        allow_missing=True,
                    )
                    if not isinstance(existing, dict):
                        existing = self._find_accessible_project(
                            current_user=current_user,
                            leaf_name=leaf_name,
                            expected_path=project_path,
                        )
                    if isinstance(existing, dict) and self._should_reap_preexisting_project(
                        project_path,
                        existing,
                        current_user=current_user,
                    ):
                        self._reap_preexisting_project(project_path, existing)
                        project = self._gitlab_request_json(
                            "POST",
                            "/api/v4/projects",
                            json_body={
                                "name": leaf_name,
                                "path": leaf_name,
                                "description": str(description_template or "").strip(),
                                "initialize_with_readme": True,
                                "visibility": "private",
                            },
                        )
                    else:
                        raise classified from exc
                else:
                    raise classified from exc
            else:
                raise
        if not isinstance(project, dict):
            raise EditorError(
                "invalid_project_create", "gitlab project create returned invalid payload"
            )
        project_id = project.get("id")
        if project_id in (None, ""):
            raise EditorError("invalid_project_create", "gitlab project create missing id")
        resolved_path = str(project.get("path_with_namespace") or project_path)
        result = {
            "project_id": project_id,
            "project_path": resolved_path,
            "read_surface_urls": self._project_surface_urls(project, resolved_path),
            "read_surface_provenance_source": "editor_api_response",
        }
        self._created_projects_by_path[project_path.lower()] = dict(result)
        self._push_cleanup(lambda project_id=project_id: self.delete_project(project_id))
        return result

    @editor_method(
        kinds=frozenset(),  # dangling parent — never a valid Option A attach
        http=("POST", "/api/v4/groups"),
        bindings={
            "name_template": FreeText(),
            "description_template": FreeText(required=False),
            "path_template": FreeText(required=False),
        },
    )
    def create_group(
        self,
        *,
        name_template: str,
        description_template: str | None = None,
        path_template: str | None = None,
    ) -> dict[str, Any]:
        group_path = self._slugify(path_template or name_template)
        groups = self._gitlab_request_json(
            "GET",
            "/api/v4/groups",
            params={"search": group_path, "per_page": 100},
        )
        if isinstance(groups, list):
            for group in groups:
                if not isinstance(group, dict):
                    continue
                if str(group.get("path", "")).strip().lower() == group_path.lower():
                    group_full_path = str(group.get("full_path") or group_path)
                    return {
                        "group_id": group["id"],
                        "group_path": group_full_path,
                        "read_surface_urls": self._group_surface_urls(group, group_full_path),
                        "read_surface_provenance_source": "editor_api_response",
                    }
        created = self._gitlab_request_json(
            "POST",
            "/api/v4/groups",
            json_body={
                "name": str(name_template).strip(),
                "path": group_path,
                "description": str(description_template or "").strip(),
            },
        )
        if not isinstance(created, dict) or created.get("id") in (None, ""):
            raise EditorError(
                "invalid_group_create", "gitlab group create returned invalid payload"
            )
        group_id = created["id"]
        self._push_cleanup(lambda group_id=group_id: self.delete_group(group_id))
        group_full_path = str(created.get("full_path") or group_path)
        return {
            "group_id": group_id,
            "group_path": group_full_path,
            "read_surface_urls": self._group_surface_urls(created, group_full_path),
            "read_surface_provenance_source": "editor_api_response",
        }

    @editor_method(
        kinds=frozenset(),  # creates a new issue — not in resolver _ATTACH_SURFACES
        http=("POST", "/api/v4/projects/{project_id}/issues"),
        bindings={
            "title_template": FreeText(),
            "body_template": FreeText(required=False),
            "project_id": SelectorGroup("project", "{benign_project_id}"),
            "project_path_template": SelectorGroup(
                "project", "{benign_project_path}", required=False
            ),
            "project_name_template": SelectorGroup("project", required=False),
            "project_description_template": FreeText(required=False),
        },
    )
    def create_issue(
        self,
        *,
        title_template: str,
        body_template: str | None = None,
        project_id: Any | None = None,
        project_name_template: str | None = None,
        project_path_template: str | None = None,
        project_description_template: str | None = None,
    ) -> dict[str, Any]:
        project = self._ensure_project(
            project_id=project_id,
            project_name_template=project_name_template,
            project_path_template=project_path_template,
            project_description_template=project_description_template,
        )
        title = str(title_template).strip()
        description = str(body_template or "").strip()
        issue = self._gitlab_request_json(
            "POST",
            f"/api/v4/projects/{project['project_id']}/issues",
            json_body={"title": title, "description": description},
        )
        if not isinstance(issue, dict) or issue.get("iid") in (None, ""):
            raise EditorError(
                "invalid_issue_create", "gitlab issue create returned invalid payload"
            )
        issue_iid = issue["iid"]
        self._push_cleanup(
            lambda project_id=project["project_id"], issue_iid=issue_iid: self._close_issue(
                project_id, issue_iid
            )
        )
        return {
            "project_id": project["project_id"],
            "project_path": project["project_path"],
            "issue_iid": issue_iid,
            "read_surface_urls": self._issue_surface_urls(
                issue, project["project_path"], issue_iid
            ),
            "read_surface_provenance_source": "editor_api_response",
        }

    @editor_method(
        kinds=frozenset({"gitlab_issue", "gitlab_search_result", "gitlab_dashboard_list"}),
        http=("POST", "/api/v4/projects/{project_id}/issues/{issue_iid}/notes"),
        bindings={
            # LLM-facing "body" maps to note_body via seeding._editor_arg_name.
            "body": FreeText(),
            "project_id": SelectorGroup("project", "{benign_project_id}"),
            "project_path_template": SelectorGroup(
                "project", "{benign_project_path}", required=False
            ),
            "project_name_template": SelectorGroup("project", required=False),
            "project_description_template": FreeText(required=False),
            "issue_iid": Token("{benign_issue_iid}"),
            "issue_title_template": FreeText(required=False),
            "issue_body_template": FreeText(required=False),
        },
        surface_id_per_kind={
            "gitlab_issue": "note_on_issue",
            "gitlab_search_result": "note_on_issue",
            "gitlab_dashboard_list": "note_on_issue",
        },
        required_editor_args=("project_id", "issue_iid", "body"),
    )
    def create_issue_note(
        self,
        *,
        note_body: str,
        project_id: Any | None = None,
        project_name_template: str | None = None,
        project_path_template: str | None = None,
        project_description_template: str | None = None,
        issue_iid: Any | None = None,
        issue_title_template: str | None = None,
        issue_body_template: str | None = None,
    ) -> dict[str, Any]:
        project = self._ensure_project(
            project_id=project_id,
            project_name_template=project_name_template,
            project_path_template=project_path_template,
            project_description_template=project_description_template,
        )
        issue = self._ensure_issue(
            project_id=project["project_id"],
            issue_iid=issue_iid,
            title_template=issue_title_template,
            body_template=issue_body_template,
        )
        created = self._gitlab_request_json(
            "POST",
            f"/api/v4/projects/{project['project_id']}/issues/{issue['issue_iid']}/notes",
            json_body={"body": str(note_body)},
        )
        if not isinstance(created, dict) or created.get("id") in (None, ""):
            raise EditorError(
                "invalid_issue_note_create", "gitlab issue note create returned invalid payload"
            )
        note_id = created["id"]
        self._push_cleanup(
            lambda project_id=project["project_id"], issue_iid=issue["issue_iid"], note_id=note_id: (
                self._delete_issue_note(project_id, issue_iid, note_id)
            )
        )
        return {
            **project,
            "issue_iid": issue["issue_iid"],
            "note_id": note_id,
            "read_surface_urls": self._issue_surface_urls(
                created, project["project_path"], issue["issue_iid"]
            ),
            "read_surface_provenance_source": "editor_api_response",
        }

    @editor_method(
        kinds=frozenset(),  # creates a new MR — not in resolver _ATTACH_SURFACES
        http=("POST", "/api/v4/projects/{project_id}/merge_requests"),
        bindings={
            "title_template": FreeText(),
            "source_branch": FreeText(),
            "body_template": FreeText(required=False),
            "target_branch": FreeText(required=False),
            "project_id": SelectorGroup("project", "{benign_project_id}"),
            "project_path_template": SelectorGroup(
                "project", "{benign_project_path}", required=False
            ),
            "project_name_template": SelectorGroup("project", required=False),
            "project_description_template": FreeText(required=False),
        },
    )
    def create_mr(
        self,
        *,
        title_template: str,
        source_branch: str,
        body_template: str | None = None,
        target_branch: str | None = None,
        project_id: Any | None = None,
        project_name_template: str | None = None,
        project_path_template: str | None = None,
        project_description_template: str | None = None,
    ) -> dict[str, Any]:
        project = self._ensure_project(
            project_id=project_id,
            project_name_template=project_name_template,
            project_path_template=project_path_template,
            project_description_template=project_description_template,
        )
        payload = {
            "title": str(title_template).strip(),
            "description": str(body_template or "").strip(),
            "source_branch": str(source_branch).strip(),
            "target_branch": str(target_branch or project.get("default_branch") or "main").strip()
            or "main",
        }
        existing = self._find_existing_merge_request(project["project_id"], payload)
        if isinstance(existing, dict):
            existing_iid = existing.get("iid")
            if existing_iid not in (None, ""):
                self._close_merge_request(project["project_id"], existing_iid)
        self._ensure_branch_and_commit(project["project_id"], payload)
        created = self._gitlab_request_json(
            "POST",
            f"/api/v4/projects/{project['project_id']}/merge_requests",
            json_body=payload,
        )
        if not isinstance(created, dict) or created.get("iid") in (None, ""):
            raise EditorError(
                "invalid_merge_request_create",
                "gitlab merge request create returned invalid payload",
            )
        mr_iid = created["iid"]
        self._push_cleanup(
            lambda project_id=project["project_id"], mr_iid=mr_iid: self._close_merge_request(
                project_id, mr_iid
            )
        )
        return {
            **project,
            "mr_iid": mr_iid,
            "read_surface_urls": self._mr_surface_urls(created, project["project_path"], mr_iid),
            "read_surface_provenance_source": "editor_api_response",
        }

    @editor_method(
        kinds=frozenset({"gitlab_mr", "gitlab_search_result", "gitlab_dashboard_list"}),
        http=("POST", "/api/v4/projects/{project_id}/merge_requests/{mr_iid}/notes"),
        bindings={
            # LLM-facing "body" maps to note_body via seeding._editor_arg_name.
            "body": FreeText(),
            "project_id": SelectorGroup("project", "{benign_project_id}"),
            "project_path_template": SelectorGroup(
                "project", "{benign_project_path}", required=False
            ),
            "project_name_template": SelectorGroup("project", required=False),
            "project_description_template": FreeText(required=False),
            "mr_iid": Token("{benign_mr_iid}"),
            "mr_title_template": FreeText(required=False),
            "mr_body_template": FreeText(required=False),
            "source_branch": FreeText(required=False),
            "target_branch": FreeText(required=False),
        },
        surface_id_per_kind={
            "gitlab_mr": "note_on_mr",
            "gitlab_search_result": "note_on_mr",
            "gitlab_dashboard_list": "note_on_mr",
        },
        required_editor_args=("project_id", "mr_iid", "body"),
    )
    def create_mr_note(
        self,
        *,
        note_body: str,
        project_id: Any | None = None,
        project_name_template: str | None = None,
        project_path_template: str | None = None,
        project_description_template: str | None = None,
        mr_iid: Any | None = None,
        mr_title_template: str | None = None,
        mr_body_template: str | None = None,
        source_branch: str | None = None,
        target_branch: str | None = None,
    ) -> dict[str, Any]:
        project = self._ensure_project(
            project_id=project_id,
            project_name_template=project_name_template,
            project_path_template=project_path_template,
            project_description_template=project_description_template,
        )
        merge_request = self._ensure_merge_request(
            project_id=project["project_id"],
            mr_iid=mr_iid,
            title_template=mr_title_template,
            body_template=mr_body_template,
            source_branch=source_branch,
            target_branch=target_branch,
        )
        created = self._gitlab_request_json(
            "POST",
            f"/api/v4/projects/{project['project_id']}/merge_requests/{merge_request['mr_iid']}/notes",
            json_body={"body": str(note_body)},
        )
        if not isinstance(created, dict) or created.get("id") in (None, ""):
            raise EditorError(
                "invalid_mr_note_create", "gitlab MR note create returned invalid payload"
            )
        note_id = created["id"]
        self._push_cleanup(
            lambda project_id=project["project_id"], mr_iid=merge_request["mr_iid"], note_id=note_id: (
                self._delete_mr_note(project_id, mr_iid, note_id)
            )
        )
        return {
            **project,
            "mr_iid": merge_request["mr_iid"],
            "note_id": note_id,
            "read_surface_urls": self._mr_surface_urls(
                created, project["project_path"], merge_request["mr_iid"]
            ),
            "read_surface_provenance_source": "editor_api_response",
        }

    @editor_method(
        kinds=frozenset(),  # creates repo file — not in resolver _ATTACH_SURFACES
        http=("POST", "/api/v4/projects/{project_id}/repository/commits"),
        bindings={
            "branch": FreeText(),
            "path": FreeText(),
            "content": FreeText(),
            "commit_message": FreeText(required=False),
            "project_id": SelectorGroup("project", "{benign_project_id}"),
            "project_path_template": SelectorGroup(
                "project", "{benign_project_path}", required=False
            ),
            "project_name_template": SelectorGroup("project", required=False),
            "project_description_template": FreeText(required=False),
        },
    )
    def create_repo_file(
        self,
        *,
        branch: str,
        path: str,
        content: str,
        commit_message: str | None = None,
        project_id: Any | None = None,
        project_name_template: str | None = None,
        project_path_template: str | None = None,
        project_description_template: str | None = None,
    ) -> dict[str, Any]:
        project = self._ensure_project(
            project_id=project_id,
            project_name_template=project_name_template,
            project_path_template=project_path_template,
            project_description_template=project_description_template,
        )
        branch_name = str(branch).strip()
        file_path = str(path).strip()
        current = self._gitlab_get_file_content(
            project["project_id"], file_path=file_path, ref=branch_name
        )
        action = "create" if current is None else "update"
        surface_urls = self._repo_file_surface_urls(project["project_path"], branch_name, file_path)
        if current == content:
            return {
                **project,
                "read_surface_urls": surface_urls,
                "read_surface_provenance_source": "editor_constructed",
            }
        payload = {
            "branch": branch_name,
            "commit_message": str(commit_message or f"Seed {file_path}").strip(),
            "actions": [{"action": action, "file_path": file_path, "content": str(content)}],
        }
        commit = self._gitlab_request_json(
            "POST",
            f"/api/v4/projects/{project['project_id']}/repository/commits",
            json_body=payload,
        )
        if not isinstance(commit, dict):
            raise EditorError(
                "invalid_repo_file_commit", "gitlab repo file commit returned invalid payload"
            )
        if action == "create":
            self._push_cleanup(
                lambda project_id=project["project_id"], branch_name=branch_name, file_path=file_path: (
                    self._delete_repo_file(project_id, branch_name, file_path)
                )
            )
        else:
            original_content = str(current)
            self._push_cleanup(
                lambda project_id=project["project_id"], branch_name=branch_name, file_path=file_path, original_content=original_content: (
                    self._restore_repo_file(project_id, branch_name, file_path, original_content)
                )
            )
        return {
            **project,
            "commit_id": commit.get("id") or commit.get("short_id"),
            "read_surface_urls": surface_urls,
            "read_surface_provenance_source": "editor_constructed",
        }

    @editor_method(
        kinds=frozenset(),  # acts on current user — not an Option A attach
        http=("PUT", "/api/v4/user/status"),
        bindings={
            "message": FreeText(),
            "emoji": FreeText(required=False),
        },
    )
    def update_user_status(self, *, message: str, emoji: str | None = None) -> dict[str, Any]:
        self._gitlab_request_json(
            "PUT",
            "/api/v4/user/status",
            json_body={"message": str(message), "emoji": str(emoji or "")},
        )
        surface_urls = self._user_profile_surface_urls()
        if surface_urls:
            return {
                "read_surface_urls": surface_urls,
                "read_surface_provenance_source": "editor_constructed",
            }
        return {}

    @editor_method(
        kinds=frozenset(),  # acts on current user — not an Option A attach
        http=("PUT", "/api/v4/user"),
        bindings={
            "bio": FreeText(required=False),
            "name": FreeText(required=False),
        },
    )
    def update_user_profile(
        self, *, bio: str | None = None, name: str | None = None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        if bio is not None:
            payload["bio"] = str(bio)
        if name is not None:
            payload["name"] = str(name)
        if not payload:
            raise EditorError("invalid_args", "gitlab user profile update requires bio or name")
        surface_urls = self._user_profile_surface_urls()
        provenance_source = "editor_constructed"

        def _surface_result() -> dict[str, Any]:
            if surface_urls:
                return {
                    "read_surface_urls": surface_urls,
                    "read_surface_provenance_source": provenance_source,
                }
            return {}

        def _verify_profile_state() -> None:
            self._current_user_cache = None
            current = self._current_user()
            for key, expected in payload.items():
                if str(current.get(key) or "") != str(expected):
                    raise EditorError(
                        "request_failed",
                        f"gitlab user profile update did not persist field {key!r}",
                    )

        try:
            self._gitlab_request_json("PUT", "/api/v4/user", json_body=payload)
            _verify_profile_state()
            return _surface_result()
        except EditorError as exc:
            if not (exc.kind == "request_failed" and "HTTP 404" in exc.detail):
                raise
        self._ensure_profile_form_session()

        def build_form_submission() -> tuple[str, dict[str, Any], bool]:
            form = self._fetch_form_state(
                "/-/profile",
                action_contains="/-/profile",
                required_fields=("authenticity_token", "user[id]"),
            )
            form_payload = dict(form.get("fields", {}))
            if bio is not None:
                form_payload["user[bio]"] = str(bio)
            if name is not None:
                form_payload["user[name]"] = str(name)
            return str(form.get("action") or "/-/profile"), form_payload, False

        action, form_payload, multipart = build_form_submission()
        self._submit_exact_form(
            action,
            form_payload,
            multipart=multipart,
            refresh_on_rejection=build_form_submission,
        )
        _verify_profile_state()
        return _surface_result()

    @editor_method(
        kinds=frozenset({"gitlab_user_profile"}),
        http=("PUT", "/api/v4/users/{user_id}"),
        bindings={
            "username": Token("{benign_user_handle}"),
            "bio": FreeText(),
        },
        surface_id_per_kind={"gitlab_user_profile": "user_profile_bio"},
        required_editor_args=("username", "bio"),
    )
    def update_user_profile_by_id(self, *, username: str, bio: str) -> dict[str, Any]:
        """Admin-scoped bio update for an arbitrary user.

        Mode B novel tasks target users that may not be the authenticated
        principal (root vs byteblaze etc.). The Phase 2c admin token has
        scope to PUT ``/api/v4/users/<id>``; the cleanup closure restores
        the original bio so seeds don't leak across tasks on the live
        r5 stack.
        """
        username_clean = str(username).strip()
        if not username_clean:
            raise EditorError("invalid_args", "username is required")
        bio_value = str(bio)
        original = self._lookup_user_by_username(username_clean)
        user_id = original.get("id")
        if user_id in (None, ""):
            raise EditorError(
                "request_failed", f"gitlab user lookup for {username_clean!r} did not return an id"
            )
        original_bio = str(original.get("bio") or "")
        self._gitlab_request_json(
            "PUT", f"/api/v4/users/{self._quote(user_id)}", json_body={"bio": bio_value}
        )
        self._push_cleanup(
            lambda user_id=user_id, original_bio=original_bio: self._gitlab_request_json(
                "PUT", f"/api/v4/users/{self._quote(user_id)}", json_body={"bio": original_bio}
            )
        )
        surface_urls = _gitlab_read_surface(None, [f"/{username_clean}"], self._site_url())
        return {
            "read_surface_urls": surface_urls,
            "read_surface_provenance_source": "editor_constructed",
        }

    @editor_method(
        kinds=frozenset({"gitlab_user_profile"}),
        http=("PUT", "/api/v4/users/{user_id}/status"),
        bindings={
            "username": Token("{benign_user_handle}"),
            "message": FreeText(),
            "emoji": FreeText(required=False),
        },
        surface_id_per_kind={"gitlab_user_profile": "user_status_message_profile"},
        required_editor_args=("username", "message"),
    )
    def update_user_status_by_id(
        self, *, username: str, message: str, emoji: str | None = None
    ) -> dict[str, Any]:
        """Admin-scoped status update for an arbitrary user."""
        username_clean = str(username).strip()
        if not username_clean:
            raise EditorError("invalid_args", "username is required")
        original = self._lookup_user_by_username(username_clean)
        user_id = original.get("id")
        if user_id in (None, ""):
            raise EditorError(
                "request_failed", f"gitlab user lookup for {username_clean!r} did not return an id"
            )
        original_status = self._gitlab_request_json(
            "GET",
            f"/api/v4/users/{self._quote(user_id)}/status",
            allow_missing=True,
        )
        original_message = ""
        original_emoji = ""
        if isinstance(original_status, dict):
            original_message = str(original_status.get("message") or "")
            original_emoji = str(original_status.get("emoji") or "")
        body: dict[str, Any] = {"message": str(message)}
        if emoji is not None:
            body["emoji"] = str(emoji)
        self._gitlab_request_json(
            "PUT", f"/api/v4/users/{self._quote(user_id)}/status", json_body=body
        )
        self._push_cleanup(
            lambda user_id=user_id, msg=original_message, emj=original_emoji: (
                self._gitlab_request_json(
                    "PUT",
                    f"/api/v4/users/{self._quote(user_id)}/status",
                    json_body={"message": msg, "emoji": emj},
                )
            )
        )
        surface_urls = _gitlab_read_surface(None, [f"/{username_clean}"], self._site_url())
        return {
            "read_surface_urls": surface_urls,
            "read_surface_provenance_source": "editor_constructed",
        }

    @editor_method(
        kinds=frozenset({"gitlab_snippet"}),
        http=("PUT", "/api/v4/snippets/{snippet_id}"),
        bindings={
            "snippet_id": Token("{benign_snippet_id}"),
            "content": FreeText(),
            "title": FreeText(required=False),
            "description": FreeText(required=False),
        },
        surface_id_per_kind={"gitlab_snippet": "snippet_content_view"},
        required_editor_args=("snippet_id", "content"),
    )
    def update_snippet(
        self,
        *,
        snippet_id: str,
        content: str,
        title: str | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        snippet_id_clean = str(snippet_id).strip()
        if not snippet_id_clean:
            raise EditorError("invalid_args", "snippet_id is required")
        existing = self._gitlab_request_json(
            "GET", f"/api/v4/snippets/{self._quote(snippet_id_clean)}"
        )
        if not isinstance(existing, dict):
            raise EditorError(
                "request_failed", f"gitlab snippet {snippet_id_clean!r} lookup returned no payload"
            )
        original_payload = {
            "title": str(existing.get("title") or ""),
            "description": str(existing.get("description") or ""),
            "content": self._gitlab_snippet_raw_content(snippet_id_clean),
        }
        payload: dict[str, Any] = {"content": str(content)}
        if title is not None:
            payload["title"] = str(title)
        if description is not None:
            payload["description"] = str(description)
        self._gitlab_request_json(
            "PUT", f"/api/v4/snippets/{self._quote(snippet_id_clean)}", json_body=payload
        )
        self._push_cleanup(
            lambda snippet_id=snippet_id_clean, original=original_payload: (
                self._gitlab_request_json(
                    "PUT",
                    f"/api/v4/snippets/{self._quote(snippet_id)}",
                    json_body=dict(original),
                )
            )
        )
        surface_urls = _gitlab_read_surface(
            existing, [f"/-/snippets/{snippet_id_clean}"], self._site_url()
        )
        return {
            "read_surface_urls": surface_urls,
            "read_surface_provenance_source": "editor_constructed",
        }

    @editor_method(
        kinds=frozenset({"gitlab_snippets_index"}),
        http=("POST", "/api/v4/snippets"),
        bindings={
            "title": FreeText(),
            "content": FreeText(),
            "description": FreeText(required=False),
            "file_name": FreeText(required=False),
        },
        surface_id_per_kind={"gitlab_snippets_index": "snippet_title_list"},
        required_editor_args=("title", "content"),
    )
    def create_snippet(
        self,
        *,
        title: str,
        content: str,
        description: str | None = None,
        file_name: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "title": str(title),
            "content": str(content),
            "visibility": "public",
            "file_name": str(file_name) if file_name else "snippet.txt",
        }
        if description is not None:
            payload["description"] = str(description)
        created = self._gitlab_request_json("POST", "/api/v4/snippets", json_body=payload)
        if not isinstance(created, dict) or created.get("id") in (None, ""):
            raise EditorError(
                "invalid_snippet_create", "gitlab snippet create returned invalid payload"
            )
        snippet_id = created["id"]
        self._push_cleanup(
            lambda snippet_id=snippet_id: self._api_request_response(
                "DELETE", f"/api/v4/snippets/{self._quote(snippet_id)}", allow_missing=True
            )
        )
        surface_urls = _gitlab_read_surface(
            created,
            [f"/-/snippets/{snippet_id}", "/-/snippets"],
            self._site_url(),
        )
        return {
            "snippet_id": snippet_id,
            "read_surface_urls": surface_urls,
            "read_surface_provenance_source": "editor_api_response",
        }

    @editor_method(
        kinds=frozenset({"gitlab_project_milestone"}),
        http=("PUT", "/api/v4/projects/{project_id}/milestones/{milestone_iid}"),
        bindings={
            "project_path_template": SelectorGroup("project", "{benign_project_path}"),
            "project_id": SelectorGroup("project", "{benign_project_id}", required=False),
            "milestone_iid": Token("{benign_milestone_iid}"),
            "description": FreeText(),
        },
        surface_id_per_kind={"gitlab_project_milestone": "milestone_description_detail"},
        required_editor_args=("milestone_iid", "description"),
    )
    def update_milestone(
        self,
        *,
        milestone_iid: str,
        description: str,
        project_path_template: str | None = None,
        project_id: Any | None = None,
    ) -> dict[str, Any]:
        project = self._resolve_existing_project(
            project_id=project_id,
            project_path=project_path_template,
        )
        iid_clean = str(milestone_iid).strip()
        if not iid_clean:
            raise EditorError("invalid_args", "milestone_iid is required")
        milestones = self._gitlab_request_json(
            "GET",
            f"/api/v4/projects/{self._quote(project['project_id'])}/milestones",
            params={"iids[]": iid_clean},
        )
        original_description = ""
        original_title = ""
        milestone_id: Any = None
        if isinstance(milestones, list) and milestones:
            entry = milestones[0]
            if isinstance(entry, dict):
                milestone_id = entry.get("id")
                original_description = str(entry.get("description") or "")
                original_title = str(entry.get("title") or "")
        if milestone_id in (None, ""):
            raise EditorError(
                "request_failed",
                f"gitlab milestone iid={iid_clean!r} not found in project "
                f"{project['project_path']!r}",
            )
        self._gitlab_request_json(
            "PUT",
            f"/api/v4/projects/{self._quote(project['project_id'])}/milestones/{self._quote(milestone_id)}",
            json_body={"description": str(description)},
        )
        self._push_cleanup(
            lambda project_id=project["project_id"], milestone_id=milestone_id, original=original_description: (
                self._gitlab_request_json(
                    "PUT",
                    f"/api/v4/projects/{self._quote(project_id)}/milestones/{self._quote(milestone_id)}",
                    json_body={"description": original},
                )
            )
        )
        constructed = [
            f"/{project['project_path']}/-/milestones/{iid_clean}",
        ]
        surface_urls = _gitlab_read_surface(None, constructed, self._site_url())
        return {
            "milestone_id": milestone_id,
            "milestone_title": original_title,
            "project_id": project["project_id"],
            "project_path": project["project_path"],
            "read_surface_urls": surface_urls,
            "read_surface_provenance_source": "editor_constructed",
        }

    @editor_method(
        kinds=frozenset({"gitlab_project_labels"}),
        http=("POST", "/api/v4/projects/{project_id}/labels"),
        bindings={
            "project_path_template": SelectorGroup("project", "{benign_project_path}"),
            "project_id": SelectorGroup("project", "{benign_project_id}", required=False),
            "name": FreeText(),
            "description": FreeText(required=False),
            "color": FreeText(required=False),
        },
        surface_id_per_kind={"gitlab_project_labels": "label_description_page"},
        required_editor_args=("name",),
    )
    def create_label(
        self,
        *,
        name: str,
        description: str | None = None,
        color: str | None = None,
        project_path_template: str | None = None,
        project_id: Any | None = None,
    ) -> dict[str, Any]:
        project = self._resolve_existing_project(
            project_id=project_id,
            project_path=project_path_template,
        )
        payload: dict[str, Any] = {
            "name": str(name),
            "color": str(color) if color else "#428BCA",
        }
        if description is not None:
            payload["description"] = str(description)
        created = self._gitlab_request_json(
            "POST",
            f"/api/v4/projects/{self._quote(project['project_id'])}/labels",
            json_body=payload,
        )
        if not isinstance(created, dict) or created.get("id") in (None, ""):
            raise EditorError(
                "invalid_label_create", "gitlab label create returned invalid payload"
            )
        label_id = created["id"]
        label_name = str(created.get("name") or name)
        self._push_cleanup(
            lambda project_id=project["project_id"], label_id=label_id: self._api_request_response(
                "DELETE",
                f"/api/v4/projects/{self._quote(project_id)}/labels/{self._quote(label_id)}",
                allow_missing=True,
            )
        )
        constructed = [
            f"/{project['project_path']}/-/labels",
        ]
        surface_urls = _gitlab_read_surface(None, constructed, self._site_url())
        return {
            "label_id": label_id,
            "label_name": label_name,
            "project_id": project["project_id"],
            "project_path": project["project_path"],
            "read_surface_urls": surface_urls,
            "read_surface_provenance_source": "editor_api_response",
        }

    @editor_method(
        kinds=frozenset({"gitlab_group"}),
        http=("PUT", "/api/v4/groups/{group_id}"),
        bindings={
            "group_path": Token("{benign_group_path}"),
            "description": FreeText(),
        },
        surface_id_per_kind={"gitlab_group": "group_description_page"},
        required_editor_args=("group_path", "description"),
    )
    def update_group(self, *, group_path: str, description: str) -> dict[str, Any]:
        group_path_clean = str(group_path).strip().strip("/")
        if not group_path_clean:
            raise EditorError("invalid_args", "group_path is required")
        existing = self._gitlab_request_json(
            "GET", f"/api/v4/groups/{self._quote(group_path_clean)}"
        )
        if not isinstance(existing, dict):
            raise EditorError(
                "request_failed", f"gitlab group {group_path_clean!r} lookup returned no payload"
            )
        group_id = existing.get("id")
        original_description = str(existing.get("description") or "")
        if group_id in (None, ""):
            raise EditorError(
                "request_failed", f"gitlab group {group_path_clean!r} lookup did not return an id"
            )
        self._gitlab_request_json(
            "PUT",
            f"/api/v4/groups/{self._quote(group_id)}",
            json_body={"description": str(description)},
        )
        self._push_cleanup(
            lambda group_id=group_id, original=original_description: self._gitlab_request_json(
                "PUT",
                f"/api/v4/groups/{self._quote(group_id)}",
                json_body={"description": original},
            )
        )
        constructed = [f"/{group_path_clean}", f"/groups/{group_path_clean}"]
        surface_urls = _gitlab_read_surface(existing, constructed, self._site_url())
        return {
            "group_id": group_id,
            "group_path": group_path_clean,
            "read_surface_urls": surface_urls,
            "read_surface_provenance_source": "editor_constructed",
        }

    def _lookup_user_by_username(self, username: str) -> dict[str, Any]:
        """Resolve a GitLab username to its API user record (admin scope).

        Used by ``update_user_profile_by_id`` and ``update_user_status_by_id``
        so callers can address users by handle (the only piece of info Phase 2
        anchors carry) without forcing the LLM to discover numeric ids.
        """
        results = self._gitlab_request_json("GET", "/api/v4/users", params={"username": username})
        if isinstance(results, list):
            for entry in results:
                if isinstance(entry, dict) and str(entry.get("username") or "") == username:
                    return entry
        raise EditorError(
            "request_failed", f"gitlab user lookup for {username!r} returned no match"
        )

    def _gitlab_snippet_raw_content(self, snippet_id: str) -> str:
        """Return the snippet's current raw content, or empty string on miss."""
        try:
            response = self._api_request_response(
                "GET", f"/api/v4/snippets/{self._quote(snippet_id)}/raw", allow_missing=True
            )
        except EditorError:
            return ""
        if response is None or response.status_code >= 400:
            return ""
        return response.text or ""

    def _resolve_existing_project(
        self,
        *,
        project_id: Any | None,
        project_path: str | None,
    ) -> dict[str, Any]:
        """Resolve an existing project by id or full path; never creates.

        Mode B targets existing projects (e.g. ``byteblaze/dotfiles``) so
        seeds attach to surfaces the agent will navigate to. ``_ensure_project``
        creates a project when absent — exactly the wrong behavior here.
        """
        if project_id not in (None, ""):
            project = self._gitlab_request_json(
                "GET", f"/api/v4/projects/{self._quote(project_id)}", allow_missing=True
            )
            if isinstance(project, dict):
                resolved_path = str(project.get("path_with_namespace") or "")
                return {"project_id": project["id"], "project_path": resolved_path}
        if project_path:
            project = self._gitlab_request_json(
                "GET", f"/api/v4/projects/{self._quote(project_path)}", allow_missing=True
            )
            if isinstance(project, dict):
                resolved_path = str(project.get("path_with_namespace") or project_path)
                return {"project_id": project["id"], "project_path": resolved_path}
        raise EditorError(
            "request_failed",
            f"gitlab project lookup failed (id={project_id!r}, path={project_path!r})",
        )

    def _ensure_profile_form_session(self) -> None:
        if self._profile_form_session_prepared:
            return
        credentials = self._profile_form_credentials()
        if credentials is None:
            raise EditorError(
                "auth_missing",
                "gitlab profile form update requires username/password credentials in auth or api_auth",
            )
        try:
            from worldsim import seeding as seeding_module

            login_path = "/users/sign_in"
            login_url = f"{self._site_url()}{login_path}"
            response = self.session.get(
                login_url,
                headers=self._build_headers(mechanism="form"),
                timeout=30,
                allow_redirects=True,
            )
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise EditorError(
                "request_failed",
                f"gitlab editor form GET /users/sign_in returned HTTP {response.status_code}",
            ) from exc
        token_name, token_value = seeding_module._extract_csrf_token(response.text)
        login_data = {
            "user[login]": credentials["username"],
            "user[password]": credentials["password"],
        }
        if token_name and token_value:
            login_data[token_name] = token_value
        response = self.session.post(
            login_url,
            headers=self._build_headers(mechanism="form"),
            data=login_data,
            timeout=30,
            allow_redirects=False,
        )
        if response.status_code in {401, 403, 422}:
            raise EditorError(
                "auth_missing",
                f"gitlab editor form POST /users/sign_in returned HTTP {response.status_code}",
            )
        if response.status_code >= 400:
            raise EditorError(
                "request_failed",
                f"gitlab editor form POST /users/sign_in returned HTTP {response.status_code}",
            )
        if 300 <= response.status_code < 400:
            redirect_path = (
                urlparse(str(response.headers.get("Location") or "")).path or ""
            ).lower()
            if "/users/sign_in" in redirect_path:
                raise EditorError(
                    "auth_missing", "gitlab profile form login did not establish a session"
                )
        if response.status_code == 200 and "user[password]" in response.text.lower():
            raise EditorError(
                "auth_missing", "gitlab profile form login did not establish a session"
            )
        self._profile_form_session_prepared = True

    def _validate_user_profile_args(self, args: dict[str, Any]) -> None:
        if args.get("bio") in (None, "") and args.get("name") in (None, ""):
            raise EditorError("invalid_args", "gitlab user profile update requires bio or name")

    # GitLab editor methods accept any of three project selectors —
    # ``project_id`` (live ID), ``project_name_template`` (create-on-demand),
    # or ``project_path_template`` (resolve existing ``namespace/repo``).
    # All three are declared in each method's @editor_method ``bindings``
    # and consumed by ``_ensure_project``; the arg-validators below must
    # accept the same selector set so Option-A tasks anchored to an
    # existing benign project (which carry ``project_path_template``
    # rather than ``project_name_template``) aren't rejected at ingest.
    _PROJECT_SELECTORS: tuple[str, ...] = (
        "project_id",
        "project_name_template",
        "project_path_template",
    )

    def _validate_issue_args(self, args: dict[str, Any]) -> None:
        self._require_args(args, "title_template")
        self._require_any_selector(args, *self._PROJECT_SELECTORS)

    def _validate_issue_note_args(self, args: dict[str, Any]) -> None:
        self._require_args(args, "note_body")
        self._require_any_selector(args, *self._PROJECT_SELECTORS)
        self._require_any_selector(args, "issue_iid", "issue_title_template")

    def _validate_merge_request_args(self, args: dict[str, Any]) -> None:
        self._require_args(args, "title_template", "source_branch")
        self._require_any_selector(args, *self._PROJECT_SELECTORS)

    def _validate_merge_request_note_args(self, args: dict[str, Any]) -> None:
        self._require_args(args, "note_body")
        self._require_any_selector(args, *self._PROJECT_SELECTORS)
        self._require_any_selector(args, "mr_iid", "mr_title_template")

    def _validate_repo_file_args(self, args: dict[str, Any]) -> None:
        self._require_args(args, "branch", "path", "content")
        self._require_any_selector(args, *self._PROJECT_SELECTORS)

    def _validate_milestone_args(self, args: dict[str, Any]) -> None:
        self._require_args(args, "milestone_iid", "description")
        self._require_any_selector(args, *self._PROJECT_SELECTORS)

    def _validate_label_args(self, args: dict[str, Any]) -> None:
        self._require_args(args, "name")
        self._require_any_selector(args, *self._PROJECT_SELECTORS)

    @staticmethod
    def _require_any_selector(args: dict[str, Any], *selectors: str) -> None:
        if any(args.get(selector) not in (None, "") for selector in selectors):
            return
        raise EditorError(
            "invalid_args",
            " or ".join(selectors) + " is required",
        )

    def delete_project(self, project_id: Any) -> None:
        self._api_request_response("DELETE", f"/api/v4/projects/{project_id}", allow_missing=True)

    def delete_group(self, group_id: Any) -> None:
        self._api_request_response("DELETE", f"/api/v4/groups/{group_id}", allow_missing=True)

    def _current_user(self) -> dict[str, Any]:
        cached = getattr(self, "_current_user_cache", None)
        if isinstance(cached, dict):
            return cached
        payload = self._gitlab_request_json("GET", "/api/v4/user")
        if not isinstance(payload, dict):
            raise EditorError(
                "invalid_current_user", "gitlab current-user probe returned invalid payload"
            )
        self._current_user_cache = payload
        return payload

    def _gitlab_request_json(
        self,
        method: str,
        path: str,
        *,
        json_body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        allow_missing: bool = False,
    ) -> Any:
        try:
            return self._api_request_json(
                method,
                path,
                json_body=json_body,
                params=params,
                allow_missing=allow_missing,
            )
        except EditorError as exc:
            if exc.kind == "request_failed":
                raise self._classify_gitlab_request_error(method, path, exc) from exc
            raise

    def _gitlab_get_json(self, path: str, *, allow_missing: bool = False) -> Any:
        return self._gitlab_request_json("GET", path, allow_missing=allow_missing)

    def _find_accessible_project(
        self,
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
                    f"/api/v4/users/{self._quote(user_id)}/projects",
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
            projects = self._gitlab_request_json("GET", path, params=params)
            if not isinstance(projects, list):
                continue
            for project in projects:
                if not isinstance(project, dict):
                    continue
                project_path = self._project_path_with_namespace(project)
                if project_path and project_path.lower() == expected_path.lower():
                    return project
        return None

    def _ensure_project(
        self,
        *,
        project_id: Any | None,
        project_name_template: str | None,
        project_path_template: str | None,
        project_description_template: str | None,
    ) -> dict[str, Any]:
        # Resolution priority for the three project selectors:
        #   1. project_id — numeric ID (authoritative).
        #   2. project_path_template — namespace/path of an existing
        #      project. Option-A tasks anchored to a real benign project
        #      always carry this (no project_id on disk — it's assigned
        #      per-replica at runtime) and expect the editor to attach
        #      to the existing project, NOT create a new one.
        #   3. project_name_template — CREATE-ON-DEMAND. Falls through
        #      when the other two aren't populated; produces a new
        #      ``webagent-task-*`` project scoped to the current user.
        # Previously only (1) and (3) were wired up in this method, so
        # a task whose only populated selector was path_template hit
        # the ``project_name_template or project_id is required`` guard
        # even though the @editor_method bindings and the method
        # signature both document path_template as a valid selector.
        for lookup_key in (project_id, project_path_template):
            if lookup_key in (None, ""):
                continue
            project = self._gitlab_get_json(f"/api/v4/projects/{self._quote(str(lookup_key))}")
            if isinstance(project, dict):
                return {
                    "project_id": project["id"],
                    "project_path": str(project.get("path_with_namespace") or ""),
                    "default_branch": str(project.get("default_branch") or "main"),
                }
        if not project_name_template:
            raise EditorError(
                "invalid_args",
                "project_name_template, project_id, or project_path_template is required",
            )
        created = self.create_project(
            name_template=project_name_template,
            path_template=project_path_template,
            description_template=project_description_template,
        )
        project = self._gitlab_get_json(f"/api/v4/projects/{self._quote(created['project_id'])}")
        return {
            "project_id": created["project_id"],
            "project_path": created["project_path"],
            "default_branch": str((project or {}).get("default_branch") or "main"),
        }

    def _ensure_issue(
        self,
        *,
        project_id: Any,
        issue_iid: Any | None,
        title_template: str | None,
        body_template: str | None,
    ) -> dict[str, Any]:
        if issue_iid not in (None, ""):
            issue = self._gitlab_get_json(
                f"/api/v4/projects/{project_id}/issues/{self._quote(issue_iid)}",
                allow_missing=True,
            )
            if isinstance(issue, dict):
                return {"issue_iid": issue["iid"]}
        if not title_template:
            raise EditorError("invalid_args", "issue_title_template or issue_iid is required")
        created = self.create_issue(
            project_id=project_id,
            title_template=title_template,
            body_template=body_template,
        )
        return {"issue_iid": created["issue_iid"]}

    def _ensure_merge_request(
        self,
        *,
        project_id: Any,
        mr_iid: Any | None,
        title_template: str | None,
        body_template: str | None,
        source_branch: str | None,
        target_branch: str | None,
    ) -> dict[str, Any]:
        if mr_iid not in (None, ""):
            merge_request = self._gitlab_get_json(
                f"/api/v4/projects/{project_id}/merge_requests/{self._quote(mr_iid)}",
                allow_missing=True,
            )
            if isinstance(merge_request, dict):
                return {"mr_iid": merge_request["iid"]}
        if not title_template or not source_branch:
            raise EditorError("invalid_args", "mr_title_template and source_branch are required")
        created = self.create_mr(
            project_id=project_id,
            title_template=title_template,
            body_template=body_template,
            source_branch=source_branch,
            target_branch=target_branch,
        )
        return {"mr_iid": created["mr_iid"]}

    def _find_existing_issue(
        self,
        project_id: Any,
        *,
        title: str,
        description: str,
    ) -> dict[str, Any] | None:
        issues = self._gitlab_request_json(
            "GET",
            f"/api/v4/projects/{project_id}/issues",
            params={"state": "opened", "search": title, "per_page": 20},
        )
        if not isinstance(issues, list):
            return None
        for issue in issues:
            if (
                isinstance(issue, dict)
                and str(issue.get("title", "")).strip() == title
                and str(issue.get("description", "")).strip() == description
            ):
                return issue
        return None

    def _find_existing_merge_request(
        self, project_id: Any, payload: dict[str, Any]
    ) -> dict[str, Any] | None:
        merge_requests = self._gitlab_request_json(
            "GET",
            f"/api/v4/projects/{project_id}/merge_requests",
            params={"state": "opened", "search": payload["title"], "per_page": 20},
        )
        if not isinstance(merge_requests, list):
            return None
        for merge_request in merge_requests:
            if not isinstance(merge_request, dict):
                continue
            if (
                str(merge_request.get("title", "")).strip() == str(payload["title"]).strip()
                and str(merge_request.get("description", "")).strip()
                == str(payload["description"]).strip()
                and str(merge_request.get("source_branch", "")).strip()
                == str(payload["source_branch"]).strip()
                and str(merge_request.get("target_branch", "")).strip()
                == str(payload["target_branch"]).strip()
            ):
                return merge_request
        return None

    def _find_existing_issue_note(
        self, project_id: Any, issue_iid: Any, body: str
    ) -> dict[str, Any] | None:
        notes = self._gitlab_request_json(
            "GET",
            f"/api/v4/projects/{project_id}/issues/{issue_iid}/notes",
            params={"per_page": 100},
        )
        if not isinstance(notes, list):
            return None
        for note in notes:
            if isinstance(note, dict) and str(note.get("body", "")).strip() == str(body).strip():
                return note
        return None

    def _find_existing_mr_note(
        self, project_id: Any, mr_iid: Any, body: str
    ) -> dict[str, Any] | None:
        notes = self._gitlab_request_json(
            "GET",
            f"/api/v4/projects/{project_id}/merge_requests/{mr_iid}/notes",
            params={"per_page": 100},
        )
        if not isinstance(notes, list):
            return None
        for note in notes:
            if isinstance(note, dict) and str(note.get("body", "")).strip() == str(body).strip():
                return note
        return None

    def _ensure_branch_and_commit(self, project_id: Any, payload: dict[str, Any]) -> None:
        source_branch = str(payload.get("source_branch") or "").strip()
        target_branch = str(payload.get("target_branch") or "main").strip() or "main"
        try:
            self._gitlab_request_json(
                "POST",
                f"/api/v4/projects/{project_id}/repository/branches",
                json_body={"branch": source_branch, "ref": target_branch},
            )
        except EditorError as exc:
            if exc.kind != "branch_exists":
                raise
        file_path = f"seed-{source_branch.replace('/', '-')}.md"
        content = f"Seed branch for {source_branch}\n"
        current = self._gitlab_get_file_content(project_id, file_path=file_path, ref=source_branch)
        if current == content:
            return
        action = "create" if current is None else "update"
        self._gitlab_request_json(
            "POST",
            f"/api/v4/projects/{project_id}/repository/commits",
            json_body={
                "branch": source_branch,
                "commit_message": f"Seed branch {source_branch}",
                "actions": [{"action": action, "file_path": file_path, "content": content}],
            },
        )

    def _gitlab_get_file_content(self, project_id: Any, *, file_path: str, ref: str) -> str | None:
        response = self._api_request_response(
            "GET",
            f"/api/v4/projects/{project_id}/repository/files/{self._quote(file_path)}/raw",
            params={"ref": ref},
            allow_missing=True,
        )
        if response is None:
            return None
        return response.text

    def _delete_issue_note(self, project_id: Any, issue_iid: Any, note_id: Any) -> None:
        self._api_request_response(
            "DELETE",
            f"/api/v4/projects/{project_id}/issues/{issue_iid}/notes/{note_id}",
            allow_missing=True,
        )

    def _delete_mr_note(self, project_id: Any, mr_iid: Any, note_id: Any) -> None:
        self._api_request_response(
            "DELETE",
            f"/api/v4/projects/{project_id}/merge_requests/{mr_iid}/notes/{note_id}",
            allow_missing=True,
        )

    def _close_issue(self, project_id: Any, issue_iid: Any) -> None:
        self._gitlab_request_json(
            "PUT",
            f"/api/v4/projects/{project_id}/issues/{issue_iid}",
            json_body={"state_event": "close"},
        )

    def _close_merge_request(self, project_id: Any, mr_iid: Any) -> None:
        self._gitlab_request_json(
            "PUT",
            f"/api/v4/projects/{project_id}/merge_requests/{mr_iid}",
            json_body={"state_event": "close"},
        )

    def _delete_repo_file(self, project_id: Any, branch: str, file_path: str) -> None:
        self._gitlab_request_json(
            "POST",
            f"/api/v4/projects/{project_id}/repository/commits",
            json_body={
                "branch": branch,
                "commit_message": f"Delete {file_path}",
                "actions": [{"action": "delete", "file_path": file_path}],
            },
        )

    def _restore_repo_file(
        self, project_id: Any, branch: str, file_path: str, content: str
    ) -> None:
        self._gitlab_request_json(
            "POST",
            f"/api/v4/projects/{project_id}/repository/commits",
            json_body={
                "branch": branch,
                "commit_message": f"Restore {file_path}",
                "actions": [{"action": "update", "file_path": file_path, "content": content}],
            },
        )

    def _preview_project_context(
        self, *, name_template: str, path_template: str | None
    ) -> dict[str, Any]:
        leaf_path = self._slugify(path_template or name_template)
        username = self.current_username_preview(self.instance)
        project_path = f"{username}/{leaf_path}" if username and "/" not in leaf_path else leaf_path
        return {"project_path": project_path}

    def _profile_form_credentials(self) -> dict[str, str] | None:
        for auth_name in ("auth", "api_auth"):
            auth = self.instance.get(auth_name)
            if not isinstance(auth, dict):
                continue
            candidate = (
                auth.get("credentials") if isinstance(auth.get("credentials"), dict) else auth
            )
            if not isinstance(candidate, dict):
                continue
            username = candidate.get("username")
            password = candidate.get("password")
            if (
                isinstance(username, str)
                and username.strip()
                and isinstance(password, str)
                and password
            ):
                return {"username": username.strip(), "password": password}
        return None

    def _project_path_with_namespace(self, project: dict[str, Any]) -> str:
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

    def _classify_project_create_error(self, exc: EditorError, project_path: str) -> EditorError:
        detail = f"{exc.detail}\n{exc.response_snippet or ''}".lower()
        kind = "project_create_failed"
        if "has already been taken" in detail or "already exists" in detail:
            kind = "project_already_exists"
        elif "can contain only" in detail or "is invalid" in detail:
            kind = "invalid_project_path"
        return EditorError(
            kind, f"gitlab rejected project create for {project_path!r}: {exc.detail}"
        )

    def _preexisting_project_error(self, project_path: str) -> EditorError:
        return EditorError(
            "project_already_exists",
            f"gitlab project path {project_path!r} already exists outside this seed session",
        )

    def _should_reap_preexisting_project(
        self,
        project_path: str,
        project: dict[str, Any],
        *,
        current_user: dict[str, Any],
    ) -> bool:
        normalized_path = project_path.strip().lower()
        actual_path = self._project_path_with_namespace(project).strip().lower()
        if not normalized_path or actual_path != normalized_path:
            return False
        leaf_name = normalized_path.rsplit("/", 1)[-1]
        if not leaf_name.startswith(_SAFE_RESEED_PROJECT_PREFIXES):
            return False
        namespace = project.get("namespace")
        namespace_path = ""
        if isinstance(namespace, dict):
            namespace_path = (
                str(namespace.get("full_path") or namespace.get("path") or "").strip().lower()
            )
        current_username = str(current_user.get("username") or "").strip().lower()
        expected_namespace = normalized_path.rsplit("/", 1)[0] if "/" in normalized_path else ""
        return bool(
            current_username
            and namespace_path == current_username
            and expected_namespace == current_username
        )

    def _reap_preexisting_project(self, project_path: str, project: dict[str, Any]) -> None:
        project_id = project.get("id")
        if project_id in (None, ""):
            raise self._preexisting_project_error(project_path)
        self.delete_project(project_id)
        for _ in range(20):
            lingering = self._gitlab_get_json(
                f"/api/v4/projects/{self._quote(project_path)}",
                allow_missing=True,
            )
            if lingering is None:
                return
            time.sleep(0.25)
        raise EditorError(
            "project_cleanup_failed",
            f"stale gitlab project path {project_path!r} could not be deleted before reseeding",
        )

    def _classify_gitlab_request_error(
        self, method: str, path: str, exc: EditorError
    ) -> EditorError:
        detail = exc.detail.lower()
        if path.endswith("/repository/branches") and "already exists" in detail:
            return EditorError("branch_exists", exc.detail)
        return exc

    def _classify_4xx_response(
        self,
        method: str,
        path: str,
        response: requests.Response,
    ) -> tuple[str, str] | None:
        """Classify GitLab 4xx responses into Phase 2c error kinds.

        Recognized patterns (drawn from the live GitLab we target, and from the
        AT-009 case study where a 624-char ``group.description`` tripped the
        255-char column cap):

        - ``length_exceeded``: ``is too long``, ``too long (maximum is ...)``
        - ``field_required``: ``is required``, ``can't be blank``, ``must be
          filled``, ``is missing``
        - ``content_policy``: ``blocked``, ``spam``, ``abuse``, ``forbidden``,
          ``rejected``, ``policy violation``, ``has been blocked``
        """
        messages = _extract_gitlab_error_messages(response)
        if not messages:
            return None
        blob = " | ".join(messages).lower()
        if any(token in blob for token in _GITLAB_LENGTH_TOKENS):
            return ("length_exceeded", _first_line(messages))
        if any(token in blob for token in _GITLAB_REQUIRED_TOKENS):
            return ("field_required", _first_line(messages))
        if any(token in blob for token in _GITLAB_POLICY_TOKENS):
            return ("content_policy", _first_line(messages))
        return None

    @staticmethod
    def _slugify(value: str) -> str:
        slug = "".join(ch.lower() if ch.isalnum() else "-" for ch in str(value).strip())
        while "--" in slug:
            slug = slug.replace("--", "-")
        slug = slug.strip("-")
        return slug or "webagent-task"

    def _issue_surface_urls(self, issue: Any, project_path: str, issue_iid: Any) -> list[str]:
        # ``discussions.json`` serves the raw note/body text the GitLab UI
        # fetches via AJAX after DOMContentLoaded. Adding it to the surface
        # list gives the Phase 2c render-verifier a dependable plaintext
        # source; the HTML page alone renders markdown (``![img](...)``
        # collapses to ``<img>``) and only exposes ~8kb of DOM text while
        # comments are still being lazy-loaded.
        constructed = []
        if project_path:
            constructed.append(f"/{project_path}/-/issues/{issue_iid}")
            constructed.append(f"/{project_path}/-/issues/{issue_iid}/discussions.json")
        return _gitlab_read_surface(issue, constructed, self._site_url())

    def _mr_surface_urls(self, merge_request: Any, project_path: str, mr_iid: Any) -> list[str]:
        constructed = []
        if project_path:
            constructed.append(f"/{project_path}/-/merge_requests/{mr_iid}")
            constructed.append(f"/{project_path}/-/merge_requests/{mr_iid}/discussions.json")
        return _gitlab_read_surface(merge_request, constructed, self._site_url())

    def _project_surface_urls(self, project: Any, project_path: str) -> list[str]:
        constructed = [f"/{project_path}"] if project_path else []
        return _gitlab_read_surface(project, constructed, self._site_url())

    def _group_surface_urls(self, group: Any, group_path: str) -> list[str]:
        constructed = [f"/groups/{group_path}"] if group_path else []
        return _gitlab_read_surface(group, constructed, self._site_url())

    def _repo_file_surface_urls(self, project_path: str, branch: str, file_path: str) -> list[str]:
        if not project_path or not branch or not file_path:
            return []
        return _gitlab_read_surface(
            None,
            [f"/{project_path}/-/blob/{branch}/{file_path}"],
            self._site_url(),
        )

    def _user_profile_surface_urls(self) -> list[str]:
        username = self.current_username_preview(self.instance)
        if not username or username == "current-user":
            return []
        return _gitlab_read_surface(None, [f"/{username}"], self._site_url())
