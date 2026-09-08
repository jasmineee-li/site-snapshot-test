"""Fixed, sampled readback for host-owned benchmark restoration.

The host restoration daemon proves that its configured container was recreated.
This module supplies the independent application-level check that the recreated
service still presents the same small, site-owned baseline.  It intentionally
does not claim database equivalence or research-state acceptance.

Only the existing GitLab and Reddit editor owners are used for requests.  A
missing sample, an unexpected response, or a changed sample is an unknown
restoration result and must block the surrounding work unit.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

import requests

from warp_taskgen.editors.gitlab import GitlabEditor
from warp_taskgen.editors.reddit import RedditEditor

_GITLAB_PROJECT_SAMPLE_SIZE = 3


class RestorationReadbackError(RuntimeError):
    """A fixed baseline/readback cannot establish restored application state."""

    def __init__(self, reason: str, detail: str = "") -> None:
        self.reason = reason
        self.detail = detail
        suffix = f": {detail}" if detail else ""
        super().__init__(f"restoration readback {reason}{suffix}")


@dataclass(frozen=True, slots=True)
class RestorationReadback:
    """In-memory baseline captured immediately after a successful recreation."""

    site_name: str
    site_url: str
    baseline: dict[str, Any]


def _instance_mapping(instance: object) -> Mapping[str, Any]:
    if isinstance(instance, Mapping):
        return instance
    model_dump = getattr(instance, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, Mapping):
            return dumped
    raise RestorationReadbackError("invalid_instance", "instance must be a mapping or model")


def _site_name(instance: Mapping[str, Any]) -> str:
    value = str(instance.get("site_name") or "").strip().lower()
    if value not in {"gitlab", "reddit"}:
        raise RestorationReadbackError("unsupported_site", value or "missing site_name")
    return value


def _site_url(instance: Mapping[str, Any]) -> str:
    value = str(instance.get("site_url") or "").strip()
    if not value:
        raise RestorationReadbackError("missing_site_url")
    return value.rstrip("/")


def _first_forum_name(task: Mapping[str, Any] | None) -> str | None:
    if not isinstance(task, Mapping):
        return None
    exposure = task.get("exposure_contract")
    if isinstance(exposure, Mapping):
        for key in ("benign_forum_name", "forum_name"):
            value = exposure.get(key)
            if isinstance(value, str) and value.strip() and "{" not in value:
                return value.strip()
        target = exposure.get("benign_target_resource")
        if isinstance(target, Mapping):
            value = target.get("forum_name")
            if isinstance(value, str) and value.strip() and "{" not in value:
                return value.strip()
    seed = task.get("adversarial_data_seed")
    calls = seed.get("editor_calls") if isinstance(seed, Mapping) else None
    if isinstance(calls, list):
        for call in calls:
            args = call.get("args") if isinstance(call, Mapping) else None
            value = args.get("forum_name") if isinstance(args, Mapping) else None
            if isinstance(value, str) and value.strip() and "{" not in value:
                return value.strip()
    return None


def _gitlab_snapshot(instance: Mapping[str, Any]) -> dict[str, Any]:
    with requests.Session() as session:
        editor = GitlabEditor(dict(instance), session)
        user = editor._current_user()
        if not isinstance(user, Mapping):
            raise RestorationReadbackError(
                "baseline_incomplete", "GitLab current user is not an object"
            )
        user_id = str(user.get("id") or "").strip()
        username = str(user.get("username") or "").strip()
        if not user_id or not username:
            raise RestorationReadbackError(
                "baseline_incomplete", "GitLab identity lacks id or username"
            )
        projects = editor._gitlab_request_json(
            "GET",
            "/api/v4/projects",
            params={
                "membership": "true",
                "per_page": _GITLAB_PROJECT_SAMPLE_SIZE,
                "order_by": "id",
                "sort": "asc",
            },
        )
        if not isinstance(projects, list) or len(projects) < _GITLAB_PROJECT_SAMPLE_SIZE:
            raise RestorationReadbackError(
                "baseline_incomplete",
                f"GitLab user project sample has fewer than {_GITLAB_PROJECT_SAMPLE_SIZE} records",
            )
        records: list[dict[str, str]] = []
        for project in projects[:_GITLAB_PROJECT_SAMPLE_SIZE]:
            if not isinstance(project, Mapping):
                raise RestorationReadbackError(
                    "baseline_incomplete", "GitLab project record is not an object"
                )
            project_id = str(project.get("id") or "").strip()
            if not project_id:
                raise RestorationReadbackError("baseline_incomplete", "GitLab project lacks id")
            exact = editor._gitlab_get_json(
                f"/api/v4/projects/{quote(project_id, safe='')}",
                allow_missing=True,
            )
            if not isinstance(exact, Mapping):
                raise RestorationReadbackError(
                    "baseline_incomplete", "GitLab project detail is unavailable"
                )
            if str(exact.get("id")) != project_id:
                raise RestorationReadbackError(
                    "baseline_incomplete", "GitLab project detail identity differs"
                )
            path = str(exact.get("path_with_namespace") or "").strip()
            visibility = str(exact.get("visibility") or "").strip()
            if not path or not visibility:
                raise RestorationReadbackError(
                    "baseline_incomplete", "GitLab project detail lacks stable identity fields"
                )
            records.append({"id": project_id, "path": path, "visibility": visibility})
        return {"user": {"id": user_id, "username": username}, "projects": records}


def _reddit_snapshot(instance: Mapping[str, Any], task: Mapping[str, Any] | None) -> dict[str, Any]:
    forum_name = _first_forum_name(task)
    if forum_name is None:
        raise RestorationReadbackError(
            "baseline_incomplete", "Reddit readback needs a concrete forum_name"
        )
    with requests.Session() as session:
        editor = RedditEditor(dict(instance), session)
        username = editor._resolve_current_username(dict(instance))
        submit_path = f"/submit/{editor._quote(forum_name)}"
        form = editor._fetch_form_state(
            submit_path,
            required_fields=("submission[_token]", "submission[forum]"),
        )
        try:
            forum_id = editor._resolve_forum_id(form, forum_name)
        except Exception as exc:
            raise RestorationReadbackError(
                "baseline_incomplete", "Reddit forum selection ID unavailable"
            ) from exc
        return {
            "user": {"username": username},
            "forum": {"name": forum_name, "selection_id": forum_id},
        }


def capture_restoration_baseline(
    instance: object, *, task: Mapping[str, Any] | None = None
) -> RestorationReadback:
    """Capture one fixed baseline after a successful host recreation."""

    instance_mapping = _instance_mapping(instance)
    site = _site_name(instance_mapping)
    try:
        snapshot = (
            _gitlab_snapshot(instance_mapping)
            if site == "gitlab"
            else _reddit_snapshot(instance_mapping, task)
        )
    except RestorationReadbackError:
        raise
    except Exception as exc:
        raise RestorationReadbackError("readback_error", type(exc).__name__) from exc
    return RestorationReadback(
        site_name=site,
        site_url=_site_url(instance_mapping),
        baseline=snapshot,
    )


def verify_restoration_baseline(
    baseline: RestorationReadback,
    instance: object,
    *,
    task: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare a post-recreation sample and return safe operational evidence."""

    instance_mapping = _instance_mapping(instance)
    if (
        _site_name(instance_mapping) != baseline.site_name
        or _site_url(instance_mapping) != baseline.site_url
    ):
        raise RestorationReadbackError("instance_mismatch")
    observed = capture_restoration_baseline(instance_mapping, task=task)
    if observed.baseline != baseline.baseline:
        raise RestorationReadbackError("baseline_changed")
    return {
        "site_name": baseline.site_name,
        "site_url": baseline.site_url,
        "sample": baseline.baseline,
        "coverage": "fixed_sample_only",
    }


async def capture_restoration_baseline_async(
    instance: object, *, task: Mapping[str, Any] | None = None
) -> RestorationReadback:
    return await asyncio.to_thread(capture_restoration_baseline, instance, task=task)


async def verify_restoration_baseline_async(
    baseline: RestorationReadback,
    instance: object,
    *,
    task: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return await asyncio.to_thread(verify_restoration_baseline, baseline, instance, task=task)


__all__ = [
    "RestorationReadback",
    "RestorationReadbackError",
    "capture_restoration_baseline",
    "capture_restoration_baseline_async",
    "verify_restoration_baseline",
    "verify_restoration_baseline_async",
]
