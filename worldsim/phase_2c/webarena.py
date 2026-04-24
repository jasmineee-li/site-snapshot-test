"""WebArena Verified Phase 2c source-data policy."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote

from worldsim.phase_2c.policy import (
    PreflightClassification,
    ProbeTarget,
    SourceDataDecision,
    register_feasibility_policy,
)
from worldsim.phases.phase_2_reachability import resolve_start_url

DEFAULT_LOGIN_REDIRECT_BAILOUT_RATIO = 0.5

_BENIGN_TOKEN_RE = re.compile(r"\{benign_([A-Za-z0-9_]+)\}")

_LOGIN_STUB_MARKERS: tuple[str, ...] = (
    'action="/users/sign_in"',
    'name="user[login]"',
    'name="user[email]"',
    "/users/sign_up",
)
_LOGIN_STUB_BODY_SIZE_LIMIT = 600

_LOGIN_REDIRECT_LOCATION_MARKERS: tuple[str, ...] = (
    "/users/sign_in",
    "/users/sign_up",
    "/login",
    "/auth/sign_in",
)


def looks_like_login_stub(body: str) -> bool:
    if not body:
        return False
    lowered = body.lower()
    for marker in _LOGIN_STUB_MARKERS:
        if marker.lower() in lowered:
            return True
    if len(body) <= _LOGIN_STUB_BODY_SIZE_LIMIT and "sign in" in lowered:
        return True
    return False


def location_is_login(location: str | None) -> bool:
    if not location:
        return False
    lower = location.lower()
    return any(marker in lower for marker in _LOGIN_REDIRECT_LOCATION_MARKERS)


def classify_webarena_probe(
    *,
    status: int | None,
    headers: dict[str, str] | None,
    body_snippet: str,
    exception_name: str | None,
) -> PreflightClassification:
    if exception_name:
        name = exception_name.lower()
        if "timeout" in name:
            return PreflightClassification(
                kind="probe_timeout",
                quarantine=False,
                http_status=None,
                detail=f"preflight probe timed out ({exception_name})",
            )
        return PreflightClassification(
            kind="host_unreachable",
            quarantine=False,
            http_status=None,
            detail=f"preflight probe raised {exception_name}",
        )
    if status is None:
        return PreflightClassification(
            kind="host_unreachable",
            quarantine=False,
            http_status=None,
            detail="preflight probe returned no status",
        )
    if status == 200:
        if looks_like_login_stub(body_snippet):
            return PreflightClassification(
                kind="login_redirect",
                quarantine=True,
                http_status=200,
                detail="200 with login-stub markers — benign user cannot reach this surface",
            )
        return PreflightClassification("reachable", False, 200, "200 OK")
    if 300 <= status < 400:
        location = headers.get("location") if headers else None
        if location_is_login(location):
            return PreflightClassification(
                kind="login_redirect",
                quarantine=True,
                http_status=status,
                detail=f"{status} redirect to {location}",
            )
        return PreflightClassification(
            kind="redirect_noncritical",
            quarantine=False,
            http_status=status,
            detail=f"{status} redirect (non-login)",
        )
    if status == 401:
        return PreflightClassification(
            "auth_missing",
            True,
            status,
            "401 Unauthorized — benign storage_state did not authenticate",
        )
    if status == 403:
        return PreflightClassification(
            "forbidden",
            True,
            status,
            "403 Forbidden — benign user lacks permission for this surface",
        )
    if status == 404:
        return PreflightClassification(
            "not_found",
            True,
            status,
            "404 Not Found — stale L4 anchor or deleted resource",
        )
    if status == 410:
        return PreflightClassification(
            "gone",
            True,
            status,
            "410 Gone — resource permanently removed",
        )
    if status == 429:
        return PreflightClassification(
            "rate_limited",
            False,
            status,
            "429 Too Many Requests — transient",
        )
    if 500 <= status < 600:
        return PreflightClassification(
            "server_error",
            False,
            status,
            f"{status} server error — transient",
        )
    return PreflightClassification(
        "unexpected_status",
        False,
        status,
        f"unexpected HTTP {status}",
    )


def task_probe_url(task: dict[str, Any], instance_site_url: str) -> str | None:
    target = task.get("benign_target_resource")
    if not isinstance(target, dict):
        return None
    start_url = target.get("start_url_resolved")
    if not isinstance(start_url, str) or not start_url.strip():
        return None
    return resolve_start_url(start_url, instance_site_url)


def render_anchor_tokens(value: Any, anchors: dict[str, Any]) -> str | None:
    if value in (None, ""):
        return None
    text = str(value)

    def repl(match: re.Match[str]) -> str:
        key = match.group(1)
        replacement = anchors.get(key)
        return "" if replacement is None else str(replacement)

    rendered = _BENIGN_TOKEN_RE.sub(repl, text).strip()
    return rendered or None


def first_value(args: dict[str, Any], anchors: dict[str, Any], *names: str) -> str | None:
    for name in names:
        if name in args:
            rendered = render_anchor_tokens(args.get(name), anchors)
            if rendered:
                return rendered
        rendered = render_anchor_tokens(anchors.get(name), anchors)
        if rendered:
            return rendered
    return None


def clean_gitlab_project_path(project_path: str) -> str:
    path = project_path.strip().strip("/")
    if "/" in path and path.split("/", 1)[0].startswith("localhost:"):
        return path.split("/", 1)[1]
    return path


def editor_surface_path(
    *,
    site: str,
    method: str,
    args: dict[str, Any],
    anchors: dict[str, Any],
) -> str | None:
    if site == "reddit":
        if method == "create_comment":
            forum = first_value(args, anchors, "forum_name")
            submission = first_value(args, anchors, "submission_id")
            if forum and submission:
                return f"/f/{quote(forum, safe='')}/{quote(submission, safe='')}"
        if method == "create_submission":
            forum = first_value(args, anchors, "forum_name")
            if forum:
                return f"/submit/{quote(forum, safe='')}"
        return None

    if site == "gitlab":
        project_path = first_value(args, anchors, "project_path", "project_path_template")
        if not project_path:
            return None
        project_path = clean_gitlab_project_path(project_path)
        if method == "create_issue_note":
            issue_iid = first_value(args, anchors, "issue_iid")
            if issue_iid:
                return f"/{project_path}/-/issues/{quote(issue_iid, safe='')}"
        if method == "create_mr_note":
            mr_iid = first_value(args, anchors, "mr_iid")
            if mr_iid:
                return f"/{project_path}/-/merge_requests/{quote(mr_iid, safe='')}"
        if method == "create_repo_file":
            branch = first_value(args, anchors, "branch") or "main"
            file_path = first_value(args, anchors, "path")
            if file_path:
                quoted_file = "/".join(quote(part, safe="") for part in file_path.split("/"))
                return f"/{project_path}/-/blob/{quote(branch, safe='')}/{quoted_file}"
    return None


def dedupe_targets(targets: list[ProbeTarget]) -> list[ProbeTarget]:
    deduped: list[ProbeTarget] = []
    seen: set[str] = set()
    for target in targets:
        if target.url in seen:
            continue
        seen.add(target.url)
        deduped.append(target)
    return deduped


@dataclass(frozen=True)
class WebArenaFeasibilityPolicy:
    site: str
    auth_path: str | None = None
    benchmark: str = "webarena_verified"

    def auth_self_test_path(self) -> str | None:
        return self.auth_path

    def requires_authenticated_preflight(self) -> bool:
        return self.site.strip().lower() == "gitlab"

    def probe_targets(self, task: dict[str, Any], instance_site_url: str) -> list[ProbeTarget]:
        targets: list[ProbeTarget] = []
        start_url = task_probe_url(task, instance_site_url)
        if start_url:
            targets.append(ProbeTarget(url=start_url, source="benign_start_url"))

        resource = task.get("benign_target_resource")
        anchors_raw = resource.get("anchors") if isinstance(resource, dict) else None
        anchors = anchors_raw if isinstance(anchors_raw, dict) else {}
        seed = task.get("adversarial_data_seed")
        calls = seed.get("editor_calls") if isinstance(seed, dict) else None
        if not isinstance(calls, list):
            return dedupe_targets(targets)

        task_site = str(task.get("site") or "").strip().lower()
        for index, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            site = str(call.get("site") or task_site).strip().lower()
            method = str(call.get("method") or "").strip()
            args = call.get("args")
            if not method or not isinstance(args, dict):
                continue
            path = editor_surface_path(site=site, method=method, args=args, anchors=anchors)
            if path:
                targets.append(
                    ProbeTarget(
                        url=resolve_start_url(path, instance_site_url),
                        source=f"editor_call[{index}].{site}.{method}",
                    )
                )
        return dedupe_targets(targets)

    def classify_probe(
        self,
        *,
        status: int | None,
        headers: dict[str, str] | None,
        body_snippet: str,
        exception_name: str | None,
    ) -> PreflightClassification:
        return classify_webarena_probe(
            status=status,
            headers=headers,
            body_snippet=body_snippet,
            exception_name=exception_name,
        )

    def decide_source_data(
        self,
        *,
        task: dict[str, Any],
        classifications_by_target: dict[int, list[PreflightClassification]],
        target_audit: dict[int, ProbeTarget],
        login_redirect_count: int,
        probed_count: int,
        bailout_ratio: float,
    ) -> SourceDataDecision:
        for target_index, classifications in classifications_by_target.items():
            quarantine_classifications = [c for c in classifications if c.quarantine]
            if not classifications:
                continue
            quarantine_rate = len(quarantine_classifications) / len(classifications)
            if quarantine_rate <= 0.5 or not quarantine_classifications:
                continue
            kind_counts: dict[str, int] = {}
            for classification in quarantine_classifications:
                kind_counts[classification.kind] = kind_counts.get(classification.kind, 0) + 1
            dominant = max(quarantine_classifications, key=lambda c: kind_counts[c.kind])
            return SourceDataDecision(
                action="drop",
                classification=dominant,
                target=target_audit[target_index],
                evidence={
                    "replicas_probed": len(classifications_by_target[target_index]),
                    "replicas_agreeing": len(quarantine_classifications),
                },
            )
        return SourceDataDecision(action="keep")

    def counts_toward_run_bailout(self, classification: PreflightClassification) -> bool:
        return classification.kind == "login_redirect"

    def should_bailout_source_data_run(
        self,
        *,
        bailout_count: int,
        probed_count: int,
        bailout_ratio: float,
    ) -> bool:
        return bool(probed_count) and bailout_count / probed_count > bailout_ratio

    def restore_drop_on_run_bailout(self, issue: dict[str, Any]) -> bool:
        return issue.get("kind") == "login_redirect"


def register_webarena_policies() -> None:
    register_feasibility_policy(WebArenaFeasibilityPolicy(site="gitlab", auth_path="/-/profile"))
    register_feasibility_policy(WebArenaFeasibilityPolicy(site="reddit"))


register_webarena_policies()

__all__ = [
    "DEFAULT_LOGIN_REDIRECT_BAILOUT_RATIO",
    "WebArenaFeasibilityPolicy",
    "classify_webarena_probe",
    "clean_gitlab_project_path",
    "dedupe_targets",
    "editor_surface_path",
    "first_value",
    "location_is_login",
    "looks_like_login_stub",
    "register_webarena_policies",
    "render_anchor_tokens",
    "task_probe_url",
]
