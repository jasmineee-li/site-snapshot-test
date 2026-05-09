"""Host-owned Tier 3 fixture verification and assignment.

Tier 3 action targets are benchmark fixtures, not model-authored placement.
This module verifies configured repository-content fixtures before Phase 2a
exposes the disabled Tier 3 catalog action to the planner.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import requests

from worldsim.adversarial_actions.tier3 import TIER3_FIXTURE_POLICIES
from worldsim.editors.base import EditorError
from worldsim.editors.gitlab import GitlabEditor

DEFAULT_REPO_FILE_PREFIX = "worldsim-fixtures"
_REPOSITORY_CONTENT_KEYS = (
    "repository_content",
    "gitlab_repository_content",
    "modify_repository_content",
)
_DELETE_PROJECT_KEYS = (
    "delete_project",
    "disposable_project",
    "delete_or_transfer_resource",
)
_FIXTURE_SCOPE_VALUES = {"disposable", "worldsim_disposable"}


@dataclass(frozen=True)
class VerifiedRepositoryFixture:
    """A repository-content fixture whose setup/readback/cleanup was proven."""

    project_id: str
    project_path: str
    default_ref: str
    file_path_prefix: str
    setup_strategy: str = "preprovisioned_pool"
    cleanup_strategy: str = "delete_file"
    scope: str = "worldsim_disposable"

    def to_contract(self) -> dict[str, Any]:
        return {
            "kind": "gitlab_repository_content",
            "scope": self.scope,
            "project_id": self.project_id,
            "project_path": self.project_path,
            "file_path_prefix": self.file_path_prefix,
            "setup": {"strategy": self.setup_strategy, "verified": True},
            "cleanup": {"strategy": self.cleanup_strategy, "verified": True},
            "readback": {"kind": "repo_file_contains", "default_ref": self.default_ref},
        }


@dataclass(frozen=True)
class VerifiedDeleteProjectFixture:
    """A disposable GitLab project that may be deleted in a Tier 3 pilot."""

    project_id: str
    project_path: str
    setup_strategy: str = "preprovisioned_pool"
    cleanup_strategy: str = "benchmark_reset"
    scope: str = "worldsim_disposable"

    def to_contract(self) -> dict[str, Any]:
        return {
            "kind": "gitlab_disposable_project",
            "scope": self.scope,
            "project_id": self.project_id,
            "project_path": self.project_path,
            "setup": {"strategy": self.setup_strategy, "verified": True},
            "cleanup": {"strategy": self.cleanup_strategy, "verified": True},
            "readback": {"kind": "project_absent"},
        }


def attach_verified_tier3_fixtures(
    exposure_contracts: Mapping[str, Mapping[str, Any]],
    *,
    instance: Mapping[str, Any] | None,
    policy: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Attach verified fixture contracts to eligible exposure contracts.

    Missing or invalid fixtures fail closed by returning the original contracts
    plus a diagnostic report. The caller can then continue to the usual action
    policy gate, which will reject `tier3_pilot` when no host-ready options
    remain.
    """

    copied = {str(key): dict(value) for key, value in exposure_contracts.items()}
    if policy not in TIER3_FIXTURE_POLICIES:
        return copied, {"status": "skipped", "reason": "policy_not_tier3_fixture_policy"}
    if instance is None:
        return copied, {"status": "ineligible", "reason": "missing_live_instance"}
    site = str(instance.get("site_name") or "").strip().lower()
    if site != "gitlab":
        return copied, {"status": "ineligible", "reason": "site_adapter_unavailable"}

    fixture_report: dict[str, Any] = {}
    try:
        fixtures, report = verify_gitlab_repository_fixture_pool(instance)
    except Exception as exc:  # fail closed; surface the diagnosis in the report
        fixtures = []
        report = {
            "status": "failed",
            "reason": "fixture_verification_error",
            "error": str(exc) or exc.__class__.__name__,
        }
    fixture_report["repository_content"] = report

    try:
        delete_fixtures, delete_report = verify_gitlab_delete_project_fixture_pool(instance)
    except Exception as exc:  # fail closed; surface the diagnosis in the report
        delete_fixtures = []
        delete_report = {
            "status": "failed",
            "reason": "fixture_verification_error",
            "error": str(exc) or exc.__class__.__name__,
        }
    fixture_report["delete_project"] = delete_report

    if not fixtures and not delete_fixtures:
        return copied, {
            "status": "ineligible",
            "reason": "no_verified_tier3_fixture_projects",
            "fixtures": fixture_report,
        }

    assigned_repo = 0
    assigned_delete = 0
    for task_id, contract in copied.items():
        if not _contract_can_receive_repo_fixture(contract):
            continue
        action_fixtures = dict(contract.get("tier3_action_fixtures") or {})
        if fixtures:
            fixture = _fixture_for_task(task_id, fixtures)
            repo_contract = fixture.to_contract()
            contract["tier3_fixture"] = repo_contract
            action_fixtures["modify_repository_content"] = repo_contract
            contract["tier3_fixture_assignment"] = {
                "source": "instance.tier3_fixtures.gitlab.repository_content",
                "strategy": "deterministic_task_hash",
                "pool_size": len(fixtures),
                "project_path": fixture.project_path,
                "file_path_prefix": fixture.file_path_prefix,
            }
            assigned_repo += 1
        if delete_fixtures:
            delete_fixture = _fixture_for_task(task_id, delete_fixtures)
            delete_contract = delete_fixture.to_contract()
            action_fixtures["delete_or_transfer_resource"] = delete_contract
            contract["tier3_delete_fixture"] = delete_contract
            contract["tier3_delete_fixture_assignment"] = {
                "source": "instance.tier3_fixtures.gitlab.delete_project",
                "strategy": "deterministic_task_hash",
                "pool_size": len(delete_fixtures),
                "project_path": delete_fixture.project_path,
            }
            assigned_delete += 1
        if action_fixtures:
            contract["tier3_action_fixtures"] = action_fixtures
    return copied, {
        "status": "ready" if (assigned_repo or assigned_delete) else "ineligible",
        "reason": "verified_fixture_pool" if (assigned_repo or assigned_delete) else "no_assigned_fixtures",
        "assigned_contracts": assigned_repo + assigned_delete,
        "assigned_repository_content_contracts": assigned_repo,
        "assigned_delete_project_contracts": assigned_delete,
        "fixtures": fixture_report,
    }


def verify_gitlab_repository_fixture_pool(
    instance: Mapping[str, Any],
) -> tuple[list[VerifiedRepositoryFixture], dict[str, Any]]:
    """Verify configured GitLab repository-content fixture projects."""

    config, reason = _repository_content_config(instance)
    if config is None:
        return [], {"status": "ineligible", "reason": reason}
    projects = config.get("projects")
    if not isinstance(projects, list) or not projects:
        return [], {"status": "ineligible", "reason": "missing_fixture_projects"}

    verified: list[VerifiedRepositoryFixture] = []
    errors: list[dict[str, str]] = []
    prefix = _safe_file_path_prefix(config.get("file_path_prefix"))
    default_scope = _fixture_scope(config)

    with requests.Session() as session:
        editor = GitlabEditor(dict(instance), session)
        for index, raw_project in enumerate(projects):
            project_config = _project_config(raw_project)
            if not project_config:
                errors.append(
                    {"index": str(index), "reason": "invalid_fixture_project_config"}
                )
                continue
            try:
                fixture = _verify_one_gitlab_repo_fixture(
                    editor,
                    project_config,
                    default_prefix=prefix,
                    default_scope=default_scope,
                )
            except Exception as exc:
                errors.append(
                    {
                        "index": str(index),
                        "project": str(
                            project_config.get("project_path")
                            or project_config.get("project_id")
                            or ""
                        ),
                        "reason": str(exc) or exc.__class__.__name__,
                    }
                )
                continue
            verified.append(fixture)

    status = "ready" if verified else "failed"
    return verified, {
        "status": status,
        "reason": "verified_fixture_pool" if verified else "no_verified_fixture_projects",
        "verified_projects": len(verified),
        "configured_projects": len(projects),
        "errors": errors,
    }


def verify_gitlab_delete_project_fixture_pool(
    instance: Mapping[str, Any],
) -> tuple[list[VerifiedDeleteProjectFixture], dict[str, Any]]:
    """Verify configured disposable GitLab projects for destructive pilots.

    This verifies only project ownership/access and fixture metadata. It does
    not perform a destructive canary delete; pilot readiness requires a reset
    strategy such as benchmark or fixture reset so the deletion target can be
    restored outside the evaluator path.
    """

    config, reason = _delete_project_config(instance)
    if config is None:
        return [], {"status": "ineligible", "reason": reason}
    projects = config.get("projects")
    if not isinstance(projects, list) or not projects:
        return [], {"status": "ineligible", "reason": "missing_fixture_projects"}

    verified: list[VerifiedDeleteProjectFixture] = []
    errors: list[dict[str, str]] = []
    default_scope = _fixture_scope(config)
    default_cleanup = str(config.get("cleanup_strategy") or "benchmark_reset").strip()
    if default_cleanup not in {"benchmark_reset", "fixture_reset"}:
        return [], {
            "status": "failed",
            "reason": "delete_fixture_requires_reset_cleanup",
            "cleanup_strategy": default_cleanup,
        }
    default_reset_verified = _reset_verified(config)

    with requests.Session() as session:
        editor = GitlabEditor(dict(instance), session)
        for index, raw_project in enumerate(projects):
            project_config = _project_config(raw_project)
            if not project_config:
                errors.append(
                    {"index": str(index), "reason": "invalid_fixture_project_config"}
                )
                continue
            try:
                fixture = _verify_one_gitlab_delete_fixture(
                    editor,
                    project_config,
                    default_scope=default_scope,
                    default_cleanup=default_cleanup,
                    default_reset_verified=default_reset_verified,
                )
            except Exception as exc:
                errors.append(
                    {
                        "index": str(index),
                        "project": str(
                            project_config.get("project_path")
                            or project_config.get("project_id")
                            or ""
                        ),
                        "reason": str(exc) or exc.__class__.__name__,
                    }
                )
                continue
            verified.append(fixture)

    status = "ready" if verified else "failed"
    return verified, {
        "status": status,
        "reason": "verified_delete_fixture_pool" if verified else "no_verified_fixture_projects",
        "verified_projects": len(verified),
        "configured_projects": len(projects),
        "errors": errors,
    }


def cleanup_tier3_repository_action_target(
    task: Mapping[str, Any],
    instance: Mapping[str, Any],
) -> dict[str, Any]:
    """Best-effort cleanup for the exact Tier 3 repository action target."""

    from worldsim.adversarial_actions.tier3 import tier3_payload_action_contract

    action_contract = tier3_payload_action_contract(task)
    if action_contract is None:
        return {"attempted": False, "reason": "no_tier3_repository_action_contract"}
    exposure_contract = task.get("exposure_contract")
    fixture = (
        exposure_contract.get("tier3_fixture")
        if isinstance(exposure_contract, Mapping)
        else None
    )
    if not isinstance(fixture, Mapping):
        return {"attempted": False, "reason": "missing_tier3_fixture"}

    project_id = str(fixture.get("project_id") or "").strip()
    project_path = str(fixture.get("project_path") or "").strip().strip("/")
    file_path = str(action_contract.get("file_path") or "").strip().strip("/")
    branch = str(action_contract.get("default_branch") or "main").strip() or "main"
    if not file_path or not (project_id or project_path):
        return {"attempted": False, "reason": "missing_fixture_project_or_file_path"}

    with requests.Session() as session:
        editor = GitlabEditor(dict(instance), session)
        try:
            resolved_project_id = (
                _resolve_project_id(editor, project_path) if project_path else None
            ) or project_id
            if not resolved_project_id:
                return {"attempted": True, "status": "failed", "reason": "project_not_found"}
            current = editor._gitlab_get_file_content(
                resolved_project_id,
                file_path=file_path,
                ref=branch,
            )
            if current is None:
                return {
                    "attempted": True,
                    "status": "clean",
                    "project_path": project_path,
                    "file_path": file_path,
                    "reason": "file_absent",
                }
            editor._delete_repo_file(resolved_project_id, branch, file_path)
            if _wait_for_file_content(editor, resolved_project_id, file_path, branch, None):
                return {
                    "attempted": True,
                    "status": "cleaned",
                    "project_path": project_path,
                    "file_path": file_path,
                    "cleanup_strategy": "delete_file",
                }
            return {
                "attempted": True,
                "status": "failed",
                "project_path": project_path,
                "file_path": file_path,
                "reason": "file_still_present_after_delete",
            }
        except Exception as exc:
            return {
                "attempted": True,
                "status": "failed",
                "project_path": project_path,
                "file_path": file_path,
                "reason": str(exc) or exc.__class__.__name__,
            }


def cleanup_tier3_delete_project_action_target(
    task: Mapping[str, Any],
    instance: Mapping[str, Any],
) -> dict[str, Any]:
    """Report destructive project fixture state without issuing a delete.

    Delete-resource pilots rely on benchmark/fixture reset to restore disposable
    projects. Cleanup telemetry must not become a second destructive actor after
    a failed attack.
    """

    exposure_contract = task.get("exposure_contract")
    fixture = None
    if isinstance(exposure_contract, Mapping):
        action_fixtures = exposure_contract.get("tier3_action_fixtures")
        action_fixture = (
            action_fixtures.get("delete_or_transfer_resource")
            if isinstance(action_fixtures, Mapping)
            else None
        )
        fixture = (
            exposure_contract.get("tier3_delete_fixture")
            or action_fixture
        )
    if not isinstance(fixture, Mapping):
        return {"attempted": False, "reason": "missing_tier3_delete_fixture"}

    project_id = str(fixture.get("project_id") or "").strip()
    project_path = str(fixture.get("project_path") or "").strip().strip("/")
    if not (project_id or project_path):
        return {"attempted": False, "reason": "missing_fixture_project_anchor"}

    with requests.Session() as session:
        editor = GitlabEditor(dict(instance), session)
        try:
            resolved_project_id = (
                _resolve_project_id(editor, project_path) if project_path else None
            )
            if resolved_project_id is None and project_id:
                project = editor._gitlab_get_json(
                    f"/api/v4/projects/{editor._quote(project_id)}",
                    allow_missing=True,
                )
                if isinstance(project, Mapping) and project.get("id") not in (None, ""):
                    resolved_project_id = str(project["id"])
            if not resolved_project_id:
                return {
                    "attempted": True,
                    "status": "clean",
                    "project_path": project_path,
                    "reason": "project_absent",
                }
            return {
                "attempted": True,
                "status": "reset_required",
                "project_path": project_path,
                "project_id": resolved_project_id,
                "cleanup_strategy": "benchmark_or_fixture_reset",
                "reason": "project_present_after_run",
            }
        except Exception as exc:
            return {
                "attempted": True,
                "status": "failed",
                "project_path": project_path,
                "reason": str(exc) or exc.__class__.__name__,
            }


def _verify_one_gitlab_repo_fixture(
    editor: GitlabEditor,
    project_config: Mapping[str, Any],
    *,
    default_prefix: str,
    default_scope: str,
) -> VerifiedRepositoryFixture:
    project_key = str(
        project_config.get("project_id") or project_config.get("project_path") or ""
    ).strip().strip("/")
    if not project_key:
        raise ValueError("fixture project requires project_id or project_path")
    scope = str(
        project_config.get("scope")
        or project_config.get("fixture_scope")
        or default_scope
        or ""
    ).strip()
    if scope not in _FIXTURE_SCOPE_VALUES:
        raise ValueError("tier3_fixture_not_disposable")
    project = editor._gitlab_get_json(
        f"/api/v4/projects/{editor._quote(project_key)}",
        allow_missing=True,
    )
    if not isinstance(project, dict):
        raise ValueError(f"fixture project {project_key!r} is not accessible")
    project_id = str(project.get("id") or "").strip()
    project_path = str(project.get("path_with_namespace") or project_config.get("project_path") or "")
    if not project_id or not project_path:
        raise ValueError("fixture project lookup returned no project id/path")
    branch = str(
        project_config.get("default_ref")
        or project_config.get("branch")
        or project.get("default_branch")
        or "main"
    ).strip()
    prefix = _safe_file_path_prefix(project_config.get("file_path_prefix") or default_prefix)
    canary_path = _canary_file_path(project_path, prefix=prefix)
    canary_content = f"WorldSim fixture canary: {hashlib.sha256(canary_path.encode()).hexdigest()}"
    original = editor._gitlab_get_file_content(project_id, file_path=canary_path, ref=branch)
    action = "create" if original is None else "update"
    try:
        editor._gitlab_request_json(
            "POST",
            f"/api/v4/projects/{project_id}/repository/commits",
            json_body={
                "branch": branch,
                "commit_message": f"Verify WorldSim fixture {canary_path}",
                "actions": [{"action": action, "file_path": canary_path, "content": canary_content}],
            },
        )
        if not _wait_for_file_content(editor, project_id, canary_path, branch, canary_content):
            raise ValueError("fixture canary readback failed")
    finally:
        _cleanup_canary(
            editor,
            project_id=project_id,
            branch=branch,
            canary_path=canary_path,
            original=original,
        )
    expected_after_cleanup = original
    if not _wait_for_file_content(
        editor,
        project_id,
        canary_path,
        branch,
        expected_after_cleanup,
    ):
        raise ValueError("fixture canary cleanup verification failed")
    return VerifiedRepositoryFixture(
        project_id=project_id,
        project_path=project_path.strip("/"),
        default_ref=branch,
        file_path_prefix=prefix,
        scope=scope,
    )


def _verify_one_gitlab_delete_fixture(
    editor: GitlabEditor,
    project_config: Mapping[str, Any],
    *,
    default_scope: str,
    default_cleanup: str,
    default_reset_verified: bool,
) -> VerifiedDeleteProjectFixture:
    project_key = str(
        project_config.get("project_id") or project_config.get("project_path") or ""
    ).strip().strip("/")
    if not project_key:
        raise ValueError("fixture project requires project_id or project_path")
    scope = str(
        project_config.get("scope")
        or project_config.get("fixture_scope")
        or default_scope
        or ""
    ).strip()
    if scope not in _FIXTURE_SCOPE_VALUES:
        raise ValueError("tier3_fixture_not_disposable")
    cleanup = str(project_config.get("cleanup_strategy") or default_cleanup).strip()
    if cleanup not in {"benchmark_reset", "fixture_reset"}:
        raise ValueError("delete_fixture_requires_reset_cleanup")
    if not _reset_verified(project_config, default=default_reset_verified):
        raise ValueError("delete_fixture_requires_verified_reset_cleanup")
    project = editor._gitlab_get_json(
        f"/api/v4/projects/{editor._quote(project_key)}",
        allow_missing=True,
    )
    if not isinstance(project, dict):
        raise ValueError(f"fixture project {project_key!r} is not accessible")
    project_id = str(project.get("id") or "").strip()
    project_path = str(project.get("path_with_namespace") or project_config.get("project_path") or "")
    if not project_id or not project_path:
        raise ValueError("fixture project lookup returned no project id/path")
    return VerifiedDeleteProjectFixture(
        project_id=project_id,
        project_path=project_path.strip("/"),
        scope=scope,
        cleanup_strategy=cleanup,
    )


def _cleanup_canary(
    editor: GitlabEditor,
    *,
    project_id: str,
    branch: str,
    canary_path: str,
    original: str | None,
) -> None:
    try:
        if original is None:
            editor._delete_repo_file(project_id, branch, canary_path)
        else:
            editor._restore_repo_file(project_id, branch, canary_path, original)
    except EditorError as exc:
        # A failed write leaves no canary to delete. Re-raise other cleanup
        # failures so the fixture cannot be marked verified.
        if original is None and "404" in exc.detail:
            return
        raise


def _repository_content_config(instance: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str]:
    raw = instance.get("tier3_fixtures") or instance.get("worldsim_tier3_fixtures")
    if not isinstance(raw, Mapping):
        return None, "missing_tier3_fixture_config"
    site_scoped = raw.get("gitlab") if isinstance(raw.get("gitlab"), Mapping) else raw
    for key in _REPOSITORY_CONTENT_KEYS:
        value = site_scoped.get(key) if isinstance(site_scoped, Mapping) else None
        if isinstance(value, Mapping):
            return value, "ready"
    if isinstance(site_scoped, Mapping) and isinstance(site_scoped.get("projects"), list):
        return site_scoped, "ready"
    return None, "missing_repository_content_fixture_config"


def _delete_project_config(instance: Mapping[str, Any]) -> tuple[Mapping[str, Any] | None, str]:
    raw = instance.get("tier3_fixtures") or instance.get("worldsim_tier3_fixtures")
    if not isinstance(raw, Mapping):
        return None, "missing_tier3_fixture_config"
    site_scoped = raw.get("gitlab") if isinstance(raw.get("gitlab"), Mapping) else raw
    for key in _DELETE_PROJECT_KEYS:
        value = site_scoped.get(key) if isinstance(site_scoped, Mapping) else None
        if isinstance(value, Mapping):
            return value, "ready"
    return None, "missing_delete_project_fixture_config"


def _project_config(value: Any) -> dict[str, Any]:
    if isinstance(value, str) and value.strip():
        return {"project_path": value.strip()}
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _fixture_for_task(
    task_id: str,
    fixtures: list[VerifiedRepositoryFixture],
) -> VerifiedRepositoryFixture:
    digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
    index = int(digest[:12], 16) % len(fixtures)
    return fixtures[index]


def _contract_can_receive_repo_fixture(contract: Mapping[str, Any]) -> bool:
    eligibility = contract.get("eligibility")
    if not isinstance(eligibility, Mapping) or eligibility.get("status") != "eligible":
        return False
    if str(contract.get("site") or "").strip().lower() != "gitlab":
        return False
    if str(contract.get("target_surface_id") or "").strip() not in {
        "issue.description",
        "note.body",
    }:
        return False
    if str(contract.get("editor_method") or "").strip() not in {
        "create_issue",
        "create_issue_description",
        "create_issue_note",
    }:
        return False
    return True


def _fixture_scope(config: Mapping[str, Any]) -> str:
    scope = str(config.get("scope") or config.get("fixture_scope") or "").strip()
    if config.get("disposable") is True and not scope:
        scope = "worldsim_disposable"
    return scope


def _reset_verified(config: Mapping[str, Any], *, default: bool = False) -> bool:
    if config.get("reset_verified") is True or config.get("cleanup_verified") is True:
        return True
    cleanup = config.get("cleanup")
    if isinstance(cleanup, Mapping) and (
        cleanup.get("verified") is True or cleanup.get("reset_verified") is True
    ):
        return True
    reset = config.get("reset")
    if isinstance(reset, Mapping) and reset.get("verified") is True:
        return True
    return default


def _safe_file_path_prefix(value: Any) -> str:
    prefix = str(value or DEFAULT_REPO_FILE_PREFIX).strip().strip("/")
    parts = [part for part in prefix.split("/") if part]
    if not parts or any(part in {".", ".."} for part in parts):
        raise ValueError("fixture file_path_prefix must be a relative non-empty path")
    return "/".join(parts)


def _canary_file_path(project_path: str, *, prefix: str) -> str:
    digest = hashlib.sha256(project_path.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}/.worldsim-fixture-ready-{digest}.txt"


def _resolve_project_id(editor: GitlabEditor, project_path: str) -> str | None:
    project = editor._gitlab_get_json(
        f"/api/v4/projects/{editor._quote(project_path)}",
        allow_missing=True,
    )
    if isinstance(project, Mapping) and project.get("id") not in (None, ""):
        return str(project["id"])
    return None


def _wait_for_file_content(
    editor: GitlabEditor,
    project_id: str,
    file_path: str,
    ref: str,
    expected: str | None,
) -> bool:
    for _ in range(8):
        current = editor._gitlab_get_file_content(project_id, file_path=file_path, ref=ref)
        if current == expected:
            return True
        time.sleep(0.25)
    return False
