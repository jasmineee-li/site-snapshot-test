"""Git helpers for controller-managed generation worktrees."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from agentlab.benchmarks.redteam.app_artifacts import validate_path_component
from agentlab.benchmarks.redteam.behavior_ids import controller_slug_for_behavior


class GitControllerError(RuntimeError):
    """Raised when controller git invariants are violated."""


@dataclass(frozen=True)
class ControllerWorkspace:
    repo_root: Path
    worktree_path: Path
    branch: str
    raw_behavior_id: str
    controller_slug: str
    logical_output_dir: Path
    app_dir: Path
    published_app_dir: Path
    logs_dir: Path
    scratch_logs_dir: Path
    source_seed_dir: Path | None


def _run_git(args: list[str], *, cwd: str | Path) -> str:
    proc = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise GitControllerError(proc.stderr.strip() or proc.stdout.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def _find_repo_root(path: str | Path) -> Path:
    current = Path(path).resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise GitControllerError(f"Controller requires a git-backed workspace; no .git found above {current}")


def normalize_output_dir(output_dir: str | Path, *, cwd: str | Path | None = None) -> Path:
    base_dir = Path(cwd).expanduser().resolve() if cwd is not None else Path.cwd().resolve()
    output_path = Path(output_dir).expanduser()
    if not output_path.is_absolute():
        output_path = base_dir / output_path
    return output_path.resolve()


def resolve_logical_output_dir(
    repo_root: str | Path,
    output_dir: str | Path,
) -> Path:
    repo_root = Path(repo_root).resolve()
    resolved = normalize_output_dir(output_dir)
    try:
        return resolved.relative_to(repo_root)
    except ValueError as exc:
        raise GitControllerError(
            f"Controller output_dir must resolve inside repo root {repo_root}: {resolved}"
        ) from exc


def ensure_controller_workspace(
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    behavior_id: str,
) -> ControllerWorkspace:
    repo_root = _find_repo_root(repo_root)
    logical_output_dir = resolve_logical_output_dir(repo_root, output_dir)
    behavior_id = validate_path_component(behavior_id, "behavior_id")
    controller_slug = controller_slug_for_behavior(behavior_id, logical_output_dir)
    branch = f"controller/{controller_slug}"
    worktree_path = repo_root / ".controller-worktrees" / controller_slug
    scratch_logs_dir = worktree_path / "logs" / controller_slug
    logs_dir = repo_root / "logs" / controller_slug
    app_dir = worktree_path / logical_output_dir / behavior_id
    published_app_dir = repo_root / logical_output_dir / behavior_id
    source_seed_dir = published_app_dir

    existing_worktrees = _run_git(["worktree", "list", "--porcelain"], cwd=repo_root).splitlines()
    has_worktree = any(line.strip() == f"worktree {worktree_path}" for line in existing_worktrees)
    if not has_worktree:
        worktree_path.parent.mkdir(parents=True, exist_ok=True)
        if worktree_path.exists():
            shutil.rmtree(worktree_path)
        branch_exists = True
        try:
            _run_git(["show-ref", "--verify", f"refs/heads/{branch}"], cwd=repo_root)
        except GitControllerError:
            branch_exists = False
        if branch_exists:
            _run_git(["worktree", "add", str(worktree_path), branch], cwd=repo_root)
        else:
            _run_git(["worktree", "add", "-b", branch, str(worktree_path), "HEAD"], cwd=repo_root)

    scratch_logs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    if source_seed_dir.exists() and not app_dir.exists():
        app_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source_seed_dir,
            app_dir,
            ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
        )

    return ControllerWorkspace(
        repo_root=repo_root,
        worktree_path=worktree_path,
        branch=branch,
        raw_behavior_id=behavior_id,
        controller_slug=controller_slug,
        logical_output_dir=logical_output_dir,
        app_dir=app_dir,
        published_app_dir=published_app_dir,
        logs_dir=logs_dir,
        scratch_logs_dir=scratch_logs_dir,
        source_seed_dir=source_seed_dir if source_seed_dir.exists() else None,
    )


def publish_controller_workspace(workspace: ControllerWorkspace) -> None:
    workspace.published_app_dir.parent.mkdir(parents=True, exist_ok=True)
    if workspace.published_app_dir.exists():
        shutil.rmtree(workspace.published_app_dir)
    shutil.copytree(
        workspace.app_dir,
        workspace.published_app_dir,
        ignore=shutil.ignore_patterns(".git", "__pycache__", "*.pyc"),
    )
    scratch_pipeline_state = workspace.scratch_logs_dir / "pipeline_state.json"
    if scratch_pipeline_state.exists():
        workspace.logs_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(scratch_pipeline_state, workspace.logs_dir / "pipeline_state.json")


def current_head(worktree_path: str | Path) -> str:
    return _run_git(["rev-parse", "HEAD"], cwd=worktree_path)


def reset_hard(worktree_path: str | Path, commit: str) -> None:
    _run_git(["reset", "--hard", commit], cwd=worktree_path)


def scoped_dirty_files(
    worktree_path: str | Path,
    scope_path: str | Path,
) -> list[str]:
    scope_path = Path(scope_path)
    if not scope_path.exists():
        return []
    rel = os.fspath(scope_path.relative_to(worktree_path))
    output = _run_git(["status", "--short", "--", rel], cwd=worktree_path)
    dirty: list[str] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        dirty.append(line[3:] if len(line) > 3 else line)
    return dirty


def checkpoint_scope(
    worktree_path: str | Path,
    *,
    scope_path: str | Path,
    message: str,
) -> str | None:
    scope_path = Path(scope_path)
    if not scope_path.exists():
        return None
    rel = os.fspath(scope_path.relative_to(worktree_path))
    dirty = scoped_dirty_files(worktree_path, scope_path)
    if not dirty:
        return None
    _run_git(["add", "--", rel], cwd=worktree_path)
    _run_git(["commit", "-m", message], cwd=worktree_path)
    return current_head(worktree_path)
