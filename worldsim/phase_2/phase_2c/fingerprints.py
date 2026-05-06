"""Phase 2c fingerprint and idempotency behavior."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _idempotency_decision(
    existing: Any,
    *,
    current_fingerprint: dict[str, str],
    ttl_hours: float | None,
    force_reverify: bool,
) -> tuple[str, str | None]:
    """Return ``("skip", reason)`` to reuse the prior result or
    ``("verify", None)`` to re-run.

    Matches the truth table in §3.7:

    - Missing feasibility field -> verify.
    - ``verified`` + fingerprint matches -> skip (reason=fingerprint_match);
      force_reverify overrides.
    - ``verified`` + fingerprint drifts -> re-verify unless TTL covers it
      (reason=ttl_hours).
    - ``infeasible`` (any fingerprint) -> re-verify (platform may have
      changed its policy since).
    - ``unverified`` -> verify.
    """
    if force_reverify:
        return ("verify", None)
    if not isinstance(existing, dict):
        return ("verify", None)
    status = existing.get("status")
    if status != "verified":
        return ("verify", None)
    prior_fp = existing.get("host_fingerprint") or {}
    if not isinstance(prior_fp, dict):
        return ("verify", None)
    if _fingerprints_match(prior_fp, current_fingerprint):
        return ("skip", "fingerprint_match")
    if ttl_hours is not None:
        verified_at = existing.get("verified_at")
        age = _hours_since(verified_at)
        if age is not None and age <= ttl_hours:
            return ("skip", "ttl_hours")
    return ("verify", None)


def _fingerprints_match(a: dict[str, Any], b: dict[str, Any]) -> bool:
    keys = (
        "host_config",
        "instances_digest",
        "editor_commit",
        "dataset_commit",
        "task_content_hash",
    )
    return all(str(a.get(k, "")) == str(b.get(k, "")) for k in keys)


def _hours_since(timestamp: Any) -> float | None:
    if not isinstance(timestamp, str) or not timestamp:
        return None
    try:
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    delta = datetime.now(tz=UTC) - parsed
    return delta.total_seconds() / 3600.0


def _task_content_hash(editor_calls: list[Any], *, exposure_contract: Any = None) -> str:
    projection = _exposure_contract_fingerprint_projection(exposure_contract)
    payload: Any = (
        {"editor_calls": editor_calls, "exposure_contract": projection}
        if projection is not None
        else editor_calls
    )
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _exposure_contract_fingerprint_projection(contract: Any) -> dict[str, Any] | None:
    if not isinstance(contract, dict):
        return None
    keys = (
        "contract_id",
        "site",
        "kind",
        "mode",
        "benign_read_url",
        "editor_method",
        "target_surface_id",
        "payload_arg",
        "editor_args_template",
        "verification",
        "eligibility",
    )
    return {key: contract.get(key) for key in keys if key in contract}


def _host_fingerprint(instances_label: str, instances: list[dict[str, Any]]) -> dict[str, str]:
    commit = _git_head_short()
    return {
        "host_config": instances_label,
        "instances_digest": _instances_digest(instances),
        "editor_commit": commit,
        "dataset_commit": commit,
    }


def _instances_digest(instances: list[dict[str, Any]]) -> str:
    canonical = json.dumps(instances, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _git_head_short() -> str:
    override = os.environ.get("WORLDSIM_EDITOR_COMMIT_OVERRIDE")
    if override:
        return override.strip()
    repo_root = Path(__file__).resolve().parents[3]
    sync_commit = _sync_stamp_commit(repo_root)
    if sync_commit:
        return sync_commit
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.decode("utf-8", errors="replace").strip() or "unknown"


def _sync_stamp_commit(repo_root: Path) -> str | None:
    stamp_path = repo_root / ".worldsim_sync_stamp.json"
    try:
        payload = json.loads(stamp_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    local_git = payload.get("local_git")
    if not isinstance(local_git, dict):
        return None
    sha = local_git.get("sha")
    if not isinstance(sha, str):
        return None
    sha = sha.strip()
    if not sha:
        return None
    return sha[:12]


def _first_method(task: dict[str, Any]) -> str:
    seed = task.get("adversarial_data_seed") or {}
    if not isinstance(seed, dict):
        return ""
    calls = seed.get("editor_calls")
    if isinstance(calls, list) and calls and isinstance(calls[0], dict):
        return str(calls[0].get("method", ""))
    return ""

__all__ = [
    "_exposure_contract_fingerprint_projection",
    "_fingerprints_match",
    "_first_method",
    "_git_head_short",
    "_host_fingerprint",
    "_hours_since",
    "_idempotency_decision",
    "_instances_digest",
    "_sync_stamp_commit",
    "_task_content_hash",
]
