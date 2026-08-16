"""Phase 2c outcome stanza behavior."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from warp_taskgen.editors import EditorError
from warp_taskgen.phase_2.phase_2c.fingerprints import _first_method
from warp_taskgen.runtime_composition import RequiredSeedCleanupError
from warp_taskgen.seeding import SeedCleanupHandle


def _infeasible_task(
    task: dict[str, Any],
    *,
    kind: str,
    detail: str,
    fingerprint: dict[str, str],
    http_status: int | None,
    response_snippet: str | None,
    attempts: list[dict[str, Any]],
    timestamp: str,
    render_evidence: dict[str, Any] | None = None,
    reachability_evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    result = dict(task)
    error_entry: dict[str, Any] = {
        "call_index": 0,
        "method": _first_method(task),
        "kind": kind,
        "detail": detail,
    }
    if http_status is not None:
        error_entry["http_status"] = http_status
    if response_snippet is not None:
        error_entry["response_snippet"] = response_snippet
    if render_evidence is not None:
        error_entry["render_evidence"] = render_evidence
    if reachability_evidence is not None:
        error_entry["reachability_evidence"] = reachability_evidence
    result["feasibility"] = {
        "status": "infeasible",
        "host_fingerprint": fingerprint,
        "errors": [error_entry],
        "first_failed_at": timestamp,
        "attempts": attempts,
    }
    return result


def _safe_cleanup(
    handle: SeedCleanupHandle | None,
    cleanup_warnings: list[str],
    task_id: Any,
    *,
    raise_on_failure: bool = False,
) -> None:
    if handle is None:
        return
    try:
        handle.cleanup()
    except EditorError as exc:
        warning = f"task={task_id!s} cleanup_failed: {exc.detail}"
        cleanup_warnings.append(warning)
        if raise_on_failure:
            raise RequiredSeedCleanupError(warning) from exc
    except Exception as exc:  # pragma: no cover - defensive
        warning = f"task={task_id!s} cleanup_raised: {exc.__class__.__name__}: {exc}"
        cleanup_warnings.append(warning)
        if raise_on_failure:
            raise RequiredSeedCleanupError(warning) from exc


def _resolve_seed_site(task: dict[str, Any]) -> str:
    """Return the site the adversarial seed actually POSTs against.

    Phase 4 uses the same precedence: ``delivery_channel.delivery_site`` ->
    first editor_call's ``site`` -> ``task["site"]``. A shopping_admin task
    whose payload seeds a product review on the shopping storefront has
    ``delivery_site="shopping"`` and must bind to the shopping instance.
    """
    delivery = task.get("delivery_channel")
    if isinstance(delivery, dict):
        ds = delivery.get("delivery_site")
        if isinstance(ds, str) and ds.strip() and ds.strip().lower() != "none":
            return ds.strip().lower()
    seed = task.get("adversarial_data_seed") or {}
    if isinstance(seed, dict):
        calls = seed.get("editor_calls")
        if isinstance(calls, list):
            for call in calls:
                if isinstance(call, dict):
                    cs = call.get("site")
                    if isinstance(cs, str) and cs.strip():
                        return cs.strip().lower()
    return str(task.get("site", "")).strip().lower()


def _now_iso() -> str:
    return datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def skipped_task_stanza(
    task: dict[str, Any], *, reason: str = "skip_feasibility_flag"
) -> dict[str, Any]:
    """Tag ``task`` with an ``unverified`` feasibility stanza.

    Used by ``--skip-feasibility``.
    """
    result = dict(task)
    result["feasibility"] = {
        "status": "unverified",
        "skipped_at": _now_iso(),
        "reason": reason,
    }
    return result


__all__ = [
    "_infeasible_task",
    "_now_iso",
    "_resolve_seed_site",
    "_safe_cleanup",
    "skipped_task_stanza",
]
