"""Evaluation orchestration loops for the redteam app generation pipeline.

Provides eval-audit iteration, hardening rounds, final regression evaluation,
and supporting helpers.  Extracted from ``app_pipeline.py`` so that the
controller and other orchestration layers can invoke these loops without
pulling in the full generation-pipeline dependency graph.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from agentlab.benchmarks.redteam.phase_ids import (
    PHASE_4B,
    PHASE_5,
    normalize_phase_id,
)
from agentlab.benchmarks.redteam.pipeline_state import (
    _expected_eval_iteration_dir,
    _normalize_result_dir,
    _phase_iteration_state,
    _phase_progress_state,
    load_pipeline_state,
    write_pipeline_state,
)
from agentlab.benchmarks.redteam.utils import (
    sha256_file as _sha256_file,
    utc_timestamp as _generation_timestamp,
    write_json as _write_json,
    write_text as _write_text,
)

from agentlab.benchmarks.redteam.pipeline_config import (
    DEFAULT_AUDIT_EVERY,
    DEFAULT_FUNCTIONAL_THRESHOLD,
    DEFAULT_HARDENING_ROUNDS,
    DEFAULT_MAX_EVAL_ITERATIONS,
    DEFAULT_READINESS_BACKEND,
    DEFAULT_REAL_TASK_THRESHOLD,
    DEFAULT_TASKS_PER_HARDENING_ROUND,
)

logger = logging.getLogger(__name__)

# Eval-iteration status constants
EVAL_ITERATION_STATUS_PENDING_EVAL = "pending_eval"
EVAL_ITERATION_STATUS_PENDING_AUDIT = "pending_audit"
EVAL_ITERATION_STATUS_COMPLETE = "complete"

# Hardening stage constants
HARDENING_STAGE_PENDING_GENERATION = "pending_generation"
HARDENING_STAGE_PENDING_SANITY = "pending_sanity"
HARDENING_STAGE_PENDING_EVAL = "pending_eval"
HARDENING_STAGE_PENDING_AUDIT = "pending_audit"
HARDENING_STAGE_COMPLETE = "complete"
HARDENING_STAGES = {
    HARDENING_STAGE_PENDING_GENERATION,
    HARDENING_STAGE_PENDING_SANITY,
    HARDENING_STAGE_PENDING_EVAL,
    HARDENING_STAGE_PENDING_AUDIT,
    HARDENING_STAGE_COMPLETE,
}


# ---------------------------------------------------------------------------
# State / resume helpers
# ---------------------------------------------------------------------------


def _load_task_ids(tasks_file: Path) -> set[str]:
    if not tasks_file.exists():
        return set()
    try:
        tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return set()
    return {str(task.get("id")) for task in tasks if task.get("id")}


def _load_task_definitions(tasks_file: Path) -> list[dict[str, Any]]:
    if not tasks_file.exists():
        return []
    try:
        tasks = json.loads(tasks_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return [task for task in tasks if isinstance(task, dict) and task.get("id")]


def phase4_baseline_snapshot_path(app_dir: Path) -> Path:
    return app_dir / "results" / "phase_4" / "baseline_real_suite_snapshot.json"


def freeze_real_task_baseline(
    app_dir: Path,
    *,
    baseline_results: list[dict[str, Any]],
) -> dict[str, Any]:
    from agentlab.benchmarks.redteam.eval_harness import resolve_verifier_path

    tasks_file = app_dir / "real-tasks.json"
    tasks = _load_task_definitions(tasks_file)
    tasks_by_id: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_id = str(task["id"])
        verify_path = str(task.get("verify", ""))
        resolved_verify = resolve_verifier_path(app_dir, verify_path)
        tasks_by_id[task_id] = {
            "task": task,
            "verify_path": resolved_verify.relative_to(app_dir).as_posix(),
            "verify_sha256": _sha256_file(resolved_verify),
        }

    snapshot = {
        "created_at": _generation_timestamp(),
        "tasks_by_id": tasks_by_id,
        "baseline_results": list(baseline_results),
    }
    snapshot_path = phase4_baseline_snapshot_path(app_dir)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json(snapshot_path, snapshot)
    return snapshot


def load_real_task_baseline_snapshot(app_dir: Path) -> dict[str, Any]:
    path = phase4_baseline_snapshot_path(app_dir)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Ignoring unreadable hardening baseline snapshot: %s", path, exc_info=True)
        return {}


def _real_task_baseline_mutation_error(
    app_dir: Path,
    snapshot: dict[str, Any],
) -> str | None:
    from agentlab.benchmarks.redteam.eval_harness import resolve_verifier_path

    baseline_tasks = snapshot.get("tasks_by_id") or {}
    if not baseline_tasks:
        return None
    current_tasks = {
        str(task["id"]): task
        for task in _load_task_definitions(app_dir / "real-tasks.json")
    }

    for task_id, baseline in baseline_tasks.items():
        current_task = current_tasks.get(task_id)
        if current_task is None:
            return f"Baseline real task {task_id} disappeared during hardening."
        if current_task != baseline.get("task"):
            return f"Baseline real task {task_id} was mutated in place during hardening."
        verify_path = str(current_task.get("verify", ""))
        try:
            resolved_verify = resolve_verifier_path(app_dir, verify_path)
        except Exception as exc:
            return f"Baseline real task {task_id} verifier path is invalid: {exc}"
        relative_verify = resolved_verify.relative_to(app_dir).as_posix()
        if relative_verify != baseline.get("verify_path"):
            return f"Baseline real task {task_id} verifier path changed during hardening."
        if _sha256_file(resolved_verify) != baseline.get("verify_sha256"):
            return f"Baseline real task {task_id} verifier content changed during hardening."
    return None


def _next_available_numeric_dir(
    parent_dir: Path,
    *,
    prefix: str,
    start_index: int,
) -> int:
    index = max(1, start_index)
    while True:
        candidate = parent_dir / f"{prefix}_{index:02d}"
        if not candidate.exists():
            return index
        index += 1


def _load_json_file(path: Path, *, context: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"{context} missing: {path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{context} unreadable: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} is not a JSON object: {path}")
    return payload


# ---------------------------------------------------------------------------
# Eval iteration helpers
# ---------------------------------------------------------------------------


def _validated_resumed_eval_iteration_dir(
    *,
    phase_name: str,
    phase_dir: Path,
    iteration: int,
    iteration_dir_value: str | Path | None,
) -> Path:
    expected_dir = _expected_eval_iteration_dir(phase_dir, iteration)
    if not iteration_dir_value:
        return expected_dir

    persisted_dir = Path(iteration_dir_value)
    phase_root = phase_dir.resolve(strict=False)
    resolved_dir = persisted_dir.resolve(strict=False)
    try:
        resolved_dir.relative_to(phase_root)
    except ValueError as exc:
        raise RuntimeError(
            f"{phase_name} resume iteration_dir resolves outside the expected results tree: {persisted_dir}"
        ) from exc

    expected_resolved = expected_dir.resolve(strict=False)
    if resolved_dir != expected_resolved:
        raise RuntimeError(
            f"{phase_name} resume iteration_dir {persisted_dir} does not match expected {expected_dir}"
        )
    return expected_dir


def _task_result_signature(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": result.get("task_id"),
        "passed": bool(result.get("passed")),
        "message": result.get("message"),
        "instruction": result.get("instruction"),
        "verify": result.get("verify"),
        "exp_dir": result.get("exp_dir"),
        "final_state_path": result.get("final_state_path"),
        "server_events_path": result.get("server_events_path"),
    }


def _read_persisted_iteration_summary(
    app_dir: Path,
    *,
    suite: str,
    backend: str,
    iteration_dir: Path,
) -> dict[str, Any]:
    from agentlab.benchmarks.redteam.eval_harness import load_declared_backend_functional_results

    result_summary_path = iteration_dir / "result_summary.json"
    result_summary = _load_json_file(
        result_summary_path,
        context=f"{suite} pending_audit result summary",
    )
    expected_task_suite = _suite_task_filename(suite)
    if result_summary.get("task_suite") != expected_task_suite:
        raise RuntimeError(
            f"{suite} pending_audit result summary has unexpected task_suite: "
            f"{result_summary.get('task_suite')!r}"
        )
    if result_summary.get("backend") != backend:
        raise RuntimeError(
            f"{suite} pending_audit result summary backend mismatch: "
            f"{result_summary.get('backend')!r} != {backend!r}"
        )
    summary_results_dir = _normalize_result_dir(result_summary.get("results_dir"))
    expected_results_dir = _normalize_result_dir(iteration_dir)
    if summary_results_dir != expected_results_dir:
        raise RuntimeError(
            f"{suite} pending_audit result summary points to {summary_results_dir or '<missing>'}, "
            f"expected {expected_results_dir}"
        )

    backend_results = load_declared_backend_functional_results(app_dir, backend)
    if not backend_results:
        raise RuntimeError(
            f"{suite} pending_audit functional_results.json missing declared backend results for {backend}"
        )
    if backend_results.get("backend") not in (None, backend):
        raise RuntimeError(
            f"{suite} pending_audit functional_results.json backend mismatch: "
            f"{backend_results.get('backend')!r} != {backend!r}"
        )
    suite_key = "function_tasks" if suite == "function" else "real_tasks"
    backend_suite = (backend_results.get(suite_key) or {})
    backend_suite_results = backend_suite.get("results")
    if not isinstance(backend_suite_results, list):
        raise RuntimeError(f"{suite} pending_audit functional_results.json missing readable backend suite results")

    summary_results = result_summary.get("results")
    if not isinstance(summary_results, list):
        raise RuntimeError(f"{suite} pending_audit result summary missing result list")

    summary_by_id = {
        str(result.get("task_id")): result
        for result in summary_results
        if result.get("task_id")
    }
    backend_by_id = {
        str(result.get("task_id")): result
        for result in backend_suite_results
        if result.get("task_id")
    }
    if set(summary_by_id) != set(backend_by_id):
        raise RuntimeError(
            f"{suite} pending_audit backend results do not match preserved iteration task ids"
        )
    for task_id, summary_result in summary_by_id.items():
        backend_result = backend_by_id.get(task_id, {})
        if _task_result_signature(summary_result) != _task_result_signature(backend_result):
            raise RuntimeError(
                f"{suite} pending_audit backend results do not match preserved iteration artifacts for task {task_id}"
            )

    total = result_summary.get("total")
    passed = result_summary.get("passed")
    pass_rate = result_summary.get("pass_rate")
    if total != backend_suite.get("total") or passed != backend_suite.get("passed"):
        raise RuntimeError(
            f"{suite} pending_audit backend aggregate results do not match preserved iteration summary"
        )
    backend_pass_rate = backend_results.get(
        "function_pass_rate" if suite == "function" else "real_pass_rate"
    )
    if pass_rate != backend_pass_rate:
        raise RuntimeError(
            f"{suite} pending_audit backend pass_rate does not match preserved iteration summary"
        )

    return {
        "ran": True,
        "backend": backend,
        "agent_config": result_summary.get("agent_config", ""),
        "pass_rate": pass_rate,
        "total": total,
        "passed": passed,
        "results_dir": str(iteration_dir),
        "results": summary_results,
        "timestamp": result_summary.get("timestamp"),
        "error": None,
    }


def _hardening_should_audit(
    round_num: int,
    *,
    hardening_rounds: int,
    audit_every: int,
) -> bool:
    is_last_round = round_num == hardening_rounds
    return is_last_round if audit_every <= 0 else (round_num % audit_every == 0 or is_last_round)


def _phase4_round_dir(phase_dir: Path, round_num: int) -> Path:
    return phase_dir / f"round_{round_num:02d}"


def _hardening_round_state(progress_state: dict[str, Any]) -> dict[str, Any]:
    return dict(progress_state.get("round_state") or {})


def _load_round_result_summary(
    *,
    round_dir: Path,
    backend: str,
) -> dict[str, Any]:
    result_summary = _load_json_file(
        round_dir / "result_summary.json",
        context="hardening round result summary",
    )
    if result_summary.get("task_suite") != "real-tasks":
        raise RuntimeError(
            f"Hardening round result summary has unexpected task_suite: {result_summary.get('task_suite')!r}"
        )
    if result_summary.get("backend") != backend:
        raise RuntimeError(
            f"Hardening round result summary backend mismatch: {result_summary.get('backend')!r} != {backend!r}"
        )
    if _normalize_result_dir(result_summary.get("results_dir")) != _normalize_result_dir(round_dir):
        raise RuntimeError(
            "Hardening round result summary points at a different results_dir than the persisted round directory"
        )
    results = result_summary.get("results")
    if not isinstance(results, list):
        raise RuntimeError("Hardening round result summary missing result list")
    return {
        "ran": True,
        "backend": backend,
        "agent_config": result_summary.get("agent_config", ""),
        "pass_rate": result_summary.get("pass_rate"),
        "total": result_summary.get("total", 0),
        "passed": result_summary.get("passed", 0),
        "results_dir": str(round_dir),
        "results": results,
        "timestamp": result_summary.get("timestamp"),
        "error": None,
    }


def _validate_hardening_round_state(
    *,
    round_state: dict[str, Any],
    phase_dir: Path,
    hardening_rounds: int,
    audit_every: int,
) -> dict[str, Any]:
    try:
        round_num = int(round_state.get("round") or 0)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("Hardening round_state is missing a valid round number") from exc
    if round_num <= 0:
        raise RuntimeError("Hardening round_state is missing a valid round number")
    if round_num > hardening_rounds:
        raise RuntimeError(
            f"Hardening round_state references round {round_num} beyond configured limit {hardening_rounds}"
        )
    stage = str(round_state.get("stage") or "")
    if stage not in HARDENING_STAGES:
        raise RuntimeError(f"Hardening round_state has invalid stage {stage!r}")
    round_dir_value = round_state.get("round_dir")
    round_dir = Path(round_dir_value) if round_dir_value else _phase4_round_dir(phase_dir, round_num)
    if round_dir != _phase4_round_dir(phase_dir, round_num):
        raise RuntimeError("Hardening round_state round_dir does not match the expected round directory")
    if not round_dir.exists():
        raise RuntimeError(f"Hardening round_state references missing round directory: {round_dir}")
    new_task_ids = [str(task_id) for task_id in (round_state.get("new_task_ids") or []) if str(task_id)]
    should_audit = bool(round_state.get("should_audit"))
    expected_should_audit = _hardening_should_audit(
        round_num,
        hardening_rounds=hardening_rounds,
        audit_every=audit_every,
    )
    if should_audit != expected_should_audit:
        raise RuntimeError("Hardening round_state should_audit does not match configured audit cadence")
    if stage in {
        HARDENING_STAGE_PENDING_SANITY,
        HARDENING_STAGE_PENDING_EVAL,
        HARDENING_STAGE_PENDING_AUDIT,
        HARDENING_STAGE_COMPLETE,
    } and not new_task_ids:
        raise RuntimeError(f"Hardening round_state for round {round_num} is missing persisted new_task_ids")
    if stage == HARDENING_STAGE_PENDING_AUDIT and not (round_dir / "result_summary.json").exists():
        raise RuntimeError(
            f"Hardening round_state for round {round_num} is pending_audit but result_summary.json is missing"
        )
    return {
        "round": round_num,
        "round_dir": str(round_dir),
        "stage": stage,
        "new_task_ids": new_task_ids,
        "should_audit": should_audit,
    }


def _should_reuse_eval_iteration(
    state: dict[str, Any],
    *,
    phase_name: str,
    iteration: int,
) -> bool:
    phase_state = _phase_iteration_state(state, phase_name)
    try:
        current_iteration = int(phase_state.get("iteration", state.get("current_iteration", 0)) or 0)
    except (TypeError, ValueError):
        return False
    return (
        _normalize_resume_phase(state.get("current_phase")) == _normalize_resume_phase(phase_name)
        and current_iteration == iteration
        and (
            phase_state.get("status") or state.get("iteration_status")
        ) in {
            EVAL_ITERATION_STATUS_PENDING_EVAL,
            EVAL_ITERATION_STATUS_PENDING_AUDIT,
        }
    )


def _resolve_eval_iteration(
    app_dir: Path,
    *,
    phase_name: str,
    phase_dir: Path,
    start_iteration: int,
    logs_dir: str | Path | None = None,
) -> tuple[int, Path]:
    state = load_pipeline_state(app_dir, logs_dir=logs_dir)
    phase_state = _phase_iteration_state(state, phase_name)
    if (
        phase_state.get("status") in {
            EVAL_ITERATION_STATUS_PENDING_EVAL,
            EVAL_ITERATION_STATUS_PENDING_AUDIT,
        }
        and isinstance(phase_state.get("iteration"), int)
        and phase_state["iteration"] > 0
    ):
        iteration = int(phase_state["iteration"])
        iteration_dir = phase_state.get("iteration_dir")
        return iteration, _validated_resumed_eval_iteration_dir(
            phase_name=phase_name,
            phase_dir=phase_dir,
            iteration=iteration,
            iteration_dir_value=iteration_dir,
        )

    iteration = _next_available_numeric_dir(
        phase_dir,
        prefix="iter",
        start_index=start_iteration,
    )
    return iteration, _expected_eval_iteration_dir(phase_dir, iteration)


def _load_current_suite_summary(
    app_dir: str | Path,
    *,
    suite: str,
    backend: str,
    require_declared_backend: bool = False,
) -> dict[str, Any]:
    from agentlab.benchmarks.redteam.eval_harness import (
        load_backend_functional_results,
        load_declared_backend_functional_results,
    )

    loader = load_declared_backend_functional_results if require_declared_backend else load_backend_functional_results
    results = loader(app_dir, backend)
    if require_declared_backend and not results:
        raise RuntimeError(f"{suite} evaluation resume missing declared backend results for {backend}")
    if results.get("backend") not in (None, backend):
        raise RuntimeError(
            f"{suite} evaluation resume backend mismatch: {results.get('backend')!r} != {backend!r}"
        )
    suite_key = "function_tasks" if suite == "function" else "real_tasks"
    pass_rate_key = "function_pass_rate" if suite == "function" else "real_pass_rate"
    suite_summary = (results.get(suite_key) or {})
    suite_results = suite_summary.get("results")
    if require_declared_backend and not isinstance(suite_results, list):
        raise RuntimeError(f"{suite} evaluation resume missing readable backend suite results for {backend}")
    suite_results = suite_results or []
    return {
        "ran": bool(results),
        "backend": backend,
        "agent_config": "",
        "pass_rate": results.get(pass_rate_key),
        "total": suite_summary.get("total", len(suite_results)),
        "passed": suite_summary.get("passed", sum(1 for result in suite_results if result.get("passed"))),
        "results_dir": suite_summary.get("results_dir", ""),
        "results": suite_results,
        "timestamp": results.get("timestamp"),
        "error": None,
    }


def load_backend_readiness_baseline(
    app_dir: str | Path,
    *,
    backend: str,
) -> dict[str, Any]:
    from agentlab.benchmarks.redteam.eval_harness import load_backend_functional_results

    return dict(load_backend_functional_results(app_dir, backend).get("readiness_baseline") or {})


def ensure_readiness_baseline(
    app_dir: str | Path,
    *,
    backend: str,
) -> dict[str, Any]:
    from agentlab.benchmarks.redteam.eval_harness import persist_readiness_baseline

    return persist_readiness_baseline(app_dir, backend=backend)


def _load_eval_result_summary_artifact(
    *,
    summary_path: Path,
    expected_task_suite: str,
    backend: str,
    context: str,
) -> dict[str, Any]:
    result_summary = _load_json_file(summary_path, context=context)
    if result_summary.get("task_suite") != expected_task_suite:
        raise RuntimeError(
            f"{context} has unexpected task_suite: {result_summary.get('task_suite')!r}"
        )
    if result_summary.get("backend") != backend:
        raise RuntimeError(
            f"{context} backend mismatch: {result_summary.get('backend')!r} != {backend!r}"
        )
    expected_results_dir = _normalize_result_dir(summary_path.parent)
    if _normalize_result_dir(result_summary.get("results_dir")) != expected_results_dir:
        raise RuntimeError(
            f"{context} points to {_normalize_result_dir(result_summary.get('results_dir')) or '<missing>'}, "
            f"expected {expected_results_dir}"
        )
    results = result_summary.get("results")
    if not isinstance(results, list):
        raise RuntimeError(f"{context} missing result list")
    return {
        "ran": True,
        "backend": backend,
        "agent_config": result_summary.get("agent_config", ""),
        "pass_rate": result_summary.get("pass_rate"),
        "total": result_summary.get("total", 0),
        "passed": result_summary.get("passed", 0),
        "results_dir": str(summary_path.parent),
        "results": results,
        "timestamp": result_summary.get("timestamp"),
        "error": None,
    }


def resume_backend_error(
    *,
    requested_backend: str,
    manifest: dict[str, Any],
    pipeline_state: dict[str, Any],
) -> str | None:
    from agentlab.benchmarks.redteam.eval_harness import normalize_functional_backend

    requested_backend = normalize_functional_backend(requested_backend)
    stored_sources: list[tuple[str, str]] = []
    manifest_backend = ((manifest.get("generation") or {}).get("functional_backend"))
    pipeline_backend = pipeline_state.get("backend")
    if manifest_backend:
        stored_sources.append(("app_manifest.json", normalize_functional_backend(manifest_backend)))
    if pipeline_backend:
        stored_sources.append(("pipeline_state.json", normalize_functional_backend(pipeline_backend)))
    if not stored_sources:
        return None

    distinct_backends = {value for _, value in stored_sources}
    if len(distinct_backends) > 1:
        details = ", ".join(f"{source}={value}" for source, value in stored_sources)
        return f"Resume backend mismatch between persisted artifacts ({details})"
    stored_backend = stored_sources[0][1]
    if stored_backend != requested_backend:
        details = ", ".join(f"{source}={value}" for source, value in stored_sources)
        return (
            f"Resume requested backend {requested_backend!r} does not match persisted backend "
            f"{stored_backend!r} ({details})"
        )
    return None


def _load_resumed_hardening_result(
    app_dir: Path,
    *,
    backend: str,
    hardening_rounds: int,
    audit_every: int,
) -> dict[str, Any]:
    if hardening_rounds <= 0:
        return {
            "ran": False,
            "rounds": [],
            "audit_summary_path": "",
            "error": None,
        }

    phase_dir = app_dir / "results" / "phase_4"
    baseline_snapshot_path = phase4_baseline_snapshot_path(app_dir)
    if not baseline_snapshot_path.exists():
        return {
            "ran": True,
            "rounds": [],
            "audit_summary_path": "",
            "error": "Missing hardening baseline snapshot for resume.",
        }
    baseline_snapshot = load_real_task_baseline_snapshot(app_dir)
    if not baseline_snapshot:
        return {
            "ran": True,
            "rounds": [],
            "audit_summary_path": "",
            "error": "Unreadable hardening baseline snapshot for resume.",
        }

    expected_round_dirs = {
        _phase4_round_dir(phase_dir, round_num)
        for round_num in range(1, hardening_rounds + 1)
    }
    actual_round_dirs = {
        path
        for path in phase_dir.glob("round_*")
        if path.is_dir()
    }
    if actual_round_dirs != expected_round_dirs:
        missing = sorted(str(path.name) for path in expected_round_dirs - actual_round_dirs)
        unexpected = sorted(str(path.name) for path in actual_round_dirs - expected_round_dirs)
        details: list[str] = []
        if missing:
            details.append(f"missing rounds: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected rounds: {', '.join(unexpected)}")
        return {
            "ran": True,
            "rounds": [],
            "audit_summary_path": "",
            "error": f"Inconsistent hardening rounds for resume ({'; '.join(details)})",
        }

    rounds: list[dict[str, Any]] = []
    last_audit_summary_path = ""
    try:
        for round_num in range(1, hardening_rounds + 1):
            round_dir = _phase4_round_dir(phase_dir, round_num)
            evaluation = _load_round_result_summary(
                round_dir=round_dir,
                backend=backend,
            )
            should_audit = _hardening_should_audit(
                round_num,
                hardening_rounds=hardening_rounds,
                audit_every=audit_every,
            )
            audit_summary_path = round_dir / "audit_summary.md"
            if should_audit and not audit_summary_path.exists():
                raise RuntimeError(
                    f"Hardening round {round_num} missing audit_summary.md for resume"
                )
            if audit_summary_path.exists():
                last_audit_summary_path = str(audit_summary_path)
            rounds.append(
                {
                    "round": round_num,
                    "results_dir": str(round_dir),
                    "new_task_ids": [
                        str(result.get("task_id"))
                        for result in (evaluation.get("results") or [])
                        if result.get("task_id")
                    ],
                    "evaluation": evaluation,
                    "audit_summary_path": str(audit_summary_path) if audit_summary_path.exists() else "",
                    "error": None,
                }
            )
    except RuntimeError as exc:
        return {
            "ran": True,
            "rounds": rounds,
            "audit_summary_path": last_audit_summary_path,
            "error": str(exc),
        }

    return {
        "ran": True,
        "rounds": rounds,
        "audit_summary_path": last_audit_summary_path,
        "error": None,
    }


def _load_resumed_final_regression_result(
    app_dir: Path,
    *,
    backend: str,
) -> dict[str, Any]:
    regression_root = app_dir / "results" / "phase_5" / "final_regression"
    baseline_snapshot = load_real_task_baseline_snapshot(app_dir)
    if not baseline_snapshot:
        return {
            "ran": False,
            "passed": False,
            "function": {},
            "real": {},
            "regressions": {"function": [], "real": []},
            "triage_path": "",
            "error": "Missing hardening baseline snapshot for final regression resume.",
        }

    readiness_baseline = load_backend_readiness_baseline(app_dir, backend=backend)
    if not readiness_baseline:
        return {
            "ran": False,
            "passed": False,
            "function": {},
            "real": {},
            "regressions": {"function": [], "real": []},
            "triage_path": "",
            "error": "Missing readiness baseline for final regression resume.",
        }
    if readiness_baseline.get("backend") not in (None, backend):
        return {
            "ran": False,
            "passed": False,
            "function": {},
            "real": {},
            "regressions": {"function": [], "real": []},
            "triage_path": "",
            "error": "Readiness baseline backend mismatch for final regression resume.",
        }
    if not isinstance((readiness_baseline.get("function_tasks") or {}).get("results"), list):
        return {
            "ran": False,
            "passed": False,
            "function": {},
            "real": {},
            "regressions": {"function": [], "real": []},
            "triage_path": "",
            "error": "Readiness baseline missing function task results for final regression resume.",
        }
    if not isinstance((readiness_baseline.get("real_tasks") or {}).get("results"), list):
        return {
            "ran": False,
            "passed": False,
            "function": {},
            "real": {},
            "regressions": {"function": [], "real": []},
            "triage_path": "",
            "error": "Readiness baseline missing real task results for final regression resume.",
        }

    try:
        function_summary = _load_eval_result_summary_artifact(
            summary_path=regression_root / "function" / "result_summary.json",
            expected_task_suite="function-tasks",
            backend=backend,
            context="final regression function result summary",
        )
        real_summary = _load_eval_result_summary_artifact(
            summary_path=regression_root / "real" / "result_summary.json",
            expected_task_suite="real-tasks",
            backend=backend,
            context="final regression real result summary",
        )
    except RuntimeError as exc:
        return {
            "ran": False,
            "passed": False,
            "function": {},
            "real": {},
            "regressions": {"function": [], "real": []},
            "triage_path": "",
            "error": str(exc),
        }

    regressions = {
        "function": _regression_failures(
            list((readiness_baseline.get("function_tasks") or {}).get("results") or []),
            function_summary.get("results", []),
        ),
        "real": _regression_failures(
            list(baseline_snapshot.get("baseline_results") or []),
            real_summary.get("results", []),
        ),
    }
    passed = not regressions["function"] and not regressions["real"]
    triage_path = regression_root / "final_regression_triage.md"
    return {
        "ran": True,
        "passed": passed,
        "function": function_summary,
        "real": real_summary,
        "regressions": regressions,
        "triage_path": str(triage_path) if triage_path.exists() else "",
        "error": None,
    }


# ---------------------------------------------------------------------------
# Internal helpers (shared across suite helpers and orchestrators)
# ---------------------------------------------------------------------------


def _normalize_resume_phase(phase: str | None) -> str | None:
    return normalize_phase_id(phase) if phase else None


def _app_identity(app_dir: str | Path, *, fallback: str = "") -> str:
    from agentlab.benchmarks.redteam.app_artifacts import load_app_manifest

    app_dir = Path(app_dir)
    manifest = load_app_manifest(app_dir)
    return str(manifest.get("app_id") or fallback or app_dir.name)


# ---------------------------------------------------------------------------
# Suite helpers
# ---------------------------------------------------------------------------


def run_task_validation_loop(
    app_dir: str | Path,
    *,
    suite: str,
    backend: str = DEFAULT_READINESS_BACKEND,
    agent_config: str | None = None,
    workers: int = 1,
    repetitions: int = 1,
    output_dir: str | Path | None = None,
    task_id: str | None = None,
    runtime_app_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run a task-suite validation loop through the functional harness."""
    from agentlab.benchmarks.redteam.eval_harness import (
        DEFAULT_AGENT_CONFIG,
        run_functional_eval,
    )

    agent_config = agent_config or DEFAULT_AGENT_CONFIG
    eval_kwargs: dict[str, Any] = {
        "app_dir": app_dir,
        "task_suite": suite,
        "agent_config": agent_config,
        "backend": backend,
        "workers": workers,
        "repetitions": repetitions,
        "output_dir": output_dir,
        "task_id": task_id,
    }
    if runtime_app_dir is not None:
        eval_kwargs["runtime_app_dir"] = runtime_app_dir
    try:
        eval_result = run_functional_eval(**eval_kwargs)
    except TypeError as exc:
        if "runtime_app_dir" not in str(exc):
            raise
        eval_kwargs.pop("runtime_app_dir", None)
        eval_result = run_functional_eval(**eval_kwargs)
    return {
        "ran": True,
        "backend": backend,
        "agent_config": agent_config,
        "pass_rate": eval_result.get("pass_rate"),
        "total": eval_result.get("total", 0),
        "passed": eval_result.get("passed", 0),
        "results_dir": eval_result.get("results_dir", ""),
        "results": eval_result.get("results", []),
        "timestamp": eval_result.get("timestamp"),
        "error": None,
    }


def _suite_task_filename(suite: str) -> str:
    return "function-tasks" if suite == "function" else "real-tasks"


def _suite_phase_name(suite: str) -> str:
    return "phase_2b" if suite == "function" else "phase_3b"


def _suite_threshold(suite: str) -> float:
    return DEFAULT_FUNCTIONAL_THRESHOLD if suite == "function" else DEFAULT_REAL_TASK_THRESHOLD


def summarize_eval_failures(
    *,
    suite: str,
    phase_name: str,
    iteration: int,
    eval_summary: dict[str, Any],
    output_dir: str | Path,
    threshold: float,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    failures: list[dict[str, Any]] = []
    for result in eval_summary.get("results", []) or []:
        if result.get("passed"):
            continue
        exp_dir = Path(result.get("exp_dir", ""))
        summary_info_path = exp_dir / "summary_info.json" if exp_dir else None
        transcript_path = ""
        if summary_info_path and summary_info_path.exists():
            try:
                summary_info = json.loads(summary_info_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                summary_info = {}
            transcript_path = str(summary_info.get("transcript_path", ""))
        failures.append(
            {
                "task_id": result.get("task_id"),
                "instruction": result.get("instruction"),
                "message": result.get("message"),
                "verify": result.get("verify"),
                "exp_dir": result.get("exp_dir"),
                "final_state_path": result.get("final_state_path"),
                "server_events_path": result.get("server_events_path"),
                "transcript_path": transcript_path,
                "attempts": result.get("attempts", []),
            }
        )

    summary = {
        "suite": suite,
        "phase_name": phase_name,
        "iteration": iteration,
        "threshold": threshold,
        "pass_rate": eval_summary.get("pass_rate"),
        "total": eval_summary.get("total", 0),
        "passed": eval_summary.get("passed", 0),
        "failed": len(failures),
        "results_dir": eval_summary.get("results_dir", ""),
        "failures": failures,
    }
    _write_json(output_dir / "failure_summary.json", summary)
    return summary


def materialize_repair_prompt(
    *,
    app_dir: str | Path,
    suite: str,
    iteration_dir: str | Path,
    eval_summary: dict[str, Any],
    failure_summary: dict[str, Any],
) -> str:
    from agentlab.benchmarks.redteam.prompt_loading import load_prompt_template

    template_name = (
        "audit-function-eval.md" if suite == "function" else "audit-real-eval.md"
    )
    try:
        prompt = load_prompt_template(
            template_name,
            behavior_id=_app_identity(app_dir),
            app_id=_app_identity(app_dir),
            iteration=str(failure_summary["iteration"]),
            pass_rate=str(eval_summary.get("pass_rate")),
            total=str(eval_summary.get("total", 0)),
            passed=str(eval_summary.get("passed", 0)),
            threshold=str(failure_summary["threshold"]),
            results_dir=str(iteration_dir),
            result_summary_path=str(Path(iteration_dir) / "result_summary.json"),
            failure_summary_path=str(Path(iteration_dir) / "failure_summary.json"),
            failure_summary_json=json.dumps(failure_summary, indent=2, ensure_ascii=False),
        )
    except FileNotFoundError:
        prompt = (
            f"Audit {suite} evaluation failures for {_app_identity(app_dir)}.\n\n"
            f"Results dir: {iteration_dir}\n"
            f"Result summary: {Path(iteration_dir) / 'result_summary.json'}\n"
            f"Failure summary: {Path(iteration_dir) / 'failure_summary.json'}\n"
        )
    _write_text(Path(iteration_dir) / "repair_prompt.md", prompt)
    return prompt


# ---------------------------------------------------------------------------
# Main orchestrators
# ---------------------------------------------------------------------------


def run_eval_audit_loop(
    app_dir: str | Path,
    *,
    suite: str,
    backend: str = DEFAULT_READINESS_BACKEND,
    agent_config: str | None = None,
    max_iterations: int = DEFAULT_MAX_EVAL_ITERATIONS,
    threshold: float | None = None,
    start_iteration: int = 1,
    update_state: bool = True,
    logs_dir: str | Path | None = None,
    workers: int = 1,
    repetitions: int = 1,
    runtime_app_dir: str | Path | None = None,
    runtime_variant_subdir: str = "benign",
) -> dict[str, Any]:
    from agentlab.benchmarks.redteam.audit import audit_app
    from agentlab.benchmarks.redteam.eval_harness import DEFAULT_AGENT_CONFIG
    from agentlab.benchmarks.redteam.runtime_ops import materialize_app_runtime

    app_dir = Path(app_dir)
    threshold = _suite_threshold(suite) if threshold is None else threshold
    agent_config = agent_config or DEFAULT_AGENT_CONFIG
    phase_name = _suite_phase_name(suite)
    iterations: list[dict[str, Any]] = []
    last_summary: dict[str, Any] | None = None
    stop_reason = "max_iterations_exceeded"
    phase_dir = app_dir / "results" / phase_name
    iteration = start_iteration
    pipeline_state = load_pipeline_state(app_dir, logs_dir=logs_dir) if update_state else {}
    persisted_iteration = 0
    persisted_iteration_status = ""

    def _persist_eval_state(
        *,
        current_iteration: int,
        iteration_status: str,
        results_dir: Path,
        last_audit_summary_path: str | None = None,
        stop_reason_override: str | None = None,
    ) -> None:
        nonlocal persisted_iteration, persisted_iteration_status, pipeline_state
        persisted_iteration = current_iteration
        persisted_iteration_status = iteration_status
        if update_state:
            pipeline_state = write_pipeline_state(
                app_dir,
                current_phase=phase_name,
                logs_dir=logs_dir,
                current_iteration=current_iteration,
                backend=backend,
                last_results_dirs={phase_name: str(results_dir)},
                last_audit_summary_path=last_audit_summary_path,
                stop_reason=stop_reason_override,
                iteration_status=iteration_status,
                phase_iteration_phase=phase_name,
                phase_iteration_iteration=current_iteration,
                phase_iteration_status=iteration_status,
                phase_iteration_dir=str(results_dir),
            )

    def _materialize_iteration_summary(summary: dict[str, Any], *, output_dir: Path) -> None:
        payload = {
            "task_suite": _suite_task_filename(suite),
            "backend": backend,
            "agent_config": agent_config,
            "results_dir": str(output_dir),
            "pass_rate": summary.get("pass_rate"),
            "total": summary.get("total", 0),
            "passed": summary.get("passed", 0),
            "results": summary.get("results", []),
            "timestamp": summary.get("timestamp"),
        }
        _write_json(output_dir / "result_summary.json", payload)

    for _ in range(start_iteration, max_iterations + 1):
        phase_iteration = _phase_iteration_state(pipeline_state, phase_name)
        reusing_iteration = _should_reuse_eval_iteration(
            pipeline_state,
            phase_name=phase_name,
            iteration=iteration,
        )
        iteration_status = ""
        if reusing_iteration:
            iteration_dir = _validated_resumed_eval_iteration_dir(
                phase_name=phase_name,
                phase_dir=phase_dir,
                iteration=iteration,
                iteration_dir_value=phase_iteration.get("iteration_dir"),
            )
            iteration_status = str(phase_iteration.get("status") or pipeline_state.get("iteration_status") or "")
        else:
            iteration, iteration_dir = _resolve_eval_iteration(
                app_dir,
                phase_name=phase_name,
                phase_dir=phase_dir,
                start_iteration=iteration,
                logs_dir=logs_dir,
            )
        iteration_dir.mkdir(parents=True, exist_ok=True)
        if iteration_status == EVAL_ITERATION_STATUS_PENDING_AUDIT:
            eval_summary = _read_persisted_iteration_summary(
                app_dir,
                suite=suite,
                backend=backend,
                iteration_dir=iteration_dir,
            )
            last_summary = dict(eval_summary)
            failure_summary = summarize_eval_failures(
                suite=suite,
                phase_name=phase_name,
                iteration=iteration,
                eval_summary=eval_summary,
                output_dir=iteration_dir,
                threshold=threshold,
            )
            materialize_repair_prompt(
                app_dir=app_dir,
                suite=suite,
                iteration_dir=iteration_dir,
                eval_summary=eval_summary,
                failure_summary=failure_summary,
            )
        else:
            _persist_eval_state(
                current_iteration=iteration,
                iteration_status=EVAL_ITERATION_STATUS_PENDING_EVAL,
                results_dir=iteration_dir,
            )

            eval_kwargs: dict[str, Any] = {
                "app_dir": app_dir,
                "suite": suite,
                "backend": backend,
                "agent_config": agent_config,
                "workers": workers,
                "repetitions": repetitions,
                "output_dir": iteration_dir,
            }
            if runtime_app_dir is not None:
                eval_kwargs["runtime_app_dir"] = materialize_app_runtime(
                    app_dir,
                    variant_subdir=runtime_variant_subdir,
                    runtime_dir=runtime_app_dir,
                )
            try:
                eval_summary = run_task_validation_loop(**eval_kwargs)
            except TypeError as exc:
                if "runtime_app_dir" not in str(exc):
                    raise
                eval_kwargs.pop("runtime_app_dir", None)
                eval_summary = run_task_validation_loop(**eval_kwargs)
            last_summary = dict(eval_summary)
            _materialize_iteration_summary(last_summary, output_dir=iteration_dir)
            failure_summary = summarize_eval_failures(
                suite=suite,
                phase_name=phase_name,
                iteration=iteration,
                eval_summary=eval_summary,
                output_dir=iteration_dir,
                threshold=threshold,
            )
            materialize_repair_prompt(
                app_dir=app_dir,
                suite=suite,
                iteration_dir=iteration_dir,
                eval_summary=eval_summary,
                failure_summary=failure_summary,
            )

        iteration_record: dict[str, Any] = {
            "iteration": iteration,
            "pass_rate": eval_summary.get("pass_rate"),
            "total": eval_summary.get("total", 0),
            "passed": eval_summary.get("passed", 0),
            "results_dir": str(iteration_dir),
            "failure_summary_path": str(iteration_dir / "failure_summary.json"),
            "audit_summary_path": "",
            "repair_prompt_path": str(iteration_dir / "repair_prompt.md"),
            "fixed_count": 0,
            "stop_reason": "",
        }

        pass_rate = eval_summary.get("pass_rate")
        if isinstance(pass_rate, (int, float)) and pass_rate >= threshold:
            stop_reason = "threshold_reached"
            iteration_record["stop_reason"] = stop_reason
            _persist_eval_state(
                current_iteration=iteration,
                iteration_status=EVAL_ITERATION_STATUS_COMPLETE,
                results_dir=iteration_dir,
            )
            iterations.append(iteration_record)
            break

        if iteration_status != EVAL_ITERATION_STATUS_PENDING_AUDIT:
            _persist_eval_state(
                current_iteration=iteration,
                iteration_status=EVAL_ITERATION_STATUS_PENDING_AUDIT,
                results_dir=iteration_dir,
            )
        audit_report = audit_app(
            app_dir=app_dir,
            task_suite=_suite_task_filename(suite),
            agent_config=agent_config,
            backend=backend,
            max_fix_attempts=3,
            max_iterations=1,
            results_dir=iteration_dir,
        )
        audit_summary_path = Path(audit_report.results_dir) / "audit_summary.md"
        iteration_record["audit_summary_path"] = str(audit_summary_path)
        iteration_record["fixed_count"] = audit_report.fixed_count
        last_summary = _load_current_suite_summary(app_dir, suite=suite, backend=backend)
        last_summary["agent_config"] = agent_config
        last_summary["results_dir"] = str(iteration_dir)
        _materialize_iteration_summary(last_summary, output_dir=iteration_dir)
        summarize_eval_failures(
            suite=suite,
            phase_name=phase_name,
            iteration=iteration,
            eval_summary=last_summary,
            output_dir=iteration_dir,
            threshold=threshold,
        )

        if audit_report.fixed_count <= 0:
            stop_reason = "audit_no_changes"
            iteration_record["stop_reason"] = stop_reason
            _persist_eval_state(
                current_iteration=iteration,
                iteration_status=EVAL_ITERATION_STATUS_COMPLETE,
                results_dir=iteration_dir,
                last_audit_summary_path=str(audit_summary_path),
            )
            iterations.append(iteration_record)
            break

        if iteration < max_iterations:
            _persist_eval_state(
                current_iteration=iteration,
                iteration_status=EVAL_ITERATION_STATUS_COMPLETE,
                results_dir=iteration_dir,
                last_audit_summary_path=str(audit_summary_path),
            )
            iteration_record["stop_reason"] = "retry_after_audit"
        else:
            _persist_eval_state(
                current_iteration=iteration,
                iteration_status=EVAL_ITERATION_STATUS_COMPLETE,
                results_dir=iteration_dir,
                last_audit_summary_path=str(audit_summary_path),
            )
        iterations.append(iteration_record)
        iteration += 1
    else:
        iteration = max(iteration, max_iterations)

    if last_summary is None:
        return {
            "ran": False,
            "backend": backend,
            "agent_config": agent_config,
            "pass_rate": None,
            "total": 0,
            "passed": 0,
            "results_dir": "",
            "iterations": [],
            "stop_reason": "no_iterations",
            "error": "Evaluation loop did not run.",
        }

    if stop_reason == "max_iterations_exceeded" and iterations:
        iterations[-1]["stop_reason"] = stop_reason

    result = {
        "ran": True,
        "backend": backend,
        "agent_config": agent_config,
        "pass_rate": last_summary.get("pass_rate"),
        "total": last_summary.get("total", 0),
        "passed": last_summary.get("passed", 0),
        "results_dir": last_summary.get("results_dir", ""),
        "results": last_summary.get("results", []),
        "timestamp": last_summary.get("timestamp"),
        "iterations": iterations,
        "stop_reason": stop_reason,
        "error": None,
    }
    if update_state:
        write_pipeline_state(
            app_dir,
            current_phase=phase_name,
            logs_dir=logs_dir,
            current_iteration=persisted_iteration or iteration,
            backend=backend,
            last_results_dirs={phase_name: str(last_summary.get("results_dir", ""))},
            stop_reason=stop_reason,
            iteration_status=persisted_iteration_status or "",
        )
    return result


def _collect_hardening_result_summaries(app_dir: Path) -> list[dict[str, Any]]:
    summary_paths = sorted(
        list((app_dir / "results" / "phase_3b").glob("iter_*/result_summary.json"))
        + list((app_dir / "results" / "phase_4").glob("round_*/result_summary.json"))
    )
    summaries: list[dict[str, Any]] = []
    for summary_path in summary_paths:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if summary.get("task_suite") != "real-tasks":
            continue
        summary["_summary_path"] = str(summary_path)
        summaries.append(summary)
    return summaries


def build_hardening_analysis(app_dir: Path) -> str:
    summaries = _collect_hardening_result_summaries(app_dir)
    if not summaries:
        return "No prior evaluation results were found."

    lines = []
    for summary in summaries:
        lines.extend(
            [
                f"Summary path: {summary.get('_summary_path', '')}",
                f"Task suite: {summary.get('task_suite', 'unknown')}",
                f"Pass rate: {summary.get('pass_rate', 0)}",
                f"Passed: {summary.get('passed', 0)}/{summary.get('total', 0)}",
                "Per-task outcomes:",
            ]
        )
        for result in summary.get("results", []) or []:
            transcript_path = ""
            exp_dir = Path(result.get("exp_dir", ""))
            summary_info_path = exp_dir / "summary_info.json" if exp_dir else None
            if summary_info_path.exists():
                try:
                    summary_info = json.loads(summary_info_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    summary_info = {}
                transcript_path = str(summary_info.get("transcript_path", ""))
            lines.append(
                "- {task_id}: passed={passed} difficulty={difficulty} message={message} transcript={transcript}".format(
                    task_id=result.get("task_id", ""),
                    passed=result.get("passed"),
                    difficulty=result.get("difficulty", ""),
                    message=result.get("message", ""),
                    transcript=transcript_path,
                )
            )
        lines.append("")
    return "\n".join(lines)


def run_hardening_rounds(
    app_dir: str | Path,
    *,
    behavior_id: str,
    template_dir: str | Path | None = None,
    backend: str = DEFAULT_READINESS_BACKEND,
    agent_config: str | None = None,
    hardening_rounds: int = DEFAULT_HARDENING_ROUNDS,
    tasks_per_hardening_round: int = DEFAULT_TASKS_PER_HARDENING_ROUND,
    audit_every: int = DEFAULT_AUDIT_EVERY,
    start_round: int = 1,
    update_state: bool = True,
    logs_dir: str | Path | None = None,
    phase_name: str = PHASE_4B,
) -> dict[str, Any]:
    from agentlab.benchmarks.redteam.audit import audit_app
    from agentlab.benchmarks.redteam.claude_code import run_claude_code
    from agentlab.benchmarks.redteam.eval_harness import DEFAULT_AGENT_CONFIG
    from agentlab.benchmarks.redteam.prompt_loading import (
        _task_guide_prompt_kwargs,
        ensure_trusted_server_template,
        load_prompt_template,
    )

    app_dir = Path(app_dir)
    phase_name = normalize_phase_id(phase_name)
    agent_config = agent_config or DEFAULT_AGENT_CONFIG
    rounds: list[dict[str, Any]] = []
    last_audit_summary_path = ""

    if hardening_rounds <= 0:
        return {
            "ran": False,
            "rounds": [],
            "audit_summary_path": "",
            "error": None,
        }

    real_tasks_file = app_dir / "real-tasks.json"
    phase_dir = app_dir / "results" / "phase_4"
    phase_dir.mkdir(parents=True, exist_ok=True)
    pipeline_state = load_pipeline_state(app_dir, logs_dir=logs_dir) if update_state else {}
    hardening_progress = _phase_progress_state(pipeline_state, phase_name)
    persisted_round_state = _hardening_round_state(hardening_progress)
    pending_hardening_task_ids = {
        str(task_id)
        for task_id in (hardening_progress.get("pending_task_ids") or [])
        if str(task_id)
    }
    baseline_snapshot_path = Path(
        hardening_progress.get("baseline_snapshot_path")
        or phase4_baseline_snapshot_path(app_dir)
    )
    baseline_snapshot = load_real_task_baseline_snapshot(app_dir)
    if not baseline_snapshot:
        if start_round > 1:
            return {
                "ran": True,
                "rounds": [],
                "audit_summary_path": "",
                "error": "Missing hardening baseline snapshot for resume.",
            }
        else:
            baseline_snapshot = freeze_real_task_baseline(
                app_dir,
                baseline_results=_load_current_suite_summary(
                    app_dir,
                    suite="real",
                    backend=backend,
                ).get("results", []),
            )

    def _persist_hardening_state(
        *,
        current_round: int,
        current_round_dir: Path,
        round_state: dict[str, Any],
    ) -> None:
        if update_state:
            write_pipeline_state(
                app_dir,
                current_phase=phase_name,
                logs_dir=logs_dir,
                current_iteration=current_round,
                backend=backend,
                last_results_dirs={phase_name: str(current_round_dir)},
                last_audit_summary_path=last_audit_summary_path or None,
                phase_progress_phase=phase_name,
                phase_progress={
                    "pending_task_ids": sorted(pending_hardening_task_ids),
                    "baseline_snapshot_path": str(baseline_snapshot_path),
                    "round_state": round_state,
                },
            )

    try:
        if persisted_round_state:
            active_round_state = _validate_hardening_round_state(
                round_state=persisted_round_state,
                phase_dir=phase_dir,
                hardening_rounds=hardening_rounds,
                audit_every=audit_every,
            )
        else:
            active_round_state = {}
    except RuntimeError as exc:
        return {
            "ran": True,
            "rounds": [],
            "audit_summary_path": "",
            "error": str(exc),
        }

    round_num = start_round
    while round_num <= hardening_rounds:
        if active_round_state:
            round_num = int(active_round_state["round"])
            round_dir = Path(active_round_state["round_dir"])
            stage = str(active_round_state["stage"])
            new_ids = list(active_round_state.get("new_task_ids") or [])
            should_audit = bool(active_round_state.get("should_audit"))
            if stage == HARDENING_STAGE_PENDING_AUDIT and new_ids:
                pending_hardening_task_ids.update(new_ids)
        else:
            round_num = _next_available_numeric_dir(
                phase_dir,
                prefix="round",
                start_index=round_num,
            )
            round_dir = _phase4_round_dir(phase_dir, round_num)
            round_dir.mkdir(parents=True, exist_ok=True)
            should_audit = _hardening_should_audit(
                round_num,
                hardening_rounds=hardening_rounds,
                audit_every=audit_every,
            )
            stage = HARDENING_STAGE_PENDING_GENERATION
            new_ids = []
            _persist_hardening_state(
                current_round=round_num,
                current_round_dir=round_dir,
                round_state={
                    "round": round_num,
                    "round_dir": str(round_dir),
                    "stage": stage,
                    "new_task_ids": [],
                    "should_audit": should_audit,
                },
            )

        round_result: dict[str, Any] = {
            "round": round_num,
            "results_dir": str(round_dir),
            "new_task_ids": list(new_ids),
            "evaluation": None,
            "audit_summary_path": "",
            "error": None,
        }

        if stage == HARDENING_STAGE_PENDING_GENERATION:
            known_ids = _load_task_ids(real_tasks_file)
            prompt = load_prompt_template(
                "harden-tasks-from-trajectories.md",
                behavior_id=behavior_id,
                app_id=_app_identity(app_dir, fallback=behavior_id),
                round_number=str(round_num),
                tasks_per_round=str(tasks_per_hardening_round),
                results_dir=str(app_dir / "results"),
                hardening_analysis=build_hardening_analysis(app_dir),
                **_task_guide_prompt_kwargs(working_dir=app_dir),
            )
            rc, stdout, stderr = run_claude_code(
                prompt=prompt,
                working_dir=app_dir,
                timeout=3600,
            )
            server_error = ensure_trusted_server_template(app_dir, template_dir)
            if server_error:
                round_result["error"] = server_error
                rounds.append(round_result)
                break
            if rc != 0:
                round_result["error"] = f"Hardening prompt failed (rc={rc}): {(stderr or stdout)[-500:]}"
                rounds.append(round_result)
                break

            mutation_error = _real_task_baseline_mutation_error(app_dir, baseline_snapshot)
            if mutation_error:
                round_result["error"] = mutation_error
                rounds.append(round_result)
                break

            new_ids = sorted(_load_task_ids(real_tasks_file) - known_ids)
            round_result["new_task_ids"] = new_ids
            if not new_ids:
                round_result["error"] = "No new tasks generated."
                rounds.append(round_result)
                break

            stage = HARDENING_STAGE_PENDING_SANITY
            _persist_hardening_state(
                current_round=round_num,
                current_round_dir=round_dir,
                round_state={
                    "round": round_num,
                    "round_dir": str(round_dir),
                    "stage": stage,
                    "new_task_ids": new_ids,
                    "should_audit": should_audit,
                },
            )

        if stage == HARDENING_STAGE_PENDING_SANITY:
            if not new_ids:
                round_result["error"] = f"Hardening round {round_num} is missing persisted new_task_ids before sanity."
                rounds.append(round_result)
                break
            # _run_suite_sanity_fix_loop remains in app_pipeline.py; import lazily
            from agentlab.benchmarks.redteam.app_pipeline import _run_suite_sanity_fix_loop

            sanity_result = _run_suite_sanity_fix_loop(
                app_dir=app_dir,
                behavior_id=behavior_id,
                suite="real",
                template_dir=template_dir,
                fix_iterations=3,
                task_id=",".join(new_ids),
            )
            if not sanity_result["sanity_passed"]:
                round_result["error"] = (sanity_result["errors"] or ["Hardening sanity repair failed."])[-1]
                rounds.append(round_result)
                break
            stage = HARDENING_STAGE_PENDING_EVAL
            _persist_hardening_state(
                current_round=round_num,
                current_round_dir=round_dir,
                round_state={
                    "round": round_num,
                    "round_dir": str(round_dir),
                    "stage": stage,
                    "new_task_ids": new_ids,
                    "should_audit": should_audit,
                },
            )

        if stage == HARDENING_STAGE_PENDING_EVAL:
            if not new_ids:
                round_result["error"] = f"Hardening round {round_num} is missing persisted new_task_ids before evaluation."
                rounds.append(round_result)
                break
            eval_summary = run_task_validation_loop(
                app_dir=app_dir,
                suite="real",
                backend=backend,
                agent_config=agent_config,
                workers=1,
                repetitions=1,
                output_dir=round_dir,
                task_id=",".join(new_ids),
            )
            round_result["evaluation"] = eval_summary
            pending_hardening_task_ids.update(new_ids)
            stage = HARDENING_STAGE_PENDING_AUDIT if should_audit else HARDENING_STAGE_COMPLETE
            _persist_hardening_state(
                current_round=round_num,
                current_round_dir=round_dir,
                round_state={
                    "round": round_num,
                    "round_dir": str(round_dir),
                    "stage": stage,
                    "new_task_ids": new_ids,
                    "should_audit": should_audit,
                },
            )

        if stage == HARDENING_STAGE_PENDING_AUDIT:
            if not new_ids:
                round_result["error"] = f"Hardening round {round_num} is missing persisted new_task_ids before audit."
                rounds.append(round_result)
                break
            if round_result["evaluation"] is None:
                round_result["evaluation"] = _load_round_result_summary(
                    round_dir=round_dir,
                    backend=backend,
                )
            audit_target_ids = set(pending_hardening_task_ids)
            audit_report = audit_app(
                app_dir=app_dir,
                task_suite="real-tasks",
                agent_config=agent_config,
                backend=backend,
                max_fix_attempts=3,
                max_iterations=1,
                results_dir=round_dir,
                allowed_task_ids=audit_target_ids,
            )
            last_audit_summary_path = str(Path(audit_report.results_dir) / "audit_summary.md")
            round_result["audit_summary_path"] = last_audit_summary_path
            pending_hardening_task_ids.difference_update(audit_report.audited_task_ids)
            stage = HARDENING_STAGE_COMPLETE
            _persist_hardening_state(
                current_round=round_num,
                current_round_dir=round_dir,
                round_state={
                    "round": round_num,
                    "round_dir": str(round_dir),
                    "stage": stage,
                    "new_task_ids": new_ids,
                    "should_audit": should_audit,
                },
            )

        rounds.append(round_result)
        active_round_state = {}
        round_num += 1

    if update_state:
        write_pipeline_state(
            app_dir,
            current_phase=phase_name,
            logs_dir=logs_dir,
            current_iteration=(rounds[-1]["round"] + 1) if rounds and not rounds[-1].get("error") else (rounds[-1]["round"] if rounds else 0),
            backend=backend,
            last_results_dirs={phase_name: str((app_dir / "results" / "phase_4").resolve())},
            last_audit_summary_path=last_audit_summary_path,
            phase_progress_phase=phase_name,
            phase_progress={
                "pending_task_ids": sorted(pending_hardening_task_ids),
                "baseline_snapshot_path": str(baseline_snapshot_path),
                "round_state": {},
            },
        )

    return {
        "ran": True,
        "rounds": rounds,
        "audit_summary_path": last_audit_summary_path,
        "error": next((round_result["error"] for round_result in rounds if round_result.get("error")), None),
    }


def _regression_failures(
    baseline_results: list[dict[str, Any]],
    current_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_passed = {
        result.get("task_id"): bool(result.get("passed"))
        for result in baseline_results
        if result.get("task_id")
    }
    current_by_id = {
        result.get("task_id"): result
        for result in current_results
        if result.get("task_id")
    }
    regressions: list[dict[str, Any]] = []
    for task_id, was_passing in baseline_passed.items():
        if not was_passing:
            continue
        current = current_by_id.get(task_id) or {}
        if current.get("passed") is True:
            continue
        regressions.append(
            {
                "task_id": task_id,
                "previously_passed": True,
                "current_message": current.get("message", "Task missing from regression run."),
                "exp_dir": current.get("exp_dir", ""),
            }
        )
    return regressions


def run_final_regression_eval(
    app_dir: str | Path,
    *,
    behavior_id: str,
    backend: str = DEFAULT_READINESS_BACKEND,
    agent_config: str | None = None,
    baseline_results: dict[str, list[dict[str, Any]]] | None = None,
    update_state: bool = True,
    logs_dir: str | Path | None = None,
) -> dict[str, Any]:
    from agentlab.benchmarks.redteam.eval_harness import DEFAULT_AGENT_CONFIG
    from agentlab.benchmarks.redteam.prompt_loading import load_prompt_template

    app_dir = Path(app_dir)
    agent_config = agent_config or DEFAULT_AGENT_CONFIG
    regression_root = app_dir / "results" / "phase_5" / "final_regression"
    regression_root.mkdir(parents=True, exist_ok=True)
    baseline_snapshot = load_real_task_baseline_snapshot(app_dir)
    if not baseline_snapshot:
        return {
            "ran": False,
            "passed": False,
            "function": {},
            "real": {},
            "regressions": {"function": [], "real": []},
            "triage_path": "",
            "error": "Missing hardening baseline snapshot for final regression.",
        }

    if update_state:
        write_pipeline_state(
            app_dir,
            current_phase=PHASE_5,
            logs_dir=logs_dir,
            current_iteration=1,
            backend=backend,
            last_results_dirs={PHASE_5: str(regression_root)},
            regression_status="running",
        )

    function_summary = run_task_validation_loop(
        app_dir=app_dir,
        suite="function",
        backend=backend,
        agent_config=agent_config,
        workers=1,
        repetitions=1,
        output_dir=regression_root / "function",
    )
    mutation_error = _real_task_baseline_mutation_error(app_dir, baseline_snapshot)
    if mutation_error:
        real_summary = {
            "ran": False,
            "backend": backend,
            "agent_config": agent_config,
            "pass_rate": None,
            "total": 0,
            "passed": 0,
            "results_dir": str(regression_root / "real"),
            "results": [],
            "timestamp": None,
            "error": mutation_error,
        }
    else:
        baseline_real_ids = sorted((baseline_snapshot.get("tasks_by_id") or {}).keys())
        if baseline_real_ids:
            real_summary = run_task_validation_loop(
                app_dir=app_dir,
                suite="real",
                backend=backend,
                agent_config=agent_config,
                workers=1,
                repetitions=1,
                output_dir=regression_root / "real",
                task_id=",".join(baseline_real_ids),
            )
        else:
            real_summary = {
                "ran": True,
                "backend": backend,
                "agent_config": agent_config,
                "pass_rate": 0.0,
                "total": 0,
                "passed": 0,
                "results_dir": str(regression_root / "real"),
                "results": [],
                "timestamp": _generation_timestamp(),
                "error": None,
            }

    baseline_results = baseline_results or {}
    regressions = {
        "function": _regression_failures(
            baseline_results.get("function", []),
            function_summary.get("results", []),
        ),
        "real": _regression_failures(
            list(baseline_snapshot.get("baseline_results") or baseline_results.get("real", [])),
            real_summary.get("results", []),
        ),
    }
    if mutation_error:
        regressions["real"].append(
            {
                "task_id": "__baseline_mutation__",
                "previously_passed": True,
                "current_message": mutation_error,
                "exp_dir": "",
            }
        )
    regression_passed = not regressions["function"] and not regressions["real"]
    triage_path = regression_root / "final_regression_triage.md"
    if not regression_passed:
        try:
            triage = load_prompt_template(
                "final-regression-triage.md",
                behavior_id=behavior_id,
                app_id=_app_identity(app_dir, fallback=behavior_id),
                function_results_dir=str(function_summary.get("results_dir", "")),
                real_results_dir=str(real_summary.get("results_dir", "")),
                regressions_json=json.dumps(regressions, indent=2, ensure_ascii=False),
            )
        except FileNotFoundError:
            triage = json.dumps(regressions, indent=2, ensure_ascii=False)
        _write_text(triage_path, triage)

    if update_state:
        write_pipeline_state(
            app_dir,
            current_phase=PHASE_5,
            logs_dir=logs_dir,
            current_iteration=1,
            backend=backend,
            last_results_dirs={
                "final_regression_function": str(function_summary.get("results_dir", "")),
                "final_regression_real": str(real_summary.get("results_dir", "")),
            },
            stop_reason="regression_passed" if regression_passed else "regression_failed",
            regression_status="passed" if regression_passed else "failed",
        )

    return {
        "ran": True,
        "passed": regression_passed,
        "function": function_summary,
        "real": real_summary,
        "regressions": regressions,
        "triage_path": str(triage_path) if triage_path.exists() else "",
        "error": None,
    }
