"""Re-score an existing Phase 3 run with the agent-response transform.

Reads a prior Phase 3 output directory (e.g. ``logs/phase_3_gemini-3-flash/``),
iterates each task's trajectory dir, pulls the agent's final text, runs the
4-strategy :func:`warp_taskgen.agent_response_transform.transform_agent_response`
on it, and re-runs the WebArena-Verified ``AgentResponseEvaluator`` (and only
that evaluator -- ``NetworkEventEvaluator`` configs are skipped because
re-scoring is about adapter-rigor parity, not re-visiting a fresh network
trace) with the transformed value.

Output: ``<phase_3_dir>/results_rescored.json`` alongside the original
``results.json``. The original is never mutated. Prints a 1-line summary:

    N tasks: {original_pass}/{rescored_pass} pass, flipped={\u0394}
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from warp_taskgen.agent_response_transform import detect_strategy, transform_agent_response
from warp_taskgen.rewards.vendor_webarena import (
    WEBARENA_EVAL_MODULE,
    WEBARENA_EVAL_PYTHON_ENV,
    webarena_eval_python_override,
)
from warp_taskgen.rewards.webarena_sites import build_supported_webarena_environments

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_benign_tasks(state_dir: Path) -> dict[str, dict[str, Any]]:
    """Return a task_id -> task_dict mapping from ``logs/phase_1/benign_tasks.json``."""
    phase_1 = state_dir / "phase_1" / "benign_tasks.json"
    if phase_1.exists():
        data = _read_json(phase_1)
        if isinstance(data, list):
            return {str(t.get("id")): t for t in data if isinstance(t, dict)}
    logger.warning("benign_tasks.json not found at %s", phase_1)
    return {}


def _find_final_result(task_dir: Path) -> str | None:
    """Pull the agent's final text answer from a Phase 3 task directory.

    Prefers ``final_response.json`` (the canonical per-task artifact written
    by ``warp_taskgen.trajectory.save_result``), falling back to ``result.json``
    which also carries ``final_result`` as part of the full
    ``run_task`` return dict.
    """
    for name in ("final_response.json", "result.json"):
        candidate = task_dir / name
        if not candidate.exists():
            continue
        try:
            data = _read_json(candidate)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(data, dict):
            value = data.get("final_result")
            if isinstance(value, str):
                return value
    return None


# ---------------------------------------------------------------------------
# URL placeholder resolution for the evaluator's config-validation step
# ---------------------------------------------------------------------------


def _normalize_environments(
    url_placeholders: dict[str, str] | None,
) -> dict[str, list[str]]:
    """Translate ``__SITE__`` placeholders to a site-name -> URLs mapping.

    Mirrors :func:`warp_taskgen.rewards.vendor_webarena._build_webarena_environment_payload` but
    takes the placeholder mapping directly (rescore does not thread through
    a live benchmark instance).
    """
    return build_supported_webarena_environments(url_placeholders)


def _default_placeholders(instances_path: Path | None) -> dict[str, str]:
    """Read URL placeholders from an instances JSON if supplied."""
    if instances_path is None or not instances_path.exists():
        return {}
    try:
        data = _read_json(instances_path)
    except (OSError, json.JSONDecodeError):
        return {}
    mapping = data.get("url_placeholders") if isinstance(data, dict) else None
    return mapping if isinstance(mapping, dict) else {}


# ---------------------------------------------------------------------------
# Evaluator wiring
# ---------------------------------------------------------------------------


def _agent_response_configs_only(reward: dict[str, Any]) -> list[dict[str, Any]]:
    """Return only the ``AgentResponseEvaluator`` entries from a reward spec."""
    eval_configs = reward.get("eval") or []
    return [
        cfg
        for cfg in eval_configs
        if isinstance(cfg, dict) and cfg.get("evaluator") == "AgentResponseEvaluator"
    ]


def _build_inprocess_evaluator(environments: dict[str, list[str]]) -> Any:
    """Instantiate ``WebArenaVerified`` in the current Python env.

    Raises ``ImportError`` when the vendor package isn't installed. Callers
    should fall back to the subprocess path in that case.
    """
    from webarena_verified.api import WebArenaVerified
    from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
    from webarena_verified.types.task import WebArenaSite

    env_configs: dict[Any, Any] = {}
    for site_name, urls in environments.items():
        if not urls:
            continue
        try:
            site_key = WebArenaSite(site_name)
        except ValueError:
            continue
        env_configs[site_key] = EnvironmentConfig(urls=urls)
    config = WebArenaVerifiedConfig(environments=env_configs if env_configs else None)
    return WebArenaVerified(config=config)


def _status_from_evaluators_results(
    evaluators_results: list[dict[str, Any]] | None,
) -> tuple[bool, str]:
    """Parse subprocess ``evaluators_results`` and pick the AgentResponseEvaluator.

    We filter to the AgentResponseEvaluator status specifically, ignoring
    NetworkEventEvaluator outcomes. That is the whole point of the rescore:
    adapter-rigor parity on the text-response path only.
    """
    if not evaluators_results:
        return False, "vendor returned no evaluator results"

    for er in evaluators_results:
        if not isinstance(er, dict):
            continue
        if er.get("evaluator_name") != "AgentResponseEvaluator":
            continue
        status = str(er.get("status", "")).upper()
        err = er.get("error_msg")
        msg = f"[AgentResponseEvaluator] {status}"
        if err:
            msg += f": {err}"
        return status == "SUCCESS", msg

    return False, "vendor returned no AgentResponseEvaluator result"


def _minimal_har_trace_entry() -> dict[str, Any]:
    """Return a minimal HAR-shaped entry that satisfies the vendor's parser.

    ``load_trace_from_content`` refuses empty lists and recognizes two formats:
    Playwright ``resource-snapshot`` events, or HAR entries with ``request``
    and ``response`` keys. Since we filter out the NetworkEventEvaluator
    result afterwards, the content itself is a no-op; we just need it to pass
    the format gate.
    """
    return {
        "request": {"method": "GET", "url": "about:blank", "headers": [], "cookies": []},
        "response": {"status": 0, "headers": [], "cookies": [], "content": {}},
    }


def _run_subprocess(
    *,
    python_executable: str,
    task_id: int,
    agent_response: Any,
    environments: dict[str, list[str]],
) -> tuple[bool, str]:
    """Invoke the vendor evaluator in its own Python env and pick out the ARS status.

    Mirrors :func:`warp_taskgen.rewards.vendor_webarena._run_webarena_verified_subprocess` for the
    payload shape. Passes a minimal HAR-shaped placeholder trace only so the
    vendor's trace parser accepts the input; any NetworkEventEvaluator
    outcome is discarded by :func:`_status_from_evaluators_results`.
    """
    payload = {
        "task_id": task_id,
        "agent_response": agent_response,
        "network_trace": [_minimal_har_trace_entry()],
        "environments": environments,
    }
    try:
        completed = subprocess.run(
            [python_executable, "-m", WEBARENA_EVAL_MODULE],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except Exception as exc:
        return False, f"evaluator subprocess failed to start: {exc}"

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        return False, f"evaluator subprocess failed: {stderr}"

    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return False, f"evaluator subprocess returned invalid JSON: {exc}"

    return _status_from_evaluators_results(response.get("evaluators_results"))


def _inprocess_evaluate(
    *,
    evaluator: Any,
    task_id: int,
    agent_response_value: Any,
) -> tuple[bool, str]:
    """In-process evaluation; runs all configured evaluators, then filters."""
    result = evaluator.evaluate_task(
        task_id=task_id,
        agent_response=agent_response_value,
        network_trace=[_minimal_har_trace_entry()],
    )
    ers = [
        {
            "evaluator_name": er.evaluator_name,
            "status": er.status if isinstance(er.status, str) else er.status.value,
            "error_msg": er.error_msg,
        }
        for er in getattr(result, "evaluators_results", []) or []
    ]
    return _status_from_evaluators_results(ers)


def _select_evaluator_runner(
    environments: dict[str, list[str]],
) -> tuple[str, Any | None]:
    """Decide whether we run in-process or via the subprocess helper.

    Returns a tuple ``(mode, handle)`` where ``mode`` is ``"subprocess"`` or
    ``"inprocess"`` and ``handle`` is either the python executable path
    (subprocess) or a prepared ``WebArenaVerified`` evaluator (in-process).
    """
    subprocess_python = webarena_eval_python_override()
    if not subprocess_python:
        # Look for the vendored venv as a default.
        repo_root = Path(__file__).resolve().parents[2]
        candidate = (
            repo_root / "packages" / "warp-taskgen-webarena-verified" / ".venv" / "bin" / "python"
        )
        if candidate.exists():
            subprocess_python = str(candidate)

    if subprocess_python:
        return "subprocess", subprocess_python

    try:
        return "inprocess", _build_inprocess_evaluator(environments)
    except ImportError as exc:
        raise RuntimeError(
            "vendor webarena_verified unavailable and "
            f"{WEBARENA_EVAL_PYTHON_ENV} is unset; cannot rescore"
        ) from exc


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def _resolve_task_dir(entry: dict[str, Any], phase_3_dir: Path) -> Path | None:
    """Return the trajectory dir for a task, tolerating relocated runs.

    ``results.json`` records ``trajectory_dir`` as an absolute-ish path that
    may have been moved (e.g. archived from ``logs/phase_3/`` into
    ``logs/phase_3_gemini-3-flash/``). Try the recorded path first, then the
    stem under the supplied ``phase_3_dir``.
    """
    raw = entry.get("trajectory_dir")
    if isinstance(raw, str) and raw:
        recorded = Path(raw)
        if recorded.exists():
            return recorded
        # Try remapping: preserve everything after the phase_3 parent.
        parts = recorded.parts
        for idx, part in enumerate(parts):
            if part.startswith("phase_3"):
                candidate = phase_3_dir.joinpath(*parts[idx + 1 :])
                if candidate.exists():
                    return candidate
                break
    # Fallback: scan children of phase_3_dir for a dir matching the task_id.
    task_id = str(entry.get("task_id", ""))
    for child in phase_3_dir.glob(f"*/{task_id}"):
        if child.is_dir():
            return child
    return None


def rescore_phase_3(
    *,
    phase_3_dir: Path,
    instances_path: Path | None = None,
) -> dict[str, Any]:
    """Run the rescore pass and write ``results_rescored.json``.

    Returns a summary dict useful for tests / callers.
    """
    results_path = phase_3_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"expected results.json at {results_path}")

    results = _read_json(results_path)
    if not isinstance(results, list):
        raise ValueError(f"{results_path} is not a JSON array")

    # Repo root is three levels up from this module file
    # (warp_taskgen/phases/phase_3_rescore.py -> repo root).
    repo_root = Path(__file__).resolve().parents[2]
    state_dir = repo_root / "logs"
    tasks_by_id = _load_benign_tasks(state_dir)

    placeholders = _default_placeholders(instances_path)
    environments = _normalize_environments(placeholders)
    mode, handle = _select_evaluator_runner(environments)

    rescored: list[dict[str, Any]] = []
    flipped = 0
    orig_pass = 0
    resc_pass = 0

    for entry in results:
        if not isinstance(entry, dict):
            continue
        task_id_raw = entry.get("task_id")
        task_id_str = str(task_id_raw)
        try:
            task_id_int = int(task_id_str.split("__")[0])
        except (TypeError, ValueError):
            rescored.append(
                {
                    "task_id": task_id_str,
                    "original_pass": bool(entry.get("passed", False)),
                    "rescored_pass": bool(entry.get("passed", False)),
                    "original_score": 1.0 if entry.get("passed") else 0.0,
                    "rescored_score": 1.0 if entry.get("passed") else 0.0,
                    "extraction_strategy_used": None,
                    "transformed_value": None,
                    "reason": "non-integer task_id; skipped",
                }
            )
            continue

        original_pass = bool(entry.get("passed", False))
        orig_pass += 1 if original_pass else 0

        task = tasks_by_id.get(task_id_str) or tasks_by_id.get(task_id_str.split("__")[0])
        if not task:
            rescored.append(
                {
                    "task_id": task_id_str,
                    "original_pass": original_pass,
                    "rescored_pass": original_pass,
                    "original_score": 1.0 if original_pass else 0.0,
                    "rescored_score": 1.0 if original_pass else 0.0,
                    "extraction_strategy_used": None,
                    "transformed_value": None,
                    "reason": "benign task not found; kept original",
                }
            )
            resc_pass += 1 if original_pass else 0
            continue

        reward = task.get("reward_function") or {}
        ars_configs = _agent_response_configs_only(reward)
        if not ars_configs:
            rescored.append(
                {
                    "task_id": task_id_str,
                    "original_pass": original_pass,
                    "rescored_pass": original_pass,
                    "original_score": 1.0 if original_pass else 0.0,
                    "rescored_score": 1.0 if original_pass else 0.0,
                    "extraction_strategy_used": None,
                    "transformed_value": None,
                    "reason": "task has no AgentResponseEvaluator; kept original",
                }
            )
            resc_pass += 1 if original_pass else 0
            continue

        task_dir = _resolve_task_dir(entry, phase_3_dir)
        if task_dir is None:
            rescored.append(
                {
                    "task_id": task_id_str,
                    "original_pass": original_pass,
                    "rescored_pass": original_pass,
                    "original_score": 1.0 if original_pass else 0.0,
                    "rescored_score": 1.0 if original_pass else 0.0,
                    "extraction_strategy_used": None,
                    "transformed_value": None,
                    "reason": "trajectory dir not found; kept original",
                }
            )
            resc_pass += 1 if original_pass else 0
            continue

        final_text = _find_final_result(task_dir)
        transformed = (
            transform_agent_response(final_text)
            if isinstance(final_text, str) and final_text.strip()
            else None
        )
        strategy_used = (
            detect_strategy(final_text)
            if isinstance(final_text, str) and final_text.strip()
            else None
        )

        if transformed is None:
            # Nothing to flip: without a successful transform, the AgentResponseEvaluator
            # receives no new information. Keep the original verdict, annotate why.
            rescored.append(
                {
                    "task_id": task_id_str,
                    "original_pass": original_pass,
                    "rescored_pass": original_pass,
                    "original_score": 1.0 if original_pass else 0.0,
                    "rescored_score": 1.0 if original_pass else 0.0,
                    "extraction_strategy_used": None,
                    "transformed_value": None,
                    "reason": (
                        "transform yielded no JSON; no adapter-rigor gap to close"
                        if final_text
                        else "no final_result text; kept original"
                    ),
                }
            )
            resc_pass += 1 if original_pass else 0
            continue

        if mode == "subprocess":
            passed, msg = _run_subprocess(
                python_executable=handle,
                task_id=task_id_int,
                agent_response=transformed,
                environments=environments,
            )
        else:
            passed, msg = _inprocess_evaluate(
                evaluator=handle,
                task_id=task_id_int,
                agent_response_value=transformed,
            )
        if passed:
            resc_pass += 1
        if passed != original_pass:
            flipped += 1

        rescored.append(
            {
                "task_id": task_id_str,
                "original_pass": original_pass,
                "rescored_pass": passed,
                "original_score": 1.0 if original_pass else 0.0,
                "rescored_score": 1.0 if passed else 0.0,
                "extraction_strategy_used": strategy_used,
                "transformed_value": transformed,
                "reason": msg,
            }
        )

    out_path = phase_3_dir / "results_rescored.json"
    out_path.write_text(json.dumps(rescored, indent=2) + "\n", encoding="utf-8")

    total = len(rescored)
    summary_line = f"{total} tasks: {orig_pass}/{resc_pass} pass, flipped={flipped}"
    print(summary_line)

    return {
        "total": total,
        "original_pass": orig_pass,
        "rescored_pass": resc_pass,
        "flipped": flipped,
        "output_path": str(out_path),
        "summary": summary_line,
    }


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> int:
    """Argparse entrypoint compatible with ``warp_taskgen.main._dispatch_phase``."""
    phase_3_dir = Path(getattr(args, "phase_3_dir", Path("logs/phase_3_gemini-3-flash")))
    instances_path = getattr(args, "instances", None)
    if instances_path is not None:
        instances_path = Path(instances_path)
    try:
        rescore_phase_3(phase_3_dir=phase_3_dir, instances_path=instances_path)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return 1
    except Exception as exc:
        logger.exception("rescore-phase-3 failed: %s", exc)
        return 1
    return 0
