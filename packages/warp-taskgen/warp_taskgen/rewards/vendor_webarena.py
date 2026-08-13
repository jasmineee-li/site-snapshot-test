from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any

from warp_taskgen._paths import find_repo_root
from warp_taskgen.har_converter import (
    NetworkTraceUnavailableError,
    ensure_har_trace,
    strict_runtime_har_trace,
)
from warp_taskgen.rewards.agent_response import _build_agent_response
from warp_taskgen.rewards.webarena_sites import build_supported_webarena_environments

logger = logging.getLogger(__name__)

WEBARENA_EVAL_PYTHON_ENV = "WARP_TASKGEN_WEBARENA_EVAL_PYTHON"
WEBARENA_EVAL_MODULE = "warp_taskgen_webarena_verified.evaluate"


def _is_network_event_evaluator_name(name: Any) -> bool:
    return isinstance(name, str) and name in {"NetworkEventEvaluator", "network_event"}


def _default_eval_python() -> str:
    """Return the repo-relative evaluator venv python when present, else ''.

    WARP Taskgen runs the WebArena Verified evaluator in its own venv
    (conflicting deps vs. the root pyproject; uv workspaces are explicitly
    contraindicated for conflicting deps). When
    ``WARP_TASKGEN_WEBARENA_EVAL_PYTHON`` is unset, fall back to the
    conventional repo-relative location. POSIX only; Windows would need
    ``.venv/Scripts/python.exe``.
    """
    try:
        root = find_repo_root()
    except RuntimeError:
        return ""
    candidate = root / "packages" / "warp-taskgen-webarena-verified" / ".venv" / "bin" / "python"
    if candidate.is_file() and os.access(candidate, os.X_OK):
        return str(candidate)
    return ""


def webarena_eval_python_override() -> str:
    """Return the configured evaluator Python override."""
    return os.environ.get(WEBARENA_EVAL_PYTHON_ENV, "").strip()


def _apply_webarena_vendor_shims(eval_configs: list[dict]) -> list[dict]:
    """Work around three upstream bugs in ServiceNow/webarena-verified v1.2.3.

    Bug 1 (value_normalizer.py:149-151): `normalize_array` raises
    unconditionally when `schema.type != "array"`, ignoring `strict=False`.
    Triggered by 514/812 tasks that use `results_schema: {"type": "null"}`
    when the agent returns non-null data. Rewriting to an array schema makes
    the vendor route non-compliant outputs to FAILURE instead of ERROR, while
    compliant null outputs still early-return via the existing falsy short
    circuit in agent_response_evaluator.py:138-140.

    Bug 2 (same line): `{"type": "object"}` schemas would also crash.
    Currently unused in the dataset but we shim proactively.

    Returns a deep-copied list so the original reward dict is never mutated
    across probe repeats.
    """
    import copy

    patched = copy.deepcopy(eval_configs)
    for cfg in patched:
        if not isinstance(cfg, dict):
            continue
        if _is_network_event_evaluator_name(cfg.get("evaluator")):
            cfg["evaluator"] = "NetworkEventEvaluator"
        if cfg.get("evaluator") != "AgentResponseEvaluator":
            continue
        rs = cfg.get("results_schema")
        if isinstance(rs, dict) and rs.get("type") in {"null", "object"}:
            cfg["results_schema"] = {"type": "array", "items": {"type": "string"}}
    return patched


def _coerce_agent_response_strings(agent_response: dict[str, Any]) -> dict[str, Any]:
    """Work around Bug 3: agent_response_evaluator.py:120 does `.strip()` on
    non-strings. Coerce task_type/status to strings before handing to vendor."""
    if not isinstance(agent_response, dict):
        return agent_response
    for key in ("task_type", "status"):
        val = agent_response.get(key)
        if val is not None and not isinstance(val, str):
            agent_response[key] = str(val)
    return agent_response


def _run_webarena_verified_eval(
    reward: dict[str, Any],
    instance: dict[str, Any],
    agent_result: Any | None,
    network_trace: list[dict] | None,
) -> tuple[bool, str]:
    """Evaluate using the vendor WebArena Verified evaluator.

    Delegates to the ``webarena_verified`` package for full normalization
    (NFKC, unidecode, TM-stripping, type-dispatch across 17 data types, etc.).
    Fail closed if the vendor package is unavailable or the reward spec lacks
    the canonical ``task_id`` needed to locate the evaluator config.
    """
    task_id = reward.get("task_id")
    if task_id is None:
        logger.error("Reward spec missing task_id; refusing non-canonical evaluation")
        return False, "reward spec missing canonical WebArena Verified task_id"

    eval_configs = _apply_webarena_vendor_shims(reward["eval"])
    agent_response = _build_agent_response(eval_configs, agent_result)
    agent_response = _coerce_agent_response_strings(agent_response)
    environments = _build_webarena_environment_payload(instance)
    # AgentResponse-only tasks can use a placeholder trace for parity with the
    # dedicated rescore path. Runtime tasks that require network evidence must
    # fail closed when trace capture is missing or malformed.
    if _reward_requires_network_trace(eval_configs):
        try:
            har_trace = strict_runtime_har_trace(network_trace)
        except NetworkTraceUnavailableError as exc:
            return False, str(exc)
    else:
        har_trace = ensure_har_trace(network_trace)

    subprocess_python = webarena_eval_python_override() or _default_eval_python()
    if subprocess_python:
        return _run_webarena_verified_subprocess(
            python_executable=subprocess_python,
            task_id=task_id,
            agent_response=agent_response,
            network_trace=har_trace,
            environments=environments,
        )

    try:
        from webarena_verified.api import WebArenaVerified
        from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
        from webarena_verified.types.task import WebArenaSite
    except ImportError:
        logger.error(
            "webarena_verified package not installed and %s is unset; refusing non-canonical evaluation",
            WEBARENA_EVAL_PYTHON_ENV,
        )
        return (
            False,
            "canonical WebArena Verified evaluation unavailable: configure a separate "
            "WebArena Verified adapter environment via "
            f"{WEBARENA_EVAL_PYTHON_ENV}, "
            "or install 'webarena-verified' in the current environment",
        )

    config = _build_webarena_config(
        environments,
        WebArenaVerifiedConfig,
        EnvironmentConfig,
        WebArenaSite,
    )

    try:
        wv = WebArenaVerified(config=config)
        result = wv.evaluate_task(
            task_id=task_id,
            agent_response=agent_response,
            network_trace=har_trace,
        )

        passed = result.score == 1.0
        parts = []
        for er in result.evaluators_results:
            status = er.status if isinstance(er.status, str) else er.status.value
            part = f"[{er.evaluator_name}] {status.upper()}"
            if er.error_msg:
                part += f": {er.error_msg}"
            parts.append(part)
        message = "; ".join(parts) if parts else f"score={result.score}, status={result.status}"
        return passed, message

    except Exception as e:
        logger.exception("Vendor evaluator failed for task %s", task_id)
        return False, f"vendor evaluator failed for task {task_id}: {e}"


def _build_webarena_config(
    environments: dict[str, list[str]],
    config_cls: type,
    env_config_cls: type,
    site_enum: type,
) -> Any:
    """Build a WebArenaVerifiedConfig from normalized site-name -> urls payload."""
    config_environments: dict[Any, Any] = {}
    for site_name, urls in environments.items():
        if not urls:
            continue
        try:
            site_key = site_enum(site_name)
        except ValueError:
            continue
        config_environments[site_key] = env_config_cls(urls=urls)

    return config_cls(environments=config_environments if config_environments else None)


def _reward_requires_network_trace(eval_configs: list[dict[str, Any]]) -> bool:
    """Return True when any evaluator config depends on network-trace evidence."""
    for config in eval_configs:
        if isinstance(config, dict) and _is_network_event_evaluator_name(config.get("evaluator")):
            return True
    return False


def _build_webarena_environment_payload(instance: dict[str, Any]) -> dict[str, list[str]]:
    """Normalize instance placeholder data into site-name -> urls payload."""
    return build_supported_webarena_environments(
        instance.get("url_placeholders"),
        site_name=str(instance.get("site_name", "")),
        site_url=str(instance.get("site_url", "")),
    )


def _run_webarena_verified_subprocess(
    *,
    python_executable: str,
    task_id: Any,
    agent_response: dict[str, Any],
    network_trace: list[dict[str, Any]],
    environments: dict[str, list[str]],
) -> tuple[bool, str]:
    """Run canonical WebArena evaluation in a separate Python environment."""
    payload = {
        "task_id": task_id,
        "agent_response": agent_response,
        "network_trace": network_trace,
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
        logger.exception("WebArena evaluator subprocess failed to start")
        return False, f"canonical WebArena evaluator process failed to start: {exc}"

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        return False, f"canonical WebArena evaluator failed: {stderr}"

    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        return False, f"canonical WebArena evaluator returned invalid JSON: {exc}"

    passed = bool(response.get("passed", False))
    message = (
        str(response.get("message", "")).strip()
        or "canonical WebArena evaluator returned no message"
    )
    return passed, message
