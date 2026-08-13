"""Phase 4 execution helper behavior."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import requests

from warp_taskgen.agent_config import task_reset_endpoints
from warp_taskgen.config import has_effective_agent_auth
from warp_taskgen.phase_2.text_fill.seed import (
    materialize_adversarial_seed,
    validate_seed_template_contract,
)
from warp_taskgen.phase_2.text_fill.validation import validate_text_post_hoc

logger = logging.getLogger(__name__)
_RESET_TIMEOUT = 300
_RESET_MAX_RETRIES = 2
_RESET_RETRY_DELAY = 10


async def _reset_task_environment(task: dict[str, Any]) -> None:
    """Reset every benchmark instance the task may interact with."""
    endpoints = task_reset_endpoints(task)
    if not endpoints:
        return
    await asyncio.gather(*[asyncio.to_thread(_post_reset, ep) for ep in endpoints])
    await asyncio.sleep(2)


def _post_reset(endpoint: str) -> None:
    """Call a benchmark reset endpoint with retries for transient failures.

    Retries on connection errors, timeouts, and 5xx responses. The generous
    timeout (300s) is needed because some WebArena sites (especially GitLab)
    block for minutes while reconfiguring.
    """
    last_exc: Exception | None = None
    for attempt in range(1, _RESET_MAX_RETRIES + 1):
        try:
            response = requests.post(endpoint, timeout=_RESET_TIMEOUT)
            if response.status_code >= 500:
                logger.warning(
                    "Reset endpoint %s returned %d on attempt %d/%d",
                    endpoint,
                    response.status_code,
                    attempt,
                    _RESET_MAX_RETRIES,
                )
                last_exc = requests.HTTPError(
                    f"{response.status_code} Server Error for url: {endpoint}",
                    response=response,
                )
                if attempt < _RESET_MAX_RETRIES:
                    time.sleep(_RESET_RETRY_DELAY)
                    continue
                response.raise_for_status()
            # 4xx errors are not retried — they indicate a client-side problem.
            response.raise_for_status()
            return
        except requests.ConnectionError as exc:
            logger.warning(
                "Reset endpoint %s connection error on attempt %d/%d: %s",
                endpoint,
                attempt,
                _RESET_MAX_RETRIES,
                exc,
            )
            last_exc = exc
            if attempt < _RESET_MAX_RETRIES:
                time.sleep(_RESET_RETRY_DELAY)
        except requests.Timeout as exc:
            logger.warning(
                "Reset endpoint %s timed out after %ds on attempt %d/%d",
                endpoint,
                _RESET_TIMEOUT,
                attempt,
                _RESET_MAX_RETRIES,
            )
            last_exc = exc
            if attempt < _RESET_MAX_RETRIES:
                time.sleep(_RESET_RETRY_DELAY)

    raise RuntimeError(
        f"Reset endpoint {endpoint} failed after {_RESET_MAX_RETRIES} attempts"
    ) from last_exc


def _effective_adversarial_seed(adversarial_task: dict[str, Any]) -> Any:
    seed_template = adversarial_task.get("seed_template")
    payload_texts = adversarial_task.get("payload_texts")
    if seed_template is None and payload_texts is None:
        return adversarial_task.get("adversarial_data_seed")
    if not isinstance(seed_template, dict):
        raise ValueError("v2 adversarial task is missing a valid seed_template object")
    validate_seed_template_contract(seed_template)
    if not isinstance(payload_texts, list) or not payload_texts:
        raise ValueError("v2 adversarial task is missing payload_texts")
    for payload_index, payload in enumerate(payload_texts):
        if not isinstance(payload, dict):
            raise ValueError(f"payload_texts[{payload_index}] must be an object")
        payload_errors = validate_text_post_hoc(payload, adversarial_task)
        if payload_errors:
            raise ValueError(f"payload_texts[{payload_index}] invalid: {'; '.join(payload_errors)}")
    if isinstance(seed_template, dict) and isinstance(payload_texts, list) and payload_texts:
        if "selected_payload_index" not in adversarial_task:
            raise ValueError("selected_payload_index must be present")
        selected_index = adversarial_task.get("selected_payload_index")
        if not isinstance(selected_index, int):
            raise ValueError("selected_payload_index must be an integer")
        if selected_index < 0 or selected_index >= len(payload_texts):
            raise ValueError("selected_payload_index is out of range for payload_texts")
        selected = payload_texts[selected_index]
        return materialize_adversarial_seed(seed_template, str(selected["rendered_payload"]))
    return adversarial_task.get("adversarial_data_seed")


def _agent_context_with_instance_auth(
    agent_context: Any,
    instance_agent_auth: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(agent_context, dict):
        agent_context = {}
    merged = json.loads(json.dumps(agent_context))
    if not isinstance(instance_agent_auth, dict):
        return merged or None
    if str(instance_agent_auth.get("type", "")).strip() == "none":
        return merged or None
    if has_effective_agent_auth(instance_agent_auth):
        merged["authentication"] = {
            "pre_authenticated": True,
            "credentials": None,
            "description": "Pre-authenticated via deployment config.",
        }
    elif "authentication" not in merged:
        merged["authentication"] = {
            "pre_authenticated": True,
            "credentials": None,
            "description": "Pre-authenticated via deployment config.",
        }
    return merged or None


def _is_non_interpretable_benign_eval(
    benign_reward: dict[str, Any],
    benign_passed: bool,
    benign_message: str,
) -> bool:
    """Return True when the benign reward could not be interpreted canonically."""
    if benign_passed or "eval" not in benign_reward:
        return False
    normalized = benign_message.lower()
    return any(
        marker in normalized
        for marker in (
            "canonical webarena verified evaluation unavailable",
            "reward spec missing canonical webarena verified task_id",
            "canonical webarena evaluator failed",
            "canonical webarena evaluator process failed to start",
            "canonical webarena evaluator returned invalid json",
            "vendor evaluator failed",
        )
    )


def _has_scoreable_agent_output(result: Any) -> bool:
    """Return True when a non-success run still produced usable evaluation artifacts."""
    return bool(
        getattr(result, "steps", 0) > 0
        or getattr(result, "final_result", None)
        or getattr(result, "network_trace", None)
    )


def _delivery_site_name(delivery_channel: Any) -> str:
    if not isinstance(delivery_channel, dict):
        return ""
    delivery_site = delivery_channel.get("delivery_site")
    if isinstance(delivery_site, str):
        normalized = delivery_site.strip()
        if normalized.lower() == "none":
            return ""
        return normalized
    return ""
