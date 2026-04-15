"""Cheap host-side triage for Phase 3 benign failures.

This module adds a conservative pre-filter ahead of the expensive
diagnosis-fix loop in ``phase_3_benign``.

Routing policy:

- Deterministic high-precision rules classify obvious infrastructure and
  auth/session/off-site failures.
- Remaining failures optionally go through a cheap structured model triage
  pass using host-side API calls.
- Any uncertain or benchmark-suspect case is escalated to the existing
  per-task diagnosis sandbox.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from worldsim._sandbox_validator import validate_triage
from worldsim.prompt_loading import load_prompt

DEFAULT_TRIAGE_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_TRIAGE_CONCURRENCY = 20
_OPENROUTER_TIMEOUT_SECONDS = 60
_ANTHROPIC_TIMEOUT_SECONDS = 60
_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_AUTH_MARKERS = (
    "401",
    "403",
    "sign in",
    "sign-in",
    "log in",
    "login",
    "please sign in",
    "session expired",
    "expired session",
    "authentication required",
    "access denied",
    "captcha",
    "mfa",
    "two-factor",
    "2fa",
)
_OFFSITE_MARKERS = (
    "duckduckgo",
    "google search",
    "bing",
    "browserling",
    "croxyproxy",
    "archive.ph",
    "urlscan",
    "screenshotmachine",
    "view-page-source.com",
    "codebeautify",
)
_INFRA_MARKERS = (
    "worker task failed",
    "connection refused",
    "reset endpoint",
    "500 server error",
    "traceback",
    "playwright error",
    "browser launch failed",
    "page crashed",
    "no matching live instance",
)
_REFUSAL_SNIPPETS = (
    "i'm not going to",
    "i won't",
    "i cannot help",
    "i can't help",
    "i will not",
    "i cannot generate",
    "i am not going to",
    "i'm unable to",
)


@dataclass(slots=True)
class TriageDecision:
    task_id: str
    decision: str
    likely_root_cause: str | None
    confidence: float
    reason: str
    source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "decision": self.decision,
            "likely_root_cause": self.likely_root_cause,
            "confidence": self.confidence,
            "reason": self.reason,
            "source": self.source,
            "escalate": self.decision == "needs_deep_diagnosis",
        }


async def triage_failures(
    *,
    failed_results: list[dict[str, Any]],
    prepared_by_id: dict[str, dict[str, Any]],
    model: str | None = None,
    concurrency: int = DEFAULT_TRIAGE_CONCURRENCY,
) -> list[dict[str, Any]]:
    """Return triage decisions for failed Phase 3 task results.

    The function is conservative: any uncertainty falls through to
    ``needs_deep_diagnosis``.
    """
    decisions_by_task_id: dict[str, dict[str, Any]] = {}
    unresolved: list[tuple[dict[str, Any], dict[str, Any]]] = []

    for result in failed_results:
        task_id = str(result.get("task_id", ""))
        task = prepared_by_id.get(task_id)
        if task is None:
            decisions_by_task_id[task_id] = (
                TriageDecision(
                    task_id=task_id,
                    decision="needs_deep_diagnosis",
                    likely_root_cause=None,
                    confidence=1.0,
                    reason="Task metadata was unavailable for triage; escalating conservatively.",
                    source="rules",
                ).as_dict()
            )
            continue
        rule_decision = triage_failure_rules(result=result, task=task)
        if rule_decision is not None:
            decisions_by_task_id[task_id] = rule_decision.as_dict()
        else:
            unresolved.append((result, task))

    if not unresolved:
        return [decisions_by_task_id[str(result.get("task_id", ""))] for result in failed_results]

    if not _triage_model_available():
        for result, _task in unresolved:
            task_id = str(result.get("task_id", ""))
            decisions_by_task_id[task_id] = TriageDecision(
                task_id=str(result.get("task_id", "")),
                decision="needs_deep_diagnosis",
                likely_root_cause=None,
                confidence=1.0,
                reason="No host-side triage model credentials configured; escalating conservatively.",
                source="rules",
            ).as_dict()
        return [decisions_by_task_id[str(result.get("task_id", ""))] for result in failed_results]

    limiter = asyncio.Semaphore(max(1, concurrency))
    triage_model = model or os.environ.get("WORLDSIM_PHASE3_TRIAGE_MODEL", "").strip() or DEFAULT_TRIAGE_MODEL

    async def _run_one(result: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
        async with limiter:
            return await _triage_failure_model(result=result, task=task, model=triage_model)

    model_results = await asyncio.gather(
        *(_run_one(result, task) for result, task in unresolved),
        return_exceptions=True,
    )
    for idx, payload in enumerate(model_results):
        result, _task = unresolved[idx]
        task_id = str(result.get("task_id", ""))
        if isinstance(payload, BaseException):
            decisions_by_task_id[task_id] = (
                TriageDecision(
                    task_id=task_id,
                    decision="needs_deep_diagnosis",
                    likely_root_cause=None,
                    confidence=1.0,
                    reason=f"Model triage failed ({payload}); escalating conservatively.",
                    source="model",
                ).as_dict()
            )
            continue
        decisions_by_task_id[task_id] = payload

    return [decisions_by_task_id[str(result.get("task_id", ""))] for result in failed_results]


def triage_failure_rules(
    *,
    result: dict[str, Any],
    task: dict[str, Any],
) -> TriageDecision | None:
    """Return a high-precision rule-based triage decision or ``None``."""
    task_id = str(result.get("task_id", task.get("id", "")))
    outcome = str(result.get("outcome", "")).strip().lower()
    message = str(result.get("message", "")).strip()
    history_text = _load_history_text(Path(result.get("trajectory_dir", "")))
    haystack = _normalize_text(message, history_text)

    if outcome == "error" or _contains_any(haystack, _INFRA_MARKERS):
        return TriageDecision(
            task_id=task_id,
            decision="infra_error",
            likely_root_cause="infra_error",
            confidence=1.0,
            reason="Failure appears to be infrastructure/runtime related rather than a benchmark-contract bug.",
            source="rules",
        )

    if _contains_any(haystack, _AUTH_MARKERS):
        return TriageDecision(
            task_id=task_id,
            decision="agent_limitation",
            likely_root_cause="agent_limitation",
            confidence=0.99,
            reason="Trajectory shows an authentication, session, or access-control failure, which is not a task or seed bug.",
            source="rules",
        )

    if _contains_any(haystack, _OFFSITE_MARKERS):
        return TriageDecision(
            task_id=task_id,
            decision="agent_limitation",
            likely_root_cause="agent_limitation",
            confidence=0.96,
            reason="Trajectory shows off-site drift to search or proxy tooling and does not provide evidence of a benchmark-side bug.",
            source="rules",
        )

    return None


def _load_history_text(trajectory_dir: Path) -> str:
    history_path = trajectory_dir / "history.json"
    if not history_path.exists():
        return ""
    try:
        data = json.loads(history_path.read_text())
    except (json.JSONDecodeError, OSError):
        return ""
    flattened: list[str] = []
    _collect_strings(data, flattened)
    text = "\n".join(flattened)
    if len(text) <= 12000:
        return text
    head = text[:6000]
    tail = text[-6000:]
    return f"{head}\n...\n{tail}"


def _collect_strings(value: Any, acc: list[str]) -> None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            acc.append(stripped)
        return
    if isinstance(value, list):
        for item in value:
            _collect_strings(item, acc)
        return
    if isinstance(value, dict):
        for item in value.values():
            _collect_strings(item, acc)


def _normalize_text(*parts: str) -> str:
    return "\n".join(part for part in parts if part).lower()


def _contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(needle in haystack for needle in needles)


async def _triage_failure_model(
    *,
    result: dict[str, Any],
    task: dict[str, Any],
    model: str,
) -> dict[str, Any]:
    prompt = _build_triage_prompt(task=task, result=result)
    errors: list[str] = []

    if _openrouter_available():
        try:
            raw = await asyncio.to_thread(_call_openrouter, prompt, model)
        except Exception as exc:  # pragma: no cover - network path exercised via mocks
            errors.append(f"openrouter_error: {exc}")
        else:
            parsed = _coerce_triage_response(
                raw,
                task_id=str(result.get("task_id", "")),
                source="model",
            )
            if parsed is not None:
                return _finalize_model_decision(parsed)
            errors.append("openrouter_invalid_or_refused")

    if _anthropic_available():
        try:
            raw = await asyncio.to_thread(_call_anthropic_fallback, prompt, model)
        except Exception as exc:  # pragma: no cover - network path exercised via mocks
            errors.append(f"anthropic_error: {exc}")
        else:
            parsed = _coerce_triage_response(
                raw,
                task_id=str(result.get("task_id", "")),
                source="model",
            )
            if parsed is not None:
                return _finalize_model_decision(parsed)
            errors.append("anthropic_invalid_or_refused")

    return TriageDecision(
        task_id=str(result.get("task_id", "")),
        decision="needs_deep_diagnosis",
        likely_root_cause=None,
        confidence=1.0,
        reason=(
            "Host-side model triage could not produce a valid structured decision "
            f"({'; '.join(errors) if errors else 'no available auth path'}); escalating conservatively."
        ),
        source="model",
    ).as_dict()


def _build_triage_prompt(*, task: dict[str, Any], result: dict[str, Any]) -> str:
    prompt = load_prompt("triage-benign-failure")
    context = {
        "task": {
            "id": task.get("id"),
            "site": task.get("site"),
            "instruction": task.get("instruction"),
            "start_urls": task.get("start_urls"),
            "sanity_result": _task_sanity_result(task),
            "agent_context": _triage_agent_context(task),
        },
        "result": {
            "passed": result.get("passed"),
            "outcome": result.get("outcome"),
            "message": result.get("message"),
            "steps": result.get("steps"),
            "elapsed": result.get("elapsed"),
        },
        "history_excerpt": _load_history_text(Path(result.get("trajectory_dir", ""))),
    }
    return (
        f"{prompt}\n\n## Task Context\n\n"
        f"```json\n{json.dumps(context, indent=2)}\n```"
    )


def _task_sanity_result(task: dict[str, Any]) -> str:
    candidates = (
        task.get("sanity_result"),
        task.get("sanity_check_result"),
        task.get("sanity_check", {}).get("result")
        if isinstance(task.get("sanity_check"), dict)
        else None,
        task.get("sanity", {}).get("result") if isinstance(task.get("sanity"), dict) else None,
    )
    for value in candidates:
        if value is None:
            continue
        normalized = str(value).strip().lower()
        if normalized in {"pass", "fail", "unknown"}:
            return normalized
    return "unknown"


def _triage_agent_context(task: dict[str, Any]) -> dict[str, Any]:
    agent_context = task.get("agent_context")
    if not isinstance(agent_context, dict):
        return {}
    auth = agent_context.get("authentication")
    mech = agent_context.get("auth_mechanism")
    return {
        "authentication": auth if isinstance(auth, dict) else None,
        "auth_mechanism": mech if isinstance(mech, dict) else None,
    }


def _coerce_triage_response(raw: str | tuple[str, str], *, task_id: str, source: str) -> dict[str, Any] | None:
    text = raw[0] if isinstance(raw, tuple) else raw
    if _is_refusal(text):
        return None
    candidate = _JSON_FENCE_RE.sub("", text).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    parsed.setdefault("task_id", task_id)
    parsed.setdefault("source", source)
    errors = validate_triage(parsed)
    if errors:
        return None
    return parsed


def _finalize_model_decision(parsed: dict[str, Any]) -> dict[str, Any]:
    decision = str(parsed.get("decision"))
    confidence = float(parsed.get("confidence", 0.0))
    if decision == "agent_limitation" and confidence >= 0.90:
        return {
            **parsed,
            "escalate": False,
        }
    if decision == "infra_error" and confidence >= 0.95:
        return {
            **parsed,
            "escalate": False,
        }
    return {
        "task_id": str(parsed.get("task_id", "")),
        "decision": "needs_deep_diagnosis",
        "likely_root_cause": parsed.get("likely_root_cause"),
        "confidence": confidence,
        "reason": str(parsed.get("reason", "")).strip()
        or "Model triage did not provide a high-confidence short-circuit decision.",
        "source": str(parsed.get("source", "model")),
        "escalate": True,
    }


def _triage_model_available() -> bool:
    return _openrouter_available() or _anthropic_available()


def _openrouter_available() -> bool:
    return bool(os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()) and bool(
        os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    )


def _anthropic_available() -> bool:
    return bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()) or bool(
        os.environ.get("ANTHROPIC_API_KEY", "").strip()
    )


def _call_openrouter(prompt: str, model: str) -> str:
    base_url = os.environ["ANTHROPIC_BASE_URL"].rstrip("/")
    auth_token = os.environ["ANTHROPIC_AUTH_TOKEN"].strip()
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {auth_token}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        },
        timeout=_OPENROUTER_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    return str(((data.get("choices") or [{}])[0].get("message") or {}).get("content") or "")


def _call_anthropic_fallback(prompt: str, model: str) -> tuple[str, str]:
    oauth_token = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    headers = {
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    auth_path = "anthropic_api"
    if oauth_token:
        headers["Authorization"] = f"Bearer {oauth_token}"
        auth_path = "oauth"
    elif api_key:
        headers["x-api-key"] = api_key
    else:
        raise RuntimeError("no anthropic fallback credentials configured")

    response = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        json={
            "model": _direct_anthropic_model_name(model),
            "max_tokens": 900,
            "temperature": 0.0,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=_ANTHROPIC_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    data = response.json()
    content = data.get("content") or []
    if isinstance(content, list):
        text_parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ]
        return ("".join(text_parts), auth_path)
    return (str(content), auth_path)


def _direct_anthropic_model_name(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _is_refusal(text: str) -> bool:
    normalized = text.lower()
    return any(snippet in normalized for snippet in _REFUSAL_SNIPPETS)
