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
from urllib.parse import urlparse

import requests

from worldsim._sandbox_validator import validate_triage_record
from worldsim.prompt_loading import load_prompt

DEFAULT_TRIAGE_MODEL = "anthropic/claude-sonnet-4.5"
DEFAULT_TRIAGE_CONCURRENCY = 20
TRIAGE_CACHE_VERSION = 2
_OPENROUTER_TIMEOUT_SECONDS = 60
_ANTHROPIC_TIMEOUT_SECONDS = 60
_JSON_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)
_MAX_MODEL_HISTORY_CHARS = 4000
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
_URL_SIGNAL_KEYS = {"url", "current_url", "document_url", "page_url", "target_url", "href"}
_ERROR_SIGNAL_KEYS = {"error", "errors", "exception", "failure_reason"}
_STATUS_SIGNAL_KEYS = {"status", "state", "status_code", "http_status", "response_status"}
_AUTH_URL_MARKERS = ("/login", "/signin", "/sign-in", "/auth", "/captcha", "/mfa", "/2fa")
_OFFSITE_HOST_MARKERS = (
    "duckduckgo.",
    "google.",
    "bing.",
    "browserling.",
    "croxyproxy.",
    "archive.ph",
    "urlscan.io",
    "screenshotmachine.",
    "view-page-source.com",
    "codebeautify.",
)


@dataclass(slots=True)
class TriageContext:
    task_id: str
    task: dict[str, Any]
    result: dict[str, Any]
    history_payload: Any
    history_text: str
    history_urls: list[str]
    history_errors: list[str]
    history_statuses: list[str]
    allowed_hosts: set[str]


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
    unresolved: list[TriageContext] = []

    for result in failed_results:
        task_id = str(result.get("task_id", ""))
        cached = _cached_triage_record(result)
        if cached is not None:
            decisions_by_task_id[task_id] = cached
            continue
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
        context = _build_triage_context(result=result, task=task)
        rule_decision = triage_failure_rules(context=context)
        if rule_decision is not None:
            decisions_by_task_id[task_id] = rule_decision.as_dict()
        else:
            unresolved.append(context)

    if not unresolved:
        return [decisions_by_task_id[str(result.get("task_id", ""))] for result in failed_results]

    if not _triage_model_available():
        for context in unresolved:
            task_id = context.task_id
            decisions_by_task_id[task_id] = TriageDecision(
                task_id=task_id,
                decision="needs_deep_diagnosis",
                likely_root_cause=None,
                confidence=1.0,
                reason="No host-side triage model credentials configured; escalating conservatively.",
                source="rules",
            ).as_dict()
        return [decisions_by_task_id[str(result.get("task_id", ""))] for result in failed_results]

    limiter = asyncio.Semaphore(max(1, concurrency))
    triage_model = (
        model or os.environ.get("WORLDSIM_PHASE3_TRIAGE_MODEL", "").strip() or DEFAULT_TRIAGE_MODEL
    )

    async def _run_one(context: TriageContext) -> dict[str, Any]:
        async with limiter:
            return await _triage_failure_model(context=context, model=triage_model)

    model_results = await asyncio.gather(
        *(_run_one(context) for context in unresolved),
        return_exceptions=True,
    )
    for idx, payload in enumerate(model_results):
        context = unresolved[idx]
        task_id = context.task_id
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
    context: TriageContext,
) -> TriageDecision | None:
    """Return a high-precision rule-based triage decision or ``None``."""
    task_id = context.task_id
    outcome = str(context.result.get("outcome", "")).strip().lower()

    if outcome == "error" or _contains_any(str(context.result.get("message", "")).lower(), _INFRA_MARKERS):
        return TriageDecision(
            task_id=task_id,
            decision="infra_error",
            likely_root_cause="infra_error",
            confidence=1.0,
            reason="Failure appears to be infrastructure/runtime related rather than a benchmark-contract bug.",
            source="rules",
        )

    if _has_auth_failure_signal(context):
        return TriageDecision(
            task_id=task_id,
            decision="agent_limitation",
            likely_root_cause="agent_limitation",
            confidence=0.99,
            reason="Trajectory shows an authentication, session, or access-control failure, which is not a task or seed bug.",
            source="rules",
        )

    if _has_offsite_drift_signal(context):
        return TriageDecision(
            task_id=task_id,
            decision="agent_limitation",
            likely_root_cause="agent_limitation",
            confidence=0.96,
            reason="Trajectory shows off-site drift to search or proxy tooling and does not provide evidence of a benchmark-side bug.",
            source="rules",
        )

    return None


def _build_triage_context(*, result: dict[str, Any], task: dict[str, Any]) -> TriageContext:
    task_id = str(result.get("task_id", task.get("id", "")))
    trajectory_dir = Path(str(result.get("trajectory_dir", "")).strip())
    history_payload = _load_history_payload(trajectory_dir)
    history_text = _render_history_text(history_payload)
    history_urls, history_errors, history_statuses = _extract_history_signals(history_payload)
    return TriageContext(
        task_id=task_id,
        task=task,
        result=result,
        history_payload=history_payload,
        history_text=history_text,
        history_urls=history_urls,
        history_errors=history_errors,
        history_statuses=history_statuses,
        allowed_hosts=_allowed_hosts_for_task(task),
    )


def _load_history_payload(trajectory_dir: Path) -> Any:
    history_path = trajectory_dir / "history.json"
    if not history_path.exists():
        return None
    try:
        return json.loads(history_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def _render_history_text(history_payload: Any) -> str:
    if history_payload is None:
        return ""
    flattened: list[str] = []
    _collect_strings(history_payload, flattened)
    text = "\n".join(flattened)
    if len(text) <= 12000:
        return text
    head = text[:6000]
    tail = text[-6000:]
    return f"{head}\n...\n{tail}"


def _extract_history_signals(history_payload: Any) -> tuple[list[str], list[str], list[str]]:
    urls: list[str] = []
    errors: list[str] = []
    statuses: list[str] = []
    _collect_signal_strings(history_payload, urls=urls, errors=errors, statuses=statuses)
    return (urls, errors, statuses)


def _collect_signal_strings(
    value: Any,
    *,
    urls: list[str],
    errors: list[str],
    statuses: list[str],
    key: str | None = None,
) -> None:
    if isinstance(value, dict):
        for child_key, child_value in value.items():
            _collect_signal_strings(
                child_value,
                urls=urls,
                errors=errors,
                statuses=statuses,
                key=str(child_key).lower(),
            )
        return
    if isinstance(value, list):
        for item in value:
            _collect_signal_strings(item, urls=urls, errors=errors, statuses=statuses, key=key)
        return
    if isinstance(value, (int, float)) and (key or "").lower() in _STATUS_SIGNAL_KEYS:
        statuses.append(str(value))
        return
    if not isinstance(value, str):
        return
    stripped = value.strip()
    if not stripped:
        return
    lowered_key = (key or "").lower()
    if lowered_key in _URL_SIGNAL_KEYS:
        urls.append(stripped)
    elif lowered_key in _ERROR_SIGNAL_KEYS:
        errors.append(stripped)
    elif lowered_key in _STATUS_SIGNAL_KEYS:
        statuses.append(stripped)


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


def _contains_any(haystack: str, needles: tuple[str, ...]) -> bool:
    return any(needle in haystack for needle in needles)


def _has_auth_failure_signal(context: TriageContext) -> bool:
    error_text = "\n".join(context.history_errors).lower()
    status_text = "\n".join(context.history_statuses).lower()
    if _contains_any(error_text, _AUTH_MARKERS) or _contains_any(status_text, _AUTH_MARKERS):
        return True
    for raw_url in context.history_urls:
        parsed = _safe_urlparse(raw_url)
        if parsed is None:
            continue
        path = parsed.path.lower()
        if any(marker in path for marker in _AUTH_URL_MARKERS):
            return True
    return False


def _has_offsite_drift_signal(context: TriageContext) -> bool:
    external_hosts: list[str] = []
    for raw_url in context.history_urls:
        parsed = _safe_urlparse(raw_url)
        if parsed is None or not parsed.netloc:
            continue
        host = parsed.netloc.lower()
        if host in context.allowed_hosts:
            continue
        external_hosts.append(host)
    if not external_hosts:
        return False
    final_host = external_hosts[-1]
    repeated_external = len(external_hosts) >= 2
    named_offsite = any(_contains_any(host, _OFFSITE_HOST_MARKERS) for host in external_hosts)
    return repeated_external and named_offsite and _contains_any(final_host, _OFFSITE_HOST_MARKERS)


async def _triage_failure_model(
    *,
    context: TriageContext,
    model: str,
) -> dict[str, Any]:
    prompt = _build_triage_prompt(context=context)
    errors: list[str] = []

    if _openrouter_available():
        try:
            raw = await asyncio.to_thread(_call_openrouter, prompt, model)
        except Exception as exc:  # pragma: no cover - network path exercised via mocks
            errors.append(f"openrouter_error: {exc}")
        else:
            parsed = _coerce_triage_response(
                raw,
                task_id=context.task_id,
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
                task_id=context.task_id,
                source="model",
            )
            if parsed is not None:
                return _finalize_model_decision(parsed)
            errors.append("anthropic_invalid_or_refused")

    return TriageDecision(
        task_id=context.task_id,
        decision="needs_deep_diagnosis",
        likely_root_cause=None,
        confidence=1.0,
        reason=(
            "Host-side model triage could not produce a valid structured decision "
            f"({'; '.join(errors) if errors else 'no available auth path'}); escalating conservatively."
        ),
        source="model",
    ).as_dict()


def _build_triage_prompt(*, context: TriageContext) -> str:
    prompt = load_prompt("triage-benign-failure")
    payload = {
        "task": {
            "id": context.task.get("id"),
            "site": context.task.get("site"),
            "instruction": context.task.get("instruction"),
            "start_urls": context.task.get("start_urls"),
            "sanity_result": _task_sanity_result(context.task),
            "agent_context": _triage_agent_context(context.task),
        },
        "result": {
            "passed": context.result.get("passed"),
            "outcome": context.result.get("outcome"),
            "message": context.result.get("message"),
            "steps": context.result.get("steps"),
            "elapsed": context.result.get("elapsed"),
        },
        "history_excerpt": _history_excerpt_for_model(context),
    }
    return (
        f"{prompt}\n\n## Task Context\n\n"
        f"```json\n{json.dumps(payload, indent=2)}\n```"
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
    sanitized_auth: dict[str, Any] | None = None
    if isinstance(auth, dict):
        credentials = auth.get("credentials")
        credential_keys: list[str] = []
        if isinstance(credentials, dict):
            credential_keys = sorted(str(key) for key in credentials.keys())
        sanitized_auth = {
            "pre_authenticated": auth.get("pre_authenticated"),
            "description": auth.get("description"),
            "has_credentials": bool(credentials),
            "credential_keys": credential_keys,
        }
    sanitized_mech: dict[str, Any] | None = None
    if isinstance(mech, dict):
        mech_type = mech.get("type")
        sanitized_mech = {"type": mech_type}
        if mech_type == "storage_state":
            sanitized_mech["has_path"] = bool(mech.get("path"))
        if mech_type == "http_headers":
            headers = mech.get("http_headers", {}).get("headers")
            if isinstance(headers, dict):
                sanitized_mech["header_names"] = sorted(str(key) for key in headers.keys())
    return {
        "authentication": sanitized_auth,
        "auth_mechanism": sanitized_mech,
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
    parsed.setdefault("escalate", True)
    errors = validate_triage_record(parsed)
    if errors:
        return None
    return parsed


def _finalize_model_decision(parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": str(parsed.get("task_id", "")),
        "decision": "needs_deep_diagnosis",
        "likely_root_cause": parsed.get("likely_root_cause"),
        "confidence": float(parsed.get("confidence", 0.0)),
        "reason": str(parsed.get("reason", "")).strip()
        or "Model triage returned advisory metadata only; escalating conservatively.",
        "source": str(parsed.get("source", "model")),
        "escalate": True,
    }


def _triage_model_available() -> bool:
    return _openrouter_available() or _anthropic_available()


def _openrouter_available() -> bool:
    return _openrouter_credentials() is not None


def _anthropic_available() -> bool:
    return bool(os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "").strip()) or bool(
        os.environ.get("ANTHROPIC_API_KEY", "").strip()
    )


def _call_openrouter(prompt: str, model: str) -> str:
    credentials = _openrouter_credentials()
    if credentials is None:
        raise RuntimeError("no openrouter credentials configured")
    base_url, auth_token = credentials
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


def _openrouter_credentials() -> tuple[str, str] | None:
    anthropic_base = os.environ.get("ANTHROPIC_BASE_URL", "").strip()
    anthropic_token = os.environ.get("ANTHROPIC_AUTH_TOKEN", "").strip()
    if anthropic_base and anthropic_token:
        return (anthropic_base.rstrip("/"), anthropic_token)
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if openrouter_key:
        return ("https://openrouter.ai/api", openrouter_key)
    return None


def _history_excerpt_for_model(context: TriageContext) -> str:
    excerpt = context.history_text.strip()
    if not excerpt:
        return ""
    lines = [line.strip() for line in excerpt.splitlines() if line.strip()]
    salient: list[str] = []
    for line in lines:
        lower = line.lower()
        if (
            _contains_any(lower, _AUTH_MARKERS)
            or _contains_any(lower, _OFFSITE_MARKERS)
            or _contains_any(lower, _INFRA_MARKERS)
            or "error" in lower
            or "reward" in lower
            or "final" in lower
        ):
            salient.append(line)
    tail = lines[-20:]
    merged = "\n".join((salient + tail)[-40:])
    minimized = merged[:_MAX_MODEL_HISTORY_CHARS]
    return _redact_sensitive_text(minimized, task=context.task)


def _redact_sensitive_text(text: str, *, task: dict[str, Any]) -> str:
    redacted = text
    credentials = (
        task.get("agent_context", {})
        .get("authentication", {})
        .get("credentials")
        if isinstance(task.get("agent_context"), dict)
        and isinstance(task.get("agent_context", {}).get("authentication"), dict)
        else None
    )
    if isinstance(credentials, dict):
        for value in credentials.values():
            if isinstance(value, str) and value.strip():
                redacted = redacted.replace(value, "[REDACTED_CREDENTIAL]")
    redacted = re.sub(
        r"(?i)(password|token|secret|api[_-]?key|cookie|authorization)\s*[:=]\s*\S+",
        r"\1=[REDACTED]",
        redacted,
    )
    redacted = re.sub(
        r"(?i)(bearer\s+token)\s+[A-Za-z0-9._-]+",
        r"\1 [REDACTED]",
        redacted,
    )
    redacted = re.sub(r"Bearer\s+[A-Za-z0-9._-]+", "Bearer [REDACTED]", redacted, flags=re.IGNORECASE)
    return redacted


def _cached_triage_record(result: dict[str, Any]) -> dict[str, Any] | None:
    if "triage_decision" not in result:
        return None
    if result.get("triage_cache_version") != TRIAGE_CACHE_VERSION:
        return None
    source = result.get("triage_source")
    decision = result.get("triage_decision")
    if source != "rules" and decision != "needs_deep_diagnosis":
        return None
    record = {
        "task_id": str(result.get("task_id", "")),
        "decision": decision,
        "likely_root_cause": result.get("triage_likely_root_cause"),
        "confidence": result.get("triage_confidence"),
        "reason": result.get("triage_reason"),
        "source": source,
        "escalate": bool(result.get("triage_escalate")),
    }
    errors = validate_triage_record(record)
    if errors:
        return None
    return record


def _allowed_hosts_for_task(task: dict[str, Any]) -> set[str]:
    hosts: set[str] = set()
    for raw_url in task.get("start_urls", []) or []:
        parsed = _safe_urlparse(str(raw_url))
        if parsed is not None and parsed.netloc:
            hosts.add(parsed.netloc.lower())
    return hosts


def _safe_urlparse(value: str) -> Any | None:
    try:
        parsed = urlparse(value)
    except Exception:
        return None
    if not parsed.scheme and not parsed.netloc:
        return None
    return parsed
