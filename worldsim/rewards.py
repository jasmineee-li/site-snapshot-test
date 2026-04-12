"""Reward function dispatcher.

Supports two evaluation paths:

1. **WebArena Verified API** — for tasks wrapped from the WebArena Verified
   benchmark. Uses the ``webarena_verified`` package's evaluator (
   ``AgentResponseEvaluator`` + ``NetworkEventEvaluator``).

2. **Custom checkers** — for eval types not covered by WebArena Verified
   (e.g. ``db_query_match`` for injection verification in Phase 4). Dispatched
   via the ``_CHECKERS`` registry.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` Phase 3
"Evaluation Infrastructure".
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def run_reward_function(
    reward: dict[str, Any],
    instance: dict[str, Any],
    agent_result: Any | None = None,
    network_trace: list[dict] | None = None,
) -> tuple[bool, str]:
    """Run one reward spec against a benchmark instance.

    Tries WebArena Verified API first (if reward has evaluator-style ``eval``
    array). Falls back to the ``_CHECKERS`` registry for other eval types.

    Args:
        reward: Reward spec. For WebArena Verified tasks, contains ``eval``
            (array of evaluator configs) and ``task_id``. For custom checks,
            contains ``type`` + type-specific fields.
        instance: Running benchmark instance dict with ``site_url`` etc.
        agent_result: The agent's ``AgentResult`` from Browser Use. Provides
            ``final_result`` for retrieve tasks.
        network_trace: HAR-format network events captured during the agent run.
            Required for tasks with ``NetworkEventEvaluator`` configs.

    Returns:
        ``(passed, message)`` tuple.
    """
    # WebArena Verified evaluation path
    if "eval" in reward and isinstance(reward["eval"], list):
        return _run_webarena_verified_eval(reward, instance, agent_result, network_trace)

    # Custom checker path (legacy / extension)
    eval_type = reward.get("type")
    if eval_type is None:
        return False, "Reward spec has neither 'eval' array nor 'type' field"

    checker = _CHECKERS.get(eval_type)
    if checker is None:
        raise NotImplementedError(
            f"Reward type {eval_type!r} not registered in worldsim.rewards. "
            f"Extend _CHECKERS to add support."
        )
    return checker(reward, instance)


def _run_webarena_verified_eval(
    reward: dict[str, Any],
    instance: dict[str, Any],
    agent_result: Any | None,
    network_trace: list[dict] | None,
) -> tuple[bool, str]:
    """Evaluate using the vendor WebArena Verified evaluator.

    Delegates to the ``webarena_verified`` package for full normalization
    (NFKC, unidecode, TM-stripping, type-dispatch across 17 data types, etc.).
    Falls back to the homebrew evaluator if the vendor package is not installed
    or the task has no ``task_id`` (required to look up the canonical eval config
    in the vendor dataset).
    """
    try:
        from webarena_verified.api import WebArenaVerified
        from webarena_verified.types.config import EnvironmentConfig, WebArenaVerifiedConfig
        from webarena_verified.types.task import WebArenaSite
    except ImportError:
        logger.warning("webarena_verified package not installed; falling back to homebrew evaluator")
        return _run_homebrew_eval(reward, instance, agent_result, network_trace)

    task_id = reward.get("task_id")
    if task_id is None:
        logger.debug("No task_id in reward spec; falling back to homebrew evaluator")
        return _run_homebrew_eval(reward, instance, agent_result, network_trace)

    # Build the agent response dict in FinalAgentResponse format
    eval_configs = reward["eval"]
    agent_response = _build_agent_response(eval_configs, agent_result)

    # Build a WebArenaVerifiedConfig with environments from our instance dict
    config = _build_webarena_config(instance, WebArenaVerifiedConfig, EnvironmentConfig, WebArenaSite)

    try:
        wv = WebArenaVerified(config=config)
        result = wv.evaluate_task(
            task_id=task_id,
            agent_response=agent_response,
            network_trace=network_trace or [],
        )

        passed = result.score == 1.0
        # Build a human-readable message from the evaluator results
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
        logger.warning(
            "Vendor evaluator failed for task %s: %s; falling back to homebrew",
            task_id, e,
        )
        return _run_homebrew_eval(reward, instance, agent_result, network_trace)


def _build_webarena_config(
    instance: dict[str, Any],
    config_cls: type,
    env_config_cls: type,
    site_enum: type,
) -> Any:
    """Build a WebArenaVerifiedConfig from our instance dict.

    Maps ``url_placeholders`` (e.g. ``{"__SHOPPING__": "http://..."}```) and
    ``site_url`` into the vendor config's ``environments`` dict.
    """
    site_url = instance.get("site_url", "")
    explicit = instance.get("url_placeholders", {})

    # Map placeholder tokens to WebArenaSite enum keys
    _PLACEHOLDER_TO_SITE = {
        "__SHOPPING__": "shopping",
        "__SHOPPING_ADMIN__": "shopping_admin",
        "__GITLAB__": "gitlab",
        "__REDDIT__": "reddit",
        "__WIKIPEDIA__": "wikipedia",
        "__MAP__": "map",
    }

    environments: dict = {}
    for placeholder, site_value in _PLACEHOLDER_TO_SITE.items():
        url = explicit.get(placeholder) or site_url
        if url:
            try:
                site_key = site_enum(site_value)
                environments[site_key] = env_config_cls(urls=[url])
            except ValueError:
                pass

    # Also add any non-standard placeholders from the instance
    for key, val in explicit.items():
        if key not in _PLACEHOLDER_TO_SITE and val:
            stripped = key.strip("_").lower()
            try:
                site_key = site_enum(stripped)
                if site_key not in environments:
                    environments[site_key] = env_config_cls(urls=[val])
            except ValueError:
                pass

    return config_cls(environments=environments if environments else None)


def _run_homebrew_eval(
    reward: dict[str, Any],
    instance: dict[str, Any],
    agent_result: Any | None,
    network_trace: list[dict] | None,
) -> tuple[bool, str]:
    """Homebrew evaluator — fallback when the vendor package is unavailable.

    Iterates the ``eval`` array manually and applies simplified comparison
    logic. Does NOT perform full Unicode normalization or type dispatch.
    """
    eval_configs = reward["eval"]

    # Build agent response in WebArena Verified format
    agent_response = _build_agent_response(eval_configs, agent_result)

    all_passed = True
    messages: list[str] = []

    for config in eval_configs:
        evaluator_type = config.get("evaluator", "")

        if evaluator_type == "AgentResponseEvaluator":
            passed, msg = _eval_agent_response(config, agent_response)
        elif evaluator_type == "NetworkEventEvaluator":
            passed, msg = _eval_network_event(config, network_trace, instance)
        else:
            passed, msg = False, f"Unknown evaluator type: {evaluator_type}"

        if not passed:
            all_passed = False
        messages.append(f"[{evaluator_type}] {'PASS' if passed else 'FAIL'}: {msg}")

    combined = "; ".join(messages)
    return all_passed, combined


def _build_agent_response(
    eval_configs: list[dict], agent_result: Any | None
) -> dict[str, Any]:
    """Build a WebArena Verified-compatible agent response dict."""
    # Infer task type from the expected response in the first AgentResponseEvaluator
    task_type = "retrieve"
    for config in eval_configs:
        if config.get("evaluator") == "AgentResponseEvaluator":
            expected = config.get("expected", {})
            task_type = expected.get("task_type", "retrieve")
            break

    final_result = None
    if agent_result is not None:
        final_result = getattr(agent_result, "final_result", None)

    # Parse the agent's final answer into retrieved_data format
    retrieved_data = None
    if task_type == "retrieve" and final_result:
        retrieved_data = _parse_retrieved_data(final_result)

    is_done = agent_result is not None and getattr(agent_result, "is_done", False)

    return {
        "task_type": task_type,
        "status": "SUCCESS" if is_done else "FAILURE",
        "retrieved_data": retrieved_data,
    }


def _parse_retrieved_data(final_result: str | None) -> list | None:
    """Parse the agent's final text answer into structured retrieved_data.

    Attempts JSON parsing first, then falls back to treating the answer as
    a single string item.
    """
    if not final_result:
        return None

    # Try JSON parse first (agent may have returned structured data)
    try:
        parsed = json.loads(final_result)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        return [parsed]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fall back to single string
    return [final_result.strip()]


def _eval_agent_response(
    config: dict, agent_response: dict
) -> tuple[bool, str]:
    """Evaluate an AgentResponseEvaluator config against agent response."""
    expected = config.get("expected", {})

    # Check task type
    expected_type = expected.get("task_type", "").lower()
    actual_type = agent_response.get("task_type", "").lower()
    if expected_type and actual_type != expected_type:
        return False, f"task_type mismatch: expected {expected_type}, got {actual_type}"

    # Check status
    expected_status = expected.get("status", "").upper()
    actual_status = agent_response.get("status", "").upper()
    if expected_status and actual_status != expected_status:
        return False, f"status mismatch: expected {expected_status}, got {actual_status}"

    # Check retrieved_data
    expected_data = expected.get("retrieved_data")
    if expected_data is None:
        # mutate/navigate tasks don't require retrieved_data
        return True, "status and task_type match"

    actual_data = agent_response.get("retrieved_data")
    if actual_data is None:
        return False, "expected retrieved_data but got None"

    # Compare data (case-insensitive, stripped)
    ordered = config.get("ordered", False)
    if _compare_data(expected_data, actual_data, ordered=ordered):
        return True, "retrieved_data matches"

    return False, f"retrieved_data mismatch: expected {expected_data!r}, got {actual_data!r}"


def _compare_data(expected: list, actual: list, ordered: bool = False) -> bool:
    """Compare expected vs actual retrieved_data with normalization."""
    def normalize(v: Any) -> str:
        return str(v).strip().lower()

    expected_norm = [normalize(e) for e in expected]
    actual_norm = [normalize(a) for a in actual]

    if ordered:
        return expected_norm == actual_norm

    # Unordered: every expected item (including duplicates) must appear in actual
    return Counter(expected_norm) <= Counter(actual_norm)


def _eval_network_event(
    config: dict, network_trace: list[dict] | None, instance: dict
) -> tuple[bool, str]:
    """Evaluate a NetworkEventEvaluator config against captured network trace.

    This is a simplified implementation that checks URL and HTTP method
    matching. The full WebArena Verified evaluator handles JSONPath, regex,
    query params, post data, response content, etc. For complete evaluation,
    use the ``webarena_verified`` package directly.
    """
    should_not_exist = config.get("should_not_exist", False)

    # Check should_not_exist BEFORE the None trace check: if we expect an
    # event NOT to exist and there's no trace at all, the event is absent.
    if network_trace is None:
        if should_not_exist:
            return True, "no network trace, event correctly absent"
        return False, "no network trace captured (required for NetworkEventEvaluator)"

    expected = config.get("expected", {})
    expected_url = expected.get("url", "")
    expected_method = expected.get("http_method", "GET").upper()

    # Resolve URL placeholders using instance config
    resolved_url = _resolve_url_placeholders(expected_url, instance)

    # Search network trace for matching events
    last_only = config.get("last_event_only", True)
    matching_events = []

    for event in network_trace:
        event_url = event.get("url", "")
        event_method = event.get("method", "GET").upper()

        if event_method == expected_method and _url_matches(resolved_url, event_url):
            matching_events.append(event)

    if last_only and matching_events:
        matching_events = [matching_events[-1]]

    if should_not_exist:
        if matching_events:
            return False, f"event found but should_not_exist: {resolved_url}"
        return True, "event correctly absent"

    if not matching_events:
        return False, f"no matching network event for {expected_method} {resolved_url}"

    return True, f"found {len(matching_events)} matching event(s)"


def _resolve_url_placeholders(url: str, instance: dict) -> str:
    """Replace __SITE__ placeholders in URLs with actual instance URLs.

    For multi-site tasks, ``instance["url_placeholders"]`` provides the
    authoritative mapping of all site tokens to their real URLs.  For the
    common single-site case the instance's own ``site_url`` is used as a
    fallback for any placeholder not in the explicit mapping.
    """
    site_url = instance.get("site_url", "")
    # Explicit per-instance overrides first (handles multi-site tasks)
    explicit = instance.get("url_placeholders", {})

    # Standard WebArena placeholders — fallback to current site_url only
    # when the explicit mapping does not provide a value.
    all_placeholders = (
        "__SHOPPING__",
        "__SHOPPING_ADMIN__",
        "__GITLAB__",
        "__REDDIT__",
        "__WIKIPEDIA__",
        "__MAP__",
    )
    for placeholder in all_placeholders:
        replacement = explicit.get(placeholder, site_url)
        url = url.replace(placeholder, replacement)

    # Apply any non-standard explicit placeholders not in the list above
    for key, val in explicit.items():
        if key not in all_placeholders:
            url = url.replace(key, val)

    return url


def _url_matches(expected: str, actual: str) -> bool:
    """Check if expected URL matches actual URL (substring / path match)."""
    # Strip trailing slashes for comparison
    expected = expected.rstrip("/")
    actual = actual.rstrip("/")

    # Exact match
    if expected == actual:
        return True

    # "GOLD in PRED" — expected appears as substring of actual
    if expected in actual:
        return True

    # Path-only comparison (ignore host differences)
    from urllib.parse import urlparse
    exp_parsed = urlparse(expected)
    act_parsed = urlparse(actual)
    if exp_parsed.path and exp_parsed.path.rstrip("/") in act_parsed.path:
        return True

    return False


# ---------------------------------------------------------------------------
# Custom checker registry (fallback for non-WebArena-Verified eval types)
# ---------------------------------------------------------------------------


def _db_query_match(
    reward: dict[str, Any], instance: dict[str, Any]
) -> tuple[bool, str]:
    """Execute a SQL query and compare result to expected value.

    Used for injection verification in Phase 4 where we need to check
    database state directly.
    """
    query = reward.get("query", "")
    expected = reward.get("expected")
    db_conn = instance.get("db_connection")

    if not db_conn:
        return False, "no db_connection on instance"
    if not query:
        return False, "no query in reward spec"

    try:
        # execute_sql is for writes; for reads we need a cursor
        import urllib.parse
        parsed = urllib.parse.urlparse(db_conn)

        if parsed.scheme == "mysql":
            import pymysql
            conn = pymysql.connect(
                host=parsed.hostname,
                port=parsed.port or 3306,
                user=parsed.username,
                password=parsed.password,
                database=(parsed.path or "").lstrip("/"),
            )
        elif parsed.scheme in ("postgresql", "postgres"):
            import psycopg2  # late import — only needed on the PostgreSQL path
            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                dbname=(parsed.path or "").lstrip("/"),
            )
        else:
            return False, f"unsupported DB dialect: {parsed.scheme}"

        try:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
        finally:
            conn.close()

        # Compare result
        if expected is None:
            passed = len(rows) > 0
            return passed, f"got {len(rows)} rows"

        actual_str = str(rows)
        expected_str = str(expected)
        passed = expected_str.lower() in actual_str.lower()
        return passed, f"expected {expected_str!r} in {actual_str[:200]!r}"

    except Exception as e:
        return False, f"db_query_match error: {e}"


_CHECKERS: dict[str, Any] = {
    "db_query_match": _db_query_match,
}
