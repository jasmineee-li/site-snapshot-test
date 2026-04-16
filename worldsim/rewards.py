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
import os
import re
import subprocess
from collections import Counter
from typing import Any

from worldsim.agent_response_transform import transform_agent_response
from worldsim.db_urls import parse_supported_db_connection
from worldsim.har_converter import ensure_har_trace
from worldsim.placeholders import apply_placeholders, placeholder_for_site

logger = logging.getLogger(__name__)

WEBARENA_EVAL_PYTHON_ENV = "WORLDSIM_WEBARENA_EVAL_PYTHON"
WEBARENA_EVAL_MODULE = "worldsim_webarena_verified.evaluate"

_MULTI_STATEMENT_PATTERN = re.compile(r";(?=(?:[^']|'[^']*')*$)")
_READ_ONLY_QUERY_PREFIX = re.compile(r"^\s*(SELECT|WITH)\b", re.IGNORECASE)
_UNSAFE_QUERY_KEYWORDS = re.compile(
    r"\b("
    r"INSERT|UPDATE|DELETE|DROP|TRUNCATE|ALTER|CREATE|GRANT|REVOKE|REPLACE|MERGE|CALL|"
    r"DO|EXEC|EXECUTE|COPY|LOAD|ATTACH|DETACH|VACUUM|ANALYZE|REFRESH|SET"
    r")\b",
    re.IGNORECASE,
)
_UNSAFE_QUERY_PATTERNS = (
    re.compile(r"\bINTO\b", re.IGNORECASE),
    re.compile(r"\bOUTFILE\b", re.IGNORECASE),
    re.compile(r"\bDUMPFILE\b", re.IGNORECASE),
    re.compile(r"\bLOCK\b", re.IGNORECASE),
    re.compile(r"\bFOR\s+(UPDATE|SHARE)\b", re.IGNORECASE),
)


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
    # WebArena Verified evaluation path (canonical tasks with task_id)
    if "eval" in reward and isinstance(reward["eval"], list):
        if reward.get("task_id") is not None:
            return _run_webarena_verified_eval(reward, instance, agent_result, network_trace)
        # Novel tasks (Mode B) omit task_id, use homebrew evaluator directly
        return _run_homebrew_eval(reward, instance, agent_result, network_trace)

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

    # Deep-copy and apply upstream-bug shims. See _apply_webarena_vendor_shims
    # for the detailed rationale on bugs Bug 1/2/3 in ServiceNow/webarena-verified v1.2.3.
    eval_configs = _apply_webarena_vendor_shims(reward["eval"])
    agent_response = _build_agent_response(eval_configs, agent_result)
    agent_response = _coerce_agent_response_strings(agent_response)
    environments = _build_webarena_environment_payload(instance)
    # Vendor parser rejects both empty lists and our flat CDP format. Convert
    # once here so every downstream call sees a valid non-empty HAR trace.
    har_trace = ensure_har_trace(network_trace)

    subprocess_python = os.environ.get(WEBARENA_EVAL_PYTHON_ENV, "").strip()
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
            "worldsim-webarena-verified environment via "
            f"{WEBARENA_EVAL_PYTHON_ENV} or install 'webarena-verified' in the current environment",
        )

    # Build a WebArenaVerifiedConfig with environments from our instance dict
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


def _build_webarena_environment_payload(instance: dict[str, Any]) -> dict[str, list[str]]:
    """Normalize instance placeholder data into site-name -> urls payload."""
    site_url = instance.get("site_url", "")
    explicit = dict(instance.get("url_placeholders", {}))
    primary_placeholder = placeholder_for_site(instance.get("site_name", ""))
    if primary_placeholder and primary_placeholder not in explicit and site_url:
        explicit[primary_placeholder] = site_url

    placeholder_to_site = {
        "__SHOPPING__": "shopping",
        "__SHOPPING_ADMIN__": "shopping_admin",
        "__GITLAB__": "gitlab",
        "__REDDIT__": "reddit",
        "__WIKIPEDIA__": "wikipedia",
        "__MAP__": "map",
    }

    environments: dict[str, list[str]] = {}
    for placeholder, site_name in placeholder_to_site.items():
        url = explicit.get(placeholder)
        if url:
            environments.setdefault(site_name, []).append(url)

    for key, val in explicit.items():
        if key not in placeholder_to_site and val:
            site_name = key.strip("_").lower()
            environments.setdefault(site_name, []).append(val)

    return environments


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


def _build_agent_response(eval_configs: list[dict], agent_result: Any | None) -> dict[str, Any]:
    """Build a WebArena Verified-compatible agent response dict.

    When the task has an ``AgentResponseEvaluator``, first try the 4-strategy
    extractor from ``worldsim.agent_response_transform``. This matches the
    vendor's canonical CLI behavior: ``eval-tasks`` wires the extractor in via
    ``--agent-response-transform`` and prefers the transformed dict over the
    raw text when it yields a JSON object (see
    ``vendors/webarena-verified/src/webarena_verified/__main__.py`` around
    ``_transform_agent_response`` at lines 944-950). Without this we score 0
    on runs where the content is correct but the JSON wrapper is mangled,
    while the benchmark's own CLI scores 1. Scope is intentionally limited to
    tasks that declare an ``AgentResponseEvaluator`` -- other evaluator types
    (and other benchmarks) are left untouched.
    """

    # Infer task type from the expected response in the first AgentResponseEvaluator
    task_type = "retrieve"
    has_agent_response_evaluator = False
    for config in eval_configs:
        if config.get("evaluator") == "AgentResponseEvaluator":
            has_agent_response_evaluator = True
            expected = config.get("expected", {})
            task_type = expected.get("task_type", "retrieve")
            break

    final_result = None
    if agent_result is not None:
        final_result = getattr(agent_result, "final_result", None)

    is_done = agent_result is not None and getattr(agent_result, "is_done", False)

    # Prefer a transform-extracted dict for WebArena-Verified
    # AgentResponseEvaluator tasks. Fall back to the prose-wrapping behavior
    # when no strategy matches (transform returns None).
    if has_agent_response_evaluator and isinstance(final_result, str) and final_result.strip():
        transformed = transform_agent_response(final_result)
        if transformed is not None:
            return transformed

    # Parse the agent's final answer into retrieved_data format
    retrieved_data = None
    if task_type == "retrieve" and final_result:
        retrieved_data = _parse_retrieved_data(final_result)

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


def _eval_agent_response(config: dict, agent_response: dict) -> tuple[bool, str]:
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
    try:
        resolved_url = _resolve_url_placeholders(expected_url, instance)
    except ValueError as exc:
        return False, str(exc)

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
    explicit = dict(instance.get("url_placeholders", {}))
    primary_placeholder = placeholder_for_site(instance.get("site_name", ""))
    site_url = instance.get("site_url", "")
    if primary_placeholder and primary_placeholder not in explicit and site_url:
        explicit[primary_placeholder] = site_url
    return apply_placeholders(url, explicit, strict=True)


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


def _db_query_match(reward: dict[str, Any], instance: dict[str, Any]) -> tuple[bool, str]:
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
        _validate_read_only_query(query)
    except ValueError as exc:
        return False, str(exc)

    try:
        parsed = parse_supported_db_connection(
            db_conn,
            purpose="db_query_match requires instance['db_connection']",
        )
        scheme = parsed.scheme.lower()

        if scheme == "mysql":
            import pymysql

            conn = pymysql.connect(
                host=parsed.hostname,
                port=parsed.port or 3306,
                user=parsed.username,
                password=parsed.password,
                database=(parsed.path or "").lstrip("/"),
            )
        elif scheme in ("postgresql", "postgres"):
            import psycopg2  # late import — only needed on the PostgreSQL path

            conn = psycopg2.connect(
                host=parsed.hostname,
                port=parsed.port or 5432,
                user=parsed.username,
                password=parsed.password,
                dbname=(parsed.path or "").lstrip("/"),
            )
        else:
            return False, f"unsupported DB dialect: {scheme}"

        try:
            _configure_read_only_connection(conn, scheme)
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
        finally:
            try:
                conn.rollback()
            except Exception:
                logger.debug("Failed to rollback read-only query connection", exc_info=True)
            conn.close()

        # Compare result
        if expected is None:
            passed = len(rows) > 0
            return passed, f"got {len(rows)} rows"

        passed = _rows_match_expected(rows, expected)
        return passed, f"expected {expected!r}; got rows {rows[:5]!r}"

    except Exception as e:
        return False, f"db_query_match error: {e}"


def _rows_match_expected(rows: Any, expected: Any) -> bool:
    """Compare SQL rows against an expected value without substring matching."""
    if expected is None:
        return bool(rows)
    if not isinstance(rows, (list, tuple)):
        return _values_match(rows, expected)

    if _values_match(rows, expected):
        return True

    for row in rows:
        if _values_match(row, expected):
            return True
        if isinstance(row, (list, tuple)):
            for cell in row:
                if _values_match(cell, expected):
                    return True
    return False


def _values_match(actual: Any, expected: Any) -> bool:
    """Return True when two SQL result values are materially equal."""
    if isinstance(actual, tuple):
        actual = list(actual)
    if isinstance(expected, tuple):
        expected = list(expected)

    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _values_match(left, right) for left, right in zip(actual, expected, strict=True)
        )

    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            return False
        return all(_values_match(actual[key], expected[key]) for key in actual)

    if type(actual) is type(expected):
        return actual == expected

    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return actual == expected

    if isinstance(actual, str) or isinstance(expected, str):
        return str(actual).strip().casefold() == str(expected).strip().casefold()

    return actual == expected


_CHECKERS: dict[str, Any] = {
    "db_query_match": _db_query_match,
}


def _validate_read_only_query(query: str) -> None:
    """Restrict db_query_match to a single read-only query."""
    normalized = query.strip()
    if not normalized:
        raise ValueError("db_query_match query is empty")
    if _MULTI_STATEMENT_PATTERN.search(normalized.rstrip(";")):
        raise ValueError("db_query_match must be a single statement")
    if not _READ_ONLY_QUERY_PREFIX.match(normalized):
        raise ValueError("db_query_match only permits SELECT or WITH queries")
    if _UNSAFE_QUERY_KEYWORDS.search(normalized):
        raise ValueError("db_query_match query contains a write-capable SQL keyword")
    for pattern in _UNSAFE_QUERY_PATTERNS:
        if pattern.search(normalized):
            raise ValueError("db_query_match query contains a write-capable SQL clause")


def _configure_read_only_connection(conn: Any, scheme: str) -> None:
    """Enable a required read-only transaction guard for SQL reward checks."""
    try:
        if hasattr(conn, "autocommit"):
            conn.autocommit = False
        with conn.cursor() as cursor:
            if scheme == "mysql":
                cursor.execute("SET SESSION TRANSACTION READ ONLY")
                cursor.execute("START TRANSACTION READ ONLY")
            elif scheme in ("postgresql", "postgres"):
                cursor.execute("BEGIN")
                cursor.execute("SET TRANSACTION READ ONLY")
            else:
                raise ValueError(f"unsupported DB dialect: {scheme}")
    except Exception as exc:
        raise RuntimeError("could not enable read-only transaction guard") from exc
