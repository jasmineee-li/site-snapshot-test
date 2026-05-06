from __future__ import annotations

from typing import Any

from worldsim.rewards.network_trace import (
    _expected_http_methods,
    _network_event_expected_fields_match,
    _network_event_method,
    _network_event_url_candidates,
    _resolve_url_placeholders,
    _url_matches,
)


def _eval_network_event(
    config: dict, network_trace: list[dict] | None, instance: dict
) -> tuple[bool, str]:
    """Evaluate a NetworkEventEvaluator config against captured network trace.

    This homebrew path is used for novel task rewards without canonical
    WebArena task IDs. It supports the deterministic subset WorldSim emits:
    URL/method, optional response status, query params, and request body
    evidence.
    """
    if network_trace is None:
        return False, "no network trace captured (required for NetworkEventEvaluator)"

    expected = config.get("expected", {})
    unsupported = set(expected) - {
        "url",
        "reference_url",
        "http_method",
        "response_status",
        "query_params",
        "post_data",
        "post_data_contains",
        "method_requirements",
    }
    if unsupported:
        return (
            False,
            "NetworkEventEvaluator fallback does not support expected fields: "
            + ", ".join(sorted(str(key) for key in unsupported)),
        )
    expected_url = expected.get("url", "")
    expected_methods = _expected_http_methods(expected.get("http_method", "GET"))

    try:
        expected_urls = expected_url if isinstance(expected_url, list) else [expected_url]
        resolved_urls = [
            _resolve_url_placeholders(str(url), instance)
            for url in expected_urls
            if isinstance(url, (str, int, float))
        ]
    except ValueError as exc:
        return False, str(exc)
    if not resolved_urls:
        return False, "NetworkEventEvaluator expected.url must contain at least one URL"

    last_only = config.get("last_event_only", True)
    matching_events = []

    for event in network_trace:
        event_urls = _network_event_url_candidates(event, instance)
        event_method = _network_event_method(event)

        if event_method in expected_methods and any(
            _url_matches(resolved_url, event_url)
            for event_url in event_urls
            for resolved_url in resolved_urls
        ):
            matching_events.append(event)

    if last_only and matching_events:
        matching_events = [matching_events[-1]]

    if not matching_events:
        return False, f"no matching network event for {sorted(expected_methods)} {resolved_urls}"

    filtered_events = []
    for event in matching_events:
        ok, reason = _network_event_expected_fields_match(event, expected)
        if ok:
            filtered_events.append(event)
        elif last_only:
            return False, reason

    if not filtered_events:
        return False, "matching network events failed expected status/query/body checks"

    return True, f"found {len(filtered_events)} matching event(s)"


def _matching_network_events(
    expected: dict[str, Any],
    network_trace: list[dict],
    instance: dict[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    expected_url = expected.get("url", "")
    expected_methods = _expected_http_methods(expected.get("http_method", "GET"))
    try:
        expected_urls = expected_url if isinstance(expected_url, list) else [expected_url]
        resolved_urls = [
            _resolve_url_placeholders(str(url), instance)
            for url in expected_urls
            if isinstance(url, (str, int, float))
        ]
    except ValueError as exc:
        return [], str(exc)
    if not resolved_urls:
        return [], "expected.url must contain at least one URL"

    matched: list[dict[str, Any]] = []
    for event in network_trace:
        if _network_event_method(event) not in expected_methods:
            continue
        event_urls = _network_event_url_candidates(event, instance)
        if not any(
            _url_matches(resolved_url, event_url)
            for event_url in event_urls
            for resolved_url in resolved_urls
        ):
            continue
        ok, reason = _network_event_expected_fields_match(event, expected)
        if ok:
            matched.append(event)
        elif expected.get("last_event_only"):
            return [], reason
    if not matched:
        return [], f"no matching network event for {sorted(expected_methods)} {resolved_urls}"
    return matched, f"found {len(matched)} matching event(s)"


def _expected_network_event_allows_url(
    expected: dict[str, Any],
    event_url: str,
    instance: dict[str, Any],
) -> bool:
    expected_url = expected.get("url", "")
    try:
        expected_urls = expected_url if isinstance(expected_url, list) else [expected_url]
        resolved_urls = [
            _resolve_url_placeholders(str(url), instance)
            for url in expected_urls
            if isinstance(url, (str, int, float))
        ]
    except ValueError:
        return False
    event = {"url": event_url}
    return any(
        _url_matches(resolved_url, candidate_url)
        for candidate_url in _network_event_url_candidates(event, instance)
        for resolved_url in resolved_urls
    )
