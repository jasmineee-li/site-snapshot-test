from __future__ import annotations

from typing import Any

from worldsim.rewards.final_state_webarena_verified_gitlab import (
    _gitlab_mutation_source_event_valid,
)
from worldsim.rewards.network_trace import (
    _expected_http_methods,
    _network_event_expected_fields_match,
    _network_event_method,
    _network_event_post_text,
    _network_event_status,
    _network_event_url,
    _network_event_url_candidates,
    _resolve_url_placeholders,
    _url_matches,
)

_NETWORK_EVENT_REWARD_TYPES = frozenset({"NetworkEventEvaluator", "network_event"})
_FINAL_STATE_REWARD_TYPES = frozenset({"FinalStateEvaluator", "final_state"})


def _is_network_event_evaluator_name(name: Any) -> bool:
    return isinstance(name, str) and name in _NETWORK_EVENT_REWARD_TYPES


def extract_network_action_attempt(
    reward: dict[str, Any],
    instance: dict[str, Any],
    network_trace: list[dict] | None,
) -> dict[str, Any]:
    """Return non-scoring network-action attempt evidence for a reward spec.

    This intentionally stops at URL/method target matching. Response status,
    request body requirements, and final-state checks remain stricter success
    signals evaluated by ``run_reward_function``.
    """

    base = {
        "attempted": False,
        "attempt_count": 0,
        "attempt_status": "not_applicable",
        "attempt_last_status": None,
        "attempt_failed_reason": "no_network_action_evaluator",
        "attempt_evidence": [],
    }
    configs = _network_event_attempt_configs(reward)
    if not configs:
        return base
    if network_trace is None:
        return {
            **base,
            "attempt_status": "trace_missing",
            "attempt_failed_reason": "no_network_trace",
        }

    all_evidence: list[dict[str, Any]] = []
    config_errors: list[str] = []
    field_failures: list[str] = []
    for index, config in configs:
        evidence, error = _network_action_attempt_evidence_for_config(
            config,
            network_trace,
            instance,
            evaluator_index=index,
        )
        if error:
            config_errors.append(error)
        all_evidence.extend(evidence)
        field_failures.extend(
            str(item["field_mismatch_reason"])
            for item in evidence
            if item.get("field_mismatch_reason")
        )

    if all_evidence:
        last = all_evidence[-1]
        return {
            **base,
            "attempted": True,
            "attempt_count": len(all_evidence),
            "attempt_status": "attempted",
            "attempt_last_status": last.get("status"),
            "attempt_failed_reason": field_failures[-1] if field_failures else None,
            "attempt_evidence": all_evidence[-3:],
        }
    reason = config_errors[-1] if config_errors else "no_matching_network_event"
    return {
        **base,
        "attempt_status": "not_attempted",
        "attempt_failed_reason": reason,
    }


def _network_event_attempt_configs(
    reward: dict[str, Any],
) -> list[tuple[int | None, dict[str, Any]]]:
    if not isinstance(reward, dict):
        return []
    if "eval" in reward and isinstance(reward["eval"], list):
        configs: list[tuple[int | None, dict[str, Any]]] = []
        for index, config in enumerate(reward["eval"]):
            if not isinstance(config, dict):
                continue
            if _is_network_event_evaluator_name(config.get("evaluator") or config.get("type")):
                configs.append((index, config))
                continue
            final_state_network = _final_state_network_event_attempt_config(config)
            if final_state_network is not None:
                configs.append((index, final_state_network))
        return configs
    eval_type = reward.get("type") or reward.get("evaluator")
    if eval_type in _NETWORK_EVENT_REWARD_TYPES:
        return [(None, reward)]
    final_state_network = _final_state_network_event_attempt_config(reward)
    if final_state_network is not None:
        return [(None, final_state_network)]
    return []


def _final_state_network_event_attempt_config(config: dict[str, Any]) -> dict[str, Any] | None:
    eval_type = config.get("type") or config.get("evaluator")
    if eval_type not in _FINAL_STATE_REWARD_TYPES:
        return None
    expected = config.get("expected")
    if not isinstance(expected, dict):
        return None
    network_event = expected.get("network_event")
    if not isinstance(network_event, dict):
        return None
    return {
        "type": "NetworkEventEvaluator",
        "expected": {
            **dict(network_event),
            "__worldsim_action_kind": expected.get("action_kind"),
            "__worldsim_contains": expected.get("contains"),
        },
        "last_event_only": False,
    }


def _network_action_attempt_evidence_for_config(
    config: dict[str, Any],
    network_trace: list[dict],
    instance: dict[str, Any],
    *,
    evaluator_index: int | None,
) -> tuple[list[dict[str, Any]], str | None]:
    expected = config.get("expected", {})
    if not isinstance(expected, dict):
        return [], "network_event_expected_not_object"
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

    evidence: list[dict[str, Any]] = []
    for event in network_trace:
        event_method = _network_event_method(event)
        if event_method not in expected_methods:
            continue
        event_urls = _network_event_url_candidates(event, instance)
        matched_expected_url = next(
            (
                resolved_url
                for event_url in event_urls
                for resolved_url in resolved_urls
                if _url_matches(resolved_url, event_url)
            ),
            None,
        )
        if matched_expected_url is None:
            continue
        if not _source_action_event_valid_for_attempt(expected, event, instance):
            continue
        fields_ok, field_reason = _network_event_expected_fields_match(event, expected)
        post_text = _network_event_post_text(event)
        evidence.append(
            {
                "evaluator_index": evaluator_index,
                "method": event_method,
                "url": _network_event_url(event),
                "status": _network_event_status(event),
                "matched_expected_url": matched_expected_url,
                "matched_expected_fields": fields_ok,
                "field_mismatch_reason": None if fields_ok else field_reason,
                "post_data_present": bool(post_text),
                "post_data_redacted": post_text == "<redacted>",
            }
        )
    if not evidence:
        return [], f"no matching network event for {sorted(expected_methods)} {resolved_urls}"
    return evidence, None


def _source_action_event_valid_for_attempt(
    expected: dict[str, Any],
    event: dict[str, Any],
    instance: dict[str, Any],
) -> bool:
    site = str(instance.get("site_name") or "").strip().lower()
    action_kind = str(expected.get("__worldsim_action_kind") or "").strip()
    witness = str(expected.get("__worldsim_contains") or "").strip()
    if site == "gitlab" and action_kind in {"create_issue", "create_issue_note"}:
        return _gitlab_mutation_source_event_valid(
            action_kind=action_kind,
            witness=witness,
            expected=expected,
            event=event,
            instance=instance,
        )
    return True
