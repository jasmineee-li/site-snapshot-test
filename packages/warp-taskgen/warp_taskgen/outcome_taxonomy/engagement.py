"""Injection exposure and attacker-engagement signal exports."""

from __future__ import annotations

from warp_taskgen.outcome_taxonomy._impl import (
    _any_document_nav_matches,
    _check_attacker_token_in_output,
    _check_attacker_url_visited,
    _check_injection_content_parsed,
    _check_injection_surface_visited,
    _classify_action,
    _classify_refusal_awareness,
    _collect_agent_text_corpus,
    _collect_platform_observable_corpus,
    _legacy_registry_match,
    _normalize_match_text,
    _payload_witnesses,
    _rendered_payloads,
    _witness_matches_corpus,
)

__all__ = [
    "_any_document_nav_matches",
    "_check_attacker_token_in_output",
    "_check_attacker_url_visited",
    "_check_injection_content_parsed",
    "_check_injection_surface_visited",
    "_classify_action",
    "_classify_refusal_awareness",
    "_collect_agent_text_corpus",
    "_collect_platform_observable_corpus",
    "_legacy_registry_match",
    "_normalize_match_text",
    "_payload_witnesses",
    "_rendered_payloads",
    "_witness_matches_corpus",
]
