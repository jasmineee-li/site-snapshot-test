"""Phase 2c source-data and storage-state preflight exports."""

from __future__ import annotations

from worldsim.phase_2.phase_2c._impl import (
    _cookie_domain_matches_host,
    _infer_records_benchmark,
    _playwright_storage_state_for_preflight,
    _playwright_storage_state_payload_for_preflight,
    _preflight_request_context_options,
    _read_storage_state_payload_for_preflight,
    _resolve_agent_auth_headers,
    _resolve_benign_storage_state_path,
    _run_preflight_and_filter_raw,
    _storage_state_cookie_hosts,
    _storage_state_origin_hosts,
    _storage_state_preflight_error,
    _storage_state_preflight_error_for_payload,
    _storage_state_recorded_hosts,
)

__all__ = [
    "_cookie_domain_matches_host",
    "_infer_records_benchmark",
    "_playwright_storage_state_for_preflight",
    "_playwright_storage_state_payload_for_preflight",
    "_preflight_request_context_options",
    "_read_storage_state_payload_for_preflight",
    "_resolve_agent_auth_headers",
    "_resolve_benign_storage_state_path",
    "_run_preflight_and_filter_raw",
    "_storage_state_cookie_hosts",
    "_storage_state_origin_hosts",
    "_storage_state_preflight_error",
    "_storage_state_preflight_error_for_payload",
    "_storage_state_recorded_hosts",
]
