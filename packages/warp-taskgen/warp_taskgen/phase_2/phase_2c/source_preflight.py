"""Phase 2c source-data and storage-state preflight exports."""

from __future__ import annotations

from warp_taskgen.phase_2.phase_2c import source_data_preflight as _source_data_preflight
from warp_taskgen.phase_2.phase_2c._impl import (
    _infer_records_benchmark,
)
from warp_taskgen.phase_2.phase_2c.auth_preflight import (
    _agent_auth_type,
    _cookie_domain_matches_host,
    _playwright_storage_state_for_preflight,
    _playwright_storage_state_payload_for_preflight,
    _preflight_request_context_options,
    _read_storage_state_payload_for_preflight,
    _resolve_agent_auth_headers,
    _resolve_benign_storage_state_path,
    _storage_state_cookie_hosts,
    _storage_state_origin_hosts,
    _storage_state_preflight_error,
    _storage_state_preflight_error_for_payload,
    _storage_state_recorded_hosts,
)


async def _run_preflight_and_filter_raw(*args, **kwargs):
    source_data_state = _source_data_preflight._patchable_globals()
    _source_data_preflight._agent_auth_type = _agent_auth_type
    _source_data_preflight._preflight_request_context_options = _preflight_request_context_options
    try:
        return await _source_data_preflight._run_preflight_and_filter_raw(*args, **kwargs)
    finally:
        _source_data_preflight._restore_patchable_globals(source_data_state)


__all__ = [
    "_agent_auth_type",
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
