"""Shared import and constant context for split Phase 4 modules."""

# ruff: noqa: F401,I001
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import shutil
import time
import types
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
import requests
from worldsim import outcome_taxonomy
from worldsim.agent_auth import resolve_agent_auth_headers
from worldsim.agent_config import (
    DEFAULT_MODEL,
    RUNTIME_METADATA_KEY,
    bind_task_to_instance,
    cap_tasks_per_site,
    execution_instance_dict,
    execution_site_instance_dict,
    instances_for_site,
    make_agent_factory,
    resolve_task_inputs,
    run_tasks_by_site,
    task_reset_endpoints,
)
from worldsim.agent_prompt import build_agent_prompt
from worldsim.atomic_io import write_json_atomic
from worldsim.auth_tokens import acquire_tokens_for_instances
from worldsim.benchmark_capabilities import get_benchmark_capabilities, infer_benchmark_name
from worldsim.browser_use_agent import AgentResult, AgentRunner
from worldsim.config import (
    BenchmarkConfig,
    BenchmarkInstance,
    has_effective_agent_auth,
    load_benchmark_config,
)
from worldsim.cost_tracker import tracker as cost_tracker
from worldsim.editors import EDITOR_REGISTRY, EditorError
from worldsim.failpoints import crash_if_enabled
from worldsim.instance_selection import select_task_site_instance
from worldsim.modal_sandbox import preflight_auth_check
from worldsim.phase_4 import result_summary as phase4_result_summary
from worldsim.phase_4.aer_trajectory_extract import (
    ExtractedTrajectory,
    extract_task_intent,
    extract_trajectory,
)
from worldsim.phase_4.encounter_detection import EncounterResult, determine_encounter
from worldsim.phase_4.exposure_admission import (
    exposure_admission_error as _exposure_admission_error,
)
from worldsim.phase_4.needham_trace import format_trace_xml, write_trace_artifacts
from worldsim.phase_4.payload_witnesses import witness_texts_for_task
from worldsim.phase_4.placement_api import run_placement_api
from worldsim.phase_4.strategy_catalog import ALLOWED_STRATEGIES as _ALLOWED_STRATEGIES
from worldsim.phase_4.transcript_purpose_api import (
    _failure_result as _transcript_purpose_failure_result,
)
from worldsim.phase_4.transcript_purpose_api import (
    run_transcript_purpose_api,
)
from worldsim.phase_4.verbalized_eval_awareness_api import (
    _failure_result as _vea_failure_result,
)
from worldsim.phase_4.verbalized_eval_awareness_api import (
    run_verbalized_eval_awareness_api,
)
from worldsim.phases.phase_2_text_fill import (
    materialize_adversarial_seed,
    validate_seed_template_contract,
    validate_text_post_hoc,
)
from worldsim.placeholders import (
    merge_placeholder_maps,
    normalize_site_name,
    normalize_task_sites,
    placeholder_for_site,
)
from worldsim.profile_validation import load_and_validate_profile, profile_requires_agent_auth
from worldsim.pvpo_endpoint import canonical_pvpo_endpoint_identity, validate_pvpo_cdp_url
from worldsim.resume_metadata import (
    RESULT_FINGERPRINT_KEY,
    fingerprint_payload,
    instance_identity,
    instances_identity,
)
from worldsim.adversarial_actions import (
    action_metadata_for_task,
    benign_reward_shape_from_task,
    reward_signal_for_task,
)
from worldsim.adversarial_actions.tier3_fixtures import (
    cleanup_tier3_delete_project_action_target,
    cleanup_tier3_repository_action_target,
)
from worldsim.rewards import extract_network_action_attempt, run_reward_function
from worldsim.seeding import (
    apply_data_seed_async,
    collect_seed_runtime_errors,
    preflight_editor_seed_calls,
    self_contained_adversarial_seed_error,
    validate_data_seed,
)
from worldsim.seeding import (
    seed_has_actions as _seed_contract_has_actions,
)
from worldsim.seeding import (
    seed_requires_reset as _seed_contract_requires_reset,
)
from worldsim.site_lock import task_lock
from worldsim.state import get_state_dir, save_state
from worldsim.storage_state_preflight import (
    StorageStatePreflightError,
    apply_skip_auth_for_host_bound_storage_states,
    inspect_storage_state_preflight,
)
from worldsim.task_paths import safe_task_path_component
from worldsim.task_reset_cache import (
    TaskResetCache,
    callable_accepts_keyword,
    result_likely_mutated_state,
)
from worldsim.trajectory import save_result

logger = logging.getLogger(__name__)
_CHECKPOINT_FINGERPRINT_KEY = "_source_fingerprint"
_PLACEMENT_FIX_CHECKPOINT = "placement_fix_checkpoint.json"
_VARIANT_GENERATION_RECORDS_KEY = "variant_generation_records"
_VARIANT_RESULT_METADATA = "resume_metadata.json"
_PHASE_4_RESUME_VERSION = "2026-04-20b"
PLACEMENT_FIX_MAX_ITERATIONS = 2
_LEGACY_AER_INFLIGHT_SENTINEL = ".aer_inflight"
_ACTION_REWARD_SIGNALS = frozenset({"network_action", "external_navigation"})
_PVPO_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "pvpo_status",
        "pvpo_failure",
        "transcript_purpose_classification",
        "transcript_purpose_is_aware",
        "transcript_purpose_was_refused",
        "transcript_purpose_guess",
        "transcript_purpose_explanation",
        "transcript_purpose_failure",
        "transcript_purpose_format",
        "transcript_purpose_attempts",
        "transcript_purpose_raw_response_path",
        "transcript_purpose_apollo_continue_classification",
        "transcript_purpose_apollo_continue_is_aware",
        "transcript_purpose_apollo_continue_was_refused",
        "transcript_purpose_apollo_continue_guess",
        "transcript_purpose_apollo_continue_explanation",
        "transcript_purpose_apollo_continue_failure",
        "transcript_purpose_apollo_continue_attempts",
        "transcript_purpose_apollo_continue_raw_response_path",
        "verbalized_eval_awareness_classification",
        "verbalized_eval_awareness_is_aware",
        "verbalized_eval_awareness_evidence",
        "verbalized_eval_awareness_failure",
        "verbalized_eval_awareness_transcript_format",
        "verbalized_eval_awareness_raw_response_path",
    }
)
_PAYLOAD_BODY_FIELD_HINTS: tuple[str, ...] = (
    "detail",
    "body",
    "content",
    "message",
    "comment",
    "bio",
    "description",
    "text",
    "value",
    "html",
)
_MARKDOWN_SYSTEM_BLOCK_RE = re.compile(r"```system\s*(.*?)```", re.DOTALL | re.IGNORECASE)
STRICT_FEASIBILITY_ADMISSION = True
LAYOUT_SCROLL_BUCKETS: tuple[tuple[str, float], ...] = (
    ("entry", 0),
    ("near", 3000),
    ("deep", 10000),
)
_FINGERPRINT_RESULT_KEYS: tuple[str, ...] = (
    "task_id",
    "outcome",
    "encounter",
    "final_status",
    "seed_preflight_mismatches",
    "seed_preflight_mismatch_records",
    "benign_passed",
    "benign_reward_message",
    "adversarial_passed",
    "adversarial_reward_message",
    "adversarial_reward_signal",
    "adversarial_final_state_checked",
    "adversarial_final_state_passed",
    "adversarial_final_state_message",
    "adversarial_action_kind",
    "adversarial_action_tier",
    "adversarial_action_family",
    "adversarial_action_enabled",
    "attack_action_attempted",
    "attack_action_attempt_reason",
    "adversarial_action_attempted",
    "adversarial_action_attempt_count",
    "adversarial_action_attempt_evidence",
    "adversarial_action_attempt_status",
    "adversarial_action_attempt_last_status",
    "adversarial_action_attempt_failed_reason",
    "attack_action_success",
    "state_confirmed_action_success",
    "tier3_state_confirmed_action_success",
    "semantic_hijack_success",
    "benign_success_under_attack",
    "stealth_success",
    "trajectory_dir",
    "elapsed",
    "steps",
    "final_result",
    "error",
    "pvpo_status",
    "pvpo_failure",
    "pvpo_observation",
    "transcript_purpose_classification",
    "transcript_purpose_is_aware",
    "transcript_purpose_guess",
    "transcript_purpose_explanation",
    "transcript_purpose_failure",
    "transcript_purpose_format",
    "transcript_purpose_attempts",
    "transcript_purpose_raw_response_path",
    "verbalized_eval_awareness_classification",
    "verbalized_eval_awareness_is_aware",
    "verbalized_eval_awareness_evidence",
    "verbalized_eval_awareness_failure",
    "verbalized_eval_awareness_transcript_format",
    "verbalized_eval_awareness_raw_response_path",
    "infrastructure_retry",
    "outcome_fine",
    "flags",
    "diagnosable_confidence",
    "signals",
    "classifier_version",
)
_RESET_TIMEOUT = 300
_RESET_MAX_RETRIES = 2
_RESET_RETRY_DELAY = 10

_VARIANT_ROUNDS_KEY = "variant_rounds"
_ADAPTIVE_VARIANT_BUDGET = (3, 3, 1)
_PHASE_4_VARIANT_BUDGET_PRESETS: dict[str, tuple[int, ...]] = {
    "adaptive-3-3-1": _ADAPTIVE_VARIANT_BUDGET,
    "smoke-3-probe": (3,),
}
_DEFAULT_PHASE_4_VARIANT_BUDGET_PRESET = "adaptive-3-3-1"


def phase_4_variant_budget_choices() -> tuple[str, ...]:
    return tuple(_PHASE_4_VARIANT_BUDGET_PRESETS)


def _phase_4_variant_budget_shape(preset: str | None) -> tuple[int, ...]:
    normalized = (preset or _DEFAULT_PHASE_4_VARIANT_BUDGET_PRESET).strip()
    return _PHASE_4_VARIANT_BUDGET_PRESETS.get(
        normalized,
        _PHASE_4_VARIANT_BUDGET_PRESETS[_DEFAULT_PHASE_4_VARIANT_BUDGET_PRESET],
    )


def install_context(namespace: dict[str, object]) -> None:
    for name, value in globals().items():
        if name.startswith("__") or name == "install_context" or name == "link_modules":
            continue
        namespace.setdefault(name, value)


def link_modules(modules: list[object]) -> None:
    class _LinkedModule(types.ModuleType):
        def __setattr__(self, name: str, value: object) -> None:
            types.ModuleType.__setattr__(self, name, value)
            for module in getattr(self, "_linked_modules", []):
                vars(module)[name] = value

    combined: dict[str, object] = {}
    for module in modules:
        for name, value in vars(module).items():
            if name.startswith("__") or name in {"install_context", "link_modules"}:
                continue
            combined[name] = value
    for module in modules:
        vars(module).update(combined)
        vars(module)["_linked_modules"] = modules
        if isinstance(module, types.ModuleType):
            module.__class__ = _LinkedModule
