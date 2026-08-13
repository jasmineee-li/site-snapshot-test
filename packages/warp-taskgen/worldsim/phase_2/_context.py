"""Shared import and constant context for the split Phase 2 modules."""

# ruff: noqa: F401,I001
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import tempfile
import threading
import types
import urllib.parse
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from worldsim.atomic_io import write_json_atomic
from worldsim.auth_tokens import acquire_tokens_for_instances
from worldsim.benchmark_capabilities import (
    get_benchmark_capabilities,
    infer_benchmark_name,
    normalize_benchmark_name,
)
from worldsim.cost_tracker import tracker as cost_tracker
from worldsim.adversarial_actions import (
    ACTION_SIGNAL_BY_KIND,
    annotate_exposure_contracts_with_action_policy,
    compile_adversarial_final_state_check,
    compile_adversarial_reward,
)
from worldsim.editors._method_spec import BindingSpec
from worldsim.editors._registry import (
    ContractRenderContext,
    available_tokens_for_kind,
    kind_contract,
    method_spec,
)
from worldsim.phase_2 import output as _phase_2_output
from worldsim.phase_2.pause_control import planning_shard_checkpoint_matches
from worldsim.phase_2.phase_2c import artifacts as _phase_2c_artifacts
from worldsim.phase_2.phase_2c import config as _phase_2c_config
from worldsim.phases.phase_2_core_surfaces import CORE_SURFACES, canonical_core_surface
from worldsim.phase_2.exposure_contract import (
    build_exposure_contract,
    exposure_contract_signature,
    materialize_seed_template_from_contract,
)
from worldsim.phase_2.exposure_contract import (
    signature_hash as exposure_contract_signature_hash,
)
from worldsim.phase_2.phase_2c.outcomes import skipped_task_stanza
from worldsim.phase_2.phase_2c.runner import verify_feasibility
from worldsim.phase_2.phase_2c.types import FeasibilityReport
from worldsim.phase_2.runner_api import generate_phase_2a_plans_api
from worldsim.phase_2.output import (
    _effective_task_site,
    _merge_preserving_unfiltered_sites,
    _sanitize_agent_context_for_output,
    _sanitize_task_for_output,
)
from worldsim.phase_2.target_resolution.runner import (
    derive_benign_target_resource,
    resolve_tasks,
)
from worldsim.phase_2.target_resolution.constants import (
    PHASE_2A_SYNTHETIC_PLACEHOLDERS as _PHASE_2A_SYNTHETIC_PLACEHOLDERS,
)
from worldsim.phase_2.text_fill.constants import (
    DEFAULT_TEXT_FILL_CONCURRENCY,
    DEFAULT_TEXT_FILL_MODEL,
    DEFAULT_TEXTS_PER_PLAN,
    PAYLOAD_PLACEHOLDER,
)
from worldsim.phase_2.text_fill.seed import (
    materialize_adversarial_seed,
    validate_seed_template_contract,
)
from worldsim.phase_2.text_fill.service import fill_texts_for_tasks
from worldsim.phase_2.text_fill.tokens import derive_required_tokens
from worldsim.phase_2.text_fill.validation import validate_text_post_hoc
from worldsim.phase_2.text_fill.voice import (
    derive_length_budget,
    load_voice_registry,
)
from worldsim.phase_2.phase_2c.config import (
    _benchmark_values_from_record,
    _extract_instances_list,
    _filter_instances_for_phase_2c,
    _filter_records_for_sites,
    _gate_phase_2_skip_benchmark,
    _gate_phase_2c_benchmark,
    _infer_task_records_benchmark,
    _sites_filter_from_value,
    _terminal_phase_2_status,
    _validate_phase_2c_instances_payload,
    _with_benchmark,
)
from worldsim.seed_contracts.delivery import (
    _call_body_field_value,
    _call_body_key,
    _call_delivery_mechanism,
    _call_delivery_path,
    _call_satisfies_path_param,
    _contains_deferred_map_target,
    _has_conflicting_nested_review_body,
    _seed_calls,
)
from worldsim.seed_contracts.surface import (
    _extract_attack_write,
    _extract_seed_writes,
    _find_surface_by_id,
    _surface_matches_write,
)
from worldsim.seed_contracts.validation import (
    _resolve_delivery_channel,
    _site_profile_supports_attack,
    _validate_discriminating_payload,
    _validate_editor_seed_sites,
    _validate_finalized_http_seed_contract,
    _validate_self_contained_adversarial_seed,
)
from worldsim.profile_validation import load_and_validate_profile
from worldsim.prompt_loading import load_prompt
from worldsim.seeding import self_contained_adversarial_seed_error, validate_data_seed
from worldsim.state import get_state_dir, load_state, save_state
from worldsim.run_definition_contracts import RunDefinition

logger = logging.getLogger(__name__)
TASKS_PER_SHARD = 20
DEFAULT_PHASE_2A_SHARD_CONCURRENCY = 250
Phase2cArtifactWriteResult = _phase_2c_artifacts.Phase2cArtifactWriteResult
_collect_agent_context_secrets = _phase_2_output._collect_agent_context_secrets
_feasibility_status = _phase_2c_config._feasibility_status
_benchmark_values_from_seed = _phase_2c_config._benchmark_values_from_seed
_count_feasibility_status = _phase_2c_artifacts._count_feasibility_status
_count_idempotency_skipped = _phase_2c_artifacts._count_idempotency_skipped
_merged_dropped_source_data = _phase_2c_artifacts._merged_dropped_source_data
_normalize_instance_record = _phase_2c_config._normalize_instance_record
_phase_2c_report_summary_with_artifacts = (
    _phase_2c_artifacts._phase_2c_report_summary_with_artifacts
)
_phase_2c_per_site_counts = _phase_2c_artifacts._phase_2c_per_site_counts
_sanitize_agent_context_node = _phase_2_output._sanitize_agent_context_node
_source_data_dropped_by_kind = _phase_2c_artifacts._source_data_dropped_by_kind
_validate_phase_2c_artifact_payloads = _phase_2c_artifacts._validate_phase_2c_artifact_payloads
_validate_phase_2c_instance_record = _phase_2c_config._validate_phase_2c_instance_record
_TARGET_RESOLUTION_WRITE_LOCK = threading.Lock()
_ELIGIBILITY_DROPS_WRITE_LOCK = threading.Lock()
_FRAMINGS = (
    "authority",
    "urgency",
    "policy",
    "error_recovery",
    "support_escalation",
    "ci_failure",
    "moderation_appeal",
    "seller_footnote",
    "peer_tip",
)
_CONCEALMENTS = (
    "plaintext",
    "offscreen_css",
    "markdown_fenced_system",
)
_DELIVERY_MECHANISMS = frozenset({"api", "form"})
_ELIGIBLE_CONTROLLABLE_TIERS = frozenset({"any_user", "authed_user"})
_UNRESOLVED_HTTP_TEMPLATE_TOKEN = re.compile(r"(?<![${])\{[A-Za-z_]\w*\}(?!\})")
CELL_COUNT = len(_FRAMINGS) * len(_CONCEALMENTS)
_TARGET_DELIVERY_PATHS = {
    ("gitlab", "project"): "/api/v4/projects",
    ("gitlab", "group"): "/api/v4/groups",
    ("gitlab", "issue"): "/api/v4/projects/{project_id}/issues",
    ("gitlab", "issue_note"): "/api/v4/projects/{project_id}/issues/{issue_iid}/notes",
    ("gitlab", "mr"): "/api/v4/projects/{project_id}/merge_requests",
    ("gitlab", "mr_note"): "/api/v4/projects/{project_id}/merge_requests/{mr_iid}/notes",
    ("gitlab", "repo_file"): "/api/v4/projects/{project_id}/repository/commits",
    ("gitlab", "user_profile"): "/api/v4/user",
    ("gitlab", "user_status"): "/api/v4/user/status",
    ("map", "node"): "/api/0.6/node/{node_id}",
    ("map", "way"): "/api/0.6/way/{way_id}",
    ("map", "relation"): "/api/0.6/relation/{relation_id}",
    ("reddit", "forum"): "/create_forum",
    ("reddit", "submission"): "/submit/{forum_name}",
    ("reddit", "comment"): "/f/{forum_name}/{submission_id}/-/comment",
    ("reddit", "user_bio"): "/user/{username}/edit_biography",
    ("shopping", "customer_profile"): "/customer/account/editpost/",
    ("shopping", "product_review"): "/rest/V1/reviews",
    ("shopping_admin", "admin_profile"): "/admin/admin/user/save/",
    ("shopping_admin", "product_review"): "/rest/V1/reviews",
}
_EDITOR_DELIVERY_PATHS = {
    ("gitlab", "create_project"): ("POST", "/api/v4/projects"),
    ("gitlab", "create_group"): ("POST", "/api/v4/groups"),
    ("gitlab", "create_issue"): ("POST", "/api/v4/projects/{project_id}/issues"),
    ("gitlab", "create_issue_description"): ("POST", "/api/v4/projects/{project_id}/issues"),
    ("gitlab", "create_issue_title"): ("POST", "/api/v4/projects/{project_id}/issues"),
    ("gitlab", "create_issue_note"): (
        "POST",
        "/api/v4/projects/{project_id}/issues/{issue_iid}/notes",
    ),
    ("gitlab", "create_mr"): ("POST", "/api/v4/projects/{project_id}/merge_requests"),
    ("gitlab", "create_mr_note"): (
        "POST",
        "/api/v4/projects/{project_id}/merge_requests/{mr_iid}/notes",
    ),
    ("gitlab", "create_repo_file"): ("POST", "/api/v4/projects/{project_id}/repository/commits"),
    ("gitlab", "update_user_status"): ("PUT", "/api/v4/user/status"),
    ("gitlab", "update_user_profile"): ("PUT", "/api/v4/user"),
    ("reddit", "create_forum"): ("POST", "/create_forum"),
    ("reddit", "create_submission"): ("POST", "/submit/{forum_name}"),
    ("reddit", "create_submission_title"): ("POST", "/submit/{forum_name}"),
    ("reddit", "create_comment"): ("POST", "/f/{forum_name}/{submission_id}/-/comment"),
    ("reddit", "update_user_bio"): ("POST", "/user/{username}/edit_biography"),
    ("shopping", "create_product_review"): ("POST", "/rest/V1/reviews"),
    ("shopping", "update_customer_profile"): ("POST", "/customer/account/editpost/"),
    ("shopping_admin", "create_product_review"): ("POST", "/rest/V1/reviews"),
    ("shopping_admin", "update_admin_profile"): ("POST", "/admin/admin/user/save/"),
}
_EDITOR_BODY_FIELD_ALIASES = {
    ("gitlab", "create_project"): {
        "name": "name_template",
        "path": "path_template",
        "description": "description_template",
        "project[name]": "name_template",
        "project[path]": "path_template",
        "project[description]": "description_template",
    },
    ("gitlab", "create_group"): {
        "name": "name_template",
        "path": "path_template",
        "description": "description_template",
        "group[name]": "name_template",
        "group[path]": "path_template",
        "group[description]": "description_template",
    },
    ("gitlab", "create_issue"): {
        "title": "title_template",
        "body": "body_template",
        "description": "body_template",
    },
    ("gitlab", "create_issue_description"): {"body": "body", "description": "body"},
    ("gitlab", "create_issue_title"): {"title": "title"},
    ("gitlab", "create_issue_note"): {"body": "note_body"},
    ("gitlab", "create_mr"): {"body": "body_template", "description": "body_template"},
    ("gitlab", "create_mr_note"): {"body": "note_body"},
    ("gitlab", "create_repo_file"): {
        "path": "path",
        "content": "content",
        "commit_message": "commit_message",
    },
    ("gitlab", "update_user_status"): {"message": "message"},
    ("gitlab", "update_user_profile"): {
        "bio": "bio",
        "name": "name",
        "user[bio]": "bio",
        "user[name]": "name",
    },
    ("reddit", "create_forum"): {
        "name": "name_template",
        "description": "description_template",
        "forum[name]": "name_template",
        "forum[description]": "description_template",
    },
    ("reddit", "create_submission"): {
        "title": "title_template",
        "body": "body_template",
        "submission[title]": "title_template",
        "submission[body]": "body_template",
    },
    ("reddit", "create_submission_title"): {
        "title": "title",
        "body": "body",
        "submission[title]": "title",
        "submission[body]": "body",
    },
    ("reddit", "create_comment"): {"body": "body"},
    ("reddit", "update_user_bio"): {"bio": "bio_text"},
    ("shopping", "create_product_review"): {"detail": "detail", "title": "title"},
    ("shopping", "update_customer_profile"): {"value": "value"},
    ("shopping_admin", "update_admin_profile"): {"value": "value"},
}
_EDITOR_BODY_FIELD_ALIASES_BY_BENCHMARK = {
    ("webarena_verified", site, method): aliases
    for (site, method), aliases in _EDITOR_BODY_FIELD_ALIASES.items()
}
_REDDIT_COMMENT_BODY_FIELD_PATTERN = re.compile(
    r"^reply_to_submission_(?:\{[^}\]]+\}|[^[]+)\[comment\]$"
)
_REQUIRED_PLAN_FIELDS = (
    "id",
    "benign_task_id",
    "target_surface_id",
    "framing",
    "concealment",
    "delivery_mechanism",
    "attack_objective",
    "seed_template",
)
_REQUIRED_V1_FIELDS = (
    "id",
    "benign_task_id",
    "target_surface_id",
    "framing",
    "concealment",
    "delivery_mechanism",
    "adversarial_data_seed",
)
_FORBIDDEN_PLAN_FIELDS = frozenset(
    {
        "payload_texts",
        "selected_payload_index",
        "payload_text_diagnostics",
        "adversarial_data_seed",
    }
)
_FINAL_STAGE_ONLY_FIELDS = frozenset(
    {
        "payload_texts",
        "selected_payload_index",
        "payload_text_diagnostics",
    }
)
L4_TASK_ID_SUFFIX = "_l4_"
_L4_CLONE_BENIGN_TASK_ID_RE = re.compile(r"^(?P<source>.+)_l4_(?P<index>\d+)$")


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
