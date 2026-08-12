from __future__ import annotations

# ruff: noqa: F401
from worldsim.phase_2.text_fill.api import (
    _anthropic_available,
    _call_anthropic_fallback,
    _call_openrouter,
    _call_text_fill_api,
    _direct_anthropic_model_name,
    _format_instructor_retry_exception,
    _openrouter_available,
    _parse_text_fill_response,
    _text_fill_max_tokens,
    is_refusal,
)
from worldsim.phase_2.text_fill.api_errors import TextFillAPIError
from worldsim.phase_2.text_fill.checkpoint_runner import (
    fill_plans_with_checkpoints,
    validate_unique_text_fill_task_ids,
)
from worldsim.phase_2.text_fill.checkpoints import (
    CHECKPOINT_SCHEMA_VERSION,
    CHECKPOINT_STAGE,
    load_text_fill_checkpoint,
    text_fill_checkpoint_matches,
    text_fill_checkpoint_path,
    text_fill_input_digest,
    text_fill_task_id,
    write_text_fill_checkpoint,
)
from worldsim.phase_2.text_fill.constants import (
    _TEXT_FILL_DEFAULT_MAX_TOKENS,
    _TEXT_FILL_EXPANDED_MAX_TOKENS,
    _TEXT_FILL_SHORT_TITLE_MAX_TOKENS,
    DEFAULT_TEXT_FILL_CONCURRENCY,
    DEFAULT_TEXT_FILL_MODEL,
    DEFAULT_TEXTS_PER_PLAN,
    PAYLOAD_PLACEHOLDER,
    REFUSAL_SNIPPETS,
    TEXT_FILL_PROMPT_VARIANTS,
    TEXT_FILL_STRUCTURED_RETRIES,
)
from worldsim.phase_2.text_fill.context import (
    _content_capacity_for_surface,
    _route_evidence_from_phase4,
    _string_or_unknown,
    _surface_route_context,
)
from worldsim.phase_2.text_fill.pause import run_text_fill_units
from worldsim.phase_2.text_fill.payload_views import (
    _contains_offscreen_css,
    _is_offscreen_style,
    _normalize_text_snapshot,
    _payload_text_views,
)
from worldsim.phase_2.text_fill.prompt import (
    _action_text_fill_guidance,
    _adversarial_action_summary,
    _benign_fill_context,
    _binary_semantic_guidance,
    _concealment_instruction,
    _required_content_instruction,
    render_fill_prompt,
)
from worldsim.phase_2.text_fill.seed import (
    _count_placeholder_occurrences,
    _replace_payload_placeholder,
    materialize_adversarial_seed,
    validate_seed_template_contract,
)
from worldsim.phase_2.text_fill.service import (
    _fill_one_task,
    _finalize_task_payloads,
    _generate_single_payload,
    _validate_cached_payloads,
    fill_texts_for_tasks,
)
from worldsim.phase_2.text_fill.tokens import derive_required_tokens
from worldsim.phase_2.text_fill.validation import (
    _action_witness_has_substance,
    _action_witness_is_only_required_token,
    _binary_semantic_post_hoc_errors,
    _blank_semantic_post_hoc_errors,
    _contains_any,
    _encounter_window_post_hoc_errors,
    _length_budget_bounds,
    _no_link_semantic_post_hoc_errors,
    _private_payload_anchor_errors,
    _public_mutation_action_witness_errors,
    _required_token_values,
    _requires_attack_action_witness,
    _scenario_alignment_post_hoc_errors,
    _semantic_evidence_window,
    _tier3_action_contract_post_hoc_errors,
    _tier3_delete_resource_contract_post_hoc_errors,
    _witness_overlaps_benign_instruction,
    validate_text_post_hoc,
)
from worldsim.phase_2.text_fill.voice import (
    _classify_source_field,
    _exemplar_length_budget,
    _field_type,
    _find_surface_by_id,
    _load_site_kind_payload,
    _route_kind_from_task,
    _route_method_from_task,
    _select_exemplars,
    _varchar_limit,
    derive_length_budget,
    load_voice_registry,
    resolve_site_kind,
)
