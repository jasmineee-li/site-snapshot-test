"""Phase 2 reuse behavior."""
# ruff: noqa: F821

from __future__ import annotations

from worldsim.phase_2._context import install_context

install_context(globals())


def _load_reusable_phase_2_plans(
    *,
    prior_state: dict[str, Any],
    plans_path: Path,
    sites_filter: set[str] | None,
    expected_benign_task_ids: set[str],
    benign_by_id: dict[str, dict[str, Any]],
    site_profiles: dict[str, dict[str, Any]],
    current_sandbox_model: str,
    current_phase_2a_resolution_signature: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    if prior_state.get("step") != "phase_2":
        return None
    if prior_state.get("phase_2_stage") not in {None, "planning", "text_fill", "feasibility"}:
        return None
    if prior_state.get("status") not in {"running", "failed", "paused"}:
        return None
    if not _resume_setting_matches(
        prior_state,
        field="sandbox_model",
        current_value=current_sandbox_model,
    ):
        return None
    if current_phase_2a_resolution_signature is not None and not _resume_setting_matches(
        prior_state,
        field="phase_2a_resolution_signature",
        current_value=current_phase_2a_resolution_signature,
    ):
        return None
    if not plans_path.exists():
        return None
    try:
        plans = json.loads(plans_path.read_text())
    except Exception:
        return None
    if not isinstance(plans, list):
        return None
    filtered_plans = (
        plans
        if sites_filter is None
        else [plan for plan in plans if str(plan.get("site", "")) in sites_filter]
    )
    if not filtered_plans:
        return None
    _normalize_l4_benign_task_ids_in_place(
        filtered_plans,
        expected_ids=expected_benign_task_ids,
    )
    # Subset check: every plan's benign_task_id must exist in expected, but we
    # don't require every benign task to have a plan (569 plans for 812 tasks is valid).
    plan_benign_ids = {str(p.get("benign_task_id", "")) for p in filtered_plans}
    if not plan_benign_ids.issubset(expected_benign_task_ids) or not plan_benign_ids:
        return None
    if not _identifiers_are_unique(filtered_plans, field="id"):
        return None
    for index, plan in enumerate(filtered_plans):
        if not isinstance(plan, dict):
            return None
        site_profile = site_profiles.get(str(plan.get("site", "")))
        if not isinstance(site_profile, dict):
            return None
        problem = _validate_generated_adversarial_task(
            plan,
            index,
            benign_by_id,
            site_profile,
        )
        if problem is not None:
            logger.warning("Phase 2: ignoring saved adversarial plan reuse because %s", problem)
            return None
    return filtered_plans


def _load_reusable_phase_2_tasks(
    *,
    prior_state: dict[str, Any],
    output_path: Path,
    sites_filter: set[str] | None,
    expected_task_ids: set[str] | None,
    expected_benign_task_ids: set[str] | None,
    texts_per_plan: int,
    benign_by_id: dict[str, dict[str, Any]],
    site_profiles: dict[str, dict[str, Any]],
    current_sandbox_model: str,
    current_text_model: str,
    current_phase_2a_resolution_signature: dict[str, Any] | None = None,
) -> list[dict[str, Any]] | None:
    if prior_state.get("step") != "phase_2":
        return None
    if prior_state.get("status") not in {"running", "failed", "paused"}:
        return None
    if not _resume_setting_matches(
        prior_state,
        field="sandbox_model",
        current_value=current_sandbox_model,
    ):
        return None
    if not _resume_setting_matches(
        prior_state,
        field="phase_2_text_model",
        current_value=current_text_model,
    ):
        return None
    if current_phase_2a_resolution_signature is not None and not _resume_setting_matches(
        prior_state,
        field="phase_2a_resolution_signature",
        current_value=current_phase_2a_resolution_signature,
    ):
        return None
    stage = prior_state.get("phase_2_stage")
    if stage not in {None, "text_fill", "feasibility"}:
        return None
    if stage == "text_fill" and not expected_task_ids:
        return None
    if not output_path.exists():
        return None
    try:
        loaded = json.loads(output_path.read_text())
    except Exception:
        return None
    if not isinstance(loaded, list):
        return None
    tasks = (
        loaded
        if sites_filter is None
        else [task for task in loaded if str(task.get("site", "")) in sites_filter]
    )
    if not tasks:
        return None
    _normalize_l4_benign_task_ids_in_place(
        tasks,
        expected_ids=expected_benign_task_ids,
    )
    if expected_task_ids is not None:
        if not _identifiers_match_exactly(tasks, field="id", expected_ids=expected_task_ids):
            return None
    elif not _identifiers_are_unique(tasks, field="id"):
        return None
    if expected_benign_task_ids is not None and not _identifiers_cover_expected_set(
        tasks,
        field="benign_task_id",
        expected_ids=expected_benign_task_ids,
    ):
        return None
    elif expected_benign_task_ids is None and not _identifiers_are_unique(
        tasks,
        field="benign_task_id",
    ):
        return None
    for index, task in enumerate(tasks):
        problem = _validate_reusable_phase_2_task(
            task,
            task_index=index,
            texts_per_plan=texts_per_plan,
            benign_by_id=benign_by_id,
            site_profiles=site_profiles,
        )
        if problem is not None:
            logger.warning("Phase 2: ignoring saved adversarial task reuse because %s", problem)
            return None
    return tasks


def _resume_setting_matches(
    prior_state: dict[str, Any],
    *,
    field: str,
    current_value: Any,
) -> bool:
    sentinel = object()
    prior_value = prior_state.get(field, sentinel)
    if prior_value is sentinel:
        if field == "phase_2a_resolution_signature" and current_value is not None:
            return False
        return True
    if field == "phase_2a_resolution_signature":
        prior_value = _phase_2a_resolution_signature_comparable(prior_value)
        current_value = _phase_2a_resolution_signature_comparable(current_value)
    return prior_value == current_value


def _phase_2a_resolution_signature_comparable(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    comparable = dict(value)
    comparable.pop("instances_path", None)
    return comparable


def _identifiers_match_exactly(
    items: list[dict[str, Any]],
    *,
    field: str,
    expected_ids: set[str],
) -> bool:
    identifiers = [str(item.get(field, "")) for item in items if isinstance(item, dict)]
    return len(identifiers) == len(expected_ids) and set(identifiers) == expected_ids


def _identifiers_cover_expected_set(
    items: list[dict[str, Any]],
    *,
    field: str,
    expected_ids: set[str],
) -> bool:
    identifiers = [str(item.get(field, "")) for item in items if isinstance(item, dict)]
    return bool(identifiers) and set(identifiers) == expected_ids


def _identifiers_are_unique(
    items: list[dict[str, Any]],
    *,
    field: str,
) -> bool:
    identifiers = [str(item.get(field, "")) for item in items if isinstance(item, dict)]
    return len(identifiers) == len(set(identifiers))


def _validate_reusable_phase_2_task(
    task: object,
    *,
    task_index: int,
    texts_per_plan: int,
    benign_by_id: dict[str, dict[str, Any]],
    site_profiles: dict[str, dict[str, Any]],
) -> str | None:
    if not isinstance(task, dict):
        return f"saved task {task_index} is not an object"
    task_name = f"saved task {task_index} ({task.get('id', '?')})"
    pre_feasibility_only_fields = _phase_2c_only_fields_present(task)
    if pre_feasibility_only_fields:
        return f"{task_name} must not include Phase 2c output fields {pre_feasibility_only_fields}"
    benign_parent = benign_by_id.get(str(task.get("benign_task_id", "")))
    if benign_parent is None:
        return f"{task_name} references unknown benign_task_id {task.get('benign_task_id')!r}"
    site_profile = site_profiles.get(str(task.get("site", "")))
    if not isinstance(site_profile, dict):
        return f"{task_name} references unknown site {task.get('site')!r}"

    violation = _validate_adversarial_task_contract(task, benign_parent, site_profile)
    if violation is not None:
        return f"{task_name} violates adversarial task contract: {violation}"

    stale_contract_reason = _stale_reusable_exposure_contract_reason(task)
    if stale_contract_reason is not None:
        return f"{task_name} has stale exposure_contract: {stale_contract_reason}"

    if "seed_template" not in task:
        final_stage_fields = sorted(_FINAL_STAGE_ONLY_FIELDS.intersection(task.keys()))
        if final_stage_fields:
            return (
                f"{task_name} legacy-shaped task must not include Phase 2b/final-task "
                f"fields {final_stage_fields}"
            )
        return None

    try:
        validate_seed_template_contract(task.get("seed_template"))
    except ValueError as exc:
        return f"{task_name} seed_template invalid: {exc}"

    payload_texts = task.get("payload_texts")
    if not isinstance(payload_texts, list) or len(payload_texts) < texts_per_plan:
        return f"{task_name} payload_texts must contain at least {texts_per_plan} entries"
    if "selected_payload_index" not in task:
        return f"{task_name} missing selected_payload_index"
    selected_index = task.get("selected_payload_index")
    if not isinstance(selected_index, int):
        return f"{task_name} selected_payload_index must be an integer"
    if selected_index < 0 or selected_index >= len(payload_texts):
        return f"{task_name} selected_payload_index is out of range"

    for ordinal, payload in enumerate(payload_texts):
        if not isinstance(payload, dict):
            return f"{task_name} payload_texts[{ordinal}] must be an object"
        payload_errors = validate_text_post_hoc(payload, task)
        if payload_errors:
            return f"{task_name} payload_texts[{ordinal}] invalid: {'; '.join(payload_errors)}"

    selected_payload = payload_texts[selected_index].get("rendered_payload")
    if not isinstance(selected_payload, str) or not selected_payload:
        return f"{task_name} selected payload is missing rendered_payload"
    try:
        rematerialized_seed = materialize_adversarial_seed(task["seed_template"], selected_payload)
    except ValueError as exc:
        return f"{task_name} seed rematerialization failed: {exc}"
    if task.get("adversarial_data_seed") != rematerialized_seed:
        return f"{task_name} adversarial_data_seed does not match seed_template + selected payload"
    return None


def _stale_reusable_exposure_contract_reason(task: dict[str, Any]) -> str | None:
    if str(task.get("site") or "").strip().lower() != "reddit":
        return None
    if task.get("target_surface_id") != "comment.body":
        return None
    seed = task.get("seed_template")
    editor_calls = seed.get("editor_calls") if isinstance(seed, dict) else None
    if not isinstance(editor_calls, list):
        return None
    if not any(
        isinstance(call, dict)
        and call.get("site") == "reddit"
        and call.get("method") == "create_comment"
        for call in editor_calls
    ):
        return None
    contract = task.get("exposure_contract")
    exposure = contract.get("phase4_exposure") if isinstance(contract, dict) else None
    if not isinstance(exposure, dict):
        return "missing_phase4_exposure"
    if exposure.get("requires_exact_comment_region") is not True:
        return "reddit_create_comment_missing_exact_comment_region_gate"
    if exposure.get("encounter_surface") != "seed_appended_comment_region":
        return "reddit_create_comment_uses_legacy_benign_read_surface"
    if exposure.get("admissible") is not True:
        return "reddit_create_comment_exposure_not_admissible"
    if exposure.get("requires_visual_comment_region") is not True:
        return "reddit_create_comment_missing_visual_comment_region_gate"
    if exposure.get("visual_comment_region_forced_by_task") is not True:
        return "reddit_create_comment_visual_comment_region_not_forced"
    if exposure.get("requires_seeded_comment_visibility") is not True:
        return "reddit_create_comment_missing_seeded_comment_visibility_gate"
    if exposure.get("seeded_comment_visible_forced_by_task") is not True:
        return "reddit_create_comment_seeded_comment_visibility_not_forced"
    return None


def _load_text_fill_diagnostics(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        loaded = json.loads(path.read_text())
    except Exception:
        return []
    return loaded if isinstance(loaded, list) else []
