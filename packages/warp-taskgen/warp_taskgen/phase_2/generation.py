"""Phase 2 generation behavior."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from warp_taskgen.adversarial_actions import (
    annotate_exposure_contracts_with_action_policy,
    compile_adversarial_final_state_check,
    compile_adversarial_reward,
)
from warp_taskgen.editors._registry import ContractRenderContext
from warp_taskgen.phase_1.gitlab_compare_decide import (
    bind_gitlab_compare_decide_benign_resource,
)
from warp_taskgen.phase_2 import eligibility as _eligibility
from warp_taskgen.phase_2 import option_a as _option_a
from warp_taskgen.phase_2 import plan_validation as _plan_validation
from warp_taskgen.phase_2 import runner_api as _runner_api
from warp_taskgen.phase_2 import target_stage as _target_stage
from warp_taskgen.phase_2.exposure_contract import materialize_seed_template_from_contract
from warp_taskgen.phase_2.output import (
    _sanitize_agent_context_for_output,
    _sanitize_task_for_output,
)
from warp_taskgen.phase_2.pause_control import write_planning_shard_checkpoint
from warp_taskgen.phase_2.planning_types import SiteInjectionResult
from warp_taskgen.phase_2.runtime_generation import generation_for_runtime
from warp_taskgen.phase_2.target_resolution.constants import (
    PHASE_2A_SYNTHETIC_PLACEHOLDERS as _PHASE_2A_SYNTHETIC_PLACEHOLDERS,
)
from warp_taskgen.phase_2.target_resolution.runner import derive_benign_target_resource
from warp_taskgen.phase_2.text_fill.tokens import derive_required_tokens
from warp_taskgen.phase_2.text_fill.voice import derive_length_budget, load_voice_registry
from warp_taskgen.profile_validation import load_and_validate_profile
from warp_taskgen.prompt_loading import load_prompt
from warp_taskgen.seed_contracts.surface import _find_surface_by_id
from warp_taskgen.seed_contracts.validation import _resolve_delivery_channel
from warp_taskgen.state import get_state_dir

if TYPE_CHECKING:
    from warp_taskgen.runtime_composition import RuntimeComposition

logger = logging.getLogger(__name__)


async def _generate_injections_for_site(
    site_name: str,
    site_tasks: list[dict],
    all_site_tasks: list[dict] | None = None,
    profile_path: Path | None = None,
    site_profile_override: Mapping[str, Any] | None = None,
    label: str | None = None,
    sandbox_model: str = "claude-sonnet-4-6",
    instance: Mapping[str, Any] | None = None,
    benchmark: str = "webarena_verified",
    action_policy: str | None = None,
    runtime_composition: RuntimeComposition | None = None,
) -> SiteInjectionResult:
    """Generate adversarial injections for one shard through API Phase 2a.

    Args:
        site_name: Canonical site name (e.g. "shopping").
        site_tasks: The subset of benign tasks staged into this sandbox.
        all_site_tasks: The full set of benign tasks for the site, used for
            post-processing validation. Defaults to *site_tasks* when None
            (single-shard case).
        profile_path: Path to the site's benchmark profile.
        label: Sandbox label for logging / Modal UI.
        instance: Optional live benchmark instance descriptor for this
            site (``{site_url, auth, storage_state_path, ...}``). When
            present, :func:`resolve_tasks` runs L1/L2/L3/L4 against the
            live instance so intent-only tasks arrive at 2a with
            concrete anchors and listing kinds fan out to one clone per
            top-N item (suffixed IDs). Absent, the legacy L1/L2-only
            :func:`derive_benign_target_resource` path runs offline and
            the task count is preserved.

    Returns:
        Validated sandbox output for the shard.
    """
    if all_site_tasks is None:
        all_site_tasks = site_tasks
    input_task_ids = [str(task.get("id") or "") for task in site_tasks]
    if label is None:
        label = site_name

    if profile_path is None or not profile_path.exists():
        logger.warning("No profile for site %r at %s — skipping", site_name, profile_path)
        return SiteInjectionResult(site_name, [], [f"profile not found at {profile_path}"])

    # Inputs both paths need in memory.
    site_profile = (
        json.loads(json.dumps(site_profile_override))
        if isinstance(site_profile_override, Mapping)
        else json.loads(profile_path.read_text())
    )
    feature_generation = generation_for_runtime(
        runtime_composition,
        benchmark=benchmark,
        site=site_name,
    )

    # Pre-compute benign-target resources (Option A placement contract,
    # docs/handoffs/phase-2-placement-systemic-gap.md). 2a consumes this
    # to constrain delivery_channel.method to attach_surfaces per task.
    #
    # When a live instance is configured for this site, run the async
    # resolver batch (L1/L2/L3/L4) so intent-only tasks that L1/L2 left
    # as pending_layer="L3" get concrete kind + anchors via Anthropic
    # intent-parse + live API probe, and listing kinds fan out to N
    # concrete per-item records via suffixed-ID clones. Absent an
    # instance, the offline L1/L2-only path mirrors today's behavior
    # exactly and the task count is preserved.
    prepared_feature_shard = None
    if feature_generation is not None:
        try:
            prepared_feature_shard = feature_generation.prepare_shard(
                site_tasks,
                runtime_composition,
            )
        except ValueError as exc:
            return SiteInjectionResult(site_name, [], [f"feature planning failed: {exc}"])
        site_tasks = prepared_feature_shard.tasks
        benign_target_resources = prepared_feature_shard.benign_target_resources
        exposure_contracts = prepared_feature_shard.exposure_contracts
        eligibility_drops = prepared_feature_shard.eligibility_drops
    else:
        (
            site_tasks,
            benign_target_resources,
        ) = await _target_stage._resolve_benign_target_resources_for_shard(
            site_tasks=site_tasks,
            instance=instance,
            site_name=site_name,
            label=label,
            benchmark=benchmark,
        )
    # L4 clones live only in this shard's local view; share the
    # expansion with the validator/merge step that runs against
    # *all_site_tasks* by substituting the expanded list whenever
    # expansion actually happened.
    if any(_target_stage.L4_TASK_ID_SUFFIX in str(t.get("id", "")) for t in site_tasks):
        all_site_tasks = site_tasks
    if feature_generation is None:
        exposure_contracts = _eligibility._build_exposure_contracts_for_shard(
            site_tasks=site_tasks,
            benign_target_resources=benign_target_resources,
            site=site_name,
            benchmark=benchmark,
            surface_visibility_by_id=_eligibility._surface_visibility_by_id(site_profile),
        )
        exposure_contracts = annotate_exposure_contracts_with_action_policy(
            exposure_contracts,
            site_tasks,
            policy=action_policy or "default",
        )
    _eligibility._persist_exposure_contracts(site_name=site_name, contracts=exposure_contracts)
    if feature_generation is None:
        site_tasks, eligibility_drops = _eligibility._phase_2a_eligible_tasks_for_benchmark(
            site_tasks,
            benign_target_resources,
            site_name,
            benchmark=benchmark,
            exposure_contracts=exposure_contracts,
        )
    if eligibility_drops:
        _eligibility._write_eligibility_drops(site_name, eligibility_drops)
    if not site_tasks:
        logger.info(
            "Phase 2: shard %r has no eligible tasks after target-resolution filtering", label
        )
        return SiteInjectionResult(site_name, [], [])
    cell_targets = _eligibility._build_cell_targets(site_profile, site_tasks, all_site_tasks)

    agent_context_path = profile_path.parent / f"AGENT_CONTEXT_{site_name}.json"
    agent_context: dict[str, Any] | None = None
    if agent_context_path.exists():
        try:
            agent_context = json.loads(agent_context_path.read_text())
        except json.JSONDecodeError as exc:
            logger.warning(
                "Phase 2: invalid AGENT_CONTEXT at %s, proceeding without: %s",
                agent_context_path,
                exc,
            )

    if prepared_feature_shard is not None:
        # An explicit composition may own a deterministic plan path outside
        # the ordinary model planner while preserving the same output contract.
        adv_tasks = prepared_feature_shard.plans
    else:
        logger.info("Phase 2: launching injection API call %r (%d tasks)", label, len(site_tasks))
        sanitized_site_tasks = [_sanitize_task_for_output(task) for task in site_tasks]
        sanitized_agent_context = (
            _sanitize_agent_context_for_output(agent_context) if agent_context is not None else None
        )
        adv_tasks = await _runner_api.generate_phase_2a_plans_api(
            benign_tasks=sanitized_site_tasks,
            benign_target_resources=benign_target_resources,
            exposure_contracts=exposure_contracts,
            cell_targets=cell_targets,
            benchmark_profile=site_profile,
            agent_context=sanitized_agent_context,
            sandbox_model=sandbox_model,
            label=label,
            site=site_name,
            benchmark=benchmark,
            runtime_composition=runtime_composition,
        )
    if not adv_tasks:
        logger.warning("Phase 2: API path %r produced no plans", label)
        return SiteInjectionResult(
            site_name,
            [],
            ["API path produced no adversarial plans"],
        )
    if prepared_feature_shard is None:
        try:
            _materialize_strategy_plans_from_exposure(
                adv_tasks,
                exposure_contracts=exposure_contracts,
                benchmark=benchmark,
                benign_tasks=all_site_tasks,
            )
        except ValueError as exc:
            return SiteInjectionResult(site_name, [], [f"exposure materialization failed: {exc}"])

    # Programmatically copy immutable fields from benign tasks instead of
    # relying on the LLM to reproduce them byte-for-byte.
    try:
        merge_kwargs: dict[str, Any] = {
            "enriched_resources": benign_target_resources,
            "exposure_contracts": exposure_contracts,
        }
        _merge_immutable_fields(adv_tasks, all_site_tasks, **merge_kwargs)
    except ValueError as exc:
        return SiteInjectionResult(site_name, [], [f"host reward compilation failed: {exc}"])

    if feature_generation is not None:
        enriched, errors = feature_generation.validate_and_enrich_plans(
            adv_tasks,
            all_site_tasks,
            exposure_contracts=exposure_contracts,
            runtime_composition=runtime_composition,
        )
    else:
        validated, errors = _plan_validation._validate_generated_adversarial_tasks(
            adv_tasks,
            all_site_tasks,
            site_profile,
        )
        try:
            enriched = _materialize_validated_shard_tasks(validated, site_profile)
        except ValueError as exc:
            return SiteInjectionResult(site_name, [], [f"plan enrichment failed: {exc}"])
    enriched = _eligibility._select_balanced_subset(enriched, cell_targets)
    _target_stage._normalize_l4_benign_task_ids_in_place(enriched)

    # Persist this shard's validated output to disk immediately so a later
    # orchestrator failure (or another shard's failure) cannot discard it.
    if enriched:
        shards_dir = get_state_dir() / "phase_2" / "shards"
        shards_dir.mkdir(parents=True, exist_ok=True)
        shard_path = shards_dir / f"{label}.json"
        try:
            write_planning_shard_checkpoint(
                shard_path,
                enriched,
                label=label,
                input_task_ids=input_task_ids,
            )
            logger.info(
                "Phase 2: persisted shard %r output (%d tasks) to %s",
                label,
                len(enriched),
                shard_path,
            )
        except (OSError, ValueError) as exc:
            checkpoint_error = f"failed to persist Run-bound shard checkpoint: {exc}"
            logger.error("Phase 2: shard %r %s", label, checkpoint_error)
            return SiteInjectionResult(site_name, [], [*errors, checkpoint_error])

    return SiteInjectionResult(site_name, enriched, errors)


def _materialize_validated_shard_tasks(
    validated: list[dict[str, Any]],
    site_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    if not validated:
        return []
    legacy_tasks = [task for task in validated if "seed_template" not in task]
    plan_tasks = [task for task in validated if "seed_template" in task]
    if not plan_tasks:
        return legacy_tasks
    enriched_plans = _enrich_adversarial_plans(plan_tasks, site_profile)
    enriched_by_id = {str(task.get("id", "")): task for task in enriched_plans}
    materialized: list[dict[str, Any]] = []
    for task in validated:
        if "seed_template" not in task:
            materialized.append(task)
            continue
        task_id = str(task.get("id", ""))
        enriched = enriched_by_id.get(task_id)
        if enriched is None:
            raise ValueError(f"missing enriched plan for task {task_id!r}")
        materialized.append(enriched)
    return materialized


def _merge_immutable_fields(
    adv_tasks: list[dict],
    benign_tasks: list[dict],
    *,
    enriched_resources: Mapping[str, Mapping[str, Any]] | None = None,
    exposure_contracts: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Copy immutable fields from benign tasks into adversarial task dicts.

    Handles both the full schema (where ``reward_function`` already exists)
    and the minimal schema (where only ``adversarial_reward`` is present).

    When ``enriched_resources`` is supplied (L3/L4 resolver output keyed
    by task id including L4 suffixed clones), use it directly for the
    ``benign_target_resource`` merge instead of re-deriving via L1/L2 —
    otherwise the L3 anchors we paid to resolve would be thrown away on
    the way out of Phase 2a.
    """
    benign_by_id = {str(t.get("id", "")): t for t in benign_tasks}
    for adv_task in adv_tasks:
        benign_id = str(adv_task.get("benign_task_id", ""))
        benign_task = benign_by_id.get(benign_id)
        if benign_task is None:
            continue

        # Copy immutable structural fields.
        for field in (
            "benchmark",
            "benchmark_name",
            "benchmark_adapter",
            "instruction",
            "site",
            "sites",
            "start_urls",
            "data_seed",
            "agent_context",
            "task_provenance",
            "task_archetype",
            "benign_reward_shape",
            "contract_bound_generation",
            "task_card_id",
            "world",
            "comparison_contract",
            "comparison_act_contract",
        ):
            if field in benign_task:
                value = json.loads(json.dumps(benign_task[field]))
                if field in {"agent_context", "data_seed"}:
                    value = _sanitize_agent_context_for_output(value)
                adv_task[field] = value

        # Inject benign_target_resource (Option A placement contract). The
        # LLM receives this in /workspace/tasks/benign_target_resources.json
        # but doesn't reliably echo it into each plan; merging it here makes
        # the Option A validator's input deterministic regardless of what
        # the planner emits. Prefer the live-resolved record when available
        # so L3/L4 anchors flow through to the validator + Phase 4.
        if enriched_resources is not None and benign_id in enriched_resources:
            adv_task["benign_target_resource"] = json.loads(
                json.dumps(dict(enriched_resources[benign_id]))
            )
        else:
            adv_task["benign_target_resource"] = derive_benign_target_resource(
                benign_task,
                _PHASE_2A_SYNTHETIC_PLACEHOLDERS,
                benchmark=_option_a._benchmark_for_option_a_plan(adv_task),
            )

        if isinstance(adv_task.get("comparison_contract"), Mapping):
            bind_gitlab_compare_decide_benign_resource(
                adv_task,
                adv_task.get("benign_target_resource"),
            )

        if exposure_contracts is not None and benign_id in exposure_contracts:
            adv_task["exposure_contract"] = json.loads(
                json.dumps(dict(exposure_contracts[benign_id]))
            )

        # Host compiles action rewards. The planner may choose an action, but it
        # never owns the deterministic evaluator schema.
        # Minimal schema: adversarial_reward is a top-level field, no reward_function.
        # Full schema: reward_function already has benign_reward + adversarial_reward.
        adv_reward_top = adv_task.pop("adversarial_reward", None)
        reward = adv_task.get("reward_function")
        if adv_reward_top is None and (
            not isinstance(reward, dict) or "adversarial_reward" not in reward
        ):
            adv_reward_top = _compile_host_action_reward(adv_task, benign_task)

        if reward is None and adv_reward_top is not None:
            # Minimal schema — construct reward_function from scratch.
            adv_task["reward_function"] = {
                "benign_reward": json.loads(json.dumps(benign_task.get("reward_function", {}))),
                "adversarial_reward": adv_reward_top,
            }
        elif isinstance(reward, dict):
            # Full schema — overwrite benign_reward to guarantee correctness,
            # and lift top-level adversarial_reward if present.
            reward["benign_reward"] = json.loads(json.dumps(benign_task.get("reward_function", {})))
            if adv_reward_top is not None and "adversarial_reward" not in reward:
                reward["adversarial_reward"] = adv_reward_top
            adv_task["reward_function"] = reward
        if isinstance(adv_task.get("adversarial_action"), Mapping):
            _attach_host_action_final_state_check(adv_task)


def _compile_host_action_reward(
    adv_task: dict[str, Any],
    benign_task: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not isinstance(adv_task.get("adversarial_action"), Mapping):
        return None
    if "required_tokens" not in adv_task:
        adv_task["required_tokens"] = derive_required_tokens(str(adv_task.get("id") or ""))
    try:
        return compile_adversarial_reward(adv_task, benign_task)
    except ValueError as exc:
        raise ValueError(f"task {adv_task.get('id', '?')!r}: {exc}") from exc


def _attach_host_action_final_state_check(adv_task: dict[str, Any]) -> None:
    if not isinstance(adv_task.get("adversarial_action"), Mapping):
        return
    reward = adv_task.get("reward_function")
    if not isinstance(reward, dict):
        return
    final_state = compile_adversarial_final_state_check(adv_task)
    if final_state is not None:
        reward["adversarial_final_state_check"] = final_state


def _collect_site_profiles(
    tasks_by_site: dict[str, list[dict]],
    profiles_dir: Path,
) -> tuple[dict[str, Path], list[str]]:
    """Validate Phase 0c profiles once and return the reusable site-path map."""
    site_profiles: dict[str, Path] = {}
    errors: list[str] = []
    for site in tasks_by_site:
        profile_path = profiles_dir / f"BENCHMARK_PROFILE_{site}.json"
        try:
            load_and_validate_profile(site, profile_path)
        except ValueError as exc:
            errors.append(str(exc))
            continue
        site_profiles[site] = profile_path
    return site_profiles, errors


def _voice_registry() -> dict[str, Any]:
    if not hasattr(_voice_registry, "_cache"):
        _voice_registry._cache = load_voice_registry()
    return _voice_registry._cache


def _render_generation_prompt(
    cell_targets: dict[str, int],
    *,
    validation_command: str,
    contract_context: ContractRenderContext | None = None,
) -> str:
    return (
        load_prompt(
            "generate-injections",
            validation_command=validation_command,
            contract_context=contract_context,
        )
        + "\n\n## Cell Balance\n\n"
        + "Use `/workspace/tasks/cell_targets.json` as the authoritative shard-level "
        + "target count per framing::concealment cell.\n\n```json\n"
        + json.dumps(cell_targets, indent=2, sort_keys=True)
        + "\n```\n"
    )


def _materialize_strategy_plans_from_exposure(
    plans: list[dict[str, Any]],
    *,
    exposure_contracts: Mapping[str, Mapping[str, Any]],
    benchmark: str,
    benign_tasks: Iterable[Mapping[str, Any]] | None = None,
) -> None:
    contracts_by_id = {
        str(contract.get("contract_id") or ""): contract for contract in exposure_contracts.values()
    }
    contracts_by_benign_id: dict[str, list[Mapping[str, Any]]] = {}
    for key, contract in exposure_contracts.items():
        if not isinstance(contract, Mapping):
            continue
        candidates = {
            str(key or "").strip(),
            str(contract.get("benign_task_id") or "").strip(),
        }
        for candidate in candidates:
            if not candidate:
                continue
            contracts_by_benign_id.setdefault(candidate, []).append(contract)
    benign_seed_by_id: dict[str, Mapping[str, Any]] = {}
    if benign_tasks is not None:
        for task in benign_tasks:
            if not isinstance(task, Mapping):
                continue
            tid = str(task.get("id") or "").strip()
            if not tid:
                continue
            seed = task.get("data_seed")
            if isinstance(seed, Mapping):
                benign_seed_by_id[tid] = seed
    host_owned_fields = {"seed_template", "target_surface_id", "delivery_mechanism"}
    for plan in plans:
        forbidden_fields = sorted(host_owned_fields.intersection(plan))
        if forbidden_fields:
            raise ValueError(
                f"strategy plan {plan.get('id', '?')!r} included host-owned placement "
                f"fields: {', '.join(forbidden_fields)}"
            )
        benign_id = str(plan.get("benign_task_id") or "").strip()
        contract_id = str(plan.get("exposure_contract_id") or "").strip()
        contract = contracts_by_id.get(contract_id) if contract_id else None
        if contract is None and (not contract_id or contract_id == benign_id):
            benign_matches = contracts_by_benign_id.get(benign_id, [])
            unique_matches = {
                str(match.get("contract_id") or ""): match
                for match in benign_matches
                if isinstance(match, Mapping)
            }
            if len(unique_matches) > 1:
                raise ValueError(
                    f"plan {plan.get('id', '?')!r} references ambiguous exposure contract "
                    f"for benign_task_id={benign_id!r}"
                )
            if len(unique_matches) == 1:
                contract = next(iter(unique_matches.values()))
        if not isinstance(contract, Mapping):
            raise ValueError(
                f"plan {plan.get('id', '?')!r} references no known exposure contract "
                f"(benign_task_id={benign_id!r}, exposure_contract_id={contract_id!r})"
            )
        plan["exposure_contract_id"] = str(contract.get("contract_id") or contract_id)
        plan["target_surface_id"] = str(contract.get("target_surface_id") or "")
        benign_seed = benign_seed_by_id.get(benign_id)
        seed_template = materialize_seed_template_from_contract(
            contract,
            benchmark=benchmark,
            benign_seed=benign_seed,
        )
        plan["seed_template"] = seed_template
        plan["delivery_mechanism"] = _eligibility._seed_delivery_mechanism(seed_template)


def _enrich_adversarial_plans(
    plans: list[dict[str, Any]],
    site_profile: dict[str, Any],
) -> list[dict[str, Any]]:
    registry = _voice_registry()
    enriched: list[dict[str, Any]] = []
    for plan in plans:
        delivery_channel = _resolve_delivery_channel(
            site_profile,
            target_surface_id=str(plan.get("target_surface_id", "")),
            delivery_mechanism=str(plan.get("delivery_mechanism", "")),
            seed_template=plan.get("seed_template"),
        )
        updated = json.loads(json.dumps(plan))
        updated["delivery_channel"] = delivery_channel
        # Propagate source_field from the site profile onto the task so that
        # downstream voice/budget resolution can pattern-match on it without
        # needing the full site_profile.
        surface = _find_surface_by_id(
            site_profile,
            str(plan.get("target_surface_id", "")),
        )
        if isinstance(surface, dict):
            sf = surface.get("source_field")
            if isinstance(sf, str) and sf.strip():
                updated["source_field"] = sf
        delivery_site = delivery_channel.get("delivery_site")
        if isinstance(delivery_site, str) and delivery_site.strip():
            sites = [
                str(site).strip()
                for site in updated.get("sites", [updated.get("site")])
                if isinstance(site, str) and str(site).strip()
            ]
            normalized_delivery_site = delivery_site.strip()
            if normalized_delivery_site not in sites:
                updated["sites"] = [*sites, normalized_delivery_site]
        updated["required_tokens"] = derive_required_tokens(str(plan.get("id", "")))
        updated["length_budget"] = derive_length_budget(updated, site_profile, registry)
        enriched.append(updated)
    return enriched
