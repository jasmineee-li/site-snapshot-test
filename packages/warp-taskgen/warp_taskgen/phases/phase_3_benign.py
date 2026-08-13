"""Phase 3: contract validity gate.

Canonical source: ``docs/warp-taskgen-technical-spec.md`` "Phase 3".

Validates each benign task's contract (reward function, start URLs, data seed)
and each adversarial task's reference to a benign task. Writes
``phase_3/contracts.json`` with one entry per benign task, stamped with origin
(``existing_task`` for vendored benchmark tasks, ``new_task`` for Phase 1 generations)
and a ``validity_status`` Phase 4 uses for admission.

No agent runs happen here. Capability measurement comes from Phase 4's
benign-under-attack outcome, not from this phase.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from warp_taskgen.atomic_io import write_json_atomic
from warp_taskgen.seeding import validate_data_seed
from warp_taskgen.state import get_state_dir, save_state

logger = logging.getLogger(__name__)


def _filter_tasks_by_sites(
    tasks: list[dict[str, Any]],
    sites_filter_raw: str | None,
    *,
    phase_label: str,
) -> list[dict[str, Any]]:
    if not sites_filter_raw:
        return tasks
    sites_filter = {site.strip() for site in sites_filter_raw.split(",") if site.strip()}
    known_sites = {str(task.get("site", "")).strip() for task in tasks if task.get("site")}
    unknown = sites_filter - known_sites
    if unknown:
        raise ValueError(
            f"{phase_label}: --sites includes unknown site(s): {sorted(unknown)}. "
            f"Known sites: {sorted(known_sites)}"
        )
    filtered = [task for task in tasks if str(task.get("site", "")).strip() in sites_filter]
    logger.info("%s: --sites filter active, running only %s", phase_label, sorted(sites_filter))
    return filtered


def _filter_tasks_by_origin(
    tasks: list[dict[str, Any]],
    task_origin: str | None,
    *,
    phase_label: str,
) -> list[dict[str, Any]]:
    if task_origin in (None, "", "all"):
        return tasks
    if task_origin not in ("existing_task", "new_task"):
        raise ValueError(
            f"{phase_label}: --task-origin must be one of all, existing_task, new_task; "
            f"got {task_origin!r}"
        )
    filtered = [task for task in tasks if _classify_origin(task) == task_origin]
    logger.info("%s: --task-origin filter active, running only %s", phase_label, task_origin)
    return filtered


def _classify_origin(task: dict[str, Any]) -> str:
    # Prefer the stamped origin field on the task (Phase 1 sets this at emit).
    # Fall back to id-prefix and seed-shape inference for legacy snapshots.
    stamped = task.get("origin")
    if isinstance(stamped, str) and stamped in ("existing_task", "new_task"):
        return stamped
    # new_task ids are enforced as `novel_<site>_<n>` at generation
    # (phase_1_generate_new_tasks_validation). Seed shape is not a reliable
    # signal: a new_task navigate-only task legitimately carries
    # `mechanism: "none"`.
    task_id = str(task.get("id", "")).strip()
    if task_id.startswith("novel_"):
        return "new_task"
    seed = task.get("data_seed") or {}
    mechanism = seed.get("mechanism") if isinstance(seed, dict) else None
    editor_calls = seed.get("editor_calls") if isinstance(seed, dict) else None
    has_editor_calls = isinstance(editor_calls, list) and bool(editor_calls)
    if mechanism not in (None, "none") or has_editor_calls:
        return "new_task"
    return "existing_task"


def _validate_benign_task(task: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    reward_function = task.get("reward_function")
    if not isinstance(reward_function, dict) or not reward_function:
        errors.append("reward_function must be a non-empty object")

    start_urls = task.get("start_urls")
    if not isinstance(start_urls, list) or not start_urls:
        errors.append("start_urls must be a non-empty list")
    else:
        for url in start_urls:
            if not isinstance(url, str) or not url.strip():
                errors.append(f"start_urls entries must be non-empty strings, got {url!r}")
                break

    try:
        validate_data_seed(task.get("data_seed") or {}, allow_none=True)
    except ValueError as exc:
        errors.append(f"data_seed invalid: {exc}")

    return errors


def _annotate_adversarially_exhausted(
    contracts: list[dict[str, Any]],
    adversarial_tasks: list[dict[str, Any]],
) -> None:
    """Tag benign contracts whose every linked adversarial is ``infeasible``.

    Annotation only: the contract stays ``valid``; Phase 4 reads
    ``adversarially_exhausted`` to decide whether to run baseline-only for
    these benigns (they have no usable adversarial this run).
    """
    linked: dict[str, list[str]] = {}
    for adv_task in adversarial_tasks:
        benign_task_id = str(adv_task.get("benign_task_id", "")).strip()
        if not benign_task_id:
            continue
        feasibility = adv_task.get("feasibility")
        status = feasibility.get("status") if isinstance(feasibility, dict) else None
        linked.setdefault(benign_task_id, []).append(str(status or "missing"))

    for contract in contracts:
        benign_id = str(contract.get("id", "")).strip()
        statuses = linked.get(benign_id)
        if not statuses:
            continue
        # "skipped" is admitted here as defense-in-depth: the spec-conformant
        # Phase 2c never emits ``status="skipped"`` (idempotent reuse preserves
        # ``status="verified"``), but a stale dataset written by an earlier
        # buggy build can still be on disk — treat those as exhausted so
        # Phase 4 doesn't get handed a benign whose only adversarial is a
        # Phase 4-inadmissible sentinel.
        if all(status in ("infeasible", "skipped") for status in statuses):
            contract["adversarially_exhausted"] = True


def _adversarial_task_errors(
    adv_task: dict[str, Any],
    benign_by_id: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    benign_task_id = str(adv_task.get("benign_task_id", "")).strip()
    if not benign_task_id:
        errors.append("adversarial task missing benign_task_id")
        return errors
    if benign_task_id not in benign_by_id:
        errors.append(f"adversarial task references unknown benign_task_id {benign_task_id!r}")
        return errors

    reward = adv_task.get("reward_function")
    if not isinstance(reward, dict):
        errors.append("reward_function must be an object")
        return errors
    adversarial_reward = reward.get("adversarial_reward")
    if not isinstance(adversarial_reward, dict) or not adversarial_reward:
        errors.append("reward_function.adversarial_reward must be a non-empty object")
    elif adversarial_reward.get("type") == "db_query_match":
        errors.append(
            "reward_function.adversarial_reward uses removed legacy evaluator "
            "'db_query_match'; regenerate with NetworkEventEvaluator, "
            "FinalStateEvaluator, or AgentResponseEvaluator"
        )
    return errors


def _filter_adversarial_tasks_by_selected_benigns(
    adversarial_tasks: list[dict[str, Any]],
    selected_benign_ids: set[str],
) -> list[dict[str, Any]]:
    if not selected_benign_ids:
        return []
    return [
        adv_task
        for adv_task in adversarial_tasks
        if str(adv_task.get("benign_task_id", "")).strip() in selected_benign_ids
    ]


async def run(args: argparse.Namespace) -> int:
    state_dir = get_state_dir()
    tasks_path = state_dir / "phase_1" / "benign_tasks.json"
    if not tasks_path.exists():
        logger.error("Benign tasks not found at %s — run phase 1 first", tasks_path)
        return 1
    benign_tasks = json.loads(tasks_path.read_text())

    sites_filter_raw = getattr(args, "sites", None)
    try:
        benign_tasks = _filter_tasks_by_sites(benign_tasks, sites_filter_raw, phase_label="Phase 3")
        task_origin = getattr(args, "task_origin", None) or "all"
        benign_tasks = _filter_tasks_by_origin(
            benign_tasks,
            task_origin,
            phase_label="Phase 3",
        )
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

    state_payload: dict[str, Any] = {
        "status": "running",
        "tasks_path": str(tasks_path),
    }
    if sites_filter_raw:
        state_payload["sites"] = sites_filter_raw
    if task_origin != "all":
        state_payload["task_origin"] = task_origin
    save_state("phase_3", **state_payload)

    adv_tasks_path = state_dir / "phase_2" / "adversarial_tasks.json"
    adversarial_tasks: list[dict[str, Any]] = []
    if adv_tasks_path.exists():
        raw = json.loads(adv_tasks_path.read_text())
        if isinstance(raw, list):
            adversarial_tasks = raw
        else:
            logger.warning("Phase 3: adversarial_tasks.json is not an array, ignoring")

    # Also include Phase 2c's quarantined infeasible tasks so the
    # ``adversarially_exhausted`` annotation can fire for benigns whose only
    # linked adversarials all landed in quarantine. These tasks will not
    # reach Phase 4, but the annotation lets reviewers distinguish "no
    # usable adversarial for this benign" from "agent resisted".
    infeasible_path = adv_tasks_path.with_name(adv_tasks_path.stem + ".infeasible.json")
    if infeasible_path.exists():
        try:
            infeasible_raw = json.loads(infeasible_path.read_text())
        except json.JSONDecodeError:
            infeasible_raw = []
            logger.warning("Phase 3: %s is not valid JSON, ignoring", infeasible_path)
        if isinstance(infeasible_raw, list):
            adversarial_tasks.extend(infeasible_raw)

    # Bug I (2026-04-23): ingest Phase 2c's source-data quarantine so
    # benigns whose only adversarial partners were preflight-dropped
    # (login_redirect / 404 stale-L4 / etc.) still get flagged
    # ``adversarially_exhausted``. Otherwise those benigns look as if no
    # adversarial was ever generated for them. Stamp a synthetic
    # feasibility wrapper so ``_annotate_adversarially_exhausted``'s
    # "infeasible" check catches them uniformly.
    dropped_source_path = adv_tasks_path.with_name(
        adv_tasks_path.stem + ".dropped_source_data.json"
    )
    if dropped_source_path.exists():
        try:
            dropped_raw = json.loads(dropped_source_path.read_text())
        except json.JSONDecodeError:
            dropped_raw = []
            logger.warning("Phase 3: %s is not valid JSON, ignoring", dropped_source_path)
        if isinstance(dropped_raw, list):
            for record in dropped_raw:
                if not isinstance(record, dict):
                    continue
                synthetic = dict(record)
                prior_feasibility = synthetic.get("feasibility")
                if isinstance(prior_feasibility, dict) and prior_feasibility:
                    synthetic["prior_feasibility"] = prior_feasibility
                source_data_issue = synthetic.get("source_data_issue")
                synthetic["feasibility"] = {
                    "status": "infeasible",
                    "kind": "source_data_issue",
                }
                if isinstance(source_data_issue, dict):
                    synthetic["feasibility"]["source_data_issue"] = source_data_issue
                adversarial_tasks.append(synthetic)

    filter_active = bool(sites_filter_raw) or task_origin != "all"
    if filter_active and adversarial_tasks:
        selected_benign_ids = {
            str(task.get("id", "")).strip() for task in benign_tasks if task.get("id")
        }
        adversarial_tasks = _filter_adversarial_tasks_by_selected_benigns(
            adversarial_tasks,
            selected_benign_ids,
        )

    benign_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for task in benign_tasks:
        task_id = str(task.get("id", "")).strip()
        if not task_id:
            continue
        if task_id in benign_by_id:
            duplicate_ids.append(task_id)
            continue
        benign_by_id[task_id] = task
    if duplicate_ids:
        logger.error(
            "Phase 3: benign task ids are not unique across existing_task and new_task: %s",
            ", ".join(sorted(set(duplicate_ids))),
        )
        save_state(
            "phase_3",
            status="failed",
            reason="duplicate_benign_ids",
            duplicate_benign_ids=sorted(set(duplicate_ids)),
        )
        return 1

    contracts: list[dict[str, Any]] = []
    valid_count = 0
    existing_task_valid = 0
    new_task_valid = 0
    for task in benign_tasks:
        task_id = str(task.get("id", "")).strip()
        if not task_id:
            logger.warning("Skipping benign task without id")
            continue
        origin = _classify_origin(task)
        errors = _validate_benign_task(task)
        validity_status = "valid" if not errors else "invalid"
        if validity_status == "valid":
            valid_count += 1
            if origin == "existing_task":
                existing_task_valid += 1
            else:
                new_task_valid += 1
        contracts.append(
            {
                "id": task_id,
                "origin": origin,
                "validity_status": validity_status,
                "validity_errors": errors,
                "task": task,
            }
        )

    adversarial_errors: list[dict[str, Any]] = []
    for adv_task in adversarial_tasks:
        errors = _adversarial_task_errors(adv_task, benign_by_id)
        if errors:
            adversarial_errors.append(
                {
                    "id": adv_task.get("id"),
                    "benign_task_id": adv_task.get("benign_task_id"),
                    "errors": errors,
                }
            )

    _annotate_adversarially_exhausted(contracts, adversarial_tasks)

    output_dir = state_dir / "phase_3"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        output_dir / "contracts.json",
        contracts,
        failpoint_base="phase_3.outputs.contracts",
    )

    state_payload: dict[str, Any] = {
        "status": "complete",
        "contracts_path": str(output_dir / "contracts.json"),
        "total": len(contracts),
        "valid": valid_count,
        "invalid": len(contracts) - valid_count,
        "existing_task_valid": existing_task_valid,
        "new_task_valid": new_task_valid,
        "adversarial_reference_errors": len(adversarial_errors),
    }
    if sites_filter_raw:
        state_payload["sites"] = sites_filter_raw
    if task_origin != "all":
        state_payload["task_origin"] = task_origin
    save_state("phase_3", **state_payload)

    if adversarial_errors:
        # Soft-warn per spec (Phase 3 is the benign validity gate); reference
        # errors indicate a Phase 2 bug, so surface at ERROR for CI greppers.
        logger.error(
            "Phase 3: %d adversarial task(s) reference invalid or unknown benign contracts:\n%s",
            len(adversarial_errors),
            "\n".join(
                f"  - adv {entry['id']}: {', '.join(entry['errors'])}"
                for entry in adversarial_errors
            ),
        )

    logger.info(
        "Phase 3 complete — %d/%d contracts valid (%d existing_task, %d new_task); wrote %s",
        valid_count,
        len(contracts),
        existing_task_valid,
        new_task_valid,
        output_dir / "contracts.json",
    )
    return 0
