"""Phase 3: contract validity gate.

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Phase 3".

Validates each benign task's contract (reward function, start URLs, data seed)
and each adversarial task's reference to a benign task. Writes
``phase_3/contracts.json`` with one entry per benign task, stamped with origin
(``mode_a`` for vendored benchmark tasks, ``mode_b`` for Phase 1 generations)
and a ``validity_status`` Phase 4 uses for admission.

No agent runs happen here. Capability measurement comes from Phase 4's
benign-under-attack outcome, not from this phase.
"""

from __future__ import annotations

import argparse
import json
import logging
from typing import Any

from worldsim.atomic_io import write_json_atomic
from worldsim.seeding import validate_data_seed
from worldsim.state import get_state_dir, save_state

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


def _classify_origin(task: dict[str, Any]) -> str:
    # Mode B ids are enforced as `novel_<site>_<n>` at generation
    # (phase_1_mode_b_validation). Seed shape is not a reliable signal: a Mode B
    # navigate-only task legitimately carries `mechanism: "none"`.
    task_id = str(task.get("id", "")).strip()
    if task_id.startswith("novel_"):
        return "mode_b"
    seed = task.get("data_seed") or {}
    mechanism = seed.get("mechanism") if isinstance(seed, dict) else None
    editor_calls = seed.get("editor_calls") if isinstance(seed, dict) else None
    has_editor_calls = isinstance(editor_calls, list) and bool(editor_calls)
    if mechanism not in (None, "none") or has_editor_calls:
        return "mode_b"
    return "mode_a"


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
    return errors


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
    except ValueError as exc:
        logger.error("%s", exc)
        return 1

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

    if sites_filter_raw and adversarial_tasks:
        benign_site_set = {str(task.get("site", "")).strip() for task in benign_tasks}
        adversarial_tasks = [
            adv for adv in adversarial_tasks if str(adv.get("site", "")).strip() in benign_site_set
        ]

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
            "Phase 3: benign task ids are not unique across Mode A and Mode B: %s",
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
    mode_a_valid = 0
    mode_b_valid = 0
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
            if origin == "mode_a":
                mode_a_valid += 1
            else:
                mode_b_valid += 1
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
        "mode_a_valid": mode_a_valid,
        "mode_b_valid": mode_b_valid,
        "adversarial_reference_errors": len(adversarial_errors),
    }
    if sites_filter_raw:
        state_payload["sites"] = sites_filter_raw
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
        "Phase 3 complete — %d/%d contracts valid (%d mode_a, %d mode_b); wrote %s",
        valid_count,
        len(contracts),
        mode_a_valid,
        mode_b_valid,
        output_dir / "contracts.json",
    )
    return 0
