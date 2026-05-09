"""Hugging Face dataset export for Phase 4 trajectory cohorts.

This module is report-only. It reads completed Phase 4 results and writes
queryable JSONL splits plus optional safe trajectory bundles. It does not change
Phase 4 scoring, PVPO, admission, rewards, or iterator behavior.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from worldsim.phase_4 import result_summary
from worldsim.phase_4.artifact_audit import (
    default_task_paths,
    load_json,
    load_task_lookup,
    resolve_phase4_results_path,
)
from worldsim.phase_4.variant_accounting import semantic_variant_accounting

DATASET_SCHEMA_VERSION = "warp_taskgen_phase4_hf_dataset_v1"
DEFAULT_DATASET_ID = "warp-taskgen-generated-ipi-tasks-50"
DEFAULT_DATASET_PRETTY_NAME = "WARP Taskgen Generated IPI Tasks 50"
DEFAULT_HF_REPO_OWNER = "ash256"

PUBLIC_TRAJECTORY_FILES = {
    "result.json",
    "processed_result.json",
    "history.json",
    "final_response.json",
    "needham_trace.json",
    "needham_trace.xml",
    "browser_runtime.json",
    "network_trace.json",
    "network.har",
    "capture_summary.json",
}
PUBLIC_TRAJECTORY_DIRS = {"screenshots", "pvpo"}
PRIVATE_TRAJECTORY_PARTS = {
    "reward_private",
    "storage_state",
    "storage_states",
    "auth",
    "secrets",
    "prompt_input.json",
    "redacted_prompt_input.json",
}


@dataclass(frozen=True)
class RunSpec:
    model_key: str
    run_dir: Path


def export_hf_dataset(
    *,
    runs: list[RunSpec],
    output_dir: Path,
    root: Path | None = None,
    dataset_id: str = DEFAULT_DATASET_ID,
    bundle_artifacts: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Export Phase 4 runs into a local Hugging Face-compatible dataset folder."""

    if not runs:
        raise ValueError("at least one run is required")
    root = (root or Path.cwd()).resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{output_dir} is not empty; pass --overwrite to replace files")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "artifacts").mkdir(exist_ok=True)

    run_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []

    for spec in runs:
        run_root = _resolve_run_dir(spec.run_dir, root=root)
        results_path = resolve_phase4_results_path(run_root)
        raw_results = load_json(results_path)
        if not isinstance(raw_results, list):
            raise ValueError(f"{results_path} must contain a list of result objects")
        results = [item for item in raw_results if isinstance(item, dict)]
        task_lookup = load_task_lookup(default_task_paths(results_path))
        summary = result_summary.summarize_results(results, task_lookup=task_lookup)
        pipeline_state = _load_json_or_empty(run_root / "pipeline_state.json")
        progress = _load_json_or_empty(run_root / "phase_4" / "progress.json")
        model_key = spec.model_key
        run_id = _run_id(run_root)

        run_rows.append(
            _run_row(
                model_key=model_key,
                run_id=run_id,
                run_root=run_root,
                root=root,
                summary=summary,
                pipeline_state=pipeline_state,
                progress=progress,
            )
        )
        for result_index, result in enumerate(results):
            row, row_warnings = _task_row(
                model_key=model_key,
                run_id=run_id,
                run_root=run_root,
                root=root,
                result_index=result_index,
                result=result,
                task=task_lookup.get(str(result.get("task_id") or ""), {}),
                output_dir=output_dir,
                bundle_artifacts=bundle_artifacts,
            )
            task_rows.append(row)
            warnings.extend(row_warnings)
            variant_rows.extend(
                _variant_rows(
                    model_key=model_key,
                    run_id=run_id,
                    task_row=row,
                    result=result,
                    root=root,
                )
            )

    _write_jsonl(output_dir / "runs.jsonl", run_rows)
    _write_jsonl(output_dir / "tasks.jsonl", task_rows)
    _write_jsonl(output_dir / "variants.jsonl", variant_rows)
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_id": dataset_id,
        "created_at": datetime.now(UTC).isoformat(),
        "root": str(root),
        "splits": {
            "runs": {"path": "runs.jsonl", "rows": len(run_rows)},
            "tasks": {"path": "tasks.jsonl", "rows": len(task_rows)},
            "variants": {"path": "variants.jsonl", "rows": len(variant_rows)},
        },
        "runs": [
            {
                "model_key": spec.model_key,
                "run_dir": _relpath(_resolve_run_dir(spec.run_dir, root=root), root),
            }
            for spec in runs
        ],
        "warnings": warnings,
    }
    _write_json(output_dir / "metadata.json", manifest)
    _write_dataset_card(output_dir / "README.md", manifest=manifest)
    return manifest


def _run_row(
    *,
    model_key: str,
    run_id: str,
    run_root: Path,
    root: Path,
    summary: dict[str, Any],
    pipeline_state: dict[str, Any],
    progress: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "model_key": model_key,
        "run_id": run_id,
        "run_dir": _relpath(run_root, root),
        "runner": pipeline_state.get("runner") or _maybe(progress, "runner") or "agentlab",
        "agent_model": pipeline_state.get("agent_model") or model_key,
        "agent_provider": pipeline_state.get("agent_provider"),
        "agent_service_tier": pipeline_state.get("agent_service_tier"),
        "phase_4_variant_system": pipeline_state.get("phase_4_variant_system")
        or progress.get("variant_progress", {}).get("variant_system"),
        "phase_4_eval_awareness_max_iterations": pipeline_state.get(
            "phase_4_eval_awareness_max_iterations"
        )
        or progress.get("variant_progress", {}).get("eval_awareness_max_iterations"),
        "phase_4_max_workers": pipeline_state.get("phase_4_max_workers")
        or progress.get("phase_4_max_workers"),
        "total": summary.get("total"),
        "scorable": summary.get("scorable"),
        "gate1_valid": summary.get("gate1_valid"),
        "headline_asr": summary.get("headline_asr"),
        "headline_asr_numerator": summary.get("headline_asr_numerator"),
        "headline_asr_denominator": summary.get("headline_asr_denominator"),
        "gate1_asr": summary.get("gate1_asr"),
        "gate1_asr_numerator": summary.get("gate1_asr_numerator"),
        "gate1_asr_denominator": summary.get("gate1_asr_denominator"),
        "benign_capability": summary.get("benign_capability"),
        "final_status_counts": summary.get("final_status_counts", {}),
        "outcome_fine_counts": summary.get("outcome_fine_counts", {}),
        "pvpo_observation_counts": summary.get("pvpo_observation_counts", {}),
        "variant_regeneration_audit": summary.get("variant_regeneration_audit", {}),
    }


def _task_row(
    *,
    model_key: str,
    run_id: str,
    run_root: Path,
    root: Path,
    result_index: int,
    result: dict[str, Any],
    task: dict[str, Any],
    output_dir: Path,
    bundle_artifacts: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    task_id = str(result.get("task_id") or f"row_{result_index}")
    variation = _variation_record(result)
    variant_results = _variant_results(variation)
    generation_errors = _generation_errors(variation)
    accounting = semantic_variant_accounting(
        variant_results=variant_results,
        generation_errors=generation_errors,
    )
    selected_variant = _selected_variant(result)
    selected_trace = _trace_from_variant(selected_variant)
    trace = _trace_path(result)
    artifact_bundle = None
    artifact_manifest: dict[str, Any] | None = None
    warnings: list[dict[str, Any]] = []
    if bundle_artifacts:
        artifact_bundle, artifact_manifest = _write_task_bundle(
            output_dir=output_dir,
            root=root,
            model_key=model_key,
            task_id=task_id,
            trace_values=[trace, selected_trace],
        )
        if artifact_manifest.get("warning"):
            warnings.append(
                {
                    "model_key": model_key,
                    "task_id": task_id,
                    "warning": artifact_manifest["warning"],
                }
            )
    encounter = _dict(result.get("encounter"))
    return (
        {
            "schema_version": DATASET_SCHEMA_VERSION,
            "model_key": model_key,
            "run_id": run_id,
            "result_index": result_index,
            "task_id": task_id,
            "site": result_summary.task_site(result, task),
            "origin": result_summary.task_origin(result, task),
            "scenario_template": result_summary.task_scenario_template(task),
            "surface": result_summary.task_surface(task),
            "editor_method": result_summary.task_editor_method(task),
            "route_variant": result_summary.task_route_variant(task),
            "adversarial_action_kind": result.get("adversarial_action_kind"),
            "adversarial_action_family": result.get("adversarial_action_family"),
            "adversarial_action_tier": result.get("adversarial_action_tier"),
            "final_status": result.get("final_status"),
            "outcome": result.get("outcome"),
            "outcome_fine": result.get("outcome_fine"),
            "is_complied": result_summary.final_result_is_compliant(result),
            "is_resistant": result.get("final_status") == "resistant",
            "is_inconclusive": result.get("final_status") == "inconclusive",
            "is_injection_not_encountered": result.get("final_status")
            == "injection_not_encountered",
            "is_task_broke": result.get("final_status") == "task_broke",
            "is_error": result.get("final_status") == "error",
            "headline_asr_denominator_included": result.get("final_status")
            not in result_summary.NON_SCORABLE_FINAL_STATUSES,
            "gate1_denominator_included": result_summary.ecologically_valid(result),
            "benign_capability_success": result_summary.final_metric_success(
                result,
                "benign_success_under_attack",
            ),
            "benign_passed": result.get("benign_passed"),
            "adversarial_passed": result.get("adversarial_passed"),
            "attack_action_attempted": result_summary.final_metric_success(
                result,
                "attack_action_attempted",
            ),
            "attack_action_success": result_summary.final_metric_success(
                result,
                "attack_action_success",
            ),
            "state_confirmed_action_success": result_summary.final_metric_success(
                result,
                "state_confirmed_action_success",
            ),
            "signals": _dict(result.get("signals")),
            "flags": result.get("flags") if isinstance(result.get("flags"), list) else [],
            "diagnosable_confidence": result.get("diagnosable_confidence"),
            "pvpo_status": result.get("pvpo_status") or encounter.get("pvpo_status"),
            "pvpo_failure": result.get("pvpo_failure") or encounter.get("pvpo_failure"),
            "pvpo_max_coverage": _float_or_none(encounter.get("max_coverage")),
            "pvpo_observation": encounter.get("pvpo_observation"),
            "pvpo_reference_step": encounter.get("reference_step"),
            "baseline_tp_label": result.get("transcript_purpose_classification"),
            "baseline_tp_explanation": result.get("transcript_purpose_explanation"),
            "baseline_tp_is_aware": result.get("transcript_purpose_is_aware"),
            "baseline_vea_label": result.get("verbalized_eval_awareness_classification"),
            "baseline_vea_is_aware": result.get("verbalized_eval_awareness_is_aware"),
            "baseline_vea_evidence": result.get("verbalized_eval_awareness_evidence")
            if isinstance(result.get("verbalized_eval_awareness_evidence"), list)
            else [],
            "selected_variant_id": _variant_id(selected_variant),
            "selected_variant_iteration": selected_variant.get("iteration")
            if isinstance(selected_variant, dict)
            else None,
            "selected_tp_label": _value_from_selected_or_result(
                selected_variant,
                result,
                "transcript_purpose_classification",
            ),
            "selected_vea_label": _value_from_selected_or_result(
                selected_variant,
                result,
                "verbalized_eval_awareness_classification",
            ),
            "tp_transition": _transition(
                result.get("transcript_purpose_classification"),
                _value_from_selected_or_result(
                    selected_variant,
                    result,
                    "transcript_purpose_classification",
                ),
            ),
            "vea_transition": _transition(
                result.get("verbalized_eval_awareness_classification"),
                _value_from_selected_or_result(
                    selected_variant,
                    result,
                    "verbalized_eval_awareness_classification",
                ),
            ),
            "iterator_entered": bool(variation and variation.get("status") != "skipped"),
            "iterator_status": variation.get("status") if variation else None,
            "iterator_stop_reason": variation.get("stop_reason") if variation else None,
            "iterations_planned": variation.get("budget") if variation else None,
            "variants_generated": len(variant_results),
            "variants_evaluated": len(variant_results),
            "pvpo_valid_variants": sum(
                1 for variant in variant_results if result_summary.ecologically_valid(variant)
            ),
            "complied_variants": sum(
                1
                for variant in variant_results
                if result_summary.ecologically_valid(variant)
                and variant.get("outcome") == "complied"
            ),
            **accounting,
            "artifact_bundle_path": artifact_bundle,
            "artifact_manifest": artifact_manifest or {},
            "trajectory_dir": _portable_path(trace, root),
            "selected_variant_trace": _portable_path(selected_trace, root),
        },
        warnings,
    )


def _variant_rows(
    *,
    model_key: str,
    run_id: str,
    task_row: dict[str, Any],
    result: dict[str, Any],
    root: Path,
) -> list[dict[str, Any]]:
    variation = _variation_record(result)
    if variation is None:
        return []
    selected_id = task_row.get("selected_variant_id")
    rows: list[dict[str, Any]] = []
    for index, variant in enumerate(_variant_results(variation)):
        vid = _variant_id(variant) or str(index)
        encounter = _dict(variant.get("encounter"))
        rows.append(
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "model_key": model_key,
                "run_id": run_id,
                "task_id": task_row["task_id"],
                "variant_id": vid,
                "variant_index": index,
                "global_variant_index": variant.get("global_variant_index"),
                "round_index": variant.get("round_index"),
                "iteration": variant.get("iteration"),
                "strategy": variant.get("strategy"),
                "status": "evaluated",
                "is_selected": vid == selected_id,
                "outcome": variant.get("outcome"),
                "outcome_fine": variant.get("outcome_fine"),
                "ecologically_valid": result_summary.ecologically_valid(variant),
                "adversarial_passed": variant.get("adversarial_passed"),
                "benign_passed": variant.get("benign_passed"),
                "attack_action_attempted": variant.get("attack_action_attempted"),
                "attack_action_success": variant.get("attack_action_success"),
                "pvpo_status": variant.get("pvpo_status") or encounter.get("pvpo_status"),
                "pvpo_failure": variant.get("pvpo_failure") or encounter.get("pvpo_failure"),
                "pvpo_max_coverage": _float_or_none(encounter.get("max_coverage")),
                "tp_label": variant.get("transcript_purpose_classification"),
                "vea_label": variant.get("verbalized_eval_awareness_classification"),
                "tp_transition": _transition(
                    result.get("transcript_purpose_classification"),
                    variant.get("transcript_purpose_classification"),
                ),
                "trajectory_dir": _portable_path(_trace_from_variant(variant), root),
            }
        )
    for index, error in enumerate(_generation_errors(variation)):
        error = error if isinstance(error, dict) else {"reason": str(error)}
        rows.append(
            {
                "schema_version": DATASET_SCHEMA_VERSION,
                "model_key": model_key,
                "run_id": run_id,
                "task_id": task_row["task_id"],
                "variant_id": str(error.get("global_variant_index") or error.get("index") or index),
                "variant_index": error.get("index", index),
                "global_variant_index": error.get("global_variant_index"),
                "round_index": error.get("round_index"),
                "iteration": error.get("iteration"),
                "strategy": error.get("strategy"),
                "status": "rejected",
                "failure_class": error.get("failure_class") or error.get("status"),
                "reason": error.get("reason") or error.get("error"),
                "is_post_eval_rejection": error.get("failure_class")
                in {"tp_regression", "task_broke", "lost_pvpo_encounter"},
            }
        )
    return rows


def _write_task_bundle(
    *,
    output_dir: Path,
    root: Path,
    model_key: str,
    task_id: str,
    trace_values: Iterable[str | None],
) -> tuple[str, dict[str, Any]]:
    bundle_rel = Path("artifacts") / model_key / f"{_safe_name(task_id)}.tar.gz"
    bundle_path = output_dir / bundle_rel
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    warnings: list[str] = []
    for trace_value in trace_values:
        trace = _resolve_existing_path(trace_value, root=root)
        if trace is None:
            continue
        for item in _iter_public_trajectory_files(trace):
            if item not in files:
                files.append(item)
    if not files:
        manifest = {"files": [], "sha256": None, "warning": "no_public_trajectory_files_found"}
        return bundle_rel.as_posix(), manifest
    with tarfile.open(bundle_path, "w:gz") as tar:
        for file_path in sorted(files):
            try:
                arcname = file_path.relative_to(root).as_posix()
            except ValueError:
                arcname = f"external/{_safe_name(task_id)}/{file_path.name}"
            tar.add(file_path, arcname=arcname)
    digest = _sha256(bundle_path)
    manifest = {
        "files": [_portable_path(path, root) for path in sorted(files)],
        "file_count": len(files),
        "sha256": digest,
        "size_bytes": bundle_path.stat().st_size,
    }
    if warnings:
        manifest["warnings"] = warnings
    return bundle_rel.as_posix(), manifest


def _iter_public_trajectory_files(trace: Path) -> Iterable[Path]:
    if not trace.exists() or not trace.is_dir():
        return []
    files: list[Path] = []
    for item in trace.rglob("*"):
        if not item.is_file() or item.is_symlink():
            continue
        rel = item.relative_to(trace)
        if any(part in PRIVATE_TRAJECTORY_PARTS for part in rel.parts):
            continue
        if item.name in PUBLIC_TRAJECTORY_FILES or (
            bool(rel.parts) and rel.parts[0] in PUBLIC_TRAJECTORY_DIRS
        ):
            files.append(item)
    return files


def _resolve_run_dir(path: Path, *, root: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def _resolve_existing_path(value: str | None, *, root: Path) -> Path | None:
    if not value:
        return None
    path = Path(value)
    candidates = [path] if path.is_absolute() else [root / path, path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _write_dataset_card(path: Path, *, manifest: dict[str, Any]) -> None:
    split_rows = manifest.get("splits", {})
    runs_rows = (
        _dict(split_rows.get("runs") if isinstance(split_rows, dict) else {}).get("rows")
        or "unknown"
    )
    tasks_rows = (
        _dict(split_rows.get("tasks") if isinstance(split_rows, dict) else {}).get("rows")
        or "unknown"
    )
    variants_rows = (
        _dict(split_rows.get("variants") if isinstance(split_rows, dict) else {}).get("rows")
        or "unknown"
    )
    model_keys = [
        str(run.get("model_key"))
        for run in manifest.get("runs", [])
        if isinstance(run, dict) and run.get("model_key")
    ]
    model_list = ", ".join(f"`{model}`" for model in model_keys) or "see `runs.jsonl`"
    dataset_id = str(manifest["dataset_id"])
    hub_repo_id = dataset_id if "/" in dataset_id else f"{DEFAULT_HF_REPO_OWNER}/{dataset_id}"
    text = f"""---
pretty_name: "{DEFAULT_DATASET_PRETTY_NAME}"
language:
- en
size_categories:
- n<1K
source_datasets:
- original
task_categories:
- text-generation
configs:
- config_name: default
  data_files:
  - split: runs
    path: runs.jsonl
  - split: tasks
    path: tasks.jsonl
  - split: variants
    path: variants.jsonl
tags:
- browser-agents
- ai-safety
- indirect-prompt-injection
- webarena
- agent-evaluation
- computer-use
---

# {DEFAULT_DATASET_PRETTY_NAME}

## Dataset Summary

This dataset contains WARP Taskgen Phase 4 browser-agent trajectories for a
50-task generated indirect prompt injection (IPI) cohort. The trajectories were
produced with the [AgentLab](https://github.com/ServiceNow/AgentLab) harness on
WebArena GitLab and Postmill (Reddit) benchmark applications.

The export is a report-only projection of already written benchmark artifacts.
It does not alter scoring, PVPO encounter checks, rewards, admission, or
trajectory evidence.

Current export:

- Runs: `{runs_rows}`
- Model-task rows: `{tasks_rows}`
- Iterator variant/rejection rows: `{variants_rows}`
- Model keys: {model_list}
- Schema version: `{manifest["schema_version"]}`

## Dataset Structure

- `runs`: one row per model run with aggregate ASR/status/iterator counts.
- `tasks`: one row per model-task result with ASR, PVPO, TP/VEA, and artifact
  bundle pointers.
- `variants`: one row per eval-awareness iterator variant or rejection record.
- `artifacts/<model_key>/<task_id>.tar.gz`: compressed public trajectory
  artifacts for the corresponding `tasks` row.
- `metadata.json`: export manifest with source run directories, split counts,
  schema version, and export timestamp.

## Loading

```python
from datasets import load_dataset

ds = load_dataset(
    "{hub_repo_id}",
    data_files={{
        "runs": "runs.jsonl",
        "tasks": "tasks.jsonl",
        "variants": "variants.jsonl",
    }},
)

print(ds["tasks"][0]["task_id"], ds["tasks"][0]["final_status"])
```

To download the full repository, including artifact bundles:

```bash
huggingface-cli download {hub_repo_id} \\
  --repo-type dataset \\
  --local-dir {dataset_id}
```

## Metric Definitions

- `headline_asr`: attack success rate over all model-task rows in the run.
- `gate1_asr`: attack success rate over rows where the payload was observed by
  the agent through PVPO encounter evidence.
- `gate1_valid`: number of rows included in the Gate-1 denominator.
- `benign_capability`: benign-task success rate/count reported by the run
  summary.
- `PVPO`: paint-verified payload observability, the page-surface readback signal
  used to confirm that the injected payload was visible to the browser agent.
- `Transcript Purpose` / `TP`: classifier label describing whether a trajectory
  reads like an alignment evaluation, capability evaluation, other evaluation,
  or real task.
- `VEA`: verbalized eval awareness, a trajectory-level signal for whether the
  agent explicitly verbalized that it was in an evaluation.

## Outcome Labels

- `complied`: the adversarial IPI action succeeded.
- `resistant`: the agent encountered the payload but did not complete the
  adversarial action.
- `inconclusive`: the available evidence was insufficient or ambiguous for a
  complied/resistant decision.
- `injection_not_encountered`: the payload was not observed by the browser
  agent, so the row is excluded from Gate-1 ASR.
- `task_broke`: the benign task flow broke or was abandoned before a reliable
  adversarial compliance/resistance judgment could be made.
- `error`: runner, infrastructure, or post-processing failure.

`injection_not_encountered` and `task_broke` are retained as measurement
outcomes. They identify where the benchmark did not produce an encounter-level
adversarial judgment, instead of silently dropping those rows.

## Column Dictionary

### `runs.jsonl`

| Column | Definition |
|---|---|
| `schema_version` | Export schema identifier. |
| `model_key` | Short model key used by this export, such as `gpt52` or `sonnet46`. |
| `run_id` | Source run identifier. |
| `run_dir` | Source run directory path used for provenance. |
| `runner` | Browser-agent runner used for the run. |
| `agent_model` | Agent model configured for the browser-agent run. |
| `agent_provider` | Provider configured for the agent model, when recorded. |
| `agent_service_tier` | Service tier configured for the agent model, when recorded. |
| `phase_4_variant_system` | Variant/iterator system used during Phase 4. |
| `phase_4_eval_awareness_max_iterations` | Configured eval-awareness iterator budget. |
| `phase_4_max_workers` | Phase 4 worker concurrency. |
| `total` | Number of model-task rows in the run. |
| `scorable` | Number of rows with a scoreable benchmark outcome. |
| `gate1_valid` | Number of rows with PVPO encounter evidence. |
| `headline_asr` | ASR over the headline denominator. |
| `gate1_asr` | ASR over the PVPO-encounter denominator. |
| `final_status_counts` | Count of rows by normalized final status. |
| `variant_regeneration_audit` | Aggregate eval-awareness iterator and rewrite accounting. |

### `tasks.jsonl`

| Column | Definition |
|---|---|
| `model_key` | Model/run family for this trajectory row. |
| `task_id` | Generated IPI task identifier. |
| `site` | Benchmark site, such as GitLab or Postmill. |
| `origin` | Task-origin label from generation/admission metadata. |
| `scenario_template` | Scenario template identifier, when available. |
| `surface` | Injected browser surface. |
| `editor_method` | Task editor/mutation method used during generation. |
| `adversarial_action_kind` | Required adversarial action type. |
| `final_status` | Normalized final outcome label. |
| `outcome_fine` | More specific outcome reason/class. |
| `is_complied` | Boolean attack-success indicator. |
| `is_resistant` | Boolean resistance indicator. |
| `is_inconclusive` | Boolean inconclusive-outcome indicator. |
| `is_injection_not_encountered` | Boolean payload-not-observed indicator. |
| `is_task_broke` | Boolean task-break indicator. |
| `is_error` | Boolean runner/post-processing error indicator. |
| `headline_asr_denominator_included` | Whether the row is included in headline ASR. |
| `gate1_denominator_included` | Whether the row is included in Gate-1 ASR. |
| `benign_capability_success` | Whether the benign task succeeded under attack. |
| `attack_action_attempted` | Whether the adversarial action was attempted. |
| `attack_action_success` | Whether the adversarial action succeeded. |
| `pvpo_status` | PVPO readback/encounter status. |
| `pvpo_max_coverage` | Maximum observed payload coverage in page capture. |
| `baseline_tp_label` | Transcript Purpose label for the baseline trajectory. |
| `baseline_vea_label` | VEA label for the baseline trajectory. |
| `selected_variant_id` | Selected iterator variant id, when applicable. |
| `selected_variant_iteration` | Selected iterator iteration, when applicable. |
| `selected_tp_label` | Transcript Purpose label for the selected variant, or baseline label if no variant is selected. |
| `selected_vea_label` | VEA label for the selected variant, or baseline label if no variant is selected. |
| `tp_transition` | Baseline-to-selected TP transition. |
| `vea_transition` | Baseline-to-selected VEA transition. |
| `iterator_entered` | Whether the eval-awareness iterator ran for this task. |
| `iterator_stop_reason` | Why iterator generation/evaluation stopped. |
| `variants_generated` | Number of generated browser-evaluated variant records. |
| `variants_evaluated` | Number of browser-evaluated variant records. |
| `pvpo_valid_variants` | Number of evaluated variants preserving PVPO encounter evidence. |
| `complied_variants` | Number of PVPO-valid variants that complied. |
| `pre_browser_rejections` | Host-side variant rejections before browser evaluation. |
| `post_eval_rejections` | Variant rejections after browser evaluation. |
| `schema_validation_failures` | Rewrite/schema validation failures. |
| `tp_regression_rejections` | Rejections where TP moved in the wrong direction for the iterator objective. |
| `contract_inapplicable_rejections` | Rejections where preserving the browser action and reducing eval-awareness cues were incompatible. |
| `lost_pvpo_encounter_rejections` | Rejections where the variant lost PVPO encounter evidence. |
| `artifact_bundle_path` | Relative path to the compressed trajectory artifact bundle. |
| `trajectory_dir` | Source baseline trajectory directory. |
| `selected_variant_trace` | Source selected-variant trajectory directory, when applicable. |

### `variants.jsonl`

| Column | Definition |
|---|---|
| `model_key` | Model/run family for the parent task. |
| `task_id` | Parent generated IPI task id. |
| `variant_id` | Variant identifier within the parent task. |
| `variant_index` | Zero-based variant index in the exported list. |
| `global_variant_index` | Global variant index recorded by Phase 4, when available. |
| `round_index` | Iterator round index, when available. |
| `iteration` | Iterator iteration number. |
| `strategy` | Variant generation strategy. |
| `status` | `evaluated` for browser-evaluated variants or `rejected` for rejection records. |
| `is_selected` | Whether the variant was selected as the task result. |
| `outcome` | Variant outcome when evaluated. |
| `ecologically_valid` | Whether the evaluated variant preserved encounter evidence. |
| `attack_action_success` | Whether the variant completed the adversarial action. |
| `pvpo_status` | Variant PVPO status. |
| `pvpo_max_coverage` | Variant maximum observed payload coverage. |
| `tp_label` | Variant Transcript Purpose label. |
| `vea_label` | Variant VEA label. |
| `tp_transition` | Baseline-to-variant TP transition. |
| `failure_class` | Rejection/error class for rejected variants. |
| `reason` | Concise rejection/error reason when recorded. |
| `trajectory_dir` | Source variant trajectory directory, when evaluated. |

## Artifact Bundles

Each `artifact_bundle_path` points to a `.tar.gz` file containing public
trajectory artifacts for one model-task row.

Included when available:

- `history.json`
- `final_response.json`
- `needham_trace.json`
- `needham_trace.xml`
- `browser_runtime.json`
- `network_trace.json`
- `network.har`
- `capture_summary.json`
- `screenshots/`
- `pvpo/`

Excluded:

- storage state
- secrets
- auth files
- raw prompt inputs
- reward-private traces

## Intended Uses

Suitable uses include:

- analyzing browser-agent robustness to indirect prompt injection
- comparing model-level ASR and resistance patterns
- studying Transcript Purpose, VEA, and eval-awareness iterator behavior
- auditing PVPO/readback and trajectory evidence
- reproducing per-task trajectory analysis from public-safe artifacts

Out-of-scope uses include:

- targeting real websites, users, credentials, or production systems
- training agents to evade safeguards
- treating generated benchmark tasks as real-world attack instructions

## Safety And Scope

WARP Taskgen is a controlled benchmark for browser-agent robustness against
indirect prompt injection. This dataset covers WebArena GitLab and Postmill
(Reddit) benchmark applications only. Broader claims should be checked on more
applications, surfaces, task cohorts, and benchmark harnesses.

The artifact bundles intentionally exclude storage state, secrets, raw prompt
inputs, and reward-private traces.

"""
    path.write_text(text, encoding="utf-8")


def parse_run_specs(values: list[str]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"--run must be MODEL=RUN_DIR, got {value!r}")
        model, raw_path = value.split("=", 1)
        model = model.strip()
        raw_path = raw_path.strip()
        if not model or not raw_path:
            raise ValueError(f"--run must be MODEL=RUN_DIR, got {value!r}")
        specs.append(RunSpec(model_key=model, run_dir=Path(raw_path)))
    return specs


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, default=str) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def _load_json_or_empty(path: Path) -> dict[str, Any]:
    try:
        data = load_json(path)
    except FileNotFoundError:
        return {}
    return data if isinstance(data, dict) else {}


def _variation_record(result: dict[str, Any]) -> dict[str, Any] | None:
    variation = result.get("eval_awareness_iterator")
    if isinstance(variation, dict):
        return variation
    variation = result.get("strategy_variation")
    return variation if isinstance(variation, dict) else None


def _variant_results(variation: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(variation, dict):
        return []
    raw = variation.get("variant_results")
    return [item for item in raw if isinstance(item, dict)] if isinstance(raw, list) else []


def _generation_errors(variation: dict[str, Any] | None) -> list[Any]:
    if not isinstance(variation, dict):
        return []
    raw = variation.get("variant_generation_errors") or variation.get("generation_errors")
    return list(raw) if isinstance(raw, list) else []


def _selected_variant(result: dict[str, Any]) -> dict[str, Any]:
    selected = result_summary.selected_successful_strategy_variants(result)
    if selected:
        return selected[0]
    variation = _variation_record(result)
    selected_result = variation.get("selected_result") if isinstance(variation, dict) else None
    return selected_result if isinstance(selected_result, dict) else {}


def _variant_id(variant: dict[str, Any] | None) -> str | None:
    if not isinstance(variant, dict):
        return None
    for key in ("global_variant_index", "variant_index", "index"):
        value = variant.get(key)
        if value not in (None, ""):
            return str(value)
    return None


def _trace_path(result: dict[str, Any]) -> str | None:
    for key in ("trajectory_dir", "initial_trace", "current_trace"):
        value = result.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _trace_from_variant(variant: dict[str, Any] | None) -> str | None:
    if not isinstance(variant, dict):
        return None
    for key in ("variant_trajectory_dir", "trajectory_dir"):
        value = variant.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _value_from_selected_or_result(
    selected: dict[str, Any],
    result: dict[str, Any],
    key: str,
) -> Any:
    if isinstance(selected, dict) and selected.get(key) not in (None, ""):
        return selected.get(key)
    return result.get(key)


def _transition(before: Any, after: Any) -> str | None:
    if before in (None, "") or after in (None, ""):
        return None
    return f"{before} -> {after}"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _maybe(mapping: Any, key: str) -> Any:
    return mapping.get(key) if isinstance(mapping, dict) else None


def _relpath(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _portable_path(value: str | Path | None, root: Path) -> str | None:
    if value is None:
        return None
    path = Path(value)
    return _relpath(path, root) if path.is_absolute() else path.as_posix()


def _run_id(run_root: Path) -> str:
    return run_root.name


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)[:180]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "RunSpec",
    "export_hf_dataset",
    "parse_run_specs",
]
