"""Phase 2c instance/config gating helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from warp_taskgen.benchmark_capabilities import (
    infer_benchmark_name,
    infer_instances_config_benchmark,
)
from warp_taskgen.config import BenchmarkInstance
from warp_taskgen.phase_2.output import _effective_task_site


def _feasibility_status(record: Mapping[str, Any]) -> str | None:
    feasibility = record.get("feasibility")
    if not isinstance(feasibility, Mapping):
        return None
    status = feasibility.get("status")
    return str(status) if isinstance(status, str) else None


def _sites_filter_from_value(value: Any) -> set[str] | None:
    if not isinstance(value, str) or not value.strip():
        return None
    sites = {site.strip() for site in value.split(",") if site.strip()}
    return sites or None


def _filter_records_for_sites(
    records: list[dict[str, Any]],
    sites_filter: set[str] | None,
) -> list[dict[str, Any]]:
    if sites_filter is None:
        return records
    return [record for record in records if _effective_task_site(record) in sites_filter]


def _filter_instances_for_phase_2c(
    instances: list[dict[str, Any]],
    selected_records: list[dict[str, Any]],
    *,
    sites_filter: set[str] | None,
) -> list[dict[str, Any]]:
    """Return only benchmark instances needed by the selected Phase 2c tasks.

    ``--sites`` already filters the task JSON handed to ``verify_feasibility``.
    Keep the instances in lockstep so preflight token acquisition does not try
    to mint credentials for unrelated local services that are intentionally down
    on a scoped run.
    """
    if sites_filter is None:
        return instances
    active_sites = {
        _effective_task_site(record)
        for record in selected_records
        if isinstance(record, dict) and _effective_task_site(record)
    }
    if not active_sites:
        active_sites = set(sites_filter)
    return [
        instance
        for instance in instances
        if str(instance.get("site_name", "")).strip() in active_sites
    ]


def _terminal_phase_2_status(prior_phase_2_status: str | None) -> str:
    """Map transient pre-2c state into the terminal Phase 2 checkpoint."""
    if prior_phase_2_status == "partial_complete":
        return "partial_complete"
    return "complete"


def _extract_instances_list(payload: Any) -> list[dict[str, Any]]:
    """Accept both the wrapper shape (``{"instances": [...]}``) and a raw list.

    The production ``instances.smoke.json`` / ``instances.scale.json`` files
    are wrapper dicts; some fixtures (and older tooling) hand back a flat list.
    """
    if isinstance(payload, list):
        return [
            _normalize_instance_record(item, None) for item in payload if isinstance(item, dict)
        ]
    if isinstance(payload, dict):
        nested = payload.get("instances")
        try:
            wrapper_benchmark = infer_instances_config_benchmark(payload)
        except ValueError:
            wrapper_benchmark = None
        if isinstance(nested, list):
            return [
                _normalize_instance_record(item, wrapper_benchmark)
                for item in nested
                if isinstance(item, dict)
            ]
    return []


def _validate_phase_2c_instances_payload(payload: Any) -> None:
    """Run config validators before Phase 2c uses raw instance dicts."""
    if isinstance(payload, dict):
        nested = payload.get("instances")
        if not isinstance(nested, list):
            raise ValueError("wrapper object must contain an instances list")
        for index, item in enumerate(nested):
            if not isinstance(item, dict):
                raise ValueError(f"instances[{index}] must be an object")
            _validate_phase_2c_instance_record(item, label=f"instances[{index}]")
        return
    if isinstance(payload, list):
        for index, item in enumerate(payload):
            if not isinstance(item, dict):
                raise ValueError(f"instance[{index}] must be an object")
            _validate_phase_2c_instance_record(item, label=f"instance[{index}]")
        return
    raise ValueError("expected wrapper object with instances or a raw instance list")


def _validate_phase_2c_instance_record(instance: dict[str, Any], *, label: str) -> None:
    try:
        BenchmarkInstance.model_validate(instance)
    except ValidationError as exc:
        messages: list[str] = []
        for error in exc.errors(include_input=False):
            loc = ".".join(str(part) for part in error.get("loc", ())) or "<root>"
            error_type = str(error.get("type") or "validation_error")
            msg = str(error.get("msg") or error_type)
            messages.append(f"{label}.{loc}: {msg} ({error_type})")
        raise ValueError("; ".join(messages) or f"{label}: invalid instance") from exc


def _normalize_instance_record(
    instance: dict[str, Any],
    wrapper_benchmark: str | None,
) -> dict[str, Any]:
    normalized = dict(instance)
    values = [
        wrapper_benchmark,
        normalized.get("benchmark"),
        normalized.get("benchmark_name"),
        normalized.get("benchmark_adapter"),
    ]
    try:
        benchmark = infer_benchmark_name(values)
    except ValueError:
        benchmark = None
    if benchmark is not None:
        normalized["benchmark"] = benchmark
    return normalized


def _gate_phase_2c_benchmark(
    *,
    task_records: list[dict[str, Any]],
    raw_instances: Any,
    instances: list[dict[str, Any]],
    runtime_composition: Any | None = None,
) -> str:
    task_benchmark = _infer_task_records_benchmark(
        task_records,
        label="Phase 2 adversarial tasks",
    )
    instances_benchmark: str | None = None
    if isinstance(raw_instances, dict):
        instances_benchmark = infer_instances_config_benchmark(raw_instances)
    if instances_benchmark is None:
        instances_benchmark = _infer_task_records_benchmark(
            instances,
            label="Phase 2c instances",
        )
    if task_benchmark != instances_benchmark:
        raise ValueError(
            "mixed benchmark metadata between Phase 2 tasks and Phase 2c instances: "
            f"tasks={task_benchmark!r}, instances={instances_benchmark!r}"
        )
    try:
        from warp_taskgen.runtime_composition import benchmark_capabilities_for_runtime

        capabilities = benchmark_capabilities_for_runtime(
            task_benchmark, runtime_composition
        ).require("phase_2_feasibility")
    except ValueError as exc:
        raise ValueError(
            f"benchmark {task_benchmark!r} does not support WARP Taskgen Phase 2c"
        ) from exc
    return capabilities.canonical_name


def _gate_phase_2_skip_benchmark(
    task_records: list[dict[str, Any]], runtime_composition: Any | None = None
) -> str:
    benchmark = _infer_task_records_benchmark(
        task_records,
        label="Phase 2 adversarial tasks",
    )
    try:
        from warp_taskgen.runtime_composition import benchmark_capabilities_for_runtime

        capabilities = benchmark_capabilities_for_runtime(
            benchmark, runtime_composition
        ).require("phase_2_generation")
    except ValueError as exc:
        raise ValueError(f"benchmark {benchmark!r} does not support WARP Taskgen Phase 2") from exc
    try:
        capabilities = capabilities.require("phase_2_feasibility")
    except ValueError as exc:
        raise ValueError(f"benchmark {benchmark!r} does not support WARP Taskgen Phase 2c") from exc
    return capabilities.canonical_name


def _infer_task_records_benchmark(records: list[dict[str, Any]], *, label: str) -> str:
    values: list[Any] = []
    for record in records:
        if not isinstance(record, dict):
            continue
        values.extend(_benchmark_values_from_record(record))
    try:
        benchmark = infer_benchmark_name(values)
    except ValueError as exc:
        raise ValueError(f"{label} contain {exc}") from exc
    if benchmark is None:
        raise ValueError(f"{label} are missing benchmark metadata")
    return benchmark


def _benchmark_values_from_record(record: Mapping[str, Any]) -> list[Any]:
    values: list[Any] = [
        record.get("benchmark"),
        record.get("benchmark_name"),
        record.get("benchmark_adapter"),
    ]
    seed = record.get("adversarial_data_seed")
    values.extend(_benchmark_values_from_seed(seed))
    seed_template = record.get("seed_template")
    values.extend(_benchmark_values_from_seed(seed_template))
    return values


def _benchmark_values_from_seed(seed: Any) -> list[Any]:
    values: list[Any] = []
    calls = seed.get("editor_calls") if isinstance(seed, Mapping) else None
    if not isinstance(calls, list):
        return values
    for call in calls:
        if isinstance(call, Mapping):
            values.extend(
                (
                    call.get("benchmark"),
                    call.get("benchmark_name"),
                    call.get("benchmark_adapter"),
                )
            )
    return values


def _with_benchmark(instance: dict[str, Any], benchmark: str) -> dict[str, Any]:
    item = dict(instance)
    item["benchmark"] = benchmark
    return item
