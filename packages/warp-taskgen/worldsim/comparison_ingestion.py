"""Native comparison-result ingestion for AgentLab sidecar runs.

Comparison runs have a different authority and result vocabulary from WARP
evaluation.  This module is the narrow boundary between the native AgentLab
JSON payload and the rest of Taskgen.  It deliberately owns no browser,
reset, scoring, or Phase 4 behavior.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from worldsim.atomic_io import write_json_atomic
from worldsim.benchmark_capabilities import (
    DEFAULT_BENCHMARK_CATALOG,
    infer_benchmark_from_metadata,
)

ComparisonStatus = Literal["success", "failure", "error", "timeout"]
ComparisonResultKind = Literal["comparison"]
ComparisonEvaluatorAuthority = Literal["comparison_runner"]

_SCHEMA_VERSION = 1
COMPARISON_RESULT_FILENAME = "comparison_result.json"
_COMPARISON_OUTCOME_MODES = frozenset({"attack_success", "resistance", "capability"})
_STATUSES = frozenset({"success", "failure", "error", "timeout"})

# These fields belong to WARP evaluation or Phase 4.  Accepting them in a
# native comparison envelope would make it possible to mistake a WARP result
# for a benchmark-native result.  The native AgentLab ``run`` payload does not
# emit any of these fields.
_WARP_FIELDS = frozenset(
    {
        "adversarial_passed",
        "adversarial_reward",
        "asr",
        "final_status",
        "network_trace",
        "outcome",
        "phase4",
        "phase_4",
        "pvpo",
        "warp_evaluation",
        "warp_reward",
    }
)
_NESTED_WARP_FIELDS = _WARP_FIELDS - {"network_trace"}


def _freeze(value: Any) -> Any:
    """Recursively freeze JSON-shaped values for an immutable record."""

    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("comparison metadata keys must be non-empty strings")
            frozen[key] = _freeze(item)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("comparison metadata numbers must be finite")
        return value
    raise ValueError(
        f"comparison metadata must contain only JSON-shaped values, got {type(value).__name__}"
    )


def _thaw(value: Any) -> Any:
    """Convert frozen values back to JSON-compatible containers."""

    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _nested_warp_fields(value: object, *, prefix: str) -> list[str]:
    """Return reserved WARP field paths inside preserved native metadata."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}"
            if key in _NESTED_WARP_FIELDS:
                found.append(path)
            found.extend(_nested_warp_fields(item, prefix=path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found.extend(_nested_warp_fields(item, prefix=f"{prefix}[{index}]"))
    return found


def _require_mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"comparison payload {field} must be a mapping")
    for key in value:
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"comparison payload {field} keys must be non-empty strings")
    return value


def _require_nonempty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"comparison payload {field} must be a non-empty string")
    return value.strip()


def _required_mapping_field(
    payload: Mapping[str, object],
    field: str,
) -> Mapping[str, object]:
    if field not in payload:
        raise ValueError(f"comparison payload {field} is required")
    return _require_mapping(payload[field], field=field)


def _identity(mapping: Mapping[str, object], *, label: str) -> str | None:
    values: list[str] = []
    for field in ("id", "task_id"):
        if field not in mapping:
            continue
        value = mapping[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"comparison {label} {field} must be a non-empty string")
        values.append(value.strip())
    if not values:
        return None
    if len(set(values)) != 1:
        raise ValueError(f"comparison {label} has conflicting task identity")
    return values[0]


def _finite_number(value: object, *, field: str, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"comparison payload {field} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"comparison payload {field} must be a finite number")
    return number


def _nonnegative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"comparison payload {field} must be a non-negative integer")
    return value


def _schema_version(payload: Mapping[str, object]) -> int:
    value = payload.get("schema_version")
    if isinstance(value, bool) or not isinstance(value, int) or value != _SCHEMA_VERSION:
        raise ValueError(f"comparison payload schema_version must be exactly {_SCHEMA_VERSION}")
    return value


@dataclass(frozen=True)
class ComparisonRecord:
    """Immutable, benchmark-native result envelope.

    ``native_reward`` is intentionally named to prevent accidental routing to
    WARP reward dispatch.  All mappings are frozen recursively at construction
    and serialized through :meth:`to_dict`.
    """

    schema_version: Literal[1]
    result_kind: ComparisonResultKind
    benchmark_name: str
    comparison_outcome_mode: str
    evaluator_authority: ComparisonEvaluatorAuthority
    task_id: str
    status: ComparisonStatus
    passed: bool
    native_reward: float
    steps: int
    elapsed_s: float
    error: str | None
    summary_info: Mapping[str, object]
    artifact_refs: Mapping[str, object]
    versions: Mapping[str, object]
    model: Mapping[str, object]
    benchmark_config: Mapping[str, object]
    provenance: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("comparison record has unsupported schema_version")
        if self.result_kind != "comparison":
            raise ValueError("comparison record result_kind must be 'comparison'")
        if self.comparison_outcome_mode not in _COMPARISON_OUTCOME_MODES:
            raise ValueError("comparison record has unsupported outcome mode")
        if self.evaluator_authority != "comparison_runner":
            raise ValueError("comparison record requires comparison_runner authority")
        benchmark_name = _require_nonempty_string(
            self.benchmark_name,
            field="benchmark_name",
        )
        capabilities = DEFAULT_BENCHMARK_CATALOG.require(
            benchmark_name,
            "comparison_ingestion",
        )
        if capabilities.canonical_name != benchmark_name:
            raise ValueError("comparison record benchmark_name must be canonical")
        if capabilities.evaluator_authorities != ("comparison_runner",):
            raise ValueError("comparison record benchmark requires comparison_runner authority")
        if capabilities.comparison_outcome_mode != self.comparison_outcome_mode:
            raise ValueError("comparison record outcome mode conflicts with Benchmark Contract")
        _require_nonempty_string(self.task_id, field="task_id")
        if self.status not in _STATUSES:
            raise ValueError("comparison record has unsupported status")
        if not isinstance(self.passed, bool):
            raise ValueError("comparison record passed must be boolean")
        if self.passed != (self.status == "success"):
            raise ValueError("comparison record status and passed conflict")
        _finite_number(self.native_reward, field="native_reward")
        _nonnegative_int(self.steps, field="steps")
        _finite_number(self.elapsed_s, field="elapsed_s")
        if self.elapsed_s < 0:
            raise ValueError("comparison record elapsed_s must be non-negative")
        if self.error is not None and not isinstance(self.error, str):
            raise ValueError("comparison record error must be a string or null")
        if self.status in {"success", "failure"} and self.error is not None:
            raise ValueError("comparison record non-error status cannot include error")
        if self.status in {"error", "timeout"} and not self.error:
            raise ValueError("comparison record error status requires error detail")
        for field in (
            "summary_info",
            "artifact_refs",
            "versions",
            "model",
            "benchmark_config",
            "provenance",
        ):
            _require_mapping(getattr(self, field), field=field)
        nested_warp_fields: list[str] = []
        for field in (
            "summary_info",
            "versions",
            "model",
            "benchmark_config",
            "provenance",
        ):
            nested_warp_fields.extend(_nested_warp_fields(getattr(self, field), prefix=field))
        if nested_warp_fields:
            raise ValueError(
                f"comparison record contains nested WARP-only fields: {nested_warp_fields}"
            )
        summary_steps = _nonnegative_int(
            self.summary_info.get("n_steps"),
            field="summary_info.n_steps",
        )
        if summary_steps != self.steps:
            raise ValueError("comparison record steps conflict with summary_info.n_steps")
        summary_reward = _finite_number(
            self.summary_info.get("cum_reward"),
            field="summary_info.cum_reward",
        )
        if summary_reward != self.native_reward:
            raise ValueError(
                "comparison record native_reward conflicts with summary_info.cum_reward"
            )
        summary_error = self.summary_info.get("err_msg")
        if summary_error is not None and not isinstance(summary_error, str):
            raise ValueError("comparison record summary_info.err_msg must be a string or null")
        if summary_error != self.error:
            raise ValueError("comparison record error conflicts with summary_info.err_msg")
        for field in ("terminated", "truncated"):
            if not isinstance(self.summary_info.get(field), bool):
                raise ValueError(f"comparison record summary_info.{field} must be boolean")
        if self.benchmark_config.get("status") != "applied":
            raise ValueError("comparison record benchmark_config status must be 'applied'")
        config_benchmark = infer_benchmark_from_metadata((self.benchmark_config,))
        if config_benchmark != self.benchmark_name:
            raise ValueError("comparison record benchmark_config identity conflicts with benchmark")
        for field in (
            "summary_info",
            "artifact_refs",
            "versions",
            "model",
            "benchmark_config",
            "provenance",
        ):
            object.__setattr__(self, field, _freeze(getattr(self, field)))

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-compatible copy of this comparison envelope."""

        return {
            "schema_version": self.schema_version,
            "result_kind": self.result_kind,
            "benchmark_name": self.benchmark_name,
            "comparison_outcome_mode": self.comparison_outcome_mode,
            "evaluator_authority": self.evaluator_authority,
            "task_id": self.task_id,
            "status": self.status,
            "passed": self.passed,
            "native_reward": self.native_reward,
            "steps": self.steps,
            "elapsed_s": self.elapsed_s,
            "error": self.error,
            "summary_info": _thaw(self.summary_info),
            "artifact_refs": _thaw(self.artifact_refs),
            "versions": _thaw(self.versions),
            "model": _thaw(self.model),
            "benchmark_config": _thaw(self.benchmark_config),
            "provenance": _thaw(self.provenance),
        }


def ingest_comparison_payload(
    task: Mapping[str, object],
    payload: Mapping[str, object],
    *,
    artifact_dir: Path | None = None,
) -> ComparisonRecord:
    """Validate one native AgentLab payload and return a comparison record.

    This function intentionally performs no reward inference.  A native reward
    is read from the sidecar's top-level ``reward`` (or explicit
    ``native_reward``) field and checked against ``summary_info.cum_reward``.
    """

    task_mapping = _require_mapping(task, field="task")
    payload_mapping = _require_mapping(payload, field="root")
    _schema_version(payload_mapping)
    unexpected_warp_fields = sorted(field for field in _WARP_FIELDS if field in payload_mapping)
    if unexpected_warp_fields:
        raise ValueError(f"comparison payload contains WARP-only fields: {unexpected_warp_fields}")
    result_kind = payload_mapping.get("result_kind")
    if result_kind is not None and result_kind != "comparison":
        raise ValueError("comparison payload result_kind must be 'comparison'")
    if payload_mapping.get("mode") not in (None, "comparison"):
        raise ValueError("comparison ingestion accepts only native comparison mode")

    benchmark_config = _required_mapping_field(payload_mapping, "benchmark_config")
    if benchmark_config.get("status") != "applied":
        raise ValueError("comparison payload benchmark_config status must be 'applied'")
    if "benchmark_name" not in benchmark_config:
        raise ValueError("comparison payload benchmark_config benchmark_name is required")
    benchmark_name = infer_benchmark_from_metadata(
        (task_mapping, payload_mapping, benchmark_config)
    )
    if benchmark_name is None:
        raise ValueError("comparison task is missing benchmark metadata")
    capabilities = DEFAULT_BENCHMARK_CATALOG.require(benchmark_name, "comparison_ingestion")
    if capabilities.evaluator_authorities != ("comparison_runner",):
        raise ValueError("comparison ingestion requires comparison_runner authority")
    outcome_mode = capabilities.comparison_outcome_mode
    if outcome_mode not in _COMPARISON_OUTCOME_MODES:
        raise ValueError(f"benchmark {benchmark_name!r} has no comparison outcome mode")
    explicit_mode = payload_mapping.get("comparison_outcome_mode")
    if explicit_mode is None:
        explicit_mode = payload_mapping.get("outcome_mode")
    if explicit_mode is not None and explicit_mode != outcome_mode:
        raise ValueError(
            f"comparison outcome mode {explicit_mode!r} conflicts with catalog {outcome_mode!r}"
        )

    expected_task_id = _identity(task_mapping, label="task")
    payload_task_id = _identity(payload_mapping, label="payload")
    if expected_task_id is None:
        raise ValueError("comparison task is missing id/task_id")
    if payload_task_id is None:
        raise ValueError("comparison payload is missing task_id")
    if expected_task_id != payload_task_id:
        raise ValueError(
            f"comparison task identity {expected_task_id!r} does not match payload "
            f"{payload_task_id!r}"
        )

    status = payload_mapping.get("status")
    if not isinstance(status, str) or status not in _STATUSES:
        raise ValueError(f"comparison payload status must be one of {sorted(_STATUSES)}")
    passed = payload_mapping.get("passed")
    if not isinstance(passed, bool):
        raise ValueError("comparison payload passed must be boolean")
    if passed != (status == "success"):
        raise ValueError("comparison payload status and passed conflict")

    summary_info = _required_mapping_field(payload_mapping, "summary_info")
    artifact_refs = _required_mapping_field(payload_mapping, "artifacts")
    versions = _required_mapping_field(payload_mapping, "versions")
    model = _required_mapping_field(payload_mapping, "model")
    for value in (summary_info, artifact_refs, versions, model, benchmark_config):
        _freeze(value)
    nested_warp_fields: list[str] = []
    for field, value in (
        ("summary_info", summary_info),
        ("versions", versions),
        ("model", model),
        ("benchmark_config", benchmark_config),
    ):
        nested_warp_fields.extend(_nested_warp_fields(value, prefix=field))
    if nested_warp_fields:
        raise ValueError(
            f"comparison payload contains nested WARP-only fields: {nested_warp_fields}"
        )

    reward_values: list[float] = []
    for field in ("native_reward", "reward"):
        if field in payload_mapping:
            reward_values.append(
                _finite_number(payload_mapping[field], field=field, allow_none=False)
            )
    if not reward_values:
        raise ValueError("comparison payload reward or native_reward is required")
    elif len(reward_values) == 1:
        native_reward = reward_values[0]
    elif reward_values[0] != reward_values[1]:
        raise ValueError("comparison payload native_reward and reward conflict")
    else:
        native_reward = reward_values[0]
    assert native_reward is not None

    if "steps" not in payload_mapping:
        raise ValueError("comparison payload steps is required")
    steps = _nonnegative_int(payload_mapping["steps"], field="steps")
    summary_steps = _nonnegative_int(
        summary_info.get("n_steps"),
        field="summary_info.n_steps",
    )
    if summary_steps != steps:
        raise ValueError("comparison payload steps conflict with summary_info.n_steps")
    summary_reward = _finite_number(
        summary_info.get("cum_reward"),
        field="summary_info.cum_reward",
    )
    if summary_reward != native_reward:
        raise ValueError("comparison payload native_reward conflicts with summary_info.cum_reward")
    summary_error = summary_info.get("err_msg")
    if summary_error is not None and not isinstance(summary_error, str):
        raise ValueError("comparison payload summary_info.err_msg must be a string or null")
    for field in ("terminated", "truncated"):
        if not isinstance(summary_info.get(field), bool):
            raise ValueError(f"comparison payload summary_info.{field} must be boolean")
    elapsed_value = payload_mapping.get("elapsed", payload_mapping.get("elapsed_s"))
    elapsed_s = _finite_number(elapsed_value, field="elapsed", allow_none=False)
    assert elapsed_s is not None
    if elapsed_s < 0:
        raise ValueError("comparison payload elapsed must be non-negative")
    error = payload_mapping.get("error")
    if error is not None and not isinstance(error, str):
        raise ValueError("comparison payload error must be a string or null")
    if status in {"success", "failure"} and error is not None:
        raise ValueError("comparison payload non-error status cannot include error")
    if status in {"error", "timeout"} and not error:
        raise ValueError("comparison payload error status requires error detail")
    if summary_error != error:
        raise ValueError("comparison payload error conflicts with summary_info.err_msg")
    authority = payload_mapping.get("evaluator_authority", "comparison_runner")
    if authority != "comparison_runner":
        raise ValueError("comparison payload requires comparison_runner authority")

    provenance: dict[str, object] = {
        "source": "agentlab_native_run",
        "sidecar_schema_version": _SCHEMA_VERSION,
    }
    if artifact_dir is not None:
        provenance["artifact_dir"] = str(artifact_dir)

    return ComparisonRecord(
        schema_version=_SCHEMA_VERSION,
        result_kind="comparison",
        benchmark_name=benchmark_name,
        comparison_outcome_mode=outcome_mode,
        evaluator_authority="comparison_runner",
        task_id=expected_task_id,
        status=status,  # type: ignore[arg-type]
        passed=passed,
        native_reward=native_reward,
        steps=steps,
        elapsed_s=elapsed_s,
        error=error,
        summary_info=summary_info,
        artifact_refs=artifact_refs,
        versions=versions,
        model=model,
        benchmark_config=benchmark_config,
        provenance=provenance,
    )


def write_comparison_result(path: Path, record: ComparisonRecord) -> None:
    """Atomically persist a validated comparison record as JSON."""

    if not isinstance(path, Path):
        raise TypeError("comparison result path must be a pathlib.Path")
    if not isinstance(record, ComparisonRecord):
        raise TypeError("comparison result must be a ComparisonRecord")
    write_json_atomic(path, record.to_dict())


__all__ = [
    "COMPARISON_RESULT_FILENAME",
    "ComparisonRecord",
    "ingest_comparison_payload",
    "write_comparison_result",
]
