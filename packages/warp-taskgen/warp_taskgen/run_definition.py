"""Run Definition projection, transition, and advisory resume planning.

This module owns the pure identity decision. State persistence and CLI routing
consume its value objects, while feature-owned checkpoint validators retain
reuse authority. The CLI's explicit Derived Run operation owns materialization;
legacy runs are never assigned an inferred identity.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit

from warp_taskgen.run_definition_contracts import (
    _SCHEMA_VERSION,
    CheckpointAction,
    CheckpointDecision,
    LifecycleAction,
    ResumeMode,
    ResumePlan,
    RunDefinition,
    _digest,
    _freeze,
    _optional_identity,
    _thaw,
)

_PHASE_ORDER = tuple("phase_0a phase_0b phase_0c phase_0d phase_1 phase_2 phase_3 phase_4".split())


def _fields(names: str) -> tuple[str, ...]:
    return tuple(names.split())


# Only non-secret, result-affecting state fields may enter the definition.
# Feature owners retain authority over the finer-grained fingerprints used to
# accept or reject their own checkpoints.
_CONTRIBUTOR_FIELDS: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "pipeline": _fields(
            "benchmark_name benchmark_path manifest_path instances_path "
            "host_inventory_instances_path host_inventory_instances_sha256 sites "
            "task_origin max_tasks_per_site"
        ),
        "phase_1": _fields(
            "sandbox_model generate_novel novel_tasks_per_site task_card_plan_path "
            "task_card_plan_digest task_capability_profile phase_1_action_counts action_counts"
        ),
        "phase_2": _fields(
            "phase_2b_texts_per_plan phase_2_text_model phase_2a_action_policy "
            "phase_2a_resolution_signature exposure_contract_signature skip_feasibility "
            "feasibility_only feasibility_instances feasibility_retry_count "
            "feasibility_ttl_hours force_reverify no_l3_l4 runtime_composition"
        ),
        "phase_4": _fields(
            "agent_model agent_runner agent_provider agent_service_tier agent_llm_timeout "
            "agent_step_timeout agent_task_timeout phase_4_variant_budget "
            "phase_4_variant_system phase_4_eval_awareness_max_iterations "
            "adversarial_action_kind phase_4_task_id skip_intermediate_asr "
            "intermediate_asr_max_steps_per_task allow_unknown_auth "
            "skip_host_bound_storage_state_auth runtime_composition"
        ),
    }
)
_ORDER_INSENSITIVE_FIELDS = frozenset({"sites"})
_PATH_FIELDS = frozenset({"feasibility_instances"})
_SENSITIVE_KEY_PARTS = (
    "auth",
    "authorization",
    "cookie",
    "database",
    "header",
    "password",
    "proxy",
    "secret",
    "session",
    "token",
    "api_key",
    "db_connection",
    "credentials",
)


def _normalise_value(value: object, *, field: str) -> object:
    if isinstance(value, Path):
        return str(value.expanduser().resolve(strict=False))
    if isinstance(value, Mapping):
        normalised: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError(f"run definition {field} keys must be non-empty strings")
            if _is_sensitive_key(key):
                normalised[key] = _safe_sensitive_identity(item)
            else:
                normalised[key] = _normalise_value(item, field=f"{field}.{key}")
        return normalised
    if field == "sites" and isinstance(value, str):
        return sorted({item.strip() for item in value.split(",") if item.strip()})
    if isinstance(value, (set, frozenset)):
        item_field = f"{field}[]" if field in _ORDER_INSENSITIVE_FIELDS else field
        return sorted(
            (_normalise_value(item, field=item_field) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if isinstance(value, (list, tuple)):
        item_field = f"{field}[]" if field in _ORDER_INSENSITIVE_FIELDS else field
        items = [_normalise_value(item, field=item_field) for item in value]
        if field in _ORDER_INSENSITIVE_FIELDS:
            unique = {json.dumps(item, sort_keys=True): item for item in items}
            return [unique[key] for key in sorted(unique)]
        return items
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not (float("-inf") < value < float("inf")):
            raise ValueError(f"run definition {field} must contain finite numbers")
        if isinstance(value, str) and _has_url_userinfo(value):
            return _safe_sensitive_identity(value)
        if (field.endswith(("_path", "_dir", "_root")) or field in _PATH_FIELDS) and isinstance(
            value, str
        ):
            if not value.strip():
                raise ValueError(f"run definition {field} path must not be blank")
            return str(Path(value).expanduser().resolve(strict=False))
        return value
    raise ValueError(
        f"run definition {field} must contain JSON-shaped values, got {type(value).__name__}"
    )


def _is_sensitive_key(key: str) -> bool:
    compact = "".join(character for character in key.strip().lower() if character.isalnum())
    return any(
        "".join(character for character in part if character.isalnum()) in compact
        for part in _SENSITIVE_KEY_PARTS
    )


def _has_url_userinfo(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return False
    return bool(parsed.scheme and parsed.netloc and (parsed.username or parsed.password))


def _safe_sensitive_identity(value: object) -> object:
    """Hash secret-shaped state so drift remains visible without disclosure."""

    if isinstance(value, Mapping) and set(value) == {"identity_sha256"}:
        candidate = value.get("identity_sha256")
        if _is_tagged_sha256(candidate):
            return {"identity_sha256": str(candidate).lower()}
    canonical = json.dumps(
        _normalise_secret_shape(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {"identity_sha256": f"sha256:{digest}"}


def _is_tagged_sha256(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False
    digest = value.removeprefix("sha256:")
    if len(digest) != 64:
        return False
    try:
        int(digest, 16)
    except ValueError:
        return False
    return True


def _normalise_secret_shape(value: object) -> object:
    if isinstance(value, Path):
        return str(value.expanduser().resolve(strict=False))
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("secret identity keys must be non-empty strings")
            out[key] = _normalise_secret_shape(item)
        return out
    if isinstance(value, (set, frozenset)):
        return sorted(
            (_normalise_secret_shape(item) for item in value),
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    if isinstance(value, (list, tuple)):
        return [_normalise_secret_shape(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not (float("-inf") < value < float("inf")):
            raise ValueError("secret identity must contain finite numbers")
        return value
    raise ValueError("secret identity must contain JSON-shaped values")


def _redact_state_for_status(value: object) -> object:
    """Preserve legacy status shape while removing secret-bearing values."""

    if isinstance(value, Mapping):
        redacted: dict[str, object] = {}
        for key, item in value.items():
            name = str(key)
            if _is_sensitive_key(name):
                redacted[name] = "<redacted>"
            elif name in {"run_id", "source_run_id"} and item is not None:
                try:
                    redacted[name] = _optional_identity(item, field=name)
                except ValueError:
                    redacted[name] = "<invalid>"
            else:
                redacted[name] = _redact_state_for_status(item)
        return redacted
    if isinstance(value, list):
        return [_redact_state_for_status(item) for item in value]
    if isinstance(value, str) and _has_url_userinfo(value):
        return "<redacted>"
    return value


def _load_definition_inputs(source: Mapping[str, object] | Path) -> Mapping[str, object]:
    if isinstance(source, Mapping):
        return source
    if not isinstance(source, Path):
        raise ValueError("run definition inputs must be a mapping or pathlib.Path")
    state_path = source / "pipeline_state.json" if source.is_dir() else source
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read run definition state at {state_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("run definition state must contain a JSON object")
    return payload


def _definition_from_envelope(envelope: Mapping[str, object]) -> RunDefinition:
    contributions = envelope.get("contributions")
    if not isinstance(contributions, Mapping):
        raise ValueError("persisted run definition contributions must be a mapping")
    normalised_contributions: dict[str, object] = {}
    for owner, values in contributions.items():
        if not isinstance(owner, str) or not owner.strip() or not isinstance(values, Mapping):
            raise ValueError("persisted run definition contributors must be named mappings")
        allowed_fields = _CONTRIBUTOR_FIELDS.get(owner)
        if allowed_fields is None:
            raise ValueError(f"unknown persisted run definition contributor {owner!r}")
        unknown_fields = set(values).difference(allowed_fields)
        if unknown_fields:
            raise ValueError(
                f"unknown persisted run definition fields for {owner}: "
                + ", ".join(sorted(str(field) for field in unknown_fields))
            )
        normalised_contributions[owner] = {
            str(field): _normalise_value(item, field=str(field)) for field, item in values.items()
        }
    digest = envelope.get("definition_digest")
    if not isinstance(digest, str):
        raise ValueError("persisted run definition digest must be a string")
    return RunDefinition(
        schema_version=envelope.get("schema_version"),  # type: ignore[arg-type]
        run_id=_optional_identity(envelope.get("run_id"), field="run_id"),
        source_run_id=_optional_identity(envelope.get("source_run_id"), field="source_run_id"),
        definition_digest=digest,
        contributions=normalised_contributions,  # type: ignore[arg-type]
        legacy=envelope.get("legacy"),  # type: ignore[arg-type]
    )


def define_run(effective_inputs: Mapping[str, object] | Path) -> RunDefinition:
    """Project one immutable, non-secret Run Definition from effective inputs.

    Unknown state fields are deliberately excluded. Feature owners add new
    result-affecting inputs by extending their allowlisted contributor here and
    pinning the change with digest tests.
    """

    effective_inputs = _load_definition_inputs(effective_inputs)
    envelope = effective_inputs.get("run_definition")
    if envelope is not None:
        if not isinstance(envelope, Mapping):
            raise ValueError("persisted run_definition envelope must be a mapping")
        return _definition_from_envelope(envelope)
    contributions: dict[str, dict[str, object]] = {}
    for owner, fields in _CONTRIBUTOR_FIELDS.items():
        values: dict[str, object] = {}
        for field in fields:
            if field not in effective_inputs:
                continue
            values[field] = _normalise_value(effective_inputs[field], field=field)
        if values:
            contributions[owner] = values
    frozen = _freeze(contributions)
    assert isinstance(frozen, Mapping)
    persisted_schema = effective_inputs.get("run_definition_schema_version")
    legacy = persisted_schema is None
    if legacy:
        run_id = None
        source_run_id = None
    elif type(persisted_schema) is not int or persisted_schema != _SCHEMA_VERSION:
        raise ValueError("unsupported persisted run definition schema_version")
    else:
        run_id = _optional_identity(effective_inputs.get("run_id"), field="run_id")
        source_run_id = _optional_identity(
            effective_inputs.get("source_run_id"), field="source_run_id"
        )
    digest = _digest(frozen)
    persisted_digest = effective_inputs.get("definition_digest")
    if persisted_digest is not None and persisted_digest != digest:
        raise ValueError("persisted definition digest does not match effective inputs")
    return RunDefinition(
        schema_version=_SCHEMA_VERSION,
        run_id=run_id,
        source_run_id=source_run_id,
        definition_digest=digest,
        contributions=frozen,  # type: ignore[arg-type]
        legacy=legacy,
    )


def _project_requested_definition(
    source: RunDefinition,
    effective_inputs: Mapping[str, object],
    *,
    persisted_state: Mapping[str, object] | None = None,
) -> tuple[RunDefinition, tuple[str, ...]]:
    """Merge explicit inputs into a source definition and report semantic drift."""

    if not isinstance(source, RunDefinition) or not isinstance(effective_inputs, Mapping):
        raise ValueError("source and effective_inputs must be Run Definition values")
    source_inputs = source.input_projection()
    if persisted_state is not None and not source.legacy:
        projected_state = define_run(
            {
                field: persisted_state[field]
                for fields in _CONTRIBUTOR_FIELDS.values()
                for field in fields
                if field in persisted_state and field in source_inputs
            }
        )
        observed = {
            field: _thaw(value)
            for values in projected_state.contributions.values()
            for field, value in values.items()
        }
        for field, expected in source_inputs.items():
            if field in observed and observed[field] != expected:
                raise ValueError(f"pipeline state field {field!r} conflicts with run_definition")
    merged = dict(source_inputs)
    for fields in _CONTRIBUTOR_FIELDS.values():
        for field in fields:
            if field in effective_inputs:
                merged[field] = effective_inputs[field]
    requested = define_run(merged)
    return requested, _drift_fields(source, requested)


def _flatten(contributions: Mapping[str, Mapping[str, object]]) -> dict[str, object]:
    return {
        f"{owner}.{field}": _thaw(value)
        for owner, values in contributions.items()
        for field, value in values.items()
    }


def _drift_fields(source: RunDefinition, requested: RunDefinition) -> tuple[str, ...]:
    source_fields = _flatten(source.contributions)
    requested_fields = _flatten(requested.contributions)
    return tuple(
        key
        for key in sorted(set(source_fields) | set(requested_fields))
        if source_fields.get(key) != requested_fields.get(key)
    )


def _lifecycle(step: str, status: str) -> tuple[LifecycleAction, str | None, str]:
    if step not in _PHASE_ORDER:
        return ("reject", None, "unknown_step")
    if status in {"complete", "partial_complete"}:
        index = _PHASE_ORDER.index(step)
        if index + 1 == len(_PHASE_ORDER):
            return ("finished", None, "pipeline_finished")
        return ("advance_phase", _PHASE_ORDER[index + 1], "pipeline_checkpoint_complete")
    if status == "running":
        return ("rerun_phase", step, "pipeline_checkpoint_running")
    if status == "failed":
        return ("rerun_phase", step, "pipeline_checkpoint_failed")
    if status == "paused":
        return ("rerun_phase", step, "pipeline_checkpoint_paused")
    if status == "interrupted":
        return ("rerun_phase", step, "pipeline_checkpoint_interrupted")
    return ("reject", None, "unknown_status")


def _authoritative_pipeline_state_matches(
    run_root: Path,
    pipeline_state: Mapping[str, object],
    source_definition: RunDefinition,
) -> bool:
    state_path = run_root / "pipeline_state.json"
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(payload, Mapping):
        return False
    for field in ("step", "status", "timestamp"):
        if payload.get(field) != pipeline_state.get(field):
            return False
    expected_logs_dir = pipeline_state.get("logs_dir")
    observed_logs_dir = payload.get("logs_dir")
    if expected_logs_dir is not None or observed_logs_dir is not None:
        try:
            if Path(str(expected_logs_dir)).expanduser().resolve(strict=False) != Path(
                str(observed_logs_dir)
            ).expanduser().resolve(strict=False):
                return False
        except (OSError, RuntimeError):
            return False
    try:
        requested_source = define_run(pipeline_state)
        observed_definition = define_run(payload)
    except ValueError:
        return False
    return requested_source == source_definition and observed_definition == source_definition


def plan_resume(
    source_definition: RunDefinition,
    pipeline_state: Mapping[str, object],
    *,
    run_root: Path,
    requested_inputs: Mapping[str, object] | None = None,
) -> ResumePlan:
    """Return a read-only Resume Plan without accepting any checkpoint.

    Feature checkpoint rows remain ``not_inspected`` because Phase 2 and Phase
    4 validators retain reuse authority. Definition drift is reported without
    claiming a feature-level reuse decision; R1 does not create the Derived Run.
    """

    if not isinstance(source_definition, RunDefinition):
        raise ValueError("source_definition must be a RunDefinition")
    if not isinstance(pipeline_state, Mapping):
        raise ValueError("pipeline_state must be a mapping")
    if not isinstance(run_root, Path):
        raise ValueError("run_root must be a pathlib.Path")
    step = pipeline_state.get("step")
    status = pipeline_state.get("status")
    if not isinstance(step, str) or not step.strip():
        step = ""
    if not isinstance(status, str) or not status.strip():
        status = ""
    requested_definition = (
        source_definition if requested_inputs is None else define_run(requested_inputs)
    )
    drift = _drift_fields(source_definition, requested_definition)
    lifecycle_action, target_step, pipeline_reason = _lifecycle(step, status)
    authoritative_state = _authoritative_pipeline_state_matches(
        run_root,
        pipeline_state,
        source_definition,
    )
    errors: tuple[str, ...] = ()
    if lifecycle_action == "reject":
        mode: ResumeMode = "rejected"
        errors = (pipeline_reason,)
        pipeline_action: CheckpointAction = "reject"
    elif drift:
        mode = "derived_required"
        if authoritative_state:
            pipeline_action = "rerun"
            pipeline_reason = "definition_drift"
        else:
            pipeline_action = "not_inspected"
            pipeline_reason = "definition_drift_pipeline_state_not_verified"
    elif not authoritative_state:
        mode = "legacy" if source_definition.legacy else "exact"
        pipeline_action = "not_inspected"
        pipeline_reason = "pipeline_state_not_verified"
    else:
        mode = "legacy" if source_definition.legacy else "exact"
        pipeline_action = "rerun" if lifecycle_action == "rerun_phase" else "reuse"
    decisions = [
        CheckpointDecision(
            checkpoint_id="pipeline_state",
            owner="pipeline",
            action=pipeline_action,
            reason_code=pipeline_reason,
            path=str((run_root / "pipeline_state.json").resolve(strict=False)),
        )
    ]
    for owner, relative in (("phase_2", "phase_2"), ("phase_4", "phase_4")):
        path = run_root / relative
        if not path.exists():
            continue
        decisions.append(
            CheckpointDecision(
                checkpoint_id=f"{owner}_checkpoints",
                owner=owner,
                action="not_inspected",
                reason_code=(
                    "definition_drift_feature_validator_required"
                    if drift
                    else "feature_validator_required"
                ),
                path=str(path.resolve(strict=False)),
            )
        )
    return ResumePlan(
        schema_version=_SCHEMA_VERSION,
        mode=mode,
        lifecycle_action=lifecycle_action,
        source_run_id=source_definition.run_id,
        source_digest=source_definition.definition_digest,
        requested_digest=requested_definition.definition_digest,
        current_step=step,
        target_step=target_step,
        state_status=status,
        drift_fields=drift,
        checkpoint_decisions=tuple(decisions),
        errors=errors,
    )


__all__ = [
    "CheckpointDecision",
    "ResumePlan",
    "RunDefinition",
    "define_run",
    "plan_resume",
]
