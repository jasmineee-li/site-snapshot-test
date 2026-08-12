"""Atomic, idempotent materialization of isolated Derived Run roots."""

from __future__ import annotations

import fcntl
import hashlib
import json
import uuid
from dataclasses import dataclass
from pathlib import Path

from worldsim.atomic_io import write_json_atomic
from worldsim.run_definition import define_run
from worldsim.run_definition_contracts import RunDefinition, RunTransition
from worldsim.state import initialize_isolated_run_state

_SCHEMA_VERSION = 1
_RESTART_STEP = "phase_0a"
_COLLECTION_NAME = ".warp-derived-runs"
_RESERVATIONS_NAME = ".reservations"
_CHILD_POINTER_NAME = "last_run_state.json"


@dataclass(frozen=True)
class DerivedRunContext:
    """One materialized child root and its persisted lineage."""

    child_root: Path
    definition: RunDefinition
    source_root: Path
    reservation_path: Path
    created: bool
    restart_step: str = _RESTART_STEP


def _canonical(path: Path) -> Path:
    return path.expanduser().resolve(strict=False)


def _source_authority(source_root: Path, expected: RunDefinition) -> tuple[dict, str]:
    state_path = source_root / "pipeline_state.json"
    try:
        raw = state_path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Derived Run source checkpoint is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("Derived Run source checkpoint must contain an object")
    observed = define_run(payload)
    if observed.legacy or observed != expected:
        raise ValueError("Derived Run source authority does not match the requested transition")
    logs_dir = payload.get("logs_dir")
    if logs_dir and _canonical(Path(str(logs_dir))) != source_root:
        raise ValueError("Derived Run source checkpoint belongs to a different state root")
    return payload, hashlib.sha256(raw).hexdigest()


def _confirm_source_unchanged(
    source_root: Path,
    expected: RunDefinition,
    expected_sha256: str,
) -> None:
    """Re-read source authority and require the expected byte identity."""

    _, observed_sha256 = _source_authority(source_root, expected)
    if observed_sha256 != expected_sha256:
        raise ValueError("Derived Run source checkpoint changed during materialization")


def _materialization_key(source_root: Path, source: RunDefinition, requested: RunDefinition) -> str:
    payload = json.dumps(
        {
            "source_root": str(source_root),
            "source_run_id": source.run_id,
            "requested_digest": requested.definition_digest,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_reservation(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("Derived Run reservation is unreadable") from exc
    if not isinstance(payload, dict):
        raise ValueError("Derived Run reservation must contain an object")
    return payload


def _child_definition(requested: RunDefinition, *, child_id: str, source_id: str) -> RunDefinition:
    return RunDefinition(
        schema_version=requested.schema_version,
        run_id=child_id,
        source_run_id=source_id,
        definition_digest=requested.definition_digest,
        contributions=requested.contributions,
        legacy=False,
    )


def _validate_reservation(
    reservation: dict,
    *,
    source_root: Path,
    source: RunDefinition,
    requested: RunDefinition,
    source_state_sha256: str,
) -> tuple[Path, RunDefinition]:
    expected = {
        "schema_version": _SCHEMA_VERSION,
        "source_root": str(source_root),
        "source_run_id": source.run_id,
        "source_definition_digest": source.definition_digest,
        "requested_definition_digest": requested.definition_digest,
        "source_state_sha256": source_state_sha256,
        "restart_step": _RESTART_STEP,
    }
    for field, value in expected.items():
        if reservation.get(field) != value:
            raise ValueError(f"Derived Run reservation field {field!r} does not match")
    child_root_raw = reservation.get("child_root")
    definition_raw = reservation.get("run_definition")
    if not isinstance(child_root_raw, str) or not isinstance(definition_raw, dict):
        raise ValueError("Derived Run reservation is missing child identity")
    definition = define_run({"run_definition": definition_raw})
    if definition.source_run_id != source.run_id:
        raise ValueError("Derived Run reservation has invalid lineage")
    if definition.definition_digest != requested.definition_digest:
        raise ValueError("Derived Run reservation has the wrong requested definition")
    child_root = _canonical(Path(child_root_raw))
    if child_root.name != definition.run_id:
        raise ValueError("Derived Run child path does not match its Run ID")
    return child_root, definition


def _initialize_child(
    child_root: Path,
    *,
    definition: RunDefinition,
    reservation: dict,
) -> bool:
    if child_root.is_symlink():
        raise ValueError("Derived Run child root must not be a symlink")
    if child_root.exists() and not child_root.is_dir():
        raise ValueError("Derived Run child root must be a directory")
    child_root.mkdir(parents=True, exist_ok=True)
    manifest_path = child_root / "derived_run.json"
    state_path = child_root / "pipeline_state.json"
    pointer_path = child_root / _CHILD_POINTER_NAME
    if manifest_path.is_symlink() or state_path.is_symlink() or pointer_path.is_symlink():
        raise ValueError("Derived Run child metadata must not be symlinked")
    created = not state_path.exists()
    manifest_valid = False
    if manifest_path.exists():
        existing = _load_reservation(manifest_path)
        if existing != reservation:
            raise ValueError("Derived Run manifest does not match its reservation")
        manifest_valid = True
    state_valid = False
    if state_path.exists():
        try:
            payload = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("Derived Run child checkpoint is unreadable") from exc
        if not isinstance(payload, dict) or define_run(payload) != definition:
            raise ValueError("Derived Run child checkpoint has conflicting identity")
        logs_dir = payload.get("logs_dir")
        if not logs_dir or _canonical(Path(str(logs_dir))) != child_root:
            raise ValueError("Derived Run child checkpoint belongs to a different state root")
        state_valid = True

    # Once both immutable lineage records and the authoritative checkpoint are
    # valid, this is an established child Run. Normal phase/root artifacts are
    # expected and must survive idempotent derive-and-resume retries.
    if manifest_valid and state_valid:
        return False

    # Fresh and crash-incomplete roots have not established both identities;
    # fail closed instead of adopting arbitrary artifacts into the child.
    unexpected = {
        path.name
        for path in child_root.iterdir()
        if path not in {manifest_path, state_path, pointer_path}
    }
    if unexpected:
        raise ValueError(
            "Derived Run child root contains unrelated files: " + ", ".join(sorted(unexpected))
        )
    if not manifest_valid:
        write_json_atomic(
            manifest_path,
            reservation,
            failpoint_base="run_materialization.child_manifest",
        )
    if state_valid:
        return False
    initialize_isolated_run_state(
        child_root,
        definition,
        step=_RESTART_STEP,
        status="failed",
        reason="derived_run_materialized",
    )
    return created


def materialize_derived_run(
    source_root: Path,
    transition: RunTransition,
    *,
    collection_root: Path | None = None,
) -> DerivedRunContext:
    """Create or recover one isolated child for an identified drift request."""

    if not isinstance(transition, RunTransition) or transition.kind != "derived_required":
        raise ValueError("materialization requires a derived_required transition")
    source = transition.source_definition
    requested = transition.definition
    if source is None or source.legacy or source.run_id is None or requested is None:
        raise ValueError("materialization requires an identified source and requested definition")
    raw_source_root = source_root.expanduser()
    if raw_source_root.is_symlink():
        raise ValueError("Derived Run source root must not be a symlink")
    source_root = _canonical(raw_source_root)
    if not requested.input_projection().get("benchmark_path"):
        raise ValueError("Derived Run restart from Phase 0a requires benchmark_path")

    raw_collection = (collection_root or (source_root.parent / _COLLECTION_NAME)).expanduser()
    if raw_collection.exists() and raw_collection.is_symlink():
        raise ValueError("Derived Run collection must not be a symlink")
    collection = _canonical(raw_collection)
    if collection == source_root or collection.is_relative_to(source_root):
        raise ValueError("Derived Run collection must be outside the source root")
    collection.mkdir(parents=True, exist_ok=True)

    key = _materialization_key(source_root, source, requested)
    reservation_dir = collection / key
    reservations_dir = collection / _RESERVATIONS_NAME
    if reservations_dir.exists() and reservations_dir.is_symlink():
        raise ValueError("Derived Run reservation store must not be a symlink")
    reservations_dir.mkdir(exist_ok=True)
    reservation_path = reservations_dir / f"{key}.json"
    lock_path = collection / ".materialization.lock"
    if reservation_path.is_symlink() or lock_path.is_symlink():
        raise ValueError("Derived Run reservation metadata must not be symlinked")
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            _, source_state_sha256 = _source_authority(source_root, source)
            reservation = _load_reservation(reservation_path)
            new_reservation = reservation is None
            if new_reservation:
                if reservation_dir.exists():
                    raise ValueError("Derived Run reservation directory is incomplete")
                child_id = f"run-{uuid.uuid4().hex}"
                definition = _child_definition(
                    requested,
                    child_id=child_id,
                    source_id=source.run_id,
                )
                child_root = reservation_dir / child_id
                reservation = {
                    "schema_version": _SCHEMA_VERSION,
                    "source_root": str(source_root),
                    "source_run_id": source.run_id,
                    "source_definition_digest": source.definition_digest,
                    "requested_definition_digest": requested.definition_digest,
                    "source_state_sha256": source_state_sha256,
                    "restart_step": _RESTART_STEP,
                    "child_root": str(child_root),
                    "run_definition": definition.to_dict(),
                }
            child_root, definition = _validate_reservation(
                reservation,
                source_root=source_root,
                source=source,
                requested=requested,
                source_state_sha256=source_state_sha256,
            )
            if child_root.parent != reservation_dir:
                raise ValueError("Derived Run child root escapes its reservation")
            if reservation_dir.is_symlink():
                raise ValueError("Derived Run reservation directory must not be a symlink")
            if reservation_dir.exists():
                unexpected = {path for path in reservation_dir.iterdir() if path != child_root}
                if unexpected:
                    raise ValueError("Derived Run reservation directory contains unrelated files")

            _confirm_source_unchanged(source_root, source, source_state_sha256)
            if new_reservation:
                write_json_atomic(
                    reservation_path,
                    reservation,
                    failpoint_base="run_materialization.reservation",
                )
                try:
                    _confirm_source_unchanged(source_root, source, source_state_sha256)
                except ValueError as source_error:
                    # The new reservation is not accepted until this post-write
                    # check succeeds. Roll back only while no child directory
                    # exists and the file still contains our exact reservation;
                    # later source writes remain outside this narrow protocol.
                    observed = _load_reservation(reservation_path)
                    if observed != reservation or reservation_dir.exists():
                        raise ValueError(
                            "Derived Run source changed while reservation acceptance "
                            "could not be rolled back safely"
                        ) from source_error
                    reservation_path.unlink()
                    raise
            if not reservation_dir.exists():
                reservation_dir.mkdir(parents=False)
            created = _initialize_child(
                child_root,
                definition=definition,
                reservation=reservation,
            )
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return DerivedRunContext(
        child_root=child_root,
        definition=definition,
        source_root=source_root,
        reservation_path=reservation_path,
        created=created,
    )


__all__ = ["DerivedRunContext", "materialize_derived_run"]
