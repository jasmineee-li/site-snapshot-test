"""Phase 0c artifact, provenance, and trace helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from worldsim.failpoints import crash_if_enabled

PROFILE_METADATA_PREFIX = "PROFILE_METADATA_"
TIER_METADATA_PREFIX = "TIER_METADATA_"
PHASE_0C_TRACE = "PHASE_0C_TRACE.jsonl"
PHASE_0C_TIMINGS = "PHASE_0C_TIMINGS.json"
REACHABILITY_REPORT = "REACHABILITY_REPORT.json"


def write_text_atomic(path: Path, text: str) -> None:
    """Atomically replace *path* with *text*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        crash_if_enabled("phase_0c.outputs.before_replace")
        os.replace(tmp_path, path)
        crash_if_enabled("phase_0c.outputs.after_replace")
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def write_json_atomic(path: Path, payload: object) -> None:
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def profile_metadata_path(output_dir: Path, site_name: str) -> Path:
    return output_dir / f"{PROFILE_METADATA_PREFIX}{site_name}.json"


def reachability_report_path(output_dir: Path) -> Path:
    return output_dir / REACHABILITY_REPORT


def phase_0c_trace_path(output_dir: Path) -> Path:
    return output_dir / PHASE_0C_TRACE


def phase_0c_timings_path(output_dir: Path) -> Path:
    return output_dir / PHASE_0C_TIMINGS


def tier_metadata_path(output_dir: Path, site_name: str, tier_name: str) -> Path:
    return output_dir / f"{TIER_METADATA_PREFIX}{site_name}_{tier_name}.json"


def tier_artifact_path(output_dir: Path, site_name: str, artifact_stem: str) -> Path:
    return output_dir / f"{artifact_stem}_{site_name}.json"


def sidecar_artifact_path(output_dir: Path, site_name: str, artifact_stem: str) -> Path:
    return output_dir / f"{artifact_stem}_{site_name}.json"


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_json(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return text_sha256(canonical)


def hash_file_list(file_list: Iterable[str], benchmark_root: Path) -> str:
    """Hash relative file paths and contents for a staged Phase 0c site."""
    resolved_root = Path(benchmark_root).resolve()
    hasher = hashlib.sha256()
    for raw_path in sorted(str(path) for path in file_list):
        path = Path(raw_path).resolve()
        rel = path.relative_to(resolved_root).as_posix()
        hasher.update(rel.encode("utf-8"))
        hasher.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        hasher.update(b"\0")
    return hasher.hexdigest()


def hash_routed_inputs(routed_inputs: Mapping[str, str]) -> str:
    """Hash remote paths and local file contents for generated input indexes."""
    hasher = hashlib.sha256()
    for remote_path, local_path in sorted(routed_inputs.items()):
        local = Path(local_path)
        hasher.update(remote_path.encode("utf-8"))
        hasher.update(b"\0")
        if local.is_file():
            with local.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
        hasher.update(b"\0")
    return hasher.hexdigest()


def build_tier_metadata(
    *,
    site_name: str,
    tier_name: str,
    prompt_name: str,
    prompt_hash: str,
    validation_command: str,
    output_path: str,
    sandbox_model: str,
    benchmark_digest: str,
    manifest_eval_types: Iterable[str],
    instance_site_url: str | None,
    host_inventory_instance_fingerprint: str | None,
    verification_proxy: object,
    evidence_index_digest: str | None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "site_name": site_name,
        "tier_name": tier_name,
        "prompt_name": prompt_name,
        "prompt_sha256": prompt_hash,
        "validation_command": validation_command,
        "output_path": output_path,
        "sandbox_model": sandbox_model,
        "benchmark_digest": benchmark_digest,
        "manifest_eval_types": sorted({str(value) for value in manifest_eval_types if value}),
        "instance_site_url": instance_site_url,
        "host_inventory_instance_fingerprint": host_inventory_instance_fingerprint,
        "verification_proxy": verification_proxy,
        "evidence_index_digest": evidence_index_digest,
    }
    if extra:
        metadata.update(dict(extra))
    return metadata


def load_reusable_tier_output(
    *,
    output_dir: Path,
    site_name: str,
    tier_name: str,
    artifact_stem: str,
    expected_metadata: Mapping[str, Any],
    validate_parsed: Callable[[object], list[str]],
) -> Any | None:
    """Load a tier artifact only when metadata and validation still match."""
    artifact_path = tier_artifact_path(output_dir, site_name, artifact_stem)
    metadata_path = tier_metadata_path(output_dir, site_name, tier_name)
    if not (artifact_path.exists() and metadata_path.exists()):
        return None

    try:
        raw = artifact_path.read_text(encoding="utf-8")
        parsed = json.loads(raw)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None

    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            return None
    if metadata.get("artifact_sha256") != text_sha256(raw):
        return None
    if validate_parsed(parsed):
        return None
    return parsed


def publish_tier_output(
    *,
    output_dir: Path,
    site_name: str,
    tier_name: str,
    artifact_stem: str,
    payload: object,
    metadata: Mapping[str, Any],
    sandbox_outputs: Mapping[str, Any] | None = None,
) -> None:
    raw = json.dumps(payload, indent=2)
    artifact_path = tier_artifact_path(output_dir, site_name, artifact_stem)
    write_text_atomic(artifact_path, raw)
    full_metadata = dict(metadata)
    full_metadata.update(
        {
            "artifact_path": artifact_path.name,
            "artifact_sha256": text_sha256(raw),
        }
    )
    telemetry = sandbox_outputs.get("_telemetry") if sandbox_outputs else None
    if telemetry is not None:
        if isinstance(telemetry, str):
            try:
                full_metadata["sandbox_telemetry"] = json.loads(telemetry)
            except json.JSONDecodeError:
                full_metadata["sandbox_telemetry"] = telemetry
        else:
            full_metadata["sandbox_telemetry"] = telemetry
    summary = sandbox_outputs.get("_summary") if sandbox_outputs else None
    if summary:
        try:
            full_metadata["sandbox_summary"] = json.loads(str(summary))
        except json.JSONDecodeError:
            full_metadata["sandbox_summary"] = summary
    write_json_atomic(tier_metadata_path(output_dir, site_name, tier_name), full_metadata)


def publish_json_sidecar(
    *,
    output_dir: Path,
    site_name: str,
    artifact_stem: str,
    raw_text: str,
) -> bool:
    """Publish an optional JSON sidecar; return False when it is malformed."""
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError:
        return False
    write_json_atomic(sidecar_artifact_path(output_dir, site_name, artifact_stem), payload)
    return True


class Phase0cTraceWriter:
    """Append-only Phase 0c trace writer with a compact timing summary."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.started_monotonic = time.monotonic()
        self.started_wall = time.time()
        self._lock = threading.Lock()
        self._records: list[dict[str, Any]] = []
        trace_path = phase_0c_trace_path(self.output_dir)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text("", encoding="utf-8")

    def record(self, event: str, **fields: Any) -> None:
        now_monotonic = time.monotonic()
        record = {
            "schema_version": 1,
            "event": event,
            "wall_time": time.time(),
            "elapsed_ms": int((now_monotonic - self.started_monotonic) * 1000),
            **fields,
        }
        line = json.dumps(record, sort_keys=True, default=str)
        with self._lock:
            self._records.append(record)
            with phase_0c_trace_path(self.output_dir).open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def write_timings_summary(self, *, failures: Iterable[str] = ()) -> None:
        completed_wall = time.time()
        tier_records = [
            record
            for record in self._records
            if record.get("event") in {"tier_generated", "tier_reused", "tier_failed"}
        ]
        payload = {
            "schema_version": 1,
            "started_wall": self.started_wall,
            "completed_wall": completed_wall,
            "elapsed_ms": int((completed_wall - self.started_wall) * 1000),
            "event_counts": _event_counts(self._records),
            "tier_records": tier_records,
            "failures": list(failures),
        }
        write_json_atomic(phase_0c_timings_path(self.output_dir), payload)


def _event_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        event = str(record.get("event") or "unknown")
        counts[event] = counts.get(event, 0) + 1
    return dict(sorted(counts.items()))
