#!/usr/bin/env python3
"""Export compact Phase 4 audit artifacts from local or remote run directories."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# This script is streamed to host system Python during remote exports. r5 has
# Python 3.10, so use timezone.utc instead of the Python 3.11 datetime.UTC alias.
_UTC = timezone.utc  # noqa: UP017

COMPACT_PATTERNS = (
    "pipeline_state.json",
    "artifact_manifest.json",
    "phase_2/adversarial_tasks.json",
    "phase_2/exposure_contracts.json",
    "phase_2/feasibility_report.json",
    "phase_2/adversarial_tasks.infeasible.json",
    "phase_2/adversarial_tasks.dropped_source_data.json",
    "phase_2/dropped_no_contract.json",
    "phase_2/exposure_ineligible.json",
    "phase_4/results.json",
    "phase_4/summary.txt",
    "phase_4/variant_audit.txt",
    "phase_4/*/*/processed_result.json",
    "phase_4/*/*/result.json",
    "phase_4/*/*/final_response.json",
    "phase_4/*/*/pvpo/*.json",
    "phase_4/*/*/variant_generation/*/*/request_summary.json",
    "phase_4/*/*/variant_generation/*/*/host_validation.json",
    "phase_4/*/*/variant_generation/*/*/failure_context.json",
    "phase_4/*/*/variant_generation/*/*/payload_diff.json",
    "phase_4/*/*/variant_generation/*/*/contract_qa.json",
)
PHASE3_CONTRACT_PATTERNS = ("phase_3/contracts.json",)
NETWORK_TRACE_PATTERNS = ("phase_4/*/*/network_trace.json",)

EXCLUDED_NAMES = {
    "history.json",
    "needham_trace.json",
    "needham_trace.xml",
    "redacted_prompt_input.json",
    "prompt_input.json",
}
EXCLUDED_PARTS = {"screenshots", "videos", "video", "recordings"}
EXCLUDED_SUFFIXES = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".mp4",
    ".png",
    ".webm",
    ".webp",
    ".zip",
}


@dataclass(frozen=True)
class PlannedFile:
    path: Path
    archive_path: str
    size: int
    sha256: str


def utc_timestamp() -> str:
    return datetime.now(_UTC).strftime("%Y%m%dT%H%M%SZ")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_relative_path(raw: str, *, label: str) -> Path:
    value = raw.strip()
    if not value:
        raise ValueError(f"{label} cannot be empty")
    path = Path(value)
    if path.is_absolute():
        raise ValueError(f"{label} must be relative to the remote checkout: {raw}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{label} contains an unsafe path component: {raw}")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_excluded(path: Path) -> bool:
    if path.name in EXCLUDED_NAMES:
        return True
    if path.suffix.lower() in EXCLUDED_SUFFIXES:
        return True
    return bool(EXCLUDED_PARTS.intersection(path.parts))


def run_dirs_from_sweep_state(path: Path, *, include_failed: bool = False) -> list[str]:
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    run_dirs: list[str] = []
    for row in data.get("completed_runs") or []:
        if isinstance(row, dict) and isinstance(row.get("run_dir"), str):
            run_dirs.append(row["run_dir"])
    for row in data.get("records") or []:
        if not isinstance(row, dict) or not isinstance(row.get("run_dir"), str):
            continue
        if include_failed or row.get("status") == "completed":
            run_dirs.append(row["run_dir"])
    return run_dirs


def collect_run_dirs(
    explicit_run_dirs: list[str],
    sweep_states: list[Path],
    *,
    include_failed: bool = False,
) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in [
        *explicit_run_dirs,
        *[
            item
            for state_path in sweep_states
            for item in run_dirs_from_sweep_state(state_path, include_failed=include_failed)
        ],
    ]:
        safe = _safe_relative_path(raw, label="run_dir").as_posix()
        if safe not in seen:
            seen.add(safe)
            ordered.append(safe)
    if not ordered:
        raise ValueError("at least one --run-dir or --sweep-state with completed runs is required")
    return ordered


def build_manifest(
    root: Path,
    run_dirs: list[str],
    *,
    max_file_bytes: int,
    include_phase3_contracts: bool = False,
    include_network_traces: bool = False,
    network_trace_task_ids: set[str] | None = None,
    summarize_network_traces: bool = False,
) -> dict[str, Any]:
    root = root.resolve(strict=False)
    if max_file_bytes <= 0:
        raise ValueError("--max-file-bytes must be positive")
    network_trace_task_ids = network_trace_task_ids or set()

    files: dict[str, PlannedFile] = {}
    skipped: list[dict[str, Any]] = []
    missing_runs: list[str] = []
    network_trace_summaries: list[dict[str, Any]] = []

    patterns = [
        *COMPACT_PATTERNS,
        *(PHASE3_CONTRACT_PATTERNS if include_phase3_contracts else ()),
        *(NETWORK_TRACE_PATTERNS if include_network_traces else ()),
    ]

    for raw_run_dir in run_dirs:
        run_dir = _safe_relative_path(raw_run_dir, label="run_dir")
        run_root = root / run_dir
        if not run_root.exists():
            missing_runs.append(run_dir.as_posix())
            continue
        for pattern in patterns:
            for candidate in run_root.glob(pattern):
                if not candidate.is_file() or candidate.is_symlink():
                    continue
                rel_to_run = candidate.relative_to(run_root)
                if (
                    pattern in NETWORK_TRACE_PATTERNS
                    and network_trace_task_ids
                    and not _network_trace_matches_task(rel_to_run, network_trace_task_ids)
                ):
                    continue
                if _is_excluded(rel_to_run):
                    skipped.append(
                        {
                            "path": candidate.relative_to(root).as_posix(),
                            "reason": "excluded_by_policy",
                        }
                    )
                    continue
                size = candidate.stat().st_size
                archive_path = candidate.relative_to(root).as_posix()
                if size > max_file_bytes:
                    skipped.append(
                        {
                            "path": archive_path,
                            "reason": "over_max_file_bytes",
                            "size": size,
                        }
                    )
                    continue
                files[archive_path] = PlannedFile(
                    path=candidate,
                    archive_path=archive_path,
                    size=size,
                    sha256=_sha256(candidate),
                )

        if summarize_network_traces:
            for pattern in NETWORK_TRACE_PATTERNS:
                for candidate in run_root.glob(pattern):
                    if not candidate.is_file() or candidate.is_symlink():
                        continue
                    rel_to_run = candidate.relative_to(run_root)
                    if network_trace_task_ids and not _network_trace_matches_task(
                        rel_to_run, network_trace_task_ids
                    ):
                        continue
                    network_trace_summaries.append(
                        summarize_network_trace(
                            candidate, archive_path=candidate.relative_to(root).as_posix()
                        )
                    )

    ordered_files = [files[key] for key in sorted(files)]
    return {
        "schema_version": "phase4_compact_artifact_export_v1",
        "created_at": datetime.now(_UTC).isoformat(),
        "root": str(root),
        "run_dirs": run_dirs,
        "patterns": patterns,
        "max_file_bytes": max_file_bytes,
        "include_phase3_contracts": include_phase3_contracts,
        "include_network_traces": include_network_traces,
        "network_trace_task_ids": sorted(network_trace_task_ids),
        "summarize_network_traces": summarize_network_traces,
        "network_trace_summaries": sorted(
            network_trace_summaries,
            key=lambda row: str(row.get("path") or ""),
        ),
        "file_count": len(ordered_files),
        "total_bytes": sum(item.size for item in ordered_files),
        "files": [
            {
                "path": item.archive_path,
                "size": item.size,
                "sha256": item.sha256,
            }
            for item in ordered_files
        ],
        "skipped": skipped,
        "missing_runs": missing_runs,
    }


def _network_trace_matches_task(rel_to_run: Path, task_ids: set[str]) -> bool:
    parts = rel_to_run.parts
    if len(parts) < 4 or parts[0] != "phase_4" or parts[-1] != "network_trace.json":
        return False
    task_dir = parts[2]
    return any(
        task_dir == task_id
        or task_dir.startswith(f"{task_id}_variant_")
        or task_dir.startswith(f"{task_id}__")
        for task_id in task_ids
    )


def _base_task_id(task_dir: str) -> str:
    for marker in ("_variant_", "__"):
        if marker in task_dir:
            return task_dir.split(marker, 1)[0]
    return task_dir


def summarize_network_trace(path: Path, *, archive_path: str) -> dict[str, Any]:
    try:
        data = _load_json(path)
    except Exception as exc:
        return {
            "path": archive_path,
            "status": "error",
            "error": str(exc),
            "size": path.stat().st_size if path.exists() else None,
        }
    if not isinstance(data, list):
        return {
            "path": archive_path,
            "status": "error",
            "error": "network trace is not a JSON array",
            "size": path.stat().st_size,
        }
    rel = Path(archive_path)
    task_dir = rel.parent.name
    urls = [str(item.get("url") or "") for item in data if isinstance(item, dict)]
    document_events = [
        item
        for item in data
        if isinstance(item, dict)
        and (item.get("is_document_load") or item.get("resource_type") == "document")
    ]
    document_urls = [str(item.get("url") or "") for item in document_events]
    status_counts = Counter(
        str(item.get("response_status"))
        for item in data
        if isinstance(item, dict) and item.get("response_status") is not None
    )
    resource_counts = Counter(
        str(item.get("resource_type") or "unknown") for item in data if isinstance(item, dict)
    )
    host_counts = Counter(urlparse(url).netloc for url in urls if url)
    return {
        "path": archive_path,
        "status": "ok",
        "size": path.stat().st_size,
        "task_dir": task_dir,
        "task_id": _base_task_id(task_dir),
        "event_count": len(data),
        "document_load_count": len(document_events),
        "first_document_url": document_urls[0] if document_urls else None,
        "last_document_url": document_urls[-1] if document_urls else None,
        "document_urls": document_urls[:12],
        "attacker_invalid_requested": any("attacker.invalid" in url for url in urls),
        "attacker_invalid_document_requested": any(
            "attacker.invalid" in url for url in document_urls
        ),
        "status_counts": dict(sorted(status_counts.items())),
        "resource_counts": dict(sorted(resource_counts.items())),
        "host_counts": dict(host_counts.most_common(12)),
    }


def write_tar(root: Path, manifest: dict[str, Any], output: io.BufferedIOBase) -> None:
    root = root.resolve(strict=False)
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
        manifest_info = tarfile.TarInfo("artifact_export_manifest.json")
        manifest_info.size = len(manifest_bytes)
        manifest_info.mtime = int(datetime.now(_UTC).timestamp())
        archive.addfile(manifest_info, io.BytesIO(manifest_bytes))

        for row in manifest.get("files", []):
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                continue
            rel_path = _safe_relative_path(row["path"], label="archive path")
            archive.add(root / rel_path, arcname=rel_path.as_posix(), recursive=False)


def extract_tar_bytes(payload: bytes, output_dir: Path, *, overwrite: bool = False) -> None:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output_dir} already exists and is not empty; pass --overwrite intentionally"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve(strict=False)
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            member_path = Path(member.name)
            if member_path.is_absolute() or any(
                part in {"", ".", ".."} for part in member_path.parts
            ):
                raise ValueError(f"unsafe tar member path: {member.name}")
            target = (root / member_path).resolve(strict=False)
            if not str(target).startswith(str(root)):
                raise ValueError(f"tar member escapes output directory: {member.name}")
        archive.extractall(root, filter="data")


def _parse_host_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line or line.startswith(" ") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().strip("'\"")
    host = values.get("advertise_host") or values.get("orchestrator_host")
    if not host:
        raise ValueError(f"{path} missing advertise_host/orchestrator_host")
    return {
        "host": host,
        "ssh_user": values.get("ssh_user") or "ubuntu",
        "compose_dir_remote": values.get("compose_dir_remote") or "/home/ubuntu",
    }


def _ssh_key_path(raw: str | None) -> str:
    value = raw or os.environ.get("SSH_KEY") or "~/.ssh/webarena-key.pem"
    return str(Path(os.path.expandvars(os.path.expanduser(value))).resolve(strict=False))


def _remote_dir_from_config(host_config: dict[str, str], remote_dir: str | None) -> str:
    if remote_dir:
        return remote_dir
    return f"{host_config['compose_dir_remote'].rstrip('/')}/browser-sim"


def _remote_script_args(
    *,
    remote_root: str,
    run_dirs: list[str],
    max_file_bytes: int,
    emit_tar: bool,
    dry_run: bool,
    include_phase3_contracts: bool,
    include_network_traces: bool,
    network_trace_task_ids: set[str],
    summarize_network_traces: bool,
) -> list[str]:
    args = [
        "--root",
        remote_root,
        "--max-file-bytes",
        str(max_file_bytes),
    ]
    for run_dir in run_dirs:
        args.extend(["--run-dir", run_dir])
    if emit_tar:
        args.append("--emit-tar")
    if dry_run:
        args.append("--dry-run")
    if include_phase3_contracts:
        args.append("--include-phase3-contracts")
    if include_network_traces:
        args.append("--include-network-traces")
    if summarize_network_traces:
        args.append("--summarize-network-traces")
    for task_id in sorted(network_trace_task_ids):
        args.extend(["--network-trace-task-id", task_id])
    return args


def run_remote_export(
    *,
    host_config_path: Path,
    remote_dir: str | None,
    ssh_key: str | None,
    run_dirs: list[str],
    output_dir: Path | None,
    max_file_bytes: int,
    dry_run: bool,
    overwrite: bool,
    include_phase3_contracts: bool,
    include_network_traces: bool,
    network_trace_task_ids: set[str],
    summarize_network_traces: bool,
) -> int:
    config = _parse_host_config(host_config_path)
    remote_root = _remote_dir_from_config(config, remote_dir)
    target = f"{config['ssh_user']}@{config['host']}"
    ssh_args = [
        "ssh",
        "-i",
        _ssh_key_path(ssh_key),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=120",
        "-o",
        "ConnectTimeout=15",
        target,
        "python3",
        "-",
        *_remote_script_args(
            remote_root=remote_root,
            run_dirs=run_dirs,
            max_file_bytes=max_file_bytes,
            emit_tar=not dry_run,
            dry_run=dry_run,
            include_phase3_contracts=include_phase3_contracts,
            include_network_traces=include_network_traces,
            network_trace_task_ids=network_trace_task_ids,
            summarize_network_traces=summarize_network_traces,
        ),
    ]
    source = Path(__file__).read_bytes()
    completed = subprocess.run(
        ssh_args,
        input=source,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        sys.stderr.write(completed.stderr.decode("utf-8", errors="replace"))
        return completed.returncode
    if completed.stderr:
        sys.stderr.write(completed.stderr.decode("utf-8", errors="replace"))
    if dry_run:
        sys.stdout.write(completed.stdout.decode("utf-8"))
        return 0
    if output_dir is None:
        raise ValueError("--output-dir is required for remote export")
    extract_tar_bytes(completed.stdout, output_dir, overwrite=overwrite)
    print(f"exported compact Phase 4 artifacts to {output_dir}")
    return 0


def run_local_export(
    *,
    root: Path,
    run_dirs: list[str],
    output_dir: Path | None,
    max_file_bytes: int,
    dry_run: bool,
    emit_tar: bool,
    overwrite: bool,
    include_phase3_contracts: bool,
    include_network_traces: bool,
    network_trace_task_ids: set[str],
    summarize_network_traces: bool,
) -> int:
    manifest = build_manifest(
        root,
        run_dirs,
        max_file_bytes=max_file_bytes,
        include_phase3_contracts=include_phase3_contracts,
        include_network_traces=include_network_traces,
        network_trace_task_ids=network_trace_task_ids,
        summarize_network_traces=summarize_network_traces,
    )
    if dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    if emit_tar:
        write_tar(root, manifest, sys.stdout.buffer)
        return 0
    if output_dir is None:
        raise ValueError("--output-dir is required unless --dry-run or --emit-tar is used")
    buffer = io.BytesIO()
    write_tar(root, manifest, buffer)
    extract_tar_bytes(buffer.getvalue(), output_dir, overwrite=overwrite)
    print(f"exported compact Phase 4 artifacts to {output_dir}")
    return 0


def default_output_dir() -> Path:
    return Path("logs") / "phase4_artifact_exports" / f"export_{utc_timestamp()}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-config", type=Path, default=None, help="Remote host YAML.")
    parser.add_argument("--remote-dir", default=None, help="Remote checkout directory.")
    parser.add_argument("--ssh-key", default=None, help="SSH key path.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Local checkout/log root for local exports. Omit when using --host-config.",
    )
    parser.add_argument("--run-dir", action="append", default=[], help="Relative run dir.")
    parser.add_argument(
        "--sweep-state",
        type=Path,
        action="append",
        default=[],
        help="Local sweep_state.json; can be repeated.",
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include failed records from sweep_state.json.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Local destination for extracted compact artifacts.",
    )
    parser.add_argument(
        "--max-file-bytes",
        type=int,
        default=5_000_000,
        help="Skip individual files larger than this many bytes.",
    )
    parser.add_argument(
        "--include-phase3-contracts",
        action="store_true",
        help="Include large Phase 3 contracts.json files.",
    )
    parser.add_argument(
        "--include-network-traces",
        action="store_true",
        help="Include network_trace.json files; these can be tens of MB per task.",
    )
    parser.add_argument(
        "--network-trace-task-id",
        action="append",
        default=[],
        help=(
            "When including network traces, limit them to this task id and its "
            "variant/retry task directories. Repeatable."
        ),
    )
    parser.add_argument(
        "--summarize-network-traces",
        action="store_true",
        help="Add compact network trace summaries to the export manifest.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow non-empty output dir.")
    parser.add_argument("--dry-run", action="store_true", help="Print the remote/local manifest.")
    parser.add_argument(
        "--emit-tar",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_dirs = collect_run_dirs(
            args.run_dir,
            args.sweep_state,
            include_failed=args.include_failed,
        )
        if args.host_config:
            return run_remote_export(
                host_config_path=args.host_config,
                remote_dir=args.remote_dir,
                ssh_key=args.ssh_key,
                run_dirs=run_dirs,
                output_dir=args.output_dir or (None if args.dry_run else default_output_dir()),
                max_file_bytes=args.max_file_bytes,
                dry_run=args.dry_run,
                overwrite=args.overwrite,
                include_phase3_contracts=args.include_phase3_contracts,
                include_network_traces=args.include_network_traces,
                network_trace_task_ids=set(args.network_trace_task_id),
                summarize_network_traces=args.summarize_network_traces,
            )
        return run_local_export(
            root=args.root or Path.cwd(),
            run_dirs=run_dirs,
            output_dir=args.output_dir
            or (None if args.dry_run or args.emit_tar else default_output_dir()),
            max_file_bytes=args.max_file_bytes,
            dry_run=args.dry_run,
            emit_tar=args.emit_tar,
            overwrite=args.overwrite,
            include_phase3_contracts=args.include_phase3_contracts,
            include_network_traces=args.include_network_traces,
            network_trace_task_ids=set(args.network_trace_task_id),
            summarize_network_traces=args.summarize_network_traces,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
