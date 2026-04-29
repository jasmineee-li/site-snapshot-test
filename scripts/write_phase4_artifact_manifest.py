#!/usr/bin/env python3
"""Write a provenance manifest for Phase 4 input artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_ARTIFACTS: tuple[str, ...] = (
    "phase_0c",
    "phase_2",
    "phase_3",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(args: list[str], *, cwd: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=cwd,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def file_record(path: Path, *, base: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.relative_to(base)),
        "size": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, UTC).isoformat(),
        "sha256": sha256_file(path),
    }


def collect_artifact_records(state_dir: Path, rel_path: str) -> dict[str, Any]:
    path = state_dir / rel_path
    record: dict[str, Any] = {
        "path": rel_path,
        "exists": path.exists(),
        "kind": "missing",
        "files": [],
    }
    if not path.exists():
        return record
    if path.is_file():
        record["kind"] = "file"
        record["files"] = [file_record(path, base=state_dir)]
        return record
    if path.is_dir():
        files = [
            file_record(candidate, base=state_dir)
            for candidate in sorted(path.rglob("*"))
            if candidate.is_file()
        ]
        record["kind"] = "directory"
        record["files"] = files
        record["file_count"] = len(files)
        aggregate = hashlib.sha256()
        for item in files:
            aggregate.update(str(item["path"]).encode())
            aggregate.update(str(item["sha256"]).encode())
        record["tree_sha256"] = aggregate.hexdigest()
        return record
    record["kind"] = "other"
    return record


def build_manifest(
    *,
    state_dir: Path,
    artifacts_source: str | None,
    instances_path: Path | None,
    repo_root: Path,
) -> dict[str, Any]:
    state_dir = state_dir.resolve(strict=False)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "phase4_artifact_manifest",
        "generated_at": datetime.now(UTC).isoformat(),
        "state_dir": str(state_dir),
        "artifacts_source": artifacts_source or "local_existing",
        "git": {
            "sha": git_value(["rev-parse", "HEAD"], cwd=repo_root),
            "branch": git_value(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root),
            "dirty": bool(git_value(["status", "--porcelain"], cwd=repo_root)),
        },
        "artifacts": [
            collect_artifact_records(state_dir, rel_path) for rel_path in DEFAULT_ARTIFACTS
        ],
    }
    if instances_path is not None:
        resolved_instances = instances_path.resolve(strict=False)
        manifest["instances"] = {
            "path": str(resolved_instances),
            "exists": resolved_instances.exists(),
        }
        if resolved_instances.exists() and resolved_instances.is_file():
            manifest["instances"]["size"] = resolved_instances.stat().st_size
            manifest["instances"]["sha256"] = sha256_file(resolved_instances)
    return manifest


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=Path("logs"))
    parser.add_argument("--artifacts-source", default=None)
    parser.add_argument("--instances", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    output = args.output or args.state_dir / "artifact_manifest.json"
    manifest = build_manifest(
        state_dir=args.state_dir,
        artifacts_source=args.artifacts_source,
        instances_path=args.instances,
        repo_root=repo_root,
    )
    write_json_atomic(output, manifest)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
