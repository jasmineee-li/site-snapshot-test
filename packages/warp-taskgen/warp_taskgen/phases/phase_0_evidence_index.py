"""Neutral source indexes staged into Phase 0c profiling sandboxes."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from itertools import islice
from pathlib import Path
from typing import Any

from warp_taskgen.phases.phase_0c_artifacts import (
    file_sha256,
    hash_file_records,
    hash_json,
    write_json_atomic,
)

_ROUTE_LITERAL = re.compile(r"""["'](/[A-Za-z0-9_./{}:$<>\-\[\]]{1,180})["']""")
_MAX_TASK_RECORDS = 1000
_MAX_ROUTE_RECORDS = 2000
_MAX_TEXT_INDEX_BYTES = 2 * 1024 * 1024
_MAX_TEXT_INDEX_LINES = 20000
_MAX_TEXT_LINE_BYTES = 64 * 1024
_MAX_JSON_TASK_BYTES = 2 * 1024 * 1024
_MAX_JSONL_TASK_BYTES = 2 * 1024 * 1024
_MAX_JSONL_TASK_LINES = 5000
_MAX_JSONL_LINE_BYTES = 64 * 1024
_MAX_JSON_WALK_NODES = 5000
_MAX_JSON_WALK_DEPTH = 12
_TASK_KEY_HINTS = frozenset(
    {
        "task",
        "task_id",
        "task_name",
        "intent",
        "goal",
        "start_url",
        "start_urls",
        "eval_type",
        "sites",
        "site",
    }
)
_TEXT_SUFFIXES = {
    ".cfg",
    ".conf",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".jsonl",
    ".jsx",
    ".md",
    ".php",
    ".py",
    ".rb",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}


def build_phase_0c_evidence_indexes(
    *,
    file_list: Iterable[str],
    benchmark_root: Path,
    manifest: Mapping[str, Any],
    site_name: str,
    output_dir: Path,
) -> dict[str, str]:
    """Write neutral evidence indexes and return sandbox route mappings.

    The indexes intentionally avoid eligibility decisions. They expose file
    paths, manifest slices, task-like records, and route-like string literals so
    Phase 0c prompts can cite evidence without making the host code pre-decide
    the research answer.
    """
    artifacts = build_phase_0c_evidence_payloads(
        file_list=file_list,
        benchmark_root=benchmark_root,
        manifest=manifest,
        site_name=site_name,
    )
    return write_phase_0c_evidence_indexes(artifacts, output_dir=output_dir)


def build_phase_0c_evidence_payloads(
    *,
    file_list: Iterable[str],
    benchmark_root: Path,
    manifest: Mapping[str, Any],
    site_name: str,
) -> dict[str, object]:
    """Build neutral evidence payloads without writing them to disk."""
    root = Path(benchmark_root).resolve()
    files = _indexed_files(file_list, root)
    routes = _route_candidates(files, root)
    tasks = _task_candidates(files, root, manifest)
    manifest_slice = _manifest_slice(manifest, site_name)

    return {
        "FILES_INDEX.json": {
            "schema_version": 1,
            "site_name": site_name,
            "files": files,
        },
        "ROUTES_INDEX.json": {
            "schema_version": 1,
            "site_name": site_name,
            "route_candidates": routes,
        },
        "TASKS_INDEX.json": {
            "schema_version": 1,
            "site_name": site_name,
            "task_candidates": tasks,
        },
        "MANIFEST_SLICE.json": {
            "schema_version": 1,
            "site_name": site_name,
            "manifest": manifest_slice,
        },
    }


def write_phase_0c_evidence_indexes(
    artifacts: Mapping[str, object],
    *,
    output_dir: Path,
) -> dict[str, str]:
    """Write prebuilt evidence indexes and return sandbox route mappings."""
    output_dir.mkdir(parents=True, exist_ok=True)
    routed: dict[str, str] = {}
    for filename, payload in artifacts.items():
        path = output_dir / filename
        write_json_atomic(path, payload)
        routed[f"/workspace/inputs/{filename}"] = str(path)
    return routed


def hash_phase_0c_evidence_payloads(artifacts: Mapping[str, object]) -> str:
    """Hash evidence payloads without depending on temp paths or formatting."""
    return hash_json({filename: artifacts[filename] for filename in sorted(artifacts)})


def benchmark_digest_from_evidence_payloads(artifacts: Mapping[str, object]) -> str:
    """Return the benchmark content digest from FILES_INDEX records."""
    files_index = artifacts.get("FILES_INDEX.json")
    if not isinstance(files_index, Mapping):
        return hash_json([])
    files = files_index.get("files")
    return hash_file_records(files if isinstance(files, list) else [])


def _indexed_files(file_list: Iterable[str], benchmark_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_path in sorted({str(path) for path in file_list}):
        path = Path(raw_path).resolve()
        try:
            rel = path.relative_to(benchmark_root).as_posix()
        except ValueError:
            continue
        if not path.is_file():
            continue
        stat = path.stat()
        records.append(
            {
                "path": rel,
                "suffix": path.suffix.lower(),
                "size_bytes": stat.st_size,
                "sha256": file_sha256(path),
                "text_indexed": path.suffix.lower() in _TEXT_SUFFIXES,
            }
        )
    return records


def _route_candidates(
    files: list[dict[str, Any]],
    benchmark_root: Path,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for record in files:
        if len(records) >= _MAX_ROUTE_RECORDS:
            break
        if not record.get("text_indexed"):
            continue
        rel = str(record["path"])
        if int(record.get("size_bytes") or 0) > _MAX_TEXT_INDEX_BYTES:
            records.append(
                {
                    "file": rel,
                    "skipped": "text_file_too_large_for_route_index",
                    "size_bytes": record.get("size_bytes"),
                }
            )
            continue
        path = benchmark_root / rel
        seen: set[str] = set()
        line_limit_hit = False
        long_line_seen = False
        for line_number, line, status in _bounded_text_lines(path):
            if status == "line_too_long":
                long_line_seen = True
                continue
            if status == "line_limit_reached":
                line_limit_hit = True
                break
            for match in _ROUTE_LITERAL.finditer(line):
                route = match.group(1)
                if route in seen:
                    continue
                seen.add(route)
                records.append({"file": rel, "line": line_number, "literal": route})
                if len(seen) >= 100 or len(records) >= _MAX_ROUTE_RECORDS:
                    break
            if len(seen) >= 100 or len(records) >= _MAX_ROUTE_RECORDS:
                break
        if long_line_seen and len(records) < _MAX_ROUTE_RECORDS:
            records.append({"file": rel, "skipped": "route_index_line_too_long"})
        if line_limit_hit and len(records) < _MAX_ROUTE_RECORDS:
            records.append(
                {
                    "file": rel,
                    "skipped": "route_index_line_cap_reached",
                    "max_lines": _MAX_TEXT_INDEX_LINES,
                }
            )
    return records


def _task_candidates(
    files: list[dict[str, Any]],
    benchmark_root: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records.extend(_task_candidates_from_manifest(manifest))
    for record in files:
        if len(records) >= _MAX_TASK_RECORDS:
            break
        rel = str(record["path"])
        suffix = str(record.get("suffix") or "")
        if suffix not in {".json", ".jsonl"}:
            continue
        path = benchmark_root / rel
        if suffix == ".json":
            if int(record.get("size_bytes") or 0) > _MAX_JSON_TASK_BYTES:
                records.append(
                    {
                        "source": rel,
                        "skipped": "json_file_too_large_for_task_index",
                        "size_bytes": record.get("size_bytes"),
                    }
                )
                continue
            payload = _read_json(path)
            if payload is not None:
                remaining = max(0, _MAX_TASK_RECORDS - len(records))
                records.extend(_extract_task_like_records(payload, source=rel, limit=remaining))
        else:
            if int(record.get("size_bytes") or 0) > _MAX_JSONL_TASK_BYTES:
                records.append(
                    {
                        "source": rel,
                        "skipped": "jsonl_file_too_large_for_task_index",
                        "size_bytes": record.get("size_bytes"),
                    }
                )
                continue
            remaining = max(0, _MAX_TASK_RECORDS - len(records))
            records.extend(_jsonl_task_records(path, source=rel, limit=remaining))
    return records[:_MAX_TASK_RECORDS]


def _task_candidates_from_manifest(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    evaluation = manifest.get("evaluation")
    if not isinstance(evaluation, Mapping):
        return records
    for key in ("task_definition_paths", "harness_paths", "eval_types"):
        value = evaluation.get(key)
        if value:
            records.append({"source": "manifest", "key": key, "value": value})
    return records


def _manifest_slice(manifest: Mapping[str, Any], site_name: str) -> dict[str, Any]:
    sites = manifest.get("sites")
    site_records = []
    if isinstance(sites, list):
        for site in sites:
            if not isinstance(site, Mapping):
                continue
            if str(site.get("name") or "") == site_name:
                site_records.append(dict(site))
    evaluation = manifest.get("evaluation")
    return {
        "sites": site_records,
        "evaluation": evaluation if isinstance(evaluation, Mapping) else {},
    }


def _extract_task_like_records(
    payload: object,
    *,
    source: str,
    limit: int = _MAX_TASK_RECORDS,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if limit <= 0:
        return records
    for path, value in _walk_json(payload):
        if not isinstance(value, Mapping):
            continue
        keys = {str(key) for key in value.keys()}
        if not keys.intersection(_TASK_KEY_HINTS):
            continue
        compact = {
            key: value[key]
            for key in sorted(value)
            if str(key) in _TASK_KEY_HINTS and _is_compact_json_value(value[key])
        }
        if compact:
            records.append({"source": source, "json_path": path, "fields": compact})
            if len(records) >= limit:
                break
    return records


def _jsonl_task_records(
    path: Path,
    *,
    source: str,
    limit: int = _MAX_TASK_RECORDS,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if limit <= 0:
        return records
    try:
        with path.open("rb") as handle:
            for index in range(1, _MAX_JSONL_TASK_LINES + 1):
                raw_line = handle.readline(_MAX_JSONL_LINE_BYTES + 1)
                if not raw_line:
                    break
                if len(raw_line) > _MAX_JSONL_LINE_BYTES:
                    records.append(
                        {
                            "source": source,
                            "line": index,
                            "skipped": "jsonl_line_too_large_for_task_index",
                            "max_line_bytes": _MAX_JSONL_LINE_BYTES,
                        }
                    )
                    if not raw_line.endswith(b"\n"):
                        _drain_line(handle)
                    if len(records) >= limit:
                        break
                    continue
                line = raw_line.decode("utf-8", errors="replace")
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for item in _extract_task_like_records(
                    payload,
                    source=source,
                    limit=limit - len(records),
                ):
                    item["line"] = index
                    records.append(item)
                if len(records) >= limit:
                    break
            else:
                records.append(
                    {
                        "source": source,
                        "skipped": "jsonl_line_cap_reached",
                        "max_lines": _MAX_JSONL_TASK_LINES,
                    }
                )
    except OSError:
        return []
    return records


def _walk_json(payload: object, path: str = "$") -> Iterable[tuple[str, object]]:
    seen = 0
    stack: list[tuple[str, object, int]] = [(path, payload, 0)]
    while stack and seen < _MAX_JSON_WALK_NODES:
        current_path, current, depth = stack.pop()
        seen += 1
        yield current_path, current
        if depth >= _MAX_JSON_WALK_DEPTH:
            continue
        remaining_slots = _MAX_JSON_WALK_NODES - seen - len(stack)
        if remaining_slots <= 0:
            continue
        if isinstance(current, Mapping):
            items = list(islice(current.items(), remaining_slots))
            for key, value in reversed(items):
                stack.append((f"{current_path}.{key}", value, depth + 1))
        elif isinstance(current, list):
            upper = min(len(current), remaining_slots)
            for index in range(upper - 1, -1, -1):
                stack.append((f"{current_path}[{index}]", current[index], depth + 1))
    return


def _bounded_text_lines(path: Path) -> Iterable[tuple[int, str, str | None]]:
    try:
        with path.open("rb") as handle:
            for line_number in range(1, _MAX_TEXT_INDEX_LINES + 1):
                raw_line = handle.readline(_MAX_TEXT_LINE_BYTES + 1)
                if not raw_line:
                    return
                if len(raw_line) > _MAX_TEXT_LINE_BYTES:
                    if not raw_line.endswith(b"\n"):
                        _drain_line(handle)
                    yield line_number, "", "line_too_long"
                    continue
                yield line_number, raw_line.decode("utf-8", errors="replace"), None
            yield _MAX_TEXT_INDEX_LINES, "", "line_limit_reached"
    except OSError:
        return


def _drain_line(handle: Any) -> None:
    while True:
        chunk = handle.readline(8192)
        if not chunk or chunk.endswith(b"\n"):
            return


def _is_compact_json_value(value: object) -> bool:
    if value is None or isinstance(value, bool | int | float):
        return True
    if isinstance(value, str):
        return len(value) <= 500
    if isinstance(value, list):
        return len(value) <= 12 and all(
            isinstance(item, str | int | float | bool) for item in value
        )
    return False


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
