"""Neutral source indexes staged into Phase 0c profiling sandboxes."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from worldsim.phases.phase_0c_artifacts import file_sha256, write_json_atomic

_ROUTE_LITERAL = re.compile(r"""["'](/[A-Za-z0-9_./{}:$<>\-\[\]]{1,180})["']""")
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
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(benchmark_root).resolve()
    files = _indexed_files(file_list, root)
    routes = _route_candidates(files, root)
    tasks = _task_candidates(files, root, manifest)
    manifest_slice = _manifest_slice(manifest, site_name)

    artifacts = {
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

    routed: dict[str, str] = {}
    for filename, payload in artifacts.items():
        path = output_dir / filename
        write_json_atomic(path, payload)
        routed[f"/workspace/inputs/{filename}"] = str(path)
    return routed


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
        if not record.get("text_indexed"):
            continue
        rel = str(record["path"])
        path = benchmark_root / rel
        text = _read_text(path)
        if text is None:
            continue
        seen: set[str] = set()
        for match in _ROUTE_LITERAL.finditer(text):
            route = match.group(1)
            if route in seen:
                continue
            seen.add(route)
            line = text.count("\n", 0, match.start()) + 1
            records.append({"file": rel, "line": line, "literal": route})
            if len(seen) >= 100:
                break
    return records


def _task_candidates(
    files: list[dict[str, Any]],
    benchmark_root: Path,
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    records.extend(_task_candidates_from_manifest(manifest))
    for record in files:
        rel = str(record["path"])
        suffix = str(record.get("suffix") or "")
        if suffix not in {".json", ".jsonl"}:
            continue
        path = benchmark_root / rel
        if suffix == ".json":
            payload = _read_json(path)
            if payload is not None:
                records.extend(_extract_task_like_records(payload, source=rel))
        else:
            records.extend(_jsonl_task_records(path, source=rel))
    return records[:1000]


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


def _extract_task_like_records(payload: object, *, source: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
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
    return records


def _jsonl_task_records(path: Path, *, source: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for item in _extract_task_like_records(payload, source=source):
                    item["line"] = index
                    records.append(item)
                if len(records) >= 200:
                    break
    except OSError:
        return []
    return records


def _walk_json(payload: object, path: str = "$") -> Iterable[tuple[str, object]]:
    yield path, payload
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            yield from _walk_json(value, f"{path}.{key}")
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            yield from _walk_json(value, f"{path}[{index}]")


def _is_compact_json_value(value: object) -> bool:
    if value is None or isinstance(value, bool | int | float):
        return True
    if isinstance(value, str):
        return len(value) <= 500
    if isinstance(value, list):
        return len(value) <= 12 and all(isinstance(item, str | int | float | bool) for item in value)
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
