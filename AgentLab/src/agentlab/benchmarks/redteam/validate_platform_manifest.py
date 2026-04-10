"""CLI for validating redteam ``platform_manifest.json`` authoring data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from agentlab.benchmarks.redteam.app_artifacts import (
    load_platform_manifest,
    platform_manifest_docs_path_errors,
    repo_root,
    validate_platform_manifest,
)
from agentlab.benchmarks.redteam.behavior_ids import normalize_behavior_spec, resolve_behavior_id


def _load_behavior_ids(benchmark_file: str | Path) -> set[str]:
    path = Path(benchmark_file)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        behaviors = payload
    elif isinstance(payload, dict):
        behaviors = payload.get("data", payload.get("behaviors", []))
    else:
        raise ValueError(f"Unexpected benchmark format in {path}")
    if not isinstance(behaviors, list):
        raise ValueError(f"Benchmark file {path} missing list of behaviors")
    return {
        resolve_behavior_id(normalize_behavior_spec(case, case_idx))
        for case_idx, case in enumerate(behaviors)
        if isinstance(case, dict)
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate shared-app redteam platform_manifest.json authoring data.",
    )
    parser.add_argument(
        "--platform-manifest",
        default=None,
        help="Path to platform_manifest.json (default: repo-root platform_manifest.json).",
    )
    parser.add_argument(
        "--benchmark-file",
        default=None,
        help="Optional behaviors JSON file used to verify mapped behavior_ids are known.",
    )
    parser.add_argument(
        "--require-full-coverage",
        action="store_true",
        help="Require the platform manifest to map every behavior_id in --benchmark-file.",
    )
    parser.add_argument(
        "--check-docs-paths",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify each app docs_path exists relative to --repo-root (default: enabled).",
    )
    parser.add_argument(
        "--repo-root",
        default=str(repo_root()),
        help="Repository root used for relative docs_path resolution (default: Browser-Sim repo root).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest_path = Path(args.platform_manifest) if args.platform_manifest else repo_root() / "platform_manifest.json"

    try:
        platform_manifest = load_platform_manifest(manifest_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"platform manifest invalid: {exc}", file=sys.stderr)
        return 1

    known_behavior_ids: set[str] | None = None
    if args.benchmark_file:
        try:
            known_behavior_ids = _load_behavior_ids(args.benchmark_file)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"benchmark file invalid: {exc}", file=sys.stderr)
            return 1

    errors = validate_platform_manifest(
        platform_manifest,
        known_behavior_ids=known_behavior_ids,
        repo_root_path=args.repo_root,
        require_full_coverage=args.require_full_coverage,
    )
    if args.check_docs_paths:
        errors.extend(
            platform_manifest_docs_path_errors(
                platform_manifest,
                repo_root_path=args.repo_root,
            )
        )

    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    apps = platform_manifest.get("apps") or []
    mapping_count = sum(
        len(app.get("behaviors") or [])
        for app in apps
        if isinstance(app, dict)
    )
    print(f"platform manifest valid: {len(apps)} app(s), {mapping_count} behavior mapping(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
