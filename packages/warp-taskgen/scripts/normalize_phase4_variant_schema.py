#!/usr/bin/env python3
"""Add semantic Phase 4 variant accounting fields to existing run artifacts.

This is a schema-normalization tool, not a scorer. It preserves legacy
``generation_*`` fields and appends clearer counters derived from existing
``results.json`` rows. Use it for completed historical runs whose
``progress.json`` predates semantic variant accounting.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from worldsim.atomic_io import write_json_atomic
from worldsim.phase_4.variant_accounting import semantic_variant_accounting


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_phase4_dir(path: Path) -> Path:
    if path.name == "results.json":
        return path.parent
    if path.name == "phase_4":
        return path
    candidate = path / "phase_4"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"could not resolve phase_4 directory from {path}")


def _variation(row: dict[str, Any]) -> dict[str, Any] | None:
    for key in ("eval_awareness_iterator", "strategy_variation"):
        value = row.get(key)
        if isinstance(value, dict):
            return value
    return None


def _variation_entered(variation: dict[str, Any]) -> bool:
    return variation.get("status") != "skipped"


def _variant_accounting_for_variation(variation: dict[str, Any]) -> dict[str, int]:
    raw_variants = variation.get("variant_results")
    raw_errors = variation.get("variant_generation_errors")
    variant_results = [item for item in raw_variants if isinstance(item, dict)] if isinstance(raw_variants, list) else []
    generation_errors = raw_errors if isinstance(raw_errors, list) else []
    return semantic_variant_accounting(
        variant_results=variant_results,
        generation_errors=generation_errors,
    )


def _ecologically_valid(row: dict[str, Any]) -> bool:
    encounter = row.get("encounter")
    if not isinstance(encounter, dict):
        return False
    try:
        return float(encounter.get("max_coverage") or 0) > 0
    except (TypeError, ValueError):
        return False


def normalize_run(path: Path, *, write: bool) -> dict[str, Any]:
    phase4_dir = _resolve_phase4_dir(path)
    results_path = phase4_dir / "results.json"
    progress_path = phase4_dir / "progress.json"
    rows = _load_json(results_path)
    if not isinstance(rows, list):
        raise ValueError(f"expected JSON array at {results_path}")

    totals: Counter[str] = Counter()
    entered = 0
    touched_rows = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        variation = _variation(row)
        if not isinstance(variation, dict):
            continue
        if not _variation_entered(variation):
            continue
        entered += 1
        accounting = _variant_accounting_for_variation(variation)
        totals.update(accounting)
        payload = {"schema_version": 1, **accounting}
        if variation.get("variant_outcome_accounting") != payload:
            touched_rows += 1
            if write:
                variation["variant_outcome_accounting"] = payload

    progress_touched = False
    progress = _load_json(progress_path) if progress_path.exists() else None
    if isinstance(progress, dict):
        variant_progress = progress.get("variant_progress")
        if isinstance(variant_progress, dict):
            for key, value in totals.items():
                if variant_progress.get(key) != value:
                    progress_touched = True
                    if write:
                        variant_progress[key] = value
            if variant_progress.get("entered_tasks") != entered:
                progress_touched = True
                if write:
                    variant_progress["entered_tasks"] = entered
            if write and progress_touched:
                migrations = progress.setdefault("schema_migrations", [])
                if isinstance(migrations, list):
                    migrations.append(
                        {
                            "name": "phase4_semantic_variant_accounting",
                            "applied_at": datetime.now(UTC).isoformat(),
                        }
                    )

    if write:
        if touched_rows:
            write_json_atomic(
                results_path,
                rows,
                failpoint_base="phase_4.schema_normalize.results",
            )
        if progress_touched and isinstance(progress, dict):
            write_json_atomic(
                progress_path,
                progress,
                failpoint_base="phase_4.schema_normalize.progress",
            )

    return {
        "phase4_dir": str(phase4_dir),
        "write": write,
        "entered_tasks": entered,
        "rows_with_added_accounting": touched_rows,
        "progress_updated": progress_touched,
        **dict(sorted(totals.items())),
        "pvpo_valid_variants": sum(
            1
            for row in rows
            if isinstance(row, dict)
            for variation in [_variation(row)]
            if isinstance(variation, dict)
            for variant in variation.get("variant_results", [])
            if isinstance(variant, dict) and _ecologically_valid(variant)
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Run dir, phase_4 dir, or results.json")
    parser.add_argument("--write", action="store_true", help="Persist additive schema fields.")
    args = parser.parse_args(argv)
    reports = [normalize_run(path, write=args.write) for path in args.runs]
    print(json.dumps(reports, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
