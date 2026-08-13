#!/usr/bin/env python3
"""Audit Phase 0c profile artifacts for deterministic provenance issues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from warp_taskgen.phases.phase_0c_audit import audit_phase_0c_profiles


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles_dir", type=Path)
    parser.add_argument("--benchmark-root", type=Path, default=None)
    parser.add_argument(
        "--manifest-eval-type",
        action="append",
        default=[],
        help="Manifest eval type to enforce; repeat for multiple values.",
    )
    parser.add_argument("--json", action="store_true", help="Print the full JSON report.")
    args = parser.parse_args(argv)

    report = audit_phase_0c_profiles(
        args.profiles_dir,
        benchmark_root=args.benchmark_root,
        manifest_eval_types=args.manifest_eval_type,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        summary = report["summary"]
        print(
            "Phase 0c profile audit: "
            f"{summary['sites']} site(s), {summary['errors']} error(s), "
            f"{summary['warnings']} warning(s)"
        )
        for finding in report["errors"][:20]:
            print(f"ERROR {finding.get('site')}: {finding.get('code')}: {finding.get('message')}")
        for finding in report["warnings"][:20]:
            print(f"WARN {finding.get('site')}: {finding.get('code')}: {finding.get('message')}")
    return 1 if report["summary"]["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
