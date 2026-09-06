"""Export matched-study reports from existing retained result.json artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from warp_taskgen.phase_4.matched_rewrite_analysis import analyze_matched_rewrite_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="Retained matched result.json files")
    parser.add_argument(
        "--families",
        required=True,
        nargs=7,
        metavar="TASK_CARD_ID",
        help="The seven frozen allocation task-card IDs, including empty families",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    try:
        report = analyze_matched_rewrite_results(
            args.results,
            expected_families=args.families,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
        )
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
