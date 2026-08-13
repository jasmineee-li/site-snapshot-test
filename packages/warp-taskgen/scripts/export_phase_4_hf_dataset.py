#!/usr/bin/env python3
"""Export Phase 4 AgentLab runs as a Hugging Face-compatible local dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from warp_taskgen.phase_4.hf_dataset_export import (
    DEFAULT_DATASET_ID,
    export_hf_dataset,
    parse_run_specs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        metavar="MODEL=RUN_DIR",
        help=(
            "Model key and Phase 4 run directory. Repeat once per model, e.g. "
            "--run sonnet46=logs/agentlab_linknaturalization_true50_sonnet46_w48_20260509. "
            "Optional comma-separated metadata is supported after RUN_DIR, e.g. "
            "...,agent_provider=openrouter,agent_service_tier=priority."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="Repository/root directory used to resolve relative run dirs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(f"data/hf/{DEFAULT_DATASET_ID}"),
        help="Output dataset directory.",
    )
    parser.add_argument(
        "--dataset-id",
        default=DEFAULT_DATASET_ID,
        help="Dataset id written into metadata and README.",
    )
    parser.add_argument(
        "--no-bundles",
        action="store_true",
        help="Write only JSONL indexes; skip per-task trajectory tarballs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    args = parser.parse_args(argv)

    manifest = export_hf_dataset(
        runs=parse_run_specs(args.run),
        output_dir=args.output_dir,
        root=args.root,
        dataset_id=args.dataset_id,
        bundle_artifacts=not args.no_bundles,
        overwrite=args.overwrite,
    )
    print(args.output_dir / "metadata.json")
    for split in ("runs", "tasks", "variants"):
        print(args.output_dir / manifest["splits"][split]["path"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
