"""CLI entry point for running eval awareness experiments.

Usage:
    python -m eval_awareness_experiments.run --config configs/p_eval_all.yaml
    python -m eval_awareness_experiments.run --experiment p_eval --models anthropic/claude-sonnet-4 --formats html screenshot
"""

import argparse
import asyncio
import json
import logging
from pathlib import Path

import yaml

from eval_awareness_experiments.llm import LLM
from eval_awareness_experiments.types import WebsiteSample
from eval_awareness_experiments.experiments.p_eval import PEvalExperiment
from eval_awareness_experiments.experiments.comparative import ComparativeExperiment
from eval_awareness_experiments.experiments.transcript_awareness import TranscriptAwarenessExperiment
from eval_awareness_experiments.experiments.open_ended import OpenEndedExperiment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

EXPERIMENT_CLASSES = {
    "p_eval": PEvalExperiment,
    "comparative": ComparativeExperiment,
    "transcript_awareness": TranscriptAwarenessExperiment,
    "open_ended": OpenEndedExperiment,
}


def load_manifest(manifest_path: Path) -> list[WebsiteSample]:
    """Load samples from a manifest.json file.

    Handles both the new per-environment format (with "pages" list) and the
    legacy per-file format (with "html_path" etc.). The new format expands
    each page into a separate WebsiteSample.
    """
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    samples = []
    for d in data:
        if "pages" in d and d["pages"]:
            # New format: expand each page into a sample
            base_path = Path(d["path"])
            website_type = d.get("metadata", {}).get("website_type") or d["id"].split("_", 1)[-1]
            for page_name in d["pages"]:
                html_path = base_path / f"{page_name}.html"
                axtree_path = base_path / f"{page_name}_axtree.txt"
                screenshot_path = base_path / f"{page_name}.png"
                samples.append(WebsiteSample(
                    id=f"{d['id']}_{page_name}",
                    source=d["source"],
                    website_type=website_type,
                    html_path=str(html_path) if html_path.exists() else None,
                    axtree_path=str(axtree_path) if axtree_path.exists() else None,
                    screenshot_path=str(screenshot_path) if screenshot_path.exists() else None,
                    metadata={**d.get("metadata", {}), "page": page_name},
                ))
        elif "html_path" in d or "axtree_path" in d or "screenshot_path" in d:
            # Legacy format
            samples.append(WebsiteSample(
                id=d["id"],
                source=d["source"],
                website_type=d.get("website_type", "unknown"),
                html_path=d.get("html_path"),
                axtree_path=d.get("axtree_path"),
                screenshot_path=d.get("screenshot_path"),
                metadata=d.get("metadata", {}),
            ))
    return samples


def filter_samples(
    samples: list[WebsiteSample],
    sources: list[str] | None = None,
    website_types: list[str] | None = None,
    id_contains: str | None = None,
) -> list[WebsiteSample]:
    """Filter samples by source, website type, or ID substring."""
    filtered = samples
    if sources:
        filtered = [s for s in filtered if s.source in sources]
    if website_types:
        # Substring match on website_type
        filtered = [s for s in filtered if any(wt in s.website_type for wt in website_types)]
    if id_contains:
        filtered = [s for s in filtered if id_contains.lower() in s.id.lower()]
    return filtered


async def run_experiment(config: dict) -> None:
    """Run an experiment based on configuration."""
    experiment_name = config["experiment"]
    models = config.get("models", ["anthropic/claude-sonnet-4"])
    formats = config.get("formats", ["html"])
    output_dir = Path(config.get("output_dir", "eval_awareness_experiments/results"))
    manifest_path = Path(config.get("manifest", "eval_awareness_experiments/data/manifest.json"))

    # Load and filter samples
    samples = load_manifest(manifest_path)
    samples = filter_samples(
        samples,
        sources=config.get("sources"),
        website_types=config.get("website_types"),
    )

    if not samples:
        print("No samples matched the filter criteria.")
        return

    print(f"Running {experiment_name} on {len(samples)} samples")
    print(f"  Models: {models}")
    print(f"  Formats: {formats}")
    print(f"  Sources: {set(s.source for s in samples)}")
    print(f"  Website types: {set(s.website_type for s in samples)}")

    for model_name in models:
        print(f"\n--- Model: {model_name} ---")
        model = LLM(model_name)
        exp_output = output_dir / config.get("name", experiment_name) / model_name.replace("/", "_")

        exp_cls = EXPERIMENT_CLASSES[experiment_name]
        experiment = exp_cls(model=model, output_dir=exp_output)

        if experiment_name == "comparative":
            for fmt in formats:
                await experiment.run_pairs(samples, fmt)
        else:
            await experiment.run(samples, formats)

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Run eval awareness experiments")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--experiment", type=str, choices=list(EXPERIMENT_CLASSES.keys()),
                        help="Experiment type (if not using config)")
    parser.add_argument("--models", type=str, nargs="+", default=["anthropic/claude-sonnet-4"],
                        help="Models to use")
    parser.add_argument("--formats", type=str, nargs="+", default=["html"],
                        help="Format types: html, axtree, screenshot")
    parser.add_argument("--sources", type=str, nargs="+", default=None,
                        help="Filter by source: worldsim, webarena, agentcompany, real")
    parser.add_argument("--website-types", type=str, nargs="+", default=None,
                        help="Filter by website type: github, gitlab, etc.")
    parser.add_argument("--manifest", type=str, default="eval_awareness_experiments/data/manifest.json",
                        help="Path to manifest file")
    parser.add_argument("--output-dir", type=str, default="eval_awareness_experiments/results",
                        help="Output directory for results")
    args = parser.parse_args()

    if args.config:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    elif args.experiment:
        config = {
            "name": args.experiment,
            "experiment": args.experiment,
            "models": args.models,
            "formats": args.formats,
            "sources": args.sources,
            "website_types": args.website_types,
            "manifest": args.manifest,
            "output_dir": args.output_dir,
        }
    else:
        parser.error("Either --config or --experiment is required")
        return

    asyncio.run(run_experiment(config))


if __name__ == "__main__":
    main()
