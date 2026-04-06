"""Extract WorldSim website data from browser-sim results/ directory.

Walks the results/ directory, finds generated HTML files and screenshots,
and copies them into the standardized data/worldsim/ structure.

Usage:
    python -m eval_awareness_experiments.extract_worldsim
    python -m eval_awareness_experiments.extract_worldsim --results-dir ../results --output data/worldsim
"""

import argparse
import json
import shutil
from pathlib import Path

from eval_awareness_experiments.types import WebsiteSample


def infer_website_type(filename: str) -> str:
    """Infer website type from the HTML filename.

    WorldSim HTML files are named like:
        github_repo_issues.html
        jenkins_manage.html
        jira_browse_JIRA-4829.html
        gitlab_explore.html
    """
    name_lower = filename.lower()

    type_keywords = {
        "github": "github",
        "gitlab": "gitlab",
        "jenkins": "jenkins",
        "jira": "jira",
        "paypal": "paypal",
        "gmail": "gmail",
        "webhook": "webhook",
        "cicd": "cicd",
    }

    for keyword, wtype in type_keywords.items():
        if keyword in name_lower:
            return wtype

    return "other"


def extract_from_results(
    results_dir: Path,
    output_dir: Path,
    max_per_type: int | None = None,
) -> list[WebsiteSample]:
    """Extract WorldSim HTML and screenshots from results directory.

    Args:
        results_dir: Path to browser-sim/results/.
        output_dir: Path to save extracted data (e.g., data/worldsim/).
        max_per_type: Maximum samples per website type (None = unlimited).

    Returns:
        List of WebsiteSample objects.
    """
    samples = []
    type_counts: dict[str, int] = {}

    # Find all experiment directories (contain variation_* subdirs with HTML)
    for exp_dir in sorted(results_dir.rglob("variation_*")):
        if not exp_dir.is_dir():
            continue

        html_files = sorted(exp_dir.glob("*.html"))
        parent_dir = exp_dir.parent  # The seed directory with screenshots

        for html_file in html_files:
            website_type = infer_website_type(html_file.stem)

            if max_per_type and type_counts.get(website_type, 0) >= max_per_type:
                continue

            # Create output subdirectory
            type_dir = output_dir / website_type
            type_dir.mkdir(parents=True, exist_ok=True)

            idx = type_counts.get(website_type, 0)
            base_name = f"sample_{idx:03d}_{html_file.stem}"

            # Copy HTML
            dst_html = type_dir / f"{base_name}.html"
            shutil.copy2(html_file, dst_html)

            # Find matching screenshot (use step_0 as the initial page state)
            screenshot_path = None
            for step_png in sorted(parent_dir.glob("screenshot_step_0.png")):
                dst_png = type_dir / f"{base_name}.png"
                shutil.copy2(step_png, dst_png)
                screenshot_path = str(dst_png)
                break

            sample = WebsiteSample(
                id=f"worldsim_{website_type}_{idx:03d}",
                source="worldsim",
                website_type=website_type,
                html_path=str(dst_html),
                screenshot_path=screenshot_path,
                metadata={
                    "source_html": str(html_file),
                    "experiment_dir": str(parent_dir),
                },
            )
            samples.append(sample)
            type_counts[website_type] = idx + 1

    return samples


def build_manifest(samples: list[WebsiteSample], output_path: Path) -> None:
    """Build a manifest.json file from a list of samples."""
    manifest = []
    for s in samples:
        manifest.append({
            "id": s.id,
            "source": s.source,
            "website_type": s.website_type,
            "html_path": s.html_path,
            "axtree_path": s.axtree_path,
            "screenshot_path": s.screenshot_path,
            "metadata": s.metadata,
        })

    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest written to {output_path} ({len(manifest)} samples)")


def main():
    parser = argparse.ArgumentParser(description="Extract WorldSim data from browser-sim results")
    parser.add_argument("--results-dir", type=str, default="results",
                        help="Path to browser-sim results/ directory")
    parser.add_argument("--output", type=str, default="eval_awareness_experiments/data/worldsim",
                        help="Output directory for extracted data")
    parser.add_argument("--max-per-type", type=int, default=None,
                        help="Max samples per website type")
    parser.add_argument("--manifest", type=str, default="eval_awareness_experiments/data/manifest.json",
                        help="Path for output manifest file")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print(f"Extracting from {results_dir} -> {output_dir}")
    samples = extract_from_results(results_dir, output_dir, args.max_per_type)

    # Print summary
    type_counts: dict[str, int] = {}
    for s in samples:
        type_counts[s.website_type] = type_counts.get(s.website_type, 0) + 1

    print(f"\nExtracted {len(samples)} samples:")
    for wtype, count in sorted(type_counts.items()):
        print(f"  {wtype}: {count}")

    # Build manifest
    build_manifest(samples, Path(args.manifest))


if __name__ == "__main__":
    main()
