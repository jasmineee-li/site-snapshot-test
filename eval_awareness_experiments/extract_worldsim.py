"""Extract WorldSim website data from browser-sim results/ directory.

Walks the results/ directory, finds generated HTML files and screenshots,
and copies them into the standardized data/worldsim/ structure grouped by
website type (jenkins, jira, gitlab, etc.).

Usage:
    python -m eval_awareness_experiments.extract_worldsim
    python -m eval_awareness_experiments.extract_worldsim --results-dir results/ --output eval_awareness_experiments/data/worldsim
"""

import argparse
import json
import shutil
from pathlib import Path

DEFAULT_OUTPUT = Path("eval_awareness_experiments/data/worldsim")
MANIFEST_PATH = Path("eval_awareness_experiments/data/manifest.json")

# Map filename keywords to website types
TYPE_KEYWORDS = {
    "github": "github",
    "gitlab": "gitlab",
    "jenkins": "jenkins",
    "jira": "jira",
    "paypal": "paypal",
    "gmail": "gmail",
    "webhook": "webhook",
    "cicd": "cicd",
}


def infer_website_type(filename: str) -> str:
    """Infer website type from the HTML filename."""
    name_lower = filename.lower()
    for keyword, wtype in TYPE_KEYWORDS.items():
        if keyword in name_lower:
            return wtype
    return "other"


def extract_from_results(
    results_dir: Path,
    output_dir: Path,
) -> dict[str, list[str]]:
    """Extract HTML and screenshots from results directory.

    Groups pages by website type. Returns {website_type: [page_names]}.
    """
    pages_by_type: dict[str, list[str]] = {}

    for exp_dir in sorted(results_dir.rglob("variation_*")):
        if not exp_dir.is_dir():
            continue

        parent_dir = exp_dir.parent  # Has screenshots

        for html_file in sorted(exp_dir.glob("*.html")):
            website_type = infer_website_type(html_file.stem)
            type_dir = output_dir / website_type
            type_dir.mkdir(parents=True, exist_ok=True)

            # Use the HTML filename (without extension) as the page name
            page_name = html_file.stem

            # Copy HTML
            dst_html = type_dir / f"{page_name}.html"
            shutil.copy2(html_file, dst_html)

            # Find matching screenshot (step_0 = initial page state)
            for step_png in sorted(parent_dir.glob("screenshot_step_0.png")):
                dst_png = type_dir / f"{page_name}.png"
                shutil.copy2(step_png, dst_png)
                break

            pages_by_type.setdefault(website_type, []).append(page_name)

    return pages_by_type


def update_manifest(
    pages_by_type: dict[str, list[str]],
    output_dir: Path,
    manifest_path: Path,
) -> None:
    """Add worldsim entries to the manifest."""
    existing = []
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())

    # Remove old worldsim entries
    existing = [e for e in existing if e.get("source") != "worldsim"]

    for website_type, pages in sorted(pages_by_type.items()):
        entry = {
            "id": f"worldsim_{website_type}",
            "source": "worldsim",
            "label": "synthetic",
            "path": str(output_dir / website_type),
            "pages": pages,
            "metadata": {
                "benchmark": "worldsim",
                "website_type": website_type,
            },
        }
        existing.append(entry)

    manifest_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"\nManifest updated: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Extract WorldSim data from browser-sim results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Path to browser-sim results/ directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output directory for extracted data",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=str(MANIFEST_PATH),
        help="Path for manifest file",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    print(f"Extracting from {results_dir} -> {output_dir}")
    pages_by_type = extract_from_results(results_dir, output_dir)

    total_pages = sum(len(p) for p in pages_by_type.values())
    print(f"\nExtracted {total_pages} pages:")
    for wtype, pages in sorted(pages_by_type.items()):
        print(f"  {wtype}: {len(pages)} pages")

    update_manifest(pages_by_type, output_dir, Path(args.manifest))


if __name__ == "__main__":
    main()
