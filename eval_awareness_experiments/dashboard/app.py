"""Minimal localhost dashboard for the causal eval-awareness pilot.

Reads the run manifest (built by `run_manifest.py`) on every request so
data stays fresh as runs complete. Three views:
  GET /            matrix view — coverage grid of (benchmark × arm × model)
  GET /cell/<bench>/<arm>/<model_slug>   per-cell detail (ASR, VEA, 5PQ, splits)
  GET /api/manifest   raw manifest JSON (for tooling / debugging)

Stack: Flask + a single Jinja template + Tailwind via CDN. No JS framework,
no build step.

Usage:
    python -m eval_awareness_experiments.dashboard \
        --results-dir eval_awareness_experiments/results/causal_pilot \
        --port 8000

Then open http://localhost:8000.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flask import Flask, abort, jsonify, render_template

from eval_awareness_experiments.run_manifest import scan as scan_manifest


def create_app(results_dir: Path) -> Flask:
    app = Flask(__name__)
    app.config["RESULTS_DIR"] = results_dir

    @app.route("/")
    def index():
        manifest = scan_manifest(results_dir)
        cells = manifest["cells"]

        # Build pivot: rows = (benchmark, model_slug), cols = arms.
        benchmarks: list[str] = []
        models: list[str] = []
        arms: list[str] = []
        seen_b, seen_m, seen_a = set(), set(), set()
        cell_index: dict[tuple[str, str, str], dict] = {}
        for c in cells:
            b, a, m = c["benchmark"], c["arm"], c["model_slug"]
            if b not in seen_b:
                benchmarks.append(b); seen_b.add(b)
            if a not in seen_a:
                arms.append(a); seen_a.add(a)
            if m not in seen_m:
                models.append(m); seen_m.add(m)
            cell_index[(b, a, m)] = c

        # Stable arm ordering: canonical first, then alphabetical
        canonical_order = ["bare", "xml_safety", "xml_scenario", "native"]
        arms.sort(key=lambda a: (canonical_order.index(a) if a in canonical_order else 999, a))

        # Build the grid view: rows × cols
        rows: list[dict] = []
        for b in benchmarks:
            for m in models:
                row_cells = []
                any_present = False
                for a in arms:
                    c = cell_index.get((b, a, m))
                    if c:
                        any_present = True
                    row_cells.append({
                        "arm": a,
                        "cell": c,
                    })
                if any_present:
                    rows.append({
                        "benchmark": b,
                        "model_slug": m,
                        "cells": row_cells,
                    })

        return render_template(
            "index.html",
            scanned_at=manifest["scanned_at"],
            results_dir=str(results_dir),
            n_total=len(cells),
            arms=arms,
            rows=rows,
        )

    @app.route("/cell/<benchmark>/<arm>/<model_slug>")
    def cell_detail(benchmark, arm, model_slug):
        manifest = scan_manifest(results_dir)
        for c in manifest["cells"]:
            if (c["benchmark"], c["arm"], c["model_slug"]) == (benchmark, arm, model_slug):
                # Try to list trajectory markdown links for clickable drill-down
                links = []
                for split in c.get("splits", []):
                    md_dir = Path(c["output_dir"]) / benchmark / split["name"] / "markdown"
                    if md_dir.exists():
                        links.append({
                            "split": split["name"],
                            "files": sorted(p.name for p in md_dir.glob("*.md") if p.name != "run_meta.md"),
                            "md_dir": str(md_dir),
                        })
                return render_template("cell.html", cell=c, links=links)
        abort(404)

    @app.route("/api/manifest")
    def api_manifest():
        return jsonify(scan_manifest(results_dir))

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Root of the causal-pilot results tree.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not args.results_dir.exists():
        # Don't fail — the dir might not exist before any runs land.
        print(f"WARNING: results-dir does not exist yet: {args.results_dir}")
        print("Dashboard will start anyway and re-scan on each request.")

    app = create_app(args.results_dir)
    print(f"Dashboard serving {args.results_dir} on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
