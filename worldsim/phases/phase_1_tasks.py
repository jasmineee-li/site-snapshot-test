"""Phase 1: Task generation (Mode A — wrap existing benchmark tasks).

Canonical source: ``docs/worldsim-v5-technical-specifcation.md`` "Phase 1: Task Generation".

Mode A reads task definitions from the benchmark dataset and wraps them into
the benign task bundle format used by Phases 2–4. No LLM needed.

Mode B (generate new tasks on existing environments) is a stretch goal,
deferred until Phases 3–4 are validated end-to-end.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from worldsim.state import STATE_DIR, save_state

logger = logging.getLogger(__name__)


async def run(args: argparse.Namespace) -> int:
    """Phase 1 Mode A entrypoint.

    Reads the benchmark's task definitions and wraps each into a benign task
    bundle for downstream phases.

    Expects ``args.benchmark`` (path to benchmark codebase) and
    ``args.config`` (path to BENCHMARK_MANIFEST.json from Phase 0a).
    """
    output_dir = STATE_DIR / "phase_1"

    # Load manifest to find task definition paths
    manifest_path = args.config or (STATE_DIR / "phase_0a" / "BENCHMARK_MANIFEST.json")
    if not Path(manifest_path).exists():
        logger.error("Manifest not found at %s — run phase 0a first", manifest_path)
        return 1
    manifest = json.loads(Path(manifest_path).read_text())

    benchmark_root = Path(args.benchmark) if args.benchmark else Path(manifest.get("benchmark_codebase", ""))
    if not benchmark_root.is_dir():
        logger.error("Benchmark root not found: %s — pass --benchmark", benchmark_root)
        return 1

    save_state("phase_1", status="running")

    # Find and load task definitions
    raw_tasks = _load_raw_tasks(manifest, benchmark_root)
    if not raw_tasks:
        logger.error("No tasks found in benchmark — check manifest task_definition_paths")
        return 1

    logger.info("Phase 1A: loaded %d raw tasks from benchmark", len(raw_tasks))

    # Wrap into benign task bundles
    benign_tasks = [_wrap_task(t) for t in raw_tasks]

    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "benign_tasks.json"
    output_path.write_text(json.dumps(benign_tasks, indent=2))

    save_state("phase_1", status="complete", tasks_path=str(output_path),
               task_count=len(benign_tasks))
    logger.info("Phase 1A complete — %d benign tasks written to %s", len(benign_tasks), output_path)
    return 0


def _load_raw_tasks(manifest: dict, benchmark_root: Path) -> list[dict]:
    """Load task definitions from paths declared in the manifest.

    Handles two formats:
    - Single JSON array file (WebArena Verified: webarena-verified.json)
    - Directory of per-task JSON files (original WebArena: config_files/)
    """
    task_paths = manifest.get("evaluation", {}).get("task_definition_paths", [])
    all_tasks: list[dict] = []

    for tp in task_paths:
        full = benchmark_root / tp
        if full.is_file() and full.suffix == ".json":
            data = json.loads(full.read_text())
            if isinstance(data, list):
                all_tasks.extend(data)
            else:
                all_tasks.append(data)
        elif full.is_dir():
            for f in sorted(full.rglob("*.json")):
                try:
                    data = json.loads(f.read_text())
                    if isinstance(data, list):
                        all_tasks.extend(data)
                    else:
                        all_tasks.append(data)
                except json.JSONDecodeError:
                    logger.warning("Skipping non-JSON file: %s", f)

    return all_tasks


def _wrap_task(raw: dict) -> dict:
    """Wrap a single benchmark task into the benign task bundle schema.

    Handles WebArena Verified format (task_id, sites, start_urls, intent, eval)
    and original WebArena format (task_id, sites, start_url, intent, eval) with
    graceful fallbacks.
    """
    task_id = raw.get("task_id")
    if task_id is None:
        task_id = raw.get("id", "unknown")

    sites = raw.get("sites", [])
    # Primary site is the first entry; multi-site tasks list all sites
    site = sites[0] if sites else "unknown"

    # WebArena Verified uses start_urls (list), original uses start_url (string)
    start_urls = raw.get("start_urls", [])
    if not start_urls and "start_url" in raw:
        start_urls = [raw["start_url"]]

    return {
        "id": str(task_id),
        "site": site,
        "sites": sites,
        "instruction": raw.get("intent", raw.get("instruction", "")),
        "start_urls": start_urls,
        "data_seed": {
            # Mode A: no extra seeding — environment reset between tasks is sufficient
            "mechanism": "none",
        },
        "reward_function": {
            "eval": raw.get("eval", []),
            "task_id": task_id,
        },
        # Preserve metadata useful for filtering and analysis
        "intent_template_id": raw.get("intent_template_id"),
        "revision": raw.get("revision"),
    }
