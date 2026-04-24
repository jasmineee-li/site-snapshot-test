"""Phase 1 Mode A helpers: load benchmark tasks and wrap them for WorldSim."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def build_mode_a_tasks(
    manifest: dict[str, Any],
    benchmark_root: Path,
    profiles_dir: Path | None = None,
    benchmark: str | None = None,
) -> list[dict[str, Any]]:
    """Load raw benchmark tasks and wrap them into benign task bundles.

    Args:
        profiles_dir: Path to Phase 0c output directory. When provided,
            ``AGENT_CONTEXT_{site}.json`` files are loaded and embedded
            in each wrapped task as the ``agent_context`` field.
    """
    agent_contexts = _load_agent_contexts(profiles_dir) if profiles_dir else {}
    return [
        _wrap_task(task, agent_contexts=agent_contexts, benchmark=benchmark)
        for task in _load_raw_tasks(manifest, benchmark_root)
    ]


def _load_agent_contexts(profiles_dir: Path) -> dict[str, dict[str, Any]]:
    """Load AGENT_CONTEXT_{site}.json files from Phase 0c output."""
    contexts: dict[str, dict[str, Any]] = {}
    for path in Path(profiles_dir).glob("AGENT_CONTEXT_*.json"):
        site_name = path.stem.replace("AGENT_CONTEXT_", "")
        try:
            contexts[site_name] = json.loads(path.read_text())
            logger.info("Loaded agent context for site %r from %s", site_name, path)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load agent context for site %r: %s", site_name, exc)
    return contexts


def _load_raw_tasks(manifest: dict[str, Any], benchmark_root: Path) -> list[dict[str, Any]]:
    """Load task definitions from paths declared in the manifest.

    Handles two formats:
    - Single JSON array file (WebArena Verified: webarena-verified.json)
    - Directory of per-task JSON files (original WebArena: config_files/)
    """
    task_paths = manifest.get("evaluation", {}).get("task_definition_paths", [])
    all_tasks: list[dict[str, Any]] = []

    for task_path in task_paths:
        full_path = benchmark_root / task_path
        if full_path.is_file() and full_path.suffix == ".json":
            data = json.loads(full_path.read_text())
            if isinstance(data, list):
                all_tasks.extend(data)
            else:
                all_tasks.append(data)
        elif full_path.is_dir():
            for path in sorted(full_path.rglob("*.json")):
                try:
                    data = json.loads(path.read_text())
                except json.JSONDecodeError:
                    logger.warning("Skipping non-JSON file: %s", path)
                    continue
                if isinstance(data, list):
                    all_tasks.extend(data)
                else:
                    all_tasks.append(data)

    return all_tasks


def _wrap_task(
    raw: dict[str, Any],
    agent_contexts: dict[str, dict[str, Any]] | None = None,
    benchmark: str | None = None,
) -> dict[str, Any]:
    """Wrap a single benchmark task into the benign task bundle schema."""
    task_id = raw.get("task_id")
    if task_id is None:
        task_id = raw.get("id", "unknown")

    sites = raw.get("sites", [])
    site = sites[0] if sites else "unknown"

    start_urls = raw.get("start_urls", [])
    if not start_urls and "start_url" in raw:
        start_urls = [raw["start_url"]]

    wrapped: dict[str, Any] = {
        "id": str(task_id),
        "site": site,
        "sites": sites,
        "instruction": raw.get("intent", raw.get("instruction", "")),
        "start_urls": start_urls,
        "data_seed": {
            "mechanism": "none",
        },
        "reward_function": {
            "eval": raw.get("eval", []),
            "task_id": task_id,
        },
        "intent_template_id": raw.get("intent_template_id"),
        "revision": raw.get("revision"),
    }
    if benchmark:
        wrapped["benchmark"] = benchmark
    if "instantiation_dict" in raw:
        wrapped["instantiation_dict"] = json.loads(json.dumps(raw["instantiation_dict"]))

    if agent_contexts and site in agent_contexts:
        # The embedded context can include benchmark-issued test credentials.
        # Keeping it on the wrapped task is intentional so downstream phases can
        # replay the same benchmark contract without re-reading Phase 0c files.
        wrapped["agent_context"] = json.loads(json.dumps(agent_contexts[site]))

    return wrapped
