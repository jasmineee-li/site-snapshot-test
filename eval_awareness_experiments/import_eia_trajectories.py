"""Import EIA (Environmental Injection Attack) trajectories from the authors'
`eval_results.zip` into our normalized per-task layout.

NO new model runs. EIA is import-only: we take the paper's pre-recorded SeeAct
trajectories, normalize them so `parse_eia_trajectory` can read them, and emit
a manifest listing the selected tasks. Judge pass runs against these later via
`run_safety_pipeline --benchmark eia --stage judge-only`.

Expected input (after `bash eval_awareness_experiments/setup_benchmarks.sh eia`):
    external_benchmarks/eia/eval_results/<task_dir>/...

SeeAct trajectory layouts vary; this importer probes common shapes and fails
loudly with a clear diagnostic if nothing matches, so a human can adjust rather
than silently dropping data.

Output layout (per selected task):
    data/eia/<task_id>/
        task.json        # {instruction, attack_type, attack_success, ...}
        trajectory.json  # [{thought, action, observation}, ...]
    data/eia/manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
from pathlib import Path

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

logger = logging.getLogger(__name__)

DEFAULT_EIA_PATH = Path(REPO_ROOT) / "external_benchmarks" / "eia" / "eval_results"
DEFAULT_OUT_DIR = Path(THIS_DIR) / "data" / "eia"

# Known keys across SeeAct-style logs. We accept any that appear.
_THOUGHT_KEYS = ("thought", "reasoning", "rationale", "analysis", "planning", "Thought")
_ACTION_KEYS = ("action", "grounded_action", "next_action", "Action", "operation")
_OBSERVATION_KEYS = ("observation", "element_text", "element", "page_state", "Observation")
_INSTRUCTION_KEYS = ("instruction", "task", "task_description", "goal", "confirmed_task")
_ATTACK_SUCCESS_KEYS = ("attack_success", "injection_success", "is_successful_attack", "attack_status")
_ATTACK_TYPE_KEYS = ("attack_type", "injection_type", "eia_type", "attack")


def _first_present(d: dict, keys) -> str | None:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            v = d[k]
            return v if isinstance(v, str) else json.dumps(v, default=str, ensure_ascii=False)
    return None


def _discover_task_dirs(eia_root: Path) -> list[Path]:
    """Find per-task directories under the EIA eval_results root.
    Heuristic: any directory that contains a JSON or JSONL file.
    """
    if not eia_root.exists():
        raise FileNotFoundError(
            f"EIA eval_results not found at {eia_root}.\n"
            "1) Run `bash eval_awareness_experiments/setup_benchmarks.sh eia` to clone the repo.\n"
            "2) Fetch eval_results.zip from the authors' Globus endpoint "
            "(not included in the git repo):\n"
            "   https://app.globus.org/file-manager?origin_id=6e0b9952-da25-4b74-a4f9-7450f0bb96b9&origin_path=%2F\n"
            "3) Place it at external_benchmarks/eia/eval_results.zip and re-run setup_benchmarks.sh eia."
        )

    candidates: list[Path] = []
    for entry in sorted(eia_root.rglob("*")):
        if not entry.is_dir():
            continue
        has_log = any(entry.glob("*.json")) or any(entry.glob("*.jsonl"))
        if has_log:
            candidates.append(entry)
    # Deduplicate parent → child (prefer leaves).
    leaves = []
    candidate_set = set(candidates)
    for c in candidates:
        if not any(other != c and other.is_relative_to(c) for other in candidate_set):
            leaves.append(c)
    if not leaves:
        raise RuntimeError(
            f"No task directories with JSON/JSONL logs found under {eia_root}. "
            "The EIA release format may have changed — inspect the zip layout "
            "and adjust `_discover_task_dirs` + `_load_trajectory` accordingly."
        )
    return leaves


def _load_trajectory(task_dir: Path) -> tuple[list[dict], dict]:
    """Return (steps, task_metadata) from a SeeAct task directory.

    steps: list of {thought, action, observation} records.
    task_metadata: at minimum {instruction}; may include attack_type, attack_success.
    """
    jsonl_files = sorted(task_dir.glob("*.jsonl"))
    json_files = sorted(task_dir.glob("*.json"))

    records: list[dict] = []
    metadata: dict = {}

    for fp in jsonl_files:
        try:
            for line in fp.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                records.append(json.loads(line))
        except json.JSONDecodeError as e:
            logger.warning(f"{fp}: malformed JSONL line, skipping — {e}")

    for fp in json_files:
        try:
            data = json.loads(fp.read_text())
        except json.JSONDecodeError as e:
            logger.warning(f"{fp}: malformed JSON — {e}")
            continue
        if isinstance(data, list):
            records.extend(data)
        elif isinstance(data, dict):
            # Either a single-step record, or a top-level doc with a `steps`/`trajectory` list.
            for key in ("steps", "trajectory", "actions", "history"):
                if key in data and isinstance(data[key], list):
                    records.extend(data[key])
                    for mk, mv in data.items():
                        if mk == key or not isinstance(mv, (str, int, float, bool)):
                            continue
                        metadata.setdefault(mk, mv)
                    break
            else:
                # Might be a metadata-only file (task.json, meta.json).
                instr = _first_present(data, _INSTRUCTION_KEYS)
                if instr:
                    metadata.setdefault("instruction", instr)
                for k in _ATTACK_SUCCESS_KEYS:
                    if k in data:
                        metadata.setdefault("attack_success", data[k])
                for k in _ATTACK_TYPE_KEYS:
                    if k in data:
                        metadata.setdefault("attack_type", data[k])
                # If record-like, also include as a step.
                if any(k in data for k in _THOUGHT_KEYS + _ACTION_KEYS + _OBSERVATION_KEYS):
                    records.append(data)

    if not records:
        raise RuntimeError(
            f"No trajectory records found in {task_dir}. "
            "Inspect the task dir manually and update this importer if EIA's "
            "log format differs from {jsonl, json with `steps` list, record-per-file} shapes."
        )

    steps = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        thought = _first_present(rec, _THOUGHT_KEYS) or ""
        action = _first_present(rec, _ACTION_KEYS) or ""
        observation = _first_present(rec, _OBSERVATION_KEYS) or ""
        if not (thought or action or observation):
            continue
        steps.append({"thought": thought, "action": action, "observation": observation})

        metadata.setdefault("instruction", _first_present(rec, _INSTRUCTION_KEYS) or "")
        for k in _ATTACK_SUCCESS_KEYS:
            if k in rec and "attack_success" not in metadata:
                metadata["attack_success"] = rec[k]
        for k in _ATTACK_TYPE_KEYS:
            if k in rec and "attack_type" not in metadata:
                metadata["attack_type"] = rec[k]

    if not steps:
        raise RuntimeError(
            f"Parsed {len(records)} records from {task_dir} but none had "
            "thought/action/observation keys. Known field aliases: "
            f"{_THOUGHT_KEYS + _ACTION_KEYS + _OBSERVATION_KEYS}"
        )
    if not metadata.get("instruction"):
        raise RuntimeError(
            f"No instruction found in {task_dir}. Known aliases: {_INSTRUCTION_KEYS}"
        )
    return steps, metadata


def _normalize_attack_success(val) -> bool | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        v = val.strip().lower()
        if v in {"true", "1", "yes", "succ", "success", "successful"}:
            return True
        if v in {"false", "0", "no", "unsucc", "failure", "failed"}:
            return False
    return None


def _write_task(
    *,
    task_id: str,
    steps: list[dict],
    metadata: dict,
    source_dir: Path,
    out_dir: Path,
) -> Path:
    task_out = out_dir / task_id
    task_out.mkdir(parents=True, exist_ok=True)

    (task_out / "trajectory.json").write_text(
        json.dumps(steps, indent=2, ensure_ascii=False)
    )
    task_json = {
        "task_id": task_id,
        "instruction": metadata.get("instruction", ""),
        "attack_type": metadata.get("attack_type", "environmental_injection"),
        "attack_success": _normalize_attack_success(metadata.get("attack_success")),
        "source_path": str(source_dir),
        "agent": "seeact",
        "benchmark": "eia",
        "injection_condition": "baseline",
    }
    (task_out / "task.json").write_text(json.dumps(task_json, indent=2, ensure_ascii=False))
    return task_out


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import EIA SeeAct trajectories (no new runs).")
    parser.add_argument("--eia-path", type=Path, default=DEFAULT_EIA_PATH,
                        help="Path to unzipped eval_results/ dir.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="Normalized output dir.")
    parser.add_argument("--n", type=int, default=20, help="Number of trajectories to select.")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed.")
    parser.add_argument("--prefer-success", action="store_true",
                        help="Prefer trajectories tagged attack_success=true.")
    parser.add_argument("--clean", action="store_true",
                        help="Remove the output dir before writing.")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args()

    task_dirs = _discover_task_dirs(args.eia_path)
    logger.info(f"Found {len(task_dirs)} candidate task directories under {args.eia_path}")

    # Load all, filter valid, optionally prefer attack-success=true.
    loaded: list[tuple[Path, list[dict], dict]] = []
    for td in task_dirs:
        try:
            steps, meta = _load_trajectory(td)
            loaded.append((td, steps, meta))
        except Exception as e:
            logger.warning(f"Skipping {td}: {e}")

    if not loaded:
        raise RuntimeError("No EIA trajectories could be parsed. Fix the importer and retry.")

    if args.prefer_success:
        successful = [x for x in loaded if _normalize_attack_success(x[2].get("attack_success")) is True]
        if len(successful) >= args.n:
            loaded = successful
            logger.info(f"Restricted to {len(loaded)} attack-success=true trajectories")
        else:
            logger.info(
                f"Only {len(successful)} attack-success=true — sampling from all {len(loaded)} instead"
            )

    rnd = random.Random(args.seed)
    rnd.shuffle(loaded)
    selected = loaded[: args.n]
    logger.info(f"Selected {len(selected)} trajectories (seed={args.seed})")

    if args.clean and args.out_dir.exists():
        logger.info(f"Cleaning {args.out_dir}")
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "benchmark": "eia",
        "source_root": str(args.eia_path),
        "seed": args.seed,
        "n_selected": len(selected),
        "tasks": [],
    }
    for idx, (src, steps, meta) in enumerate(selected):
        # Task id = stable hash of source path, plus an index for readability.
        short = src.name.replace(" ", "_") or f"task_{idx:03d}"
        task_id = f"eia_{idx:03d}_{short}"
        task_path = _write_task(
            task_id=task_id,
            steps=steps,
            metadata=meta,
            source_dir=src,
            out_dir=args.out_dir,
        )
        manifest["tasks"].append({
            "task_id": task_id,
            "path": str(task_path.relative_to(args.out_dir)),
            "instruction": meta.get("instruction", "")[:200],
            "attack_success": _normalize_attack_success(meta.get("attack_success")),
            "n_steps": len(steps),
        })

    manifest_path = args.out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    logger.info(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
