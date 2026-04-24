"""Run WASP (Web Agent Security against Prompt injection) via BrowserGym + AgentLab.

Mirrors the DoomArena runner (eval_awareness_experiments/run_doomarena_glm5_popup.py)
so reasoning tokens are captured through the same patched AgentLab ChatModel path.
Differences from DoomArena:
- Attack lives inside the task config (injection baked into a page element / URL),
  not a runtime AttackConfig. So `attack_configs=()`.
- Task source is WASP's `webarena_prompt_injections/configs/*.json`.
- Conditional injection wrapping via `injection_modifier.wrap_injection` happens
  on the config fields before the task is registered.

Prereqs:
- `bash eval_awareness_experiments/setup_benchmarks.sh wasp`
- VisualWebArena GitLab + Reddit containers running (see SAFETY_BENCHMARKS_HANDOFF.md)
- OPENROUTER_API_KEY in env or .env
"""

from __future__ import annotations

import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(THIS_DIR)
if sys.path and os.path.abspath(sys.path[0]) == THIS_DIR:
    sys.path.pop(0)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import argparse
import json
import logging
import random
import tempfile
from pathlib import Path

os.environ.setdefault("DOOMARENA_WEBARENA_BASE_URL", "http://localhost")

import bgym
from browsergym.experiments.benchmark import Benchmark
from browsergym.experiments.benchmark.configs import DEFAULT_HIGHLEVEL_ACTION_SET_ARGS
from browsergym.experiments.benchmark.metadata.utils import task_metadata
from browsergym.experiments.benchmark.utils import make_env_args_list_from_fixed_seeds
from doomarena.browsergym.scripts.run_bgym_experiment import (
    BgymExperiment,
    run_bgym_experiment,
    setup_logging,
)

from eval_awareness_experiments.agent_factory import build_openrouter_agent
from eval_awareness_experiments.extra_instructions import PRESETS as EXTRA_INSTRUCTION_PRESETS
from eval_awareness_experiments.injection_modifier import (
    ALL_CONDITIONS,
    InjectionCondition,
    wrap_injection,
)

logger = logging.getLogger(__name__)

WASP_INJECTION_FIELDS = (
    "injection_content",
    "instantiated_instruction",
    "injection_text",
    "instruction",
)


def _patch_browsergym_reachability(online_sites: set[str]) -> None:
    """Only ping sites we actually have running locally. Mirror of the DoomArena patch."""
    import requests
    import browsergym.webarena.instance as webarena_instance

    def _check_is_reachable(self, timeout):
        for site, url in self.urls.items():
            if site not in online_sites:
                continue
            try:
                requests.get(url, timeout=timeout)
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                raise RuntimeError(
                    f'WebArena site "{site}" ({url}) is not reachable: {e}'
                ) from e

    webarena_instance.WebArenaInstance._check_is_reachable = _check_is_reachable


def _load_wasp_configs(wasp_path: Path) -> list[Path]:
    """Find WASP task config JSONs. WASP's upstream layout is
    `webarena_prompt_injections/configs/`; some forks flatten to `configs/`.
    """
    for rel in ("webarena_prompt_injections/configs", "configs"):
        candidate = wasp_path / rel
        if candidate.is_dir():
            return sorted(candidate.glob("*.json"))
    raise FileNotFoundError(
        f"No WASP config dir under {wasp_path} "
        f"(expected {wasp_path}/webarena_prompt_injections/configs or {wasp_path}/configs). "
        "Did setup_benchmarks.sh clone WASP correctly?"
    )


def _apply_condition_to_config(config: dict, condition: InjectionCondition) -> dict:
    """Wrap every injection-carrying field with the causal-condition XML tag.
    No-op when condition is 'baseline'.
    """
    if condition == "baseline":
        return config

    modified = dict(config)
    modified_any = False
    for key in WASP_INJECTION_FIELDS:
        if key in modified and isinstance(modified[key], str):
            modified[key] = wrap_injection(modified[key], condition)
            modified_any = True

    nested = modified.get("prompt_injection_config")
    if isinstance(nested, dict):
        nested_mod = dict(nested)
        for key in WASP_INJECTION_FIELDS:
            if key in nested_mod and isinstance(nested_mod[key], str):
                nested_mod[key] = wrap_injection(nested_mod[key], condition)
                modified_any = True
        modified["prompt_injection_config"] = nested_mod

    if not modified_any:
        logger.warning(
            "No recognized injection field in WASP config; condition wrapping had no effect. "
            f"Known field names: {WASP_INJECTION_FIELDS}. Config keys: {list(config)}"
        )
    return modified


def _build_wasp_benchmark(
    config_dir: Path,
    site: str,
    max_tasks: int,
    shuffle: int | None,
    max_steps: int,
) -> Benchmark:
    """Build a BrowserGym Benchmark over WASP task configs for a single site.

    WASP configs sit alongside VWA task metadata — each config has a `task_id`
    matching a VWA task; running it through our `run_bgym_experiment` invokes
    the VWA task with WASP's injection already baked into the page (via WASP's
    `prompt_injector` page-load hook, which registers automatically on import).
    """
    configs = sorted(config_dir.glob("*.json"))
    if not configs:
        raise FileNotFoundError(f"No WASP configs in {config_dir}")

    filtered = []
    for cfg_path in configs:
        try:
            data = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            logger.warning(f"Skipping malformed WASP config {cfg_path}")
            continue
        cfg_site = data.get("site") or data.get("target_site")
        if site and cfg_site and cfg_site != site:
            continue
        task_id = data.get("task_id")
        if task_id is None:
            continue
        filtered.append((cfg_path, data, task_id))

    if shuffle is not None:
        rnd = random.Random(shuffle)
        rnd.shuffle(filtered)
    filtered = filtered[:max_tasks]

    task_list = [f"webarena.{t[2]}" for t in filtered]
    if not task_list:
        raise RuntimeError(
            f"No WASP tasks survived filtering (site={site!r}, max_tasks={max_tasks})"
        )

    logger.info(f"WASP benchmark: {len(task_list)} tasks for site={site!r}")
    return Benchmark(
        name=f"wasp_{site}_{len(task_list)}",
        high_level_action_set_args=DEFAULT_HIGHLEVEL_ACTION_SET_ARGS["webarena"],
        is_multi_tab=True,
        supports_parallel_seeds=True,
        backends=["webarena"],
        env_args_list=make_env_args_list_from_fixed_seeds(
            task_list=task_list,
            max_steps=max_steps,
            fixed_seeds=[0],
        ),
        task_metadata=task_metadata("webarena"),
    )


def _import_wasp_injector(modified_config_dir: Path) -> None:
    """Import WASP's prompt_injector module so its page-load hook registers.
    WASP typically points its injector at the config dir via env var.
    """
    os.environ["WASP_CONFIG_DIR"] = str(modified_config_dir)
    try:
        import webarena_prompt_injections  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "WASP (webarena_prompt_injections) is not importable. "
            "Run: bash eval_awareness_experiments/setup_benchmarks.sh wasp"
        ) from e


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WASP tasks with eval-awareness scaffolding")
    parser.add_argument("--wasp-path", type=Path,
                        default=Path(REPO_ROOT) / "external_benchmarks" / "wasp",
                        help="Path to WASP clone (default: external_benchmarks/wasp)")
    parser.add_argument("--single-site", required=True, choices=["gitlab", "reddit"],
                        help="WASP currently supports gitlab and reddit (VWA-based).")
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=42)
    parser.add_argument("--model-name", required=True,
                        help="OpenRouter model name. Append :thinking to enable reasoning mode.")
    parser.add_argument("--webarena-eval-model",
                        default=os.environ.get("WEBARENA_EVAL_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--avg-step-timeout", type=int, default=60)
    parser.add_argument("--condition", choices=ALL_CONDITIONS, default="baseline",
                        help="Causal injection-wrapping condition.")
    parser.add_argument("--extra-instructions-preset", default="none",
                        choices=list(EXTRA_INSTRUCTION_PRESETS.keys()),
                        help="Agent-side elicitation preset (same as DoomArena runner).")
    parser.add_argument("--extra-instructions", default=None,
                        help="Arbitrary extra instructions, overrides --extra-instructions-preset.")
    parser.add_argument("--online-sites", nargs="+", default=None,
                        help="Override reachability check. Defaults to just --single-site.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    online = set(args.online_sites or [args.single_site])
    _patch_browsergym_reachability(online)

    from doomarena.browsergym.scripts.run_bgym_experiment import _override_webarena_env_vars
    base_url = os.environ.get("DOOMARENA_WEBARENA_BASE_URL", "http://localhost")
    _override_webarena_env_vars(base_url)
    for var in ("REDDIT", "SHOPPING", "SHOPPING_ADMIN", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE"):
        os.environ.setdefault(var, os.environ.get(f"WA_{var}", ""))

    # Reuse DoomArena's evaluator patch verbatim by importing it.
    from eval_awareness_experiments.run_doomarena_glm5_popup import _patch_webarena_evaluator
    _patch_webarena_evaluator(args.webarena_eval_model)

    config_paths = _load_wasp_configs(args.wasp_path)
    logger.info(f"Loaded {len(config_paths)} raw WASP configs from {args.wasp_path}")

    tmp = Path(tempfile.mkdtemp(prefix=f"wasp_{args.condition}_"))
    n_wrapped = 0
    for cfg_path in config_paths:
        try:
            data = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            logger.warning(f"Skipping malformed {cfg_path}")
            continue
        modified = _apply_condition_to_config(data, args.condition)
        (tmp / cfg_path.name).write_text(json.dumps(modified, indent=2))
        n_wrapped += 1
    logger.info(f"Wrote {n_wrapped} condition-wrapped configs → {tmp}")

    _import_wasp_injector(tmp)

    benchmark = _build_wasp_benchmark(
        config_dir=tmp,
        site=args.single_site,
        max_tasks=args.max_tasks,
        shuffle=args.shuffle,
        max_steps=args.max_steps,
    )

    extra_instr = args.extra_instructions or EXTRA_INSTRUCTION_PRESETS.get(
        args.extra_instructions_preset
    )
    experiment = BgymExperiment(
        agent=build_openrouter_agent(args.model_name, extra_instructions=extra_instr),
        # Injection is baked into the task via WASP's page-load hook; no runtime attack config.
        attack_configs=(),
        benchmark=benchmark,
        defenses=[],
    )

    setup_logging()
    run_bgym_experiment(
        bgym_experiments=[experiment],
        reproducibility_mode=False,
        relaunch=False,
        n_jobs=args.n_jobs,
        max_steps=args.max_steps,
        skip_reset_and_massage=True,
    )


if __name__ == "__main__":
    main()
