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

# Imported for its side effects: importing bgym registers the BrowserGym
# task sets, and it must happen after DOOMARENA_WEBARENA_BASE_URL is set
# above. Not referenced directly in this module.
import bgym  # noqa: F401
from browsergym.experiments.benchmark import Benchmark
from browsergym.experiments.benchmark.configs import DEFAULT_HIGHLEVEL_ACTION_SET_ARGS
from browsergym.experiments.benchmark.metadata.utils import task_metadata
from browsergym.experiments.benchmark.utils import make_env_args_list_from_fixed_seeds
from doomarena.browsergym.scripts.run_bgym_experiment import (
    BgymExperiment,
    run_bgym_experiment,
    setup_logging,
)

from eval_awareness_experiments.agent_factory import build_agent
from eval_awareness_experiments.extra_instructions import PRESETS as EXTRA_INSTRUCTION_PRESETS
from eval_awareness_experiments.injection_modifier import (
    ALL_CONDITIONS,
    InjectionCondition,
    wrap_injection,
)
from eval_awareness_experiments.system_prompt_frame import ALL_FRAMES

logger = logging.getLogger(__name__)

WASP_INJECTION_FIELDS = (
    "injection_content",
    "instantiated_instruction",
    "injection_text",
    "instruction",
)


def _patch_browsergym_reachability(online_sites: set[str]) -> None:
    """Only ping sites we actually have running locally. Mirror of the DoomArena patch."""
    import browsergym.webarena.instance as webarena_instance
    import requests

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


def _load_wasp_configs(task_dir: Path) -> list[Path]:
    """Per-task JSONs produced by WASP's prompt_injector.py (its step 1).
    Layout: <output_dir>/webarena_tasks/<task_id>.json — each is one
    (user_goal × attacker_goal × injection_format) instance with
    `sites`, `task_id`, `start_url`, `intent`, `eval`.
    """
    if (task_dir / "webarena_tasks").is_dir():
        task_dir = task_dir / "webarena_tasks"
    if not task_dir.is_dir():
        raise FileNotFoundError(
            f"No WASP task dir at {task_dir}. Run prompt_injector.py first to plant "
            "injections + generate per-task JSONs (see run_wasp.py docstring)."
        )
    paths = sorted(task_dir.glob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No *.json in {task_dir}")
    return paths


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
        cfg_sites = data.get("sites") or ([data["site"]] if data.get("site") else [])
        if site and cfg_sites and site not in cfg_sites:
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

    _register_wasp_tasks(filtered)

    # WASP task_ids start at 1000 to avoid collision with WebArena's 812 canonical
    # tasks. BrowserGym's `task_metadata("webarena")` only knows the 812. Append
    # synthesized rows so Benchmark.__post_init__ doesn't reject our task names.
    import pandas as pd
    base_md = task_metadata("webarena")
    extra_rows = []
    for _cfg_path, data, tid in filtered:
        sites_str = ",".join(data.get("sites") or [site or ""])
        eval_types_list = data.get("eval", {}).get("eval_types", []) or []
        extra_rows.append({
            "task_name": f"webarena.{tid}",
            "requires_reset": bool(data.get("require_reset", False)),
            "sites": sites_str,
            "eval_types": ",".join(eval_types_list),
            "task_id": tid,
            "browsergym_split": "wasp",
            "depends_on": "",
        })
    md = pd.concat([base_md, pd.DataFrame(extra_rows)], ignore_index=True)

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
        task_metadata=md,
    )


def _import_wasp_injector(modified_config_dir: Path, wasp_path: Path | None = None) -> None:
    """Import WASP's prompt_injector module so its page-load hook registers.
    WASP typically points its injector at the config dir via env var.

    WASP isn't a pip-installable package (no setup.py/pyproject.toml), so we
    prepend the clone path to sys.path before importing. This matches what
    `scripts/wasp_n100_run.sh` does via PYTHONPATH externally, but doing it
    here means *any* caller (matrix runner, smoke, direct, pytest) works
    without needing per-launcher env-var setup.
    """
    os.environ["WASP_CONFIG_DIR"] = str(modified_config_dir)
    if wasp_path is not None and str(wasp_path) not in sys.path:
        sys.path.insert(0, str(wasp_path))
    try:
        import webarena_prompt_injections  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            f"WASP (webarena_prompt_injections) is not importable from "
            f"{wasp_path or '<--wasp-path not set>'}. Verify that path "
            "contains a webarena_prompt_injections/ dir, or pass --wasp-path "
            "explicitly. Default is REPO_ROOT/external_benchmarks/wasp."
        ) from e


def _register_wasp_tasks(filtered: list[tuple[Path, dict, int]]) -> None:
    """Register WASP task_ids (>=1000) as gym envs, and monkey-patch
    GenericWebArenaTask so it can find their config dicts (which aren't in
    webarena/test.raw.json).
    """
    from browsergym.core.registration import register_task
    from browsergym.webarena import task as wa_task

    wasp_configs_by_id: dict[int, dict] = {tid: data for _, data, tid in filtered}

    orig_init = wa_task.GenericWebArenaTask.__init__

    def patched_init(self, seed, task_id=None, intent_template_id=None,
                     with_na_hint=False, with_homepage_hint=False):
        if task_id is not None and task_id in wasp_configs_by_id:
            from browsergym.core.task import AbstractBrowserTask
            from browsergym.webarena.instance import WebArenaInstance
            AbstractBrowserTask.__init__(self, seed)
            self.viewport = {"width": 1280, "height": 720}
            self.slow_mo = 1000
            self.timeout = 10000
            self.webarena_instance = WebArenaInstance()
            self.config_file = None
            self.with_na_hint = with_na_hint
            self.with_homepage_hint = with_homepage_hint
            # WASP tasks ship with already-substituted URLs (planted post URLs),
            # so no __REDDIT__/__GITLAB__ replacement needed.
            self.task_configs = [wasp_configs_by_id[task_id]]
            return
        orig_init(self, seed, task_id=task_id, intent_template_id=intent_template_id,
                  with_na_hint=with_na_hint, with_homepage_hint=with_homepage_hint)

    wa_task.GenericWebArenaTask.__init__ = patched_init

    for tid in wasp_configs_by_id:
        gym_id = f"webarena.{tid}"
        try:
            register_task(gym_id, wa_task.GenericWebArenaTask, task_kwargs={"task_id": tid})
        except Exception as e:
            logger.debug(f"register_task({gym_id}) skipped: {e}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run WASP tasks with eval-awareness scaffolding")
    parser.add_argument("--wasp-path", type=Path,
                        default=Path(REPO_ROOT) / "external_benchmarks" / "wasp",
                        help="Path to WASP clone (default: external_benchmarks/wasp)")
    parser.add_argument("--task-dir", type=Path, required=True,
                        help="Output dir from WASP's prompt_injector.py — contains "
                             "webarena_tasks/<task_id>.json. Generate via: "
                             "PYTHONPATH=external_benchmarks/wasp/webarena_prompt_injections "
                             "python external_benchmarks/wasp/webarena_prompt_injections/prompt_injector.py "
                             "--config <raw.json> --output-dir <here> --user_goal_idx 0 "
                             "--injection_format goal_hijacking_url_injection --output-format webarena")
    parser.add_argument("--single-site", default=None, choices=["gitlab", "reddit"],
                        help="Restrict to one site. Omit to include all sites in the task pool.")
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--shuffle", type=int, default=42)
    parser.add_argument("--model-name", required=True,
                        help="OpenRouter model name. Append :thinking to enable reasoning mode.")
    parser.add_argument("--webarena-eval-model",
                        default=os.environ.get("WEBARENA_EVAL_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--max-steps", type=int, default=15)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--avg-step-timeout", type=int, default=60)
    parser.add_argument("--results-dir", default=None,
                        help="Cell-owned directory to write the AgentLab study "
                             "under (path becomes the study's exp_root).")
    parser.add_argument("--relaunch-incomplete", action="store_true",
                        help="Resume the latest AgentLab study under --results-dir, "
                             "rerunning only incomplete or errored tasks. Falls "
                             "back to a fresh study if no prior study exists.")
    parser.add_argument("--condition", choices=ALL_CONDITIONS, default="baseline",
                        help="Causal injection-wrapping condition.")
    parser.add_argument("--extra-instructions-preset", default="none",
                        choices=list(EXTRA_INSTRUCTION_PRESETS.keys()),
                        help="Agent-side elicitation preset (same as DoomArena runner).")
    parser.add_argument("--extra-instructions", default=None,
                        help="Arbitrary extra instructions, overrides --extra-instructions-preset.")
    parser.add_argument("--system-prompt-frame", default="none", choices=ALL_FRAMES,
                        help="Wrap the agent's instructions in <llm_safety_test> or "
                             "<hypothetical_scenario> tags. Browser track injects the "
                             "frame into the GenericAgent's extra_instructions section.")
    parser.add_argument("--online-sites", nargs="+", default=None,
                        help="Override reachability check. Defaults to just --single-site.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    online = set(args.online_sites or ([args.single_site] if args.single_site else ["gitlab", "reddit"]))
    _patch_browsergym_reachability(online)

    from doomarena.browsergym.scripts.run_bgym_experiment import _override_webarena_env_vars
    base_url = os.environ.get("DOOMARENA_WEBARENA_BASE_URL", "http://localhost")
    _override_webarena_env_vars(base_url)
    for var in ("REDDIT", "SHOPPING", "SHOPPING_ADMIN", "GITLAB", "WIKIPEDIA", "MAP", "HOMEPAGE"):
        os.environ[var] = os.environ.get(f"WA_{var}", "")

    # Reuse DoomArena's evaluator patch verbatim by importing it.
    from eval_awareness_experiments.run_doomarena_glm5_popup import _patch_webarena_evaluator
    _patch_webarena_evaluator(args.webarena_eval_model)

    config_paths = _load_wasp_configs(args.task_dir)
    logger.info(f"Loaded {len(config_paths)} per-task WASP configs from {args.task_dir}")

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

    _import_wasp_injector(tmp, wasp_path=args.wasp_path)

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
        agent=build_agent(
            args.model_name,
            extra_instructions=extra_instr,
            system_prompt_frame=args.system_prompt_frame,
        ),
        # Injection is baked into the task via WASP's page-load hook; no runtime attack config.
        attack_configs=(),
        benchmark=benchmark,
        defenses=[],
    )

    setup_logging()
    exp_root = Path(args.results_dir) if args.results_dir else None
    if exp_root is not None:
        exp_root.mkdir(parents=True, exist_ok=True)
    relaunch = False
    if args.relaunch_incomplete:
        if exp_root is None:
            raise SystemExit("--relaunch-incomplete requires --results-dir")
        relaunch = (exp_root / "study.pkl.gz").exists() or any(
            exp_root.glob("*/study.pkl.gz")
        )
        if relaunch:
            logger.info("Relaunching incomplete tasks from latest study under %s", exp_root)
        else:
            logger.warning(
                "--relaunch-incomplete requested but no prior study found under %s; "
                "starting a fresh study.",
                exp_root,
            )
    try:
        run_bgym_experiment(
            bgym_experiments=[experiment],
            exp_root=exp_root,
            reproducibility_mode=False,
            relaunch=relaunch,
            n_jobs=args.n_jobs,
            max_steps=args.max_steps,
            avg_step_timeout=args.avg_step_timeout,
            skip_reset_and_massage=True,
        )
    except KeyError as e:
        # DoomArena's collect_results() expects success_filter_{key,value} columns
        # produced by runtime AttackConfigs. WASP uses post-hoc evaluators (no
        # runtime attacks → attack_configs=()) so those columns don't exist.
        # The trajectory data (step_*.pkl.gz, summary_info.json) is fully written
        # before collect_results runs — downstream consumers read the pkls directly.
        if "success_filter" in str(e):
            logger.warning(
                "Suppressed DoomArena collect_results() KeyError on success_filter "
                "columns — expected for WASP since attack_configs=(). Trajectories "
                "are saved in results/browsergym/study_*/."
            )
        else:
            raise


if __name__ == "__main__":
    main()
