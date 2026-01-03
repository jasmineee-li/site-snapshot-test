import gzip
import json
import logging
import os
import pickle
import random
import time
import uuid
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from multiprocessing import Manager, Pool, Queue
from pathlib import Path

import bgym
from bgym import DEFAULT_BENCHMARKS, Benchmark
from slugify import slugify

from agentlab.agents.agent_args import AgentArgs
from agentlab.analyze import inspect_results
from agentlab.benchmarks.abstract_env import AbstractEnvArgs
from agentlab.experiments import reproducibility_util as repro
from agentlab.experiments.exp_utils import RESULTS_DIR, add_dependencies
from agentlab.experiments.launch_exp import (
    find_incomplete,
    non_dummy_count,
    run_experiments,
)
from agentlab.experiments.loop import EnvArgs, ExpArgs
from agentlab.experiments.multi_server import BaseServer
from agentlab.benchmarks.redteam_attacker import RedteamAttackerAgent

logger = logging.getLogger(__name__)


def make_study(
    agent_args: list[AgentArgs] | AgentArgs,
    benchmark: Benchmark | str,
    logging_level=logging.WARNING,
    logging_level_stdout=logging.WARNING,
    suffix="",
    comment=None,
    ignore_dependencies=False,
    parallel_servers=None,
):
    """Run a list of agents on a benchmark.

    Args:
        agent_args: list[AgentArgs] | AgentArgs
            The agent configuration(s) to run. *IMPORTANT*: these objects will be pickled and
            unpickled.  Make sure they are imported from a package that is accessible from
            PYTHONPATH. Otherwise, it won't load in agentlab-xray.
        benchmark: Benchmark | str
            The benchmark to run the agents on. See DEFAULT_BENCHMARKS for the main ones. You
            can also make your own by modifying an existing one.
        logging_level: int
            The logging level for file log.
        logging_level_stdout: int
            The logging level for the stdout of the main script. Each job will have its own logging
            level that will save into file and can be seen in agentlab-xray.
        suffix: str
            A suffix to add to the study name. This can be useful to keep track of your experiments.
            By default the study name contains agent name, benchmark name and date.
        comment: str
            Extra comments from the authors of this study to be stored in the reproducibility
            information. Leave any extra information that can explain why results could be different
            than expected.
        ignore_dependencies: bool
            If True, ignore the dependencies of the tasks in the benchmark. *Use with caution.* So
            far, only WebArena and VisualWebArena have dependencies between tasks to minimize the
            influence of solving one task before another one. This dependency graph allows
            experiments to run in parallel while respecting task dependencies. However, it still
            can't run more than 4 and, in practice it's speeding up evaluation by a factor of only
            3x compare to sequential executionz. To accelerate execution, you can ignore
            dependencies and run in full parallel. This leads to a decrease in performance of about
            1%-2%, and could be more. Note: ignore_dependencies on VisualWebArena doesn't work.
        parallel_servers: list[WebArenaInstanceVars]
            The number of parallel servers to use `if "webarena" in benchmark.name`. Use this to
            dispatch agent_args on a pool of servers in parallel. If len(agent_args) >
            len(parallel_servers), the servers will be reused for next evaluation (with a reset) as
            soon as it is done.

    Returns:
        Study | SequentialStudies | ParallelStudies object.
            SequentialStudies: if the benchmark requires manual reset after each evaluation such as
                WebArena and VisualWebArena.
            ParallelStudies: if the benchmark has multiple servers to run in parallel.
            Study: otherwise.
    """

    if not isinstance(agent_args, (list, tuple)):
        agent_args = [agent_args]

    if isinstance(benchmark, str):
        benchmark = DEFAULT_BENCHMARKS[benchmark.lower()]()

    if len(agent_args) > 1 and ("webarena" in benchmark.name or parallel_servers is not None):
        logger.warning(
            "*WebArena* requires manual reset after each evaluation. Running through SequentialStudies."
        )
        studies = []
        for agent in agent_args:
            studies.append(
                Study(
                    [agent],
                    benchmark,
                    logging_level=logging_level,
                    logging_level_stdout=logging_level_stdout,
                    suffix=suffix,
                    comment=comment,
                    ignore_dependencies=ignore_dependencies,
                )
            )
        if parallel_servers is not None:
            return ParallelStudies(studies, parallel_servers=parallel_servers)
        else:
            return SequentialStudies(studies)
    else:
        return Study(
            agent_args,
            benchmark,
            logging_level=logging_level,
            logging_level_stdout=logging_level_stdout,
            suffix=suffix,
            comment=comment,
            ignore_dependencies=ignore_dependencies,
        )


class AbstractStudy(ABC):
    """Abstract class for a study."""

    dir: Path = None
    suffix: str = ""

    @abstractmethod
    def find_incomplete(self, include_errors=True):
        """Prepare the study for relaunching by finding incomplete experiments"""

    @abstractmethod
    def run(self, n_jobs=1, parallel_backend="ray", strict_reproducibility=False, n_relaunch=3):
        """Run the study"""

    def make_dir(self, exp_root=RESULTS_DIR):
        """Create a directory for the study"""
        if self.dir is None:
            dir_name = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{self.name}"

            self.dir = Path(exp_root) / dir_name
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, exp_root=RESULTS_DIR):
        """Pickle the study to the directory"""
        # TODO perhaps remove exp_args_list before pickling and when loading bring them from the individual directories

        self.make_dir(exp_root=exp_root)
        with gzip.open(self.dir / "study.pkl.gz", "wb") as f:
            pickle.dump(self, f)

    def get_results(self, suffix="", also_save=False):
        """Recursively load all results from the study directory and summarize them."""
        result_df = inspect_results.load_result_df(self.dir)
        error_report = inspect_results.error_report(result_df, max_stack_trace=3, use_log=True)
        summary_df = inspect_results.summarize_study(result_df)

        return result_df, summary_df, error_report

    def shuffle_exps(self):
        """Shuffle the experiments in the study."""
        self.exp_args_list = random.sample(self.exp_args_list, len(self.exp_args_list))


@dataclass
class Study(AbstractStudy):
    """A study coresponds to one or multiple agents evaluated on a benchmark.

    This is part of the high level API to help keep experiments organized and reproducible.

    Attributes:
        agent_args: list[AgentArgs]
            The agent configuration(s) to run. *IMPORTANT*: these objects will be pickled and
            unpickled.  Make sure they are imported from a package that is accessible from
            PYTHONPATH. Otherwise, it won't load in agentlab-xray.
        benchmark: Benchmark | str
            The benchmark to run the agents on. See DEFAULT_BENCHMARKS for the main ones. You
            can also make your own by modifying an existing one.
        dir: Path
            The directory where the study will be saved. If None, a directory will be created in
            RESULTS_DIR.
        suffix: str
            A suffix to add to the study name. This can be useful to keep track of your experiments.
            By default the study name contains agent name, benchmark name and date.
        uuid: str
            A unique identifier for the study. Will be generated automatically.
        reproducibility_info: dict
            Information about the study that may affect the reproducibility of the experiment. e.g.:
            versions of BrowserGym, benchmark, AgentLab...
        logging_level: int
            The logging level for individual jobs.
        logging_level_stdout: int
            The logging level for the stdout of the main script. Each job will have its own logging
            level that will save into file and can be seen in agentlab-xray.
        comment: str
            Extra comments from the authors of this study to be stored in the reproducibility
            information. Leave any extra information that can explain why results could be different
            than expected.
        ignore_dependencies: bool
            If True, ignore the dependencies of the tasks in the benchmark. *Use with caution*. So
            far, only WebArena and VisualWebArena have dependencies between tasks to minimize the
            influence of solving one task before another one. This dependency graph allows
            experiments to run in parallel while respecting task dependencies. However, it still
            can't run more than 4 and, in practice it's speeding up evaluation by a factor of only
            3x compare to sequential execution. To accelerate execution, you can ignore
            dependencies and run in full parallel. This leads to a decrease in performance of about
            1%-2%, and could be more. Note: ignore_dependencies on VisualWebArena doesn't work.
        avg_step_timeout: int
            The average step timeout in seconds. This is used to stop the experiments if they are
            taking too long. The default is 60 seconds.
        demo_mode: bool
            If True, the experiments will be run in demo mode, which will record videos, and enable
            visual effects for actions.
    """

    agent_args: list[AgentArgs] = None
    benchmark: Benchmark | str = None
    dir: Path = None
    suffix: str = ""  # used for adding a personnal comment to the study name
    uuid: str = None
    reproducibility_info: dict = None
    logging_level: int = logging.DEBUG
    logging_level_stdout: int = logging.WARNING
    comment: str = None  # Extra comments from the authors of this study
    ignore_dependencies: bool = False
    avg_step_timeout: int = 60
    demo_mode: bool = False

    def __post_init__(self):
        """Initialize the study. Set the uuid, and generate the exp_args_list."""
        self.uuid = uuid.uuid4()
        if isinstance(self.benchmark, str):
            self.benchmark = DEFAULT_BENCHMARKS[self.benchmark.lower()]()

        self.benchmark.env_args_list = _convert_env_args(self.benchmark.env_args_list)

        if isinstance(self.dir, str):
            self.dir = Path(self.dir)
        self.make_exp_args_list()

    def make_exp_args_list(self):
        """Generate the exp_args_list from the agent_args and the benchmark."""
        self.exp_args_list = self.agents_on_benchmark(
            self.agent_args,
            self.benchmark,
            logging_level=self.logging_level,
            logging_level_stdout=self.logging_level_stdout,
            ignore_dependencies=self.ignore_dependencies,
            demo_mode=self.demo_mode,
        )

    def find_incomplete(self, include_errors=True):
        """Find incomplete or errored experiments in the study directory for relaunching.

        Args:
            include_errors: bool
                If True, include errored experiments in the list.

        Returns:
            list[ExpArgs]: The list of all experiments with completed ones replaced by a
                dummy exp_args to keep the task dependencies.
        """
        self.exp_args_list = find_incomplete(self.dir, include_errors=include_errors)
        n_incomplete = non_dummy_count(self.exp_args_list)
        n_error = [
            getattr(exp_args, "status", "incomplete") == "error" for exp_args in self.exp_args_list
        ].count(True)
        return n_incomplete, n_error

    def load_exp_args_list(self):
        logger.info(f"Loading experiments from {self.dir}")
        self.exp_args_list = list(inspect_results.yield_all_exp_results(savedir_base=self.dir))

    def set_reproducibility_info(self, strict_reproducibility=False, comment=None):
        """Gather relevant information that may affect the reproducibility of the experiment

        e.g.: versions of BrowserGym, benchmark, AgentLab...

        Args:
            strict_reproducibility: bool
                If True, all modifications have to be committed before running the experiments.
                Also, if relaunching a study, it will not be possible if the code has changed.
            comment: str
                Extra comment to add to the reproducibility information.
        """
        agent_names = [a.agent_name for a in self.agent_args]
        info = repro.get_reproducibility_info(
            agent_names,
            self.benchmark,
            self.uuid,
            ignore_changes=not strict_reproducibility,
            comment=comment,
            allow_bypass_benchmark_version=not strict_reproducibility,
        )
        if self.reproducibility_info is not None:
            repro.assert_compatible(
                self.reproducibility_info, info, raise_if_incompatible=strict_reproducibility
            )
        self.reproducibility_info = info

    def run(
        self,
        n_jobs=1,
        parallel_backend="ray",
        strict_reproducibility=False,
        n_relaunch=1,
        relaunch_errors=True,
        exp_root=RESULTS_DIR,
    ):
        start_time = time.time()

        self.set_reproducibility_info(
            strict_reproducibility=strict_reproducibility, comment=self.comment
        )
        self.save(exp_root=exp_root)

        n_exp = len(self.exp_args_list)
        last_error_count = None
        result_df = None
        summary_df = None

        for i in range(n_relaunch):
            logger.info(f"Launching study {self.name} - trial {i + 1} / {n_relaunch}")
            self._run(n_jobs, parallel_backend, strict_reproducibility)

            result_df, summary_df, error_report = self.get_results()
            logger.info("\n" + str(summary_df))

            n_incomplete, n_error = self.find_incomplete(include_errors=relaunch_errors)

            if n_error / n_exp > 0.3:
                logger.warning("More than 30% of the experiments errored. Stopping the retries.")
                break

            if last_error_count is not None and n_error >= last_error_count:
                logger.warning(
                    "Last trial did not reduce the number of errors. Stopping the retries."
                )
                break

            if n_incomplete == 0:
                logger.info(f"Study {self.name} finished.")
                break

        logger.info("# Error Report:\n-------------\n\n" + error_report)

        if n_incomplete != 0:
            logger.warning(
                f"Study {self.name} did not finish after {n_relaunch} trials. There are {n_incomplete} incomplete experiments."
            )

        # Save structured run log
        self._save_run_log(result_df, summary_df, start_time)

    def _run(self, n_jobs=1, parallel_backend="joblib", strict_reproducibility=False):
        """Run all experiments in the study in parallel when possible.

        Args:
            n_jobs: int
                Number of parallel jobs.
            parallel_backend: str
                Parallel backend to use. Either "joblib", "ray" or "sequential".
            strict_reproducibility: bool
                If True, all modifications have to be committed before running the experiments.
                Also, if relaunching a study, it will not be possible if the code has changed.

        Raises:
            ValueError: If the exp_args_list is None.
        """

        if self.exp_args_list is None:
            raise ValueError("exp_args_list is None. Please set exp_args_list before running.")

        logger.info("Preparing backends...")

        self.benchmark.prepare_backends()
        logger.info("Backends ready.")

        run_experiments(
            n_jobs,
            self.exp_args_list,
            self.dir,
            parallel_backend=parallel_backend,
            avg_step_timeout=self.avg_step_timeout,
        )

    def _save_run_log(self, result_df, summary_df, start_time: float):
        """Save a structured log.json with comprehensive run information.

        Args:
            result_df: DataFrame with experiment results
            summary_df: DataFrame with summary statistics
            start_time: Unix timestamp when the study started
        """
        from agentlab.experiments.loop import ExpResult, get_exp_result

        end_time = time.time()
        duration_seconds = end_time - start_time

        # Build args section
        benchmark_file = None
        if hasattr(self.benchmark, "benchmark_file"):
            benchmark_file = str(self.benchmark.benchmark_file)

        # Extract model names from first experiment
        target_model = None
        attacker_model = None
        world_sim_model = None
        adversarial_mode = None
        max_steps = None

        if self.exp_args_list and len(self.exp_args_list) > 0:
            first_exp = self.exp_args_list[0]
            if hasattr(first_exp.agent_args, "chat_model_args"):
                target_model = getattr(first_exp.agent_args.chat_model_args, "model_name", None)
            if hasattr(first_exp, "env_args"):
                env_args = first_exp.env_args
                attacker_model = getattr(env_args, "attacker_model", None)
                world_sim_model = getattr(env_args, "world_sim_model", None)
                adversarial_mode = getattr(env_args, "adversarial_mode", None)
                max_steps = getattr(env_args, "max_steps", None)

        log_data = {
            "args": {
                "study_name": self.name,
                "benchmark_name": self.benchmark.name if hasattr(self.benchmark, "name") else None,
                "benchmark_file": benchmark_file,
                "target_model": target_model,
                "attacker_model": attacker_model,
                "world_sim_model": world_sim_model,
                "adversarial_mode": adversarial_mode,
                "max_steps": max_steps,
            },
            "time": datetime.now().isoformat(),
            "duration_seconds": round(duration_seconds, 2),
            "summary": {
                "n_experiments": len(self.exp_args_list) if self.exp_args_list else 0,
                "n_completed": 0,
                "n_errors": 0,
                "avg_reward": 0.0,
                "total_cost": 0.0,
            },
            "experiments": [],
            "token_usage": {
                "target": {"input": 0, "output": 0, "total": 0},
                "attacker": {"input": 0, "output": 0, "total": 0},
                "world_sim": {"input": 0, "output": 0, "total": 0},
                "totals": {"input": 0, "output": 0, "total": 0},
            },
        }

        # Extract summary stats from summary_df if available
        if summary_df is not None and len(summary_df) > 0:
            try:
                log_data["summary"]["avg_reward"] = (
                    float(summary_df["avg_reward"].iloc[0]) if "avg_reward" in summary_df else 0.0
                )
                if "n_completed" in summary_df:
                    n_completed_str = str(summary_df["n_completed"].iloc[0])
                    if "/" in n_completed_str:
                        log_data["summary"]["n_completed"] = int(n_completed_str.split("/")[0])
                if "n_err" in summary_df:
                    log_data["summary"]["n_errors"] = int(summary_df["n_err"].iloc[0])
                if "cum_cost" in summary_df:
                    log_data["summary"]["total_cost"] = float(summary_df["cum_cost"].iloc[0])
            except Exception as e:
                logger.warning(f"Error extracting summary stats: {e}")

        # Process each experiment
        total_input_tokens = 0
        total_output_tokens = 0
        attacker_total_tokens = 0
        judge_successes = 0
        judge_scores = []

        try:
            for exp_result in inspect_results.yield_all_exp_results(self.dir, progress_fn=None):
                try:
                    exp_data = self._extract_experiment_data(exp_result)
                    log_data["experiments"].append(exp_data)

                    # Aggregate target token usage
                    if "token_usage" in exp_data:
                        total_input_tokens += exp_data["token_usage"].get("input", 0)
                        total_output_tokens += exp_data["token_usage"].get("output", 0)

                    # Aggregate attacker token usage
                    if "attacker" in exp_data and exp_data["attacker"]:
                        attacker_usage = exp_data["attacker"].get("token_usage", {})
                        attacker_total_tokens += attacker_usage.get("total", 0)

                    # Aggregate judge results
                    if exp_data.get("judge_result"):
                        judge_result = exp_data["judge_result"]
                        if judge_result.get("overall_success") is True:
                            judge_successes += 1
                        if judge_result.get("overall_score") is not None:
                            judge_scores.append(judge_result["overall_score"])
                except Exception as e:
                    logger.warning(f"Error processing experiment {exp_result.exp_dir}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"Error iterating experiments: {e}")

        # Calculate attack success rate and average judge score
        n_experiments = len(log_data["experiments"])
        if n_experiments > 0:
            log_data["summary"]["attack_success_rate"] = judge_successes / n_experiments
        else:
            log_data["summary"]["attack_success_rate"] = 0.0

        if judge_scores:
            log_data["summary"]["avg_judge_score"] = sum(judge_scores) / len(judge_scores)
        else:
            log_data["summary"]["avg_judge_score"] = None

        # Update total token usage
        log_data["token_usage"]["target"]["input"] = total_input_tokens
        log_data["token_usage"]["target"]["output"] = total_output_tokens
        log_data["token_usage"]["target"]["total"] = total_input_tokens + total_output_tokens
        log_data["token_usage"]["attacker"]["total"] = attacker_total_tokens
        # Include attacker tokens in totals
        grand_total = total_input_tokens + total_output_tokens + attacker_total_tokens
        log_data["token_usage"]["totals"]["input"] = total_input_tokens
        log_data["token_usage"]["totals"]["output"] = total_output_tokens
        log_data["token_usage"]["totals"]["total"] = grand_total

        # Save log.json
        log_path = self.dir / "log.json"
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=2, default=str)
            logger.info(f"Saved run log to {log_path}")
        except Exception as e:
            logger.error(f"Error saving log.json: {e}")

    def _extract_experiment_data(self, exp_result) -> dict:
        """Extract structured data from an experiment result.

        Args:
            exp_result: ExpResult object

        Returns:
            Dictionary with experiment data for log.json
        """
        exp_args = exp_result.exp_args
        env_args = exp_args.env_args

        # Extract behavior/task info
        behavior_id = (
            getattr(env_args, "task_name", "").split(".")[-1]
            if hasattr(env_args, "task_name")
            else None
        )

        # Extract goal/doc from env_args
        goal = (
            getattr(env_args, "doc", None)
            or getattr(env_args, "goal", None)
            or getattr(env_args, "objective", "")
        )

        # Extract pages info
        pages = []
        if hasattr(env_args, "pages") and env_args.pages:
            for page in env_args.pages:
                if isinstance(page, dict):
                    pages.append(
                        {
                            "domain": page.get("domain", ""),
                            "real_site": page.get("real_site", ""),
                            "html_path": page.get("existing_path", ""),
                        }
                    )

        # Extract target info
        target_model = None
        if hasattr(exp_args, "agent_args") and hasattr(exp_args.agent_args, "chat_model_args"):
            target_model = getattr(exp_args.agent_args.chat_model_args, "model_name", None)

        target_system_prompt = getattr(env_args, "target_system_prompt", None)

        # Build conversation from steps
        conversation = []

        # Add each step to conversation
        try:
            steps_info = exp_result.steps_info
            seen_user_turns = set()  # Track which user turns we've added

            for step_info in steps_info:
                if not step_info.obs or not isinstance(step_info.obs, dict):
                    continue

                # Extract user messages from chat_messages
                # or from obs["goal"] (for single-turn where attacker message is injected as goal)
                chat_messages = step_info.obs.get("chat_messages", [])
                for msg in chat_messages:
                    if msg.get("role") == "user":
                        msg_content = msg.get("message", "")
                        # Use timestamp or message hash to dedupe
                        msg_key = (msg.get("timestamp"), msg_content[:100] if msg_content else "")
                        if msg_key not in seen_user_turns and msg_content:
                            conversation.append(
                                {
                                    "role": "user",
                                    "content": msg_content,
                                }
                            )
                            seen_user_turns.add(msg_key)

                # Fallback: if no chat_messages but we have a goal (single-turn mode)
                if not chat_messages and not seen_user_turns:
                    first_message = step_info.obs.get("goal", "")
                    if first_message:
                        conversation.append({"role": "user", "content": first_message})
                        seen_user_turns.add(("fallback", first_message[:100]))

                # Add target agent's response for this step
                step_entry = {
                    "role": "target",
                    "step": step_info.step,
                    "action": step_info.action,
                }

                # Extract think/reasoning if available
                if step_info.agent_info:
                    if "think" in step_info.agent_info:
                        step_entry["think"] = step_info.agent_info["think"]

                conversation.append(step_entry)
        except Exception as e:
            logger.warning(f"Error extracting steps: {e}")

        # Extract result info from summary
        result_data = {
            "n_steps": 0,
            "terminated": False,
            "truncated": False,
            "reward": 0,
            "cost": 0.0,
        }

        try:
            summary = exp_result.summary_info
            result_data["n_steps"] = summary.get("n_steps", 0)
            result_data["terminated"] = summary.get("terminated", False)
            result_data["truncated"] = summary.get("truncated", False)
            result_data["reward"] = summary.get("cum_reward", 0)
            result_data["cost"] = summary.get("stats.cum_cost", 0.0)
        except Exception:
            pass

        # Extract criteria scores if available
        criteria_scores = {}
        criteria_id_list = []  # Ordered list for mapping criterion_N -> actual ID
        if hasattr(env_args, "criteria") and env_args.criteria:
            for criterion in env_args.criteria:
                if isinstance(criterion, dict):
                    crit_id = criterion.get("id", "")
                    criteria_id_list.append(crit_id)
                    criteria_scores[crit_id] = {
                        "name": criterion.get("name", ""),
                        "threshold": criterion.get("threshold", 0),
                        "score": None,  # Will be filled from judge results
                    }

        # Load judge results if available
        judge_result = None
        try:
            judge_result_path = exp_result.exp_dir / "judge_result.json"
            if judge_result_path.exists():
                with open(judge_result_path, "r", encoding="utf-8") as f:
                    judge_result = json.load(f)
                logger.debug(f"Loaded judge result from {judge_result_path}")

                # Update criteria scores from judge results if present
                if judge_result and "individual_results" in judge_result:
                    for ind_result in judge_result.get("individual_results", []):
                        if "details" in ind_result and "criteria_scores" in ind_result["details"]:
                            judge_criteria = ind_result["details"]["criteria_scores"]
                            for judge_key, crit_data in judge_criteria.items():
                                # Try direct ID match first
                                if judge_key in criteria_scores:
                                    criteria_scores[judge_key]["score"] = crit_data.get("score")
                                # Also try mapping criterion_N to Nth criterion ID
                                elif judge_key.startswith("criterion_"):
                                    try:
                                        idx = int(judge_key.split("_")[1]) - 1  # 1-indexed
                                        if 0 <= idx < len(criteria_id_list):
                                            actual_id = criteria_id_list[idx]
                                            criteria_scores[actual_id]["score"] = crit_data.get(
                                                "score"
                                            )
                                    except (ValueError, IndexError):
                                        pass
        except Exception as e:
            logger.debug(f"Could not load judge result: {e}")

        # Token usage from summary
        token_usage = {
            "input": 0,
            "output": 0,
            "total": 0,
        }
        attacker_token_usage = {
            "input": 0,
            "output": 0,
            "total": 0,
        }
        try:
            summary = exp_result.summary_info
            token_usage["input"] = summary.get("stats.cum_input_tokens", 0)
            token_usage["output"] = summary.get("stats.cum_output_tokens", 0)
            token_usage["total"] = token_usage["input"] + token_usage["output"]

            # Extract attacker token usage if available
            attacker_usage = summary.get("attacker_token_usage", {})
            if attacker_usage:
                attacker_token_usage["input"] = attacker_usage.get("input", 0)
                attacker_token_usage["output"] = attacker_usage.get("output", 0)
                attacker_token_usage["total"] = attacker_usage.get("total", 0)
        except Exception:
            pass

        return {
            "behavior_id": behavior_id,
            "behavior_name": getattr(env_args, "target", None) or behavior_id,
            "goal": goal,
            "pages": pages,
            "target": {
                "model_name": target_model,
                "system_prompt": (
                    target_system_prompt[:500]
                    if target_system_prompt and len(target_system_prompt) > 500
                    else target_system_prompt
                ),
            },
            "attacker": {
                "model_name": getattr(env_args, "attacker_model", None),
                "token_usage": attacker_token_usage,
            },
            "world_sim": {
                "model_name": getattr(env_args, "world_sim_model", None),
            },
            "conversation": conversation,
            "result": result_data,
            "judge_result": judge_result,
            "criteria_scores": criteria_scores if criteria_scores else None,
            "token_usage": token_usage,
            "exp_dir": str(exp_result.exp_dir),
        }

    def append_to_journal(self, strict_reproducibility=True):
        """Append the study to the journal.

        Args:
            strict_reproducibility: bool
                If True, incomplete experiments will raise an error.
        """
        _, summary_df, _ = self.get_results()
        repro.append_to_journal(
            self.reproducibility_info,
            summary_df,
            strict_reproducibility=strict_reproducibility,
        )

    @property
    def name(self):
        agent_names = [a.agent_name for a in self.agent_args]
        return _make_study_name(agent_names, [self.benchmark.name], self.suffix)

    def override_max_steps(self, max_steps):
        for exp_args in self.exp_args_list:
            exp_args.env_args.max_steps = max_steps

    @staticmethod
    def load(dir: Path) -> "Study":
        dir = Path(dir)
        study_path = dir / "study.pkl.gz"
        if not study_path.exists() and dir.is_dir():
            # For backward compatibility
            first_result = next(
                inspect_results.yield_all_exp_results(savedir_base=dir, progress_fn=None)
            )
            benchmark_name = first_result.exp_args.env_args.task_name.split(".")[0]
            agent_args = first_result.exp_args.agent_args
            study = Study(agent_args=agent_args, benchmark=benchmark_name, dir=dir)
        else:
            with gzip.open(dir / "study.pkl.gz", "rb") as f:
                study = pickle.load(f)  # type: Study
            study.dir = dir

            # # just a check
            # for i, exp_args in enumerate(study.exp_args_list):
            #     if exp_args.order != i:
            #         logging.warning(f"The order of the experiments is not correct. {exp_args.order} != {i}")

        return study

    @staticmethod
    def load_most_recent(root_dir: Path = None, contains=None) -> "Study":
        return Study.load(get_most_recent_study(root_dir, contains=contains))

    def agents_on_benchmark(
        self,
        agents: list[AgentArgs] | AgentArgs,
        benchmark: Benchmark,
        demo_mode=False,
        logging_level: int = logging.INFO,
        logging_level_stdout: int = logging.INFO,
        ignore_dependencies=False,
    ):
        """Run one or multiple agents on a benchmark.

        Args:
            agents: list[AgentArgs] | AgentArgs
                The agent configuration(s) to run.
            benchmark: Benchmark
                The benchmark to run the agents on.
            demo_mode: bool
                If True, the experiments will be run in demo mode.
            logging_level: int
                The logging level for individual jobs.
            logging_level_stdout: int
                The logging level for the stdout.
            ignore_dependencies: bool
                If True, the dependencies will be ignored and all experiments can be run in parallel.

        Returns:
            list[ExpArgs]: The list of experiments to run.

        Raises:
            ValueError: If multiple agents are run on a benchmark that requires manual reset.
        """

        if not isinstance(agents, (list, tuple)):
            agents = [agents]

        if benchmark.name.startswith("visualwebarena") or benchmark.name.startswith("webarena"):
            if len(agents) > 1:
                raise ValueError(
                    f"Only one agent can be run on {benchmark.name} since the instance requires manual reset after each evaluation."
                )

        for agent in agents:
            agent.set_benchmark(
                benchmark, demo_mode
            )  # the agent can adapt (lightly?) to the benchmark

        env_args_list = benchmark.env_args_list
        if demo_mode:
            set_demo_mode(env_args_list)

        exp_args_list = []

        for agent in agents:
            for env_args in env_args_list:

                exp_args = ExpArgs(
                    agent_args=agent,
                    env_args=env_args,
                    attacker_agent_args=env_args.attacker_agent,
                    max_conversation_turns=env_args.max_conversation_turns,
                    logging_level=logging_level,
                    logging_level_stdout=logging_level_stdout,
                )
                exp_args_list.append(exp_args)

        for i, exp_args in enumerate(exp_args_list):
            exp_args.order = i

        # not required with ray, but keeping around if we would need it for visualwebareana on joblib
        # _flag_sequential_exp(exp_args_list, benchmark)

        if not ignore_dependencies:
            # populate the depends_on field based on the task dependencies in the benchmark
            exp_args_list = add_dependencies(exp_args_list, benchmark.dependency_graph_over_tasks())
        else:
            logger.warning(
                f"Ignoring dependencies for benchmark {benchmark.name}. This could lead to different results."
            )

        return exp_args_list


def _make_study_name(agent_names, benchmark_names, suffix=None):
    """Make a study name from the agent and benchmark names."""
    # extract unique agent and benchmark names
    agent_names = list(set(agent_names))
    benchmark_names = list(set(benchmark_names))

    if len(agent_names) == 1:
        agent_name = agent_names[0]
    else:
        agent_name = f"{len(agent_names)}_agents"

    if len(benchmark_names) == 1:
        benchmark_name = benchmark_names[0]
    else:
        benchmark_name = f"{len(benchmark_names)}_benchmarks"

    study_name = f"{agent_name}_on_{benchmark_name}_{suffix if suffix else ''}"

    return slugify(study_name, max_length=200, allow_unicode=True)


@dataclass
class SequentialStudies(AbstractStudy):
    """
    Sequential execution of multiple studies.

    This is required for e.g. WebArena, where a server reset is required between evaluations of each agent.
    """

    studies: list[Study]

    @property
    def name(self):
        """The name of the study."""
        agent_names = [a.agent_name for study in self.studies for a in study.agent_args]
        benchmark_names = [study.benchmark.name for study in self.studies]
        return _make_study_name(agent_names, benchmark_names, self.suffix)

    def find_incomplete(self, include_errors=True):
        for study in self.studies:
            study.find_incomplete(include_errors=include_errors)

    def run(
        self,
        n_jobs=1,
        parallel_backend="ray",
        strict_reproducibility=False,
        n_relaunch=1,
        exp_root=RESULTS_DIR,
    ):
        # This sequence of of making directories is important to make sure objects are materialized
        # properly before saving. Otherwise relaunch may not work properly.
        self.make_dir()
        for study in self.studies:
            study.make_dir(exp_root=self.dir)

        self.save(exp_root=exp_root)
        self._run(n_jobs, parallel_backend, strict_reproducibility, n_relaunch)
        _, summary_df, _ = self.get_results()
        logger.info("\n" + str(summary_df))
        logger.info(f"SequentialStudies {self.name} finished.")

    def _run(self, n_jobs=1, parallel_backend="ray", strict_reproducibility=False, n_relaunch=3):
        for study in self.studies:
            study.run(n_jobs, parallel_backend, strict_reproducibility, n_relaunch)

    def override_max_steps(self, max_steps):
        for study in self.studies:
            study.override_max_steps(max_steps)

    def append_to_journal(self, strict_reproducibility=True):
        for study in self.studies:
            study.append_to_journal(strict_reproducibility=strict_reproducibility)


def _init_worker(server_queue: Queue):
    """Run once at the initialization of the worker in the multiprocessing.Pool.

    This is typically used to initialize different environment variables of the WebArena server for
    multiple instances in parallel.

    Args:
        server_queue: Queue
            A queue of object implementing BaseServer to initialize (or anything with a init
            method).
    """
    print("initializing server instance with on process", os.getpid())
    print(f"using queue {server_queue}")
    server_instance = server_queue.get()  # type: "WebArenaInstanceVars"
    logger.warning(f"Initializing server instance {server_instance} from process {os.getpid()}")
    server_instance.init()


def _run_study(study: Study, n_jobs, parallel_backend, strict_reproducibility, n_relaunch):
    """Wrapper to run a study remotely."""
    study.run(n_jobs, parallel_backend, strict_reproducibility, n_relaunch)


@dataclass
class ParallelStudies(SequentialStudies):
    parallel_servers: list[BaseServer] | int = None

    def _run(
        self,
        n_jobs=1,
        parallel_backend="ray",
        strict_reproducibility=False,
        n_relaunch=3,
    ):
        parallel_servers = self.parallel_servers
        if isinstance(parallel_servers, int):
            parallel_servers = [BaseServer() for _ in range(parallel_servers)]

        server_queue = Manager().Queue()
        for server in parallel_servers:
            server_queue.put(server)

        with ProcessPoolExecutor(
            max_workers=len(parallel_servers), initializer=_init_worker, initargs=(server_queue,)
        ) as executor:
            # Create list of arguments for each study
            study_args = [
                (study, n_jobs, parallel_backend, strict_reproducibility, n_relaunch)
                for study in self.studies
            ]

            # Submit all tasks and wait for completion
            futures = [executor.submit(_run_study, *args) for args in study_args]

            # Wait for all futures to complete and raise any exceptions
            for future in futures:
                future.result()


@dataclass
class ParallelStudies_alt(SequentialStudies):
    parallel_servers: list[BaseServer] | int = None

    def _run(
        self,
        n_jobs=1,
        parallel_backend="ray",
        strict_reproducibility=False,
        n_relaunch=3,
    ):
        parallel_servers = self.parallel_servers
        if isinstance(parallel_servers, int):
            parallel_servers = [BaseServer() for _ in range(parallel_servers)]

        server_queue = Manager().Queue()
        for server in parallel_servers:
            server_queue.put(server)

        with Pool(len(parallel_servers), initializer=_init_worker, initargs=(server_queue,)) as p:
            p.starmap(
                _run_study,
                [
                    (study, n_jobs, parallel_backend, strict_reproducibility, n_relaunch)
                    for study in self.studies
                ],
            )


def get_most_recent_study(
    root_dir: Path = None, date_format: str = "%Y-%m-%d_%H-%M-%S", contains=None
):
    """Return the most recent directory based on the date in the folder name.

    Args:
        root_dir: The directory to search in
        date_format: The format of the date in the folder name
        contains: If not None, only consider folders that contains this string

    Returns:
        Path: The most recent folder satisfying the conditions
    """

    if root_dir is None:
        root_dir = RESULTS_DIR

    most_recent_folder = None
    most_recent_time = datetime.min

    for item in root_dir.iterdir():
        if item.is_dir() and not item.name.startswith("_"):
            if contains is not None and contains not in item.name:
                continue
            try:
                folder_date = datetime.strptime("_".join(item.name.split("_")[:2]), date_format)
                if folder_date > most_recent_time:
                    most_recent_time = folder_date
                    most_recent_folder = item
            except (ValueError, IndexError):
                continue

    return most_recent_folder


def set_demo_mode(env_args_list: list[EnvArgs]):
    """Set the demo mode for the experiments. This can be useful for generating videos for demos."""
    for env_args in env_args_list:
        env_args.viewport = {"width": 1280, "height": 720}
        env_args.record_video = True
        env_args.wait_for_user_message = False
        env_args.slow_mo = 1000


def _convert_env_args(env_args_list) -> list[EnvArgs]:
    """Return a list where every element is the *new* EnvArgs.

    For backward compatibility, we need to convert the old EnvArgs to the new one.

    Args:
        env_args_list (list): list of EnvArgs objects to convert

    Returns:
        list: list of converted EnvArgs objects

    Raises:
        TypeError: If an element in env_args_list is not of expected type.
    """
    from bgym import EnvArgs as BGymEnvArgs

    new_list = []
    for ea in env_args_list:
        # already new → keep as‑is
        if isinstance(ea, (EnvArgs, AbstractEnvArgs)):
            new_list.append(ea)
        # old → convert
        elif isinstance(ea, BGymEnvArgs):
            new_list.append(EnvArgs(**asdict(ea)))
        else:
            raise TypeError(f"Unexpected type: {type(ea)}")
    return new_list


# def _flag_sequential_exp(exp_args_list: list[ExpArgs], benchmark: Benchmark):
#     if benchmark.name.startswith("visualwebarena"):
#         sequential_subset = benchmark.subset_from_glob("requires_reset", "True")
#         sequential_subset = set(
#             [env_args.task_name for env_args in sequential_subset.env_args_list]
#         )
#         for exp_args in exp_args_list:
#             if exp_args.env_args.task_name in sequential_subset:
#                 exp_args.sequential = True


# def ablation_study(start_agent: AgentArgs, changes, benchmark: str, demo_mode=False):
#     """Ablation study of an agent.

#     Changes is a list of tuples (path_to_attribute, value) to change in the agent
#     configuration.

#     Args:
#         start_agent: AgentArgs
#             The agent configuration to start from.

#         changes: list[tuple]
#             The changes to apply to the agent configuration.

#         benchmark: str
#             The benchmark to use.

#         demo_mode: bool
#             If True, the experiments will be run in demo mode.

#     Returns:
#         Study
#     """
#     agents = args.make_ablation_study(start_agent, changes)
#     study = run_agents_on_benchmark(agents, benchmark, demo_mode=demo_mode)
#     study.suffix = "ablation_study"
#     return study


# def random_search(
#     random_agent: AgentArgs = RANDOM_SEARCH_AGENT,
#     n_samples=10,
#     benchmark: str = "miniwob",
#     demo_mode=False,
# ):
#     """
#     Random search of AgentArgs (NOTE: not fully tested since refactoring)

#     The random search mechanism will recursively search through dataclasses and
#     dict to find attributes of type args.Choice. It will sample iid and replace
#     with the corresponding value.

#     *WARINING* The standard errror of the experiment will usually be relatively high and
#     the search space is usually big so the false discovery rate will likely be
#     high. Make sure to analyze the results with caution and don't actually draw
#     final conclusions from these experiments.

#     Args:
#         agent: AgentArgs
#             The agent configuration, with some sub-arguments defined as args.Choice.

#         n_samples: int
#             The number of samples to take.

#         benchmark: str
#             The benchmark to use.

#         demo_mode: bool
#             If True, the experiments will be run in demo mode.

#     Returns:
#         Study
#     """
#     agents = args.sample_and_expand_cross_product(random_agent, n_samples)
#     study = run_agents_on_benchmark(agents, benchmark, demo_mode=demo_mode)
#     study.suffix = "random_search"
#     return study
