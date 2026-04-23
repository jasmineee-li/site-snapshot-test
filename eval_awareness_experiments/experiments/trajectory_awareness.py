"""Trajectory-based eval awareness experiment.

Loads agent trajectories from safety benchmarks (DoomArena, WASP, OS-harm)
and existing benchmarks (WebArena-Infinity, TAC), converts them to a
standardized format, then runs configurable judges to detect eval awareness
and scenario recognition in the agent's reasoning traces.

Default judges:
- test_awareness_steering: 3-dimension (model_awareness, hypothetical_recognition, task_performance)
- verbalized_awareness: 4-category spontaneous awareness classification

Results include both eval-awareness scores and attack success rate (ASR) from
the benchmark's own evaluator, enabling ASR × awareness correlation analysis.
"""

import logging
from pathlib import Path

from eval_awareness_experiments.experiments.base import BaseExperiment
from eval_awareness_experiments.judges import get_judge
from eval_awareness_experiments.llm import LLM
from eval_awareness_experiments.trajectory_converter import (
    parse_trajectory,
    trajectory_to_transcript,
)
from eval_awareness_experiments.types import (
    WebsiteExperimentResult,
    WebsiteSample,
)

logger = logging.getLogger(__name__)

DEFAULT_JUDGES = [
    "test_awareness_steering",
    "verbalized_awareness",
    "purpose_continue_5q",
]


class TrajectoryAwarenessExperiment(BaseExperiment):
    """Run eval awareness judges on agent trajectories from safety benchmarks."""

    name = "trajectory_awareness"

    def __init__(
        self,
        model: LLM,
        output_dir: Path,
        judge_names: list[str] | None = None,
    ):
        super().__init__(model, output_dir)
        self.judge_names = judge_names or DEFAULT_JUDGES

    async def run_sample(
        self, sample: WebsiteSample, format_type: str,
    ) -> list[WebsiteExperimentResult]:
        """Run trajectory awareness judges on a sample's trajectory.

        The sample's metadata must contain either:
        - "trajectory_dir": path to a trajectory directory, or
        - "trajectory": an already-parsed trajectory dict
        """
        traj_dir = sample.metadata.get("trajectory_dir")
        if not traj_dir:
            return []

        traj_path = Path(traj_dir)
        if not traj_path.exists():
            logger.warning(f"Trajectory dir not found: {traj_dir}")
            return []

        # Parse the trajectory
        source = sample.metadata.get("benchmark", sample.source)
        task_info = sample.metadata.get("task_info")
        traj = parse_trajectory(source, traj_path, task_info=task_info)

        if traj is None:
            logger.warning(f"Failed to parse trajectory: {traj_dir}")
            return []

        # Convert to Transcript for judge consumption
        transcript = trajectory_to_transcript(traj)

        results = []
        for judge_name in self.judge_names:
            try:
                judge = get_judge(
                    judge_name,
                    grader_model=self.model,
                    classifier_model=self.model,
                )
                judge_result = await judge.judge(transcript)

                # Build metadata with both awareness and ASR info
                meta = {
                    "judge_name": judge_name,
                    "reasoning": judge_result.reasoning,
                    "source": sample.source,
                    "benchmark": traj.benchmark,
                    "agent": traj.agent,
                    "task_instruction": traj.task_instruction,
                    "attack_type": traj.attack_type,
                    "attack_success": traj.attack_success,
                    "task_passed": traj.passed,
                    "injection_condition": traj.injection_condition,
                    "n_steps": len(traj.steps),
                }

                # For TAS judge, surface the 3 dimensions directly
                if judge_name == "test_awareness_steering":
                    meta["model_awareness"] = judge_result.metadata.get(
                        "model_awareness", {}
                    )
                    meta["hypothetical_recognition"] = judge_result.metadata.get(
                        "hypothetical_recognition", {}
                    )
                    meta["task_performance"] = judge_result.metadata.get(
                        "task_performance", {}
                    )

                # For 5PQ judge, surface both shares + per-variant labels.
                if judge_name == "purpose_continue_5q":
                    meta["alignment_eval_share"] = judge_result.metadata.get(
                        "alignment_eval_share"
                    )
                    meta["any_eval_share"] = judge_result.metadata.get(
                        "any_eval_share"
                    )
                    meta["variant_categories"] = [
                        {"prompt_id": v["prompt_id"], "category": v["category"]}
                        for v in judge_result.metadata.get("variants", [])
                    ]

                results.append(WebsiteExperimentResult(
                    sample_id=sample.id,
                    experiment_type=f"trajectory_{judge_name}",
                    model=self.model.model,
                    format_used="trajectory",
                    score=judge_result.confidence,
                    classification=judge_result.category.value,
                    raw_response=judge_result.raw_response,
                    metadata=meta,
                ))
            except Exception as e:
                logger.error(f"Judge {judge_name} failed on {sample.id}: {e}")
                results.append(WebsiteExperimentResult(
                    sample_id=sample.id,
                    experiment_type=f"trajectory_{judge_name}",
                    model=self.model.model,
                    format_used="trajectory",
                    metadata={"error": str(e), "judge_name": judge_name},
                ))

        return results
