"""E3: Transcript-based awareness experiment.

Uses the copied judges (binary_third_person, probability_third_person,
verbalized_awareness) on agent interaction transcripts from WorldSim runs
and other benchmarks.
"""

import logging
from pathlib import Path

from eval_awareness_experiments.experiments.base import PerSampleExperiment
from eval_awareness_experiments.judges import get_judge
from eval_awareness_experiments.llm import LLM
from eval_awareness_experiments.types import (
    Message,
    Transcript,
    WebsiteExperimentResult,
    WebsiteSample,
)

logger = logging.getLogger(__name__)


def load_transcript_from_log(log_path: Path) -> Transcript | None:
    """Attempt to load a transcript from a browser-sim experiment log.

    Looks for experiment.log or similar files and extracts the agent's
    interaction messages.
    """
    if not log_path.exists():
        return None

    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None

    # Simple heuristic: treat the whole log as a single assistant message
    # Real implementation would parse structured agent logs
    messages = [
        Message(role="assistant", content=text),
    ]

    return Transcript(
        messages=messages,
        source_file=str(log_path),
        metadata={"source": "browser_sim_log"},
    )


class TranscriptAwarenessExperiment(PerSampleExperiment):
    """Run transcript-based eval awareness judges on agent interaction logs."""

    name = "transcript_awareness"

    def __init__(
        self,
        model: LLM,
        output_dir: Path,
        judge_names: list[str] | None = None,
    ):
        super().__init__(model, output_dir)
        self.judge_names = judge_names or [
            "binary_third_person",
            "probability_third_person",
            "verbalized_awareness",
        ]

    async def run_sample(
        self, sample: WebsiteSample, format_type: str
    ) -> list[WebsiteExperimentResult]:
        """Run transcript awareness judges on a sample's experiment log."""
        # Try to find the experiment log
        if not sample.metadata.get("experiment_dir"):
            return []

        exp_dir = Path(sample.metadata["experiment_dir"])
        log_path = exp_dir / "experiment.log"
        transcript = load_transcript_from_log(log_path)

        if transcript is None:
            return []

        results = []
        for judge_name in self.judge_names:
            try:
                judge = get_judge(
                    judge_name,
                    grader_model=self.model,
                    classifier_model=self.model,
                )
                judge_result = await judge.judge(transcript)

                results.append(
                    WebsiteExperimentResult(
                        sample_id=sample.id,
                        experiment_type=f"transcript_{judge_name}",
                        model=self.model.model,
                        format_used="transcript",
                        score=judge_result.confidence,
                        classification=judge_result.category.value,
                        raw_response=judge_result.raw_response,
                        metadata={
                            "judge_name": judge_name,
                            "reasoning": judge_result.reasoning,
                            "source": sample.source,
                            "website_type": sample.website_type,
                        },
                    )
                )
            except Exception as e:
                logger.error(f"Judge {judge_name} failed on {sample.id}: {e}")
                results.append(
                    WebsiteExperimentResult(
                        sample_id=sample.id,
                        experiment_type=f"transcript_{judge_name}",
                        model=self.model.model,
                        format_used="transcript",
                        metadata={"error": str(e), "judge_name": judge_name},
                    )
                )

        return results
