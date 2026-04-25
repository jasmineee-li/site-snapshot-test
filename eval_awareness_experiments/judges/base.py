"""Base class for eval awareness judges.

Adapted from eval-awareness-testbed: uses LLM instead of inspect_ai.model.Model.
"""

import asyncio
from abc import ABC, abstractmethod

from eval_awareness_experiments.llm import LLM
from eval_awareness_experiments.types import JudgeResult, Transcript

DEFAULT_MAX_CONCURRENCY = 50

# Judges are doing classification, not creative writing — sampling diversity
# buys nothing and inflates variance across runs. Pin to 0 by default so
# repeated judge calls over the same transcript give the same label.
JUDGE_DEFAULT_TEMPERATURE = 0.0


class BaseJudge(ABC):
    """Base class for eval awareness judges.

    Two model roles:
    - grader_model (actor): Used for continuation/generation
    - classifier_model: Used for analyzing/classifying responses

    String-named models are wrapped in `LLM(..., temperature=0)` for
    reproducibility. Pre-built `LLM` instances are used as-is — callers
    that pass one in are responsible for its sampling settings.
    """

    name: str = "base"
    description: str = "Base judge class"
    judge_type: str = "meta_eval"  # "meta_eval" or "live_eval"

    def __init__(
        self,
        grader_model: LLM | str,
        classifier_model: LLM | str | None = None,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
        judge_temperature: float = JUDGE_DEFAULT_TEMPERATURE,
        **kwargs,
    ):
        if grader_model is None:
            raise ValueError("grader_model must be specified")
        if isinstance(grader_model, str):
            grader_model = LLM(grader_model, temperature=judge_temperature)
        self.grader_model = grader_model

        if classifier_model is None:
            self.classifier_model = grader_model
        elif isinstance(classifier_model, str):
            self.classifier_model = LLM(classifier_model, temperature=judge_temperature)
        else:
            self.classifier_model = classifier_model

        self.max_concurrency = max_concurrency
        self.config = kwargs

    @abstractmethod
    async def judge(self, transcript: Transcript) -> JudgeResult:
        """Judge a transcript for eval awareness."""
        pass

    async def judge_batch(self, transcripts: list[Transcript]) -> list[JudgeResult]:
        """Judge multiple transcripts concurrently."""
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def _judge_one(transcript: Transcript) -> JudgeResult:
            async with semaphore:
                return await self.judge(transcript)

        return await asyncio.gather(*[_judge_one(t) for t in transcripts])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
