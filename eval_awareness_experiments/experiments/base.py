"""Base class for eval awareness experiments."""

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from eval_awareness_experiments.llm import LLM
from eval_awareness_experiments.types import WebsiteExperimentResult, WebsiteSample

logger = logging.getLogger(__name__)


class BaseExperiment(ABC):
    """Base class for website eval awareness experiments."""

    name: str = "base"

    def __init__(self, model: LLM, output_dir: Path):
        self.model = model
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_website_content(self, sample: WebsiteSample, format_type: str) -> str | None:
        """Load website content in the specified format.

        Args:
            sample: The website sample.
            format_type: One of "html", "axtree", "screenshot".

        Returns:
            Content string, or None if not available.
        """
        if format_type == "html" and sample.html_path:
            path = Path(sample.html_path)
            if path.exists():
                content = path.read_text(encoding="utf-8", errors="replace")
                # Truncate very large HTML to avoid token limits
                if len(content) > 100_000:
                    content = content[:100_000] + "\n\n[... HTML truncated at 100k chars ...]"
                return content
        elif format_type == "axtree" and sample.axtree_path:
            path = Path(sample.axtree_path)
            if path.exists():
                return path.read_text(encoding="utf-8", errors="replace")
        elif format_type == "screenshot" and sample.screenshot_path:
            path = Path(sample.screenshot_path)
            if path.exists():
                return str(path)  # Return path for image handling

        return None

    @abstractmethod
    async def run_sample(
        self, sample: WebsiteSample, format_type: str
    ) -> list[WebsiteExperimentResult]:
        """Run experiment on a single sample.

        Args:
            sample: The website sample.
            format_type: One of "html", "axtree", "screenshot".

        Returns:
            List of experiment results.
        """
        pass

    async def run(
        self,
        samples: list[WebsiteSample],
        format_types: list[str],
    ) -> list[WebsiteExperimentResult]:
        """Run experiment on all samples and formats concurrently.

        Concurrency is bounded by the LLM client's semaphore.
        Saves results to JSONL when all tasks complete.
        """
        import asyncio

        output_file = self.output_dir / f"{self.name}_results.jsonl"

        async def _run_one(sample: WebsiteSample, fmt: str) -> list[WebsiteExperimentResult]:
            try:
                return await self.run_sample(sample, fmt)
            except Exception as e:
                logger.error(f"Error on {sample.id}/{fmt}: {e}")
                return [WebsiteExperimentResult(
                    sample_id=sample.id,
                    experiment_type=self.name,
                    model=self.model.model,
                    format_used=fmt,
                    metadata={"error": str(e)},
                )]

        tasks = [
            _run_one(sample, fmt)
            for sample in samples
            for fmt in format_types
        ]
        nested_results = await asyncio.gather(*tasks)
        all_results = [r for batch in nested_results for r in batch]

        # Save all results
        with open(output_file, "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps({
                    "sample_id": r.sample_id,
                    "experiment_type": r.experiment_type,
                    "model": r.model,
                    "format_used": r.format_used,
                    "score": r.score,
                    "classification": r.classification,
                    "raw_response": r.raw_response,
                    "metadata": r.metadata,
                }) + "\n")

        print(f"  {self.name}: {len(all_results)} results saved to {output_file}")
        return all_results
