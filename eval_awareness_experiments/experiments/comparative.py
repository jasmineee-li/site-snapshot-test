"""E2: Comparative scoring experiment.

Presents pairs of websites from different sources and asks which looks
more like an evaluation environment. Counterbalanced (AB and BA orderings).
"""

import asyncio
import json
import logging
from itertools import combinations
from pathlib import Path

from eval_awareness_experiments.experiments.base import BaseExperiment
from eval_awareness_experiments.experiments.prompts import WEBSITE_COMPARATIVE_PROMPT
from eval_awareness_experiments.judges.json_parsing import parse_json_safe
from eval_awareness_experiments.llm import LLM
from eval_awareness_experiments.types import WebsiteExperimentResult, WebsiteSample

logger = logging.getLogger(__name__)


class ComparativeExperiment(BaseExperiment):
    """Pairwise comparison of website environments from different sources."""

    name = "comparative"

    async def run_sample(
        self, sample: WebsiteSample, format_type: str
    ) -> list[WebsiteExperimentResult]:
        # Not used directly — we override run() to handle pairs
        return []

    async def run_pairs(
        self,
        samples: list[WebsiteSample],
        format_type: str,
    ) -> list[WebsiteExperimentResult]:
        """Run pairwise comparisons between samples from different sources.

        Groups samples by website_type, then creates all cross-source pairs.
        Each pair is run in both AB and BA order for counterbalancing.
        """
        # Group by website_type
        by_type: dict[str, list[WebsiteSample]] = {}
        for s in samples:
            by_type.setdefault(s.website_type, []).append(s)

        all_results = []
        output_file = self.output_dir / f"{self.name}_results.jsonl"

        for wtype, type_samples in by_type.items():
            # Group by source within this website type
            by_source: dict[str, list[WebsiteSample]] = {}
            for s in type_samples:
                by_source.setdefault(s.source, []).append(s)

            sources = list(by_source.keys())
            if len(sources) < 2:
                continue

            # Generate cross-source pairs
            for src_a, src_b in combinations(sources, 2):
                for sample_a in by_source[src_a]:
                    for sample_b in by_source[src_b]:
                        # Run AB order
                        result_ab = await self._compare_pair(
                            sample_a, sample_b, format_type, "AB"
                        )
                        all_results.append(result_ab)

                        # Run BA order (counterbalanced)
                        result_ba = await self._compare_pair(
                            sample_b, sample_a, format_type, "BA"
                        )
                        all_results.append(result_ba)

                        # Incremental save
                        with open(output_file, "a", encoding="utf-8") as f:
                            for r in [result_ab, result_ba]:
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

    async def _compare_pair(
        self,
        sample_a: WebsiteSample,
        sample_b: WebsiteSample,
        format_type: str,
        order_label: str,
    ) -> WebsiteExperimentResult:
        """Compare two samples."""
        content_a = self.load_website_content(sample_a, format_type)
        content_b = self.load_website_content(sample_b, format_type)

        if content_a is None or content_b is None:
            return WebsiteExperimentResult(
                sample_id=f"{sample_a.id}_vs_{sample_b.id}",
                experiment_type=self.name,
                model=self.model.model,
                format_used=format_type,
                metadata={"error": "missing_content", "order": order_label},
            )

        format_label = {
            "html": "HTML source code",
            "axtree": "accessibility tree",
            "screenshot": "screenshot",
        }.get(format_type, format_type)

        if format_type == "screenshot":
            # For screenshots we'd need multi-image support; use text prompt as fallback
            prompt = WEBSITE_COMPARATIVE_PROMPT.format(
                format_type=format_label,
                content_a="[Screenshot A - see image]",
                content_b="[Screenshot B - see image]",
            )
            output = await self.model.generate(prompt)
        else:
            prompt = WEBSITE_COMPARATIVE_PROMPT.format(
                format_type=format_label,
                content_a=content_a,
                content_b=content_b,
            )
            output = await self.model.generate(prompt)

        response_text = output.message.text
        parsed = parse_json_safe(response_text)

        choice = None
        confidence = None
        reasoning = ""
        if parsed:
            choice = parsed.get("more_likely_eval")
            confidence = parsed.get("confidence")
            reasoning = parsed.get("reasoning", "")

        # Map choice back to actual source
        chosen_source = None
        if choice == "A":
            chosen_source = sample_a.source
        elif choice == "B":
            chosen_source = sample_b.source

        return WebsiteExperimentResult(
            sample_id=f"{sample_a.id}_vs_{sample_b.id}",
            experiment_type=self.name,
            model=self.model.model,
            format_used=format_type,
            score=confidence,
            classification=chosen_source,
            raw_response=response_text,
            metadata={
                "order": order_label,
                "sample_a_id": sample_a.id,
                "sample_b_id": sample_b.id,
                "sample_a_source": sample_a.source,
                "sample_b_source": sample_b.source,
                "website_type": sample_a.website_type,
                "choice": choice,
                "reasoning": reasoning,
            },
        )
