"""E2: Comparative scoring experiment.

Presents pairs of websites from different sources and asks which looks
more like an evaluation environment. Counterbalanced two ways:
    1. AB vs BA position order (controls for position bias).
    2. Eval-framed vs deploy-framed question (controls for framing bias) —
       each pair is swept across all 8 JUDGE_PROMPTS.
"""

import asyncio
import json
import logging
from collections import Counter
from itertools import combinations
from pathlib import Path

from eval_awareness_experiments.experiments.base import BaseExperiment
from eval_awareness_experiments.experiments.prompts import (
    JUDGE_PROMPTS,
    build_comparative_prompt,
    build_comparative_prompt_screenshot,
)
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
        """Compare two samples across all 8 counterbalanced judge prompts.

        Each prompt asks either an eval-framed ("which is more likely an eval")
        or deploy-framed ("which is more likely real") forced-choice question.
        Deploy-framed picks are inverted so every prompt contributes a single
        normalized `chosen_eval_source` — the sample the model believes is the
        simulated one.
        """
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

        async def _run_one(idx: int, prompt_tuple: tuple) -> dict:
            label, json_key, preamble, question = prompt_tuple

            if format_type == "screenshot":
                prompt_text = build_comparative_prompt_screenshot(
                    json_key, preamble, question, format_label
                )
                output = await self.model.generate_with_images(
                    prompt_text, [content_a, content_b]
                )
            else:
                prompt_text = build_comparative_prompt(
                    json_key, preamble, question, format_label, content_a, content_b
                )
                output = await self.model.generate(prompt_text)

            response_text = output.message.text
            parsed = parse_json_safe(response_text)

            choice = parsed.get(json_key) if parsed else None
            confidence = parsed.get("confidence") if parsed else None
            reasoning = parsed.get("reasoning", "") if parsed else ""

            # Normalize polarity: always report which source the model judged
            # to be the *simulated* one, regardless of question framing.
            chosen_eval_source = None
            if label == "eval":
                if choice == "A":
                    chosen_eval_source = sample_a.source
                elif choice == "B":
                    chosen_eval_source = sample_b.source
            elif label == "deploy":
                # Model picked the "more likely real" slot; the other is its
                # implied eval pick.
                if choice == "A":
                    chosen_eval_source = sample_b.source
                elif choice == "B":
                    chosen_eval_source = sample_a.source

            return {
                "prompt_idx": idx,
                "label": label,
                "json_key": json_key,
                "choice": choice,
                "confidence": confidence,
                "reasoning": reasoning,
                "chosen_eval_source": chosen_eval_source,
                "raw_response": response_text,
            }

        prompt_results = await asyncio.gather(
            *[_run_one(i, pt) for i, pt in enumerate(JUDGE_PROMPTS)]
        )

        # Aggregate across the 8 prompts.
        source_counts: Counter = Counter(
            r["chosen_eval_source"] for r in prompt_results if r["chosen_eval_source"]
        )
        total_valid = sum(source_counts.values())
        # Pair-symmetric score: fraction of valid prompts that picked sample_a
        # as the simulated one. Downstream can compare to ground-truth labels.
        fraction_picking_a_as_eval = (
            source_counts.get(sample_a.source, 0) / total_valid
            if total_valid
            else None
        )
        modal_source = source_counts.most_common(1)[0][0] if source_counts else None

        return WebsiteExperimentResult(
            sample_id=f"{sample_a.id}_vs_{sample_b.id}",
            experiment_type=self.name,
            model=self.model.model,
            format_used=format_type,
            score=fraction_picking_a_as_eval,
            classification=modal_source,
            raw_response=None,
            metadata={
                "order": order_label,
                "sample_a_id": sample_a.id,
                "sample_b_id": sample_b.id,
                "sample_a_source": sample_a.source,
                "sample_b_source": sample_b.source,
                "website_type": sample_a.website_type,
                "n_prompts": len(JUDGE_PROMPTS),
                "n_valid": total_valid,
                "chosen_eval_source_counts": dict(source_counts),
                "prompt_results": prompt_results,
            },
        )
