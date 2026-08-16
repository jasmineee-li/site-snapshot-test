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
import os
import random
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

# When set, _run_one_inner dumps the exact prompt sent and the exact response
# received. Each dump goes to a separate file under $COMPARATIVE_DEBUG_DIR
# (one file per {pair, order, judge prompt}).
_DEBUG = os.environ.get("COMPARATIVE_DEBUG") == "1"
_DEBUG_DIR = os.environ.get("COMPARATIVE_DEBUG_DIR")


def _safe_name(s: str, max_len: int = 50) -> str:
    """Sanitize a sample id for safe use in filenames."""
    s = re.sub(r"[^a-zA-Z0-9._-]", "_", s)
    if len(s) > max_len:
        s = s[:max_len]
    return s


def _debug_dump(
    *,
    model: str,
    sample_a,
    sample_b,
    format_type: str,
    order_label: str,
    idx: int,
    label: str,
    json_key: str,
    prompt_text: str,
    content_a: str | None,
    content_b: str | None,
    image_paths: list[str] | None,
    response_text: str,
) -> None:
    """Dump the exact prompt sent + response received to a per-call file.

    Fires when COMPARATIVE_DEBUG=1 is set. Each (pair, order, judge prompt)
    gets its own file at $COMPARATIVE_DEBUG_DIR. If the env var is not set,
    writes to ./comparative_debug.
    """
    out_dir = Path(_DEBUG_DIR or "comparative_debug")
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_a = _safe_name(sample_a.id)
    safe_b = _safe_name(sample_b.id)
    filename = f"{format_type}__{safe_a}__vs__{safe_b}__{order_label}__p{idx:02d}_{label}.txt"
    out_file = out_dir / filename

    sep = "=" * 100
    sub = "-" * 100

    abs_images = None
    if image_paths:
        abs_images = [str(Path(p).resolve()) for p in image_paths]

    lines = [
        sep,
        f"model              = {model}",
        f"format             = {format_type}",
        f"order              = {order_label}",
        f"judge_prompt_idx   = {idx}  ({label}-framed, json_key={json_key})",
        f"sample_A.id        = {sample_a.id}",
        f"sample_A.source    = {sample_a.source}",
        f"sample_A.wtype     = {sample_a.website_type}",
        f"sample_B.id        = {sample_b.id}",
        f"sample_B.source    = {sample_b.source}",
        f"sample_B.wtype     = {sample_b.website_type}",
    ]
    if abs_images:
        lines.append(f"image_A (A slot)   = {abs_images[0]}")
        lines.append(f"image_B (B slot)   = {abs_images[1]}")
    lines.extend([
        sep,
        "",
        "=== PROMPT TEXT SENT TO MODEL ===",
        sub,
        prompt_text,
        sub,
    ])
    if content_a is not None:
        lines.append(f"\n(content_a: {len(content_a)} chars, embedded in prompt above)")
        lines.append(f"(content_b: {len(content_b) if content_b else 0} chars, embedded in prompt above)")
    lines.extend([
        "",
        "=== RAW RESPONSE TEXT FROM MODEL ===",
        sub,
        response_text,
        sub,
    ])
    out_file.write_text("\n".join(lines), encoding="utf-8")

from eval_awareness_experiments.experiments.base import BaseExperiment
from eval_awareness_experiments.experiments.prompts import (
    JUDGE_PROMPTS,
    build_comparative_prompt,
    build_comparative_prompt_screenshot,
)
from eval_awareness_experiments.judges.json_parsing import parse_json_safe
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

    @staticmethod
    def _normalize_website_type(wtype: str) -> str:
        """Normalize website_type for cross-source matching.

        WAI uses subtypes like 'gmail-accounts-and-contacts' or
        'gitlab-plan-and-track'; normalize to base product name so they
        can be matched against real/TAC sources.
        """
        # Known base product names — order matters (longest prefix first)
        BASE_PRODUCTS = [
            "elation", "figma", "gitlab", "gmail", "handshake",
            "linear", "paypal", "superhuman", "xero",
        ]
        lower = wtype.lower()
        for base in BASE_PRODUCTS:
            if lower == base or lower.startswith(base + "-"):
                return base
        return wtype

    async def run_pairs(
        self,
        samples: list[WebsiteSample],
        format_type: str,
        max_per_side: int | None = None,
        seed: int = 42,
        cross_type: bool = False,
    ) -> list[WebsiteExperimentResult]:
        """Run pairwise comparisons between samples from different sources.

        Groups samples by normalized website_type, then creates cross-source
        pairs. Each pair is run in both AB and BA order for counterbalancing.

        Args:
            max_per_side: If set, sample at most this many pages per source
                per website type to limit the cartesian product.
            seed: Random seed for sampling.
            cross_type: If True, ignore website_type entirely — pair any
                sample with any other cross-source sample. Useful when
                sources have no overlapping website types (e.g. worldsim).
        """
        rng = random.Random(seed)

        # Group by normalized website_type (or one bucket if cross_type)
        by_type: dict[str, list[WebsiteSample]] = {}
        for s in samples:
            key = "__all__" if cross_type else self._normalize_website_type(s.website_type)
            by_type.setdefault(key, []).append(s)

        all_results = []
        output_file = self.output_dir / f"{self.name}_results.jsonl"

        # Resume: load already-completed (sample_id, order) pairs for this format
        done: set[tuple[str, str]] = set()
        if output_file.exists():
            for line in output_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("format_used") != format_type:
                    continue
                order = (r.get("metadata") or {}).get("order")
                sid = r.get("sample_id")
                if sid and order:
                    done.add((sid, order))
            if done:
                logger.info(f"Resume: skipping {len(done)} already-completed entries for format={format_type}")

        for wtype, type_samples in sorted(by_type.items()):
            # Group by source within this website type
            by_source: dict[str, list[WebsiteSample]] = {}
            for s in type_samples:
                by_source.setdefault(s.source, []).append(s)

            sources = list(by_source.keys())
            if len(sources) < 2:
                continue

            # Optionally subsample per source
            if max_per_side:
                for src in by_source:
                    if len(by_source[src]) > max_per_side:
                        by_source[src] = rng.sample(by_source[src], max_per_side)

            logger.info(f"Comparative group '{wtype}': {', '.join(f'{s}({len(v)})' for s, v in by_source.items())}")

            # Generate cross-source pairs
            for src_a, src_b in combinations(sources, 2):
                for sample_a in by_source[src_a]:
                    for sample_b in by_source[src_b]:
                        ab_id = f"{sample_a.id}_vs_{sample_b.id}"
                        ba_id = f"{sample_b.id}_vs_{sample_a.id}"

                        pair_results = []
                        if (ab_id, "AB") not in done:
                            try:
                                result_ab = await self._compare_pair(
                                    sample_a, sample_b, format_type, "AB"
                                )
                                pair_results.append(result_ab)
                            except Exception as e:
                                logger.error(f"Pair {ab_id} AB crashed: {e}")
                        if (ba_id, "BA") not in done:
                            try:
                                result_ba = await self._compare_pair(
                                    sample_b, sample_a, format_type, "BA"
                                )
                                pair_results.append(result_ba)
                            except Exception as e:
                                logger.error(f"Pair {ba_id} BA crashed: {e}")

                        all_results.extend(pair_results)

                        # Incremental save
                        if pair_results:
                            with open(output_file, "a", encoding="utf-8") as f:
                                for r in pair_results:
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
            label, json_key, _preamble, _question = prompt_tuple

            try:
                return await _run_one_inner(idx, prompt_tuple)
            except Exception as e:
                logger.warning(f"pair {sample_a.id} vs {sample_b.id} prompt {idx} failed: {e}")
                return {
                    "prompt_idx": idx,
                    "label": label,
                    "json_key": json_key,
                    "choice": None,
                    "confidence": None,
                    "reasoning": "",
                    "chosen_eval_source": None,
                    "raw_response": None,
                    "error": str(e),
                }

        async def _run_one_inner(idx: int, prompt_tuple: tuple) -> dict:
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

            if _DEBUG:
                _debug_dump(
                    model=self.model.model,
                    sample_a=sample_a, sample_b=sample_b,
                    format_type=format_type, order_label=order_label,
                    idx=idx, label=label, json_key=json_key,
                    prompt_text=prompt_text,
                    content_a=content_a if format_type != "screenshot" else None,
                    content_b=content_b if format_type != "screenshot" else None,
                    image_paths=[content_a, content_b] if format_type == "screenshot" else None,
                    response_text=response_text,
                )

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
