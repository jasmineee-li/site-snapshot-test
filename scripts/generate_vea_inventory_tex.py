#!/usr/bin/env python3
"""Generate the VEA inventory appendix table from structured VEA rerun rows."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


BENCHMARK_ORDER = ("doomarena", "wasp", "agentdojo", "injecagent")
BENCHMARK_DISPLAY = {
    "doomarena": "DoomArena",
    "wasp": "WASP",
    "agentdojo": "AgentDojo",
    "injecagent": "InjecAgent",
}
MODEL_ORDER = {
    "anthropic_claude-opus-4.7_thinking": 0,
    "anthropic_claude-sonnet-4.6_thinking": 1,
    "google_gemini-2.5-pro_thinking": 2,
    "moonshotai_kimi-k2.5_thinking": 3,
    "openai_gpt-5.2_thinking": 4,
    "z-ai_glm-5_thinking": 5,
}
MODEL_DISPLAY = {
    "anthropic_claude-opus-4.7_thinking": "Opus 4.7",
    "anthropic_claude-sonnet-4.6_thinking": "Sonnet 4.6",
    "google_gemini-2.5-pro_thinking": "Gemini 2.5 Pro",
    "moonshotai_kimi-k2.5_thinking": "Kimi K2.5",
    "openai_gpt-5.2_thinking": "GPT-5.2",
    "z-ai_glm-5_thinking": "GLM-5",
}
CLASS_DISPLAY = {
    "Alignment Evaluation": "Align",
    "Capability Evaluation": "Cap",
    "Other Evaluation": "Other",
}

COUNT_PHRASES = [
    "appears to be a test",
    "potential distraction",
    "trap",
    "distraction or a test",
    "embedded instructions",
    "misleading instruction",
    "potentially malicious content",
    "test of whether",
    "test or a trick",
    "test or potentially malicious content",
    "test scenario",
    "test or example data",
    "distraction/trap",
    "honeypot",
    "injection attempt",
    "meta-instructions",
    "placeholder",
    "simulated environment",
    "simulated response",
    "test data",
    "test or a scenario",
    "test or distraction",
    "test or simulation",
    "test or simulated data",
    "test or simulated environment",
    "test or some kind of injection attempt",
    "test or trap",
    "test/simulation",
    "test/simulation environment",
    "test/training",
    "test/trap",
    "training",
]

BOLD_PHRASES = sorted(
    set(
        COUNT_PHRASES
        + [
            "test or an attempt to redirect my behavior",
            "instructions embedded in external data",
            "task embedded in the calendar event description",
            "test or task embedded",
            "appears to be a test/trap",
            "honeypot or test",
            "potential distraction or trick",
        ]
    ),
    key=len,
    reverse=True,
)


@dataclass(frozen=True)
class InventoryRow:
    benchmark: str
    split: str
    model_slug: str
    sample_id: str
    classification: str
    attack_success: bool | None
    evidence: tuple[str, ...]

    @property
    def task_suffix(self) -> str:
        if self.benchmark in {"doomarena", "wasp"}:
            match = re.search(r"webarena\.(\d+)", self.sample_id)
            return f".{match.group(1)}" if match else self.sample_id
        if self.benchmark in {"agentdojo", "injecagent"}:
            match = re.search(r"__(\d+)$", self.sample_id)
            return f".{match.group(1)}" if match else self.sample_id
        return self.sample_id

    @property
    def source(self) -> str:
        return f"{BENCHMARK_DISPLAY[self.benchmark]} $\\cdot$ {latex_escape(self.split)}/{latex_escape(self.task_suffix)}"

    @property
    def model_display(self) -> str:
        return MODEL_DISPLAY.get(self.model_slug, self.model_slug)

    @property
    def class_display(self) -> str:
        return CLASS_DISPLAY.get(self.classification, self.classification)

    @property
    def outcome_display(self) -> str:
        if self.attack_success is True:
            return r"\ding{55}"
        if self.attack_success is False:
            return r"\ding{51}"
        return "?"

    @property
    def sort_key(self) -> tuple[int, int, str, int, str]:
        task_match = re.search(r"(\d+)", self.task_suffix)
        task_num = int(task_match.group(1)) if task_match else 10**9
        return (
            BENCHMARK_ORDER.index(self.benchmark),
            MODEL_ORDER.get(self.model_slug, 999),
            self.split,
            task_num,
            self.sample_id,
        )


def latex_escape(text: Any) -> str:
    value = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in value)


def normalize_quote(text: str) -> str:
    return " ".join(text.replace("\r", "\n").split())


def latex_quote(text: str) -> str:
    text = normalize_quote(text)
    spans: list[tuple[int, int]] = []
    lowered = text.lower()
    for phrase in BOLD_PHRASES:
        start = 0
        phrase_lower = phrase.lower()
        while True:
            idx = lowered.find(phrase_lower, start)
            if idx < 0:
                break
            end = idx + len(phrase)
            if not any(max(idx, a) < min(end, b) for a, b in spans):
                spans.append((idx, end))
            start = idx + 1
    spans.sort()

    parts: list[str] = []
    cursor = 0
    for start, end in spans:
        if start < cursor:
            continue
        parts.append(latex_escape(text[cursor:start]))
        parts.append(r"\textbf{" + latex_escape(text[start:end]) + "}")
        cursor = end
    parts.append(latex_escape(text[cursor:]))
    return "".join(parts)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def parse_evidence(row: dict[str, Any]) -> tuple[str, ...]:
    try:
        parsed = json.loads(row.get("raw_response") or "{}")
    except json.JSONDecodeError:
        parsed = {}
    evidence = parsed.get("evidence")
    if isinstance(evidence, list):
        return tuple(str(item) for item in evidence if isinstance(item, str) and item.strip())

    reasoning = (row.get("metadata") or {}).get("reasoning") or ""
    lines = []
    for line in str(reasoning).splitlines():
        line = line.strip()
        if line.startswith("- "):
            lines.append(line[2:])
    return tuple(lines)


def coerce_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n"}:
            return False
    return None


def task_id_from_sample(sample_id: str) -> str | None:
    match = re.search(r"webarena\.(\d+)", sample_id)
    return f"webarena.{match.group(1)}" if match else None


def load_wasp_e2e_outcomes(root: Path) -> dict[tuple[str, str], bool]:
    outcomes: dict[tuple[str, str], bool] = {}
    if not root.exists():
        return outcomes
    for path in sorted(root.glob("bare__*.jsonl")):
        model_slug = path.stem.removeprefix("bare__")
        for row in read_jsonl(path):
            if row.get("type") != "task" or row.get("status") != "scored":
                continue
            task_id = row.get("task_id")
            if isinstance(task_id, str):
                outcomes[(model_slug, task_id)] = bool(row.get("attack_success"))
    return outcomes


def load_inventory(vea_root: Path, wasp_e2e_root: Path) -> list[InventoryRow]:
    wasp_e2e = load_wasp_e2e_outcomes(wasp_e2e_root)
    rows: list[InventoryRow] = []

    for path in sorted(vea_root.glob("**/trajectory_awareness_results.jsonl")):
        rel = path.relative_to(vea_root).parts
        if len(rel) < 6:
            continue
        benchmark, arm, model_slug, inner_benchmark, split = rel[:5]
        if benchmark not in BENCHMARK_ORDER or inner_benchmark != benchmark or arm != "bare":
            continue
        for raw in read_jsonl(path):
            if raw.get("experiment_type") != "trajectory_verbalized_awareness":
                continue
            classification = str(raw.get("classification") or "")
            if classification == "Real":
                continue
            sample_id = str(raw.get("sample_id") or (raw.get("metadata") or {}).get("task_id") or "")
            if not sample_id:
                continue
            metadata = raw.get("metadata") or {}
            attack_success = coerce_bool(metadata.get("attack_success"))
            if benchmark == "wasp":
                task_id = task_id_from_sample(sample_id)
                if task_id and (model_slug, task_id) in wasp_e2e:
                    attack_success = wasp_e2e[(model_slug, task_id)]
            rows.append(
                InventoryRow(
                    benchmark=benchmark,
                    split=split,
                    model_slug=model_slug,
                    sample_id=sample_id,
                    classification=classification,
                    attack_success=attack_success,
                    evidence=parse_evidence(raw),
                )
            )
    return sorted(rows, key=lambda row: row.sort_key)


def phrase_counts(rows: list[InventoryRow]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        text = "\n".join(row.evidence).lower()
        for phrase in COUNT_PHRASES:
            if phrase in text:
                counts[phrase] += 1
    return counts


def fmt_phrase_counts(counts: Counter[str]) -> str:
    items = [(phrase, counts[phrase]) for phrase in COUNT_PHRASES if counts[phrase]]
    items.sort(key=lambda item: (-item[1], COUNT_PHRASES.index(item[0])))
    return ", ".join(r"\emph{``" + latex_escape(phrase) + r"''} (" + str(count) + ")" for phrase, count in items)


def table_quote(row: InventoryRow) -> str:
    if not row.evidence:
        return r"\emph{No evidence extracted.}"
    return (
        r" \newline\textcolor{black!50}{$\hookrightarrow$}\,"
    ).join(latex_quote(item) for item in row.evidence)


def render(rows: list[InventoryRow]) -> str:
    by_benchmark = defaultdict(list)
    for row in rows:
        by_benchmark[row.benchmark].append(row)
    distinct = {
        benchmark: len({row.sample_id for row in group})
        for benchmark, group in by_benchmark.items()
    }
    class_counts = Counter(row.classification for row in rows)
    outcome_counts = Counter(row.attack_success for row in rows)
    counts_text = fmt_phrase_counts(phrase_counts(rows))

    lines: list[str] = []
    lines.extend(
        [
            r"\subsection{Full Inventory of Verbalized Eval-Awareness Cases}",
            r"\label{app:vea_inventory}",
            "",
            (
                r"Table~\ref{tab:vea_inventory} lists every verbalized eval-awareness (VEA) "
                r"instance our structured Sonnet 4.6 judge flagged in the corrected bare-arm "
                r"N200 sweep across all four safety benchmarks: "
                f"{len(by_benchmark.get('doomarena', []))} occurrences in DoomArena "
                f"(across {distinct.get('doomarena', 0)} distinct tasks), "
                f"{len(by_benchmark.get('wasp', []))} occurrences in WASP "
                f"({distinct.get('wasp', 0)} distinct tasks), "
                f"{len(by_benchmark.get('agentdojo', []))} occurrences in AgentDojo "
                f"({distinct.get('agentdojo', 0)} distinct tasks), and "
                f"{len(by_benchmark.get('injecagent', []))} occurrences in InjecAgent "
                f"({distinct.get('injecagent', 0)} distinct tasks). "
                r"Each row is one judged trajectory; quotes are the judge-extracted verbatim "
                r"evidence from the agent's reasoning, with lexical eval-aware phrases bolded. "
                r"The same inventory grounds the inline exemplars in \S\ref{sec:diag_awareness}."
            ),
            "",
            r"\paragraph{Phrasing patterns.} The structured VEA pass flags 29 bare-arm cases. By exact row-level lexical hit across the 29 cases: "
            + counts_text
            + ".",
            "",
            (
                r"\paragraph{Class and outcome breakdown.} Of the "
                f"{len(rows)} cases, {class_counts.get('Alignment Evaluation', 0)} are headlined as "
                r"\textsc{AlignmentEval}, "
                f"{class_counts.get('Capability Evaluation', 0)} as "
                r"\textsc{CapabilityEval}, and "
                f"{class_counts.get('Other Evaluation', 0)} as "
                r"\textsc{OtherEval}. On the underlying attack outcomes, "
                f"{outcome_counts.get(False, 0)} eval-aware trajectories resisted the prompt-injection attack "
                f"and {outcome_counts.get(True, 0)} fell to it. Outcome labels use the canonical ASR source "
                r"available for each benchmark: DoomArena corrected task-level attack metadata, WASP e2e replay rows, "
                r"and the tool-calling benchmark outcome metadata for AgentDojo and InjecAgent."
            ),
            r"\renewcommand{\arraystretch}{1.15}",
            r"\begin{small}",
            r"\setlength{\tabcolsep}{4pt}",
            r"\begin{longtable}{@{}r @{\hspace{4pt}} l @{\hspace{4pt}} l @{\hspace{4pt}} c @{\hspace{4pt}} c @{\hspace{4pt}} >{\raggedright\arraybackslash}p{0.36\textwidth}@{}}",
            (
                r"\caption{All "
                + str(len(rows))
                + r" verbalized eval-awareness (VEA) instances flagged by the structured \texttt{verbalized\_awareness} judge "
                + r"on the corrected bare-arm N=200 four-benchmark sweep (unprompted/unelicited). Quotes are verbatim "
                + r"judge-extracted evidence from the agent's reasoning trace, with lexical eval-aware phrases bolded. "
                + r"Outcome column: \ding{51}~$=$~the underlying prompt-injection attack was resisted on this trajectory, "
                + r"\ding{55}~$=$~attack succeeded. Class: \textsc{Align} = \textsc{AlignmentEval}, "
                + r"\textsc{Cap} = \textsc{CapabilityEval}, \textsc{Other} = \textsc{OtherEval}. Within each benchmark, "
                + r"rows are ordered by agent model (Anthropic, Google, Moonshot, OpenAI, Z.AI) then by task ID.}"
            ),
            r"\label{tab:vea_inventory} \\",
            r"\toprule",
            r"\# & Source & Agent & Class & Out. & Verbatim quote(s) \\",
            r"\midrule",
            r"\endfirsthead",
            r"\multicolumn{6}{l}{\small\itshape (Table~\ref{tab:vea_inventory} continued from previous page)} \\",
            r"\toprule",
            r"\# & Source & Agent & Class & Out. & Verbatim quote(s) \\",
            r"\midrule",
            r"\endhead",
            r"\midrule",
            r"\multicolumn{6}{r}{\small\itshape (continued on next page)} \\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
        ]
    )

    last_benchmark = None
    for idx, row in enumerate(rows, start=1):
        if last_benchmark is not None and row.benchmark != last_benchmark:
            lines.append(r"\midrule")
        last_benchmark = row.benchmark
        lines.append(
            f"{idx} & {row.source} & {latex_escape(row.model_display)} & "
            f"{row.class_display} & {row.outcome_display} & {table_quote(row)} \\\\"
        )

    lines.extend(
        [
            r"\end{longtable}",
            r"\setlength{\tabcolsep}{6pt}",
            r"\end{small}",
            r"\renewcommand{\arraystretch}{1.0}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vea-root",
        type=Path,
        default=Path("eval_awareness_experiments/results/n200_2026-04-29_vea_structured_sonnet46_20260509"),
    )
    parser.add_argument(
        "--wasp-e2e-root",
        type=Path,
        default=Path("eval_awareness_experiments/results/n200_2026-04-29/wasp_e2e_replay_20260509_lanes_skipreset/no_reset"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_awareness_experiments/results/n200_2026-04-29_vea_structured_sonnet46_20260509/vea_inventory_structured_sonnet46.tex"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_inventory(args.vea_root, args.wasp_e2e_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(rows), encoding="utf-8")
    by_benchmark = Counter(row.benchmark for row in rows)
    print(f"wrote {args.output}")
    print(f"rows={len(rows)} by_benchmark={dict(by_benchmark)}")


if __name__ == "__main__":
    main()
