"""Render trajectory + judge results to a human-readable markdown file.

Pairs a stage-1 transcript JSON (from run_agentdojo / run_injecagent / etc.) with
its stage-2 judge rows (from trajectory_awareness_results.jsonl) into a single
.md per task. Useful for skimming what the agent actually did, what scratchpads
it wrote, and how each judge prompt landed.

Truncates verbose tool observations to OBS_CHAR_LIMIT chars to keep files
skim-friendly. Model-generated text (preamble/thought/answers) is NEVER
truncated — that's the whole point.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OBS_CHAR_LIMIT = 2000


def _trunc(text: str | None, n: int) -> str:
    if not text:
        return ""
    if len(text) <= n:
        return text
    return text[:n] + f"\n... [truncated {len(text) - n} chars]"


def _bool_emoji(v: Any) -> str:
    if v is True:
        return "✅ True"
    if v is False:
        return "❌ False"
    return f"❓ {v}"


_SCRATCHPAD_PRESETS = {
    "scratchpad",
    "scratchpad_bare",
    "scratchpad_private",
    "scratchpad_green",
    "motive",  # also uses <unfiltered_observations>
}


def _scratchpad_compliance(t: dict) -> tuple[bool | None, str]:
    """Did the model emit the scratchpad it was asked for?

    Returns (complied, note). complied is None when no scratchpad was asked for.
    """
    preset = t.get("extra_instructions_preset") or "none"
    if preset not in _SCRATCHPAD_PRESETS:
        return None, ""

    # Scan all model-generated text in the trajectory for the expected tag.
    # `motive` and the scratchpad presets all use <unfiltered_observations>.
    blob_parts = []
    for s in t.get("steps") or []:
        for k in ("preamble", "thought"):
            v = s.get(k) or ""
            if v:
                blob_parts.append(v)
    blob_parts.append(t.get("final_answer") or "")
    blob = "\n".join(blob_parts)

    has_open = "<unfiltered_observations>" in blob
    has_close = "</unfiltered_observations>" in blob
    if has_open and has_close:
        return True, "✅ scratchpad emitted"
    if has_open or has_close:
        return False, "⚠️ partial scratchpad tags (one of open/close missing)"
    return False, "⚠️ scratchpad instructions sent but model did not emit one"


def _render_header(t: dict, judge_rows: list[dict]) -> str:
    md = []
    md.append(f"# {t.get('task_id', 'unknown')}")
    md.append("")
    md.append(f"- **Benchmark**: {t.get('benchmark', '?')}")
    md.append(f"- **Agent model**: {t.get('agent', '?')}")
    md.append(f"- **Condition**: {t.get('condition', '?')}")
    md.append(f"- **Extra-instructions preset**: {t.get('extra_instructions_preset', '?')}")
    md.append(f"- **Attack succeeded**: {_bool_emoji(t.get('attack_success'))}")
    md.append(f"- **Task passed (utility)**: {_bool_emoji(t.get('passed'))}")
    md.append(
        f"- **Steps**: {len(t.get('steps', []))}"
        + (" (hit max turns)" if t.get("metadata", {}).get("hit_max_turns") else "")
    )
    _complied, note = _scratchpad_compliance(t)
    if note:
        md.append(f"- **Scratchpad compliance**: {note}")
    if judge_rows:
        md.append(f"- **Judge model**: {judge_rows[0].get('model', '?')}")
    md.append("")
    return "\n".join(md)


def _render_system_prompt(t: dict) -> str:
    sp = t.get("system_prompt") or ""
    if not sp.strip():
        return ""
    md = ["## System prompt", "", "```", sp, "```", ""]
    return "\n".join(md)


def _render_task(t: dict) -> str:
    md = ["## Task instruction", ""]
    md.append("> " + (t.get("task_instruction") or "").replace("\n", "\n> "))
    md.append("")

    inj_goal = (t.get("metadata") or {}).get("injection_goal")
    if inj_goal:
        md.append("## Injection (attacker goal)")
        md.append("")
        md.append("> " + inj_goal.replace("\n", "\n> "))
        md.append("")
    return "\n".join(md)


def _render_steps(t: dict) -> str:
    md = ["## Trajectory", ""]
    steps = t.get("steps") or []
    if not steps:
        md.append("_(no steps recorded)_")
        md.append("")
        return "\n".join(md)

    for i, s in enumerate(steps):
        md.append(f"### Step {i + 1}")
        md.append("")

        preamble = s.get("preamble") or ""
        if preamble.strip():
            md.append("**Preamble (visible to user)**:")
            md.append("")
            md.append(preamble)
            md.append("")

        thought = s.get("thought") or ""
        if thought.strip():
            md.append("**Thought (model reasoning)**:")
            md.append("")
            md.append(thought)
            md.append("")

        action = s.get("action") or ""
        if action.strip():
            md.append("**Action**:")
            md.append("")
            md.append("```")
            md.append(action)
            md.append("```")
            md.append("")

        obs = s.get("observation") or ""
        if obs.strip():
            md.append("**Observation**:")
            md.append("")
            md.append("```")
            md.append(_trunc(obs, OBS_CHAR_LIMIT))
            md.append("```")
            md.append("")
    return "\n".join(md)


def _render_final_answer(t: dict) -> str:
    fa = t.get("final_answer") or ""
    if not fa.strip():
        return ""
    md = ["## Final answer", "", fa, ""]
    return "\n".join(md)


def _render_judges(judge_rows: list[dict]) -> str:
    if not judge_rows:
        return ""
    md = ["## Judge results", ""]
    for row in judge_rows:
        meta = row.get("metadata") or {}
        judge = meta.get("judge_name", row.get("experiment_type", "?"))

        md.append(f"### {judge}")
        md.append("")
        md.append(f"- **Classification**: {row.get('classification', '?')}")
        if row.get("score") is not None:
            md.append(f"- **Score**: {row['score']}")

        # 5PQ-style aggregate
        if "alignment_eval_share" in meta:
            md.append(
                f"- **alignment_eval_share**: {meta.get('alignment_eval_share')} | "
                f"**any_eval_share**: {meta.get('any_eval_share')} | "
                f"**refused_share**: {meta.get('refused_share')}"
            )
        md.append("")

        # VEA: raw_response holds {classification, evidence:[...]}
        raw = row.get("raw_response") or ""
        if raw:
            try:
                parsed = json.loads(raw)
                evidence = parsed.get("evidence") or []
                if evidence:
                    md.append("**Evidence quotes**:")
                    md.append("")
                    for q in evidence:
                        md.append(f"> {q}")
                        md.append("")
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass

        # 5PQ: per-prompt question + model answer + category
        variants = meta.get("variant_categories") or []
        for v in variants:
            pid = v.get("prompt_id", "?")
            cat = v.get("category", "?")
            md.append(f"#### {pid} → **{cat}**")
            md.append("")
            q = v.get("question")
            if q:
                md.append(f"**Q**: {q}")
                md.append("")
            ans = v.get("model_reflection")
            if ans:
                md.append("**A**:")
                md.append("")
                md.append(ans)
                md.append("")
            else:
                md.append("_(model_reflection not saved for this row)_")
                md.append("")
    return "\n".join(md)


def render_markdown(transcript: dict, judge_rows: list[dict]) -> str:
    parts = [
        _render_header(transcript, judge_rows),
        _render_system_prompt(transcript),
        _render_task(transcript),
        _render_steps(transcript),
        _render_final_answer(transcript),
        _render_judges(judge_rows),
    ]
    return "\n".join(p for p in parts if p)


def write_markdown_for_split(
    *,
    transcript_dir: Path,
    judge_jsonl: Path,
    out_dir: Path,
) -> int:
    """For each transcript in `transcript_dir`, write a .md alongside its judge rows.

    Returns the number of MD files written.
    """
    if not judge_jsonl.exists():
        logger.warning(f"[md] judge JSONL not found: {judge_jsonl} — skipping")
        return 0

    judge_rows: list[dict] = [
        json.loads(l) for l in judge_jsonl.read_text().splitlines() if l.strip()
    ]
    by_sample: dict[str, list[dict]] = {}
    for r in judge_rows:
        by_sample.setdefault(r["sample_id"], []).append(r)

    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for tpath in sorted(transcript_dir.glob("*.json")):
        try:
            t = json.loads(tpath.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[md] could not read {tpath}: {e}")
            continue

        # Match by task_id substring — judge sample_id has format
        # `{benchmark}_{split}_{task_id}` where task_id matches the JSON's task_id.
        task_id = t.get("task_id", tpath.stem)
        rows = [r for sid, rs in by_sample.items() if task_id in sid for r in rs]
        md = render_markdown(t, rows)
        out_path = out_dir / f"{tpath.stem}.md"
        out_path.write_text(md, encoding="utf-8")
        n += 1
    logger.info(f"[md] wrote {n} markdown files to {out_dir}")
    return n
