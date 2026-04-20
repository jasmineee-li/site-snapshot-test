# Codex handoff: audit Needham et al canonical prompts vs orphan-branch port

> **Outcome (2026-04-20):** This audit informed the Phase-4 VEA rewrite that landed on `feat/worldsim-v5` (commits `0cabb4db`, `d383ed01`, `f6bc7dc2`). The shipped split: VEA uses **Jasmine's 4-way classification prompt** (`origin/claude/find-eval-aware-trajectories-IpDdi`, `eval_awareness_experiments/judges/prompts.py::VERBALIZED_AWARENESS_PROMPT`) over **Needham's exact XML transcript format** (`jjpn97/eval_awareness` `pipeline/apply_pq.py::_format_xml`, byte-equivalent port at `worldsim/phase_4/needham_xml.py`). Jasmine's `transcript_to_xml` flat-tag serializer was explicitly NOT adopted. IPI-AER p(eval) stays on its existing AER text format (separate metric, separate plan). The new judge module is `worldsim/phase_4/verbalized_eval_awareness_api.py`; the prompt body is at `worldsim/prompts/verbalized-eval-awareness.md` with one surgical edit retagging the "What Counts as 'the Model's Output'" block from flat tags to Needham's `<message role="assistant">` shape. See `docs/worldsim-v5-technical-specifcation.md` §"Gate 1 Stage 2: Observational IPI-AER p(eval) + VEA" for the current design.

## Context (self-contained)

This repository is `feat/worldsim-v5` branch of an IPI safety evaluation benchmark project. Track 1 of `docs/TODO-2-paper-experiments.md` plans a rigorous three-signal eval-awareness probe battery (verbalized eval awareness, external P(eval), self-reported P(eval)) for Phase 5. Two teammates (Jasmine + Max) have a partial implementation on an orphan branch that we plan to port.

A recent GPT Deep Research pass (summarized in `docs/research/gpt-deep-research-2026-04-14.md`, "External P(eval) and VEA prompts" section) located Needham et al.'s public reproduction repo for the canonical prompts those probes are based on. Before porting the orphan branch, we want a comparison showing which parts are verbatim vs paraphrased vs divergent. The output is a research decision document, not code.

## Sources

**Needham et al reproduction repo (external)**: `https://github.com/jjpn97/eval_awareness`
Pinned commit: `1543ffe644d3f8f3a43b5dfd7814dbe9b12f890d`

Key files to read:
- `cot_classifiers/prompts.py` (VEA / reasoning-flag classifier prompts, Figure 16 style flags)
- `constants.py` (5-probe-question bank, Table 4 text)
- `run.py` (orchestration showing how prompts are assembled into probes)
- `analysis/roc.py` (ROC curve generation; reference layout)
- `analysis/purpose.py` (MCQ vs open-ended purpose classification, Table 9)
- `analysis/cot.py` (CoT analysis pipeline)

Paper reference: "Large Language Models Often Know When They Are Being Evaluated" (Needham et al, 2025). Judge model in the paper is GPT-4.1; human agreement metric is ICC(3) with 95% CI; output format is XML-ish `<label>YES</label>`.

**Orphan branch (internal)**: `origin/claude/plan-eval-experiments-LNd9F`

Key files to read (use `git show origin/claude/plan-eval-experiments-LNd9F:<path>`):
- `eval_awareness_experiments/judges/verbalized_awareness.py` (VEA extractor, 4-category classification)
- `eval_awareness_experiments/experiments/p_eval.py` (five-prompt bank across screenshot / HTML / axtree formats)
- `eval_awareness_experiments/analysis_causal.py` (ASR x awareness plots using numpy + matplotlib)

If a file named above does not exist at that branch + path, grep the branch for the closest match and note the actual path in your output.

## Deliverable

A single markdown file at `docs/research/needham-vs-orphan-port-diff.md` with this structure:

1. **Summary** (2-3 sentences): overall verdict on how close the orphan port is to Needham canonical.
2. **Prompt-by-prompt comparison**: for each Needham prompt (VEA flags, P(eval) PQs, purpose classification), a table or bullet list showing:
   - Needham source (file:line in jjpn97/eval_awareness)
   - Orphan source (file:line in plan-eval-experiments-LNd9F)
   - Relationship: verbatim / paraphrased / rewritten / missing
   - Any semantic divergence in one sentence
3. **Analysis-script comparison**: for each of Needham's `analysis/{roc,purpose,cot}.py` files, note whether the orphan `analysis_causal.py` covers equivalent functionality, has partial overlap, or is missing it. Line-level citations.
4. **Judge-model note**: single paragraph confirming Needham's judge (GPT-4.1 per the paper) vs what the orphan port uses, and whether the orphan port is judge-agnostic.
5. **Recommendation**: one of three options, with a 2-3 sentence rationale:
   - "Keep orphan port as-is" (minor divergence, not worth the re-port cost)
   - "Replace with Needham canonical" (significant divergence, canonical wins)
   - "Merge" (keep orphan's adaptations but pull specific Needham prompts verbatim; list which)

## Verification

- The diff document cites at least one file:line on both sides for every Needham prompt mentioned.
- The recommendation matches one of the three enumerated options verbatim.
- The document opens with a summary paragraph that someone not already familiar with either side can read in under 60 seconds.

## Non-goals

- Do NOT edit the orphan branch or merge it.
- Do NOT port any prompt into `worldsim/` or modify any code under `worldsim/`, `tests/`, or `docs/` other than the new diff file.
- Do NOT run the Needham codebase; reading is sufficient.
- Do NOT update `docs/TODO-2-paper-experiments.md`; that is separately tracked.

## Operating constraints

- No commits unless the user explicitly asks.
- No pushes ever.
- Do not switch branches; use `git show` for cross-branch reads.
- No em dashes in prose; use commas or parentheses.
- If GitHub access is restricted, fetch the Needham repo content via `git clone` into `.cache/` rather than relying on live URLs.

## Expected output to the user when done

A one-paragraph summary: path to the diff document, the recommendation chosen (one of the three), and any surprising findings (e.g., large divergence, missing prompts, or semantic drift).
