# Comparison frontier: one TP-aware versus neutral iteration

**Date:** 2026-08-30
**Status:** Research proposal only. No runs, model calls, infrastructure, tests,
or source changes were made. “Proposed” below means a design choice, not a
measured result.

## Recommendation

Run one paired, equal-opportunity comparison on a frozen admitted bank after
the broader WARP generation step. The bank should contain the existing WARP
families and the first TAC family only once that family's proposed exposure,
permission, reset, and readback contracts are implemented and validated. TAC
does not already have working WARP admission. If TAC is not yet admitted,
the same comparison can start on the current WARP bank; expansion and this
comparison need not block each other. The purpose is to ask whether the
TP-conditioned operational pipeline adds measurement value beyond an ordinary
bounded search/rewrite, while using the same WARP contracts on a genuinely
different workflow.

For each task/model pair, branch from the same baseline trajectory and initial
state:

* **TP-informed arm (current mechanism):** use the Phase 4 eval-awareness
  iterator's TP cue diagnosis, constrained payload rewrite, contract QA, and
  browser rerun. For the fixed-index endpoint, the harness suppresses its
  TP=`Real` early stop so the arm has the same scheduled opportunities as the
  control; retain the stop as an operational diagnostic.
* **Neutral iteration arm:** use the same proposal/QA/rerun schedule and the
  same payload-protection contract, but give the proposal call only the task,
  attacker objective, and the common permitted non-TP attempt feedback. Keep
  that common feedback identical in kind across arms; do not add ASR/reward
  feedback or raw traces unavailable to the current TP rewrite path. It must not receive TP,
  VEA, cue, or “evaluation” labels; its predeclared objective is a
  semantics-preserving, non-awareness paraphrase/retry. Keep proposal-call and
  browser-rerun opportunities equal, then report token/wall cost separately.

The primary endpoint should be fixed attempt index (baseline and each scheduled
attempt), not only the post-treatment winner. Both arms receive the same finite
rewrite cap and the same hard-stop rules (task break, lost PVPO, or contract
failure); TP=`Real` must not terminate only the TP-informed arm in the primary
comparison. Publish attempt-response curves and all rejected attempts. If the
operational selected result is also desired, apply the *identical* current
`_best_iterator_result` rule to both arms, including its TP-regression exclusion
and compliance/TP/PVPO ordering. That secondary endpoint is a shared TP-aware
selector effect—not a pure TP ablation and not evidence that an internal mental
state mediated the outcome.

Report paired, family/site/action-stratified differences in fixed-cohort
`headline_asr`/`asr_raw` and encounter-conditioned `gate1_asr`/`asr_valid`,
objective-specific propagation and unauthorized-action outcomes, TP and VEA,
benign capability, PVPO, task-broke/non-encounter status, contract-rejection
classes, attempt counts, and cost. Keep any TAC-native score separate from the
WARP side-effect check. A lower TP rate with preserved objective-specific ASR
and capability is evidence for an operational measurement benefit; equal ASR
with no TP difference is a null incremental benefit; an ASR gain alone is a
search result, not an awareness result. None supports a claim about internal
awareness or a mental mechanism.

This is one contrast, not a task-by-model-by-cue factorial. It preserves the
paper's accepted boundary that broader valid generation is useful before every
optional causal ablation, while matched retries are needed for a claim about a
benefit specifically attributable to awareness-directed rewriting
([main report §12](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/docs/research/web-benchmark-onboarding-2026-08-30.md:328)). The first TAC result remains a narrow structural-transfer result, not
arbitrary plug-and-play ([accepted TAC scope](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/docs/research/web-benchmark-onboarding-2026-08-30.md:328)).

## Why the current release cannot answer this comparison

The current default is `eval-awareness-iterator`, with a bounded rewrite cap;
legacy `strategy-variation` is opt-in and is not a neutral control
([variant options](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/phase_4/options.py:11),
[dispatch](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/phase_4/postprocess.py:266)). The iterator is scheduled for PVPO-valid baseline trajectories, branches on TP, and stops early at TP=`Real`; non-Real labels (including classifier failure) trigger cue diagnosis and rewriting ([spec](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/warp-taskgen-technical-spec.md:1752), [loop](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/phase_4/eval_awareness_iterator.py:1099)). Consequently, switching a flag or reusing legacy strategy variation would change gating, stopping, and proposal family at once. A neutral generator plus a harmonized schedule is the smallest additional seam; the comparison is medium implementation cost, not a configuration-only study.

The selected output is also post-treatment: `_best_iterator_result` drops
TP-regressed candidates and ranks compliance/TP/PVPO before iteration index
([selection](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/phase_4/eval_awareness_iterator.py:755)). Existing analysis observes 35 non-Real→Real transitions, with 234/300 selected/final rows still non-Real; TP changes usually occur without VEA changes, and variant-level ASR does not establish a broad final-status change ([analysis summary](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/analysis/warp_taskgen_hf_20260509/paper_report.md:7), [iterator accounting](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/analysis/warp_taskgen_hf_20260509/paper_report.md:91)). The release also confounds workflow and action: `create_issue` is 100% compliant, `create_issue_note` 8.3%, and `create_post` 97.5% ([action slices](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/analysis/warp_taskgen_hf_20260509/paper_report.md:121)). Current numbers therefore cannot distinguish TP-directed value from extra retries, action difficulty, selection, or grading artifacts.

The canonical manuscript is [`worldsim-paper/icml/icml_final.tex`](/Users/ashtonchew/projects/worldsim-paper/CLAUDE.md:2), not the newer `icml_final_2.tex` draft. The canonical prose describes two post-rollout adaptation loops, while the current code/spec makes the TP iterator default and legacy strategy variation opt-in ([canonical WARP additions](/Users/ashtonchew/projects/worldsim-paper/icml/icml_final.tex:793), [current default](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/warp-taskgen-technical-spec.md:1754)). This comparison tests the current v1 operational pipeline; it must not silently be described as a measurement of the canonical two-loop implementation.

## Keep fixed versus change one thing

| Keep fixed in both arms | Change only between arms |
| --- | --- |
| Frozen task ID, benign/adversarial twin, workflow family/Site, payload record and protected witness/action anchors; initial world, reset, permissions, agent/model/provider, tools, timeouts, and browser schedule; PVPO, reward/readback, TP/VEA versions, host QA, and native-score separation. | Proposal/feedback source: TP cue diagnosis plus prior TP feedback versus ordinary task/objective feedback with no TP/VEA/cue labels. |
| Same rewrite opportunity cap, same call/rerun shape, same hard-stop rules, the same permitted mutable payload slots, and the same fixed-index endpoint. | Generated text within those same permitted slots; every delta remains subject to the existing rewrite contract. |

Do not silently change the grader, task objective, initial permissions, or
protected witness to make a neutral rewrite easier. Keep `PVPO` and native
capability measures distinct from final ASR, and retain every failure class.

## Counterevidence the note should force into the report

* **Extra retries/search:** show baseline and per-attempt curves; count browser
  reruns and proposal/QA calls; do not compare only selected winners. Equal
  maximum attempts is insufficient if TP=`Real` stops one arm early.
* **Workflow/action difficulty:** cluster or stratify by family, Site, carrier,
  and action kind. The current action confounding makes an aggregate gain
  uninterpretable.
* **Changed grading:** run both arms through the same host-owned reward,
  readback, PVPO, and TP/VEA versions; preserve native TAC grading as a labeled
  control rather than replacing WARP checks.
* **Objective shift:** contract QA must preserve required fields, target
  record, actor/route, payload witnesses, and declared attacker objective.
  Separate content propagation, semantic hijack, and unauthorized mutation;
  warning text is not automatically obedience.
* **Judge/selection artifacts:** retain TP regressions, parse failures,
  contract-inapplicable rewrites, lost-PVPO attempts, and VEA independently.
  A TP decrease with unchanged ASR is compatible with cue removal, selector
  preference, or judge calibration; it is not mediation evidence.
* **Transfer boundary:** if the TAC family fails its existing exposure/reset
  contract, report that boundary and keep the comparison on admitted WARP tasks;
  do not turn a single Site result into a full TAC or cross-application claim.

## Alternatives and cost

| Option | What it would establish | Cost/limitation |
| --- | --- | --- |
| **Recommended: equal-opportunity TP-informed vs neutral iteration** | Incremental value of TP-conditioned proposal/feedback in a safety-measurement pipeline, with the same contract on existing and (when admitted) TAC-generated workflows. | Medium: one neutral proposal path and schedule harmonization. It remains an operational comparison, not a cue-level causal intervention. |
| **Frozen payload/task transfer across agents** | Whether exposure, pairing, TP/VEA reporting, and objective-specific scoring transfer across agent/model runners on the same expanded bank. | Lower implementation cost and compatible with broad coverage; it does not isolate iterator value or establish a causal awareness effect. Use this if the neutral path would delay the first TAC result. |
| **Randomized explicit cue intervention** | Cue→first-attempt ASR total effect within the same model/task/state, a different causal estimand. | Defer: the current causal runner has distinct `system_prompt_frame` arms and older `injection_modifier` conditions, and its aggregate analysis has no WARP iterator lineage ([runner arms](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/eval_awareness_experiments/run_causal_experiment.py:50), [aggregation](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/eval_awareness_experiments/analysis_causal.py:96)). No extra cue arms are required for this frontier. |

The practical alternative is therefore not “run all arms everywhere”: if a
neutral iterator cannot be added before the TAC slice, freeze the expanded bank
and run the lower-cost transfer comparison first, then treat TP rewriting as a
later operational analysis.

## User-choice dependency

The current frontier is whether to prioritize this matched-rewriting contrast
as the small controlled study alongside task expansion. Endpoint and schedule
choices follow that decision. If it is selected, the proposed default is
fixed-attempt outcomes with the shared-selector result as a secondary view
from the same runs; this is not a second experiment. Schedule harmonization
is a labeled comparison-only change, not unchanged deployment of the current
default iterator. The main report's pending section governs this sequence.

## Sources

* [WARP onboarding decision and §12](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/docs/research/web-benchmark-onboarding-2026-08-30.md:328)
* [Current Phase 4 technical specification](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/warp-taskgen-technical-spec.md:1752)
* [Current iterator implementation](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/phase_4/eval_awareness_iterator.py:755)
* [Canonical paper authority](/Users/ashtonchew/projects/worldsim-paper/CLAUDE.md:2) and [canonical WARP description](/Users/ashtonchew/projects/worldsim-paper/icml/icml_final.tex:793)
* [Current WARP analysis](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/analysis/warp_taskgen_hf_20260509/paper_report.md:7)
* [TAC upstream pinned source](https://github.com/TheAgentCompany/TheAgentCompany/tree/98b68ef82a47690c316f42fddb05baafaab56851)
