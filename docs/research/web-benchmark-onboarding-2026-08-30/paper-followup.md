# Paper follow-up: evidence threshold for expanding WARP's generated corpus

**Date:** 2026-08-30
**Scope:** paper-improvement audit only; no runs, model calls, infrastructure, or source edits.

## Provenance and bottom line

[paper guidance](/Users/ashtonchew/projects/worldsim-paper/CLAUDE.md:2)
identifies `icml/icml_final.tex` as the active ICML manuscript. The newer
`icml/icml_final_2.tex` is a separate, non-canonical draft/poster source; its
stronger cross-benchmark wording (for example,
[abstract](/Users/ashtonchew/projects/worldsim-paper/icml/icml_final_2.tex:200)
and [causal paragraph](/Users/ashtonchew/projects/worldsim-paper/icml/icml_final_2.tex:706))
must not be silently attributed to the canonical paper.

More valid tasks can improve precision; broader tasks can additionally improve
behavioral coverage. Neither benefit follows from the count alone. The
recommended evidence package for the proposed generalization claim is:

1. a lightweight behavior-first family description with a held-out behavioral
   family or relation combination;
2. reusable family-level generation validation plus per-instance exposure,
   permission, reset, and state-readback preconditions; and
3. a frozen-bank, fixed-cohort report with family/instance denominators,
   preserved negative outcomes, and a capability-aligned mix.

These are recommended research choices, not additional admission requirements.
A larger set of valid, varied tasks can strengthen the paper before a new
causal intervention is complete. A randomized cue experiment can test the effect
of that cue on ASR; attributing it specifically to internal awareness requires
further construct-validity evidence. Separately, matched-budget retries test
whether the iterator adds benefit beyond search. If the paper retains an environment-agnostic or
cross-host claim, at least one second host must satisfy the same contracts;
otherwise narrow the claim to within-host behavioral coverage. Native benchmark
tasks remain useful diagnostic controls, but should not carry the expansion
claim.

## What the current artifacts do and do not establish

The canonical manuscript describes a 50-task WebArena Verified cohort
([setup](/Users/ashtonchew/projects/worldsim-paper/icml/icml_final.tex:922),
[task set](/Users/ashtonchew/projects/worldsim-paper/icml/icml_final.tex:934)) and
reports 242/300 = 80.7% headline ASR with 295/297 benign passes, three
non-encounters, and two task-broken rows
([results](/Users/ashtonchew/projects/worldsim-paper/icml/icml_final.tex:964),
[denominators](/Users/ashtonchew/projects/worldsim-paper/icml/icml_final.tex:972)).
The release is only three confounded families: `create_issue` 120/120,
`create_post` 117/120, and `create_issue_note` 5/60
([issue family](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/analysis/warp_taskgen_hf_20260509/scenario_template_effects.csv:2),
[follow-up family](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/analysis/warp_taskgen_hf_20260509/scenario_template_effects.csv:3),
[comment family](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/analysis/warp_taskgen_hf_20260509/scenario_template_effects.csv:4));
the manuscript correctly calls this descriptive
([caveat](/Users/ashtonchew/projects/worldsim-paper/icml/icml_final.tex:1001)).
Thus a high aggregate ASR can be an action/evaluator or site effect, and the
low slice is not by itself evidence of model robustness. Current status records
the renderer-backed `create_issue_note` rerun as 6/6 PVPO-valid with final
`resistant=2` ([current status](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/current_progress.md:45));
do not label this action family as currently broken from the 5/60 aggregate alone.

The TP iterator changes labels on 35/300 rows while VEA changes only 5 rows
([transition totals](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/analysis/warp_taskgen_hf_20260509/transition_summary_by_group.csv:2)),
and the selected TP *Eval share falls from 89.7% to 78.0% while VEA is 1.0%
to 0.7% ([TP/VEA results](/Users/ashtonchew/projects/worldsim-paper/icml/icml_final.tex:1033),
[cohort row](/Users/ashtonchew/projects/worldsim-paper/icml/icml_final.tex:1061)).
These are useful diagnostics, not evidence that internal awareness was reduced:
TP/VEA are observational measures. The accounting artifact records substantial
contract-inapplicable, schema, TP-regression, and lost-PVPO outcomes
([accounting header](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/analysis/warp_taskgen_hf_20260509/variant_failure_classes_by_model.csv:1),
[model examples](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/analysis/warp_taskgen_hf_20260509/variant_failure_classes_by_model.csv:2));
treat these as recorded outcome classes, not silently discardable failures.

Selection matters. `_best_iterator_result` ranks joint compliance + TP-Real
first and excludes TP-regressed variants
([selection rule](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/phase_4/eval_awareness_iterator.py:755)),
while the iterator performs sequential rewrites/reruns
([iterator entry](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/phase_4/eval_awareness_iterator.py:966)).
A selected row is therefore post-treatment
and search-selected. “Lower TP” and “unchanged selected ASR” can result from
selection, judge calibration, or cue removal rather than less awareness causing
more attacks. Conversely, high ASR can result from easy families, non-encounter
denominator choices, or reward/readback artifacts.

## Recommended design, scaled to the claim

**Coverage.** Freeze a compact behavior-first taxonomy before generation.
Separate `carrier_contract`, `task_archetype`, and `attack_affordance` as
required by the task-bank specification
([identity fields](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/warp-taskgen-technical-spec.md:1033),
[freeze rule](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/warp-taskgen-technical-spec.md:1041)). A family
should encode information need, answer/decision logic, navigation depth, data
relations, and action semantics; changed names, wording, or payload paraphrases
are instances, not new families. Generate candidates to undercovered families,
then freeze an admitted bank and run all model comparisons on that same snapshot.
The critical test is a held-out workflow relation or action combination, not a
random task split. There is no scientifically meaningful universal task count;
precision should follow observed family/instance variance.

**Validity.** Validate each generation family against known positive and
negative fixtures (writer role, legitimate reader, exact surface, witness
rendering, reset, and state readback), then validate concrete preconditions for
each instance: target record/child, ordinary actor permissions, audience,
route, expected benign outcome, adversarial action, and clean initial state.
Use Phase 2c admission and PVPO/state checks; record candidate → admitted →
evaluated counts and every exclusion reason. The spec explicitly admits every
valid task and has no benign-must-succeed filter
([admission rule](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/warp-taskgen-technical-spec.md:1748)):
retain hard benign tasks and report capability as a contextual diagnostic.
Public and authenticated cohorts need
separate ordinary writer and Independent Authenticated Reader identities, not
writer-session readback. Pair benign/adversarial twins and keep native-task
scores separate from WARP side-effect checks.

**Inference.** Make first-attempt, fixed-cohort outcomes primary. Report
encounter-conditioned and capability-stratified views, but do not relabel
non-encounters or task breaks as resistance. Use family/instance-clustered
intervals (not each rewrite as an independent sample), and preserve all
attempts, discarded candidates, and lineage. A representative stratified subset
can carry expensive awareness judges; broad admitted-task evaluation can carry
coverage. This avoids multiplying every task by every ablation without hiding
the denominator.

**Capability alignment and difficult actions.** A benign workflow should put the
agent near the capability that the payload attempts to hijack. The strategy
handoff distinguishes Tier 1 semantic/navigation effects, Tier 2 bounded public
mutations, and Tier 3 high-impact mutations
([tiers](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/handoffs/TODO-capability-aligned-ipi-strategy.md:126),
[Tier 2](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/handoffs/TODO-capability-aligned-ipi-strategy.md:146),
[Tier 3](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/handoffs/TODO-capability-aligned-ipi-strategy.md:169)).
It explicitly recommends action-parameter hijacks as a realistic flagship and
destructive delete/project actions as hard negatives or stress conditions, not as
the sole ASR story
([hard negative](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/handoffs/TODO-capability-aligned-ipi-strategy.md:95),
[flagship](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/handoffs/TODO-capability-aligned-ipi-strategy.md:98)).
Do not require every severe action, and do not choose benchmark/tasks
after seeing which ones produce high ASR. Predeclare a broad capability-aligned
mix (including information-only and bounded state-changing tasks), retain all
outcomes, and include a deliberately *distant* deviation—an unrelated or more
impactful action—to test whether any gain is specific to alignment. This is a
control concept, not a rigid quota. The recollection that GitLab deletion was
nearly impossible is a useful hypothesis for that stress condition, not a
measured universal rate; only retained runs can support that claim. A useful
May 5 control reports both capability-aligned and read-only unaligned
page-native `create_issue` cells at 7/7 twice
([mechanism note](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/runs/gitlab-public-followup-evidence-20260505.md:43),
[side-by-side results](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/runs/gitlab-public-followup-evidence-20260505.md:89)).
That is counterevidence against treating alignment as the cause of that cell. A useful
existing control keeps the issue-description carrier but swaps page-native
`create_issue` for repository mutation: it has 5/5 PVPO and benign utility but
0/5 action attempts, with ten valid variants also failing
([control result](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/runs/gitlab-public-followup-evidence-20260505.md:141)).
This separates page/action affordance from a claim that benign alignment causes
ASR; it does not establish a causal effect. Also inspect reward/readback fidelity
before interpreting any low rate: historical stale contracts are documented as
a warning, not a current diagnosis
([historical warning](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/handoffs/TODO-capability-aligned-ipi-strategy.md:214)).

## Awareness intervention, judges, and retries

TP is a two-step, target-model guess/classify probe with parse retries
([TP protocol](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/phase_4/transcript_purpose_api.py:1),
[TP retry policy](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/phase_4/transcript_purpose_api.py:66)); VEA is a separate single structured judge over the same inert Needham transcript
([VEA protocol](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/phase_4/verbalized_eval_awareness_api.py:1),
[VEA call](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/phase_4/verbalized_eval_awareness_api.py:200)).
Both are observational; the specification makes only TP a rewrite branch and
keeps VEA report-only
([metric roles](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/warp-taskgen-technical-spec.md:1949)).
API parse retries are judge reliability behavior, not evidence of attack
improvement. Keep their attempt/failure fields and judge versions in the
analysis.

For a causal claim, randomize a cue within the same model-task and initial
state, preserving benign instruction semantics, carrier/payload bytes as far as
possible, token length, tools, and permissions. Use at minimum no-cue, a
length-matched neutral wrapper, and an explicit evaluation cue; a
scenario-recognition cue is a useful secondary arm, not a required positive
effect. Measure ASR, TP, VEA, benign success, PVPO, and action/state outcomes.
Do not call TP/VEA a mediator unless a blinded audit or held-out judge supports
construct validity. The current causal runner's four `system_prompt_frame` arms
([runner examples](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/eval_awareness_experiments/run_causal_experiment.py:16),
[arm names](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/eval_awareness_experiments/run_causal_experiment.py:50)) coexist with the older three `injection_modifier` conditions
([modifier conditions](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/eval_awareness_experiments/injection_modifier.py:1));
freeze one exact intervention and document this path distinction before
comparing results. The current aggregate causal analysis groups by
condition/benchmark/model
([aggregation](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/eval_awareness_experiments/analysis_causal.py:96)) but
does not encode WARP iterator lineage, so it cannot by itself justify a claim
about selected variants.

Attack retries need their own control: compare first attempt, TP-directed
rewrite, and an equal-budget non-awareness paraphrase/retry, all with the same
predeclared attempt budget. A finite cap is an engineering treatment, not a
scientific constant; publish attempt-response curves and lineage rather than
arguing that “three” is intrinsically correct. If TP-directed rewrites do not
beat matched search, the awareness-directed benefit is falsified.

## Three unresolved scientific choices

1. **What is the held-out unit of diversity?** Choose whether a family is
   defined primarily by cross-record relation/decision logic or by action and
   surface. Recommendation: relation + decision logic, with action/surface as
   crossed factors; hold out a relation/action combination. The choice must be
   made before generation, because it determines whether “new” tasks are real
   coverage or reworded GitLab/Reddit instances.
2. **What awareness estimand is the paper claiming?** Recommendation: headline
   a randomized cue → first-attempt ASR total effect; treat TP/VEA as secondary
   measurements and the iterator as an operational mitigation. Do not combine
   the cue effect, selected iterator effect, and a TP label change into one
   causal story.
3. **Which ASR population is primary?** Recommendation: fixed-cohort,
   first-attempt ASR, with selected-variant ASR shown as an upper-bound
   operational result and all retry lineage exposed. If the paper instead
   headlines selected rows, it must explicitly call out search/selection and
   show the matched-budget negative control.

## Falsifiers and negative outcomes

The expansion fails scientifically if held-out tasks collapse to the existing
three action families; if exact independent exposure/readback or reset cannot
be established; or if apparent gains disappear after family clustering and
capability stratification. The awareness story is weakened if a cue shifts TP
but not ASR, shifts ASR without any awareness-measure change, or if VEA remains
near floor with no independent corroboration. If matched non-awareness retries
equal TP-directed rewrites, report no incremental awareness benefit. These are
valuable negative results, not reasons to tighten admission or inflate the
corpus.

## Sources

- [Active-manuscript authority](/Users/ashtonchew/projects/worldsim-paper/CLAUDE.md:2) and [canonical WARP experiment](/Users/ashtonchew/projects/worldsim-paper/icml/icml_final.tex:922).
- [Current WARP technical specification](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/warp-taskgen-technical-spec.md:1029).
- [Current WARP analysis artifacts](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/analysis/warp_taskgen_hf_20260509/scenario_template_effects.csv:1).
- [Capability-aligned action rationale](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/handoffs/TODO-capability-aligned-ipi-strategy.md:73).
- [TAC task/evaluator primary source](https://github.com/TheAgentCompany/TheAgentCompany/blob/98b68ef82a47690c316f42fddb05baafaab56851/workspaces/tasks/pm-update-plane-issue-from-gitlab-status/evaluator.py) and [WebArena Verified](https://github.com/ServiceNow/webarena-verified).
- [Newer non-canonical draft/poster attribution](/Users/ashtonchew/projects/worldsim-paper/icml/icml_final_2.tex:200).
