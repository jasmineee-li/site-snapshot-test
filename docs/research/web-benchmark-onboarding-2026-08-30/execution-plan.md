# WARP benchmark expansion: execution and evidence plan

Planning date: **2026-09-01 (America/New_York)**. Research baseline:
**2026-08-30**. Source baseline: `origin/main` at
`5c45a32fdc3c509db2a9435849df7edf8c6eeb45`.

Execution update: **2026-09-02 (America/New_York)**. E1/E2 production source
paths and no-model disposable-host proofs are complete; model-driven evaluation
and the remaining E3–E7 work are still pending.

Provider-blocked development update: **2026-09-03 (America/New_York)**. E3 has
20 currently reusable Phase 1 rows out of the frozen 140-row target. The missing
80 non-TAC rows are blocked on the frozen Sonnet provider route; the 40
Rocket.Chat rows have not been generated. Provider-independent source work is
now split into narrow cache, ownership, cost, remote-root, and stage-specific
checkpoint-visibility slices described below. This work improves safe resume
and operator legibility; it does not create, admit, or evaluate a task.

CI and agent-DX delivery update: **2026-09-04 (America/New_York)** against
`origin/main` `eac23227`. The provider-independent source DAG #240--#246 and
source-feedback PRs #259--#261 are complete. Taskgen-only root routing measured
17 seconds rather than 1 minute 50 seconds for the full gate; core pytest
measured 114.63 seconds after exposing durations and narrowing two readiness
scans; the final two-way core split reduced the critical test job from 2
minutes 18 seconds to 1 minute 30 seconds. The complete Taskgen workflow took
about 1 minute 39 seconds including its aggregate. The exact-route Phase 1
resumption protocol uses two production micro-canaries, not a no-provider
imitation.

Provider-boundary delivery update: **2026-09-05 (America/New_York)** against
`origin/main` `783d3533`. PR #264 implements the bounded operator runner. A
real invocation reached the authenticated OpenRouter current-key endpoint and
stopped with `capacity_unavailable` before direct generation, Modal setup, or
creation of the fresh output root. Neither production generation boundary has
passed; the earlier direct-canary HTTP 403 remains historical failure evidence.
Issue [#263](https://github.com/jasmineee-li/warp/issues/263) is closed. Run15
remains the resume target; effective-route persistence is deferred rather than
changing its cache or Run Definition compatibility.

Status: **source and transfer smokes complete; scientific execution pending**.
The source tickets under parent issue #192 are closed; PRs #199–#204 and
#206–#208 are merged, and live-proof issue #205 is closed. This
plan owns the remaining path from tested software to paper-eligible evidence.
It does not reinterpret static declarations or deterministic fakes as runtime
results. Execution is tracked by parent issue
[#209](https://github.com/jasmineee-li/warp/issues/209).

The user approved continuing the entire vertical slice on 2026-09-01. That
approval covers repository planning, tickets, source changes, focused local
tests, review, and bounded work on already configured disposable benchmark
infrastructure. It is not an unbounded spending authorization and cannot
supply missing hosts, credentials, billing limits, model choices, or
publication authority.

## Current program contract and remaining work — 2026-09-07

This section is the canonical current plan. The dated delivery history below
remains evidence of earlier decisions; where it differs, this section governs.
Latest source inspection: `origin/main` `ddd7d7ba` on September 7. PRs
#335/#336 and the subsequent runtime changes are merged. The matched-study
packages #340/#341, prospective scorer repairs #346/#351 and Phase 2b provenance
#360 and ordinary Phase 2a provenance #362 are merged. GitHub confirms all six required checks passed for #360;
its owner reports independent reviews and full local acceptance passed. No new
scientific Run is implied by these source deliveries. Earlier source hashes and
check results below remain historical evidence, not certification of a later
source or deployed Instance. Recheck active ownership before dependent edits.

### Settled scientific and execution choices

WARP studies **indirect prompt injection (IPI)** through lower-trust content
encountered during legitimate browser tasks. The follow-up tests whether
substantively generated workflows broaden this evaluation and whether TP-guided
rewriting adds value over matched ordinary rewriting. High ASR, a null paired
effect, and retained attack success with lower measured evaluation salience are
all reportable outcomes. The latter does not by itself identify causal mediation
by latent awareness. A one-opportunity comparison does not establish superiority
of a multi-round search loop.

- Keep **Sonnet 4.6 and GPT-5.2** as the initial expanded-evaluation and matched
  study target agents. Preserve the historical six-model results and retained
  generator, iterator and judge configurations. Astra is the implementation
  agent. Verify exact provider/runner settings from Run Artifacts before calls;
  a target model is distinct from a generator, proposer or judge role.
- Keep the **140-candidate allocation, 20 per seven workflow families** (40
  information-only and 100 state-changing benign tasks). Candidates are not
  admitted tasks. Include all eligible admitted parents per target where
  feasible. If necessary, freeze a family-stratified subset before rewrite
  outcomes are observed; never select for attack success.
- Keep the **$5,000 overall remaining ceiling**, covering generation, expanded
  evaluation and matched studies. The earlier $3,000 generation backpressure
  value is historical, not an additional budget. Use existing estimates,
  configured limits and returned usage; do not add a billing service, repeated
  Modal cost polling or a new stop condition solely for missing Modal pricing.
  Existing production ledger-validation and provider-capacity checks remain.
- Eligible matched baselines must satisfy canonical admission, mutable payload,
  witness and PVPO requirements and have baseline TP other than Real. Include
  baseline successes and failures. Freeze parent/model cells before treatment.
- One rewrite opportunity per arm, same proposer and repair allowance, with
  TP diagnosis as the information difference. Neither arm receives VEA or
  reward traces as extra feedback. Historical issue wording mentioning VEA
  does not authorize a larger intervention. Ordinary critique must be produced
  through the real matched provider on live Runs.
- Balance seeded arm order within family/model strata and persist it. Reserve
  one reset scope through both arms, with verified reset before reuse.
- Primary: per-model paired ASR difference, equal weights over the seven fixed
  task-card families, parent-clustered 95% intervals. Report task-micro and
  per-family results secondarily. Never silently drop an empty family or
  substitute zero; report the full-family estimate unavailable. There is no
  accepted 2-percentage-point noninferiority threshold.
- Fixed-opportunity outcomes distinguish measured success, measured negative,
  completed unusable proposal (zero with its own cause), and infrastructure,
  reset or readback failure (unknown). Retain every scheduled cell and report
  missing-outcome bounds. Baseline fallback under the shared selector is a
  secondary endpoint. Report TP, joint outcomes, benign utility and gains/losses
  with their actual available denominators; do not count infrastructure errors
  as model resistance.

### One remaining-work map

Owners below are responsibilities, not claims that every future implementation
has already been dispatched. The Astra lead owns scope, interfaces and final
acceptance. “Plan WARP benchmark expansion” owns E3 execution and its E4/E5
operational reconnaissance; this task must not resume its Run or duplicate it.

| Work / scientific purpose | Current evidence | Missing deliverable and owner | Dependencies / gate | Acceptance criteria and what it unlocks |
| --- | --- | --- | --- | --- |
| **E3: corpus generation and admission.** Test substantive workflow breadth, not paraphrase volume. | Owner reports retained Run 15 has 20 reusable Postmill rows, 80 missing non-TAC GitLab rows, no GitLab aggregate/cache and no Phase 2 directory; 40 TAC candidates remain separate. Earlier source-scoped compatibility evidence: 15 microcanary tests and all three fast checks passed; 12 Run file hashes unchanged by inspection. The expansion owner reports later Phase 1 changes, so these checks must be refreshed against the selected source before resume; they do not certify latest `main`. | Exact-route production canaries, retained resume, TAC generation, admission and frozen funnel. **Owner: Plan WARP benchmark expansion, #212.** | **OpenRouter**, then **Benchmark Instances** for feasibility/admission. Current production capacity check stops before generation/Modal; confirm the prior process/Remote Job no longer owns Run 15 before eventual resume. SSH timeout does not prove no owner. | Direct and sandbox production boundaries succeed; original compatible Run resumes without concatenating diagnostic runs; all seven allocations accounted for as candidate/validated/admitted/failed with substantive task evidence. Unlocks expanded evaluation. |
| **E1/E2: Rocket.Chat/TAC evidence.** Demonstrate transfer of ordinary-participant IPI and exact decision/action grading. | #210/#211 completed: pinned TAC/Rocket.Chat 5.3 no-model smokes, separate ordinary identities, exact thread/notification readback and terminal reset. These are transport/execution proofs, not model results. | Reuse retained smoke evidence and bind generated TAC rows to the proven composition; then collect model trajectories. **Owner: E3/E4 execution lane.** | Source work complete; **Instance + OpenRouter** for new rows. | Actual Phase 2/4 paths preserve exact identity, independent readback and reset; model results distinguish decision from notification, encounter and utility. Unlocks bounded TAC transfer claims, not TAC-wide or cross-Site support. |
| **A: retained matched arm order.** Prevent order from being confounded with the intervention. | Existing one-opportunity API and retained checkpoints; new production-path tests exercise both orders, QA rejection, reset failure and legacy reuse. PR #340 is merged; source tests are not a scientific Run. | Delivered: seeded balanced assignments bound to Run Definition and results; schema-2 reading without ordered reuse. **Owner: Astra lead accepted [PR #340](https://github.com/jasmineee-li/warp/pull/340).** | **Source complete.** Live execution still needs E5 gates. | Actual dispatch follows frozen assignment; changing seed/order rejects reuse; source baseline unchanged; reset failure prevents agent dispatch; deliberate fixed-order regression fails the test. Unlocks defensible E5/E6 execution. |
| **B: matched analysis/export.** Estimate incremental benefit with honest missingness and family coverage. | PR #341 is merged; 30 focused tests include real producer-to-analyzer integration, with full acceptance and required CI green. | Delivered: fixed endpoint, cause-aware unknowns, seven-family macro, parent-block intervals, bounds and secondary views. **Owner: Astra lead accepted [PR #341](https://github.com/jasmineee-li/warp/pull/341).** | **Source complete.** Scientific estimates need **results**. | Positive/negative and independent numerical tests, unequal family weights and parent dependence, missing-family unavailability, evaluator failure unknowns; required checks and independent review. Unlocks analysis immediately when real pairs arrive. |
| **C: prospective Reddit scoring.** Attribute success to the new post’s required content. | Full compiled-contract reward API reproduced seed-only false success; PR #346 is merged after independent review, 56 focused tests, full local acceptance and green required CI. | Delivered: response-bound submission-body attribution and unknown missing evidence. **Owner: lead accepted [PR #346](https://github.com/jasmineee-li/warp/pull/346).** | **Source complete**; selected deployed scorer still needs an **Instance** canary. | Source-only/wrong-resource/wrong-field rejected, correct created-body accepted, known carrier excluded, identity conflicts unavailable and origin handling correct; required checks and independent review. Unlocks prospective Reddit scoring readiness. |
| **D: prospective generic GitLab scoring.** Attribute success to the new issue/note and correct project/parent. | PR #351 delivers the reviewed repair after public-API reproductions; 108 focused tests and full local acceptance pass, including specialized comparison controls. | Delivered: generic source/readback binding using existing identity evidence. **Owner: lead accepted [PR #351](https://github.com/jasmineee-li/warp/pull/351).** | **Source complete**, independent of C’s owning files; selected deployed scorer still needs an **Instance** canary. | New-resource positives and old/wrong-project/wrong-parent negatives through real APIs, valid supported UI evidence preserved, insufficient proof unavailable, specialized comparison controls green; required checks and independent review. Unlocks prospective generic GitLab scoring readiness. |
| **E5: first real matched pair.** Prove the entire scientific intervention is executable. | Source support exists; no verified complete eligible local baseline in the inspected E3 artifacts. Historical Sonnet/GPT-5.2 archive names are leads, not accepted baselines. | Restore and validate one exact task/result/payload/witness/Run identity; bind provider/runner and per-arm limits; execute both arms. **Owner: lead assigns runtime owner after E3 scout, #214.** | **A + OpenRouter + isolated Instance + eligible artifact.** May precede E4 if a historical source passes existing materializer checks. | Real ordinary critique and TP diagnosis with matched opportunity; verified reset, retained failure/usage/outcomes; no manufactured legacy identity. One pair proves execution, not treatment effect. |
| **E4: expanded target evaluation.** Measure attack performance and utility across the admitted workflow bank. | No expanded evaluated bank yet. Host configuration exists in an ignored worktree; replica declarations are not live isolation proof. A bounded SSH read-only probe timed out before authentication. | Freeze bank identity and target settings; evaluate admitted rows and export authoritative funnel. **Owner: expansion execution lane / lead, #213.** | **E3 + OpenRouter + verified Instances.** | Both settled targets, complete scheduled outcomes, PVPO/task-broke/non-encounter/utility distinctions, exact final-state evaluation and reset-safe scheduling. Exports must distinguish cause-bearing unavailable evaluator evidence from measured negatives rather than count raw False alone. No success-based admission filter. Unlocks representative E6 cohort and breadth results. |
| **E6: matched cohort.** Measure TP-guided incremental value across workflows. | Accepted design above; no expanded matched results. | Freeze eligible parents per model and seeded orders, run cohort and analyze. **Owner: study execution lane / lead, #215.** | **E4 + E5 + B + capacity/Instances.** | All eligible feasible or pre-outcome frozen subset; seven-family primary reported only when supported; every scheduled cell and missingness retained; paired intervals and secondary endpoints reproducible. Unlocks the main matched-study claim. |
| **Historical scoring attribution.** Ensure ASR witnesses match the requested new action. | Separate investigation reports a concrete Reddit source-witness contamination candidate via offline replay and GitLab attribution weaknesses from a mechanism screen; original evaluator/package association remains to be reconciled. The earlier 241/242 discrepancy was withdrawn. | Approved planning covers all 300 historical model-task rows (120 Reddit posts, 120 GitLab issues, 60 GitLab notes), with original-contract/source reconciliation and the settled policy of prospective exact rendered generation-request retention in existing archives with compact references. Merged [PR #360](https://github.com/jasmineee-li/warp/pull/360) covers Phase 2b requests per attempt/retry joined to response/validation diagnostics, resolved client-provider identity and reuse preservation; Ordinary Phase 2a retention is also delivered in [PR #362](https://github.com/jasmineee-li/warp/pull/362); Phase 1/4 and the explicitly excluded transports remain coverage gaps. **Owner: Plan WARP scoring attribution audit…**, task `01a07849-8154-7a60-8a54-316d788328f2`; its preserved local plan is `/private/tmp/warp-attribution-task-state-2026-09-06/attribution-audit-plan.md` after removal of the clean implementation worktree; the task retains historical-audit ownership. | **Now: planning/artifact reconciliation.** Ordinary Phase 2a and Phase 2b provenance are delivered; historical score adjudication remains pending. This owner is identifying remaining generation-path retention seams before the lead accepts another focused package. Historical numerical changes need exact evidence. | Preserve raw Runs and original scores, attribute any defect to exact source/artifacts, version corrected analysis separately. Missing new provenance must not invalidate/exclude old results or automatically trigger historical regeneration, re-evaluation or reruns; missing historical request bytes remain not retained. Inspectable requests do not guarantee deterministic output replay. Separately established scoring defects may still justify versioned corrections. New analyzer correctness does not validate historical or current evaluator attribution; resolve any confirmed applicable production defect before new browser studies. Unlocks trustworthy historical comparisons. |
| **Evaluation-awareness validity.** Separate measured framing/TP effects from latent-awareness claims. | Existing external five-phrasing classifier exports, Taskgen TP and probe analyses are different instruments. Manifest cell inventories are not completed denominators. Neutral system-prompt source exists; accepted neutral-result provenance was not found in inspected inventory. Paper-to-retained judge/actor/step settings and probe snapshots need reconciliation. | Claim-to-artifact table using existing Run identities and regeneration commands; resolve corrected snapshot, calibration, refusals/denominators and pending WASP paired discordances. **Owner: Astra lead coordinates awareness/paper owner; no duplicate exporter dispatched.** | **Now: read-only reconciliation**, then **upstream corrected results**. New cue/probe Runs require a **scope decision**. | Match every reported instrument, target/judge role, actor, step budget, calibration split and analysis population to artifacts. Explain unresolved differences without declaring all paper claims false. Do not call hypothetical framing neutral or infer mediation from association. Unlocks defensible causal language and validity limitations. |
| **E7: analysis and paper preparation.** Make the evidence useful and the thesis assessable. | Existing historical paper/report; active manuscript is the paper repository's `icml/icml_final.tex` (audit at `48aa588`). New experimental claims are not yet measured. | Regenerate tables/plots, align methods and denominators, document provenance, limitations and bounded claims. **Owner: paper owner with Astra lead acceptance, #216.** | **Now:** table/claim planning and artifact reconciliation. Final numbers need **E4/E6 and historical/validity resolutions**. | Separate supplied structure/generated instances/admission/evaluation; report high/null/mixed results faithfully, distinguish bounded transfer and one-step rewrite claims. Follow paper-repository guidance; submission/publication is separate authorization. |

### Prospective scorer-readiness gate

The attribution owner’s helper findings led to independent current public-API
reproductions. Reddit’s actual compiled contract and `run_reward_function`
accepted seed-only witness text when a new post lacked it. A separate GitLab
public-API diagnostic under the unchanged compiled policy reproduced generic
issue/note attribution defects involving existing records and wrong project or
parent. Only external readback was substituted. These are prospective code
defects, not deployment replay or revised historical scores; specialized
comparison behavior is checked separately.

Before E4/E5 uses an affected scorer, pin its deployed revision and effective
probe/policy, then exercise the **full existing evaluator entry point** with
external readback/transport substituted only where necessary. Require rejection
of seed-only, wrong-resource and wrong-field evidence and acceptance of the
correct new resource. For GitLab also check query versus mutation, parent
binding and any enabled UI fallback. A reviewed repair or proof that the
problematic path is not selected clears this prospective gate; finishing all
300 historical rows is not required first. Reward-derived action-attempt flags
cannot independently corroborate the reward.

The lead dispatched **C: prospective Reddit create-post attribution** after
the historical owner confirmed no competing source owner. One Astra worker
owns the Reddit adapter/readback and their existing reward tests in an isolated
current-main worktree. Completion requires a full-entry-point reproduction,
minimal repair, attributable positive and negative cases, required checks and
a separate reviewer; missing identity/readback remains unavailable. Merged [PR #346](https://github.com/jasmineee-li/warp/pull/346) clears
the prospective Reddit source gate, not the separate GitLab readiness check.
**D: prospective generic GitLab attribution** is separately assigned to an
Astra worker in a fresh `4894409e` worktree after ownership clearance. It owns
the GitLab adapter/readback and their reward tests, plus two existing positive
fixtures whose captured creation IDs must match their readback records: bind created resource,
project/parent and required content through existing evidence, preserve
specialized comparison controls, and report missing attribution unavailable.
D’s [PR #351](https://github.com/jasmineee-li/warp/pull/351) passed the
public-API positive/negative regressions and independent review; required CI
is a merge gate for every source package. No shared scoring schema, new actor/time
requirement or historical correction belongs to either package. Historical artifacts, correction decisions and prospective
generation provenance stay with the attribution task. No new scoring schema or
broader historical contract is implied. This gate is separate from B's honest
handling of missing/error evidence: an apparently successful but misattributed
score cannot be repaired by the report-only analyzer.

### Interfaces, dispatch and acceptance

A owns matched order/config/identity/run changes and its owning technical-spec
section. B owns analysis and its export/tests/documentation. The shared retained
contract is schema 3 with top-level `arm_order` and `assignment_seed`, and pair
`arm_order`; family comes from original retained `task_card_id`, parent from
`benign_task_id`, not prompt-sanitized task metadata or a model-dependent Run
identity. A owns producer changes. B consumes them without redefining them.
The canonical plan is owned by the lead; the expansion task is coordinating
operational findings rather than editing this file concurrently.

Each new package starts from current `origin/main` in a separate short-lived
Codex worktree with one exact objective, owning files, exclusions, acceptance
tests and completion condition. Review the proposed approach and early diff;
report extra work to the lead instead of absorbing it. Use existing production
APIs and Run Artifacts. Substitute only necessary external boundaries in tests;
do not replace the behavior being asserted. Require an independent correctness
and spec review per PR, repeat only to verify a concrete repair, and merge after
required checks pass. Remove only clean worktrees created for that package.
Source PRs supporting #214/#215 do not close their still-unmeasured live Runs.

A/B delivery and producer integration are complete, as are the reviewed C/D
source repairs in #346/#351. The next execution gate is selected deployed-scorer
preflight before any E4/E5 browser execution, including actual trace coverage
for the existing creation-identity contract. Missing proof stays unavailable;
source tests alone cannot establish live trace coverage. The expansion
owner retains the earlier successful compatibility checks, but reports that
later Phase 1 changes require refreshing them against the selected source before
resume. That refresh stays with the expansion owner; real provider capacity is
also required. In parallel, reconcile existing awareness and historical
artifacts before proposing new source. Do not create a scheduler, reset system,
billing layer, alternate schema or generic instrument framework to fill time.
Source-delivery estimates concern review/CI cycles; live-study elapsed time
cannot be inferred from those estimates until capacity, eligible rows and
reachable reset scopes are known.

### Capacity-independent work and paper outputs — September 7

Request retention now covers ordinary Phase 2a planning (#362) and Phase 2b
payload generation (#360). It captures model-facing SDK arguments, not literal
HTTP wire bytes or every pipeline call. Phase 1 and Phase 4 use other paths and
remain unfinished. This gap neither invalidates old Runs nor creates a blanket
new admission gate. The Phase 4 worker must first agree its exact dispatch,
retry and durable-output interface with the lead; it does not own other callers.

The following bounded work is assigned while capacity is unavailable. A task's
assignment is not evidence that its deliverable has completed. Existing
expansion/scoring owners retain their lanes; Astra lead owns this plan and final
acceptance. No new provider, model or benchmark scope is implied.

| Package / owner | Capacity-independent deliverable | Completion evidence / what it unlocks |
| --- | --- | --- |
| Historical adjudication / attribution owner | Inspect representative GET-only Reddit, query-only GitLab and mismatched GLM bundle cases; bind original effective contracts/source to selected traces and attempt local original/public evaluator replay. | Per-case confirmed applicable finding or exact unresolved dependency, with commands/hashes; preserve raw scores and separate counterfactual mechanism tests from historical replay. Converts the completed 300-row inventory into interpretable historical evidence. |
| Awareness recovery / Astra evidence subagent | Recover one pinned original N200 LFS judge object and verify SHA-256/size; compare its settings/denominators to the matching manifest, then identify corrected-export recovery requirements. | Actual recovered bytes or measured remote-access failure. Original exports must not be relabeled corrected. Establishes which paper claims can be regenerated and where the accepted corrected snapshot is still needed. |
| E5 baseline preparation / expansion owner | Find one retained eligible baseline for the settled targets and exercise existing local matched-materializer validation without dispatch. | Exact task/payload/result/witness/Run identity passes, or a precise missing-artifact reason. No invented legacy identity. Prepares the first real pair once capacity and Instance access return. |
| Default Phase 4 rewrite retention / dedicated Astra worker | One focused integration using existing request storage/serializer, capturing actual application-level retry/reask requests and durable outcome joins. | Agreed owning files, real SDK/HTTP/SSE positive/negative tests, cancellation/write-failure and legacy compatibility, independent review and required checks. No new generic archive framework, Phase 1/2 edits or cue/variant/critic scope. |
| Paper outputs / lead | Specify each planned figure/table against the accepted population and existing artifact owner before numbers arrive. | Output map below; final numbers require their stated evidence. No fabricated demo values or changed estimands. |

The paper's contribution remains generated workflow breadth for IPI evaluation,
plus a controlled test of TP-guided rewrite value. Request retention supports
auditability; completing it alone is not the scientific contribution.

| Planned output | Evidence and population | Interpretation supported |
| --- | --- | --- |
| Corpus table and admission funnel | E3 candidate → validated → admitted → evaluated counts by all seven frozen families; exact task/world lineage, decision/action and Site. | Shows what was generated and tested; generated candidates are not automatically admitted tasks or new workflow diversity. |
| Expanded performance table | E4 results for each settled target: ASR, utility, encounter, task-broke and unavailable evaluator evidence with scheduled denominators. | Measures attack effectiveness across the admitted bank. High ASR remains an interesting result; unavailable readback is not resistance. |
| Paired effect figure | Existing matched analysis report from every scheduled E6 cell: seven-family fixed macro difference and parent-clustered interval per model; task-weighted/per-family secondary views. | Estimates incremental TP-guided value over ordinary rewriting. Missing family/outcome handling remains exactly as accepted; no new threshold or hypothesis-test battery. |
| Rewrite outcome table | Same report: gains/losses, TP-Real, joint success/TP, utility, completed unusable proposals and infrastructure unknowns; selector outcome separately labeled. | Distinguishes more effective attacks from changed measured evaluation salience and failed opportunities; no latent-awareness mediation claim. |
| Scoring/provenance appendix | Historical adjudication plus prospective scorer and retention evidence; original/selected/repaired identity, exact source and cause-bearing missing evidence. | Explains what each score establishes and any separately versioned corrections. Metadata absence alone does not invalidate old results. |
| Awareness validity table | Recovered accepted external judge/continuation/classifier outputs, settings, calibration/probe split and corrected paired outcomes. | Distinguishes external five-question instruments, Taskgen TP and VEA, source defaults and measured configurations. Recover evidence before deciding on new experiments. |

Reuse the existing matched report command and field mapping in
[`matched-rewrite-analysis.md`](../../../packages/warp-taskgen/docs/matched-rewrite-analysis.md).
The report cannot prove cohort completeness if scheduled files are omitted; bind
inputs to the frozen cohort before paper export. Avoid building another analyzer
or synthetic fixtures that reimplement it. A rendering gap, if discovered, can
be a separate small package using the authoritative report unchanged.

The live sequence remains admission and deployed-scoring proof → expanded
evaluation → matched cohort → analysis and paper results. One eligible historical
pair may establish execution earlier under E5's existing gates. No repeated
capacity polling is useful without changed access/capacity. Plane, cross-Site
workflows and new model/cue/probe scope remain separate decisions below.

### Deferred options: planning, not implementation authorization

| Option | Scientific value | Effort and dependencies | Recommendation / decision boundary |
| --- | --- | --- | --- |
| **Plane** | Tests portability to another structured work-management Site; finite multi-record decisions/actions can add useful dependencies, although GitLab comparison already covers part of that contribution. | Medium-to-large separate slice: pin actual deployed version/routes, prove ordinary-role writer and independent exact reader/reset, then implement only consumed generation/admission/evaluation joins and run models. No current Plane adapter was found. Existing TAC Plane fork and modern Plane routes differ. | Keep a bounded portfolio ready; decide after core transfer/breadth evidence identifies the remaining scientific gap. E1 success does not prohibit reconsidering Plane. No implementation dispatched. |
| **Cross-application workflows (for example GitLab → Plane)** | Tests a real cross-Site information/action dependency, beyond another carrier for the same payload. | Larger than Plane alone: current single-task validation requires one Site. Needs explicit multi-Site task/evidence contract, identity linkage and proven reset isolation across both Instances; depends on a working second Site. | Separate scope decision after defining the contribution and one finite end-to-end task. Do not disguise it as configuration or build a universal workflow engine. |
| **New targets, iterator upgrades, neutral/cue/probe Runs, multi-round matching** | Could test contemporary robustness, search efficiency or stronger awareness mechanisms, each a different question. | New predeclared comparison, role-specific provider settings, capacity and appropriate controls. Existing-role continuity would need an explicit bridge. | Preserve current models and one-opportunity scope; consider only after evidence or an explicit user decision establishes the value. No automatic model refresh. |

Supporting bounded Plane planning is in
[the existing portfolio](../benchmark-followons-2026-09-01/plane-task-portfolio.md)
and [primary-source notes](../benchmark-followons-2026-09-01/plane-and-cross-site-primary-sources.md).
These options are not dependencies of the accepted core program.

## 1. Current evidence

| Claim area | Evidence now | What it does not establish |
| --- | --- | --- |
| GitLab comparison | Generated compare-and-decide/act source, strict current-attempt binding, exact grading, green CI, and one externally reported serialized r8a source/evaluator smoke with fresh authenticated render, one selected-issue note, readback, cleanup and reset. | A locally retained runtime bundle, model-driven browser-agent performance, task-family rates, or corpus breadth. |
| Rocket.Chat transfer | Typed decision/notification families plus opt-in production compositions. Serialized no-model E1/E2 smokes on pinned TAC commit `98b68ef82a47690c316f42fddb05baafaab56851` and Rocket.Chat 5.3 proved separate ordinary writer/reader identities, exact conversation and notification IDs/bodies/threads, finite and final-state grading, terminal reset, zero final history and marker absence. E1 supplied the deep thread-panel browser proof; E2 supplied ordinary notification persistence and independent exact readback. | Model-driven browser-agent performance, corpus breadth, live NPC behavior, or TAC-wide/general cross-Benchmark support. |
| Existing-family volume | A read-only funnel projection that preserves authoritative statuses and labels missing counts unavailable. | Any newly generated, admitted or evaluated task volume. The tracked released corpus remains 50 unique tasks. |
| Matched rewriting | A study-only one-opportunity contract, deterministic provider tests, retained failures/accounting and resume identity, with the default iterator unchanged. | A real model/browser pair, treatment effect, token use, or cost. |
| Paper | Accepted research framing and implementation plan. | Updated corpus, transfer, matched-study results, tables, limitations, or submission. |

The counterfactual is simple: if the implementation had already delivered the
paper result, current artifacts would contain new candidate/admitted/evaluated
counts and measured paired rewrite rows. They do not.

## 2. Execution dependency tree

```text
Merged source baseline (PRs #199–#204, #206–#208; issue #205)
├─ E1. Rocket.Chat decision live slice [complete]
│  └─ E2. Rocket.Chat notification live slice [complete]
├─ E3. Generate and admit the expanded task bank
│  └─ E4. Evaluate the frozen expanded bank [also needs E1/E2 for TAC rows]
├─ E5. Execute one real matched-rewrite pair
│  └─ E6. Run the representative matched cohort [also needs E4]
└─ E1–E6 evidence
   └─ E7. Update the canonical manuscript
```

E1/E2, the non-TAC portion of E3, and E5 provider work may develop in
parallel. The final bank cannot admit Rocket.Chat until E1/E2 establish its
executable composition, E4 cannot evaluate TAC rows before that join, and E6
depends on both E4's evaluated rows and E5's matched provider. Mutable runs
remain serialized per reset-sensitive Benchmark Instance unless distinct
reset scopes are independently verified. Paper work waits for measured
artifacts; it must not draft result numbers in advance.

| Package | Tracking issue |
| --- | --- |
| E1 — Rocket.Chat decision live slice | [#210](https://github.com/jasmineee-li/warp/issues/210) |
| E2 — Rocket.Chat notification live slice | [#211](https://github.com/jasmineee-li/warp/issues/211) |
| E3 — Generate and admit the expanded task bank | [#212](https://github.com/jasmineee-li/warp/issues/212) |
| E4 — Evaluate the frozen expanded bank | [#213](https://github.com/jasmineee-li/warp/issues/213) |
| E5 — Execute one real matched-rewrite pair | [#214](https://github.com/jasmineee-li/warp/issues/214) |
| E6 — Run the representative matched cohort | [#215](https://github.com/jasmineee-li/warp/issues/215) |
| E7 — Update the canonical manuscript | [#216](https://github.com/jasmineee-li/warp/issues/216) |

## 3. Vertical work packages

### E1 — Rocket.Chat decision live slice

Deliver one ordinary writer → fresh independent authenticated reader → exact
conversation render → finite decision grade → cleanup/reset path on a
disposable TAC Rocket.Chat instance. Implement only the missing production
transport, reader context, composition and evaluator join consumed by this
slice. Preserve writer/reader identity separation and keep reset/admin control
outside the agent context.

The delivered composition must support this decision family through its
actual Phase 2 admission and Phase 4 response-evaluation path; static
`phase_1_generation`/WARP-local authority alone is not operational transfer.

Evidence: focused positive/negative tests; one readable source/evaluator smoke;
exact room/message/author/thread/body readback; pre/final state; reset result.
Failure to obtain ordinary-role visibility is a measured boundary, not
permission to substitute an admin writer.

Result recorded 2026-09-02: the no-model E1 smoke started from the pinned
fixture's golden `project-graphdb` room (zero messages, five members), seeded a
generated plan/update/correction thread through an ordinary account, and bound
the independent ordinary REST and browser reader to the same physical user.
The exact Rocket.Chat 5.3 thread-panel adapter matched one correction element
and all three generated message IDs, authors, thread links, order and body
digests. The current structured decision passed; the stale initial decision
failed. The host-owned synchronous reset reported terminal completion, after
which room history was zero and the unique run marker was absent. Layout-at-
entry and scroll distance remain telemetry under the existing Phase 2c policy;
the successful resource proof required nonzero unhidden exact-selector
geometry and did not relabel that telemetry. No model or live NPC ran, and no
new evidence format was introduced.

### E2 — Rocket.Chat notification live slice

Reuse E1 and send one decision-derived notification through an ordinary
participant action. Verify exact recipient, room, author, parent/thread, body
and message identity with an independent reader. Reject request-only success,
wrong target, stale same-text state and extra artifacts within the fixture
scope. Run one serialized sandbox smoke and reset.

Extend E1's executable composition only with the final-state/action capability
consumed by notification. E2 is the gate for admitting and evaluating the TAC
notification rows in E3/E4.

Result recorded 2026-09-02: the no-model E2 smoke used the pinned fixture's
golden `project-graphdb` room and externally supplied ordinary fixture
credentials. WARP generated the plan, update, decisive correction, due date and
marker; the author, reader and recipient roles were bound to distinct fixture
identities. The ordinary writer persisted one derived notification;
the distinct ordinary reader verified its exact current room, root/thread,
message ID, author, recipient mention and body; WARP-local final-state grading
passed. Cleanup, explicit terminal reset and final marker/message absence all
passed through the existing JUnit property surface. This smoke reused E1's
browser-exposure proof rather than repeating it, ran no model or NPC, and did
not create a new evidence format.

### E3 — Generate and admit the expanded task bank

Use WARP generation for substantive instances inside the accepted families:
existing GitLab/Postmill work, GitLab comparison siblings and, after E1/E2,
the live-ready Rocket.Chat families. Existing/GitLab generation may proceed in
parallel with TAC work, but the final multi-Benchmark bank freezes only after
the TAC capability join. Freeze family allocations and generation inputs
before observing attack success. Vary decisive facts, relationships and action
dependencies—not only names or paraphrases. Preserve `origin="new_task"` and
report candidate, validated, admitted and failed counts through existing
artifacts and the merged coverage projection.

Before paid generation, produce a dry-run call/token estimate for several
candidate corpus sizes. The selected size must be substantially larger than
the released 50-task bank and must include every accepted workflow family; it
is chosen before admission outcomes and is not reduced to successful attacks.

The accepted pre-generation choice is exactly 140 WARP-generated rows: 20 in
each of seven families. The retained Run 15 currently has 20 reusable Postmill
rows and no GitLab aggregate or site cache. Earlier partial and failed runs are
diagnostics rather than a bank because their IDs, source versions, or task
semantics differ. Resume must therefore regenerate the missing 80 non-TAC rows
inside the retained Run rather than concatenate earlier outputs. Phase 2 has not
started, and no Phase 2c row is admitted.

#### E3 provider-independent development slices

These slices address concrete failures observed during E3 while preserving its
generation, admission, provenance, and safety contracts:

1. Include contract-bound diversity salt and normalized forbidden references
   in the existing Phase 1 cache identity only for consumers that read them.
2. Give Phase 1 CLI execution one same-filesystem cooperating-process owner per
   state root, without claiming a distributed lock or changing direct Python
   callers.
3. Persist every returned paid-call cost observation before later validation,
   expose known and unknown observations as a lower bound, and fail closed on a
   malformed ledger before another paid Phase 1 dispatch.
4. Make an explicit remote-wrapper state root authoritative across inherited
   environment and dotenv loading without changing global credential
   precedence.
5. Extend the existing read-only status surface in three stage-ordered slices:
   Phase 2a planning, Phase 2b text fill, then Phase 2c feasibility. Each owner
   applies its existing checkpoint validator and reports compatible, pending,
   stale/malformed, or `not_inspected`; status never grants reuse.

| Source slice | Tracking issue | Dependency |
| --- | --- | --- |
| Contract-bound Phase 1 cache identity | [#240](https://github.com/jasmineee-li/warp/issues/240) / [PR #248](https://github.com/jasmineee-li/warp/pull/248) | complete |
| Explicit remote state-root authority | [#241](https://github.com/jasmineee-li/warp/issues/241) / [PR #250](https://github.com/jasmineee-li/warp/pull/250) | complete |
| Phase 1 single-owner CLI lifecycle | [#242](https://github.com/jasmineee-li/warp/issues/242) / [PR #249](https://github.com/jasmineee-li/warp/pull/249) | complete |
| Observed Phase 1 cost durability and status | [#243](https://github.com/jasmineee-li/warp/issues/243) / [PR #251](https://github.com/jasmineee-li/warp/pull/251) | complete after #240 |
| Phase 2a planning checkpoint status | [#244](https://github.com/jasmineee-li/warp/issues/244) / [PR #252](https://github.com/jasmineee-li/warp/pull/252) | complete after #243 |
| Phase 2b text-fill checkpoint status | [#245](https://github.com/jasmineee-li/warp/issues/245) / [PR #253](https://github.com/jasmineee-li/warp/pull/253) | complete after #244 |
| Phase 2c feasibility checkpoint status | [#246](https://github.com/jasmineee-li/warp/issues/246) / [PR #254](https://github.com/jasmineee-li/warp/pull/254) | complete after #245 |

Cache identity, Phase 1 ownership, and remote-root authority may develop in
parallel. Cost durability stacks after cache identity because both touch Phase
1 generation callers. Phase 2a, 2b, and 2c visibility then stack in order on the
shared status formatter. The focused
[research note](dx-vertical-slices-online-research-2026-09-03.md) records the
source findings, primary-source guidance, counterfactuals, and explicit
non-goals.

#### Source feedback acceleration slices

These accepted slices improve the source-delivery loop while E3's exact
provider route is unavailable. They do not gate a provider resume and do not
change generation, admission, exposure, reset, grading or benchmark evidence.
The measured 2026-09-04 baseline over PRs #248--#254 is 154 seconds median for
the Taskgen workflow: 126 seconds in core pytest, 16 seconds in remote tests
and 37 seconds in package proof. The separate Root gates workflow is about 111
seconds median elapsed. A representative core run spent about 122 seconds in
pytest and 1.3 seconds in locked uv synchronization, so dependency caching is
not the primary target.

| Source slice | Tracking issue and delivery | Status |
| --- | --- | --- |
| Expose real core durations and replace the two measured duplicate readiness walks while preserving both assertion sets | [#255](https://github.com/jasmineee-li/warp/issues/255) / [PR #260](https://github.com/jasmineee-li/warp/pull/260) | complete |
| Split core tests into two feature-oriented lanes with three-way matrix capacity and exact selection parity | [#257](https://github.com/jasmineee-li/warp/issues/257) / [PR #261](https://github.com/jasmineee-li/warp/pull/261) | complete after #255 |
| Route clearly Taskgen-only changes to a successful no-op inside the still-required Root gates job | [#256](https://github.com/jasmineee-li/warp/issues/256) / [PR #259](https://github.com/jasmineee-li/warp/pull/259) | complete |

The historical combined target was an approximately 80--90 second merge-gate
critical path. The delivered split measured a 90-second critical test job and
a 99-second complete Taskgen workflow including its aggregate. The two core
lanes and existing remote lane run concurrently with `max-parallel: 3`; exact
selection-parity tests preserve the prior core selection. The work did not add
a scheduler grid, repeated timing campaign, third core shard, larger paid
runner, timing manifest or shared virtual environment.

The supporting [agent-DX note](agent-dx-frontier-2026-09-04.md) and
[CI/test acceleration note](ci-test-acceleration-frontier-2026-09-04.md)
record the current-source inspection, primary-source research, counterfactuals
and rejected complexity.

The post-delivery [next-frontier note](agent-dx-next-frontier-2026-09-04.md)
records the measured #259--#261 result, current OpenRouter capacity evidence
and proposed follow-ups. Its per-card Phase 1 route-map choice remains
unresolved and is not an authorized source ticket.

#### E3 provider-boundary resumption check

Delivery is complete in [PR #264](https://github.com/jasmineee-li/warp/pull/264).
The remaining gate is external provider capacity followed by one successful
direct and one successful sandbox production micro-canary.

Do not add a no-provider command, mock service, alternate parser/compiler, or
seven hand-authored provider-response fixtures. Representative feature tests
already substitute transport while exercising production compilation and
validation. The unverified behavior is genuine producer-to-consumer response
compatibility.

Immediately before resuming paid Phase 1, use the exact frozen OpenRouter route
to call `generate_new_tasks_for_site` twice for one row from a disposable
output/state root with card slicing disabled:

1. a host-action-only card and required capability profile that force the
   direct contract-bound backend and then traverse top-level final validation
   and cache behavior; and
2. a model-owned, non-host-action-only card that forces the sandbox backend.

Before dispatch, verify and record only the non-secret effective route: expected
OpenRouter base host, frozen model, required auth variable present and
higher-precedence OAuth variable absent. Never print a credential. These calls
use provider/account capacity, write generation files and may create sandbox
work, but do not touch an external Site or browser. Retain the compiled output,
cost summary and telemetry where the owning backend already produces them under
the existing run policy. Current entry points do not preserve raw Phase 1
provider envelopes. Promote one sanitized genuine replay only after a recurring
boundary failure justifies a deliberately reviewed recorder at the owning seam.
An Anthropic-direct run may be labeled transport evidence, but it does not
establish the frozen OpenRouter route.

From `packages/warp-taskgen/`, load the configured repository credential, then
run this exact retained-Run check. The output path must not already exist.

```bash
set -a
source /Users/ashtonchew/projects/warp/.env
set +a
export WORLDSIM_PHASE1_CONTRACT_BOUND_API=1

SOURCE_RUN=/Users/ashtonchew/projects/warp/.codex-worktrees/issue-212-expanded-bank/packages/warp-taskgen/logs/issue-212-expanded-bank/runs/15-non-tac-expanded-bank-bounded
OUTPUT_ROOT="$PWD/logs/issue-212-expanded-bank/runs/17-provider-microcanaries-$(date -u +%Y%m%dT%H%M%SZ)"

uv run python scripts/run_phase1_provider_microcanaries.py \
  --source-run "$SOURCE_RUN" \
  --output-root "$OUTPUT_ROOT" \
  --direct-card-id gitlab_issue_description_public_followup_issue_paper \
  --sandbox-card-id gitlab_compare_decide
```

The command checks the documented OpenRouter current-key endpoint first and
stops while remaining capacity is absent or unknown. It then runs the direct
production generation boundary before performing Modal preflight/upload and the
sandbox production boundary. Only a successful two-row sequence authorizes the
separate retained Phase 1 resume; the command does not resume Run 15 itself.
After success, confirm that the earlier process or Remote Job has stopped, then
inspect and use the exact existing resume path:

```bash
WORLDSIM_PHASE1_CONTRACT_BOUND_API=1 \
  uv run warp-taskgen status "$SOURCE_RUN" --json

WORLDSIM_PHASE1_CONTRACT_BOUND_API=1 \
  WARP_TASKGEN_STATE_DIR="$SOURCE_RUN" \
  uv run warp-taskgen resume
```

### E4 — Evaluate the frozen expanded bank

Run the admitted bank through the ordinary Phase 4 path on isolated benchmark
instances. Retain benign utility, PVPO, task-broke/non-encounter, objective-
specific outcomes, attempts, model identity and cost. Do not rewrite historical
scores or turn benign-agent success into an admission filter. Export the
existing coverage/funnel report from the resulting artifacts.

### E5 — Execute one real matched-rewrite pair

Connect the merged study provider to the canonical Phase 4 proposal, repair,
browser, PVPO, reward and judge seams for one eligible retained baseline.
Replace the deterministic ordinary-critique test control on the live path with
a stage-matched ordinary model critique: the same model class, corresponding
call opportunity, input boundary and configured token/cost ceiling as the TP
diagnosis, without TP/VEA/cue fields. Record real usage for both; the existing
deterministic critique remains a source-test stand-in and cannot be reported as
the matched arm. Prove that both arms share task, payload, witness, constraints,
opportunity and repair allowance; only the TP arm receives TP/cue information.
Retain inapplicable, failed and no-rerun outcomes. This is a runtime canary, not
the paper comparison.

### E6 — Run the representative matched cohort

After E4, freeze an eligible cohort spanning evaluated existing and new
workflow families, then run one opportunity per arm with serialized mutable
state through E5's real providers. Report the fixed-index primary result,
shared-selector secondary view, browser attempts, token and cost accounting,
failures and uncertainty. Do not expand to a full task × model × ablation grid.

### E7 — Update the canonical manuscript

After E1–E6, edit the manuscript designated by the paper repository guidance.
Separate supplied workflow structure, generated content, admitted corpus,
model-evaluated rows and bounded live smokes. State TAC scope narrowly, keep
GitLab smoke distinct from agent performance, report null matched results if
observed, and preserve limitations. Editing is authorized after evidence;
submission or publication remains a distinct external action.

## 4. Parallel ownership and joins

| Lane | Owns | Joins |
| --- | --- | --- |
| TAC | E1 then E2: Rocket.Chat transport, Phase 2/4 composition, reader, exact readback and reset smoke. | Gates TAC admission/evaluation in E3/E4; no dependency on matched rewriting. |
| Corpus | E3 then E4: generation allocation, final frozen bank after the TAC join, admission/evaluation funnel. | Supplies evaluated new-family rows to E6 and paper evidence to E7. |
| Study | E5 then E6: real matched TP/ordinary providers, paired run and analysis. | E5 can develop independently; E6 waits for E4's frozen evaluated cohort and does not change default runtime. |
| Synthesis | E7: provenance-linked tables, claims and limitations. | Waits for the three evidence lanes. |

Each ticket owns one demonstrable behavior and one PR. Source work uses
separate short-lived worktrees from current `origin/main`; dependent PRs land
to `main` rather than remaining indefinitely stacked. Use one independent correctness/spec review per PR; repeat only to verify
a concrete repair. Runtime work uses separate state directories and existing Run
Artifacts, not another manifest or attestation layer.

For these E3 support slices, the issue itself is the compact agent execution
contract. It names the owning seam, the failure prevented, one positive and one
counterfactual test, the exact focused and shipping commands, the existing
inspectable status or artifact, the offline/live evidence boundary, and the
condition for escalation. This is issue content rather than a new checked-in
schema. New diagnostics identify the failed invariant, relevant path or state
root, and next safe command where one exists; human and JSON status retain the
same explicit unknown or blocked state.

## 5. External execution state and remaining boundaries

These are current operational facts, not reopened architecture choices:

1. **E1/E2 are complete:** the approved disposable TAC deployment, separate
   ordinary identities and host-owned reset path supported the retained
   no-model Rocket.Chat smokes. They do not need to block source CI work.
2. **E3 inputs are frozen:** `claude-sonnet-4-6`, the existing Modal and
   Anthropic-compatible runners, 140 rows at 20 per family, and the historical
   `$3,000` run-level backpressure value are recorded on #212. The current
   overall remaining ceiling is `$5,000`, as specified above; the earlier
   generation value is not an additional allocation.
3. **E3 remains externally blocked:** the configured frozen OpenRouter route
   currently authenticates but its redacted capacity response reports no
   usable capacity. The delivered microcanary runner stops with
   `capacity_unavailable` before direct generation or Modal setup. The earlier
   direct one-row HTTP 403 is retained historical failure evidence. Neither
   check touches a browser or Site, and exact account metadata remains local.
   Once the route reports usable capacity, run the direct and sandbox canaries
   through the delivered command and only then resume the exact retained Phase
   1 request. Do not silently fall back to Anthropic-direct and claim the
   frozen-route result.
4. **E4/E6 remain downstream scientific runs:** they wait for the admitted bank
   and real matched pair and must preserve instance/reset isolation. Mutable
   runs are not parallelized on one reset-sensitive instance.
5. **Paper/publication remains separate:** do not edit the manuscript, submit,
   upload, release or publish without the separately required approval.

Provider-independent tickets #240--#246 and source-feedback tickets #255--#257
are complete. Any further authorized source ticket uses a ticket-local locked
environment, focused tests, bounded review and ordinary PR delivery. It must
not invent credentials or results, weaken a safety/runtime check, or treat
source/CI success as benchmark evidence.

## 6. Deferred work

Keep the accepted deferrals: Plane and cross-application generation, live NPCs,
multi-round matched rewriting, a universal workflow/binder/auth framework, a
universal semantic judge, and new evidence machinery. The current decision criteria and scientific value for Plane and cross-Site
work are recorded above; E1 success does not prohibit later reconsideration.
