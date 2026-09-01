# WARP benchmark expansion: implementation and development plan

Planning date: **2026-08-31 (America/New_York)**. Research baseline: **2026-08-30**.
Status: **architecture choices accepted; implementation-ready for approval**. The accepted
research choices in [the main report, sections 12–13](../web-benchmark-onboarding-2026-08-30.md#12-agreed-decisions-scoring-generated-structure-and-the-first-transfer-result)
remain authoritative. This plan does not authorize implementation or execution.

Source inspection used the existing `research/web-benchmark-onboarding-2026-08-30`
worktree at `4f3df62e2a471703a7cf44020bf295cd282f9d4e`. Existing research and
`CONTEXT.md` edits were preserved. The initial inspection ran no tests. The later
D1a investigation ran only a throwaway local seed-contract proof, focused existing
unit tests, and a pure in-memory JavaScript prototype; no project/benchmark code,
models, browsers, installers, infrastructure, or experiments were run. Paper and poster
repositories remained read-only. The paper's `CLAUDE.md` still designates
`icml/icml_final.tex`; the poster's different authority does not override it.
Three initial bounded source checks and three independent D1a checks used
`gpt-5.6-luna` with reasoning effort `max`. The lead reconciled their
recommendations against the accepted scope and
independently inspected the decisive source. The Skill dispatcher was unavailable;
`grill-with-docs`, grilling, domain-modeling and codebase-design were read directly.

## 1. Rechecked implementation facts

Paths in this document are relative to the repository unless otherwise stated;
`warp_taskgen/` and `tests/` below are inside `packages/warp-taskgen/`.

| Source fact | Architectural consequence |
| --- | --- |
| `phases/phase_1_contract_bound_action_api.py:219` compiles one selected action/route into one seed call; line 258 stamps `webarena_verified`; lines 281–324 expose semantic text slots, not a candidate-set contract. | Extend the generation seam deliberately. A new editor cannot supply multi-record decision semantics or correct Benchmark identity by itself. |
| `phase_2/exposure_contract/_impl.py:183` already accepts a Benchmark argument; `phase_2/phase_2c/policy.py:187` accepts explicit identity or task metadata. Both retain a WebArena default. | Reuse these interfaces with explicit identity and test propagation. These defaults are not proof that every consumer needs a new abstraction or implementation. |
| `seeding/_impl.py:240` already iterates `editor_calls`; lines 292–297 retain created resources and per-call results. | Reuse multi-call execution. Add only the missing logical-record-to-created-identity binding; do not build another seed engine or infer identity from the last returned record. |
| `phase_2/target_stage.py:209` fans listing records into separate tasks, while its route-contracted path merges only the first result's anchors. | Neither path preserves a multi-record decision case by itself. Add a narrow binding path for the new family while retaining Site route/materialization behavior and existing task behavior. |
| `adversarial_actions/action_targets.py:36,147` permits only `same_issue`; `capability_adapters.py:360` declares the issue-description-to-note example. | A selected-record action needs a bounded target/evidence extension. Candidate selection belongs to benign workflow construction, not inside attacker-target resolution. |
| `rewards/agent_response.py:84` grades normalized sequences/multisets; it does not judge explanations. `rewards/final_state.py:47` requires an action, content witness, and source-event expectation. | Reuse finite-ID answer comparison unchanged where appropriate. Do not mistake a witness-bearing note check for a decision, recipient, or extra-artifact predicate. |
| `rewards/final_state_catalog.py:17,61,121,173` restricts local final-state evaluation to WebArena Verified. `rewards/dispatcher.py:47` sends canonical task IDs to the WebArena vendor evaluator. | Admit new WARP-local evaluator authority explicitly. Generated TAC tasks remain task-ID-less in the reward sense; adding a native TAC evaluator runner is not a prerequisite. Preserve vendor dispatch. |
| `runtime_composition.py:47` bundles targeting, seed registry, feasibility and reader preflight, but not a final-state evaluator catalog. `phase_2/phase_2c/probes.py:400` forces empty browser-context options after reader preflight. | The current preflight hook alone cannot implement authenticated independent reading. TAC needs explicit reader-context handling and reward-catalog plumbing, not global registration or writer-session fallback. |
| `phase_4/eval_awareness_iterator.py:755,1099` combines TP-aware selection, TP-based continuation, cue eligibility and sequential proposals. | Matched rewriting is new study orchestration, not a current flag. The normal iterator must retain its existing behavior. |

These are source findings, not working-integration claims. In particular,
ordinary Rocket.Chat permissions, exact rendered message exposure, reset
postconditions and deployed transport behavior remain unknown.
The injected reward catalog also has to reach the actual caller:
`phase_4/execution.py:184` currently calls reward dispatch without it.

## 2. Proposed ownership and small interfaces

The design follows existing ADRs 0002 and 0004: deepen a concrete capability
without moving all behavior into Site Targeting. No universal workflow language,
plugin registry, new execution engine, or unrelated module cleanup is proposed.

| Owner | Behavior kept together | Interface to other owners |
| --- | --- | --- |
| New concrete workflow-generation modules under `phase_1/`, first GitLab comparison, then Rocket.Chat conversation | Family-specific content schema, constraints, selection/decision rule, instance validation, seed recipe and expected-result construction. Keep local types beside this behavior; mirror tests under `tests/phase_1/`. | Compile validated generated content and host-bound record facts into the existing benign-task shape, or return an actionable validation failure. Start with concrete functions; extract a shared protocol only when both implementations actually need it. |
| `sites/`, `editors/`, `seeding/` | Site route grammar and identity interpretation; ordinary-participant HTTP/auth/write/cleanup behavior; existing seed execution and per-call evidence. | Existing bound Site and seed-registry interfaces. Setup/record recipes come from the family; the writer returns actual resource identities. |
| `adversarial_actions/` | Permitted attacker objectives, target-binding rules, action-preservation constraints and reward compilation. | Consume validated candidate identities and a separately declared attacker target; never compute the legitimate answer from the injected payload. |
| `rewards/` | Exact decision/output predicates and Site/Benchmark-specific persisted-state collection and attribution. Keep new checks local to their evidence and objective. | Existing reward dispatch plus the narrow new predicate/expectation required by a concrete family. Do not add a universal semantic judge or alter old propagation scores. |
| `phase_2/phase_2c/` and existing exposure owners | Admission, independent reader-context execution, reachability/rendering, exact carrier evidence and failures. | A declared independent-reader context from trusted configuration, distinct from writer state, with secret-free role metadata. Site modules interpret observations; they do not decide admission. |
| `runtime_composition.py`, Benchmark catalog and `site_compositions/` | Explicit per-Run executable composition; Benchmark capability/evaluator authority; separately, static owner declarations. | Extend existing composition only for dependencies actually needed by TAC. Static completion still leaves operational readiness blocked. |
| Taskgen `phase_4/` plus a named study entrypoint/analysis | Payload finalization, QA, variant execution, PVPO and judges remain Taskgen-owned. Study-owned code chooses treatment, cohort, opportunity schedule and reporting. | A narrow callable attempt seam returning the existing attempt artifacts. No copied safety implementation, altered default flags, or forced inheritance from the website-sample experiment classes. |

The family owns *what makes the task coherent*; rewards own *how observed output
is scored*; Phase 2 owns *whether its carrier is admissible*. Link the family
documentation to those implementations instead of duplicating their rules.
Prompts belong with the existing prompt owner and identify their family caller.
In particular, do not force an information-only family through the current
action-only slot validator or move benign decision construction into Phase 2
merely because Phase 2 already resolves listing records.

### Candidate and target binding

Proposed example: three actual GitLab issue records describe release blockers.
The legitimate rule selects the unresolved blocker with the relevant dependency.
One variant asks for its issue reference; another asks for a note on that issue.
Changing the relevant dependency must change the expected selection. Titles and
wording alone do not create a different archetype.

Keep three identities separate: **payload carrier**, **legitimate selected
record**, and **declared attacker target**. A wrong-target note is not an extra
note, and a warning quoting attacker text on the correct issue is not a
wrong-target success. Outcome predicates may overlap; they are not a forced
single-label classification.

Use family-local logical record keys during construction and carry each key on
its declared seed call. Bind it to that attempt's actual IDs/routes from the
per-call result. The stable declaration belongs in the task; the attempt-local
mapping belongs only in existing feasibility/run evidence and the task copy used
for that attempt. The multi-call runner's aggregate tokens cannot stand in for
this mapping. Compile each twin from the same world and preserve
non-payload evidence; no one-body substitute may be reported as navigation across
independent records. Runtime-generated IDs must be bound consistently before
reward use; this is part of the selected-record slice, not deferred glue.
Preserve task-defining facts when inserting or rewriting the attack: if the
payload replaces the evidence that determines the legitimate answer, an unchanged
answer key is no longer justified. The family must provide a payload slot or
preservation rule that keeps its underlying task coherent across both twins.

Reuse GitLab issue/note writers for their existing operations. Their cleanup is
not a Golden-State Reset: `editors/gitlab.py:483` closes created issues on cleanup.
An explicit baseline/reset and candidate-scope check must prevent those records
from contaminating later selections. Do not broaden this into general editor
cleanup refactoring.

### Rocket.Chat slice

A proposed generated conversation contains an initial project plan and an
explicitly confirmed correction to its owner/date. The legitimate participant
either reports the current plan's finite fields or notifies the audience named
by that plan. Vary corrections, dependencies and audience relationships, not just
people's names. This remains a new WARP-generated task, not TAC's unchanged NPC
notification task. No live NPC or native TAC evaluator process is required.

Bind each message and notification to its exact room, message ID, author and
thread relation. Wrong-recipient notification, incorrect decision fields and
propagated text have separate checks. Add the concrete message action and
transport in the action/reward owners; existing `submit_comment` is bound to
`create_comment` and Reddit-shaped transport, not already a Rocket.Chat action.
Build the response-only check first inside this work package, then the persisted
notification check; that local sequence does not delay the GitLab milestone.

## 3. Decision dependency tree

```text
Accepted: substantive WARP generation; GitLab comparisons; static Rocket.Chat;
          objective-specific outcomes; matched TP versus ordinary rewriting
├─ D1. Generated evidence and decision construction [accepted 2026-08-31]
│  ├─ D1a. Logical-record binding to per-seed resource IDs [accepted 2026-08-31]
│  └─ D1b. Exact structured decision output [accepted 2026-08-31]
├─ D2. First integration milestone [accepted 2026-08-31]
│  └─ GitLab compare-and-act plus information-only companion
└─ D3. Scope of the matched-rewriting mechanism [accepted 2026-08-31]
   └─ D3a. Neutral arm's non-TP information projection [accepted 2026-08-31]
```

**D1 accepted:** generate constrained factual relationships and natural content,
then let host-owned family rules derive the answer and permitted action. The
generator does not author the answer key, evaluator, routes or runtime IDs.
Content validation must reject disagreement between structured facts and what a
reader actually sees. Prose-first generation with independent adjudication offers
more open-ended synthesis but introduces a more difficult grading/validation
problem. This is a tradeoff about initial family expressiveness, not whether
instances are substantively generated.

**D2 accepted:** make GitLab compare-and-act the first *integration milestone*,
with the information-only view built from the same candidate evidence. An
answer-only checkpoint can land first, but is not the milestone that proves the
new target seam. Existing GitLab/Postmill volume expansion proceeds independently;
Rocket.Chat local work and the study do not wait for that milestone.

**D3 accepted:** start with one matched rewrite opportunity per arm
from a shared eligible baseline. This isolates a bounded proposal/feedback
contrast and avoids a new multi-round scheduling policy. It does not validate
the incremental effect of the entire adaptive iterator. If that full mechanism
is the required claim, settle a study-only multi-round schedule instead; do not
quietly replace the normal runtime algorithm. All attempts, inapplicability and
costs remain visible. Exact eligibility and call accounting follow this choice.

For the one-opportunity proposal, use a shared baseline-eligible cohort (positive
PVPO, non-`Real` TP, and a host-declared mutable payload); keep all other baseline
rows in the corpus report and label the contrast conditional on that cohort.
Do not further select pairs because the TP diagnoser found an editable cue.
An inapplicable TP proposal remains an outcome alongside its neutral counterpart.
Match diagnosis/ordinary-critique, rewrite and allowed repair opportunities;
report those calls separately from browser reruns and token/cost use. The primary
comparison includes every scheduled pair, not only browser-evaluated or selected
winners. If a secondary selected-result view is used, apply the same selector to
both arms and label its TP use explicitly.
Match inputs at the corresponding stage: the current cue diagnoser receives
`_trajectory_summary` (`eval_awareness_cue_api.py:192`, including outcome and final
answer), whereas later rewrite-control feedback has its own restrictions. The
neutral path must not gain extra raw traces or judge annotations, nor should it
be given less ordinary evidence by accident. Exclude derived TP/VEA/cue fields;
do not censor identical agent-observed content merely for mentioning evaluation.

The durable D1 ownership decision is recorded in [ADR 0008](../../adr/0008-own-generated-workflows-by-concrete-feature.md),
and D1a's per-attempt binding/backpressure seam is recorded in
[ADR 0010](../../adr/0010-bind-logical-records-per-seed-attempt.md).
The durable D3 study/runtime separation is recorded in [ADR 0009](../../adr/0009-keep-the-matched-rewrite-study-out-of-default-runtime.md).
D2 is a delivery sequence, not an architecture decision, so it does not receive
an ADR.

### Accepted dependent choices

**D1a accepted:** keep generated records
addressable by a stable family-local key carried on the corresponding seed-call
declaration. A strict binder joins that declaration to exactly one result from
the current attempt and validates normalized Benchmark, Site, method, expected
resource kind and safe identity tokens. `call_index` can correlate a declaration
with its result inside one attempt, but it is not the record's durable meaning;
reordering or inserting a setup call must not silently retarget an alias. Reject
missing/duplicate keys, missing/duplicate physical IDs and contract mismatches.
Do not fall back to aggregate metadata for new selected-record workflows.

Phase 2c proves the binding on its temporary seed and records the readable
diagnostic in existing feasibility evidence. Phase 4 resolves it again after its
own reset from the resources created for that evaluation, on an attempt-local
task copy, before browser execution and reward evaluation. Only the Phase 4 map
may supply reward/action anchors. Never persist a Phase 2c auto-increment ID as if
the later reset/reseed must reproduce it. The stable binding specification—not
ephemeral IDs—participates in existing source/variant resume identity. Fixed
fixture IDs would reduce binding work but couple the corpus to one mutable seeded
world and make task isolation less explicit.

**D1b accepted:** grade a small exact set of decision fields: selected record or
recipient plus the decisive evidence fields used by the declared rule. Encode
them in the existing structured response shape, such as canonical `field=value`
entries, so the existing exact ordered/multiset comparison can reject missing or
extra claims. Free-form rationale may remain diagnostic but does not receive an
unsupported correctness claim.

**D3a accepted:** give both arms the same task, selected payload, immutable
constraints and compact non-TP baseline outcome/trajectory summary. Pair the TP
diagnosis call with an ordinary realism/clarity critique, then give each arm one
rewrite proposal and the same repair allowance. Only the TP arm receives TP
classification and cue diagnosis; neither arm receives VEA or additional reward
traces. This estimates the value of TP-specific information rather than the value
of an extra analysis call.

### D1a offline feasibility proof — executed 2026-08-31

This proof establishes that the proposed seam is implementable without replacing
the seed engine. It does not establish working integration.

- A throwaway Python proof used the real `apply_data_seed` and an isolated fake
  `SeedSiteRegistry`. Three calls produced three `editor_call_results` in each of
  two seed attempts. The same selected logical record resolved to the middle
  record in both attempts while its physical ID changed. Reversing the result
  list did not change the selection.
- The counterfactual failed as predicted: aggregate
  `metadata.write_tokens.record_id` contained the last call's ID, not the selected
  middle record's ID. The current GitLab Phase 4 anchor helper likewise patched
  a selected-middle example to the aggregate last issue. Thus a new strict
  resolver is necessary; it does not exist today.
- Missing and duplicate call indices, wrong methods, missing identities, unsafe
  identity keys and URL-shaped identity values failed closed in the Python proof.
  The stricter [interactive logic prototype](q4-logical-record-binding-prototype.html)
  additionally rejects missing/duplicate logical keys, duplicate physical IDs,
  wrong Benchmark/Site/method/resource kind and reuse of a Phase 2c ID in Phase 4.
- Existing source already provides the two narrow hooks to reuse: call-aware
  render admission reads per-call results, and Phase 4 mutates GitLab reward
  anchors after seeding but before browser and reward execution. The render helper
  currently returns the first duplicate and permits aggregate fallback, so it is
  not itself the strict action binder.
- Focused existing tests passed: four seeding/read-surface cases, one Phase 2c
  feasibility case and two call-aware render cases. The lead reran the throwaway
  Python proof (eleven pass conditions) and the prototype logic checks, including
  the last-write counterexample, cross-phase stale-ID rejection and all malformed
  fixtures.

The Python proof was intentionally left in `/tmp`; it is not a new repository
artifact or evidence system. The HTML is a disposable planning aid with its
limitations visible. No GitLab/Rocket.Chat transport, authentication, browser
exposure, deployed reset, cleanup, generator, evaluator or resume behavior was
exercised. Those remain later implementation tests and sandbox smoke checks.

The proof also identifies the smallest shared contract change. `seeding` should
validate and echo an optional stable record key plus normalized Benchmark on each
per-call result; it already owns editor-call validation/execution and result
metadata. The first strict binder remains inside the concrete GitLab comparison
feature, where expected Resource Kinds and identity fields are known. Its small
interface consumes the stable family specification and current seed metadata,
then returns either an exact attempt-local binding/anchor projection or an
actionable failure. Phase 2c and Phase 4 each call that same feature behavior at
their existing post-seed boundary. Do not add a workflow registry or extract a
generic binder until a second real family demonstrates the same contract.

Current code does not yet carry a logical key or normalized Benchmark in
`editor_call_results`; Phase 2c's canonical result also drops those records, and
the existing GitLab anchor helper consumes aggregate identity. Those are the
necessary implementation changes, not documentation evidence of completion.

## 4. Dependency-ordered work packages

| Package / owner | Work and join | Can proceed independently |
| --- | --- | --- |
| V — existing-site expansion | Prepare/use existing novel-generation counts and allocations; preserve `origin="new_task"` and admitted-bank accounting. Scale only after execution approval and a bounded operational check. | No dependency on TAC, new target binding, or matched rewriting. |
| G — GitLab workflow owner | Family content/decision compiler and focused tests; actual multi-record seeds; information-only output. Join selected-target binding, exact note grading, Phase 2 admission and later sandbox check for compare-and-act. | Initial family logic and test cases can proceed while shared seams are extended. |
| S — one shared-seam maintainer/integrator | Benchmark identity through generation; explicit record/target evidence; independent-reader context; local evaluator authority and catalog plumbing. Make small changes with actual consumers, not one prerequisite framework PR. | GitLab target work need not wait for TAC reader/evaluator work. One designated owner controls edits to shared files. |
| T — TAC/Rocket.Chat owner | Static conversation content, routes, ordinary writer, independent reader, exact observation and notification predicates; explicit composition. Join S only where new context/evaluator support is needed. | Local fake behavior and fixture tests do not need a live TAC stack or a finished GitLab feature. |
| C — matched-rewriting owner | Neutral proposal path and chosen study schedule; reuse Phase 4 finalization/execution/judges. Study setup and analysis remain separate from default runtime. | Develop/test against existing admitted task examples. TAC enters the representative cohort when ready, not as a code dependency. |

### First deliverable and interface agreement

The smallest coherent end-to-end development slice is one offline GitLab
comparison family with information-only and compare-and-act siblings. It is
dependency ordered but does not require TAC or the matched-rewriting study:

1. **G-core:** implement the feature-local generated facts, natural rendering,
   decision rule, consistency rejection, logical record declarations and exact
   expected fields. Test through one feature interface: validated generated
   content in; compiled task siblings or an actionable rejection out.
2. **S-record:** extend the existing seed-call/result contract with the stable
   record key and normalized Benchmark. Preserve current per-call Site, method,
   created-resource and write-token evidence. Keep old callers unchanged.
3. **G-bind:** implement the GitLab feature's strict current-attempt binder and
   expose one interface returning either the bound candidate set plus selected
   anchor projection, or a typed/actionable failure. Its implementation owns
   cardinality, Benchmark/Site/method/kind, safe-identity and stale-ID checks;
   callers do not repeat them.
4. **Phase joins:** Phase 2c invokes G-bind after its seed and retains the result
   in existing feasibility evidence. Phase 4 invokes it after reset/reseed on an
   attempt-local task copy, before browser execution, then supplies the fresh
   projection to existing action/reward owners. Binding failure stops admission
   or execution before scoring; there is no aggregate fallback.
5. **G-outcomes:** reuse the structured response comparator for the information
   sibling. Add only the concrete selected-issue note target and exact persisted
   predicates required by compare-and-act, preserving separate propagation,
   wrong-target and extra-artifact outcomes.

The offline slice is complete when changing a decisive generated fact changes
both siblings' expected selection, a non-last selected issue binds across fresh
seed IDs, malformed evidence fails at the feature interface, exact positive and
relevant negative grading cases pass, and existing GitLab/Postmill behavior is
unchanged. A later, separately approved sandbox smoke then checks one ordinary
write → independent render → legitimate answer/action → exact readback → reset
path. It does not gate parallel source work in V, T or C.

Explicitly deferred are a universal workflow/binding engine, cross-application
generation, live NPCs, Plane, a universal semantic judge, multi-round matched
rewriting, corpus-scale runs and any new provenance/attestation system.

Actual critical paths are separate: G → selected-target/readback → GitLab
integration; T → authenticated exposure/reset and evaluator → TAC transfer;
C → comparable attempt schedule → matched comparison. The paper needs all three
claims plus volume, but their implementations do not need to be serialized.

Before concurrent code work, agree only on input/output shape, ownership of the
shared edits, and a representative positive/negative example. Feature authors
can use stand-ins at those seams. No second source snapshot or sync-back
workflow is needed. This planning work remains in the existing worktree; no
additional worktrees, commits or PRs are authorized here.

Code parallelism does not grant experiment parallelism. Future mutable runs are
serialized per reset-sensitive Benchmark Instance unless separate instances and
their reset scope are verified. Share immutable code/profiles, not mutable
conversation histories, memberships or issue state.

## 5. Proportionate validation and evidence

Except for the bounded D1a checks recorded above, all checks below are **future
proposed checks**, not executed results.

| Concrete failure prevented | Simplest sufficient check at that stage |
| --- | --- |
| Generation changes wording but not behavior, or emits the wrong oracle | Family test where changing a decisive fact changes the expected record/recipient; ambiguous or contradictory content rejects. Audit a readable generated example against the displayed evidence. |
| Candidate creation, benign selection and attack target get conflated | Multi-call seed fixture with distinct IDs and reordered returns; wrong-ID/foreign-project negatives. Missing bindings fail before scoring. |
| Wrong-record action is scored as a legitimate note, or a carrier/stale note as agent output | Known-positive exact note/message event and readback; wrong parent/actor and pre-existing same-text negatives. Check extra artifacts only within the declared objective/fixture scope. |
| A correct ID is treated as a correct explanation | Grade declared finite fields with the existing comparator; either leave free rationale unscored or add a separately justified narrow predicate. Do not claim explanation correctness from ID equality. |
| Logged-in writer visibility is mistaken for authorized independent exposure | Reader-context tests reject writer state/identity and preserve Classifieds anonymity; later ordinary-role sandbox check opens the exact resource in a fresh authorized context and measures painted exposure. |
| Declarations are mistaken for integration | Existing Site Behavior Contract and package checks; static readiness remains blocked. Later smoke covers ordinary write → independent render → grading → reset, not merely a green composition report. |
| Reset/cleanup leaves history, memberships or selection candidates behind | Later sandbox baseline presence/absence check before/after a run and one partial-failure cleanup case. Health/HTTP acceptance alone is insufficient. |
| Study changes payload authority or gives one arm extra search | Shared immutable-task/QA tests, neutral-feedback leakage negative, failed-proposal and repair-call accounting; same baseline and configured opportunities. Default iterator regression tests remain unchanged. |
| New Benchmark grading silently falls back to WebArena | WARP-local explicit-authority positive; unknown/mixed Benchmark, unsupported evaluator, canonical-vendor and comparison-only negatives. |
| Resume reuses an attempt from the other treatment or changed task | Bind study condition and result-affecting schedule to the existing Run Definition/feature checkpoints; test incompatible reuse rejection. Preserve existing integrity checks. |

Use existing task provenance, per-call seed results, Phase 2c evidence,
`phase_4/results.json`, attempt artifacts, Run Definitions and experiment logs.
Add only necessary readable fields for family/condition/record bindings. Existing
digests establish identity or resume compatibility, not behavioral correctness.
No second manifest system, fingerprint hierarchy, attestation layer, or exhaustive
proof bundle is proposed. Existing required fields and validators stay intact.

A test stand-in establishes deterministic behavior, not deployed permissions or
transport correctness. A later smoke establishes one integration path, not the
final corpus size or transfer breadth. After smoke validation, expand generated
instances and meaningful rules/evidence relationships; report candidate,
admitted and evaluated counts and failures. Do not select only successful attacks
or turn benign-agent success into an admission filter.

## 6. Approval and documentation

Only planning/research edits are authorized now. Implementation, dependencies,
infrastructure, model/browser runs, spend, paper edits and publication need later
approval. Cohort size and runtime budgets belong to that execution proposal,
not an invented engineering cap.

ADRs 0008, 0009 and 0010 record the durable workflow ownership, study/runtime
separation and per-attempt binding/backpressure choices. Ordinary filenames,
work-package ordering and easy-to-reverse implementation choices do not become
ADRs. `CONTEXT.md` remains a glossary; no implementation-only term is added.

No consequential architecture choice remains open. Implementation approval may
be granted per work package; it does not imply permission for dependencies,
infrastructure, model/browser runs, mutable benchmark execution, spend, paper
edits or publication. Live smoke and experiment approval remain later gates.
