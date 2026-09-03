# WARP Taskgen modular scaling model

Inspection date: 2026-09-02 (the requested planning date is 2026-09-01; this
note records the actual inspection date).  The source snapshot inspected was
`origin/main` at `2f2ce2b4`.  The repository's research baseline is
2026-08-30.  This is source inspection and planning only: no implementation,
benchmark run, browser session, or infrastructure launch was performed.

## Executive finding

The current scaling unit is a host-owned task-card/adapter plus a deep,
feature-local world/compiler.  A generated task follows this path:

```text
per-Site generation -> generic Phase 1 validation -> host-owned card metadata
  -> optional feature-local compiler -> shared seed/exposure/readback/reward
```

This is already a multi-archetype model, but it is not a workflow engine.  A
task card selects the allowed capability, Site and route family; a feature
module owns the substantive records, rule, target binding, evaluator and
feature evidence.  Shared seams should be extended only when a real second
consumer demonstrates the same missing contract.

## What exists today

| Concern | Current owner and evidence | Boundary for a new feature |
| --- | --- | --- |
| Archetype and card metadata | `packages/warp-taskgen/warp_taskgen/adversarial_actions/capability_adapters.py`, `capability_task_cards.py`, and `phases/phase_1_task_cards.py` define immutable `CapabilityTaskAdapter`/blueprint records, compile named plans, and validate route/action/reward/precondition/scenario/target compatibility. | Add a Plane card only for a concrete host-owned capability. Do not create a second registry or put Plane rules in this layer. |
| Per-Site generation and feature dispatch | `phases/phase_1_generate_new_tasks.py` runs Sites independently, computes expected counts and cache/resume fingerprints, validates generated output, then calls `_compile_phase1_feature_tasks`. That hook dispatches the known GitLab comparison contracts and otherwise leaves tasks unchanged. | Keep dispatch to an explicit Plane card contract. A feature compiler is evidence of a real consumer, not a universal workflow plug-in point. |
| Envelope validation/admission | `phases/phase_1/novel_task_validation/` checks required generated fields, exact IDs/count, route/card alignment and stable answer diversity, and overwrites model-authored host metadata. | Reuse unchanged for `id`, `origin`, `site/sites`, instruction, start URLs, seed and reward. Plane-specific facts and grading remain host-owned. |
| Site targeting | `sites/contracts.py` and `sites/catalog.py` provide immutable `TargetingContext`, `CanonicalRoute`, `SiteAdapter`, optional profile/read-surface/readback protocols, and `BoundSite`. GitLab and Rocket.Chat keep route grammar in their feature modules. | A real Plane consumer can add `sites/plane.py` and one route/profile/readback seam. Static design or fake tests need not enlarge the default catalog. |
| Seed/readback | `seeding/site_contracts.py` supplies an immutable per-run `SeedSiteRegistry`, `EditorSeedResult`, write identities, created-resource facts and read-surface provenance. Site editors still own auth, HTTP, mutation and cleanup. | Reuse if a Plane editor can emit exact child/state identities and read surfaces. Extend one editor/readback contract narrowly if it cannot; do not add generic CRUD. |
| Exposure, reset, reward and accounting | Existing Phase 2/2c, action-card, reset/run-definition and `task_bank`/coverage paths own their existing contracts. Task signatures and archetype fields support identity/accounting, not behavioral proof. | Keep safety, exposure, reset and reward owners intact. Add only Plane-specific evidence or evaluator logic where existing contracts do not express the claim. |

The current catalog defaults to GitLab and Reddit/Postmill.  Rocket.Chat is a
feature-local first TAC route/conversation result, not a replacement Site and
not evidence that all TAC workflows share a runtime abstraction.

## Quantified current pattern

`_WEB_ARENA_VERIFIED_ADAPTERS` contains nine base `archetype_id` values:

```text
issue_description_semantic_status
submission_body_semantic_status
comment_body_semantic_status
issue_description_public_followup_issue
issue_description_public_followup_comment
submission_body_public_followup_post
comment_body_public_followup_comment
issue_description_repository_maintenance
issue_description_wasp_comment_delete_project
```

There are seven named adapter profiles (including the opt-in Classifieds
diagnostic profile); the paper Tier 2 profile reuses four of the base entries
with a different reward shape.  The source also contains the explicit
workflow-family literals `public_followup`, `discussion_reply`,
`repository_maintenance`, `task_local_prerequisite_acknowledgement`, and the
`negative_control` family.  `DEFAULT_NOVEL_TASKS_PER_SITE` is 30, with
per-run/action-count overrides.  These counts show that cards already support
multiple generated instances and profiles; they do not establish behavioral
diversity by themselves.

The closest existing two-adapter proof is
`tests/test_phase_1_tasks.py::test_compiled_tier2_profile_filters_by_site_without_route_drift`:
one profile filtered to Reddit yields both submission-body and comment-body
cards, keeps both on Reddit, and rejects accidental “latest/newest” wording.
`tests/test_phase_1_tasks.py::test_pure_action_paper_profile_includes_reddit_comments_and_uses_host_action_only`
similarly exercises four cards across GitLab and Reddit.  These tests support
card/profile composition, not a universal adapter abstraction.

## Why GitLab and Rocket.Chat are feature-local

GitLab comparison code (`phase_1/gitlab_compare_decide.py`,
`gitlab_compare_decide_content.py`, `gitlab_compare_decide_generation.py`, and
`gitlab_compare_act.py`) owns:

- the deterministic three-record world and decision rule;
- model-generated substantive facts (never route, evaluator or physical IDs);
- compilation of decide/act siblings;
- current-attempt typed physical binding and stale/foreign-target rejection;
- exact expected response and action/readback reward behavior.

Rocket.Chat code (`phase_1/rocket_chat_contracts.py`,
`rocket_chat_decisions.py`, `rocket_chat_task_envelope.py`,
`rocket_chat_evaluator.py`, and the `sites/` runtime/reader/reset modules)
similarly owns finite conversation facts, decision derivation, exact output,
and its own route/runtime protocols.  Generic Phase code contains no
Rocket.Chat branch except the explicit Phase 1 feature hook and capability
metadata.  This separation gives each feature a substantial implementation
behind a small seam while preserving shared validation and safety behavior.

## What a deep Plane module should own

For the proposed Plane-only multi-record triage slice, a Plane feature module
should own the following facts and behavior:

- a bounded project/world containing three to five known issue records;
- record states (for example Backlog, Blocked, In Progress and Done), issue
  text and any assignee/metadata that the triage predicate actually needs;
- a deterministic triage predicate and expected selected set;
- separate information-only and selective-state-update task contracts, if both
  are retained, including the exact changed-record set and unchanged sentinels;
- generated substantive issue facts and their host-owned expected answer;
- Plane route grammar, logical-to-physical record binding for the current
  attempt, and editor calls that can identify each child/state update;
- exact readback/evaluation for selected records, unchanged records, wrong
  targets and unauthorized extra artifacts;
- feature-local fixtures, ordinary writer/independent reader behavior and
  reset diagnostics.

The public seam should stay small, analogous to GitLab's world/compiler/binder:

```text
PlaneTriageWorld / PlaneRecord
generate_plane_triage_world
compile_plane_triage_task
bind_plane_attempt
evaluate_plane_triage
```

Names are illustrative, not an implementation request.  The important design
is ownership: record/state/rule/expected updates and evaluator logic remain in
Plane, while Phase 1 only invokes an explicitly recognized card contract.

## Reuse versus a real shared extension

Reuse unchanged where the existing owner already expresses the evidence:

- Phase 1 task envelopes and generated-task validation;
- task-card plan compilation/validation and action-capability contracts when a
  Plane action is genuinely one of the declared capabilities;
- per-Site generation, cache/resume and count accounting;
- generic exposure/admission, action-card safety checks, Phase 4 checks,
  run-definition/resume compatibility and Golden-State Reset bookkeeping;
- `SeedSiteRegistry`/`EditorSeedResult` and the Site behavior-contract test
  family if Plane can emit safe identities, exact resources and read surfaces;
- task-bank archetype/signature accounting (while separately measuring
  structural behavior dimensions).

Only a concrete Plane runtime should justify narrow shared changes:

1. Add Plane route/profile mapping implementing the existing `SiteAdapter` and
   profile-route capability contracts.
2. Add one Plane editor method/readback observation or action-card carrier if
   current contracts cannot represent its exact state evidence.
3. After a real GitLab-to-Plane consumer exists, extract only the common
   cross-Site binding/run seam that both implementations exercise.

The following are speculative at this stage and would weaken locality and
backpressure: a universal workflow or binding engine, a global mutable card or
Site registry, a generic multi-Site DAG, a universal state-machine/semantic
judge, a reset coordinator that owns Site cleanup, or a catch-all CRUD adapter.
The first Plane module should reveal a missing seam through tests rather than
requiring one in advance.

## Deletion test and two-adapter test

The deletion/backpressure test already exists in
`tests/sites/test_composition_package.py::test_removing_unrelated_composition_preserves_active_site_checks`,
with fail-closed checks for removed/unknown Sites in
`tests/sites/onboarding_backpressure/` and
`tests/test_seed_site_contracts.py`.  A Plane design should add the following
red-test obligations before any broader abstraction is proposed:

1. **Deletion:** remove the Plane composition/card/module and verify that
   GitLab and Reddit catalogs, card plans, composition reports, findings and
   digests are unchanged; a Plane request fails closed; no Plane noun appears
   in generic Phase modules.  If deleting Plane changes an existing Site's
   output, the seam is too broad.
2. **Two adapters:** expose two independent Plane cards (recommended initial
   pair: information-only triage and selective state update) through the same
   immutable card/profile path.  Each must compile and grade its own contract;
   wrong card/Site and route drift must fail closed.  No process-global
   mutable registration may be needed.  If the second card requires no new
   shared field, existing seams are sufficient; if both require the same
   missing field, add one narrow shared contract with a named owner.  A need
   motivated only by a hypothetical third Site is not evidence.

These tests are architectural probes, not a claim that Plane has already been
implemented.  Their counterfactuals are useful: failure of deletion means the
proposed module leaked policy; failure of two-adapter composition means either
the Plane feature contract is underspecified or a genuinely shared seam is
missing.

## Evidence and runtime gates for Plane

The smallest sufficient pre-runtime checks are focused known-positive and
negative tests for:

- deterministic record uniqueness and a predicate that selects exactly the
  intended set;
- separate information-only versus state-changing contracts and card
  alignment;
- exact physical ID/state readback, unchanged sentinels, wrong-target and
  extra-artifact rejection;
- independent writer/reader identities, route near-miss rejection, and
  deletion/two-adapter locality.

Only after those pass should a narrow sandbox smoke test use a separate
Benchmark Instance with ordinary-role access, exact routes/auth, fixture
creation, independent readback and documented reset.  State-changing runs
need current-attempt binding, independent authorized exposure and Painted
Visibility where required.  Reusing one mutable instance requires serialized
runs and a Golden-State Reset between attempts.  A hash/fingerprint, merged
source, focused test, live canary, generated/admitted corpus count or one
completed Run proves only its own category; the paper claim requires admitted
corpus plus completed Runs and outcome analysis.

## Sequencing and parallel ownership

The critical path is:

```text
Plane world/compiler + two-card/locality tests
  -> Plane Site/seed/readback feasibility and exposure checks
  -> first admitted Plane-only Runs
  -> decide whether a shared cross-Site seam is evidenced
  -> GitLab-to-Plane generation and Runs
```

Parallel, non-overlapping work can include a Plane feature/design note and
fakes; read-only audits of existing card/validation contracts; and a separate
research note on GitLab-to-Plane claim/evidence requirements.  Do not run
state-changing experiments in parallel on one Benchmark Instance, and do not
edit generic Site registries while the Plane feature contract is still being
tested.  GitLab-to-Plane should follow Plane-only: it adds confounds (two
origins, identity/route mapping and cross-Site reset) that cannot be diagnosed
until the one-Site state/readback contract is reliable.

## Recommendation and reversal conditions

The source supports Plane-only multi-record triage as the smallest next
scientifically useful slice.  It adds meaningful dimensions not present in the
current one-record GitLab/Reddit action cards or finite GitLab comparison:
multi-record selection, state-transition cardinality, selective versus
information-only outcomes, unchanged-record evidence, and wrong-target/extra
artifact attribution.  It can reuse the existing generated-task and safety
contracts without making TAC or cross-Site support a prerequisite.

Stop or reverse this recommendation if any of the following becomes true:

- Plane output varies only wording, IDs or record count and adds no new
  predicate, state transition, action/evidence burden or outcome category;
- exact readback, current-attempt binding, ordinary-role exposure or reset is
  not reproducible in a disposable instance;
- deletion or two-adapter tests fail, showing policy leakage or an
  unjustified generic seam;
- admitted tasks cannot be graded independently of the writer or cannot
  distinguish propagation, wrong-target and unauthorized-extra-artifact
  outcomes.

If those gates fail, keep Plane research-only and strengthen the current
GitLab/Rocket.Chat evidence instead.  TAC depth should move ahead of Plane
only if a specific, already-supported claim (for example a reproducible
conversation-synthesis failure that the Plane slice cannot test) is both
demonstrated in source/runtime evidence and has a narrower, independently
gradable contract.  WebChoreArena-style native workflow imports remain
references or labeled controls: importing them as the main corpus would trade
WARP's host-owned provenance, exact exposure and reset guarantees for task
volume, and would not show that WARP generated the substantive instances.

## Deferred work and approval boundary

Deferred until the Plane-only result is accepted and evidenced: GitLab-to-Plane
cross-Site generation; any generic multi-Site seam; SuiteCRM onboarding or
WARP-generated SuiteCRM slices; richer/live/additional TAC Sites; and office or
document workflows.  This note is not a specification or ticket DAG.  A
durable ADR, implementation, live benchmark run, or follow-on tickets require
explicit approval after the dependency and card decisions are settled.
