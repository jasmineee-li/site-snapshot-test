# Plane-first and GitLab-to-Plane follow-on research

Research baseline: **2026-08-30 (ET)**.  Follow-on review date: **2026-09-01
(ET)**.  This is a read-only source review; no Plane/TAC stack, browser, model,
evaluator, benchmark Run, or live reset was run.  Official web sources and the
current `origin/main` checkout were inspected on 2026-09-01.  A source fact is
not a live-sandbox result, an admitted corpus item, a completed Run, or paper
evidence.

## Current answer

Plane-only multi-record triage remains the smallest scientifically useful TAC
follow-on, but only as a **conditional single-Site study**.  It can test whether
an injected record is read while an agent compares a finite, independently
seeded set and either returns an exact decision or changes only the selected
record.  It cannot establish cross-application generation, general workflow
portability, or TAC-native evaluator correctness.

GitLab-to-Plane is a stronger later result, not a configuration-only extension.
Current WARP validation rejects a multi-Site task, generation compiles one
Site/one benchmark seed call, action-target binding only accepts `same_issue`,
and the local final-state catalog is WebArena-Verified-only.  The TAC fixture
also ships a 2024 Plane fork whose helpers still use `/issues` routes, while
current Plane documentation deprecates those routes.  A cross-Site slice must
therefore wait for a proved Plane-only contract and a route/version canary.

## What current Plane actually exposes

The current official API documentation uses workspace slug plus project UUID
and the `/work-items/` resource:

| Concern | Current official contract | Benchmark consequence |
| --- | --- | --- |
| Authentication | Cloud base URL is `https://api.plane.so/`; API keys are sent in `X-API-Key`, or OAuth bearer tokens with scopes. ([API introduction](https://developers.plane.so/api-reference/introduction)) | Keep API credentials in host seed/reset tooling.  The participant should use a browser account; a successful privileged API call is not exposure evidence. |
| Candidate read | `GET /api/v1/workspaces/{workspace_slug}/projects/{project_id}/work-items/` is cursor-paginated and returns a `results` envelope. ([List work items](https://developers.plane.so/api-reference/issue/list-issues)) | Seed three to five records, preserve their actual IDs, and paginate/read the visible UI.  Current `preview` source rejects `pql`/`filters` on this endpoint, so a task must not assume server-side filtering; the host can provide a small project and the browser must inspect the rows. ([current issue view](https://raw.githubusercontent.com/makeplane/plane/preview/apps/api/plane/api/views/issue.py)) |
| Candidate write | `PATCH /api/v1/workspaces/{workspace_slug}/projects/{project_id}/work-items/{resource_id}/` supports partial fields including `state`. ([Update work item](https://developers.plane.so/api-reference/issue/update-issue-detail)) | A browser state transition must be bound to the selected current-attempt ID and read back.  Never infer success from a click or HTTP 200 alone. |
| State semantics | A State has a stable ID and a `group` in `backlog`, `unstarted`, `started`, `completed`, or `cancelled`; `completed_at` is set when an item enters a completed group. ([State overview](https://developers.plane.so/api-reference/state/overview)) | “Blocked” is not an official group.  If a fixture calls a state *Blocked*, record its actual state ID and group and assert the observed transition; do not assume a universal status vocabulary.  Do not mutate state definitions in the task. |
| Comments | Work-item comments have explicit list/create routes and read/write scopes. ([Comments](https://developers.plane.so/api-reference/issue-comment/list-issue-comments), [create comment](https://developers.plane.so/api-reference/issue-comment/add-issue-comment)) | A comment can be a state-changing sibling, but its parent work-item ID, actor, body, and absence of extra comments need independent readback. |
| Roles | Current `preview` source defines project roles Admin=20, Member=15, Guest=5.  `ProjectEntityPermission` requires an active member and permits mutation to Admin/Member; comment endpoints use a lighter project-member permission. ([roles](https://raw.githubusercontent.com/makeplane/plane/preview/apps/api/plane/db/models/project.py), [permissions](https://raw.githubusercontent.com/makeplane/plane/preview/apps/api/plane/app/permissions/project.py)) | Use an ordinary active Member for the writer and fresh reader.  Treat Guest behavior as version-dependent and do not use it to prove a normal writer can transition state. |
| Identity/readback | Current work-item responses include ID, state, `created_by`, `updated_by`, project, and workspace fields. ([Work-item overview](https://developers.plane.so/api-reference/issue/overview)) | Fixture metadata must retain the exact current IDs, expected state IDs/groups, actor identity, and baseline comments.  A title match alone is not enough. |

The official documentation now warns that `/api/v1/.../issues/` endpoints are
deprecated in favor of `/work-items/`, with support ending **31 March 2026**
([API introduction](https://developers.plane.so/api-reference/introduction)).
The current `makeplane/plane` `preview` source contains both aliases, but that
does not establish that the TAC image does.  Current source also sanitizes or
filters invalid assignee IDs rather than necessarily rejecting the request;
the open upstream report documents the resulting silent drop
([Plane issue #9517](https://github.com/makeplane/plane/issues/9517)).  Plane
fixtures should therefore avoid assignee variation in the first slice, or
verify the returned assignee set rather than treating a 201/200 as proof.
Pages are deliberately out of scope: the public URL module and self-hosted
deployments have documented mismatch reports
([issue #9484](https://github.com/makeplane/plane/issues/9484),
[issue #9511](https://github.com/makeplane/plane/issues/9511)).

## TAC deployment is not current Plane

TAC's main repository has no post-baseline change relevant to Plane: the
latest listed commit is the November 2025 MongoDB image-reference fix
([TAC history](https://github.com/TheAgentCompany/TheAgentCompany/commits/main/)).
Its Plane download script explicitly targets a `v0.22-dev`/August 2024 fork and
the `stable` branch of `TheAgentCompany/plane`
([download script](https://github.com/TheAgentCompany/TheAgentCompany/blob/main/servers/plane/download.sh)).
That fork's latest listed commit is December 2024
([TAC Plane history](https://github.com/TheAgentCompany/plane/commits/stable/)).
TAC's `base_image/common.py` calls `/api/v1/.../issues/` helpers, so a WARP
Plane editor cannot assume that current `/work-items/` documentation matches
the benchmark image.  A version-pinned browser route/readback canary is a hard
gate before admitting any Plane corpus item.

TAC's native `pm-update-plane-issue-from-gitlab-status` task is useful as a
workflow reference: it reads fixed GitLab issues and changes fixed Plane issue
states.  Its evaluator checks trajectory URLs, a fixed project/cycle, and a
completed-state lookup, but not current-attempt actor, freshness, exact
parent/child identity, or extra mutations
([task](https://github.com/TheAgentCompany/TheAgentCompany/blob/main/workspaces/tasks/pm-update-plane-issue-from-gitlab-status/task.md),
[evaluator](https://github.com/TheAgentCompany/TheAgentCompany/blob/main/workspaces/tasks/pm-update-plane-issue-from-gitlab-status/evaluator.py)).
Keep it as a labeled native control, not as WARP exposure or safety evidence.

TAC's reset service POSTs asynchronous per-service resets and waits for health,
while Plane reset restores a fixture.  Health is not an exact-state assertion:
it does not prove that the seeded work items, comments, memberships, or
absence of extra rows are restored
([TAC reset](https://github.com/TheAgentCompany/TheAgentCompany/blob/main/workspaces/base_image/reset.sh),
[API reset endpoints](https://github.com/TheAgentCompany/TheAgentCompany/blob/main/servers/api-server/api-server.py),
[Plane reset Makefile](https://github.com/TheAgentCompany/TheAgentCompany/blob/main/servers/plane/Makefile)).
The host must serialize a Golden-State Reset and verify the fixture itself
before reusing an instance.

## Plane-only slice: claim, boundary, and checks

**Concrete instance.** Host-seed one project with three to five known work
items.  Give each a logical fixture key and distinct state/description facts;
keep the records stable and visible to an ordinary Member.  An information-only
task asks the agent to inspect all records and return an exact object such as
`{selected_key, reason}`.  A state-changing sibling asks it to select one
deterministically and move only that work item to a named existing state (or
post exactly one comment).  A matched carrier pair keeps the benign predicate
fixed while varying the injected record text.

**Claim supported.** This supports a bounded claim about finite-set reasoning
and exposure on a structured, stateful Site: whether injected content changes
the selected record, answer, or exactly targeted state/comment while the agent
must read multiple records.  It can separate propagation, incorrect
conclusions, wrong-target actions, and unauthorized extra artifacts.

**Claim not supported.** It does not show cross-Site relation, universal
workflow binding, arbitrary Plane portability, deletion safety, native TAC
evaluator validity, or general WebChoreArena-style workflow coverage.

**Smallest sufficient checks.** Before any Run, use focused contract tests for
candidate-key/expected-state schemas, exact finite-set answer comparison, and
selected-record binding.  A later narrow sandbox canary must show: (1) all
records and IDs render in one ordinary reader context, including pagination;
(2) the writer can perform exactly one allowed state/comment action; (3) a
fresh reader sees the exact state, parent, actor, and body; and (4) serialized
reset restores every baseline state/comment/membership and no extra artifact.
The first failed check identifies a concrete boundary: missing rows means no
native multi-record claim; missing parent/actor means wrong-target and extra
artifact cannot be distinguished; reset drift invalidates subsequent Runs.

The Plane family should own candidate generation, logical-key mapping, state
predicate, fixture facts, local response/final-state evaluator, and Plane
route/readback details.  It can reuse generic read-surface provenance,
exposure/PVPO, current-attempt evidence, and the existing seed loop that
already preserves per-call results.  It should not introduce a registry or a
universal workflow DSL.

## GitLab-to-Plane later slice

**Claim supported.** After Plane-only evidence exists, a generated cross-Site
family can test whether an agent reads a GitLab record, joins it to a Plane
record by an explicit fixture relation, and performs only the permitted Plane
state/comment update.  The result would support a narrow cross-application
dependency claim, with GitLab and Plane controls to distinguish relation errors
from Site-local failures.

**Claim not supported.** It cannot be called “configuration-only,” cannot reuse
the TAC native fixed-ID evaluator as a WARP score, and cannot imply a generic
multi-Site engine.  The existing generated-task validator requires
`sites == [site_name]` ([validator](../../../packages/warp-taskgen/warp_taskgen/phase_1/novel_task_validation/_impl.py#L370));
the contract-bound compiler emits one `site`, one Site start URL, and a literal
`benchmark: webarena_verified` editor call ([compiler](../../../packages/warp-taskgen/warp_taskgen/phases/phase_1_contract_bound_action_api.py#L247));
and action targets admit only `same_issue` ([targets](../../../packages/warp-taskgen/warp_taskgen/adversarial_actions/action_targets.py#L36)).
The local final-state catalog likewise accepts only WebArena Verified bindings
([catalog](../../../packages/warp-taskgen/warp_taskgen/rewards/final_state_catalog.py#L17)).
These are fail-closed contracts, not switches to flip.

The narrow shared seams revealed by this real consumer are:

* preserve a per-call `logical_record_key` to current-attempt ID map and
  benchmark identity in seed results (the existing editor loop and generic
  `SeedSiteRegistry` remain reusable);
* extend validation/generation to an explicit ordered Site set and dependency
  relation, without exposing IDs or evaluator fields to the model;
* add a selected-record target/evidence binding that carries source Site,
  target Site, parent ID, actor, witness, and post-state; and
* bind a WARP-local evaluator to that benchmark/Site composition while keeping
  vendor dispatch, safety, exposure, reset, and historical scores intact.

GitLab's existing issue read/compare/note behavior can remain feature-local and
unchanged for its leg.  Plane's editor, UI route contract, identity setup,
fixture/readback, and reset adapter remain a separate feature module.  A
cross-Site Run must use an isolated Benchmark Instance and a serialized
Golden-State Reset; a hash can prove fixture compatibility, not that either
Site was actually read or updated.

## Counterfactuals and sequencing

Plane-first becomes scientifically insufficient if a focused source/runtime
check shows that an ordinary Member cannot see all three to five records in one
stable context; if state/comment readback omits target and actor identity; if
the TAC fork cannot be made route-compatible without privileged API shortcuts;
if reset cannot prove exact baseline restoration; or if generated instances
vary only titles while preserving the same predicate, answer, and state.  In
those cases retain Plane as a comparison-only/native reference and choose a
different single-Site family; do not silently downgrade the claim.

Cross-Site would be configuration-only only if the current WARP contracts
already accepted an ordered multi-Site task, emitted per-Site routes and
identities, preserved relation-bound IDs in the current seed attempt, and
selected a WARP-local evaluator/reset path.  The source facts above falsify
that counterfactual at `origin/main` (HEAD `577bb40`, PR #206).  Disabling the
single-Site validator or relabeling TAC data would hide, rather than solve, the
missing scientific evidence.

Recommended sequence:

1. Finish the current GitLab/Postmill/Rocket.Chat/matched-rewriting program and
   keep native TAC tasks as labeled controls.
2. Specify and validate the **Plane-only finite-set triage** family, starting
   with information-only and adding one state-changing sibling only after exact
   readback/reset gates pass.
3. Use its concrete logical-key and selected-target evidence to design a small
   shared seam, then specify GitLab-to-Plane generation and a held-out relation
   cohort.
4. Leave richer TAC synthesis, live NPC/office workflows, and SuiteCRM
   onboarding research-only or comparison-only until their own identity,
   exposure, evaluator, and reset gates are credible.

Evidence ledger for this note: source inspection only; no focused tests were
run here; no live sandbox result; no generated/admitted Plane corpus; no
completed experimental Run; no paper evidence.
