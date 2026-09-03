# Plane task-family portfolio for the next WARP slice

**Review date:** 2026-09-02
**Research baseline:** 2026-08-30 (the accepted web-benchmark-onboarding notes)
**Scope:** source and documentation research only. No Plane deployment, browser/model run, benchmark generation, corpus admission, or paper result was performed.

## Bottom line

Plane remains a scientifically useful second TAC-oriented Site, but the useful
unit is a small, finite decision world, not a native-task importer. One
feature-local Plane module can support a portfolio of information-only and
state-changing families while keeping each task visually bounded at three to
five records. The first state-changing family should still be **selective
state transition** (read a finite candidate set, choose by an explicit
predicate, update exactly that record). The next scientifically distinct
families are finite inventory/reconciliation, anomaly detection, deterministic
ranking, a selected-record comment, and then sparse batch state updates.
Assignment, labels, cycles/modules, parent links, and project/issue creation
should remain deferred or comparison-only until their ordinary-role write,
readback, and exact reset contracts are demonstrated.

This ordering tests a real claim—multi-record evidence plus selective,
target-bound persistence on a second Site—without requiring a universal
workflow language. A different title, wording, record order, or generated
field value is a **variant**, not a new archetype, when the predicate, action
graph, expected output, and evidence obligations are unchanged.

## Current Plane and WARP facts

The current Plane API documents work items as records with stable IDs and
fields including `name`, description, priority, dates, `sequence_id`, state,
assignees, labels, project, parent, and audit fields; the state API models the
groups `backlog`, `unstarted`, `started`, `completed`, and `cancelled`.
The documented list route is `GET /api/v1/workspaces/{workspace_slug}/projects/{project_id}/work-items/`
and the partial update route is
`PATCH .../work-items/{resource_id}/`. Both require project work-item scopes.
See the official [work-item overview](https://developers.plane.so/api-reference/issue/overview),
[list-issues reference](https://developers.plane.so/api-reference/issue/list-issues),
[update-issue-detail reference](https://developers.plane.so/api-reference/issue/update-issue-detail),
and [state overview](https://developers.plane.so/api-reference/state/overview)
(all accessed 2026-09-02).

The API introduction specifies `X-API-Key` or OAuth bearer authentication and
deprecates the old `/issues/` spelling in favor of `/work-items/`, with support
for the old route ending 2026-03-31. That is a version risk for the TAC fork,
not a reason to add an abstraction in advance; verify the actual sandbox
route before admission. See [Plane API introduction](https://developers.plane.so/api-reference/introduction)
(accessed 2026-09-02).

Current `origin/main` (commit `2f2ce2b4`, inspected 2026-09-02) has no Plane
adapter. It does have useful but deliberately narrow building blocks:

* `packages/warp-taskgen/warp_taskgen/seeding/_impl.py` records one result per
  editor call, including `logical_record_key`, read surfaces, created
  resources, and write tokens; the seed registry remains immutable per run.
* `phase_1/gitlab_compare_decide.py` and its binding/reward siblings show the
  intended feature-local pattern: a typed finite world, generated facts and
  predicate, an attempt-local logical-to-physical map, and exact structured
  response/final-state evaluation. The GitLab world is exactly three records;
  this is a precedent for a bounded Plane world, not a generic multi-Site
  contract. See the [current GitLab comparison module](https://github.com/atla-benchmark/warp/blob/main/packages/warp-taskgen/warp_taskgen/phase_1/gitlab_compare_decide.py)
  (repository source inspected at `origin/main`, 2026-09-02).
* `phase_1/novel_task_validation/_impl.py` still enforces one `site` per
  generated task (`sites == [site_name]`); cross-Site generation therefore
  needs a real validator/compiler/evidence change later. It is not
  configuration-only. The generic phase-1 action compiler also emits one
  Site and one editor call and is not a multi-record decision binder.
* Existing exact response comparison, exposure/PVPO and current-attempt
  resource binding, Run Artifacts/provenance, safety checks, and reset
  orchestration can be reused where their owner is genuinely the same. No
  universal semantic judge, workflow DSL, or generic target registry is
  warranted by a single Plane consumer.

The accepted [workflow frontier](../web-benchmark-onboarding-2026-08-30/workflow-frontier.md)
already distinguishes finite-set comparison from select-then-follow-up and
warns that changed prose is not behavioral diversity. The current TAC and
cross-Site research in this directory records the older `/issues/` fork and
its reset limitations; those facts remain gates rather than live evidence.

## Archetypes versus variants

An **archetype** changes at least one of the decision dependency, allowed
side-effect graph, expected-output shape, or independently observable evidence
needed to attribute success and safety. A **variant** changes only wording,
record titles, key/order, non-decisive facts, threshold values, or which
logical record is decisive while preserving that graph. Generate many variants
inside an archetype; report counts by archetype and avoid counting templates as
behavioral diversity.

For example, “return the records whose state is started and priority is high”
and “return the records whose state is backlog and priority is urgent” are one
set-valued filter archetype if they have the same response and evidence
contract. “Return an ordered ranking using a tie-breaker” is a different
archetype because order is now observable. “Move the selected record to Done”
is different again because it adds a persistent state transition and
wrong-target/extra-action failure modes.

## Proposed Plane portfolio

The examples use three to five records with stable logical keys. Names and
facts are generated; the reader sees the finite list and exact details needed
for the predicate. “Reset” below means an exact Golden-State Reset and
postcondition check, not merely a healthy process or a hash.

| Family (disposition) | Concrete minimum example and attacker objective | Claim supported / cannot support | Evidence, identity, and reset gate |
| --- | --- | --- | --- |
| **Finite inventory / reconciliation** (information-only, first canary) | Read 3–5 work items and return an exact table of logical key, state group, priority, and target date. Objective: omission, fabricated fact, or propagation of one record’s fact to another. | Tests structured multi-record reading and aggregation. It cannot show state-changing safety, target binding, or cross-Site portability. | Independent ordinary reader sees every row and required details; exact structured-response comparison; verify the baseline fixture before/after (no mutation). If list/detail rendering hides a required field, narrow the contract rather than infer it. |
| **Cross-field anomaly / consistency set** (information-only) | Report every item whose state/date/priority combination violates a generated consistency rule, including the empty-set case. Objective: false positive/negative, missed member, or wrong logical key. | Tests relational reasoning over records and negative evidence. It cannot support ranking or mutation claims. | Fixture contains both known-positive and known-negative records; evaluator compares an exact set and reason code. Use state, priority, and dates before assignees. Independent readback is enough; no write reset. |
| **Deterministic ranking** (information-only) | Return all candidate keys ordered by `(state-group precedence, priority, target date, key)` with a generated tie. Objective: order inversion, tie mishandling, or dropped record. | Tests a total-order computation, not merely filtering. It cannot establish persistent action safety or a semantic judge. | Exact ordered-list response and tie fixture; record the comparator in the Run Definition. If the UI cannot expose all comparator fields consistently, defer rather than silently change the rule. |
| **Comment-history disposition** (information-only, defer until readback is proven) | Inspect an item’s existing comments and report the latest confirmed disposition/actor/time for a finite set. Objective: stale-comment selection, actor confusion, or unsupported conclusion. | Tests temporal/actor evidence within one Site. It cannot claim external communication or NPC reliability; native TAC’s RocketChat contact task is a different two-Site workflow. | Need a documented comments-list route and UI readback with parent item, actor, timestamp, and body; fixture must contain conflicting older/newer comments and an empty case. Baseline and exact readback are required; defer if TAC only exposes privileged or unstable data. |
| **Selective state transition** (state-changing, **lead**) | Read 3–5 items; choose the unique item satisfying a state/priority/date predicate; move only it to a named existing state (for example, `started`), leaving all others untouched. Objective: wrong-target action, wrong-state action, unauthorized extra update, or propagation. | Directly supports multi-record evidence-to-action and selective persistent safety on a second Site. It cannot show batch scalability, cross-Site generation, or assignment correctness. | Active ordinary **Member** writer and fresh ordinary reader; current-attempt logical→physical binding; before/after state ID and group for every record; actor and mutation count; no extra artifacts. Serialize Golden-State Reset and assert all five baseline states after each Run. Avoid destructive deletion and avoid creating new states in the first slice. |
| **Selected-record comment follow-up** (state-changing sibling, next) | Apply the same finite predicate, then post one public comment on the selected item containing an exact token/reason. Objective: wrong parent, wrong body, wrong actor, duplicate/extra comment, or omitted comment. | Adds an independently observable artifact while reusing selection/binding. It cannot prove state-transition semantics unless paired as a separate task family. | Comment-create route/UI, parent ID, exact body/token, actor, and count readback; public visibility and ordinary writer/reader checks; exact reset deletes or restores fixture comments only through a sandbox reset, never by relying on model deletion. |
| **Sparse batch state update** (state-changing, after lead) | Among five items, move exactly every item satisfying a predicate (for example, stale + open) to one existing state, leaving non-matches unchanged. Objective: overreach, omission, duplicate update, or wrong-set attribution. | Tests set-valued target binding and bounded batch safety. It cannot support arbitrary-scale workflow claims or justify a universal action engine. | Full before/after map for all records, per-record state IDs/groups and mutation audit, exact changed-set equality, independent reader, and serialized reset. Keep the world at 3–5 records; scale by worlds/tasks, not unbounded rows. |
| **Metadata/assignment correction** (deferred; comparison-only until gates pass) | Assign a selected item to a specified ordinary member or set a label/priority/date from evidence. Native controls include batch assignment and status/sprint tasks. Objective: wrong identity, silent write drop, unauthorized role use, or wrong field. | Could test field-level selective updates, but not safely on the first Plane slice. It cannot be admitted from native-task success alone. | Requires ordinary-role membership fixture, exact user/label IDs, write/readback symmetry, and reset of every field. Current Plane issue [#9517](https://github.com/makeplane/plane/issues/9517) reports assignee serialization silently filtering invalid/guest IDs; this is a direct blocker until a known-positive ordinary-member path is measured. |
| **Cycle/module/parent/link update or project/issue batch creation** (deferred/native control) | Move records between sprints/cycles, set parent links, or reproduce TAC’s `pm-assign-issues` project + three issue creation. Objective: relationship/identity errors and extra artifacts. | These are useful controls or later claims about relationship mutations, not evidence for the next selective-state result. They add another identity graph and reset surface. | Require exact relationship readback, ordinary permissions, deterministic fixture IDs, and complete cleanup/reset. TAC evaluators for `pm-assign-issues`, `pm-update-plane-issue-from-gitlab-status`, and `pm-copy-plane-issues-to-gitlab` use fixed titles/URLs and trajectory checks, so they are references rather than WARP evidence: [assign issues task](https://raw.githubusercontent.com/TheAgentCompany/TheAgentCompany/main/workspaces/tasks/pm-assign-issues/task.md), [evaluator](https://raw.githubusercontent.com/TheAgentCompany/TheAgentCompany/main/workspaces/tasks/pm-assign-issues/evaluator.py), [Plane→GitLab task](https://raw.githubusercontent.com/TheAgentCompany/TheAgentCompany/main/workspaces/tasks/pm-copy-plane-issues-to-gitlab/task.md), [evaluator](https://raw.githubusercontent.com/TheAgentCompany/TheAgentCompany/main/workspaces/tasks/pm-copy-plane-issues-to-gitlab/evaluator.py). |

### What one deep Plane module owns

Keep a feature-local module (for example, `phase_1/plane_work_items.py` and
local binding/evaluator siblings) responsible for:

* typed logical records and generated worlds (candidate facts, predicate,
  tie-breaker, expected set/order, and allowed state/comment operations);
* Plane transport/UI routes, workspace/project/state/resource identity, and
  ordinary Member reader/writer setup;
* deterministic fixture creation, exact readback, current-attempt binding, and
  exact final-state/response grading for the families above;
* the local evidence graph: all candidate baselines, changed-set equality,
  selected parent, actor, body, and extra-artifact checks.

Inventory, anomaly, ranking, and selective transition share the finite-world
record and exact-response core. The selected comment and sparse batch families
may share the local selector and binding, but should retain their own allowed
action/evaluator contracts so a comment cannot be mistaken for a state update.
Only after two concrete consumers require it should a narrow shared seam (for
example, a per-record before/after map or selected-record target binding) move
out of the module. Do not create a registry, workflow DSL, semantic judge, or
generic cross-Site action engine from this portfolio.

### Reuse versus extension

Reuse unchanged where ownership matches: the seed loop and immutable
`SeedSiteRegistry`; per-call `logical_record_key` result/provenance; independent
exposure/PVPO and current-attempt binding checks; Run Artifacts and resume
compatibility; exact list/multiset/structured response comparison; existing
public-comment reward shape; and the reset runner as an orchestration wrapper.

The narrow extensions are feature-local first: a Plane logical-record
world/serializer, a list/detail read contract, selected or set-valued binding,
state/comment transport, and exact Plane reset postconditions. The existing
single-Site novel-task validator and one-call action compiler must not be
disabled or pretended to handle this; a later GitLab→Plane family will require
an explicit multi-Site validator/compiler/evidence change after Plane proves
the seam.

## Bounded scaling and evidence ledger

Visual boundedness is per task, not a corpus cap. Keep each generated world at
three to five records and expose a finite list plus exact detail pages. Scale
the corpus by independent worlds, decisive logical keys, predicate values,
negative controls, and Run repetitions. Keep one orthogonal change per paired
variant (for example, flip only the decisive state) so a regression is
diagnosable. Do not vary prose alone or use pagination beyond the verified
visible window; if the UI cannot show the finite world, fail closed.

For every state-changing family, the minimum evidence packet is:

1. exact precondition fixture/readback for every logical key;
2. independent authorized exposure and current-attempt logical→physical map;
3. action trace with actor, parent, field, and mutation count;
4. exact postcondition/readback for changed and unchanged records;
5. serialized Golden-State Reset and post-reset baseline check.

This note contains source inspection only. Focused known-positive/negative
tests, a Plane sandbox smoke result, generated/admitted corpus size, completed
Runs, and paper evidence are all **pending** and must be reported separately.

## Counterfactuals and stop/reverse conditions

* If a proposed pair changes only wording, titles, order, or non-decisive facts,
  collapse it into one archetype; otherwise the portfolio would overstate
  behavioral diversity.
* If an ordinary reader cannot see every candidate and predicate field in the
  actual TAC/Plane UI, do not claim multi-record reasoning. Reduce the family
  to the visible contract or stop it.
* If before/after readback cannot distinguish the selected physical record from
  an aggregate or stale ID, selective or batch state claims are invalid; keep
  the work information-only/comparison-only.
* If a state PATCH succeeds in transport but the UI/readback does not expose
  the exact state ID/group, actor, and unchanged records, reverse to a
  read-only family. A hash or process health is not behavioral proof.
* If comments lack stable parent, actor, timestamp, and body readback, defer
  comment-history/follow-up rather than weakening evidence.
* If assignee/label/cycle/parent writes require privileged roles, silently
  drop invalid identities, or cannot be reset exactly, keep those families
  deferred. Plane issue [#9517](https://github.com/makeplane/plane/issues/9517)
  is a concrete example of the failure this gate prevents.
* If generated candidate worlds do not guarantee a unique expected answer (or
  a specified empty/set-valued answer), the family is not an archetype yet;
  repair the generator or stop it.
* If a local Plane result cannot be reproduced across isolated Benchmark
  Instances with serialized reset, do not advance to GitLab→Plane. A native
  TAC task passing a fixed-title evaluator is not a WARP Run or paper result.

## Sequencing recommendation

1. Verify the actual Plane/TAC route and ordinary Member read/write path with a
   tiny no-model canary, then build the feature-local finite-world contract.
2. Admit information-only inventory/anomaly (and optionally ranking) controls
   with exact responses; they establish read evidence before mutation.
3. Run the **selective state transition** vertical slice as the next
   specification: 3–5 records, one deterministic winner, one existing state,
   exact unchanged-record checks, and serialized reset.
4. Add selected-record comment follow-up only if its parent/actor/body
   readback is equally strong; add sparse batch state updates after selective
   updates show no overreach/identity failures.
5. Keep assignment/metadata and relationship/creation families deferred. Begin
   GitLab→Plane generation only after Plane local generation, execution,
   scoring, and reset gates pass; that later claim needs an explicit shared
   multi-Site change, not configuration.

The recommendation reverses to information-only or comparison-only if any
runtime gate above fails. It stops entirely if no ordinary-role, exact
readback, and reset path can be established without weakening WARP’s existing
safety checks.

## Primary references

* Plane official API: [introduction](https://developers.plane.so/api-reference/introduction), [work-item overview](https://developers.plane.so/api-reference/issue/overview), [list](https://developers.plane.so/api-reference/issue/list-issues), [update](https://developers.plane.so/api-reference/issue/update-issue-detail), and [states](https://developers.plane.so/api-reference/state/overview) (accessed 2026-09-02).
* TheAgentCompany native task source/evaluators: [pm-ask-issue-assignee-for-issue-status-and-update-in-plane](https://raw.githubusercontent.com/TheAgentCompany/TheAgentCompany/main/workspaces/tasks/pm-ask-issue-assignee-for-issue-status-and-update-in-plane/task.md), [evaluator](https://raw.githubusercontent.com/TheAgentCompany/TheAgentCompany/main/workspaces/tasks/pm-ask-issue-assignee-for-issue-status-and-update-in-plane/evaluator.py), [pm-update-plane-issue-from-gitlab-status](https://raw.githubusercontent.com/TheAgentCompany/TheAgentCompany/main/workspaces/tasks/pm-update-plane-issue-from-gitlab-status/task.md), [evaluator](https://raw.githubusercontent.com/TheAgentCompany/TheAgentCompany/main/workspaces/tasks/pm-update-plane-issue-from-gitlab-status/evaluator.py), and [pm-monitor-new-bug-issues](https://raw.githubusercontent.com/TheAgentCompany/TheAgentCompany/main/workspaces/tasks/pm-monitor-new-bug-issues/task.md) (accessed 2026-09-02).
* WARP accepted research: [workflow frontier](../web-benchmark-onboarding-2026-08-30/workflow-frontier.md), [Plane/cross-Site primary sources](plane-and-cross-site-primary-sources.md), and [WebChoreArena/TAC primary sources](webchorearena-tac-primary-sources.md) (baseline 2026-08-30; local notes inspected 2026-09-02).
* WARP current source inspected at `origin/main` commit `2f2ce2b4` on 2026-09-02: `packages/warp-taskgen/warp_taskgen/seeding/_impl.py`, `phase_1/gitlab_compare_decide.py`, `phase_1/gitlab_compare_decide_binding.py`, `phase_1/gitlab_compare_act.py`, `phase_1/novel_task_validation/_impl.py`, and `rewards/agent_response.py`.
