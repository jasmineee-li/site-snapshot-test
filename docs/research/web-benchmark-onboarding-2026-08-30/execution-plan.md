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

CI and agent-DX update: **2026-09-04 (America/New_York)** against
`origin/main` `d7d9a33e`. The provider-independent source DAG #240--#246 is
complete. Three new source-feedback tickets use measured hosted timings:
#255 supplies current slow-node visibility and removes confirmed duplicate
repository traversal only if it consumes material time, #257 depends on that
trace for a two-way core split, and
#256 independently routes the disjoint root gate inside its still-required job.
The exact-route Phase 1 resumption protocol uses two production micro-canaries,
not a no-provider imitation.

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

| Source slice | Tracking issue | Dependency |
| --- | --- | --- |
| Expose real core durations and conditionally replace material duplicate readiness walks while preserving both assertion sets | [#255](https://github.com/jasmineee-li/warp/issues/255) | independent; supplies the current duration trace |
| Split core tests into two measured feature-oriented lanes with three-way matrix capacity and exact selection parity | [#257](https://github.com/jasmineee-li/warp/issues/257) | #255 |
| Route clearly Taskgen-only changes to a successful no-op inside the still-required Root gates job | [#256](https://github.com/jasmineee-li/warp/issues/256) | independent; may develop in parallel with #255/#257 |

The critical delivery path is #255 then #257. Issue #256 is a parallel lane
with no overlapping Taskgen acceptance owner. The two core lanes and existing
remote lane must be allowed to run concurrently; otherwise a two-job
`max-parallel` limit can erase the split's benefit. The first combined target is
an approximately 80--90 second merge-gate critical path, a 45--55% projection
from current hosted measurements rather than a promised result. Each ticket
records its ordinary hosted wall time and stops after at most two focused
repairs; do not create a scheduler grid, repeated timing campaign, third core
shard, larger paid runner, timing manifest or shared virtual environment.

The supporting [agent-DX note](agent-dx-frontier-2026-09-04.md) and
[CI/test acceleration note](ci-test-acceleration-frontier-2026-09-04.md)
record the current-source inspection, primary-source research, counterfactuals
and rejected complexity.

#### E3 provider-boundary resumption check

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
to `main` rather than remaining indefinitely stacked. Review is bounded to two
passes. Runtime work uses separate state directories and existing Run
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
   Anthropic-compatible runners, 140 rows at 20 per family, and the accepted
   `$3,000` run-level backpressure value are recorded on #212. That value is
   operational backpressure, not an automatic hard cap or selection rule.
3. **E3 remains externally blocked:** the configured frozen OpenRouter route
   currently lacks usable provider credit. Once it is usable, run the two
   one-row production micro-canaries above and then resume the exact retained
   Phase 1 request. Do not silently fall back to Anthropic-direct and claim the
   frozen-route result.
4. **E4/E6 remain downstream scientific runs:** they wait for the admitted bank
   and real matched pair and must preserve instance/reset isolation. Mutable
   runs are not parallelized on one reset-sensitive instance.
5. **Paper/publication remains separate:** do not edit the manuscript, submit,
   upload, release or publish without the separately required approval.

Autonomous work may complete #255--#257 and any other already authorized
source tickets with ticket-local locked environments, focused tests, bounded
review and ordinary PR delivery. It must not invent credentials or results,
weaken a safety/runtime check, or treat source/CI success as benchmark evidence.

## 6. Deferred work

Keep the accepted deferrals: Plane and cross-application generation, live NPCs,
multi-round matched rewriting, a universal workflow/binder/auth framework, a
universal semantic judge, and new evidence machinery. Revisit Plane only if E1
demonstrates that ordinary Rocket.Chat exposure is infeasible or scientifically
collapses to message copying.
