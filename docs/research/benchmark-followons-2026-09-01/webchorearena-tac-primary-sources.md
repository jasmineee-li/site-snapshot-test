# WebChoreArena and TAC follow-on primary-source note

Research date: 2026-09-01 (America/New_York). Repository research baseline:
2026-08-30. This is a source and sequencing note only; it does not authorize
integration work, benchmark execution, or native-task import.

## Current upstream state

The official [WebChoreArena repository](https://github.com/WebChoreArena/WebChoreArena)
still describes 532 curated tasks in three challenge families—Massive Memory,
Calculation, and Long-Term Memory—over four WebArena applications. Its
[main-branch history](https://github.com/WebChoreArena/WebChoreArena/commits/main)
shows a later README-only commit (`542abc5`) on 2026-07-09 and no tagged
[release](https://github.com/WebChoreArena/WebChoreArena/releases). The latest
substantive task/configuration commit remains `e612218` from 2025-08-17. The
current repository therefore provides a stable design reference, not evidence
of a post-2025 task or runtime change.

The official [TheAgentCompany repository](https://github.com/TheAgentCompany/TheAgentCompany)
has no newer main commit than `98b68ef` (2025-11-17) in the current history and
only a [1.0.0 release](https://github.com/TheAgentCompany/TheAgentCompany/releases).
However, the current issue and PR queues contain unresolved operational signals:
[issue #1085](https://github.com/TheAgentCompany/TheAgentCompany/issues/1085)
reports Rocket.Chat channel NPC replies not arriving while direct messages work,
and [PR #1094](https://github.com/TheAgentCompany/TheAgentCompany/pull/1094)
adds a missing Rocket.Chat dependency while noting that tasks sharing a service
cannot safely run in parallel because reset endpoints conflict. These are
post-baseline reasons to keep live-NPC work out of the first authoritative TAC
result and to serialize or isolate any eventual TAC runtime.

## What WebChoreArena is useful for

The [README](https://raw.githubusercontent.com/WebChoreArena/WebChoreArena/main/README.md)
and [BrowserGym task configuration](https://github.com/WebChoreArena/WebChoreArena/tree/main/browsergym)
make the useful abstraction explicit: a finite matrix of workflow dimensions,
with per-task sites, state, intent, required observation, side-effect flags,
and an evaluator. The cross-site data gives concrete examples. In
[`test_cross.raw.json`](https://raw.githubusercontent.com/WebChoreArena/WebChoreArena/main/browsergym/config_files/test_cross.raw.json),
one task reads and sorts GitLab repositories, selects the five most recently
updated, and posts an exact summary to Reddit; other tasks transfer a computed
shopping or Wikipedia result into a Reddit post. The same data marks
`required_wait` and `affects_environment`, and the README/runner instructions
require resetting each WebArena site before a cross-site task. The site-specific
files also show calculation and memory variants on GitLab rather than merely
paraphrased single-site prompts. The [paper](https://arxiv.org/abs/2506.01952)
describes these as 532 curated tasks and explicitly lists simulation and future
online extension as limitations.

This supports a WARP planning claim that a task family can vary along at least
three independently inspectable dimensions—information aggregation (memory),
derived computation (calculation), and transfer across applications
(cross-site/long-term)—while keeping the underlying application set fixed. A
concrete WARP-shaped example would be: read three records in Plane, identify
which are stale, update only those, and leave a fresh record untouched. The
WebChoreArena analogy helps name the aggregation and state-change dimensions;
it does not supply WARP's exposure, current-attempt binding, independent
readback, outcome attribution, or Golden-State Reset contracts.

It cannot establish that imported native tasks are valid WARP corpus items.
WebChoreArena rows contain native WebArena `program_html`/string evaluators and
storage-state assumptions, not WARP-generated provenance, adversarial exposure,
exact resource evidence, or the distinct WARP outcomes (propagation, incorrect
conclusion, wrong-target action, unauthorized extra artifact). Importing those
rows would add prompts and volume, but would not test WARP generation or its
Site-local safety contract. Native tasks should remain labeled references,
diagnostics, or controls. WARP should author substantive instances inside the
finite dimension matrix and preserve WARP-owned carriers and evidence.

The falsifier is structural, not textual: if a proposed expansion changes only
IDs, names, dates, or wording while keeping the same predicate, target/action,
carrier, required evidence, and outcome class, it has increased volume but not
behavioral diversity. A small audit table over those fields is sufficient; a
new global fingerprint registry or universal workflow DSL is not.

## Deeper TAC: bounded claims and evidence burdens

The TAC [evaluation guide](https://raw.githubusercontent.com/TheAgentCompany/TheAgentCompany/main/docs/EVALUATION.md)
requires an LLM key because many tasks use LLM evaluators/NPCs, records all
trajectory steps, starts each task with a reset, and runs 175 task images. The
[Rocket.Chat NPC guide](https://raw.githubusercontent.com/TheAgentCompany/TheAgentCompany/main/servers/rocketchat/npc/NPC.md)
says one NPC responds turn-by-turn; multiple NPCs in one channel all answer the
same agent message rather than talking to one another; fine-grained filtering
requires a custom image. TAC's [reset implementation](https://raw.githubusercontent.com/TheAgentCompany/TheAgentCompany/main/servers/api-server/api-server.py)
and task initialization issue asynchronous, broad service resets. These facts
make TAC a useful stressor, but not a reason to introduce a universal workflow
layer.

| Candidate | Narrow claim it could support | Claim it cannot support | Feature-local owner and smallest evidence gate |
| --- | --- | --- | --- |
| **Richer Rocket.Chat synthesis** | A generated agent can reconcile a small, conflicting or temporal conversation into a bounded decision (for example, choose the current owner and due date from 3–5 messages and cite the decisive message). | General social reasoning, reliable NPC cooperation, or portability to arbitrary TAC sites. | Extend the Rocket.Chat decision contracts/generator/evaluator only. Reuse exact room/thread/message identity, ordinary reader/writer roles, independent authenticated readback, current outcome taxonomy, and reset/exposure checks. Require a seeded known-positive run plus negatives for stale message, wrong actor, wrong thread, and extra artifact. |
| **Live NPC diagnostic/workflow** | If made deterministic enough, sensitivity to dynamic coworker responses and diagnostic coverage for missing/late/ambiguous replies. | A general communication score or authoritative proof from an LLM-generated response. | Keep diagnostic feature-local. Require isolated instance, fresh reset, NPC identity/scenario, channel message identity, timeout/error categories, and a replayable fixed configuration. The open channel-reply issue means this is currently a failure mode to measure, not a first success claim. |
| **Additional TAC Site or TAC cross-app workflow** | Transfer of one explicitly named route/permission/reset contract across another Site after Rocket.Chat/Plane evidence exists. | Generic portability or a reusable multi-Site framework. | Add one Site-local module only after a concrete claim and exact route/readback/reset evidence are specified. Shared seams should be extended only when a real second consumer demonstrates the same need. |
| **Office/document workflow** | Multimodal file-operation safety (for example, extract a PDF table into XLSX and preserve a required value). | Browser-only communication, TAC social reasoning, or WARP Site transfer. | Research-only until a paper claim requires it. TAC's own [office tasks](https://github.com/TheAgentCompany/TheAgentCompany/tree/main/workspaces) use ownCloud, local files, PDF/XLSX/PPTX creation, links, and file permissions; evidence would need file-content semantics, provenance, independent readback, cleanup, and reset in addition to browser traces. |

For richer synthesis, simplification is scientifically harmful if it reduces the
conversation to the existing two-field decision with renamed text: that cannot
separate synthesis from lookup. Conversely, adding NPC turns, arbitrary
channels, or multiple TAC sites before a new predicate is defined would confound
the claim with environment instability. The smallest useful extension is one
additional dependency (such as conflicting owner plus due-date evidence) with
an exact, deterministic evaluator; it should remain a deep Rocket.Chat module.

Native TAC tasks illustrate why WARP should not ingest them as its main corpus.
For example, TAC's [notification evaluator](https://github.com/TheAgentCompany/TheAgentCompany/tree/main/workspaces/pm-send-notification-to-corresponding-user)
uses LLM judgments over named users, while its [channel-message evaluator](https://github.com/TheAgentCompany/TheAgentCompany/tree/main/workspaces/pm-create-channel-message)
matches message text and username without a run-bound message ID/freshness
requirement. The multi-Site backlog task's evaluator also contains hard-coded
URLs and trajectory strings. These are useful diagnostic controls and concrete
failure examples, but importing them would weaken WARP's exact evidence and
current-attempt attribution rather than strengthen it.

## Sequencing and counterfactuals

Plane-only multi-record triage remains the smaller scientifically useful next
slice. It tests selective state changes and information-only siblings in one
Site, reveals the necessary record-selection/readback seam, and can reuse WARP's
existing safety, exposure, reset, and reward machinery. WebChoreArena's matrix
can inform a docs-only family table in parallel. A TAC synthesis extension should
follow the first Plane result (and any GitLab-to-Plane result) unless all of the
following counterfactual evidence appears:

1. The Plane canary cannot provide stable multi-record rendering, exact target
   binding, or isolated reset despite a Site-local fix; and
2. Rocket.Chat's current static family has reached a demonstrated ceiling (for
   example, all admissible instances collapse to the same two-field predicate),
   while a 3–5-message synthesis family yields a new predicate or outcome split
   with exact readback; and
3. TAC service dependencies, reset serialization, and channel delivery are
   validated in an isolated instance (including resolution or containment of
   the issues above).

If condition (2) is false—generated instances differ only in wording/volume—or
if the exact-message/readback gate is false, keep deeper TAC research-only and
do not move it ahead of Plane. A live NPC success alone is not sufficient: it
would show one runtime path, not a new WARP behavioral dimension, unless the
controlled comparison demonstrates a distinct predicate and outcome burden.

Parallel work can therefore be partitioned without overlapping edits:

- a docs-only WebChoreArena dimension/coverage matrix;
- a Rocket.Chat static-synthesis claim/evidence note (no runtime changes);
- a separate, later TAC reset/NPC feasibility gate using isolated Benchmark
  Instances and serialized Golden-State Reset; and
- office/document source inspection kept comparison-only.

The critical path is Plane-only generation, exact selective readback, and reset
evidence; then cross-Site GitLab-to-Plane if its second consumer exposes a real
shared seam; only then any TAC depth justified by a new claim. The stop rule is
to reverse an expansion that adds rows or prose without a new structural
predicate, action/target relationship, evidence requirement, or outcome class.
