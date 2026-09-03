# Browser-benchmark alternatives for the next WARP follow-on

**Research date:** 2026-09-02 (America/New_York)
**Repository research baseline:** 2026-08-30
**Scope:** primary-source screening and sequencing only. No integration,
benchmark execution, corpus generation, or infrastructure work was performed.

## Decision in brief

No external benchmark or Site found here is a better canonical next
specification than the proposed **Plane-only multi-record triage and selective
state updates**. It is the smallest new WARP result that combines a second
Site, ordinary-user writes, exact target/readback evidence, persistent state,
and a genuinely new IPI outcome surface. The current WebArena, WorkArena, and
VisualWebArena releases provide useful controls and implementation references,
but importing their native tasks would add wording or volume without proving
WARP generation, current-attempt exposure, or WARP's outcome taxonomy.

The strongest *later marketplace candidate* is a deeper VisualWebArena
**Classifieds** slice, not Shopping or Shopping Admin. WARP's August 2026
Classifieds canary already proved a normal participant reply, exact independent
readback, Painted Visibility, and Golden-State Reset for one task. Shopping has
a maintained 2026 container distribution and legacy WARP plumbing, but WARP's
April 2026 history records pending-review visibility, forced approval, and
benign paths that bypassed product reviews. Shopping is therefore a research
hypothesis for a named moderated-commerce claim, not the default breadth lane.

WorkArena++ has the richest enterprise planning/composition story, but its
default task identity is an `admin` user on a remote ServiceNow instance and
its isolation is temporary-user teardown rather than a demonstrated WARP
Golden-State Reset. VisualWebArena Classifieds has a reset endpoint and
ordinary-looking logged-in tasks, but its state-changing evaluators use fixed
IDs/`last` selectors and visual/image checks; it is a multimodal grounding
reference, not current WARP IPI evidence. OpenApps and TimeWarp are useful
reliability/variation controls, not authenticated multi-actor persistent
Sites. Knows (Google Workspace) remains an open BrowserGym pull request with
no public paper, repository, or dataset and requires Google credentials.
WebArena-Infinity is a 2026 environment-generation reference, not a proven
ordinary-user safety benchmark.

## Screening gates and what counts as evidence

The WARP target is narrower than “a benchmark has many tasks.” A candidate
must have, or make a credible path to:

1. browser-only execution against a persistent web state;
2. an ordinary writer and (where required) an independently authenticated
   reader, with exact resource exposure and current-attempt binding;
3. deterministic or otherwise auditable before/after readback for the targeted
   resource and all unchanged controls;
4. an isolated, repeatable reset that can satisfy WARP's Golden-State Reset,
   not merely a process restart or a hash;
5. a threat model in which malicious/incorrect content can be distinguished
   from ordinary application behavior; and
6. a substantive WARP-generated predicate/action graph whose variants do not
   collapse to renamed wording.

The native benchmark's task count, focused tests, a live sandbox success, a
generated/admitted corpus count, completed Runs, and paper evidence remain
different categories. This note reports only upstream source facts. Any
candidate that fails the gates is retained as a labelled reference or control,
not admitted as a WARP corpus source.

## Delta from the 2026-08-30 baseline

The 2026-08-30 WARP notes already settled that WARP-generated tasks remain the
main corpus, native tasks are references/diagnostics/controls, and no universal
workflow or semantic-judge engine is a prerequisite. Since that baseline, the
following upstream facts are current on 2026-09-02:

* The current [WebArena-Verified](https://github.com/ServiceNow/webarena-verified)
  documentation reports 812 verified tasks, a 258-task hard subset,
  deterministic response and network-trace evaluation, and February 2026
  optimized Docker images with auto-login headers and a single Map container.
  Its 2026 issue queue still includes a GitLab/Reddit initialization failure
  and an unsatisfiable network evaluator report
  ([issues](https://github.com/ServiceNow/webarena-verified/issues)).
* [BrowserGym v0.14.3](https://github.com/ServiceNow/BrowserGym/releases/tag/v0.14.3)
  (20 January 2026) lists WebArena-Verified, OpenApps, TimeWarp, and a
  WebArena-Lite task subset in the default ecosystem. As of this review it has
  20 open and 249 closed pull requests; [#397](https://github.com/ServiceNow/BrowserGym/pull/397)
  (Knows) and [#398](https://github.com/ServiceNow/BrowserGym/pull/398) (an
  external ClawBench reference) are still open. Therefore neither is evidence
  of a merged BrowserGym Site integration, and WebArena-Lite adds no new Site
  or WARP exposure/reset contract.
* [TimeWarp](https://arxiv.org/abs/2603.04949) was submitted 5 March 2026 and
  is linked from BrowserGym's current metadata. It varies three web apps over
  six UI versions, but its documented goals are answer/navigation tasks rather
  than ordinary-user writes with an exposure/readback threat model.
* [WebArena-Infinity](https://github.com/web-arena-x/webarena-infinity) is a
  current 2026 repository that generates self-contained apps, tasks, and
  verifiers from documentation. Its README explicitly gives privileged code
  agents access to environment internals and asks users to run generated code;
  it does not provide WARP's actor separation, adversarial exposure, or reset
  evidence.
* [ClawBench](https://github.com/TIGER-AI-Lab/ClawBench) is now a substantial
  live-site benchmark (281 tasks across 163 sites in the current README), and
  BrowserGym #398 proposes only an external documentation link. ClawBench's
  interceptor prevents a final live request from reaching the site and records
  five trace layers, but its own README treats live-site drift as part of the
  target; it is not a resettable persistent sandbox for WARP.

Maintenance is uneven. [VisualWebArena's commit history](https://github.com/web-arena-x/visualwebarena/commits/main)
currently ends on 9 November 2024 and its README still lists reset scripts as a
TODO; [WebChoreArena's history](https://github.com/WebChoreArena/WebChoreArena/commits/main)
has a 9 July 2026 README-only update but no new task/runtime release. WorkArena
has February 2026 commits but an open deletion-validation issue, while
[OpenApps](https://github.com/facebookresearch/OpenApps/commits/main) remains
actively changed through 24 August 2026. These activity signals affect
maintenance risk, not the WARP claim: active code without an actor/exposure
contract is still not a WARP Site, and a stable reference without reset/readback
proof is still not an admitted corpus source.

These are maintenance and comparison updates, not grounds for reopening the
settled WARP corpus policy. A direct main-branch history check on 2026-09-02
found no commits after the 2026-08-30 research baseline in WorkArena,
OpenApps, WebArena-Verified, or WebArena-Infinity. WebChoreArena's latest
commit remains the 2026-07-09 README-only change described below.

## Candidate matrix

| Candidate | Narrow claim supported | What it cannot support for WARP | Access/reset/evaluator burden | Disposition |
| --- | --- | --- | --- | --- |
| **Plane-only multi-record slice** (WARP proposal) | Generated finite-world evidence-to-action: choose one (then a bounded set) of work items from 3–5 records and update only those, with information-only siblings. | Not cross-Site generation or arbitrary workflow portability. | One Site-local ordinary writer/reader, exact list/detail readback, per-record before/after map, serialized Golden-State Reset. | **Rank 1; next specification.** |
| **VisualWebArena Classifieds** | Generated public-marketplace UGC and visual/multi-listing decisions; WARP has one exact opt-in carrier canary. | The one-task opposite-label canary does not prove a generated portfolio or final-state action grading; multimodal claims need separate control. | Existing WARP regular-writer/fresh-reader/exact-ID/Painted-Visibility/reset evidence reduces risk; selected-listing reply grading and redistribution remain open. | **Rank 2 later/conditional.** |
| **WebArena Shopping customer** | Customer-facing e-commerce retrieval plus a generated safe mutation (for example, update a cart/wishlist fixture) if WARP adds a local contract. | No current natural WARP exposure oracle; purchase/order semantics may overlap existing mutation; no cross-Site claim. | Optimized Docker/env-ctrl and `X-M2-Customer-Auto-Login`; review moderation, necessary encounter, fixture readback, and side-effect containment need proof. | **Research hypothesis, not default breadth lane.** |
| **WorkArena / WorkArena++** | Enterprise UI component coverage and compositional planning/reasoning (682 composite tasks; 19,912 WorkArena-L1 instances). | Not WARP IPI safety, ordinary-role least privilege, exact exposure, or guaranteed reset. | Gated ServiceNow instance, temporary user creation/deletion, REST/frontend validators; default `user_roles=["admin"]`, teardown best-effort. | **Research/control, not next Site.** |
| **WebArena Shopping Admin** | Admin CMS workflows and batch product/content management. | Fails ordinary-user threat-model gate and adds privileged destructive state. | Admin auto-login and bulk operations; reset/evaluator burden high. | **Exclude from canonical WARP path.** |
| **WebArena GitLab/Reddit** | Existing WARP Sites and comparison controls; WebArena-Verified adds deterministic trace evaluators. | Does not add environment extensibility; native tasks do not prove WARP generation. | Existing WARP ownership; WebArena-Verified still reports 2026 image/init issues. | **Keep current/control.** |
| **WebArena Wikipedia/Map** | Retrieval/navigation and visual/map interaction. | No ordinary-user write path or persistent IPI state. | Read-only data/image volumes; no reason to add a Site for the target claim. | **Exclude.** |
| **WebChoreArena** | A finite workflow matrix for massive-memory, calculation, long-term-memory, and cross-site chores on WebArena. | Native hand-curated rows cannot prove WARP generation, exposure, current-attempt binding, or WARP outcome attribution. | Reuses WebArena reset instructions; 532 rows are references, not WARP provenance. | **Source/reference/control only.** |
| **OpenApps** | Reliability across configurable appearance/content and transparent Python ground-truth state; lightweight parallel app variants. | No ordinary authenticated actors, malicious payload/exposure oracle, or Site realism comparable to WARP. | Single CPU and isolated deployments are attractive; WARP would still need a browser-only actor/evidence wrapper. | **Research/control; not canonical.** |
| **TimeWarp** | Robustness/generalization to six temporal UI versions across Wiki, News, and Webshop. | No write-path safety, actor separation, or resettable persistent-state claim; mostly answer verifiers. | Three app servers, conda/data setup, deterministic answer verifiers plus optional judge; no WARP state contract. | **UI-variation control only.** |
| **Knows (BrowserGym PR #397)** | Potential long-horizon Docs/Sheets/Slides authoring with checkpoint rewards. | Current upstream status is open; public paper/repo/dataset and reset/ordinary access are not available. | Google account plus Cloud service account; Workspace API grading and cloud-state cleanup are unverified. | **Deferred research; do not onboard.** |
| **WebArena-Infinity** | Automated generation of realistic apps/tasks/verifiers from manuals; a useful environment-extensibility comparison. | Its privileged code-generation pipeline is not WARP-generated IPI data, and README gives no ordinary-role/exposure/reset threat model. | 19 generated environments, browser-use agents, AWS/Claude credentials, generated code risk; very high audit burden. | **Research-only reference.** |
| **ClawBench (BrowserGym #398 external reference)** | Live consumer-site completion with browser-only traces and intercepted irreversible requests. | Live-site drift is intentional; no Golden-State Reset, WARP exposure, current-attempt binding, or generated persistent fixtures. | Isolated containers and five-layer recordings, but real websites, disposable accounts, and post-session LLM judging. | **Comparison-only; not a WARP Site.** |

## Wrapper, generation, exposure, reset, and realism consequences

No candidate arrives with a drop-in WARP wrapper. BrowserGym task classes and
WebArena/VisualWebArena config files bind native task IDs, storage cookies,
fixed URLs, or benchmark-specific evaluators; they do not supply WARP's
current-attempt logical-to-physical map, independent authorized exposure, or
outcome attribution. A wrapper around one of these systems would therefore be a
new feature-local owner, not a registration entry:

* **Plane:** own generated finite records, selector/predicate, ordinary member
  identities, list/detail readback, allowed state updates, and exact reset
  postconditions. Reuse WARP safety, provenance, exposure checks, and reward
  machinery only where their existing owners match. Extract a narrow
  before/after or target-binding seam only after Plane plus GitLab-to-Plane
  provide two real consumers.
* **Shopping or Classifieds:** own product/listing fixture generation,
  customer/seller actor setup, reversible mutation, exact resource readback,
  and reset. Native URL/program-HTML/image checks are controls, not WARP
  grading. Do not share a generic “web record” or “workflow” abstraction before
  a second WARP consumer demonstrates the same contract.
* **WorkArena, OpenApps, TimeWarp, Knows, WebArena-Infinity, and ClawBench:**
  keep source inspection and comparison adapters outside the admitted WARP
  corpus. Their realism (enterprise UI, app variation, temporal UI, generated
  apps, or live production sites) is a different axis from WARP's ordinary-user
  IPI safety. Admitting them would require a new source-backed exposure and
  reset owner, not a configuration flag.

The realism trade-off is explicit: Plane is less visually broad than
WorkArena++ or Classifieds, but it can make a falsifiable claim about selective
state change under an ordinary role. A visually richer Site is only stronger
if it preserves that claim and adds a distinct visual predicate/evidence
burden; otherwise it adds screenshots and wording without behavioral diversity.

## Detailed assessments

### 1. Plane remains the smallest useful new result

The WARP proposal is a finite project containing three to five known work-item
records in distinct states. An information-only sibling asks for an exact set
or ranking; the lead state-changing family selects a unique record satisfying
a generated predicate and moves only that record to an existing state. A
concrete instance is: “From five records, move the only item that is blocked,
high priority, and past its target date to `started`; leave every other item
unchanged.”

This has a new dependency graph—multi-record evidence → selected logical key →
one persistent state transition → unchanged-record proof—without requiring a
global workflow language. The evaluator can attribute omission, wrong-target,
wrong-state, propagation, and unauthorized-extra-update outcomes directly.
The feature-local Plane module should own the logical-record world, route and
state identities, ordinary reader/writer setup, exact before/after map, and
reset postcondition. WARP's existing safety, exposure, provenance, current-
attempt binding, outcome taxonomy, and Run Artifacts should be reused only
where they already own that behavior. The later GitLab-to-Plane path must not
be described as configuration-only: the current novel-task validator assumes a
single Site, so a real multi-Site validator/compiler/evidence change would be
needed after the Plane local result.

The deletion test is decisive: remove multi-record selection and this family
collapses into an existing single-record mutation; if so, the new claim was not
actually delivered. The two-adapter test is also required: only after Plane and
the later GitLab-to-Plane consumer demand the same before/after/binding seam
should a narrow shared seam be extracted. This avoids a speculative registry or
universal workflow engine.

### 2. WebArena customer Shopping is a conditional research hypothesis

The current [WebArena-Verified README](https://github.com/ServiceNow/webarena-verified#readme)
lists six environments—shopping, shopping_admin, reddit, gitlab, wikipedia,
and map—and documents 812 verified tasks with deterministic response and
network-trace evaluation. Its 2 February 2026 announcement says optimized
Docker images include auto-login headers and a single Map container. The
[Shopping guide](https://servicenow.github.io/webarena-verified/environments/shopping/)
describes a Magento storefront with an `X-M2-Customer-Auto-Login` header;
the [Shopping Admin guide](https://servicenow.github.io/webarena-verified/environments/shopping_admin/)
uses a separate admin header and panel.

Shopping customer therefore has a plausible ordinary-user browser route. A
WARP-generated candidate could, for example, read a three-item cart fixture,
choose the only item satisfying a quantity/price rule, and update a reversible
cart field. But the current benchmark's reference tasks and evaluators are
not evidence of WARP's independent exposure: WARP would need a writer and
independent reader identity, exact product/order IDs, a bounded mutation whose
side effects cannot charge or ship, and a reset that proves every fixture row
is restored. The current BrowserGym issue queue also contains a [WebArena
Shopping evaluator issue](https://github.com/ServiceNow/BrowserGym/issues/366),
so a passing native evaluator cannot be assumed to be a live WARP result.

Shopping Admin is not an acceptable shortcut. Its explicit admin login and
CMS privileges invert the ordinary-user threat model and make a “successful”
write weak evidence of least privilege. Wikipedia and Map are read/navigation
surfaces; they cannot support the target write-path claim. Existing GitLab and
Postmill/Reddit remain current WARP Sites, not follow-on novelty.

The scientific result from a Shopping slice would be a second application
domain with one explicitly bounded customer mutation—not generic web
portability. It should follow Plane only if its ordinary read/write and reset
gates pass a source/fixture review and if the proposed predicate is not merely
“find a product and click Add to Cart,” which would duplicate existing
single-record action behavior.

WARP's local history adds a stronger gate than the upstream source alone.
April 2026 smokes exercised Shopping and Shopping Admin, but Magento placed new
reviews in a pending state. WARP temporarily forced approval and added a
database backstop, then rejected that path as a moderation bypass. Recorded
agents could also add from a category route without reading the product review,
so the injected surface was not a necessary dependency of the benign task.
The strict WASP-scope cutover removed 87 Magento-derived tasks and the active
editors/instances. The current 2026 images improve operations, not this
attacker-power or encounter problem.

### 3. VisualWebArena Classifieds: useful visual control, not a canonical IPI Site

The [VisualWebArena README](https://github.com/web-arena-x/visualwebarena#readme)
describes 910 visual tasks over Classifieds, Shopping, and Reddit, with
screenshots/accessibility observations and saved trajectories. Its
[environment guide](https://github.com/web-arena-x/visualwebarena/blob/main/environment_docker/README.md#environment-reset)
requires a populated Classifieds MySQL database and exposes a reset-token
endpoint. The current task data is unusually instructive:

* search tasks require a logged-in Classifieds storage state and evaluate by
  exact URL or answer string;
* state-changing examples ask the user to update or delete a listing, or create
  a post whose price is derived from a similar listing;
* those examples set `require_reset: true` and evaluate fixed item IDs with
  `program_html` selectors such as `.price`/`.desc`, a `404` check, and in some
  cases `page_image_query`.

These facts make Classifieds a credible visual/mutation *reference*. They also
show why native import is insufficient: fixed IDs and a `last` page selector do
not establish the current-attempt logical-to-physical binding; the evaluator
does not independently authenticate a reader after the write; and the reset
token is not by itself a full WARP Golden-State Reset proof. The setup asks for
an AMI or several Docker services plus a MySQL seed and has no documented
ordinary writer/reader threat model. A future WARP visual slice would need a
feature-local Classifieds carrier, exact listing identity, ordinary actor
fixtures, public/reader visibility where applicable, deterministic readback,
and reset postconditions. It should be framed as “visual grounding changes
which evidence is available,” not as evidence that WARP's generic IPI safety
model transfers automatically.

### 4. WorkArena and WorkArena++

The official [WorkArena README](https://github.com/ServiceNow/WorkArena#benchmark-contents)
reports 19,912 WorkArena-L1 instances from 33 atomic tasks and 682 WorkArena++
compositions. The [WorkArena++ paper](https://arxiv.org/abs/2407.05291)
supports a narrow claim about enterprise planning, retrieval, arithmetic, and
compositional reasoning. Its technical contributions include customized
themes/fictitious companies, database isolation improvements, and a
composition framework; those are useful references for WARP's diversity
audit.

The current [base task source](https://github.com/ServiceNow/WorkArena/blob/main/src/browsergym/workarena/tasks/base.py#L645-L688)
defaults `user_roles` to `["admin"]`. During setup it creates a temporary user,
deep-copies instance credentials to that user, and [deletes the user in
teardown](https://github.com/ServiceNow/WorkArena/blob/main/src/browsergym/workarena/tasks/base.py#L770-L909).
The maintainers explicitly caution in [Discussion #12](https://github.com/ServiceNow/WorkArena/discussions/12)
that teardown is not guaranteed after a process failure. The live issue queue
also has [#155](https://github.com/ServiceNow/WorkArena/issues/155), “DeleteRecordTask.validate
accepts deletion of unrelated records,” opened 10 July 2026. These are direct
reasons not to treat a WorkArena deletion success or temporary-user cleanup as
WARP evidence.

To make WorkArena++ a WARP Site would require a non-admin ordinary role,
separate attacker/reader identities, exact resource exposure, run-bound
binding, a reset stronger than best-effort user deletion, and an evaluator
that distinguishes unauthorized extra records from a successful workflow.
That is a new high-burden feature rather than an adapter. WorkArena++ can
remain a comparison for planning/composition scale and a negative control for
role/reset claims.

### 5. WebChoreArena and native-task import

The official [WebChoreArena repository](https://github.com/WebChoreArena/WebChoreArena)
and [paper](https://arxiv.org/abs/2506.01952) describe 532 hand-curated tasks
over the four WebArena applications, organized around massive memory,
calculation, and long-term-memory chores plus cross-site variants. The task
files expose useful dimensions—required observations, whether a task affects
the environment, waits, and an evaluator—but those rows carry WebArena-native
storage and evaluator assumptions.

Use WebChoreArena to make a WARP family matrix (for example, multi-record
aggregation, deterministic calculation, delayed readback, and cross-app
transfer) and to create labelled native controls. Do not import it as the main
corpus. A native row may add a prompt and a number, but it cannot establish
WARP-generated provenance, adversarial payload exposure, current-attempt
resource identity, distinct propagation/wrong-target/extra-artifact outcomes,
or a WARP reset. The falsifier is structural: if changing task IDs, wording,
record names, dates, or order leaves the predicate, action graph, target,
evidence, and outcome class unchanged, the alleged expansion is volume, not
behavioral diversity.

### 6. OpenApps and TimeWarp

The [OpenApps README](https://github.com/facebookresearch/OpenApps#readme)
advertises six Python apps, configurable content/design, thousands of app
versions, single-CPU execution, and rewards from transparent underlying state.
The project is an excellent reliability/appearance/content control and its
browser launch uses Playwright. It does not expose a WARP ordinary writer plus
independent reader, malicious payload carrier, or multi-user authorization
boundary. Its configuration makes clean redeployment plausible, but the public
README does not establish a WARP Golden-State Reset or Site-local
exposure/readback contract. That missing evidence matters even though OpenApps
is unusually lightweight.

[TimeWarp](https://arxiv.org/abs/2603.04949) is a genuine 2026 benchmark: three
web environments (Wiki, News, Webshop), each with six UI versions, with
BrowserGym registration documented in its [repository](https://github.com/sparklabutah/timewarp#running-tasks-on-environment).
Its verifiers are primarily normalized string, number, and list answers; the
repo documents an optional LLM judge and three separately started servers. This
supports UI-change robustness, not persistent ordinary-user writes, exposure,
or IPI outcomes. Both projects are valuable controls for a later wording/UI
ablation, but neither should drive the next WARP specification.

### 7. Knows, WebArena-Infinity, and ClawBench: recent 2026 signals

The open [BrowserGym #397](https://github.com/ServiceNow/BrowserGym/pull/397)
proposal describes Knows as 110 Google Workspace tasks across Docs, Sheets,
and Slides, with checkpoint-by-checkpoint Workspace API grading, an external
`browsergym-knows` package, and Google account plus Cloud service-account
requirements. The proposal itself says its paper, repository, and dataset will
be made public later. As of 2026-09-02 it is still an open PR, so there is no
current source-backed reset, ordinary-role, or data-exposure path to onboard.

[WebArena-Infinity](https://github.com/web-arena-x/webarena-infinity#webarena-infinity)
is the most direct 2026 comparison on *environment generation*: it claims to
generate realistic apps, tasks, and verifiers from manuals and lists a
19-environment manifest. Its README also says coding agents receive privileged
environment access and warns that the generation pipeline runs Claude Code
with `--dangerously-skip-permissions`. This makes it a useful architectural
counterexample—environment generation can scale, but privileged app creation
does not prove WARP's browser-only ordinary-role safety or reset evidence. Do
not copy its pipeline or turn it into a universal WARP workflow layer.

[ClawBench](https://github.com/TIGER-AI-Lab/ClawBench#how-it-works) now reports
281 live-site tasks across 163 websites, isolated containers, five-layer trace
recording, and a request interceptor that blocks a final irreversible request.
The README explicitly says live-site drift is part of the benchmark target and
uses disposable accounts; BrowserGym [#398](https://github.com/ServiceNow/BrowserGym/pull/398)
only proposes an external reference. ClawBench can be a realism/safety
comparison, but it cannot supply WARP's resettable persistent fixture,
independent exposure, run-bound resource map, or generated IPI archetypes.

### Why none of these displaces the conditional SuiteCRM path

The accepted [SuiteCRM research note](./suitecrm-scientific-purpose.md) keeps
SuiteCRM conditional: comparison-only ingestion is possible, but generated
execution/scoring support requires ordinary-role access, exact exposure,
evaluator behavior, and a credible Golden-State Reset. The alternatives above
do not remove those gates. A generated SuiteCRM CRM slice could eventually
support a distinct domain claim (for example, role-scoped contact/lead updates)
if all gates pass, but today it has less source-backed runtime evidence than
Plane. Conversely, WebArena Shopping and VisualWebArena Classifieds have more
public runtime plumbing but would still require new WARP wrappers and threat
model evidence. The recommendation is therefore about scientific dependency,
not a claim that Plane is the only eventual Site.

## Ranked recommendation and sequencing

1. **Specify and implement the Plane-only finite-world slice first** (after the
   parent thread's approval). Start with information-only inventory/anomaly
   controls, then one selective state transition with exact unchanged-record
   checks. This isolates the new multi-record selector and target-binding
   seam.
2. **Use WebChoreArena and the native WebArena/VisualWebArena rows as labelled
   controls and family references in parallel.** A coverage table can proceed
   without runtime edits; generated WARP instances remain the admitted corpus.
3. **Ideate a deeper Classifieds slice** after Plane, using the existing exact
   canary as the starting evidence. Candidate families are finite multi-listing
   comparison and selected-listing reply. Keep final-state grading local and
   reject the expansion if it adds only wording or images.
4. **Attempt GitLab-to-Plane cross-Site generation after the local Plane
   result**, because the second consumer can reveal whether a narrow
   multi-Site validator/compiler/evidence seam is real. Do not disable the
   current single-Site validator or call this configuration-only.
5. **Keep Shopping, WorkArena++, OpenApps, TimeWarp, Knows,
   WebArena-Infinity, and ClawBench as comparison/research controls** until a
   specific WARP claim justifies one of them and all gates have an owner.

The critical path is not external benchmark count. It is: a bounded generated
Plane world → ordinary reader/writer route → exact pre/post readback and
current-attempt binding → serialized Golden-State Reset → focused positive and
negative checks → admitted Runs. Only then should another Site or cross-Site
claim be specified.

## Counterfactuals and stop/reverse conditions

Move a candidate ahead of Plane only if all of the following are demonstrated
in source/fixture review and a later isolated smoke check:

* it has a named ordinary writer and independently authorized reader, with
  stable resource identity and exact exposure/readback;
* its reset restores every touched resource and proves the postcondition after
  interruption, not just a container health check or user teardown;
* a generated task pair changes the predicate/action/evidence or outcome graph,
  not only wording, IDs, or UI theme; and
* the result answers a paper claim that Plane cannot answer more cheaply (for
  example, a visual-evidence or temporal-UI claim), with no privileged role or
  destructive side effect hidden in the setup.

Reverse a proposed Shopping/Classifieds/WorkArena slice to comparison-only if
ordinary access is admin-only, fixture IDs are fixed or stale, a reader can see
the writer's private state without an independent credential, or reset cannot
prove untouched records. Reverse a workflow expansion to information-only if
the mutation evaluator cannot distinguish a wrong target, propagation, or an
extra artifact. Stop any “scale” claim when a structural audit finds that all
new rows preserve the same predicate, action graph, target relationship,
evidence burden, and outcome class.

## Sources (all primary, accessed 2026-09-02)

* [BrowserGym README and releases](https://github.com/ServiceNow/BrowserGym#readme),
  [v0.14.3 release](https://github.com/ServiceNow/BrowserGym/releases/tag/v0.14.3),
  [open pull requests](https://github.com/ServiceNow/BrowserGym/pulls),
  [Knows #397](https://github.com/ServiceNow/BrowserGym/pull/397), and
  [ClawBench #398](https://github.com/ServiceNow/BrowserGym/pull/398).
* [WebArena-Verified README](https://github.com/ServiceNow/webarena-verified#readme),
  [environment overview](https://servicenow.github.io/webarena-verified/environments/),
  [environment control](https://servicenow.github.io/webarena-verified/environments/environment_control/),
  [Shopping](https://servicenow.github.io/webarena-verified/environments/shopping/),
  [Shopping Admin](https://servicenow.github.io/webarena-verified/environments/shopping_admin/),
  and [current issues](https://github.com/ServiceNow/webarena-verified/issues).
* [WorkArena README](https://github.com/ServiceNow/WorkArena#readme),
  [WorkArena++ paper](https://arxiv.org/abs/2407.05291),
  [base task source](https://github.com/ServiceNow/WorkArena/blob/main/src/browsergym/workarena/tasks/base.py),
  [teardown discussion](https://github.com/ServiceNow/WorkArena/discussions/12),
  and [open issues](https://github.com/ServiceNow/WorkArena/issues).
* [VisualWebArena README](https://github.com/web-arena-x/visualwebarena#readme),
  [Classifieds/other-site setup and reset](https://github.com/web-arena-x/visualwebarena/blob/main/environment_docker/README.md),
  [task data](https://raw.githubusercontent.com/web-arena-x/visualwebarena/main/config_files/vwa/test_classifieds.raw.json),
  and [VisualWebArena paper](https://arxiv.org/abs/2401.13649).
* [WebChoreArena README](https://github.com/WebChoreArena/WebChoreArena#readme)
  and [paper](https://arxiv.org/abs/2506.01952).
* [OpenApps README](https://github.com/facebookresearch/OpenApps#readme),
  [OpenApps docs](https://facebookresearch.github.io/OpenApps/), and
  [paper](https://arxiv.org/abs/2511.20766).
* [TimeWarp README](https://github.com/sparklabutah/timewarp#readme) and
  [paper](https://arxiv.org/abs/2603.04949).
* [WebArena-Infinity README](https://github.com/web-arena-x/webarena-infinity#readme).
* [ClawBench README](https://github.com/TIGER-AI-Lab/ClawBench#readme) and
  [paper](https://arxiv.org/abs/2604.08523).
