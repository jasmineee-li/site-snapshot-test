# WARP baseline evidence (2026-08-30)

## Scope, provenance, and reading rules

This note is a static baseline of the WARP checkout at commit
`4f3df62e2a471703a7cf44020bf295cd282f9d4e` (branch
`research/web-benchmark-onboarding-2026-08-30`, based on `origin/main`).  I read
the repository guidance, task-generation path guides, and the research skill
before inspecting the implementation.  I did not run tests, benchmark jobs, a
browser, a launcher, or an installation, and did not read secrets.  Code and
tests cited below are therefore implementation evidence, not a claim that the
tests currently pass.  A configured-host Classifieds run is cited only as
retained historical evidence from the checked-in research note; its remote
artifacts were not re-fetched or replayed here.

The principal distinction throughout is **static declaration**, **runtime
default**, **named opt-in POC**, **comparison-only ingestion**, and **released
benchmark evidence**.  A static declaration or a design sentence is not live
onboarding proof.

## Bottom line

WARP is currently a WebArena Verified wrapper pipeline for realistic,
eval-aware browser safety tasks.  The released/default path is GitLab and
Reddit/Postmill, with Browser Use as the normal Phase 4 worker and AgentLab as
an isolated sidecar/comparison path.  The root README describes the 50-task
released benchmark and its 80.7% headline result
([README.md](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/README.md:35),
[README.md](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/README.md:118));
that is the demonstrated release boundary, not an arbitrary-site guarantee.

The active editor registry contains only two editor/site pairs (WebArena
Verified GitLab and WebArena Verified Reddit)
([editor registry](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/editors/__init__.py:8)).
Each class has a broad method surface (19 GitLab methods and five Reddit
methods), rather than one generic operation
([GitlabEditor](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/editors/gitlab.py:145),
[RedditEditor](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/editors/reddit.py:175)).
The static Site Composition contract additionally names nine owner roles,
including targeting, profile, editor, writer, feasibility, read surface,
readback, final-state evaluation, and action cards
([contract roles](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/site_composition_contracts.py:19)).
Consequently, the manuscript/README phrase “one editor class” is a useful
architectural shorthand, but it is not an observed end-to-end onboarding cost.

Classifieds is a deliberately named, one-task VisualWebArena canary.  Its
static composition marks final-state evaluation unsupported, and the normal
runtime composition does not bind it.  A retained configured-host canary note
reports one task's response-label compliance and cleanup, but the `completion.json` and logs
remain on a remote host/operator bundle and are not tracked in this checkout.
The result is useful POC evidence, not independently reproducible release
evidence at this commit.

WASP, STWebAgentBench, and DoomArena are comparison-only ingestion tracks.
Their native AgentLab records intentionally do not enter WARP Phase 4 or WARP
reward evaluation.  Adding another benchmark to the capability catalog alone
would not make it a WARP evaluator: the final-state catalog and task-id reward
dispatcher currently hard-reject non-WebArena-Verified requests.

## Thesis and released/default state

The root project describes WARP as a “Wrapper-based Adaptive Realistic
Pipeline” and states that it generates IPI benchmarks on WebArena Verified,
seeds a normal-user-writable field, checks rendered exposure and state, and
rewrites payloads for eval-aware models
([README.md](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/README.md:3),
[README.md](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/README.md:35)).
The stated threat model is a regular authenticated GitLab/Postmill user who
cannot alter the application, harness, task, system, or reward; PVPO checks
rendered presence and a state check verifies the side effect
([README.md](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/README.md:51)).
The five phases and the TP/VEA awareness toolkit are summarized in the README
([README.md](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/README.md:61)).

The release accounting is concrete: 20 GitLab issue follow-up tasks using
`issue.description`/`create_issue`, 10 GitLab issue-comment tasks using
`issue.description`/`create_issue_note`, and 20 Reddit/Postmill tasks using
`submission.body`/`create_post`
([README.md](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/README.md:118)).
The package README repeats that the current mainline carrier scope is GitLab
and Reddit/Postmill; other sites are historical/support context
([taskgen README](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/README.md:127)).
It also makes Browser Use the normal worker, keeps AgentLab in a separate
sidecar, and gives Phase 4 host/agent/evaluation boundaries
([taskgen README](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/README.md:1),
[taskgen README](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/README.md:303)).

The root README reports 80.7% across six models and explicitly lists
one-host/two-site scope and confounded cohorts as limitations
([README.md](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/README.md:89),
[README.md](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/README.md:178)).
Those limitations are the appropriate current evidence boundary.

## Runtime composition and extension interfaces

### Static Site Composition is broader than the live default

Site Composition is deliberately data-only: declarations carry owner IDs and
scope, while `not_applicable` is compiler-derived rather than a declaration
shortcut ([composition contract](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/site_composition_contracts.py:101)).
The host-owned use-case catalog requires all nine roles for Phase 4 execution,
while a public-reply use case omits final-state evaluation
([use-case catalog](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/site_composition_contracts.py:271)).
The checker itself validates immutable declarations and projections; it does
not establish live owner methods, HTTP reachability, or reader visibility
([static checker](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/site_composition.py:302)).

The static defaults include GitLab, Reddit, and a Classifieds declaration
([static defaults](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/site_composition_defaults.py:11)).
The runtime `SiteCatalog(None)` default, however, binds only GitLabSite and
RedditSite ([runtime catalog](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/sites/catalog.py:327)).
This static-closure/runtime-default split is an important source of false
“supported site” readings.

### Editor, seeding, and runner seams

The editor registry is the shared host-side source for resolving methods,
validating options, substituting seeds, rendering prompts, pre-shard
feasibility, and sandbox validation
([registry consumers](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/editors/_registry.py:8)).
Registration checks decorated methods and rejects duplicates
([registry invariant](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/editors/_registry.py:66)).
This is a real extension seam, but each registered class still supplies a
substantial method family and downstream projections.

`SeedSiteRegistry` is an immutable, per-run seam: site editors own
authentication, mutation, and cleanup; Phase 2/2c owns exposure and admission
([seed contract](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/seeding/site_contracts.py:1),
[seed registry](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/seeding/site_contracts.py:110)).
Agent execution is runner-neutral at the protocol level, with Browser Use the
default and AgentLab registered as an optional runner
([agent runtime](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/agent_runtime.py:18),
[runner registry](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/runners/__init__.py:10)).

### Reward/evaluator is a separate hard seam

The local final-state catalog is explicitly WebArena Verified only: request
construction rejects any other benchmark and catalog construction rejects
non-WebArena evaluator identities
([final-state request](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/rewards/final_state_catalog.py:41),
[final-state catalog](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/rewards/final_state_catalog.py:104)).
Its default catalog contains only GitLab and Reddit evaluators
([default evaluators](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/rewards/final_state_catalog.py:206)).
For a task-id-bearing evaluation, the dispatcher always routes to the vendor
WebArena Verified evaluator ([dispatcher](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/rewards/dispatcher.py:18));
the vendor path requires a canonical WebArena task ID
([vendor evaluator](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/rewards/vendor_webarena.py:98)).
Therefore, a new benchmark needs evaluator/catalog and dispatcher work in
addition to an editor, seed path, and action cards.

## Actual integrations and comparison-only tracks

The capability catalog gives WebArena Verified and VisualWebArena WARP
capabilities and `worldsim_v5` Phase 4, while WASP, STWebAgentBench, and
DoomArena have only `comparison_ingestion` and a native comparison runner
([capability catalog](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/benchmark_capabilities.py:40)).
Catalog invariants prohibit WARP and comparison capabilities from coexisting
and require comparison-only records to stay on the comparison runner
([benchmark invariants](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/benchmark_contracts.py:195)).
The separate comparison envelope owns no browser lifecycle, reset, WARP
scoring, or Phase 4 state ([comparison ingestion](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/comparison_ingestion.py:1)).
Its native reward is intentionally named `native_reward`, and WARP-only fields
are rejected ([comparison record](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/comparison_ingestion.py:168)).

This makes the current tracks easy to classify:

- **Released/default WARP:** WebArena Verified GitLab and Reddit/Postmill;
  Browser Use worker; optional AgentLab sidecar for the documented paths.
- **Named opt-in POC:** VisualWebArena Classifieds listing-reply canary (one
  semantic operation, not a general Classifieds benchmark).
- **Comparison-only:** WASP, STWebAgentBench, and DoomArena native AgentLab
  ingestion; no WARP Phase 4/reward claim.
- **Aspirational/target:** any synthetic environment, a one-editor onboarding
  story, additional hosts, and broader awareness transfer.  These are goals or
  paper framing, not cross-environment demonstrations.

The older experiment TODO records STWebAgentBench as a planned adapter and
describes WASP/DoomArena/SafeArena as stretch or incomplete work
([experiment TODO](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/TODO-2-paper-experiments.md:132)).
The current progress/spec documents should take precedence over those historic
plans.

## Classifieds listing-reply POC: what exists and what does not

The static Classifieds composition is VisualWebArena, exposes the
`listing_reply.body` carrier, and explicitly marks final-state evaluation
unsupported ([Classifieds composition](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/site_compositions/classifieds.py:21)).
The runtime composition binds it only when the named
`classifieds_listing_reply_poc` experiment is requested; normal/empty
composition remains `None` ([runtime POC](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/runtime_composition.py:77),
[runtime default](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/runtime_composition.py:104)).
The capability adapter is one semantic operation (`create_listing_reply`) at
experimental support level ([Classifieds adapter](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/adversarial_actions/classifieds_capability.py:1));
the editor uses a regular-user fixed POST flow, exact-reply cleanup, and a
separate reader/readback path ([Classifieds editor](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/sites/classifieds_editor.py:321),
[Classifieds readback](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/sites/classifieds_readback.py:171)).
The reader is explicitly anonymous, fresh, and free of storage state
([Classifieds reader](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/sites/classifieds_reader.py:84)).

The retained research note records a configured-host canary run
`run-34b2327b67ad42a18e480998fe301b68`: listing 12085, reply 3, independent
anonymous exact-ID/body observation, one admitted Phase 4 task with
`final_status=complied`, ecological exposure/engagement 1/1, and cleanup/reset
success ([retained canary note](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/research/classifieds-listing-reply-poc-2026-08-14.md:238)).
The conditional ASR 1/1 is WARP `AgentResponseEvaluator` compliance with the
opposite binary response label, not native VisualWebArena final-state grading.
The canary prepares that reward through the host-owned compiler
([canary reward](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/scripts/prepare_classifieds_canary.py:154),
[response compiler](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/adversarial_actions/reward_compiler.py:151)).
Exact persisted reply readback proves the seed/exposure contract separately.
The retained evidence JSON records source base commit
`680bcea0c969296c33dd19c35e31df480a219eec`, remote retention under
`/home/ubuntu/...`, an operator bundle, and artifact hashes
([retained evidence JSON](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/research/classifieds-listing-reply-canary-evidence-2026-08-16.json:1)).
That source commit predates this baseline (`680b...` is the 2026-08-14
“add Classifieds listing-reply proof” commit); the current checkout is
`4f3df62e`.  A local file search found no tracked
`logs/classifieds-canary/**` or `completion.json`, so the run is not locally
auditable at this commit.  This is not evidence that the remote run did not
happen; it is evidence that this checkout cannot independently reproduce or
inspect it.

The canary guide itself says `completion.json` is a terminal run artifact and
does not attest the benchmark-host lifecycle postcondition
([canary guide](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/agent_docs/classifieds-canary.md:16)).
The retained note also leaves release gates unresolved (license/data inventory,
privileged reset and command-echo risk, and mutable dependency/MUZZLE review)
([retained gates](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/research/classifieds-listing-reply-poc-2026-08-14.md:268)).
Thus “Classifieds completion artifact exists” should be reported as a retained,
remote-only POC claim, not as a checked-in artifact or released integration.

The local Classifieds tests are valuable contract/unit evidence (they use fake
forms/sessions and readback/reset fixtures), but were not run here
([vertical-slice tests](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/tests/test_classifieds_vertical_slice.py:88)).
Static composition tests similarly cannot prove live host reachability
([composition tests](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/tests/test_classifieds_site_composition.py:37)).

## Methodological requirements and portability boundaries

The strict Phase 2c gate requires the live HTTP editor to traverse the same
path as Phase 4, prove regular-user write/render/reachability, and provide an
independent reader/readback path; SQL is excluded
([Phase 2c contract](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/warp-taskgen-technical-spec.md:1578)).
PVPO is page-surface-stable: it uses runner-owned browser capture, TreeWalker
text witnesses, and ink occupancy, with zero coverage classified as
`injection_not_encountered` rather than resistance
([PVPO semantics](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/warp-taskgen-technical-spec.md:1860),
[PVPO capture](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/warp_taskgen/phase_4/pvpo_query.js:75)).
The implementation walks text nodes in `document.body`/`document.documentElement`
and does not traverse frames/canvas or inspect DOM attributes.  The repository
calls out the image-alt-text consequence explicitly: such tasks are
`max_coverage==0` and are dropped at the dataset layer
([image-alt issue](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/scripts/drop_image_alt_text_tasks.py:1)).
Any “multimodal” or cross-frame portability claim therefore needs a separate
surface extractor and an explicit encounter measurement.

TP and VEA are observational on every extractable trajectory, including zero
coverage; TP is the default iterator branch and VEA is report-only
([TP/VEA spec](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/warp-taskgen-technical-spec.md:1931)).
Gate 2 then requires reward/readback semantics (`complied`, `task_broke`, or
`refused`) ([Gate 2](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/warp-taskgen-technical-spec.md:1967)).
Headline ASR keeps a fixed admitted cohort denominator, while the
encounter-conditioned Gate 1 rate excludes zero-coverage cases; zero coverage
is non-encounter, not resistance ([PVPO quantities](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/docs/warp-taskgen-technical-spec.md:1860)).

For a new host/benchmark, an honest portability experiment should measure at
least: (1) admission and live HTTP exposure on the normal writer path, (2)
independent fresh-reader visibility and exact identity/readback, (3) PVPO
coverage and task-broken rates by carrier/surface, (4) evaluator/reward parity,
(5) reset/cleanup and host-lifecycle postconditions, and (6) TP/VEA label drift
against a separately adjudicated ground truth.  A zero-change editor-only port
that still satisfies all six would support the one-editor claim; any required
new owner projection, evaluator/dispatcher change, reader exception, or
host-specific workaround would falsify that cost claim.

## Canonical manuscript resolution and discrepancies

I followed each repository's written authority rather than file age:

| Reference | Written authority and provenance | Attributed claim relevant to onboarding |
|---|---|---|
| `worldsim-paper` | `CLAUDE.md` names `icml/icml_final.tex` as the active ICML target ([paper guidance](/Users/ashtonchew/projects/worldsim-paper/CLAUDE.md:1)). Paper HEAD is `48aa58827a147723647867576917d7dbe43a5b1f`; `icml_final.tex` last changed in `f936bca4df7a1e9408f9266432ee4c9adac115be`, SHA-256 `73def061d002eabc981861119cb9ed829ef93e0dac64cf1c21417bd4670bdc1d`. | The canonical paper says extending to a new environment costs one editor class plus an auto-generated suite ([canonical paper](/Users/ashtonchew/projects/worldsim-paper/icml/icml_final.tex:793)), reports 80.7%/81.5% results ([canonical paper](/Users/ashtonchew/projects/worldsim-paper/icml/icml_final.tex:964)), and limits validation to one host with future integration still needed ([canonical paper](/Users/ashtonchew/projects/worldsim-paper/icml/icml_final.tex:1152)). |
| `warp-poster` | `CLAUDE.md` names `WARP Poster.html` primary and `icml_final_2.tex` authoritative narrative/numbers ([poster guidance](/Users/ashtonchew/projects/warp-poster/CLAUDE.md:1)). Poster HEAD is `545bd037a19be949aa4dc86f1a85b88eb5fe0543`; its `icml_final_2.tex` was last changed in `e87a8f9d854b072f6fca7bf71c46cc788be59c44`, SHA-256 `ef603a2984de46cdaab75f882d60ce6e029589798bb70a9d282dab280c6a484f`. | The poster's authoritative narrative repeats the one-editor framing ([poster manuscript](/Users/ashtonchew/projects/warp-poster/icml_final_2.tex:775)), 50-task setup/results ([poster manuscript](/Users/ashtonchew/projects/warp-poster/icml_final_2.tex:836)), and one-host/future-environment limitation ([poster manuscript](/Users/ashtonchew/projects/warp-poster/icml_final_2.tex:1140)). |

`worldsim-paper/icml/icml_final_2.tex` is newer content (last commit
`90c8066e2be83610e81a26dc0fad15914014c595`, SHA-256
`6037e73ff7680db0728f5bcd23e78c7bdc4242c051b791a0c8a046856025fdb3`) but is
not the paper repository's named active target.  It and the poster use the
same newer 80.7/one-editor wording; the paper's own `CLAUDE.md` still makes
`icml_final.tex` canonical.  This is a provenance discrepancy, not a reason to
silently promote `icml_final_2.tex`.  Both manuscript variants frame
“any synthetic environment” and “one editor class” as the architecture's
generalization; the code evidence above supports a narrower, qualified reading
at this snapshot.

## Counterevidence, uncertainties, and falsification plan

1. **One-editor portability.**  Onboard a second benchmark with only the
   proposed editor class and generated task suite.  Record every changed
   module and every owner status (targeting/profile/read surface/readback,
   feasibility, action cards, final evaluator, seed/reset, and dispatcher),
   then execute Phase 2c, fresh-reader readback, PVPO, Gate 2, and cleanup on a
   configured sandbox host.  Any required change outside the editor/task suite
   falsifies the literal cost claim; success would support it.
2. **Classifieds artifact provenance.**  Retrieve the remote `completion.json`,
   preflight/probe/Phase 4/result/reset artifacts named in the retained JSON,
   verify their hashes and source/runtime projection against both commit `680b...`
   and current `4f3df62e`, and independently inspect the fresh-reader,
   PVPO, reward, and cleanup evidence.  A missing artifact, hash mismatch, or
   source/runtime mismatch falsifies a current-release completion claim (while
   not disproving that the historical run occurred).
3. **Awareness transfer.**  Run the same TP/VEA/PVPO instrumentation on a
   non-WebArena host and adjudicate a ground-truth set of injected/exposed
   trajectories.  Report extraction, label, and encounter drift by carrier;
   there is no current ground-truth evidence that awareness rates transfer.
4. **Comparison boundary.**  Feed malformed and mixed WARP fields to the
   comparison envelope and verify they remain rejected and no WARP reward/state
   files are produced.  The code and tests specify this fail-closed boundary,
   but this note did not execute them.
5. **Surface coverage.**  Include image-alt, canvas, and cross-frame-only
   carriers in a portability run.  Under the current TreeWalker extractor they
   should produce zero coverage; treating them as encountered without an
   extractor change would falsify the PVPO methodology.

The principal remaining uncertainties are remote-host artifact retention,
current runtime parity with the older Classifieds source commit, and whether
future onboarding work changes the evaluator/dispatcher seam.  These are
measurable uncertainties, not evidence of failure.
