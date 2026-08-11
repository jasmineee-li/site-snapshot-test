# Research note: benchmark adapter seam architecture

**Date:** 2026-08-10
**Status:** research and sequencing options only; this note does not choose a
final interface.

## Evidence boundary

The current source of truth is `origin/main` at `d7e8dc74` (the
`codex/agent-ergonomics-plan` worktree). I read the current package guides,
technical specification, implementation, and focused tests. I also read the
`feat/multi-benchmark` worktree at `d2c3809d` read-only. That worktree is dirty,
pre-PVPO/pre-WASP-scope, and its design notes are historical evidence rather
than an API to copy. No external or secondary source is used below.

## Finding: “benchmark adapter” is several seams, not one layer

The current system already has distinct ownership boundaries. A future
benchmark integration can compose these boundaries, but collapsing them into a
single object would make reset, scoring, browser runtime, and site behavior
hard to reason about.

| Concern | Current owner and factual contract | Boundary to preserve |
| --- | --- | --- |
| Benchmark identity and phase capability | [`worldsim/benchmark_capabilities.py`](../../worldsim/benchmark_capabilities.py) normalizes names, declares default/supported runners, and records whether a benchmark is WARP Phase 4 or comparison-only. `BenchmarkConfig` rejects unknown or mixed metadata ([`worldsim/config.py`](../../worldsim/config.py), lines 485–536). | This is benchmark metadata and gating, not an HTTP editor, evaluator, or browser runner. |
| Task, route, and surface vocabulary | [`worldsim/surface_identity.py`](../../worldsim/surface_identity.py) maps profile IDs to canonical carriers and fails closed for unknown or ambiguous mappings. Action capability adapters map generic contracts to route/editor/fixture/probe/cleanup facts ([technical spec](../warp-taskgen-technical-spec.md), lines 970–997; [action-contracts](../../agent_docs/action-contracts.md), lines 6–13). | Keep selectors, endpoint paths, fixture details, and cleanup host-owned. Model prompts must not become a hidden benchmark adapter. |
| Site behavior | [`worldsim/editors/`](../../worldsim/editors/) registers `(benchmark, site)` editor classes. GitLab and Reddit/Postmill currently implement authenticated writes and read-surface metadata; reward-local final-state modules own their readback. | A site module owns the site protocol and evidence. It is not a benchmark-wide task loader or browser-agent runtime. |
| Browser-agent runner | [`worldsim/agent_runtime.py`](../../worldsim/agent_runtime.py), [`worldsim/runners/__init__.py`](../../worldsim/runners/__init__.py), and the worker pool define runner lifecycle/results. Browser Use is the default; AgentLab is a lazy, isolated sidecar. | A runner owns browser-agent setup/run/teardown and runtime artifacts, not benchmark scoring or Phase 2 admission. |
| Canonical evaluator subprocess | [`worldsim/rewards/vendor_webarena.py`](../../worldsim/rewards/vendor_webarena.py), lines 100–270, sends canonical `task_id`, agent response, HAR/network trace, and environment URLs to a separate `warp-taskgen-webarena-verified` environment over one JSON request/response. | Preserve WebArena evaluator parity and dependency isolation. Do not turn the evaluator process into a generic browser runner or silently score a task-id-less task as canonical. |
| Environment and reset | User-supplied [`BenchmarkInstance`](../../worldsim/config.py) fields include `site_url`, optional `reset_endpoint`, auth, placeholders, and read-only DB access. Binding collects reset endpoints in task runtime metadata ([`worldsim/agent_config.py`](../../worldsim/agent_config.py), lines 585–638). WARP Phase 4 resets and seeds in [`worldsim/phase_4/execution.py`](../../worldsim/phase_4/execution.py), lines 367–388; AgentLab comparison does the analogous reset/optional seed in [`worldsim/runners/agentlab.py`](../../worldsim/runners/agentlab.py), lines 1847–1876. | Declare reset requirements and ownership explicitly. A benchmark adapter should not provision or tear down external services. BrowserGym’s per-episode reset is a separate runtime event from a benchmark site reset. |
| Comparison-only benchmark | `wasp`, `stwebagentbench`, and `doomarena` are currently declared `phase_4_mode="comparison_runner"`, AgentLab-only, and Phase 2/2c unsupported ([`worldsim/benchmark_capabilities.py`](../../worldsim/benchmark_capabilities.py), lines 43–84; [`tests/test_benchmark_capabilities.py`](../../tests/test_benchmark_capabilities.py), lines 17–32). The sidecar README distinguishes native `run` from WARP-parity `phase4-run` ([`packages/worldsim-agentlab-runner/README.md`](../../packages/worldsim-agentlab-runner/README.md), lines 62–67). | Comparison output, native benchmark reward, and WARP Phase 4 outcomes must remain separately labelled. Native AgentLab output must not be reported as WARP Phase 4 parity data. |

The technical specification states the intended split directly: AgentLab is
crossed only through a sidecar, WARP owns Phase 4 admission, seeding, rewards,
PVPO, TP/VEA, variants, and summaries, and the sidecar delegates only the
browser episode ([technical spec](../warp-taskgen-technical-spec.md), lines
489–493). The current code follows that contract. The seam should therefore be
additive and compositional, rather than a second orchestration stack.

## Invariants for any vertical slice

1. **AgentLab sidecar isolation.** Root WARP Taskgen must not import AgentLab;
   the sidecar owns the incompatible dependency graph and communicates through
   JSON. The sidecar's `run` and `phase4-run` commands remain semantically
   distinct ([`packages/worldsim-agentlab-runner/src/worldsim_agentlab_runner/cli.py`](../../packages/worldsim-agentlab-runner/src/worldsim_agentlab_runner/cli.py), lines 15–37).

2. **WebArena evaluator parity.** A canonical WebArena task is identified by its
   benchmark task ID and evaluated with the vendor package's typed response and
   HAR semantics. The separate package documents the one-object stdin/stdout
   protocol and the rule that task-id-less novel tasks use WARP-local evaluators
   ([`packages/warp-taskgen-webarena-verified/README.md`](../../packages/warp-taskgen-webarena-verified/README.md), lines 21–30). Missing evaluator environments or malformed network evidence fail closed in the current adapter.

3. **Locality of behavior.** Benchmark facts belong near benchmark capability
   metadata and task conversion; site HTTP and readback belong to editors and
   reward-local modules; browser lifecycle belongs to runners; subprocess
   protocol belongs to the evaluator package; reset/seeding orchestration stays
   in the phase or comparison task runner. This follows the feature-ownership
   guide ([`code-organization.md`](../../agent_docs/code-organization.md), lines 5–86).

4. **Modularity by feature.** Adding a benchmark should be able to add a
   capability entry and focused task/surface/editor modules without changing
   generic Phase 2c or Phase 4 scoring. The technical specification explicitly
   requires benchmark-neutral Phase 2c and editor-registry extension rather
   than benchmark strings in the verifier ([technical spec](../warp-taskgen-technical-spec.md), lines 1473–1479).

5. **Semantic names and fail-closed metadata.** Keep `benchmark`, `site`,
   `runner`, `evaluator`, `reset_endpoint`, and `agentlab_task_name` distinct.
   Normalize aliases once, reject mixed benchmark metadata, and require an
   explicit BrowserGym task name for non-WebArena tasks (the AgentLab bridge
   does this at [`worldsim/runners/agentlab.py`](../../worldsim/runners/agentlab.py), lines 164–193).

6. **Separate reset meanings.** A user-configured site reset (`POST` to one or
   more bound endpoints) restores benchmark state; a Browser Use/BrowserGym
   episode reset creates a fresh browser task context. The current Phase 4 code
   calls site resets before seeding, while `phase4-run` records the BrowserGym
   reset seed in its own runtime artifacts ([`phase4_loop.py`](../../packages/worldsim-agentlab-runner/src/worldsim_agentlab_runner/phase4_loop.py), lines 108–150). An adapter contract must not conflate these events.

## Candidate vertical slices (options, not an interface decision)

### Slice A — WebArena capability and evaluator ledger (lowest risk)

Start with the benchmark already supported by WARP Phase 2/2c/4:

- make the benchmark capability entry, canonical surface mapping, existing
  `(benchmark, site)` editor registry, and evaluator routing explicit in one
  focused set of contract tests;
- exercise one canonical task with vendor scoring and one task-id-less novel
  task with WARP-local scoring;
- keep Browser Use as the default runner and leave the current AgentLab bridge
  untouched.

Acceptance evidence: alias/mixed-metadata rejection
([`tests/test_benchmark_config.py`](../../tests/test_benchmark_config.py), lines
22–71), Phase 2c benchmark gates
([`tests/test_benchmark_capabilities.py`](../../tests/test_benchmark_capabilities.py), lines 34–84), surface fail-closed tests
([`tests/test_surface_identity.py`](../../tests/test_surface_identity.py), lines
27–122), editor registry/HTTP/readback tests, and vendor evaluator tests
([`tests/rewards/test_vendor_webarena.py`](../../tests/rewards/test_vendor_webarena.py)). This slice proves the seam without adding a new runtime.

### Slice B — Canonical evaluator subprocess as an explicit feature

If evaluator portability is the immediate risk, isolate and test the evaluator
protocol before adding another benchmark. Test JSON framing, task-ID routing,
HAR-required failure, vendor-shim behavior, timeout/non-zero exit, and redacted
diagnostics. Keep evaluator naming explicit (for example, “WebArena Verified
evaluator”) rather than calling it a runner. This is a narrow prerequisite for
both Browser Use and AgentLab because both produce the same WARP network
artifacts while using different recorders ([technical spec](../warp-taskgen-technical-spec.md), lines 491–493).

### Slice C — STWebAgentBench comparison-only task conversion

The prior worktree contains a useful, but stale, `STWebAgentBenchAdapter` at
`/Users/ashtonchew/projects/warp/.codex-worktrees/feat-multi-benchmark/worldsim/adapters/stwebagentbench.py`:
it loads `test.raw.json`, preserves `policies`, splits `|AND|` start URLs,
maps placeholders, and resolves credentials/task names (lines 102–319). The
historical plan records one important gap: `policies` was passed through but
not scored by the WARP reward path (prior worktree
`docs/TODO-2-paper-experiments.md`, lines 121–153). Treat that code as a
selective reference, not a drop-in API.

Candidate slice: wrap a small, declared task subset; run AgentLab native
`run` against its benchmark environment; write outputs under an unmistakable
comparison path; report native task completion separately from WARP safety
metrics. Do not expose it to Phase 2c or WARP Phase 4 until its native
evaluator, reset, credentials, and artifact contract are proven.

Acceptance evidence: deterministic raw-task-to-WARP-task fixtures; policy
provenance retained even if unscored; explicit `agentlab_task_name`; missing
credential/reset failure; one native result and one malformed-result fixture;
comparison summary labels that cannot be mistaken for `phase4-run`.

### Slice D — WebArena AgentLab `phase4-run` parity

Only after Slice A (and, if needed, B), use the current sidecar's
`phase4-run` path for a WebArena task. The sidecar README requires artifact,
PVPO, network, auth, timeout/retry, and resume gates before a new matrix is
called parity data ([`packages/worldsim-agentlab-runner/README.md`](../../packages/worldsim-agentlab-runner/README.md), lines 69–81). The root keeps reward dispatch, PVPO encounter, TP/VEA, variants, and summaries.

Acceptance evidence already has a strong local base in
[`tests/test_agentlab_runner.py`](../../tests/test_agentlab_runner.py): request
mapping, auth scope, BrowserGym context controls, artifact manifests, timeout
recovery, action projection, and resume/audit behavior. Add a small live smoke
only when the configured sidecar and benchmark host are available; otherwise
record the infrastructure blocker rather than weakening the gate.

### Slice E — WASP and DoomArena only after live prerequisites

Keep these as later comparison slices. The prior branch's WASP adapter exposes
an `instantiate_injection_text` helper but its matrix expansion intentionally
stores the raw template because a caller must provide the live `domain_map`
(`worldsim/adapters/wasp.py`, lines 305–365 and 441–452); no current Phase 3/4
caller proves that URL resolution, seeding, or `attacker_eval` is wired. The
prior DoomArena adapter builds attack configuration objects but documents that
it is not a complete benchmark adapter (`worldsim/adapters/doomarena.py`, lines
180–191) and its attacked environment requires DoomArena-managed
infrastructure. These are prerequisites, not reasons to broaden the first
seam.

## Acceptance-test map

- **Identity/gating:** `tests/test_benchmark_capabilities.py`,
  `tests/test_benchmark_config.py`, and Phase 4 preflight tests.
- **Surface/action locality:** `tests/test_surface_identity.py`,
  `tests/test_phase_1_tasks.py::test_capability_adapters_keep_benchmark_specific_support_out_of_prompts`,
  `tests/test_editors_registry.py`, and `tests/test_editors_registry_coverage.py`.
- **Site protocol and reset:** `tests/test_editor_http_auth.py`,
  `tests/test_read_surface_editors.py`, `tests/test_seeding.py`,
  `tests/test_task_reset_cache.py`, and the reset sections of
  `tests/test_phase_4/`.
- **Evaluator contract:** `tests/rewards/test_vendor_webarena.py`,
  `tests/rewards/test_dispatcher.py`, and `tests/preflight/test_phase_4_preflight.py`.
- **Runner/sidecar contract:** `tests/test_agentlab_runner.py`,
  `tests/test_eval_worker_pool.py`, `tests/phase_4/test_runner.py`, and
  `tests/test_agentlab_pvpo_needham_parity.py`.
- **Artifact and reporting separation:**
  `tests/test_phase_4_artifact_audit.py` plus a comparison-run fixture that
  asserts native `run` output is not accepted as WARP `phase4-run` parity.
- **Live proof:** follow [`agent_docs/verification.md`](../../agent_docs/verification.md),
  especially the evaluator-venv preflight, AgentLab sidecar sync, and focused
  adapter smoke command. Live tests remain conditional on the configured
  sandbox benchmark infrastructure.

## Migration risks and reusable prior work

- **Stale topology:** the prior branch predates current PVPO and WASP-aligned
  scope; it contains old `worldsim/` paths and adapters for deleted or inactive
  sites. Do not copy its adapter registry or site map wholesale.
- **Duplicate orchestration:** prior `worldsim/runner.py` and two-function
  runner API are useful design evidence, but current `AgentRunner` and lazy
  runner registry already exist. Porting both creates competing task-runner
  semantics and risks moving reset/reward ownership into a runner.
- **Policy/evaluator drift:** STWebAgentBench's preserved `policies` need a
  declared scoring owner. “Passed through” is not evidence of safety-policy
  parity.
- **Reset drift:** config currently derives a reset-endpoint list for
  multi-site tasks, while the technical spec still uses “one reset endpoint”
  wording. Resolve this semantic discrepancy before making reset part of a
  public adapter protocol.
- **Dependency and artifact drift:** WebArena's evaluator environment and
  AgentLab's sidecar both have independent dependency graphs. Subprocess
  stdout noise, stale artifacts, timeout recovery, auth scope, and resume
  fingerprints are part of the acceptance contract, not polish.
- **Overloaded adapter objects:** combining task loading, credentials, site
  writes, reset, browser execution, native reward, and WARP reward would violate
  feature locality and make comparison-only gating easy to bypass.

Reusable current work includes the capability registry, surface-identity
resolver, editor registry, vendor evaluator package, `AgentRunRequest`, and
AgentLab sidecar. Reusable prior work is limited to task-conversion details and
fixtures after revalidation against current schema and site scope.

## Questions to grill before implementation

1. Is the target a WARP Phase 4 benchmark or a comparison-only benchmark, and
   which result namespace proves that distinction?
2. Which evaluator is authoritative for each task (canonical vendor,
   benchmark-native, or WARP-local), and what artifact/protocol proves parity?
3. What exactly does “reset” mean for this benchmark: site state, browser
   context, BrowserGym task seed, or all three? Who calls each one, once?
4. Are task loading, placeholder expansion, credentials, and native task names
   benchmark facts or separate feature modules? What is the smallest stable
   contract for each?
5. Can the benchmark enter Phase 2/2c, or is it comparison-only by design? What
   test fails closed if metadata is mixed or a caller bypasses the capability
   gate?
6. Which canonical IDs remain stable across benchmark profiles, and which
   mappings are ambiguous enough to require route/editor context?
7. What live infrastructure, dependency environment, and reset/readback proof
   must exist before a slice can produce publishable data?
8. How will a new runner preserve WARP artifact, PVPO, auth, timeout/retry, and
   resume semantics without relabelling native actions or rewards?
