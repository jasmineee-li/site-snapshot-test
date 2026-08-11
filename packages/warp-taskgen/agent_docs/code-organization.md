# Code Organization

Use this when moving code, adding modules, or reviewing agent-readiness debt.

## Feature Ownership

Organize code by feature or domain, not by technical layer. A module should
answer "what behavior owns this?" before it answers "what kind of code is this?"

Current and target ownership should stay explicit:

- `worldsim.phase_2`: injection generation, target resolution, exposure
  contracts, text fill, and Phase 2c admission. Route/exposure/admission
  semantics still belong to the Phase 2 domain.
- `worldsim.phase_2.text_fill`: host-side payload realization behavior split by
  API calls, prompt rendering, payload views, seeding, validation, and voice
  exemplars. The legacy `worldsim.phases.phase_2_text_fill` path remains a
  patchable compatibility facade during migration; do not treat it as removable
  until in-repo imports and hidden consumers have moved.
- `worldsim.phase_2.phase_2c`: Phase 2c feasibility verification split by
  report types, constants, fingerprints, outcome stanzas, exposure projection,
  reddit attribution, admission guards, render/reachability probes,
  source-data preflight, per-task verification, and runner orchestration. The
  legacy `worldsim.phases.phase_2_feasibility` path is the patchable
  compatibility surface during migration.
- `worldsim.phase_2.exposure_contract`: deterministic Phase 2 exposure
  contracts split by signature, builder, seed-template materialization, exposure
  modes, candidate selection, Phase 4 exposure gates, route metadata,
  verification contracts, and editor argument templates. The legacy
  `worldsim.phases.phase_2_exposure_contract` path is a compatibility facade.
- `worldsim.phase_4`: adversarial execution, PVPO placement, postprocess judges,
  strategy variation, intermediate ASR post-hoc judging, resume, and results.
  Phase 4 must not own benign task eligibility.
- `worldsim.phase_4.result_summary`: Phase 4 result aggregation split by final
  metrics, task metadata labels, inspection index rows, action-tier metrics,
  variant regeneration audit, and top-level summary assembly.
- `worldsim.outcome_taxonomy`: Phase 4 trajectory classification split by signal
  extraction, read-surface matching, engagement checks, classification,
  stratified summaries, disk IO, and result-field serialization.
- `worldsim.phase_1.novel_task_validation`: Phase 1 generated-task validation
  split by batch entry points, single-task orchestration, route alignment,
  task-card alignment, placement target checks, reward checks, answer stability,
  and ordering/eligibility. The legacy
  `worldsim.phases.phase_1_generate_new_tasks_validation` path is only a
  compatibility facade.
- `worldsim.adversarial_actions`: host-owned adversarial action behavior split by
  policies, allowed options, reward compilation, public mutation rewards,
  final-state compilers, and reward introspection. Keep the old
  `compiler.py` import surface as a facade while migrating callers to the
  behavior-owned modules.
- Phase 0c profile rigor is split by behavior:
  `worldsim/phases/phase_0_recon.py` remains the compatibility runner,
  `worldsim/phases/phase_0_evidence_index.py` owns neutral source indexes,
  `worldsim/phases/phase_0c_artifacts.py` owns provenance/reuse/trace artifacts,
  and `worldsim/phases/phase_0c_audit.py` owns deterministic host audits. Do not
  move these concerns into Modal sandbox setup; `modal_sandbox.py` stays an
  infrastructure-only runner.
- `worldsim.seed_contracts`: shared seed/editor-call contract behavior used by
  Phase 2, Phase 4, seeding, and sandbox validation. This package must preserve
  sandbox packaging constraints and parity tests.
- `worldsim.benchmark_capabilities`: immutable Benchmark Contract identity,
  alias normalization, explicit WARP phase admission versus comparison-only
  ingestion, runner declarations, and evaluator-authority decisions. It must
  not own Site HTTP behavior, auth/browser/reset lifecycle, runner execution,
  evaluator subprocesses, or reward/scoring implementations. Registration is
  metadata only; new callers must require the named capability before
  admission. Legacy phase flags remain derived compatibility readers during
  the current migration cycle.
- `worldsim.comparison_ingestion`: native AgentLab comparison payload
  validation, immutable comparison-result envelopes, provenance/artifact
  references, and atomic `comparison_result.json` persistence. This module
  must not import AgentLab, start browsers, reset Sites, enter WARP phase
  admission, or dispatch rewards. The AgentLab runner owns subprocess and
  reset orchestration and calls this module only after a native `run`.
- `worldsim.run_definition`: immutable, non-secret Run Definition projection,
  deterministic Definition Digests, and read-only Resume Plans.
  `worldsim.run_transition` owns pure Run-transition decisions, while
  `worldsim.cli.run_identity` normalizes shared CLI defaults and binds those
  decisions to dispatch. `worldsim.state` owns the state-root-scoped context
  that persists an exact Run Definition into each atomic checkpoint, while the
  CLI consumes transition decisions. This seam must not accept checkpoints or
  replace Phase 2/4 fingerprint policies. `worldsim.run_materialization` owns
  the atomic reservation and isolated child-root initialization for definition
  drift. It must not copy or accept feature checkpoints, mutate the source Run,
  update the shared discovery pointer, or dispatch the child automatically.
- `worldsim.run_control`: non-secret cooperative pause requests and
  paused/interrupted lifecycle transitions. Normal Phase 4 scheduling remains
  here; `worldsim.phase_2.pause_control` owns the Phase 2a planning queue and
  its Run-bound shard manifests. Phase 2b text fill and Phase 2c feasibility do
  not yet accept pause.
  It must not cancel admitted browser/API work, accept feature checkpoints,
  reinterpret `progress.json` as routing authority, or silently extend pause to
  Phases 0, 1, or 3. `worldsim.phase_4.process_pool` owns its supervisor-specific
  claim/launch boundary and worker/result orchestration;
  `worldsim.phase_4.process_pool_control` owns root-local lifecycle metadata,
  output ownership, and the explicit wrapper continuation contract. Child
  runner and result-merge acceptance remain unchanged.
  `worldsim.cli.run_control` is the thin parser/dispatch adapter that handles
  operator output and catchable process signals after the phase stack unwinds.
- `worldsim.seeding`: host-side seed validation, context rendering, editor-call
  execution, read-surface/result metadata, reddit/map context resolution,
  runtime error validation, DB helpers, and editor-argument compatibility. This
  package remains a patchable public surface because many tests and hidden
  callers monkeypatch `worldsim.seeding` directly.
- `worldsim.cli`: WorldSim CLI parser, dispatch, phase-4 lock handling,
  verification proxy setup, task-bank commands, resume flow, and auth
  validation. `worldsim.main` remains the executable compatibility entrypoint.
- `worldsim.rewards`: reward dispatch and scoring behavior. Keep the public
  facade thin; put behavior in reward-local modules by evidence type and
  benchmark surface. Request-level evidence belongs in `network_event.py` and
  `network_trace.py`; generic persisted-readback validation and catalog
  dispatch belongs in `final_state.py`, immutable local evaluator composition
  belongs in `final_state_catalog.py`, and Benchmark/Site orchestration belongs
  in the corresponding `final_state_*_adapter.py` module. WebArena Verified
  GitLab and Reddit/Postmill transport and exact readback behavior remains in
  `final_state_webarena_verified_gitlab.py` and
  `final_state_webarena_verified_reddit.py`; vendor adapter shims belong in
  `vendor_webarena.py`; non-scoring attempt telemetry belongs in
  `action_attempt.py`.
- `worldsim.browser_use`: Browser Use runtime concerns when that runner is split.
- `worldsim.sandbox_validator`: sandbox/profile/task validation when that module
  is split. This domain has a stricter Modal runtime contract than ordinary host
  modules and should not be mechanically extracted.

Avoid `utils.py`, `helpers.py`, and global shared `types.py`. If a helper is
shared, name the domain it belongs to. Keep types next to the behavior that owns
them until multiple sibling modules need them; then move them to a feature-local
types module.

## File Size

The advisory review ceiling follows `scripts/readiness_audit.py`: 550 actual
lines is review debt and 1200 lines is urgent split debt. The 150-line floor is only a guard
against over-splitting; small files are fine for entrypoints, adapters,
protocols, `__init__` modules, and narrow local type holders.

Split at natural behavioral boundaries. Do not split a linear algorithm into
arbitrary chunks just to satisfy a number.

True exemptions should be documented in `scripts/readiness_audit.py` with a
reason. Deferred monoliths are not exemptions; keep them visible in readiness
reports until they are split.

## Tests

Keep pytest tests under top-level `tests/`, mirrored by domain where practical:

- `tests/phase_2/`
- `tests/phase_4/`
- `tests/seed_contracts/`
- `tests/test_browser_use_agent.py` and other Browser Use-focused files
- `tests/test_sandbox_validator.py`

Tests may import feature-private helpers when those helpers encode important
contracts, but they should import them from the owning feature package rather
than from compatibility wrappers once a package split exists.

## Compatibility Wrappers

When moving public or widely imported modules, keep the old import path as a
thin compatibility wrapper for one migration cycle. Wrappers may delegate and
re-export canonical names, but they should not own behavior or contain a second
implementation. They also should not monkeypatch dependencies, except for the
explicitly documented patchable compatibility surfaces that preserve old test
and hidden-consumer behavior during the current migration window.

Update in-repo imports to the canonical feature package during the same change.
Remove wrappers in a follow-up cleanup once hidden consumers have had a chance
to surface.

Wrappers are temporary agent-DX debt. They are worth paying when the refactor
touches research-critical paths because they separate behavior migration from
import-path cutover. They should have tests that prove legacy imports delegate,
and a follow-up PR should remove them when the migration window closes.

Package-backed parity modules named `_impl.py` are also transition debt. They
preserve behavior during a clean import cutover, but new behavior should land in
the behavior-owned sibling module when practical. When editing an `_impl.py`
area, prefer moving the touched function into the sibling module that owns the
behavior and leaving a re-export behind, rather than growing `_impl.py` further.

## Current Follow-Up Debt

The Phase 2 and Phase 4 modularization PR intentionally leaves a small amount of
transition debt so the main behavior split remains reviewable. Do not treat all
readiness debt as equally urgent or equally safe to fold into the same PR. Good
sequencing reduces review risk:

- First split behavior into domain-owned modules while preserving external
  import compatibility.
- For package cutovers that still have `_impl.py` parity modules, drain those
  modules incrementally. Move one behavioral cluster at a time into its sibling
  module and run the focused tests before deleting any facade or re-export.
  Priority order: Phase 2c feasibility, seeding execution, exposure contracts,
  Phase 4 result-summary, outcome taxonomy, Phase 1 validation, then CLI.
- Then remove pure compatibility wrappers in follow-up changes after downstream
  imports are moved and one validation cycle has had a chance to reveal hidden
  consumers. Some `worldsim.phases.*` modules are still patchable compatibility
  surfaces, not pure shims. Do not delete those wholesale. Update remaining
  internal and downstream imports to canonical `worldsim.phase_2.*`,
  `worldsim.phase_4.*`, and `worldsim.seed_contracts.*` paths, then delete only
  wrappers that `scripts/readiness_audit.py` identifies as legacy import
  modules.
- Unwind linked-context modules after wrapper removal. The `_context.py`
  `install_context` / `link_modules` pattern is a transition mechanism that
  preserves old monolith global lookup and monkeypatch behavior during the
  split. Replace it with explicit imports domain by domain, starting with target
  resolution because it is smaller and has focused tests.
- Split sandbox validation as a separate design task.
  `worldsim/_sandbox_validator.py` is urgent by size, but it runs inside Modal
  with stdlib-only and no-`worldsim` import constraints. Sharing seed-contract
  behavior there needs either a generated standalone validator or a deliberately
  shipped sandbox validation package, plus parity tests against
  `worldsim.seed_contracts`.
- Keep other large files visible as debt instead of allowlisting them. Examples
  include the Browser Use runner, seeding, GitLab editor, Phase 2 feasibility,
  Phase 2 exposure contracts, the main CLI, and outcome taxonomy.

Only fold one of these follow-ups into an active PR when it reduces review risk
for that PR. If it expands the behavioral surface under review, keep it separate.
