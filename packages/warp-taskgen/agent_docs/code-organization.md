# Code Organization

Use this when moving code, adding modules, or reviewing agent-readiness debt.

## Feature Ownership

Organize code by feature or domain, not by technical layer. A module should
answer "what behavior owns this?" before it answers "what kind of code is this?"

Current and target ownership should stay explicit:

- `warp_taskgen.phase_2`: injection generation, target resolution, exposure
  contracts, text fill, and Phase 2c admission. Route/exposure/admission
  semantics still belong to the Phase 2 domain.
- `warp_taskgen.phase_2.text_fill`: host-side payload realization behavior split by
  API calls, prompt rendering, payload views, seeding, validation, and voice
  exemplars.
- `warp_taskgen.phase_2.phase_2c`: Phase 2c feasibility verification split by
  report types, constants, fingerprints, outcome stanzas, exposure projection,
  reddit attribution, admission guards, render/reachability probes,
  source-data preflight, per-task verification, and runner orchestration. The
  loop takes its collaborators through `probe_bundle.Phase2cProbeBundle`.
- `warp_taskgen.phase_2.exposure_contract`: deterministic Phase 2 exposure
  contracts split by signature, builder, seed-template materialization, exposure
  modes, candidate selection, Phase 4 exposure gates, route metadata,
  verification contracts, and editor argument templates.
- `warp_taskgen.phase_4`: adversarial execution, PVPO placement, postprocess judges,
  strategy variation, intermediate ASR post-hoc judging, resume, and results.
  Phase 4 must not own benign task eligibility. The eval-awareness iterator keeps
  the sequential loop in `eval_awareness_iterator.py` and reads three siblings:
  `eval_awareness_cue_diagnosis.py` owns iteration triggers, protected witnesses,
  and cue-diagnosis normalization; `eval_awareness_iteration_feedback.py` owns
  prior-iteration feedback and contract-QA rejection; and
  `eval_awareness_iterator_budget.py` owns budget accounting, iteration selection,
  stop reasons, and the checkpoint-to-result projection.
- `warp_taskgen.phase_4.result_summary`: Phase 4 result aggregation split by final
  metrics, task metadata labels, inspection index rows, action-tier metrics,
  variant regeneration audit, and top-level summary assembly.
- `warp_taskgen.outcome_taxonomy`: Phase 4 trajectory classification split by signal
  extraction, read-surface matching, engagement checks, classification,
  stratified summaries, disk IO, and result-field serialization.
- `warp_taskgen.phase_1.novel_task_validation`: Phase 1 generated-task validation
  split by batch entry points, single-task orchestration, route alignment,
  task-card alignment, placement target checks, reward checks, answer stability,
  and ordering/eligibility.
- `warp_taskgen.adversarial_actions`: host-owned adversarial action behavior split by
  policies, allowed options, reward compilation, public mutation rewards,
  final-state compilers, and reward introspection. Import from the
  behavior-owned module or the package `__init__`; there is no `compiler.py`
  facade.
- Phase 0c profile rigor is split by behavior:
  `warp_taskgen/phases/phase_0_recon.py` remains the compatibility runner,
  `warp_taskgen/phases/phase_0_evidence_index.py` owns neutral source indexes,
  `warp_taskgen/phases/phase_0c_artifacts.py` owns provenance/reuse/trace artifacts,
  and `warp_taskgen/phases/phase_0c_audit.py` owns deterministic host audits. Do not
  move these concerns into Modal sandbox setup; `modal_sandbox.py` stays an
  infrastructure-only runner. Phase 0d auth bootstrap is split the same way:
  `warp_taskgen/phases/phase_0d_auth_bootstrap.py` remains the runner,
  `phase_0d_site_auth_specs.py` owns `AuthBootstrapError` and the per-site auth
  spec parsers, `phase_0d_generator_dispatch.py` owns input hashing, dispatch
  selection, generator loading, and declared-artifact trust, and
  `phase_0d_form_login.py` owns the built-in Playwright form-login bootstrap.
- `warp_taskgen.sites`: Site Targeting, profile identity, and carrier policy.
  Core-surface and active-carrier policy is Site-owned (`SiteCarrierPolicy` on
  each `*_profile.py` mixin) and reached only through
  `BoundSite.carrier_policy()`; a Site without the capability binds closed.
- `warp_taskgen.seed_contracts`: shared seed/editor-call contract behavior used by
  Phase 2, Phase 4, seeding, and sandbox validation. This package must preserve
  sandbox packaging constraints and parity tests.
- `warp_taskgen.benchmark_capabilities`: immutable Benchmark Contract identity,
  alias normalization, explicit WARP phase admission versus comparison-only
  ingestion, runner declarations, and evaluator-authority decisions. It must
  not own Site HTTP behavior, auth/browser/reset lifecycle, runner execution,
  evaluator subprocesses, or reward/scoring implementations. Registration is
  metadata only; new callers must require the named capability before
  admission. Legacy phase flags remain derived compatibility readers during
  the current migration cycle.
- `warp_taskgen.site_composition`: static closure of immutable, data-only
  `SiteComposition` declarations for one exact Site, Benchmark, use case,
  carrier, and action kind. `site_composition_contracts.py` owns the semantic
  owner declarations, request, finding, digest, and static-only report values;
  the Host-Owned use-case catalog owns required owner roles and derives
  `not_applicable`; `site_compositions/<site>.py` keeps each Site's pure
  declaration local, and `site_composition_defaults.py` only aggregates the
  GitLab, Reddit, and explicit Classifieds diagnostic declarations. This seam
  reads declarations only. Executable owner behavior stays in its feature
  module and is tested through `tests/sites/behavior_contract/` with fake adapters. The
  only CLI adapter is `site composition check`. Static completion does not
  grant active policy or prove live evidence.
- `warp_taskgen.comparison_ingestion`: native AgentLab comparison payload
  validation, immutable comparison-result envelopes, provenance/artifact
  references, and atomic `comparison_result.json` persistence. This module
  must not import AgentLab, start browsers, reset Sites, enter WARP phase
  admission, or dispatch rewards. The AgentLab runner owns subprocess and
  reset orchestration and calls this module only after a native `run`.
- `warp_taskgen.run_definition`: immutable, non-secret Run Definition projection,
  deterministic Definition Digests, and read-only Resume Plans.
  `warp_taskgen.run_transition` owns pure Run-transition decisions, while
  `warp_taskgen.cli.run_identity` normalizes shared CLI defaults and binds those
  decisions to dispatch. `warp_taskgen.state` owns the state-root-scoped context
  that persists an exact Run Definition into each atomic checkpoint, while the
  CLI consumes transition decisions. This seam must not accept checkpoints or
  replace Phase 2/4 fingerprint policies. `warp_taskgen.run_materialization` owns
  the atomic reservation and isolated child-root initialization for definition
  drift. It must not copy or accept feature checkpoints, mutate the source Run,
  update the shared discovery pointer, or dispatch the child automatically.
- `warp_taskgen.run_control`: non-secret cooperative pause requests and
  paused/interrupted lifecycle transitions. Normal Phase 4 scheduling remains
  here; `warp_taskgen.phase_2.pause_control` owns the Phase 2a planning queue and
  its Run-bound shard manifests. `warp_taskgen.phase_2.text_fill` owns the Phase
  2b task queue and Run-bound text-fill checkpoints. The feature-owned
  `warp_taskgen.phase_2.phase_2c.pause_control` owns bounded preflight admission,
  verification claims, drain behavior, and the aggregate-promotion boundary;
  its task checkpoints remain owned by `phase_2.phase_2c.checkpoints`.
  It must not cancel admitted browser/API work, accept feature checkpoints,
  reinterpret `progress.json` as routing authority, or silently extend pause to
  Phases 0, 1, or 3. `warp_taskgen.phase_4.process_pool` owns its supervisor-specific
  claim/launch boundary and worker/result orchestration;
  `warp_taskgen.phase_4.process_pool_control` owns root-local lifecycle metadata,
  output ownership, and the explicit wrapper continuation contract. Child
  runner and result-merge acceptance remain unchanged.
  `warp_taskgen.cli.run_control` is the thin parser/dispatch adapter that handles
  operator output and catchable process signals after the phase stack unwinds.
- `warp_taskgen.sites`: Site-owned render-probe behavior lives beside the other
  Site modules. `render_probe.py` is the Site-neutral leaf (`RenderOutcome`, the
  one text normalizer, the one body-text wait, the one same-origin check) and
  must not be re-exported from `sites/__init__.py`; `gitlab_render_probe.py` and
  `reddit_render_probe.py` own their Site's fast paths.
  `phases/phase_2_render_check.py` keeps `verify_seed_renders` and reaches them
  through a Site-keyed lookup, and `phases/phase_2_reachability.py` imports the
  same-origin check from the leaf so no `sites` module imports back into
  `warp_taskgen.phases`.
- `warp_taskgen.seeding`: host-side seed validation, context rendering, editor-call
  execution, read-surface/result metadata, reddit/map context resolution,
  runtime error validation, DB helpers, and editor-argument compatibility.
  Behavior lives in the sibling module that owns it and `__init__.py` re-exports
  an explicit, bounded surface. Patch the owning sibling (for example
  `warp_taskgen.seeding.execution`), not the package root.
  `site_contracts.default_seed_registry()` builds the default GitLab/Reddit seed
  binding a Run uses when it does not carry its own.
- `warp_taskgen.cli`: the WARP Taskgen CLI, split by owner: the parser
  (`args`), the import-time dotenv bootstrap that runs first (`env`), dispatch
  (`dispatch`), resume (`resume`, `resume_plan`, `derived_run`,
  `run_identity`), the Phase 4 run lock and bounded async shutdown
  (`phase4_lock`), verification-proxy setup (`proxy`), task-bank commands
  (`task_bank`), unknown-auth validation (`auth`), pause and lifecycle
  operator output (`run_control`), the status and inspect projections
  (`status`), and the static Site Composition check (`site_composition_check`).
  The parser is `args` assembling the root parser and the small commands plus
  the `phase_arguments`, `resume_arguments`, `agentlab_arguments`, and
  `task_bank_arguments` siblings that register one command group each, with
  `argument_types` (argparse `type=` validators) and `argument_defaults` (agent
  model and provider defaults) beside them.
  `warp_taskgen.main` is the console entrypoint only; tests import and patch the
  owning `cli.*` module.
- `warp_taskgen.rewards`: reward dispatch and scoring behavior. Keep the public
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
- `warp_taskgen.browser_use`: Browser Use runtime concerns when that runner is split.
- `warp_taskgen.runners`: the AgentLab runner keeps the agent wrapper, reset, and
  task runner; its four `agentlab_*` siblings own sidecar request construction,
  sidecar process and result parsing, sidecar redaction, and Phase 4 artifacts.
- `warp_taskgen.sandbox_validator`: sandbox/profile/task validation when that module
  is split. This domain has a stricter Modal runtime contract than ordinary host
  modules and should not be mechanically extracted.

There is no `worldsim.phases.*` facade and no executable `worldsim` import in
the tree; `agent_docs/verification.md` names the readiness check that keeps it
that way.

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
implementation. They also should not monkeypatch dependencies; no package
`__init__` is a patchable surface.

Update in-repo imports to the canonical feature package during the same change.
Remove wrappers in a follow-up cleanup once hidden consumers have had a chance
to surface.

Wrappers are temporary agent-DX debt. They are worth paying when the refactor
touches research-critical paths because they separate behavior migration from
import-path cutover. They should have tests that prove legacy imports delegate,
and a follow-up PR should remove them when the migration window closes.

Do not add a package-backed `_impl.py` parity module. No package has one:
behavior lives in the sibling module that owns it and the package `__init__`
re-exports a bounded surface.

## Current Follow-Up Debt

The Phase 2 and Phase 4 modularization PR intentionally leaves a small amount of
transition debt so the main behavior split remains reviewable. Do not treat all
readiness debt as equally urgent or equally safe to fold into the same PR. Good
sequencing reduces review risk:

- First split behavior into domain-owned modules while preserving external
  import compatibility.
- If a package cutover ever needs a parity module again, drain it
  incrementally: move one behavioral cluster at a time into its sibling module
  and run the focused tests before deleting any facade or re-export. No package
  carries one today.
- Then remove pure compatibility wrappers in follow-up changes after downstream
  imports are moved and one validation cycle has had a chance to reveal hidden
  consumers. No `warp_taskgen.phases.*` module is a patchable compatibility
  surface; patch the module that defines the behavior, or the module under
  test where it binds an imported name.
- The `_context.py` `install_context` / `link_modules` linkage is gone. The
  `tests/**/test_*_context_boundary.py` suite, run by the root
  `core-context-boundaries` lane, asserts it never returns.
- Split sandbox validation as a separate design task.
  `warp_taskgen/_sandbox_validator.py` is urgent by size, but it runs inside Modal
  with stdlib-only and no-`warp_taskgen` import constraints. Sharing seed-contract
  behavior there needs either a generated standalone validator or a deliberately
  shipped sandbox validation package, plus parity tests against
  `warp_taskgen.seed_contracts`.
- Keep other large files visible as debt instead of allowlisting them. Examples
  include the Browser Use runner, seeding, GitLab editor, Phase 2 feasibility,
  Phase 2 exposure contracts, the CLI parser, and outcome taxonomy.

Only fold one of these follow-ups into an active PR when it reduces review risk
for that PR. If it expands the behavioral surface under review, keep it separate.
