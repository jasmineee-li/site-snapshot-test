# WorldSim v5 Current Status

> **Current-status index, rewritten 2026-05-02.** This file summarizes only the current canonical pipeline state and points to durable run records. The pre-PVPO / pre-WASP chronological log formerly stored here was archived to `docs/handoffs/archive/current_progress_pre_wasp_20260417.md`.

## Current Scope

WorldSim v5 mainline is strict WASP scope:

- GitLab issues and issue comments/notes
- Reddit/Postmill submissions and comments

GitLab merge requests are historical/support plumbing, not active mainline
carriers. Shopping, shopping_admin, Wikipedia, map/OSM, Magento, and classifieds
are historical or support plumbing unless the spec explicitly reopens scope.

## Current Gates

- Phase 2c strict feasibility is admission for Phase 4.
- Phase 4 keeps benign task contracts and rewards invariant across baseline,
  eval-awareness iterator rewrites, and legacy strategy variants.
- PVPO `max_coverage > 0` is the encounter gate.
- Attack effectiveness stays reward/PVPO-grounded.
- Transcript Purpose and VEA are observational in baseline scoring. The
  explicit `eval-awareness-iterator` is scheduled for every PVPO-valid
  default-path baseline trajectory and uses Transcript Purpose only as bounded
  rewrite control; baseline ASR still comes from reward/PVPO. VEA stays
  report-only.

## Current Run Records

- Technical source of truth: `docs/worldsim-v5-technical-specifcation.md`
- Domain invariants: `agent_docs/domain-invariants.md`
- r5 and rigor runbook: `docs/handoffs/rigor-run-setup.md`
- Phase 2c floor: `docs/handoffs/phase-2c-admission-floor.md`
- Expanded Phase 4 sweep, title-inclusive historical cohort: `docs/phase4-expanded-model-sweep-20260501.md`
- No-title mainline sweep config: `configs/phase4_model_sweeps/phase4_20260502_no_title.json`
- OpenRouter smoke matrix: `docs/phase4-openrouter-agent-model-smoke.md`
- Phase 4 eval-awareness iterator renderer fix: validated 2026-05-07 on r5 in
  `logs/gitlab_create_issue_note_renderer_rerun_20260507T0200Z` at
  `258570f16203`. The narrowed GitLab `create_issue_note` Tier 2 rerun
  generated and browser-evaluated all 6 / 6 renderer-backed iterator variants
  with 6 / 6 PVPO-valid variants, 0 generation failures, and final status
  `resistant=2`. This replaces the earlier `unchanged_seed` iterator failure
  as the current evidence for that action family.

## Operator Rule

Do not treat old handoffs or TODO files as live instructions unless their top banner says they are active. For current implementation work, read the spec plus the relevant `agent_docs/` file first.
