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
- Phase 4 varies adversarial strategy only; benign task contracts and rewards stay invariant.
- PVPO `max_coverage > 0` is the encounter gate.
- Refusal judge measures attack effectiveness.
- Transcript Purpose and VEA are observational metrics only.

## Current Run Records

- Technical source of truth: `docs/worldsim-v5-technical-specifcation.md`
- Domain invariants: `agent_docs/domain-invariants.md`
- r5 and rigor runbook: `docs/handoffs/rigor-run-setup.md`
- Phase 2c floor: `docs/handoffs/phase-2c-admission-floor.md`
- Expanded Phase 4 sweep, title-inclusive historical cohort: `docs/phase4-expanded-model-sweep-20260501.md`
- No-title mainline sweep config: `configs/phase4_model_sweeps/phase4_20260502_no_title.json`
- OpenRouter smoke matrix: `docs/phase4-openrouter-agent-model-smoke.md`

## Operator Rule

Do not treat old handoffs or TODO files as live instructions unless their top banner says they are active. For current implementation work, read the spec plus the relevant `agent_docs/` file first.
