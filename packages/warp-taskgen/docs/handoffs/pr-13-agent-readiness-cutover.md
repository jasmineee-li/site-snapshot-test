# PR #13: Agent Readiness Wrapper Cutover

> **COMPLETED PR RECORD.** PR #13 cutover landed; use this for provenance only.
> Pending live gates must be checked in `docs/current_progress.md` and current
> runbooks.

## Summary

- Removes the six temporary Phase 2, Phase 2c helper, target-resolution, and Phase 4 compatibility wrappers under `worldsim.phases.*` after PR #11 restored them for one migration cycle.
- Adds a tracked import cutover guard to `scripts/readiness_audit.py` and wires it into `scripts/verify_fast.sh`. Detection covers absolute and relative import shapes.
- Updates the former compatibility-wrapper test to assert the wrappers are gone and tracked source has no legacy imports.
- Migrates the lone in-source caller (`worldsim/main.py`) and 14 test files to canonical paths.
- Lands as a single bisect-safe commit on top of v5; a follow-up commit covers post-cutover refinements (audit AST gap fix, prompt-renderer drift fix).

## Context

PR #13 targets `feat/worldsim-v5` (PR #11 merged 2026-05-05 at `8b58404c`). The cutover scope on this base is **6 wrappers, not 7**: PR #11's rebase onto post-PR-10 v5 left `worldsim.phases.phase_2_injections_api` as the canonical 542-line Shape-C streaming L3 implementation, not a shim. The deferred rename to `worldsim.phase_2.runner_api` is tracked in `docs/handoffs/TODO-phase-2-injections-api-rename.md`.

## Description

Removed compatibility wrappers (6):

- `worldsim.phases.phase_2_injections`
- `worldsim.phases.phase_2_output`
- `worldsim.phases.phase_2_target_resolver`
- `worldsim.phases.phase_2c_artifacts`
- `worldsim.phases.phase_2c_config`
- `worldsim.phases.phase_4_adversarial`

Excluded from cutover (kept on v5 as canonical):

- `worldsim.phases.phase_2_injections_api` — see TODO doc.

The readiness audit reports `legacy_phase_imports` and can fail with `--fail-on legacy-imports`. The AST detection covers:

- Absolute: `import worldsim.phases.phase_X`, `import worldsim.phases.phase_X as alias`, `from worldsim.phases.phase_X import Y`, `from worldsim.phases import phase_X`, `from worldsim.phases import phase_X, phase_Y`.
- Relative (only meaningful inside `worldsim/phases/`): `from . import phase_X`, `from .phase_X import Y`, `from ..phases import phase_X`. Resolution honors the file's package anchor and walks `level - 1` segments up before joining with the explicit `module` part.

The guard skips files under `docs/` and `tests/` (allowlist). Callers in those trees migrate as part of the cutover but are not gated by the audit.

`worldsim/main.py` previously reached the Phase 4 wrapper via `from worldsim.phases.phase_4_adversarial import phase_4_variant_budget_choices` (introduced after PR #10 merged). The cutover migrates that import to `worldsim.phase_4._context` directly. A re-export through `worldsim/phase_4/__init__.py` was attempted first but introduced a circular import via `worldsim.phases.phase_2_text_fill`; the public-surface refactor is captured in the deferred-rename TODO so it can land alongside the runner_api rename.

`tests/test_auth_mechanism.py` `monkeypatch.setattr` strings target the canonical `worldsim.phase_4.runner.run` rather than the deleted wrapper module path.

This PR intentionally makes no Phase 2c admission, Phase 4 encounter, reward, PVPO, eligibility, exposure, feasibility, target-resolution behavior, or WASP-scope changes.

## Bisect-safety

The cutover lands as a single commit. There is no intermediate state where wrappers are deleted but callers still reference them. `verify_fast.sh` and the broader pytest sweep are green at the cutover SHA and at the post-cutover-refinements SHA.

## Testing

- `uv run python -m worldsim.main --help` — passed
- `uv run python scripts/readiness_audit.py --fail-on legacy-imports --fail-on tracked-generated --fail-on tokens --json` — passed; `legacy_phase_imports=[]`, `token_findings=[]`, `tracked_generated=[]`
- `uv run pytest tests/test_phase_compat_wrappers.py tests/test_readiness_audit.py -q` — `19 passed` (17 cutover + 2 audit-AST-gap)
- `uv run pytest tests/test_phase_compat_wrappers.py tests/test_readiness_audit.py tests/test_auth_mechanism.py -q` — `71 passed`
- `uv run pytest tests/test_phase_compat_wrappers.py tests/test_readiness_audit.py tests/phase_2/target_resolution tests/phase_2/test_target_stage.py -q` — `157 passed`
- `uv run pytest tests/test_phase_compat_wrappers.py tests/test_readiness_audit.py tests/test_auth_mechanism.py tests/seed_contracts tests/phase_2 tests/phase_4 -q` — `518 passed`
- `uv run ruff check worldsim tests scripts` — passed
- `bash scripts/verify_fast.sh` — passed (ruff scoped + pytest collection + readiness audit all green)

## Live Gate / Pending

Live r5 smoke was not run. The r5 wrapper smoke remains required before merging the whole chain because it exercises the refactored Phase 2c, target resolution, seed-contract/config loading, and Phase 4 surfaces against the real browser/service topology.

```bash
bash scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r5.yaml --quiet
```

## Follow-up

- Run the r5 integration wrapper after active live work is clear.
- Land the deferred rename (`worldsim.phases.phase_2_injections_api` -> `worldsim.phase_2.runner_api`) per `docs/handoffs/TODO-phase-2-injections-api-rename.md`. That PR also expands the cutover audit set back to 7 entries and revisits the `worldsim/phase_4/__init__.py` re-export for `phase_4_variant_budget_choices`.
