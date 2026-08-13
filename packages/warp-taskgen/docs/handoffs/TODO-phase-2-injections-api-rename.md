# Completed: rename `worldsim.phases.phase_2_injections_api` to `worldsim.phase_2.runner_api`

> **COMPLETED IN ISSUE #84.** The rename was structural only; no Phase 2
> behavior, admission, or Phase 4 behavior changed. The historical import path
> remains a one-cycle compatibility alias.

## Status

**Implemented after PR #11 and PR #13.** Both PRs deferred this rename so the
cutover blast radius stayed scoped to the six pure-shim wrappers. Issue #84
completed the deferred migration.

## Historical context

When PR #11 was rebased onto the then-current authoring branch, the canonical Shape-C streaming L3 implementation lived at `worldsim/phases/phase_2_injections_api.py` (542 lines), not at the previously planned `worldsim/phase_2/runner_api.py`. PR #11's delegation test pinned this with the comment:

> `phase_2_injections_api` remains the canonical Shape-C streaming L3 implementation; the PR #11 rename to `worldsim.phase_2.runner_api` was deferred to a later migration cycle.

PR #13's cutover therefore landed against **6 wrappers** instead of 7. Issue
#84 now places the historical module in the audit inventory and migrates the
caller in `worldsim/phase_2/_context.py` to the feature-owned path.

## Delivered change

1. The 543-line implementation now lives at `worldsim/phase_2/runner_api.py`.
2. In-repository callers and tests use `worldsim.phase_2.runner_api`.
3. `worldsim/phases/phase_2_injections_api.py` is a `sys.modules` alias that
   preserves historical module and symbol identity for one migration cycle.
4. `scripts/readiness_audit.py` inventories the historical path as a legacy
   import, while the compatibility test proves the alias delegates.
5. `worldsim.phase_4.phase_4_variant_budget_choices` is available through the
   lazy public package surface without importing the Phase 4 context.

## Verification commands the rename PR should run

```bash
# Unit-level smoke for the completed move.
uv run pytest tests/test_phase_compat_wrappers.py tests/test_readiness_audit.py tests/test_prompt_contract_renderer.py -q

# Confirm the audit catches historical imports while production callers stay canonical.
uv run python scripts/readiness_audit.py --fail-on legacy-imports --json | rg legacy_phase_imports

# Confirm Phase 2a resolves from the feature-owned package.
uv run python -c "from worldsim.phase_2.runner_api import generate_phase_2a_plans_api; print(generate_phase_2a_plans_api)"

# Live r8a smoke before merging the whole chain.
bash scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r8a.local.yaml --quiet
```

## Scope guard

- No Phase 2a behavior changes; the rename is structural only.
- No `phase_2_injections_api` API surface changes; the canonical module exports
  the same symbols as the historical implementation.
- No further restructuring of `worldsim/phase_2/` beyond moving this file.

## Pointers

- Canonical content: `worldsim/phase_2/runner_api.py`.
- Compatibility alias: `worldsim/phases/phase_2_injections_api.py`.
- Canonical caller: `worldsim/phase_2/_context.py`.
- Audit set: `scripts/readiness_audit.py` (`LEGACY_PHASE_IMPORT_MODULES`).
- PR #11 delegation pin: `tests/test_phase_compat_wrappers.py` (`test_legacy_phase_helper_imports_delegate_to_canonical_functions`) on the merged branch.
- PR #13 cutover handoff: `docs/handoffs/pr-13-agent-readiness-cutover.md`.
