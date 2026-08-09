# TODO: rename `worldsim.phases.phase_2_injections_api` to `worldsim.phase_2.runner_api`

> **ACTIVE DEFERRED RENAME TODO.** Structural rename only; no Phase 2 behavior,
> admission, or Phase 4 changes belong in this task.
> Start any implementation from current `origin/main` in a short-lived
> worktree and edit the canonical `packages/warp-taskgen/` source only.

## Status

**Deferred from PR #11 and PR #13.** Both PRs explicitly punted this rename to a later migration cycle so the cutover blast radius stayed scoped to the six pure-shim wrappers.

## Why this is still pending

When PR #11 was rebased onto the then-current authoring branch, the canonical Shape-C streaming L3 implementation lived at `worldsim/phases/phase_2_injections_api.py` (542 lines), not at the previously planned `worldsim/phase_2/runner_api.py`. PR #11's delegation test pinned this with the comment:

> `phase_2_injections_api` remains the canonical Shape-C streaming L3 implementation; the PR #11 rename to `worldsim.phase_2.runner_api` was deferred to a later migration cycle.

PR #13's cutover therefore landed against **6 wrappers** instead of 7. The audit guard's `LEGACY_PHASE_IMPORT_MODULES` set in `scripts/readiness_audit.py` excludes `worldsim.phases.phase_2_injections_api` so canonical callers in `worldsim/phase_2/_context.py` keep working.

## Work this rename PR must carry

1. **Move the implementation.** Relocate the 542 lines of `worldsim/phases/phase_2_injections_api.py` to `worldsim/phase_2/runner_api.py`. Verify the module-internal imports (e.g. `from worldsim._sandbox_validator import ...`) still resolve cleanly under the new package path.
2. **Migrate callers.** At the time of writing the only canonical in-source caller is `worldsim/phase_2/_context.py:61` (`from worldsim.phases.phase_2_injections_api import generate_phase_2a_plans_api`). Repoint it to `from worldsim.phase_2.runner_api import generate_phase_2a_plans_api`. Run `git grep 'phase_2_injections_api'` to catch any new callers added after this TODO was filed.
3. **Migrate tests.** `tests/test_prompt_contract_renderer.py:275` uses `from worldsim.phases import phase_2_injections_api`. Update it (or the equivalent shape) to import from `worldsim.phase_2.runner_api`.
4. **Add a one-cycle compat shim.** Mirror PR #11's pattern: leave a tiny `worldsim/phases/phase_2_injections_api.py` that does `sys.modules[__name__] = worldsim.phase_2.runner_api` for one migration cycle so out-of-tree consumers do not break instantly. The shim and its delegation test fall away in the cutover PR that follows this rename.
5. **Expand the cutover audit.** Add `worldsim.phases.phase_2_injections_api` to `LEGACY_PHASE_IMPORT_MODULES` in `scripts/readiness_audit.py` and remove the explanatory comment block PR #13 added beneath the set. Update `tests/test_phase_compat_wrappers.py` if its assertions need to expand to seven modules.
6. **Revisit the `worldsim/phase_4/__init__.py` re-export.** PR #13 attempted to re-export `phase_4_variant_budget_choices` through the public package surface but reverted because it introduced a circular import via `worldsim.phases.phase_2_text_fill`. Once the canonical content of `phase_2_injections_api` moves into `worldsim.phase_2.runner_api`, audit whether `worldsim/phase_4/_context.py:81` still pulls in `phase_2_text_fill` at module-init time. If not, re-attempt the public re-export so `worldsim/main.py:33` can use `from worldsim.phase_4 import phase_4_variant_budget_choices` instead of reaching into the private `_context`.

## Verification commands the rename PR should run

```bash
# Unit-level smoke after the move.
uv run pytest tests/test_phase_compat_wrappers.py tests/test_readiness_audit.py tests/test_prompt_contract_renderer.py -q

# Confirm the audit catches the new legacy entry.
uv run python scripts/readiness_audit.py --fail-on legacy-imports --json | rg legacy_phase_imports

# Confirm Phase 2a still resolves.
uv run python -c "from worldsim.phase_2.runner_api import generate_phase_2a_plans_api; print(generate_phase_2a_plans_api)"

# Live r8a smoke before merging the whole chain.
bash scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r8a.local.yaml --quiet
```

## Out of scope for the rename PR

- No Phase 2a behavior changes. The rename is structural only.
- No `phase_2_injections_api` API surface changes. Re-export the same symbols the current module exports.
- No further restructuring of `worldsim/phase_2/` beyond moving the one file. The modular split landed in PR #10 / PR #11; this rename only finishes the deferred relocation.

## Pointers

- Canonical content (current location): `worldsim/phases/phase_2_injections_api.py`.
- Caller relying on the canonical path: `worldsim/phase_2/_context.py:61`.
- Audit set: `scripts/readiness_audit.py` (`LEGACY_PHASE_IMPORT_MODULES`).
- PR #11 delegation pin: `tests/test_phase_compat_wrappers.py` (`test_legacy_phase_helper_imports_delegate_to_canonical_functions`) on the merged branch.
- PR #13 cutover handoff: `docs/handoffs/pr-13-agent-readiness-cutover.md`.
