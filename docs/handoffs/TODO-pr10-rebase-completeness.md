# TODO: Prove PR #10 rebase completeness (everything except live r5)

Status: closed (Gap 4 + Gap 13 + Gap 15 resolved 2026-05-05; rebased onto `feat/worldsim-v5@446ff6cc`)
Owner: unassigned
Worktree: `/Users/ashtonchew/projects/browser-sim/.codex-worktrees/agent-readiness-standardized` (branch `feat/agent-readiness-standardized`, bug-hunt baseline head `dab75522`, base `origin/feat/worldsim-v5@446ff6cc`)
Backup branches (do not delete): `backup/pr-10-before-446ff6cc-rebase-20260505-145710`, `backup/pr-10-pre-rerebase-20260505-132359`, `backup/pr-10-before-rebase-20260505-124459`, plus three earlier dated `backup/pr-10-before-*` branches.

## What this is

PR #10 (`https://github.com/jasmineee-li/browser-sim/pull/10`, "Agent readiness: modularize Phase 2 and Phase 4 domains") now sits on `origin/feat/worldsim-v5@446ff6cc`. The original modularization/rebase work was followed by the Gap 4, Gap 13, and Gap 15 hardening commits, plus later bug-hunt fixes after `398bc38d`, `4baa1a82`, and `60933632` (`872ef27a`, `42637149`, `dab75522`). At the bug-hunt baseline head `dab75522`, `git log --oneline origin/feat/worldsim-v5..HEAD | wc -l` reported 50 commits on top of `446ff6cc`; subsequent fix or docs commits should be counted as `50 + N`.

Local validation is green: `tests/seed_contracts tests/phase_2 tests/phase_4 tests/test_readiness_audit.py tests/test_rewards.py` (`493 passed`), Phase 4 host-API + adjacent surfaces (`412 passed`), `worldsim.main --help`, `scripts/readiness_audit.py --json`, `ruff check`, `bash scripts/verify_fast.sh`. Git considers the PR `MERGEABLE` with `mergeStateStatus: CLEAN`.

The live r5 integration gate is OUT OF SCOPE for this TODO. It is the operator's call per CLAUDE.md / `agent_docs/remote-runs.md` and is already documented as Pending Live Gate in the PR body.

## Why this matters

The rebase had to reconcile ~51 upstream commits with the PR's deletion of three monolith files (`worldsim/phases/phase_2_injections.py`, `worldsim/phases/phase_2_target_resolver.py`, `worldsim/phases/phase_4_adversarial.py`). Conflict resolution chose `--theirs` (PR side) for those monoliths and forward-ported the upstream symbols I could identify into the modular layout. The local pytest suite passes, but pytest does not exercise every code path; some upstream behavior may have landed cleanly via three-way merge on unmoved files, some may be missing in the modular layout, and a few resolutions are documented adaptations rather than 1-1 ports.

Until the gaps below are closed (or explicitly judged as accepted risk), "safe to merge" is conditional on live r5 alone. After this TODO is resolved, "safe to merge" is conditional only on the live gate.

## Constraints (must hold throughout)

- WASP scope is GitLab issues/comments and Reddit/Postmill posts/comments only. Do not introduce or restore Magento / Wikipedia / OSM / shopping / classifieds.
- Phase 2c is strict admission for Phase 4. Do not loosen feasibility, do not pass `--skip-feasibility`.
- Phase 4 has exactly two gates: PVPO encounter (`max_coverage` in `[0,1]`, `0` routes to placement-fix) and refusal-judge attack effectiveness. Transcript Purpose and Verbalized Eval Awareness are observational only. Do not branch or gate on them.
- Phase 4 only varies adversarial strategy. Benign task contracts and reward functions emitted by Phase 3 remain invariant across variants.
- Refusal judge, variant generator, Transcript Purpose, VEA, and placement-fix use direct host Anthropic Messages API calls, not `run_claude_in_sandbox`. Do not reroute them through Modal.
- Compatibility wrappers `worldsim.phases.phase_2_*` and `worldsim.phases.phase_4_adversarial` stay in place for one migration cycle. Do not delete them in this TODO.
- Auth precedence lives in `worldsim/modal_sandbox.py::_build_claude_secrets` and `worldsim/phase_4/anthropic_client.py`. Do not hard-code one auth mode.
- `AgentLab/src/agentlab/benchmarks/redteam/{execution.py,claude_code.py}` are read-only references. Never import from `AgentLab/`.
- Conventional Commits required. Use `chore(rebase): <subject>`, `fix(rebase): <subject>`, `test(rebase): <subject>` for follow-up work; body cites the upstream SHA-7 and headline being absorbed plus the invariant preserved. Co-authored-by trailers preserved where appropriate.
- Push uses `--force-with-lease`, never `--force`.

## Workflow

Research, then plan, then implement, then validate. Keep each step small. Update this TODO file in place as gaps close (mark each section "resolved" with the SHA(s) of the commits that closed it). Do not delete sections.

```
cd /Users/ashtonchew/projects/browser-sim/.codex-worktrees/agent-readiness-standardized
git fetch origin
git status                                    # confirm clean tree
git log --oneline origin/feat/worldsim-v5..HEAD | wc -l   # bug-hunt baseline dab75522 was 50; expect 50 + N follow-up commits
```

For any new forward-port commit, run the local validation block at the end of this file before pushing. After every push, refresh `gh pr checks 10`.

## Gap 1: GitHub CI green on `d6d5c0f3`

**Status: resolved (no CI configured) 2026-05-05.** `gh pr checks 10` returns `no checks reported on the 'feat/agent-readiness-standardized' branch`. There is no `.github/workflows/` directory in the fork, so no GitHub Actions workflow is wired. `gh run list --repo jasmineee-li/browser-sim --branch feat/agent-readiness-standardized` returns empty. The local validation block at the end of this TODO is the authoritative gate for PR #10 until CI is set up. This is documented in the PR body.

**What.** When the rebase pushed, no CI checks were reporting on `feat/agent-readiness-standardized`. CI may not be wired to this fork, may be running, or may have failed silently.

**Why.** Even if CI is informational rather than blocking, an unsignaled or red CI is a merge-blocker for review. We need a definitive answer.

**Discovery.**
1. `gh pr checks 10` and `gh pr view 10 --json statusCheckRollup` to see the canonical state.
2. `gh run list --repo jasmineee-li/browser-sim --branch feat/agent-readiness-standardized --limit 20` to see workflow runs.
3. If "no checks" persists, look at `.github/workflows/` to confirm trigger configuration; check whether the fork has `pull_request` workflows enabled in its settings (PR is from `jasmineee-li/browser-sim` to itself, so `pull_request` workflows should fire if defined).
4. If CI is not wired in this fork, replicate the local validation block (see end of file) and document that as a substitute. Do not invent CI.

**Fix.**
- If a real check is failing, root-cause and fix in a `fix(<scope>): …` follow-up commit; never bypass with `--no-verify`.
- If checks are pending, wait, do not poll.
- If checks are not configured, document that explicitly in the PR body and treat the local validation block as the authoritative gate.

**Verification.** `gh pr checks 10` returns either all-green or "no checks" with documented justification.

## Gap 2: Verify the three behavioral forward-port commits are correct

**Status: resolved (no fix needed) 2026-05-05.** Direct audit of each behavioral forward-port confirmed correctness:

- `05a17161` (validate-then-canonicalize): `worldsim/phases/phase_1_generate_new_tasks_validation.py:1085, 1092` shows the correct order; `tests/test_phase_1_tasks.py::test_validate_generated_novel_tasks_rejects_task_card_capability_mismatch` passes.
- `8b9850d5` (action_policy=action_policy or "default"): `worldsim/phase_2/generation.py:20, 90` matches; `tests/phase_2/test_output.py::test_generate_injections_for_site_api_path_sanitizes_prompt_inputs` passes.
- `d6d5c0f3` (adopt upstream test files verbatim): source-side preservation confirmed. GraphQL URL pattern `r"/api/graphql(?:[?#].*)?$"` exists at `worldsim/adversarial_actions/compiler.py:721, 907, 931`; post-data field expectation `r"^variables(?:[\[.].*)?body.*$"` exists at `:943`. Helpers `_canonicalize_task_card_action_provenance` and `_card_benign_task_family_id` exist at `phase_1_generate_new_tasks_validation.py:1209, 1470`. `tests/test_adversarial_actions.py` runs clean. The 7 failures in `tests/test_phase_1_tasks.py` are exactly the pre-existing Gap 4 set and are out-of-scope.

The five trailing forward-port commits are the only commits in this PR that are not 1-1 with the original PR's intent. Three are behavioral. They need spot-review.

**Commit `cb9c2e9a chore(rebase): validate phase_1 capability alignment before canonicalizing card provenance`.**
- File: `worldsim/phases/phase_1_generate_new_tasks_validation.py` (call order at the end of `_validate_task_card_alignment`).
- Claim: validate-then-canonicalize preserves both upstream's mismatch detection and the user's `46552b86` intent of card-authoritative downstream metadata.
- Discovery: read `_validate_task_card_capability_alignment` and `_canonicalize_task_card_action_provenance` end-to-end. Check that any field the canonicalize function writes (`capability_family`, `benign_task_family_id`, `compatible_action_kinds`, `task_provenance.*`) is not also depended on by the validation that now runs before it. Specifically, `_validate_task_card_capability_alignment` reads `task["capability_family"]` and `card_capability_family(card)`; if both come from the model + card directly, the validation is correct. If anywhere downstream reads the canonicalized fields with the assumption that they are pre-validated, that downstream code might now see a stale value if validation rejected and the canonicalize step never ran.
- Discovery query: `rg -n "task\[.capability_family.\]|task\[.benign_task_family_id.\]|task\[.compatible_action_kinds.\]" worldsim/phases/phase_1_generate_new_tasks_validation.py worldsim/phases/phase_1_route_contracts.py worldsim/adversarial_actions/`.
- Test that proves correctness: `tests/test_phase_1_tasks.py::test_validate_generated_novel_tasks_rejects_task_card_capability_mismatch` (already passing) plus any test that runs canonicalize-then-uses-it (search for tests asserting `task_provenance.compatible_action_kinds`).
- Fix if wrong: revert ordering and instead make the canonicalize function refuse to overwrite when the values mismatch (raise `INVALID_TASK_CARD` rather than silently aligning). Keep `--theirs`-style behavior only as a last resort.

**Commit `05c7a965 chore(rebase): default phase_2 action policy when caller passes None`.**
- File: `worldsim/phase_2/generation.py` line ~88 (`policy=action_policy or "default"`).
- Claim: matches upstream signature behavior while keeping the user's `action_policy: str | None = None` parameter shape.
- Discovery: confirm that no caller in `worldsim/phase_2/runner.py` or `worldsim/main.py` distinguishes `None` from `"default"` upstream of this point. `rg -n "phase_2a_action_policy|action_policy=" worldsim/`.
- Discovery: confirm `canonical_action_policy(None)` returns `"default"` (it does at `worldsim/adversarial_actions/compiler.py` line 145), so `or "default"` and passing `None` through canonicalize would be equivalent if the validator did not run before canonicalize. The fix taken is the cheaper one (one call site).
- Test that proves correctness: `tests/phase_2/test_output.py::test_generate_injections_for_site_api_path_sanitizes_prompt_inputs` (already passing).
- Fix if wrong: instead of the `or "default"` coercion, change the upstream validator to call `canonical_action_policy` before the `if policy not in ACTION_POLICIES` guard (cleaner, but a behavioral change to a heavily-tested helper, so prefer the call-site fix unless the validator change is independently motivated).

**Commit `2b566e67 test(rebase): adopt upstream phase_1 / adversarial action test files verbatim`.**
- Files: `tests/test_phase_1_tasks.py`, `tests/test_adversarial_actions.py`. Both reset to `origin/feat/worldsim-v5` content.
- Claim: the user's `46552b86` and `d788d3a5` test edits were formatting-only or now-obsolete; the behavioral wiring those commits introduce is preserved on the source side.
- Discovery: for each of the two source files those commits modified (`worldsim/phases/phase_1_generate_new_tasks_validation.py` and `worldsim/adversarial_actions/compiler.py`), read every public function and the relevant private helpers. Confirm that:
  - GraphQL note URL pattern `r"/api/graphql(?:[?#].*)?$"` and the `r"^variables(?:[\[.].*)?body.*$"` post-data field expectation are present in `compiler.py` (search by string).
  - `_canonicalize_task_card_action_provenance` and `_card_benign_task_family_id` exist and are reachable.
- Discovery: search the upstream test files for any assertion that exercises GraphQL note recognition specifically, e.g. `rg "graphql" tests/test_phase_1_tasks.py tests/test_adversarial_actions.py`. If upstream's tests cover the behavior, no test debt. If they do not, write a focused test in `tests/phase_2/` or a new `tests/test_adversarial_actions_graphql.py` that asserts both URL-pattern and post-data expectations. New tests must be in `tests/`, not in the dropped monolith files.
- Fix if a behavior is uncovered: add `test(rebase): cover gitlab graphql note recognition end-to-end` with concrete assertions.

## Gap 3: Audit monolith conflict resolutions for missed forward-ports

**What.** During the rebase, three monoliths had `git checkout --theirs` applied so they would still be deleted later in the PR. Any upstream commit between merge-base `f2e1d2e1` and current tip `38aa2f30` that touched only the monolith might have been silently dropped. I forward-ported the ones I identified (reddit visual regex, smoke variant budget). The complete list of upstream commits that touched any of the three monoliths needs an explicit audit.

**Why.** Phase 2c admission, Phase 4 execution, target resolution, ink-occupancy, encounter detection, postprocess accounting, scenario summary, and reward provenance all live in those monoliths. A missed port could weaken admission, disable a metric, or drift a gate. Phase 4's gates must remain PVPO encounter and refusal judge with no observational metric leaking into branching.

**Discovery (deterministic).**
```
cd /Users/ashtonchew/projects/browser-sim/.codex-worktrees/agent-readiness-standardized

# 1. List every upstream commit that touched any of the three monoliths.
git log --oneline f2e1d2e1..origin/feat/worldsim-v5 -- \
  worldsim/phases/phase_2_injections.py \
  worldsim/phases/phase_2_target_resolver.py \
  worldsim/phases/phase_4_adversarial.py > /tmp/pr10-monolith-upstream.txt
wc -l /tmp/pr10-monolith-upstream.txt

# 2. For each commit, capture the public symbols it added or modified.
while read sha rest; do
  git show "$sha" --stat -- \
    worldsim/phases/phase_2_injections.py \
    worldsim/phases/phase_2_target_resolver.py \
    worldsim/phases/phase_4_adversarial.py
  git show "$sha" -- \
    worldsim/phases/phase_2_injections.py \
    worldsim/phases/phase_2_target_resolver.py \
    worldsim/phases/phase_4_adversarial.py \
    | grep -E "^\+(def |class |[A-Z_][A-Z0-9_]+\s*[:=])" | head -20
done < /tmp/pr10-monolith-upstream.txt

# 3. For each public symbol, confirm it exists in the modular layout.
#    Top-level symbols flow through worldsim/phase_2/target_resolution/_context.py,
#    worldsim/phase_2/_context.py, and worldsim/phase_4/_context.py via install_context + link_modules.
rg -n "<symbol>" worldsim/phase_2/ worldsim/phase_4/ worldsim/seed_contracts/

# 4. For private helpers (leading underscore) that are called from the same monolith
#    and that monolith is now a compat shim, the helper must exist in the modular destination
#    so the shim's `sys.modules[__name__] = _runner` redirect resolves it.
```

**Discovery (semantic).** For each upstream commit, read its commit message and decide which modular module it belongs to (cross-reference against the table in `/Users/ashtonchew/.claude/plans/i-want-you-to-velvet-orbit.md` "Conflict Surface" section). If the message says it tightens admission, the destination is `worldsim/phase_2/eligibility.py` or `worldsim/phase_2/plan_validation.py`. If it changes Phase 4 execution, the destination is `worldsim/phase_4/execution.py`, `postprocess.py`, `result_summary.py`, `runner.py`, `strategy_variation.py`, `preflight.py`, or `encounter_detection.py`.

**Specific commits known to need verification (all touch one or more monoliths since merge-base):**

- `4d450fc9 fix(phase4): distinguish offscreen pvpo witnesses` (PVPO offscreen handling)
- `b1b549dd fix(audit): accept final-state mutation evidence` (audit + maybe phase_2)
- `841fb550 feat(phase4): extract observational action attempts` (telemetry only, must not gate)
- `5b1ce58f fix(phase4): share scenario summary reporting`
- `9e7fdd38 feat(phase4): report scenario strength metrics`
- `8fcf0602 feat(phase4): report variant progress heartbeats`
- `9873fa76 feat(phase4): add smoke variant budget` (already forward-ported in `232a7a67`; verify)
- `4e2d6cd8 fix(actions): enforce host-owned action contracts`
- `ffafce36 fix(actions): harden action prompt contracts`
- `9b0c719c fix(actions): harden prompt boundaries and action drift`
- `f0d3dd2e fix(phase4): count state-confirmed action readback`
- `1ee8d3e9 fix(phase2): harden action evidence checks`
- `a7ec52fd fix(phase4): update postprocess progress incrementally`
- `950774be fix(phase4): resync tier3 action witnesses`
- `90284d59 feat(actions): enforce action-only benign rewards`
- `e704dbeb fix(actions): fail closed on capability pilot contracts`
- `80f9534c fix(phase4): harden action reward provenance`
- `68bab8b9 fix(phase4): cap browser worker reruns`
- `cacb0f32 fix(phase4): preserve host-compiled benign action rewards`
- `7830fd10 fix(phase4): persist reward evaluation diagnostics`
- `c85849f4 fix(phase2): recompile benign action evidence`
- `f618a53c fix(phase4): preserve auth across same-site origin aliases`
- `4a3186c6 fix(phase2): require seeded reddit comment visibility` (already forward-ported in `232a7a67`; verify the modular runner exposes the symbol via the shim)
- `8a534c22 fix(phase2): require visual forcing for reddit comments` (also forward-ported; verify)
- `7ca19e45 fix(phase2): block private payload anchor leaks`
- `c8ede0ba fix(phase2): resolve generated child carrier surfaces`
- `33f6e01c fix(phase2): validate generated child carrier surfaces`
- `f8ce8213 fix(gitlab): bound duplicate issue seed retries`
- `f1e1edcf fix(phase2): foreground public mutation payload actions`
- `45df2440 fix(phase0): guard terminal source path repair` (file outside the monolith trio; should have come through cleanly via three-way merge, but include in audit because a phase boundary moved)

**Fix.** For each commit confirmed to have a missing semantic, add a `chore(rebase): port <commit-headline> into <modular-destination>` follow-up commit. The body cites the upstream SHA-7, the destination module, and the invariant preserved (Phase 2c admission, Phase 4 gate, observational-only telemetry, WASP scope). One commit per logical concern; do not batch.

**Verification.**
- All upstream symbols added by the listed commits resolve through both the modular path and the compat shim. Smoke:
  ```
  uv run python -c "import worldsim.phases.phase_2_injections; import worldsim.phases.phase_2_target_resolver; import worldsim.phases.phase_4_adversarial; print('shims OK')"
  uv run python -c "from worldsim.phase_2 import eligibility, generation, plan_validation, target_stage, runner; from worldsim.phase_2.target_resolution import runner as tr_runner, l3, l4, http_probes, listing_probes, reconstruction, resolver, types, url_matching; from worldsim.phase_2.phase_2c import artifacts, config, stage; from worldsim.phase_4 import admission, execution, postprocess, preflight, runner as p4_runner, results, resume, strategy_variation, variant_eval, scenario_funnel_export, scenario_funnel_outputs; print('modular OK')"
  ```
- For every symbol the listed commits added, both `import_via_modular()` and `import_via_shim()` return the same object (`is`-identical, since `link_modules` and `sys.modules` redirect re-export the same module).
- No new pytest failures vs `feat/worldsim-v5` baseline.

## Gap 4: Triage 7 pre-existing pytest failures in `tests/test_phase_1_tasks.py`

**Status: resolved 2026-05-05 by `b39079a8`.** All 7 named tests pass on PR HEAD. Root cause: `phase_2_exposure_contract::_appended_comment_exposure_capability` hard-requires `visual_comment_region_forced_by_task` and `seeded_comment_visible_forced_by_task` for any reddit comment-body capability check, and three call sites needed those flags but could not produce them: the Phase 1 route-capability probe (no task content), the Phase 1 task validator (`_validate_phase2_placement_target` did not annotate the resource), and `_route_evidence_flags` (only inferred `exact_comment_region_forced_by_task` from `_LATEST_DISCUSSION_REGION_RE`, never `visual_comment_region_forced_by_task`). Fix sets the three forcing flags maximally on the route-capability probe in `worldsim/phases/phase_1_route_contracts.py::_pattern_has_admissible_exposure` (probe asks about reachability, not task content), sets the same flags on the task-validator resource when the task selected the reddit comment-body route in `worldsim/phases/phase_1_generate_new_tasks_validation.py::_validate_phase2_placement_target` (Phase 1 `instruction_requirements` continues to enforce instruction quality via `ROUTE_INSTRUCTION_TOO_WEAK`), and extends `worldsim/phase_2/target_resolution/l3.py::_route_evidence_flags` to set transition / exact / visual / seeded flags when the instruction matches `_REDDIT_COMMENT_VISUAL_REGION_RE` for `reddit_submission` / `reddit_dashboard_list` kinds. Phase 2c admission gates remain unchanged. Phase 4 enforcement at the seeding mechanism + PVPO checks continues to verify real comment-region forcing.

**What.** These tests fail on unmodified `origin/feat/worldsim-v5` and on the rebased PR head:
- `test_validate_generated_novel_task_accepts_phase2_eligible_reddit_submission_target`
- `test_build_task_route_contracts_includes_covered_core_carrier_surfaces`
- `test_build_task_route_contracts_handles_phase0_reddit_feed_ids_and_capitalized_entities`
- `test_build_task_route_contracts_includes_inventory_backed_reddit_comment_carriers`
- `test_validate_generated_novel_tasks_accepts_visually_forced_reddit_comment_route`
- `test_validate_generated_novel_tasks_rejects_reddit_comment_route_without_visual_region`
- `test_validate_generated_novel_tasks_rejects_generic_reddit_comment_route`

The failure mode in at least one is `UNKNOWN_ROUTE_ID` because `reddit.comment_body.reddit_submission.create_comment` is missing from `build_task_route_contracts(...)["route_families"]` even though the editor spec at `worldsim/editors/reddit.py:316` declares `surface_id_per_kind: {"reddit_submission": "comment_body_thread"}`.

**Why.** These are research correctness tests for Phase 1 route generation and Phase 1 -> Phase 2 admission. If the route is missing, Phase 1 generators cannot author reddit comment-body tasks, which Phase 2c then cannot admit, which Phase 4 then cannot vary. WASP scope already restricts us to Reddit/Postmill UGC and GitLab issues/comments; losing Reddit comment-body tasks is a real coverage gap.

These failures are NOT regressions introduced by PR #10 (confirmed: they reproduce on the main checkout). They predate the PR.

**Discovery.**
1. Reproduce on the main checkout: `cd /Users/ashtonchew/projects/browser-sim && uv run pytest tests/test_phase_1_tasks.py::test_validate_generated_novel_tasks_accepts_visually_forced_reddit_comment_route -q`. Confirm same failure mode.
2. Find the last upstream commit where the test passed: `git log --oneline -p tests/test_phase_1_tasks.py | grep -B1 "test_validate_generated_novel_tasks_accepts_visually_forced_reddit_comment_route"` and walk commits forward, running the test at each commit until it starts failing.
3. Read `worldsim/phases/phase_1_route_contracts.py::build_task_route_contracts` and `_uncovered_surface_ids` / `_covered_surface_ids` / `_route_family_for_spec` to find the path that filters out the comment-body route. The test uses `_add_reddit_submission_sample(profile)` which adds the `comment_body_thread` injection surface; check why the route generator drops it.
4. Look at `is_active_carrier_surface(site, canonical, kind=kind, method=spec.method)` and `is_core_surface(site, canonical)` to see whether `comment_body_thread`'s canonical maps to a core+active surface for `reddit_submission`.

**Fix.**
- This is upstream-side work. Resolution is NOT in PR #10's scope, but the bug should be filed and possibly fixed on `feat/worldsim-v5` directly.
- Open a separate handoff or upstream commit titled `fix(phase1): include reddit comment-body inventory-backed routes` and address there.
- Until fixed, document the failure list in PR #10's body (already done) so reviewers do not block on it.

**Verification.**
- The 7 tests pass on `feat/worldsim-v5` after the upstream fix lands.
- After upstream fix is merged, re-rebase PR #10 (clean fast-forward expected).

## Gap 5: External callsites of the compatibility wrappers

**Status: resolved (no fix needed) 2026-05-05.** Direct audit smoked every unique import that crosses a compat shim boundary (eight private helpers from `phase_2_target_resolver`, four from `phase_2_injections`, two from `phase_4_adversarial`, eight from `phase_2_injections_api`, plus the integration-test imports under `tests/integration/`). Every symbol resolves through the shim. The compat layer is `sys.modules[__name__] = <runner>` plus `install_context` + `link_modules` propagation, so any symbol on a sibling module reachable from the runner is reachable from the shim. No `chore(scripts):` migration is needed in this PR; PR #11 adds `tests/test_phase_compat_wrappers.py` for ongoing CI coverage of the same surface.

**What.** The PR retains `worldsim/phases/phase_2_injections.py`, `worldsim/phases/phase_2_target_resolver.py`, `worldsim/phases/phase_4_adversarial.py` as compat shims. Any code in `scripts/`, in vendored references, in agent harnesses, or in notebooks that imports those paths must still work.

**Why.** Silent import-time failures or partial re-exports would manifest only at runtime in live r5 runs, exactly the path we did not exercise. The "Preserve Auth Boundary" memory and the "v5 Pivot Absences" memory both warn against silent failures at boundaries.

**Discovery.**
```
cd /Users/ashtonchew/projects/browser-sim/.codex-worktrees/agent-readiness-standardized

# 1. Find every external callsite of the old paths.
rg -n "from worldsim\.phases\.phase_2_injections|from worldsim\.phases\.phase_2_target_resolver|from worldsim\.phases\.phase_4_adversarial|import worldsim\.phases\.phase_2_injections|import worldsim\.phases\.phase_2_target_resolver|import worldsim\.phases\.phase_4_adversarial" worldsim tests scripts packages

# 2. For each callsite, confirm the imported symbol exists in the modular destination.
#    The shim does sys.modules[__name__] = <modular_runner> so any symbol on the runner is reachable.
#    Symbols defined on the package's _context.py propagate via install_context.
#    Symbols defined on sibling modules propagate via link_modules.
#    Anything else has to be re-exported explicitly.

# 3. Smoke each unique import statement.
uv run python -c "from worldsim.phases.phase_2_injections import <symbol>; print(repr(<symbol>))"
```

**Fix.** For any imported symbol that does not resolve through the shim:
- Preferred: add the symbol to the appropriate modular `_context.py` so `install_context` + `link_modules` propagates it, or to the appropriate sibling module that the runner already aggregates.
- Fallback (if the symbol is genuinely deprecated and the callsite should migrate): add a `from worldsim.<modular_path> import <symbol>` re-export at the top of the compat shim. Do not add new symbols to the shim itself; the shim is `sys.modules[__name__] = <runner>` and not a place for live code.
- If the callsite is in `scripts/` and is the only consumer, migrate the script to the new modular import path in a `chore(scripts): migrate <script> to modular phase_<N> imports` commit and keep the shim untouched.

**Verification.**
- `rg ...` returns the same set of imports.
- Each unique symbol resolves successfully under both the shim path and the modular path.
- No `ImportError` from `uv run python -m worldsim.main --help` or `uv run python -m worldsim.main preflight --help`.

## Gap 6: Forward-port completeness across unmoved files

**Status: resolved (no fix needed) 2026-05-05.** Tree-diff between HEAD and the rebase target `38aa2f30` shows the unmoved files either match upstream verbatim (`worldsim/rewards.py`, `worldsim/editors/reddit.py`, `worldsim/phases/phase_2_feasibility.py` all 0+ 0-) or differ only by intentional PR changes:

- `worldsim/main.py` (37+ 13-): verification proxy token resolver migration (`e5fea6f0 security: externalize verification proxy token`).
- `worldsim/config.py` (61+ 0-): `load_benchmark_config` and `dump_verification_proxy_config` helpers for `token_env`/`token_file` indirection.
- `worldsim/adversarial_actions/compiler.py` (7+ 5-): GraphQL note URL pattern `r"/api/graphql(?:[?#].*)?$"` and the `r"^variables(?:[\[.].*)?body.*$"` post-data field expectation (`5e43e6cd fix(gitlab): recognize graphql note actions`).
- `worldsim/phases/phase_1_route_contracts.py` (6+ 6-): import path migration from `phase_2_target_resolver` to `phase_2.target_resolution.runner`.
- `worldsim/phases/phase_1_generate_new_tasks_validation.py` (46+ 3-): validate-then-canonicalize fix (`05a17161`) plus the canonicalize helper, both already verified under Gap 2.
- Other small files (phase_0_recon.py, phase_0d_auth_bootstrap.py, phase_2_exposure_contract.py, phase_2_text_fill.py, phase_2_output.py, phase_2c_artifacts.py, phase_2c_config.py, payload_guidance.py): each diff is single-digit lines and matches PR-intent module-path migrations or readiness-audit cleanup.

Upstream advanced from `38aa2f30` to `446ff6cc` while this work was in progress (three commits: `446ff6cc fix: harden reddit comment attribution`, `4479a962 docs: record reddit comment attribution rca`, `6f4f0019 docs: clarify phase4 remote run state dir`). Those changes are out of scope for this audit; PR #10 will need a fresh rebase onto the new tip after these gaps close.

**What.** The 51 upstream commits since `df71d96a` touched many files outside the monolith trio: `worldsim/rewards.py`, `worldsim/main.py`, `worldsim/config.py`, `worldsim/phases/phase_0_recon.py`, `worldsim/phases/phase_1_generate_new_tasks_validation.py`, `worldsim/adversarial_actions/*`, `scripts/*`, `docs/*`. Three-way merge most likely brought their changes through cleanly, but "most likely" is not "verified".

**Why.** Reward readback hardening (`fix(rewards): wait for gitlab note readback`, `fix(rewards): match network events across origin aliases`, `fix(rewards): prove redacted issue comments by readback`, `fix(rewards): read back reddit public mutation state`) is research-load-bearing: rewards must read back from the live UGC surface, not from internal state, per the "Preserve Auth Boundary" memory.

**Discovery.**
```
# For each upstream commit since merge-base, run a tree-equivalence check on the files it touched.
# A commit is "absorbed" iff: for every file it changed, the post-rebase tree at HEAD matches
# the post-commit tree on origin (aside from the modular-path moves and any explicit forward-ports).

cd /Users/ashtonchew/projects/browser-sim/.codex-worktrees/agent-readiness-standardized
git log --oneline f2e1d2e1..origin/feat/worldsim-v5 > /tmp/pr10-upstream-since-merge-base.txt
wc -l /tmp/pr10-upstream-since-merge-base.txt   # expect ~51

# For each commit, list the files it changed and diff each file at HEAD vs the upstream tree.
while read sha rest; do
  files=$(git show --name-only --pretty='' "$sha")
  for f in $files; do
    # Skip files PR #10 deletes; their behavior is forward-ported separately.
    case "$f" in
      worldsim/phases/phase_2_injections.py|worldsim/phases/phase_2_target_resolver.py|worldsim/phases/phase_4_adversarial.py|tests/test_phase_2_injections.py|tests/test_phase_2_target_resolver.py|tests/test_phase_4_adversarial.py|logs/phase_*) continue ;;
    esac
    if ! git show "origin/feat/worldsim-v5:$f" 2>/dev/null | diff -q - "$f" >/dev/null 2>&1; then
      echo "DIFFERS: $sha $f"
    fi
  done
done < /tmp/pr10-upstream-since-merge-base.txt > /tmp/pr10-difference-report.txt
```

**Fix.** For each `DIFFERS` line, manually compare the two file versions and decide whether the divergence is intended (PR #10 modified that file deliberately) or accidental (a hunk got dropped during conflict resolution). Accidental drops get a `chore(rebase): restore <upstream-commit-headline> in <file>` commit that brings the missing hunk forward. Intentional divergences get a one-line note in this TODO.

**Verification.** `/tmp/pr10-difference-report.txt` is either empty or every line is a documented intentional divergence.

## Gap 7: Documentation drift

**Status: resolved 2026-05-05.** Active doc references to the deleted monoliths updated:
- `README.md:308` directory listing now shows `worldsim/phase_2/`, `worldsim/phase_4/`, and `worldsim/phases/` (legacy + compat shims) explicitly.
- `docs/worldsim-v5-technical-specifcation.md:1755` updated to point at `worldsim.phase_4.metrics._ecologically_valid` with a note that the `phase_4_adversarial` compat shim re-exports it for one migration cycle.

Historical TODO/handoff docs (`docs/TODO-adversarial-rigor-mvp.md`, `docs/todo-pvpo-post-ship-review.md`, `docs/TODO-2-paper-experiments.md`, `docs/handoffs/codex-handoff-*.md`, `docs/handoffs/wasp-aligned-scoping-decision.md`, `docs/handoffs/researcher-handoff-project-status.md`, `docs/handoffs/pvpo-placement-fix-r5-integration-2026-04-19.md`, `docs/handoffs/phase-2-placement-systemic-gap.md`, `docs/handoffs/archive/*.md`) keep their original phrasing because they are point-in-time records, not active executable docs. Their line-number references were already stale before the rebase.

**What.** The PR updates `agent_docs/code-organization.md`, `agent_docs/artifacts.md`, `agent_docs/remote-runs.md`, `agent_docs/secrets.md`, `README.md`, `docs/handoffs/orchestrator-handoff-r5-migration.md`, `docs/worldsim-v5-technical-specifcation.md`. The base added new `docs/results/*`, `docs/actions/*`, and `docs/runbook/*` content. After the rebase, agent docs should describe the modular layout AND the upstream-only doc additions should still be present.

**Why.** The spec is authoritative. CLAUDE.md says: if code and spec diverge, update the spec first. After this rebase the modularized layout exists in code and the spec text should match.

**Discovery.**
- `rg -n "phase_2_injections|phase_2_target_resolver|phase_4_adversarial" agent_docs docs README.md` to find any doc that still names the deleted monoliths.
- Compare `agent_docs/code-organization.md` between PR head and `origin/feat/worldsim-v5` to confirm the PR's modular description is current.

**Fix.** `docs(rebase): align <doc> with modular phase_<N> layout` for any doc that still names the monoliths or describes their internal structure.

**Verification.** No agent doc references `phase_2_injections.py` or `phase_4_adversarial.py` as live code paths; all such references are either inside compatibility-wrapper notes or removed.

## Gap 8: Generated artifact policy preservation

**Status: resolved (no fix needed) 2026-05-05.** `git ls-files logs/` returns empty. `.gitignore` diff against `origin/feat/worldsim-v5` shows only the intended PR additions (`instances.smoke.json`, `instances.smoke.json.fragment`, `scripts/docker-compose.smoke.yml`). No regenerated logs were re-tracked by the rebase. Readiness audit `tracked_generated` field was already verified as `[]` after the rebase and remains so after the Gap 10/11/12/14 closing commits.

**What.** PR #10 deleted ~775,000 lines of tracked `logs/phase_1/*.json` and `logs/phase_2/*.json`. The PR adds artifact policy text in `agent_docs/artifacts.md`. Verify that nothing in the rebase reintroduced any tracked `logs/` artifact, and that `.gitignore` still ignores them.

**Why.** Re-tracking those files would undo the core artifact policy of PR #10 and bloat the repo by hundreds of megabytes.

**Discovery.**
- `git ls-files logs/` should be empty.
- `cat .gitignore | grep logs/` should show the ignore rule survived the rebase.
- `git diff origin/feat/worldsim-v5 -- .gitignore` should match the PR's intended `.gitignore` change.

**Fix.** If any logs file is tracked, `git rm --cached logs/<file>` and commit `chore(artifacts): re-untrack <file> per artifact policy`.

**Verification.** `git ls-files logs/` returns nothing; readiness audit `tracked_generated` field is `[]` (already verified after the rebase, but recheck after any forward-port commit lands).

## Gap 9: Final import-graph end-to-end smoke

**Status: resolved (no fix needed) 2026-05-05.** The full heredoc below printed `import graph OK` and exited 0 on HEAD `3f7097e7`. No `AttributeError` from `install_context` / `link_modules` propagation. Note that this gate cannot detect the metric-time gaps in Gap 10/11/12/13 because those failures only manifest when the corresponding metric is computed, not at import time.

**What.** Beyond unit tests, the modular layout has subtle inter-module dependencies (`install_context` + `link_modules` mutates module dicts). One missing helper anywhere can manifest as `AttributeError` at first call rather than at import time.

**Discovery / Fix.** Run a single Python import that walks the entire surface:
```
cd /Users/ashtonchew/projects/browser-sim/.codex-worktrees/agent-readiness-standardized
uv run python - <<'EOF'
import worldsim.main
import worldsim.config
import worldsim.rewards
import worldsim.phases.phase_0_recon
import worldsim.phases.phase_0d_auth_bootstrap
import worldsim.phases.phase_1_generate_new_tasks_validation
import worldsim.phases.phase_1_route_contracts
import worldsim.phases.phase_2_core_surfaces
import worldsim.phases.phase_2_exposure_contract
import worldsim.phases.phase_2_injections          # compat shim
import worldsim.phases.phase_2_target_resolver     # compat shim
import worldsim.phases.phase_2_text_fill
import worldsim.phases.phase_2c_artifacts
import worldsim.phases.phase_2c_config
import worldsim.phases.phase_4_adversarial         # compat shim
import worldsim.phase_2.runner
import worldsim.phase_2.eligibility
import worldsim.phase_2.generation
import worldsim.phase_2.plan_validation
import worldsim.phase_2.output
import worldsim.phase_2.target_stage
import worldsim.phase_2.target_resolution.runner
import worldsim.phase_2.target_resolution.l3
import worldsim.phase_2.target_resolution.l4
import worldsim.phase_2.target_resolution.url_matching
import worldsim.phase_2.target_resolution.reconstruction
import worldsim.phase_2.target_resolution.http_probes
import worldsim.phase_2.target_resolution.listing_probes
import worldsim.phase_2.target_resolution.types
import worldsim.phase_2.phase_2c.artifacts
import worldsim.phase_2.phase_2c.config
import worldsim.phase_2.phase_2c.stage
import worldsim.phase_4.runner
import worldsim.phase_4.execution
import worldsim.phase_4.postprocess
import worldsim.phase_4.preflight
import worldsim.phase_4.results
import worldsim.phase_4.resume
import worldsim.phase_4.strategy_variation
import worldsim.phase_4.variant_eval
import worldsim.phase_4.placement_loop
import worldsim.phase_4.admission
import worldsim.phase_4.metrics
import worldsim.phase_4.scenario_funnel_export
import worldsim.phase_4.scenario_funnel_outputs
import worldsim.seed_contracts.delivery
import worldsim.seed_contracts.surface
import worldsim.seed_contracts.validation
import worldsim.adversarial_actions.compiler
import worldsim.adversarial_actions.payload_guidance
print("import graph OK")
EOF
```

**Verification.** The single Python invocation prints `import graph OK` and exits 0. If anything raises, file a `fix(rebase): expose <symbol> via <modular module>` commit.

## Gap 10: `_ACTION_REWARD_SIGNALS` constant missing from modular Phase 4

**Status: resolved 2026-05-05 by `5ac93e2e`.** Constant added to `worldsim/phase_4/_context.py`; install_context propagates it to runner.py and the compat shim resolves it transparently.

**What.** Upstream commit `f0d3dd2e fix(phase4): count state-confirmed action readback` defined `_ACTION_REWARD_SIGNALS = frozenset({"network_action", "external_navigation"})` at `worldsim/phases/phase_4_adversarial.py:236`. The constant gates the binary `attack_action_success` computation and the `_final_state_action_success_fields` helper (Gap 11).

**Why.** Without the constant, any callsite that dereferences it raises NameError. Even if the modular layout never reaches the dereference because the helper is also missing (Gap 11), the constant is still needed once Gap 11 is closed.

**Discovery (verified 2026-05-05).**
```
rg -n "_ACTION_REWARD_SIGNALS|action_reward_signals|ACTION_REWARD_SIGNALS" worldsim/phase_4/
# returns empty; constant is missing.
git show 38aa2f3095:worldsim/phases/phase_4_adversarial.py | rg -n "_ACTION_REWARD_SIGNALS"
# 236:_ACTION_REWARD_SIGNALS = frozenset({"network_action", "external_navigation"})
# 2137:    attack_action_success = bool(adv_passed and action_enabled and signal in _ACTION_REWARD_SIGNALS)
# 2169:    if not action_enabled or reward_signal not in _ACTION_REWARD_SIGNALS:
# 2274:        final_state_passed is True and action_enabled and signal in _ACTION_REWARD_SIGNALS
```

**Fix.** Add to `worldsim/phase_4/_context.py` near the other top-level constants. `install_context` will propagate it to the runner namespace. Single commit `chore(rebase): port _ACTION_REWARD_SIGNALS into phase_4 _context (f0d3dd2e)`.

**Verification.** `rg -n "_ACTION_REWARD_SIGNALS" worldsim/phase_4/` returns the new declaration; downstream callsites that compute `attack_action_success` resolve cleanly under both modular and shim imports.

## Gap 11: Phase 4 per-task action-evaluation pipeline missing from modular layout

**Status: resolved 2026-05-05 by `270a3a93` (helpers) + `a2e0986b` (call-site wiring).** The six helper functions are now in `worldsim/phase_4/metrics.py` and propagate through `install_context` + the `phase_4_adversarial` compat shim. All five execution.py paths (two seed-preflight error sites, agent-error path, non-interpretable benign-eval error path, main scoreable path) now compute and persist the reward_signal_fields, final_state_fields (where applicable), and tier3_cleanup_fields per upstream behavior. Local pytest 905 green; ruff clean; verify_fast.sh green.

**Status: scope expanded by direct audit 2026-05-05.** The original gap framed the issue as a single missing helper, `_final_state_action_success_fields()`. Direct grep against `worldsim/` shows the modular layout is missing the entire upstream cluster of per-task action-evaluation helpers and their call sites in `execution.py`. Tests pass only because `tests/test_phase_4_result_summary.py` stubs the result fields directly; real Phase 4 production runs would silently report zero numerators for every action-related headline metric.

**Missing helpers (each present in upstream `worldsim/phases/phase_4_adversarial.py` but absent from modular `worldsim/phase_4/`).**

| Upstream symbol | Upstream line | Purpose |
|---|---|---|
| `_adversarial_reward_signal_fields(...)` | 2125 | Builds the `adversarial_reward_signal`, `attack_action_success`, `semantic_hijack_success`, `benign_success_under_attack`, `stealth_success`, `benign_reward_shape`, plus all `attack_action_attempted` / `_attempt_count` / `_attempt_status` / etc. fields for one task. |
| `_adversarial_action_attempt_fields(...)` | 2160 | Computes the network-action attempt evidence / status / reason / count from the network trace. |
| `_format_action_attempt_fields(...)` | 2215 | Formats the attempt fields into the canonical key set persisted on each result. |
| `_upgrade_action_attempt_from_state_confirmation(...)` | 2240 | Upgrades `attack_action_attempted` from False to True when a final-state evaluator confirms persisted state. |
| `_final_state_action_success_fields(...)` | 2259 | Returns `{state_confirmed_action_success, tier3_state_confirmed_action_success}` from the reward-signal fields and `final_state_passed`. |
| `_tier3_action_cleanup_fields(...)` (async) | 2285 | Returns reporting-only cleanup telemetry for Tier 3 disposable-fixture action targets. |

**Missing call sites in the modular `worldsim/phase_4/execution.py` (compare against upstream `worldsim/phases/phase_4_adversarial.py:4280-4710`).**
- The seed-preflight error path needs `**_adversarial_reward_signal_fields(task, benign_passed=False, adv_passed=False)` on the synthetic result payload (upstream :4294).
- The post-PVPO-gate stage-1 result construction at modular `execution.py:412-442` is missing the upstream :4630-4710 block:
  ```python
  reward_signal_fields = _adversarial_reward_signal_fields(
      task, benign_passed=benign_passed, adv_passed=adv_passed,
      network_trace=result.network_trace, instance=instance_dict,
  )
  final_state_checked, final_state_passed, final_state_message = ...
  final_state_fields = {
      "adversarial_final_state_checked": final_state_checked,
      "adversarial_final_state_passed": final_state_passed,
      "adversarial_final_state_message": final_state_message,
      **_final_state_action_success_fields(reward_signal_fields, final_state_passed=final_state_passed),
  }
  if final_state_fields["state_confirmed_action_success"]:
      reward_signal_fields = _upgrade_action_attempt_from_state_confirmation(reward_signal_fields)
  tier3_cleanup_fields = await _tier3_action_cleanup_fields(task, instance_dict)
  save_result(..., **reward_signal_fields, **final_state_fields, **tier3_cleanup_fields, ...)
  ```

**Why.** Every Phase 4 production run silently reports zero numerators for the headline action-success metrics:
- `attack_action_success_rate`
- `semantic_hijack_success_rate`
- `stealth_success_rate`
- `benign_success_under_attack_rate`
- `state_confirmed_action_success_rate`
- `tier3_state_confirmed_action_success_rate`
- All `_by_tier` breakdowns of the above
- All `_observational_action_attempt_*` reporting

The local pytest suite passes because `tests/test_phase_4_result_summary.py:24+` directly stubs `attack_action_success`, `semantic_hijack_success`, `stealth_success`, `benign_success_under_attack`, etc. on synthetic result dicts. Production runs construct results via `execution.py:save_result(...)`, which never sets these fields. **This is the kind of regression that "local pytest is green" cannot detect; only the live r5 wrapper or a real Phase 4 mock-pipeline integration test would catch it.**

This is research-load-bearing per CLAUDE.md ("Phase 4 has two gates: PVPO encounter and refusal judge ... Action attempts are observational only ... A Tier 3 action passing state readback is reported as state-confirmed action success, not folded silently into semantic ASR").

**Discovery (verified 2026-05-05).**
```
rg -n "_adversarial_reward_signal_fields|_adversarial_action_attempt_fields|_format_action_attempt_fields|_upgrade_action_attempt_from_state_confirmation|_final_state_action_success_fields|_tier3_action_cleanup_fields" worldsim/
# returns NOTHING in worldsim/. The helpers exist only in upstream's monolith.
rg -n "attack_action_success|semantic_hijack_success|stealth_success|benign_success_under_attack" worldsim/ tests/
# Only tests stub these. No production producer in worldsim/.
```

**Fix.** Port the six helpers verbatim into `worldsim/phase_4/metrics.py` (which already owns `_pvpo_metric_payload`, `_ecologically_valid`, `_classify_trajectory_outcome` from the same upstream neighborhood). Then port the call-site block into `worldsim/phase_4/execution.py:412-442` and the seed-preflight error paths at `execution.py:121, 213`. Imports for `extract_network_action_attempt`, `cleanup_tier3_delete_project_action_target`, `cleanup_tier3_repository_action_target`, `reward_signal_for_task`, `action_metadata_for_task`, `benign_reward_shape_from_task`, and `BENIGN_REWARD_HOST_ACTION_ONLY` may need to be added to `worldsim/phase_4/_context.py` if not already imported (verify first via `rg`).

This is a multi-commit forward-port:
1. `chore(rebase): port action-evaluation helper family into phase_4 metrics (f0d3dd2e, 9e7fdd38, 841fb550, 90284d59, 80f9534c, cacb0f32, 950774be)`.
2. `chore(rebase): wire reward_signal_fields and final_state_fields into phase_4 execution (f0d3dd2e and friends)`.
3. (optional) `test(rebase): cover phase_4 reward_signal_fields production from a synthetic task`.

Each commit body cites the upstream SHA-7s being absorbed and the invariant preserved (Phase 4 gates remain PVPO + refusal-judge; action attempts and state-confirmed success remain observational; benign rewards remain Phase 3 invariants).

**Verification.** A synthetic task constructed via host code paths (not stubbed) produces `attack_action_success`, `semantic_hijack_success`, `state_confirmed_action_success` keys on the result. `tests/test_phase_4_result_summary.py` stays green. New regression test asserts `_adversarial_reward_signal_fields(...)` returns the canonical key set when called directly. The existing test suite continues to pass.

**Coverage caveat.** Even after this fix, "local pytest green" is necessary but not sufficient. The live r5 gate is what proves end-to-end correctness; that is intentionally the operator's call.

## Gap 12: `_FINGERPRINT_RESULT_KEYS` missing action-attempt and state-confirmed keys

**Status: resolved 2026-05-05 by `8370f29a`.** The modular tuple now matches upstream's tuple with the addition of all action-evaluation, final-state, message, pvpo_observation, and infrastructure_retry keys. resume.py crash-resume preserves these fields across restarts. Local pytest 636 green; ruff clean.

**What.** Upstream `_FINGERPRINT_RESULT_KEYS` tuple at `worldsim/phases/phase_4_adversarial.py:2414+` includes:
- `attack_action_attempted` (added by `9e7fdd38 feat(phase4): report scenario strength metrics`)
- `attack_action_attempt_reason` (same commit)
- `state_confirmed_action_success` (added by `f0d3dd2e`)
- `tier3_state_confirmed_action_success` (added by `f0d3dd2e`)

The modular tuple at `worldsim/phase_4/_context.py:185-219` is missing all four keys.

**Why.** `_FINGERPRINT_RESULT_KEYS` is used by `worldsim/phase_4/resume.py:50, 384` to round-trip per-task result fields across crash-resume. Missing keys are silently dropped on resume, which means a crash-recovered Phase 4 run reports zero attack action attempts and zero state-confirmed successes regardless of what actually happened. This is research-load-bearing for resume correctness, not just telemetry.

**Discovery (verified 2026-05-05).**
```
# Read worldsim/phase_4/_context.py:185-219 directly. Tuple ends at "classifier_version"
# without any of the four keys above.
rg -n "attack_action_attempted|state_confirmed_action_success" worldsim/phase_4/
# Many CONSUMER citations; the keys are not in _FINGERPRINT_RESULT_KEYS.
```

**Fix.** Extend the tuple in `worldsim/phase_4/_context.py:185-219` with the four missing keys. Use the upstream order. Single commit `chore(rebase): extend _FINGERPRINT_RESULT_KEYS with action-attempt and state-confirmed keys (9e7fdd38, f0d3dd2e)`.

**Verification.** `rg -n "_FINGERPRINT_RESULT_KEYS" worldsim/phase_4/` shows all consumers; the tuple now includes the keys. `tests/test_phase_4_result_summary.py` and `tests/phase_4/` resume-related tests stay green.

## Gap 13: `variant_progress_by_task` heartbeat wiring missing from Phase 4 runner

**Status: resolved 2026-05-05 by `1e8f99ba` (helpers) + `144dc7e0` (signature plumbing) + `d24722dd` (runner wiring) + `f08f3aa5` (tests) + `35e63174` (observational hardening and adaptive parity).** New module `worldsim/phase_4/postprocess_progress.py` contains `Phase4ProgressState`, `Phase4ProgressCallback`, `_PHASE_4_PROGRESS_ACTIVE_TASK_LIMIT`, `_jsonable_payload`, `_phase_4_progress_path`, `compute_progress_extra`, `write_postprocess_progress`, `record_postprocess_start`, `record_variant_progress`, `record_postprocess_result`, and `completed_task_ids_from_task_dir_root`. `worldsim/phase_4/postprocess.py::_postprocess_one_task` and `_process_adversarial_result` plus `worldsim/phase_4/strategy_variation.py::run_strategy_variation` accept an optional `progress_callback` kwarg (default `None`); `run_strategy_variation` defines a local `_emit_variant_progress` closure that emits `judge_complete`, `variant_round_started`, `variant_generation_recorded`, `variant_evaluation_started`, and `variant_evaluation_complete` events. `worldsim/phase_4/runner.py::run` now writes best-effort initial-evaluation and postprocess heartbeats, constructs a `Phase4ProgressState` near the postprocess gather, and wraps each `_postprocess_one_task` call with a local `_postprocess_one_task_with_progress` wrapper that records start, supplies the variant callback, and records result on success or failure. Schema is preserved verbatim at `schema_version: 1` so `worldsim/cli_status.py` and `scripts/remote_job_status.sh` continue to read successfully; `tests/phase_4/test_postprocess_progress.py` plus strategy/runner tests cover schema shape, lifecycle counters, callback chain, concurrent-record correctness, ecological-validity progress counting, adaptive budget rounds, semaphore enforcement, legacy checkpoint migration, and `_jsonable_payload` coercion. Heartbeats remain observational only per CLAUDE.md; progress write failures are logged and cannot fail postprocess or strategy variation.

**What.** Upstream `8fcf0602 feat(phase4): report variant progress heartbeats` added a `variant_progress_by_task` dict, an async `_record_variant_progress()` helper, and `progress_callback=lambda event, data, task_id=task_id: _record_variant_progress(...)` wiring inside the main `run()` body at `phase_4_adversarial.py:3829, 3893, 3946`.

**Why.** Telemetry only per CLAUDE.md ("Phase 4 only varies adversarial strategy"; heartbeats do not gate). However, absent heartbeats mean no progress visibility on multi-hour variant runs. Operator UX, not research correctness.

**Discovery (verified 2026-05-05).**
```
rg -n "variant_progress|record_variant_progress|progress_callback" worldsim/phase_4/
# returns empty.
git show 38aa2f3095:worldsim/phases/phase_4_adversarial.py | rg -n "variant_progress_by_task|_record_variant_progress|progress_callback"
# 3829, 3835, 3893, 3896, 3946 (call sites in run()).
```

**Fix.** Port the heartbeat dict, helper, and callback wiring into `worldsim/phase_4/runner.py`'s `run()` function. The helper may be inlined or extracted into `execution_helpers.py`. Single commit `chore(rebase): port variant progress heartbeats into phase_4 runner (8fcf0602)`.

**Verification.** `rg -n "variant_progress_by_task" worldsim/phase_4/` returns the new wiring. Long-running Phase 4 runs emit per-task variant progress events.

## Gap 15: Re-rebase onto `feat/worldsim-v5@446ff6cc`

**Status: resolved 2026-05-05 by `6fdc9d43` (port) + the rebase itself.** Upstream advanced from `38aa2f30` to `446ff6cc` (3 new commits: `446ff6cc fix: harden reddit comment attribution`, `4479a962 docs: record reddit comment attribution rca`, `6f4f0019 docs: clarify phase4 remote run state dir`). The rebase produced two structural conflicts:

- `worldsim/phases/phase_2_injections.py`: PR commit `66816e6c refactor: split phase 2 readiness helpers` deletes ~7382 lines, while upstream `446ff6cc` adds 21 lines (`_validate_reddit_submit_comment_state_probe` + a call from `_validate_final_state_action_reward_semantics`). Resolution: accept PR's modularization (`--theirs`), then port the upstream behavior into `worldsim/phase_2/plan_validation.py` as a separate `chore(rebase)` commit (`6fdc9d43`).
- `tests/test_phase_2_injections.py`: PR commit `76074ee5` deletes the file; upstream `446ff6cc` adds 47 lines of tests for the new validator. Resolution: accept PR's deletion, port the 2 tests into `tests/phase_2/test_plan_validation_3.py` (calls reach the helper through the `phase_2_injections` compat shim).

`worldsim/editors/reddit.py` and `worldsim/rewards.py` (the substantive reddit comment attribution hardening) flowed through three-way merge cleanly because PR HEAD never touched them. The 491 new lines in `tests/test_rewards.py` and 105 in `tests/test_adversarial_actions.py` plus 91 in `tests/test_phase_2_feasibility.py` also landed cleanly.

Verification: `tests/test_rewards.py -k reddit_comment` (and the broader local validation block) passes after rebase. At bug-hunt baseline head `dab75522`, `git log --oneline origin/feat/worldsim-v5..HEAD` showed 50 commits on top of `446ff6cc`, including the Gap 15 rebase port, Gap 4 fix, Gap 13 heartbeat/adaptive/progress fixes, the Reddit attribution hardening follow-up, and the latest Phase 4 runtime-knob/progress follow-ups.

## Gap 14: Stale `tests/test_phase_4_adversarial.py` reference in `agent_docs/verification.md`

**Status: resolved 2026-05-05 by `1634dab7`.** verification.md:11 now points to existing post-rebase test paths (tests/phase_4 plus the focused tests/test_phase_4_judge_api.py, tests/test_phase_4_variant_api.py, tests/test_phase_4_pvpo_capture.py).

**What.** `agent_docs/verification.md:11` contains an executable code block: `"uv run pytest -x tests/test_phase_4_adversarial.py tests/test_phase_4_pvpo_capture.py -q"`. The file `tests/test_phase_4_adversarial.py` was deleted by PR #10 (verified via `ls` and `git ls-files`).

**Why.** Active doc drift, not historical handoff content. Anyone copy-pasting this command gets a pytest collection error.

**Discovery (verified 2026-05-05).**
```
rg -n "tests/test_phase_4_adversarial|tests/test_phase_2_injections|tests/test_phase_2_target_resolver" agent_docs docs README.md
# Active hit: agent_docs/verification.md:11
# Historical hits in docs/handoffs/codex-handoff-*.md and docs/handoffs/archive/* stay as-is.
ls tests/test_phase_4_adversarial.py
# No such file.
```

**Fix.** Replace the path with an existing post-rebase Phase 4 test file path: `tests/test_phase_4_judge_api.py tests/test_phase_4_variant_api.py` or a `tests/phase_4/` glob. Verify `tests/test_phase_4_pvpo_capture.py` exists; substitute if also missing. Single commit `docs(rebase): align verification.md test paths with modular phase_4 layout`.

**Verification.** `rg -n "tests/test_phase_4_adversarial" agent_docs/` returns empty.

## Local validation block (run after every forward-port commit)

```
cd /Users/ashtonchew/projects/browser-sim/.codex-worktrees/agent-readiness-standardized

uv run ruff check worldsim tests scripts
uv run pytest tests/seed_contracts tests/phase_2 tests/phase_4 tests/test_readiness_audit.py tests/test_rewards.py -q
uv run pytest tests/test_phase_4_judge_api.py tests/test_phase_4_strategy_catalog.py tests/test_phase_4_result_summary.py tests/test_phase_4_variant_api.py tests/test_phase_4_transcript_purpose_api.py tests/test_pvpo_live_validation.py tests/test_phase_2_text_fill.py tests/test_benchmark_config.py tests/test_generate_compose_scale.py tests/test_remote_job_scripts.py tests/test_proxy_port_map.py tests/test_state.py tests/test_host_api_observability.py tests/test_phase_0_recon.py -q
uv run python -m worldsim.main --help >/dev/null
uv run python scripts/readiness_audit.py --json >/dev/null
bash scripts/verify_fast.sh
```

A green block is necessary, not sufficient. The live r5 wrapper remains the only gate for PR #10's research correctness, and is intentionally outside this TODO.

## Push protocol

- Always `git fetch origin` before pushing.
- Always `git push --force-with-lease` after a force-push-required operation. Never `--force`.
- After pushing, refresh `gh pr checks 10` once. Do not poll.
- Keep all `backup/pr-10-*` branches local. Do not push them.

## Done definition

- Gaps 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15: closed.
- PR body lists exactly which gaps were closed and which were intentionally accepted.
- Local validation block green on the final code HEAD from the closeout sequence (verified 2026-05-05 on `35e63174`; later bug-hunt follow-ups should rerun the focused local validation listed in their handoff or PR notes).
- `gh pr checks 10` either green or "no checks" with documented justification (the fork has no `.github/workflows/`, so "no checks" is expected).
- Live r5 gate still pending; that is the operator's call and is the final merge blocker.
