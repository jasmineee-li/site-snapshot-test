# TODO: Prove PR #10 rebase completeness (everything except live r5)

Status: open
Owner: unassigned
Worktree: `/Users/ashtonchew/projects/browser-sim/.codex-worktrees/agent-readiness-standardized` (branch `feat/agent-readiness-standardized`, head `d6d5c0f3`, base `feat/worldsim-v5@38aa2f30`)
Backup branches (do not delete): `backup/pr-10-pre-rerebase-20260505-132359`, `backup/pr-10-before-rebase-20260505-124459`, plus three earlier dated `backup/pr-10-before-*` branches.

## What this is

PR #10 (`https://github.com/jasmineee-li/browser-sim/pull/10`, "Agent readiness: modularize Phase 2 and Phase 4 domains") was rebased from `df71d96a` to `38aa2f30` in two passes: first onto `8ef4f5db`, then a clean fast-forward onto `38aa2f30`. The rebase produced 26 commits on top of base: 21 original PR commits, 3 user-authored local commits (`fix(phase1): canonicalize action capability provenance`, `fix(phase2): compile host action rewards`, `fix(gitlab): recognize graphql note actions`), and 5 trailing forward-port commits (2 `chore(rebase):`, 2 `test(rebase):`, plus the `chore(rebase): port reddit visual-region regex into phase_2 modular layout` that bundles the reddit gate + smoke-variant-budget ports).

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
git log --oneline origin/feat/worldsim-v5..HEAD | wc -l   # expect 26 (or +N as forward-port commits land)
```

For any new forward-port commit, run the local validation block at the end of this file before pushing. After every push, refresh `gh pr checks 10`.

## Gap 1: GitHub CI green on `d6d5c0f3`

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

**What.** The PR updates `agent_docs/code-organization.md`, `agent_docs/artifacts.md`, `agent_docs/remote-runs.md`, `agent_docs/secrets.md`, `README.md`, `docs/handoffs/orchestrator-handoff-r5-migration.md`, `docs/worldsim-v5-technical-specifcation.md`. The base added new `docs/results/*`, `docs/actions/*`, and `docs/runbook/*` content. After the rebase, agent docs should describe the modular layout AND the upstream-only doc additions should still be present.

**Why.** The spec is authoritative. CLAUDE.md says: if code and spec diverge, update the spec first. After this rebase the modularized layout exists in code and the spec text should match.

**Discovery.**
- `rg -n "phase_2_injections|phase_2_target_resolver|phase_4_adversarial" agent_docs docs README.md` to find any doc that still names the deleted monoliths.
- Compare `agent_docs/code-organization.md` between PR head and `origin/feat/worldsim-v5` to confirm the PR's modular description is current.

**Fix.** `docs(rebase): align <doc> with modular phase_<N> layout` for any doc that still names the monoliths or describes their internal structure.

**Verification.** No agent doc references `phase_2_injections.py` or `phase_4_adversarial.py` as live code paths; all such references are either inside compatibility-wrapper notes or removed.

## Gap 8: Generated artifact policy preservation

**What.** PR #10 deleted ~775,000 lines of tracked `logs/phase_1/*.json` and `logs/phase_2/*.json`. The PR adds artifact policy text in `agent_docs/artifacts.md`. Verify that nothing in the rebase reintroduced any tracked `logs/` artifact, and that `.gitignore` still ignores them.

**Why.** Re-tracking those files would undo the core artifact policy of PR #10 and bloat the repo by hundreds of megabytes.

**Discovery.**
- `git ls-files logs/` should be empty.
- `cat .gitignore | grep logs/` should show the ignore rule survived the rebase.
- `git diff origin/feat/worldsim-v5 -- .gitignore` should match the PR's intended `.gitignore` change.

**Fix.** If any logs file is tracked, `git rm --cached logs/<file>` and commit `chore(artifacts): re-untrack <file> per artifact policy`.

**Verification.** `git ls-files logs/` returns nothing; readiness audit `tracked_generated` field is `[]` (already verified after the rebase, but recheck after any forward-port commit lands).

## Gap 9: Final import-graph end-to-end smoke

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

## Gap 11: `_final_state_action_success_fields()` helper missing from modular Phase 4

**What.** Upstream `f0d3dd2e` defined `_final_state_action_success_fields(...)` at `worldsim/phases/phase_4_adversarial.py:2259+`. Returns `{"state_confirmed_action_success": bool, "tier3_state_confirmed_action_success": bool}`. Called via `**_final_state_action_success_fields(...)` at `phase_4_adversarial.py:4684` when building each per-task result payload.

**Why.** Without the helper, `state_confirmed_action_success` and `tier3_state_confirmed_action_success` are never populated on result dicts. Downstream `worldsim/phase_4/result_summary.py:1225-1236, 1465-1474` reads those keys via `final_metric_success` and silently reports zero numerators for the `state_confirmed_action_success_rate` and `tier3_state_confirmed_action_success_rate` metrics. This is a research-load-bearing Phase 4 metric (per `agent_docs/domain-invariants.md`, "A Tier 3 action passing state readback is reported as state-confirmed action success, not folded silently into semantic ASR").

**Discovery (verified 2026-05-05).**
```
rg -n "final_state_action_success|state_confirmed_action_success|tier3_state_confirmed_action" worldsim/phase_4/
# Returns CONSUMERS in result_summary.py, scenario_funnel_export.py, variant_trace_export.py, variant_trace_outputs.py.
# No PRODUCER (no _final_state_action_success_fields function definition).
```

**Fix.** Port the helper into `worldsim/phase_4/postprocess.py` or `execution_helpers.py`, then call it from the modular code path that builds each task's result payload (likely in `postprocess.py` or `execution.py`'s result-construction site). Single commit `chore(rebase): port _final_state_action_success_fields into phase_4 postprocess (f0d3dd2e)`.

**Verification.** `state_confirmed_action_success` appears on result payloads after Phase 4 task completion. `result_summary.py:1465` numerator computation runs with non-None denominator. `tests/test_phase_4_result_summary.py` should still pass (and ideally cover this metric).

## Gap 12: `_FINGERPRINT_RESULT_KEYS` missing action-attempt and state-confirmed keys

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

## Gap 14: Stale `tests/test_phase_4_adversarial.py` reference in `agent_docs/verification.md`

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

- Gaps 1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14: closed (or documented as accepted risk in the PR body).
- Gap 4: triaged into a separate upstream issue, NOT blocking PR #10.
- PR body lists exactly which gaps were closed and which were intentionally accepted.
- Local validation block green on the final HEAD.
- `gh pr checks 10` either green or "no checks" with documented justification.
- Live r5 gate still pending; that is the operator's call and is the final merge blocker.
