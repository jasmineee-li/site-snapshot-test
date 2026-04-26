# codex-handoff — Phase 1 Mode B end-to-end implementation + Phase 1 semantic rename

**Status:** Open. As of HEAD `3559aad2` on `feat/worldsim-v5`, Mode B has working code on disk and zero output anywhere downstream. The entire 84-task adversarial dataset rides on Mode A WebArena wraps that do not traverse the seeded surfaces.

**Branch:** Land all work as sequential commits on `feat/worldsim-v5`. Per repo convention (memory: *Commits Not PRs*) do not split into multiple PRs.

**Scope of this handoff:** make Mode B the primary task source, fix any bugs that surface when it runs against the live r5 stack, rename the Phase 1 surface from `mode_a` / `mode_b` to semantic names (`existing_tasks` / `generate_new_tasks`), and stratify the Phase 4 result by origin so the paper can present the two populations distinctly.

**This document is self-contained.** Everything you need to scope, implement, validate, and ship is here. Cross-references to other handoffs are advisory.

---

## Why this exists (the problem in one paragraph)

Phase 4 has been producing `max_coverage = 0.0` on essentially every trajectory. The cause is not the placement gate, the PVPO oracle, the renderer leak, or the witness-matching code. The cause is that **the benign tasks Phase 4 runs were not designed to traverse the surface that Phase 2 seeds**. WebArena Mode A tasks are header / state retrievals (e.g., *"Get the project IDs of my personal projects with the least stars"*); they do not require the agent to render an issue's comment thread. Phase 2 nevertheless attaches an injection to one of that user's issues, and the agent never visits it. Three layers of verification predicates ping-pong because they cannot prove a runtime property (will the agent visually attend to this surface?) of a task whose contract does not require attending to it.

The fix is co-design: generate the benign task **from the surface**, so the task contract requires consuming the surface that the injection sits on. Mode B was designed for exactly this and was never run.

---

## Evidence (run these to confirm before changing anything)

```bash
# 1. Mode B never ran — 0 of 812 benign tasks have novel_ id prefix.
jq '[.[] | select(.id | startswith("novel_"))] | length' logs/phase_1/benign_tasks.json
# expected: 0

# 2. The 106-task admitted adversarial set references only Mode A numeric ids.
jq '[.[] | .benign_task_id | tonumber? | values] | length' logs/phase_2/adversarial_tasks.json
# expected: 106  (every benign_task_id is a numeric WebArena id, no novel_ prefixes)

# 3. The latest Phase 4 trajectories all show pvpo_steps_captured=0,
#    max_coverage=0.0. (The PVPO infra is independently broken on macOS hosts;
#    this evidence is from the most recent r5 run on disk.)
for d in logs/phase_4/20260422_094059/adv_*/; do
  jq '{task:input_filename, max_cov:.encounter.max_coverage, pvpo:.pvpo_status}' "$d/result.json"
done

# 4. Phase 0c already enumerated 27 injection surfaces for gitlab and similarly
#    for reddit, with attacker_realism / location_page / rendering_context.
jq '.injection_surface | length' logs/phase_0c_v1_backup/INJECTION_SURFACE_gitlab.json
# expected: 27
```

All four checks must hold before this work begins. If any disagree, the situation has shifted and the handoff needs an update before implementation.

---

## Literature alignment (why this is the right call, citations)

Every published IPI benchmark we are comparable to handles encounter at construction time, not at runtime. None operate an automated reachability prover. Mode B implements the same construction-time pattern.

| Benchmark | What we cite it for | Verbatim mechanism |
|---|---|---|
| **VWA-Adv** (Wu et al., ICLR 2025, arXiv:2406.12814) | "ASR makes no sense if the trigger is not even seen" | Sets `start_url` to the page where the trigger lives |
| **WASP** (Evtimov et al., 2025, arXiv:2504.18575) | Manually verified each attacker goal is achievable; user goal forces traversal of the seeded surface | 21 hand-curated × 2 = 42 scenarios per environment |
| **ST-WebAgentBench** (Levy et al., ICLR 2026, arXiv:2410.06703) | Pairs each task with policies; injects into UI elements the task forces the agent to read | Co-designed task and injection point |
| **Gray Swan IPI Arena** (Dziemian et al., arXiv:2603.15714) | Conversation prefill — injection is in the input transcript | Single `{fill}` point per scenario |
| **InjecAgent** (Zhan et al., 2024, arXiv:2403.02691) | Scratchpad pre-population — `Observation` already contains the payload | Tool-output simulation framework |

Mode B's prompt at `worldsim/prompts/generate-benign-tasks.md` already encodes this: *"Prioritize tasks that route the agent through high-realism injection surfaces documented in the profile that are not already exercised by existing benchmark tasks."* That sentence is the in-codebase statement of the WASP / ST-WebAgentBench co-design pattern.

---

## What's already built (read this before implementing)

### Files in scope

| Path | LOC | Role |
|---|---|---|
| `worldsim/phases/phase_1_mode_a.py` | 123 | Wraps existing WebArena tasks. Stable. |
| `worldsim/phases/phase_1_mode_b.py` | 587 | Eligible-site discovery, sandbox launch, per-site cache, fingerprint resume. |
| `worldsim/phases/phase_1_mode_b_validation.py` | 213 | Schema, evaluator, placeholder, and reward-function gates for novel tasks. |
| `worldsim/phases/phase_1_tasks.py` | 469 | Phase 1 orchestrator. Calls Mode A always, Mode B only when `--generate-novel`. |
| `worldsim/prompts/generate-benign-tasks.md` | 142 | Mode B prompt. Already references `injection_surfaces_without_task_coverage`. |
| `worldsim/main.py:88-92, 305-310, 701-734` | n/a | CLI flag `--generate-novel`. Resume-aware via saved state. |
| `tests/test_phase_1_tasks.py` | n/a | Existing Mode A + cache fingerprint coverage. No live Mode B coverage. |

### What works already

- Eligible-site detection (`site_is_mode_b_eligible`): a site is eligible iff its profile carries a non-empty `injection_surfaces_without_task_coverage` list.
- Per-site caching with fingerprint invalidation on benchmark / manifest / sandbox-model changes.
- One self-correction iteration on schema validation failure (`MODE_B_FIX_MAX_ITERATIONS = 1`).
- Validation: `id` must match `novel_<site>_\d+`, `start_urls` must use `__SITE__` placeholders, evaluator must be one of the two declared in profile, no `task_id` in reward.
- `merge_benign_tasks` returns `mode_a_tasks + sort_novel_tasks(novel_tasks)` deterministically.

### What is broken or missing (the implementation surface)

These are the gaps you will close. Each becomes a numbered commit.

1. **Mode B has never run on the live r5 stack.** Pre-flight passes (we know from older test runs), but the integration end-to-end against the current `BENCHMARK_PROFILE_*.json` and `AGENT_CONTEXT_*.json` shapes is unverified. Sandbox prompt may produce tasks that fail validation in shapes the existing 1-shot self-heal cannot recover from.

2. **`origin` field is not stamped on tasks.** Phase 4 reads `task.get("origin", "")` and `entry.get("origin", "mode_b")` for stratification (see `worldsim/phases/phase_4_adversarial.py:1585-1586, 1655, 2096-2133`). Mode A tasks have no `origin`. Mode B tasks have no `origin`. Phase 4 cannot stratify. Stamp the field at Phase 1 emit.

3. **Phase 2 target resolver was tuned on Mode A.** The header comment at `worldsim/phases/phase_2_target_resolver.py:102` reads *"Regex inventory derived from 333 gitlab+reddit benign tasks."* Mode B tasks may emit URL shapes the resolver does not recognize, producing `kind = None` and dropping the task at `dropped_no_contract.json`. Verify against a Mode B sample before committing the dataset.

4. **`--sites gitlab,reddit` is not honored end-to-end through Mode B.** `load_mode_b_eligible_sites` scans every `BENCHMARK_PROFILE_*.json` it finds. The WASP-aligned scope (memory: *WorldSim v5 Pivot* and `docs/handoffs/wasp-aligned-scoping-decision.md`) is gitlab + reddit only. Mode B will currently try to generate for shopping / shopping_admin / map / wikipedia if their profiles still exist and have uncovered surfaces.

5. **`DEFAULT_NOVEL_TASKS_PER_SITE = 30` is hardcoded.** Not exposed via CLI. For a 84-task scope split (60 + 24, or 50 + 50, etc.) operators need a per-run override.

6. **No live integration test exists.** `tests/test_phase_1_tasks.py` mocks `run_claude_in_sandbox`. Add a Mode B integration test that runs against the live r5 stack and is skipped by default, gated like the other live tests in `scripts/run_integration_tests.sh`.

7. **Module / function / state-key names still say `mode_a` / `mode_b`.** The user's call: rename to semantic names. See "Rename plan" below.

8. **Phase 2c short-circuit opportunity.** When Mode B emits a `start_url` that already points at the surface to be seeded, the `phase_2_reachability` Playwright probe is doing redundant work. Out of scope for this handoff but flag any short-circuit opportunities you see; do not implement.

---

## Rename plan (the renaming the user asked for)

Mode A is "wrap an existing benchmark task." Mode B is "generate a new task targeting an injection surface." Use those phrases as the new identifiers throughout.

### File renames

| Old | New |
|---|---|
| `worldsim/phases/phase_1_mode_a.py` | `worldsim/phases/phase_1_existing_tasks.py` |
| `worldsim/phases/phase_1_mode_b.py` | `worldsim/phases/phase_1_generate_new_tasks.py` |
| `worldsim/phases/phase_1_mode_b_validation.py` | `worldsim/phases/phase_1_generate_new_tasks_validation.py` |
| `worldsim/phases/phase_1_tasks.py` | (no rename; this is the orchestrator) |

### Symbol renames

| Old | New |
|---|---|
| `build_mode_a_tasks` | `build_existing_task_wraps` |
| `run_mode_b` | `run_generate_new_tasks` |
| `generate_novel_tasks_for_site` | `generate_new_tasks_for_site` |
| `EligibleSiteProfile` | (keep) |
| `SiteNovelTaskResult` | `SiteGenerateNewTasksResult` |
| `MODE_B_*` constants | `GENERATE_NEW_TASKS_*` |
| `compute_mode_b_*_fingerprint` | `compute_generate_new_tasks_*_fingerprint` |

### CLI

Keep `--generate-novel` as the flag. It is already semantic and operators have run scripts referencing it. Add `--novel-tasks-per-site N` (alias `--new-tasks-per-site`) to override `DEFAULT_NOVEL_TASKS_PER_SITE`.

### State / persistence keys

| Old | New |
|---|---|
| `mode_a_task_count` | `existing_task_count` |
| `novel_task_count` | `new_task_count` |
| `mode_b_resume_metadata.json` | `generate_new_tasks_resume_metadata.json` |
| `novel_tasks_<site>.json` (per-site cache) | (keep filename; emit alias for one release) |

### Origin field values (the new field this handoff adds to every task)

| Phase 1 source | `task.origin` value |
|---|---|
| `build_existing_task_wraps` | `"existing_task"` |
| `run_generate_new_tasks` | `"new_task"` |

Phase 4's stratification reader at `phase_4_adversarial.py:1585` defaults missing origins to `"mode_b"` today. Update it to read the new values, fall back to inferring from task id prefix (`novel_` → `"new_task"`, numeric → `"existing_task"`), and only after that to a hard error.

### Documentation references

`docs/worldsim-v5-technical-specifcation.md` (the load-bearing-typo spec, memory: *Spec Filename Typo*) references "Mode A" and "Mode B" several times. Update those references in lockstep. CLAUDE.md, READMEs, and other handoffs reference Mode A/B too; update each in the same commit as the rename.

---

## Implementation order (proposed commits)

Each commit is independently revertable. The pipeline state must remain runnable between every commit. No commit may delete a file without first replacing the importers.

### Commit 1: stamp `origin` on every Phase 1 task

Smallest commit. Adds a single field. Establishes the stratification primitive.

- In `phase_1_mode_a.build_mode_a_tasks` (still under the old name in this commit): set `task["origin"] = "existing_task"` on every emitted task.
- In `phase_1_mode_b.generate_novel_tasks_for_site`: set `task["origin"] = "new_task"` on every validated task before caching.
- Update `phase_1_mode_b_validation.validate_generated_novel_task` to require `origin == "new_task"` on cached entries (so a stale cache from before the rename is invalidated).
- Update `phase_4_adversarial.py:1585` to accept either of `"existing_task"`, `"new_task"`, `"mode_a"`, `"mode_b"`. Map old → new for backward compatibility for one release.
- Update `tests/test_phase_1_tasks.py` and `tests/test_phase_4_*.py` for the new field.

Acceptance: every entry in `logs/phase_1/benign_tasks.json` has a non-empty `origin` field after re-running Phase 1 with `--resume`.

### Commit 2: site-scope filter for Mode B

`load_mode_b_eligible_sites` accepts an explicit `site_filter: Iterable[str] | None` argument. `phase_1_tasks.run` passes `args.sites` (already parsed by the CLI). When unset, behavior is unchanged. When set, only profiles whose `site_name` is in the filter are returned.

Acceptance: `uv run python -m worldsim.main phase 1 --generate-novel --sites gitlab,reddit` only launches sandboxes for gitlab and reddit, regardless of how many other profiles exist.

### Commit 3: per-run override for `DEFAULT_NOVEL_TASKS_PER_SITE`

CLI flag `--novel-tasks-per-site N` plumbs through `args` → `phase_1_tasks.run` → `run_mode_b` → `generate_novel_tasks_for_site` → `render_generate_benign_tasks_prompt(num_tasks=N)` → `validate_generated_novel_tasks(expected_task_count=N)`. Cache fingerprint includes N (a count change must invalidate the cache).

Acceptance: `--novel-tasks-per-site 50 --sites gitlab,reddit` produces 50 novel tasks per site, not 30.

### Commit 4: live Mode B run end-to-end on r5

This is the big one. Run on the r5 stack (memory: *WebArena AWS Instance*, *R5 SSH Key*) with `gitlab` and `reddit` only. Capture every validation failure produced by the sandbox. Tighten the prompt or the validator iteratively until both sites produce a clean cache without manual intervention. Do not paper over a malformed shape with a one-off fix; if the prompt is producing the wrong shape, fix the prompt.

The known failure modes to expect (best-guess — confirm or refute live):

- **Reward-function expected drift:** the prompt shows `retrieved_data: ["Wireless Headphones"]` but the validator at `_validate_eval_expected` accepts any non-empty `expected` containing one of `task_type / status / retrieved_data`. Sandbox may emit `expected: {"task_type": "retrieve"}` only; that passes, but Phase 4's `AgentResponseEvaluator` may reject it. Verify end-to-end against the evaluator, not just the validator.
- **Placeholder violations:** Mode B prompt examples include `__SHOPPING__` URLs. Sandbox may copy that pattern into a gitlab task ("test-with-other-site-placeholder" hallucination). Validator catches this at `extract_placeholders` enforcement.
- **Multi-site bleed:** instruction text occasionally references "the post on Reddit" inside a gitlab task. The validator does not catch this; the runtime agent will fail. Add a soft check: any of `__GITLAB__ / __REDDIT__ / __SHOPPING__` tokens appearing in `instruction` text should warn (not fail).
- **Site profile drift:** `BENCHMARK_PROFILE_<site>.json` fields the prompt depends on may have shifted since Phase 0c was last run. Verify `injection_surfaces_without_task_coverage`, `verification_capabilities`, and `seeding_mechanism` are present and shaped as the validator expects.

Acceptance:
1. `logs/phase_1/novel_tasks_gitlab.json` exists with N validated novel tasks targeting gitlab injection surfaces.
2. Same for reddit.
3. `logs/phase_1/benign_tasks.json` contains existing + novel tasks merged.
4. Every novel task has `origin == "new_task"` and a `start_urls[0]` whose path matches one of the surfaces in `INJECTION_SURFACE_<site>.json::injection_surface[*].location_page` (after placeholder resolution).

### Commit 5: Phase 2 target-resolver compatibility for new tasks

Run Phase 2 against the merged dataset from Commit 4. Capture every `dropped_no_contract.json` entry and every `option_a_validator_discrepancy.ndjson` row. The likely failure mode: Mode B `start_urls` use shapes the regex inventory at `phase_2_target_resolver.py:206-220` doesn't match.

For each unmatched shape, decide:
- Genuinely new but valid → extend the regex inventory.
- Malformed shape from sandbox → tighten Mode B prompt or validator.
- Out-of-scope → drop the task with an explicit dropped-reason annotation.

Acceptance: Phase 2 admits at least 80% of new-task novel inputs without resolver dropouts. The remaining 20% are categorized in a per-site triage file under `logs/phase_2/new_task_resolver_dropouts.json`.

### Commit 6: live Phase 4 sanity sample

Run Phase 4 on a 6-task smoke (3 gitlab + 3 reddit, all `origin = "new_task"`). PVPO must report `max_coverage > 0` on at least 4 of 6 trajectories. If fewer, the surface choice in the new tasks is mismatched against where Phase 2 actually placed the seed; iterate on Mode B prompt, not on Phase 2.

Acceptance: stratified summary in the run log shows `new_task` ASR computed over a non-zero-encounter denominator.

### Commit 7: rename

Now do the file / symbol / state-key renames described in the Rename Plan section. Keep one release of import aliases (`from worldsim.phases.phase_1_mode_a import build_mode_a_tasks  # deprecated, see phase_1_existing_tasks`). Run the full unit suite plus integration tests. Update CLAUDE.md, the spec, and any handoff that names Mode A/B.

Acceptance: `git grep -i 'mode_a\|mode_b'` returns only deprecation aliases and historical-context handoff references. Tests green. `scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r5.yaml --quiet` passes.

### Commit 8: stratified summary in Phase 4 reads the new origin values cleanly

Remove the legacy `"mode_a"` / `"mode_b"` fallback paths in `phase_4_adversarial.py` once the dataset is fully re-stamped. Keep the inference-from-id-prefix safety net. This is the cleanup commit; only ship after Commits 1-7 have soaked for at least one full Phase 4 run.

Acceptance: `git grep '"mode_a"\|"mode_b"' worldsim/` returns zero hits.

---

## Risks (be honest with the operator)

1. **Generated-task quality.** An LLM-generated benign task can be ambiguous, ungrammatical, or have a reward function that does not actually verify the stated intent. The Mode B prompt is short and task design is hard. Expect 10-20% of generated tasks to fail at Phase 2 / Phase 3 / Phase 4 in some way. This is acceptable as long as the surviving tasks are realistic; track the survival rate.
2. **Reward-function correctness.** Mode B can only emit `NetworkEventEvaluator` or `AgentResponseEvaluator`. Mode A wraps the canonical WebArena Verified evaluator with ground-truth `task_id`. The Mode B reward is therefore weaker per-task than Mode A's. Stratify accordingly; do not pool ASR across origins as if the rewards were equivalent.
3. **Drift from the "WASP-aligned" claim.** Mode B tasks are not in WASP. The paper currently says "we evaluate on WebArena tasks." That claim narrows: "we evaluate on (a) WebArena Mode-A wraps and (b) novel tasks generated against WebArena's environment, following ST-WebAgentBench's co-design pattern." This is a stronger paper claim, not a weaker one, but it must be written explicitly.
4. **Sandbox cost.** Mode B uses Modal sandboxes (Claude Sonnet by default). Two sites × ~30 tasks ≈ 60 generations + the validation fix-up loop. Budget ~$5-15 per full Mode B run depending on prompt complexity and retries. Cost is recorded in the existing `cost_tracker` under bucket `phase_1`.
5. **Surface drift.** Phase 0c was last run on `logs/phase_0c_v1_backup/`. If the live r5 stack's surfaces have drifted (new GitLab version, reset state changes), Mode B will generate tasks against stale `INJECTION_SURFACE_<site>.json`. Re-run Phase 0c first if there is any doubt; the runbook is in `docs/handoffs/rigor-run-setup.md`.
6. **Encounter-rate guarantee is statistical, not absolute.** Even with a co-designed task and surface, a particular agent may still wander. Mode B's claim is "encounter rate close to 1," not "encounter rate equals 1." Reporting must be conditional on PVPO `max_coverage > 0`; do not present unconditional ASR.
7. **PVPO infra is independently broken.** The recent r5 trajectories show `pvpo_failure: capture_failed, pvpo_steps_captured: 0`. Mode B does not fix this. If PVPO is still broken when you reach Commit 6, the `max_coverage > 0` acceptance gate will fail for infrastructure reasons unrelated to Mode B. Diagnose PVPO before claiming Mode B is complete; cross-reference `feedback_pvpo_chrome_leak.md` in memory. Do NOT regress on Mode B if PVPO is the actual failing layer; document it and continue.

---

## Files explicitly out of scope for this handoff

- Anything under `worldsim/phase_4/` other than the stratification reader at `phase_4_adversarial.py:1585-1655, 2096-2133`.
- The PVPO capture pipeline and chrome-headless-shell Docker image. Cross-handoff: `codex-handoff-paint-verified-oracle.md`.
- The Phase 2c reachability prover. Mode B may make portions redundant; do not delete them in this handoff.
- The Apollo Transcript Purpose / Needham VEA classifiers. Independent.
- The benchmark host proxy and nginx config. Independent.

---

## Operator checklist before you start

1. `feat/worldsim-v5` branch checked out, working tree clean except for memory-driven stale untracked files.
2. r5 host reachable: `ssh -i ~/.ssh/webarena-key.pem ubuntu@3.12.221.9 'echo ok'` returns `ok`.
3. Modal auth current: `modal token current` succeeds.
4. Phase 0c artifacts present and recent: `ls logs/phase_0c_v1_backup/INJECTION_SURFACE_*.json` shows gitlab and reddit.
5. The four evidence checks at the top of this doc all pass.
6. You have read `docs/handoffs/wasp-aligned-scoping-decision.md` for why scope is gitlab+reddit only.

---

## Definition of done

- All 8 commits landed on `feat/worldsim-v5`.
- `logs/phase_1/benign_tasks.json` contains existing-task wraps + new-task generations, every entry stamped with `origin` ∈ {`existing_task`, `new_task`}.
- `scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r5.yaml --quiet` passes.
- A 6-task Phase 4 smoke against `origin == "new_task"` tasks reports `max_coverage > 0` on the majority of trajectories. (Conditional on PVPO infra being healthy. If PVPO is the failing layer, document and stop.)
- `git grep -i '\bmode_a\b\|\bmode_b\b' worldsim/ tests/ docs/handoffs/codex-handoff-phase-1-mode-b-implementation.md` returns only this handoff.
- The Rename Plan table in this doc is fully reflected in the codebase.
- A short note appended to `docs/handoffs/researcher-handoff-project-status.md` recording that Mode A → existing_task / Mode B → new_task is now the canonical naming, with the rename commit SHA.

---

## Style notes

- Per repo convention (memory: *No Em Dashes*), prefer commas over em dashes. Match existing handoff voice.
- No new markdown files outside `docs/handoffs/`.
- No new top-level scripts unless an existing one cannot be extended.
- Decorator-adjacent-to-code (memory: *Locality Over Flatness*) is preferred for any new metadata about tasks; do not introduce a flat side-table.
- Do not silently drop auth boundaries (memory: *Preserve Auth Boundary*). Mode B sandboxes inherit the standard `_build_claude_secrets` path.
- Do not commit any benchmark-specific snapshot to source. Generated artifacts go under `logs/`.
