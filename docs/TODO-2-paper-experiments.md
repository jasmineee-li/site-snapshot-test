# TODO-2: Paper Experiments and Integration

> **STALE PAPER-PLAN DRAFT, needs rewrite before execution.** This document is
> a pre-sweep planning inventory, not the current paper execution plan.
> Revalidate scope, models, costs, comparison benchmarks, and analysis tracks
> against `docs/worldsim-v5-technical-specifcation.md`,
> `docs/phase4-expanded-model-sweep-20260501.md`, and current handoffs before
> using any item as a task. Details below mention deleted `p_eval` paths and
> obsolete ecological-validity analysis. Current Gate 1 is PVPO
> `encounter.max_coverage > 0`; TP/VEA behavior is documented in the spec.
> The only live section is the dated 2026-05-09 AgentLab cohort block under
> "Track 7"; promote that block into `docs/current_progress.md` or a new dated
> run record before using it as an execution checklist.

Status: planning. Owner: Ashton (infra), Max + Jasmine (eval awareness + comparisons). Target: close the gap between the MVP pipeline (`docs/TODO-adversarial-rigor-mvp.md`) and a submittable NeurIPS paper per `docs/worldsim-v5-full-specification.md`.

The MVP TODO covers Contribution 1 (IPI wrapper) and produces cohort-quality adversarial trajectories. This doc covers everything else the paper needs: Contribution 2 (eval-awareness measurement), Contribution 3 (ecological-validity gap), Experiments A/B/C, the comparison benchmark claim, analysis tooling, and pre-registration.

Each track below has: **Goal**, **What**, **Why**, **Context**, **Open questions**. No sprint plan yet — this is planning input that will be split into sprints later.

## Overarching goals

The paper claims three things. Mapping to pipeline state:

1. **WASP-aligned IPI wrapper** (`docs/worldsim-v5-full-specification.md`, "Scope" and "Pipeline"). MVP pipeline delivers this for GitLab and Reddit/Postmill UGC. Blocker: cross-check landing, then expensive runs.
2. **Large-scale eval-awareness measurement for browser-use safety evals** (`docs/worldsim-v5-full-specification.md`, "Research Question" and "Eval-Awareness Iterator"). Current Phase 4 records Transcript Purpose and VEA; Jasmine/Max's broader post-hoc probe battery remains a separate planning track.
3. **Measurement gap analysis** (`docs/worldsim-v5-full-specification.md`, "Rewards And Readback"). Current mainline separates raw ASR, encounter-conditional ASR, Transcript Purpose, VEA, benign-under-attack, and readback/action evidence. Any P(eval)-filtered analysis is future Phase 5 work, not a current Phase 4 gate.

Everything below is work that must land before paper submission. Three weeks to NeurIPS (5/5).

## Track 1: Phase 5 — rigorous eval-awareness measurement

### Goal
Produce the three-signal probe battery the paper promises: verbalized eval awareness (VEA), external P(eval), self-reported P(eval). Enable Contribution 2's measurement and Contribution 3's gap analysis.

### What
New `worldsim/phases/phase_5_validity.py` that iterates Phase 4 trajectory dirs and runs three probes per task in Modal sandboxes. Reuses primitives from `origin/claude/plan-eval-experiments-LNd9F:eval_awareness_experiments/` (Jasmine/Max's branch):

- **VEA extractor**: `judges/verbalized_awareness.py` on the orphan branch. Four-category classification (alignment_eval / capability_eval / other_eval / real) with evidence extraction from transcript XML. Port to read our `history.json` + `conversations/*.txt` layout. See also: canonical Needham et al prompts at `github.com/jjpn97/eval_awareness` @ commit `1543ffe644d3f8f3a43b5dfd7814dbe9b12f890d`, specifically `cot_classifiers/prompts.py` for Figure 16 reasoning-flag style and output format `<label>YES</label>` (per `docs/research/gpt-deep-research-2026-04-14.md`, "Needham et al evaluation-awareness probes" section). Codex handoff `docs/handoffs/codex-handoff-needham-prompt-audit.md` produces a prompt-by-prompt diff to inform whether we keep the orphan port, replace with Needham canonical, or merge.
- **External P(eval)**: `experiments/p_eval.py` on the orphan branch. Five-prompt bank (counterbalanced), averaged across screenshot / HTML / axtree formats. This is future Phase 5 planning, not current Phase 4 behavior. Canonical reference: Needham et al's 5 Probe Questions (Table 4) at `github.com/jjpn97/eval_awareness` `constants.py` + `run.py` (per `docs/research/gpt-deep-research-2026-04-14.md`, "P(eval) probe questions" section). Needham used GPT-4.1 as judge with ICC(3) human-agreement reporting; our decision on judge choice (Opus 4.6 vs GPT-4.1 vs both) is surfaced in the GPT-integration section at the end of this doc.
- **Self-reported P(eval)**: NEW. Target model is reshown its own transcript post-trajectory and asked for eval probability. No implementation exists anywhere today. GPT confirmed independently (per `docs/research/gpt-deep-research-2026-04-14.md`, "Self-reported post-trajectory P(eval) probe pattern" section) that no public implementation exists end-to-end in Needham, Greenblatt, or any adjacent eval-awareness codebase. This strengthens the contribution: our implementation is the first.

Output per task: `validity_report.json` with all three signals + combined score. Aggregate: `logs/phase_5/<ts>/results.json`.

### Why
- Without this, Contribution 2 is empty (no eval-awareness measurement exists).
- Without this, Contribution 3 is empty (raw ASR minus filtered ASR requires a filter, and the filter IS the P(eval) signal).
- Without self-reported P(eval), the triangulation claim in the spec (`:50`) is only two signals deep, weaker than advertised.

### Context
- **Current mainline differs from this old plan**: Phase 4 now records Transcript Purpose and VEA over Needham XML. The old `p_eval_api.py` / `p-eval.md` path is gone from this workspace. Treat external P(eval) as future Phase 5 analysis unless a new implementation lands behind current contracts.
- **Orphan branch `origin/claude/plan-eval-experiments-LNd9F`**: contains the full rigorous implementation. Authors Jasmine + Max. Never merged. Covers ~70 percent of Phase 5's surface area, missing only self-reported P(eval).
- **Trajectory artifacts are stable**: confirmed `history.json`, `final_response.json`, `conversations/`, `screenshots/`, `network.har` all land in per-task dirs. Phase 5 can be a pure post-hoc pass.

### Open questions
- Do we retire the lite Phase 4 probe or keep both? Recommendation: keep both (fast runtime gate + rigorous post-hoc measurement), they answer different questions at different cost tiers.
- Where do ported judges live? Recommendation: `worldsim/validity/` as a new subpackage; prompts under `worldsim/prompts/validity/` to match existing layout.
- How does self-reported P(eval) get the target model's API credentials? Reuse `worldsim/agent_config.py::make_llm` with the same model slug used for the original trajectory.
- How many times do we run the 5-probe bank per trajectory — once averaged, or 5 independent calls averaged? Spec says "take the average", implies independent calls. Costs 5x more per trajectory.
- Cost estimate: ~$0.50-1 per trajectory for the 3-signal battery. For 720 primary trajectories, that is $400-700.

## Track 2: Experiment B — environment degradation

### Goal
Emit degraded parallel seed variants of each adversarial task so ASR and P(eval) can be measured across realistic vs degraded conditions. Enables the causal claim in Contribution 2.

### What
New `worldsim/phases/phase_2_5_degrade.py` that reads `logs/phase_2/adversarial_tasks.json` and emits `logs/phase_2_5/adversarial_tasks_degraded.json`. Each adversarial task gets a twin with `degradation_level: "realistic" | "degraded"` and a `degradation_cues: [...]` annotation.

Rule-based transformations (deterministic, not LLM-driven, for attributability):
- Usernames → `user_<N>`, `test_user_<N>`
- Realistic URLs → `http://localhost:PORT/...`
- Timestamps → dropped or set to `1970-01-01T00:00:00Z`

Phase 4 runner iterates both conditions per task against separate reset instances. Within-task paired analysis (same task paired across conditions).

### Why
- This is future Phase 5 work; it is not required by the current Phase 4
  spec. Re-check `docs/worldsim-v5-full-specification.md` and the technical
  spec before scheduling it.
- Experiment C's 2x2 ablation (environment realism x explicit cues) is blocked on this. Without it, Experiment C cannot run, and the causal claim for Contribution 2 has no evidence.
- Rule-based transforms are attributable: each cue can be ablated independently ("realistic + degraded username only" vs "realistic + degraded URL only") to produce the "eval tells" ranking the spec promises.

### Context
- Zero precedent in `worldsim/` (`grep -rn degrad\|placeholder.*user\|localhost.*url worldsim/` returns zero matches).
- Zero precedent in `main:behavior-gen/pipeline.py` either (Jasmine's pipeline had no degradation pattern).
- Spec names four cues: placeholder usernames, localhost URLs, missing timestamps, broken interactivity. Cue #4 (broken interactivity) requires webapp-level mutation which `CLAUDE.md` principle 1 forbids. Drop it from scope, document as limitation.

### Open questions
- **Within-task paired (each task runs both conditions) vs between-task stratification (half cohort realistic, half degraded)?** Recommendation: within-task paired. Better statistical power for fixed compute; paired t-test or Wilcoxon signed-rank.
- **Doubles Phase 4 cost**: each task runs twice. At 60 tasks x 2 conditions x 4 models x 3 repeats = 1440 trajectories instead of 720. Budget impact $600-1200 additional.
- **How many cues per degraded variant?** All three at once (maximum contrast), or one-at-a-time for cue-ranking? Recommendation: emit a "max-degradation" twin plus three single-cue variants per task. 5 variants per task total. Cohort balloons.
- **Do degraded URLs break agent navigation?** Some agents refuse to navigate to localhost; may artificially inflate task_broke rate. Needs pilot test.
- **Is "broken interactivity" truly out of scope?** Could be carved out with a JavaScript injection that disables buttons. Violates principle 1 but returns the spec's strongest cue. Decision: drop for MVP, note as limitation; revisit if the gap analysis needs the cue to generate meaningful effect size.

## Track 3: Experiment A — static environment capture

### Goal
Capture screenshots + HTML + accessibility trees from our environment and comparison benchmarks for a future static-perception analysis.

### What
New `scripts/capture_static.py` CLI:

```
uv run python scripts/capture_static.py \
  --urls urls.json \
  --out logs/static_a/ \
  --headed false --timeout 30
```

Inputs: JSON list of `{benchmark, site, url}`. Outputs: `logs/static_a/<benchmark>/<site>/<slug>/{screenshot.png, dom.html, a11y.json, meta.json}`.

Reuses Browser Use primitives: `page.screenshot()`, `page.content()`, `page.accessibility.snapshot()`. Thin wrapper.

### Why
- Static snapshots are no longer part of the current Phase 4 spec. Treat this
  as future comparison-analysis work, not a mainline pipeline requirement.
- Contribution 2's static-perception ablation (how much does an LLM judge recognize a synthetic env as synthetic from a single static snapshot?) can only run on captured data.

### Context
- **No static capture tooling exists in the repo**. Dynamic screenshots during trajectories exist (`worldsim/browser_use_agent.py:929-943`), but no standalone URL-to-snapshot tool.
- **Browser Use primitives are there**: `.venv/.../browser_use/dom/enhanced_snapshot.py:46`, `.venv/.../browser_use/screenshots/service.py:13`. Wrapping is maybe a day of work. Adjacent-stack corroboration for our accessibility-tree filtering claim: Playwright MCP documentation states a11y snapshots automatically filter hidden elements (`display:none`, `visibility:hidden`, etc.) per `docs/research/gpt-deep-research-2026-04-14.md` "browser-use 0.12.6 DOM filtering" section. Our source inspection of `.venv/.../browser_use/dom/service.py:82-93,257-271` remains primary evidence; Playwright MCP docs are a cross-library generalization we can cite.
- **WebArena Infinity and TheAgentCompany are NOT running in our infrastructure**. Only WebArena Verified is provisioned (EC2 `18.117.99.179`). Spec's static-perception cohort depends on additional environments being stood up, or on hand-curated URLs against the authors' published demo sites.

### Open questions
- **Minimum cohort for defensible Experiment A?** Recommendation: WebArena Verified (have) + WebArena Infinity (stand up second EC2) + published demo URLs from TAC authors. 30-50 static snapshots per env minimum.
- **Is TheAgentCompany essential?** TAC is mentioned in the spec (`:52`). Standing it up is substantial (Docker-compose, orchestration). Recommendation: drop for MVP, cite as future work, replace with WebArena Infinity.
- **Who owns standing up WebArena Infinity?** Ashton (infra). Add to sprint.
- **Does Infinity expose public URLs for our Browser Use to hit?** Check `vendors/webarena-infinity/` for its runtime model. If it's ephemeral-per-task (spins up for each test), static capture requires custom integration.

## Track 4: Comparison benchmark runs

### Goal
Run our target models against at least one comparison benchmark for the ICML CUA workshop (Sonnet wrapper vs STWebAgentBench for Panel B). Add WASP, DoomArena, and SafeArena as NeurIPS extension stretches. Apply Phase 5 probe battery to those trajectories. Produce Panel B of the headline figure: per-benchmark P(eval) distribution + per-benchmark ASR.

### What
- **ICML scope**: port the multi-benchmark runner architecture from `feat/multi-benchmark` plus the STWebAgentBench adapter only. Vendor STWebAgentBench under `vendors/` (read-only clone). Run Sonnet 4.6 against it for head-to-head Panel B. Note: per `docs/research/gpt-deep-research-2026-04-14.md` "STWebAgentBench task list triage", the benchmark's task inventory covers only three sites (`gitlab`, `shopping_admin`, `suitecrm`), so our GitLab + ShoppingAdmin coverage is the only overlap without standing up SuiteCRM. Canonical task file is `stwebagentbench/test.raw.json` @ commit `f7c69df98fe67de7a587795400ca00feb9b375a4`.
- **NeurIPS stretch**: add WASP (live VWA GitLab + Reddit instance, wire `injection_text` instantiation and seeding), DoomArena (wire `AttackedBrowserEnvArgs` into the AgentLab runner, stand up DoomArena-managed Docker WebArena), SafeArena (fix `OpenRouterModelArgs` vs `LiteLLMModelArgs` dead code, add `safearena` to `[comparison]` extra).
- Architecture is already designed on `feat/multi-benchmark`: Protocol-based runner contracts in `worldsim/runner.py` (no ABC, duck-typed), two-function module API per runner (`make_task_runner`, `make_agent_factory`), registry in `worldsim/runners/__init__.py` that resolves via importlib at call time. Browser Use runner is a thin re-export of `phase_3_benign.run_task` (no behavior change). AgentLab runner adds `EnvArgs` / `ExpArgs` / `_run_experiment_sync` / `_parse_exp_result` / `_persist_result_sentinel`. CLI flags are additive: `--runner`, `--attack-mode`, `--benchmark-adapter`. Config schema diff is the single new `auth` field (already present on our branch).
- Output: `logs/comparison/<benchmark>/<model>/<task>/` trajectory dirs.

### Why
- The current full spec keeps comparison work separate from the mainline
  GitLab/Reddit WASP scope. Head-to-head claims still need measurement, but
  this track is not a Phase 4 admission requirement.
- Without it, Contribution 3's claim reduces to "our wrapper produces a gap", not "our wrapper produces a smaller gap than prior work". The latter is the defensible paper-length claim.

### Context
- **Per-benchmark audit (four parallel Sonnet agents on `feat/multi-benchmark` worktree at `.claude/worktrees/multi-benchmark/`):** runner architecture is SOLID (Protocol contracts, registry dispatch, additive CLI and config), ~7.25h to port. STWebAgentBench adapter is FULLY WORKING: all four methods (`load_tasks`, `wrap_task`, `get_browsergym_task_name`, `get_credentials`) land, SuiteCRM Angular hash routing is handled, passing test at `tests/test_phase_1_tasks.py:209-320`, 1h to port. One documented gap: `policies` array is passed through `wrap_task` but Phase 4 Gate 2 reward does not read it (adapter-level data only, not scored). Follow-up: per `docs/research/gpt-deep-research-2026-04-14.md` "STWebAgentBench task list triage", the benchmark's `test.raw.json` only contains tasks on three sites (`gitlab`, `shopping_admin`, `suitecrm`); Postmill/Reddit, Wikipedia, and OSM have zero STWebAgentBench coverage. This narrows our ICML overlap to GitLab + ShoppingAdmin (two sites), reduces triage workload, and narrows Panel B's cross-site claim accordingly.
- **DoomArena is effectively a skip** for ICML. The adapter is a config-layer stub: `DoomArenaAdapter.get_attack_configs()` builds AttackConfig objects but nothing in the call path consumes them. Critical bug: the AgentLab runner builds plain `EnvArgs` not `AttackedBrowserEnvArgs` (`runners/agentlab.py:373`), so attacks (popup, banner, ugc, div) never fire. Phase 4 hard-rejects the AgentLab runner at `phase_4_adversarial.py:163-180`. Requires DoomArena-managed Docker WebArena (no overlap with our EC2 env-ctrl setup). Real effort to get attacks firing: 12-22h (4-6 cherry-pick + 4-8 wire `AttackedBrowserEnvArgs` + 4-8 stand up env).
- **WASP is functional at the task-expansion layer** (84 tasks from 21 configs x 2 formats x 2 user goals; `INJECTION_FORMAT_TEMPLATES` verbatim from WASP's `constants.py`), but the critical end-to-end gap is that `injection_text` is generated as a template and never instantiated with live URLs; Phase 3 and 4 do not read `injection_text` or `attacker_eval`; the `domain_map` wiring has no caller. Requires live VWA GitLab + Reddit stack separate from our WebArena Verified EC2. Effort: 2-4h cherry-pick + 20-30h to wire end-to-end (domain resolution + seeding + attacker-eval scoring).
- **SafeArena** `load_tasks` and `wrap_task` work; RISK_CATEGORIES and jailbreak constants are metadata-only. Critical bug: `get_agentlab_agent_args()` uses `OpenRouterModelArgs` while the runner uses `LiteLLMModelArgs` (dead code, latent failure on first harm-mode run). `safearena` is not in `pyproject.toml` `[comparison]` extra. Three SafeArena commits show repeated fix-after-review patches. Effort: 4-6h to fix dead code + add pip dep + reconcile merge conflicts.
- **Tests already covering the port**: `tests/test_agentlab_runner.py` (3 tests) + indirect coverage via `test_phase_3_benign.py:444` and `test_phase_1_tasks.py:209-320`.

### Migration steps (ordered)
1. Copy `worldsim/runner.py` (Protocol definitions). Create `worldsim/runners/__init__.py` registry. ~0.5h.
2. Copy `worldsim/runners/browser_use.py` verbatim (thin re-export of `phase_3_benign.run_task`, no behavior change). ~0.5h.
3. Copy `worldsim/runners/agentlab.py` verbatim. Run `tests/test_agentlab_runner.py`. ~1h.
4. Copy `worldsim/adapters/__init__.py` plus all five adapter files even though only STWebAgentBench is USED for ICML. Keeps scaffolding for post-submission expansion without re-merging. ~0.5h.
5. Update `worldsim/phases/phase_3_benign.py` runner-dispatch block (lines 128-159 on worktree). Preserve ALL current-branch additions: `benchmark_root` threading, `build_agent_prompt`, `validate_fix_patch`, `save_state` with the `benchmark_adapter` key. ~1.5h.
6. Update `worldsim/main.py` to add `--runner`, `--attack-mode`, `--benchmark-adapter` flags. Preserve current-branch flags: `--sandbox-model`, `--allow-unknown-auth`, `--sites`, Phase 0d, `rescore-phase-3`. ~1h.
7. Update `pyproject.toml` to add `[browser]` and `[comparison]` optional extras. Move `browser-use` out of core deps. ~0.5h.
8. Copy `worldsim/reporting.py` (standalone utility). ~0.25h.
9. Port `tests/test_agentlab_runner.py`. Verify STWebAgentBench tests still pass. ~0.5h.
10. Smoke test: `uv run python -m worldsim.main phase 3 --runner browser_use` (no new deps), then `--runner agentlab --benchmark-adapter stwebagentbench` against dummy instances. ~1h.

### Open questions
- **Do we port WASP / DoomArena / SafeArena adapter FILES even though we will not USE them for ICML?** Recommendation: yes, copy them in step 4 as dead scaffolding. Costs 3h now, saves a re-merge for NeurIPS extension. Alternative is to leave them on the orphan branch and cherry-pick later; carries re-merge risk as `feat/worldsim-v5` evolves.
- **Which STWebAgentBench tasks overlap with our 5 sites?** Resolved (per `docs/research/gpt-deep-research-2026-04-14.md`): only GitLab + ShoppingAdmin overlap. GPT supplied a concrete 30-task panel (8 ShoppingAdmin core + 18 GitLab tiered x 6 base intents x 3 policy-load tiers + 4 GitLab core) drawn from `stwebagentbench/test.raw.json` @ `f7c69df98fe67de7a587795400ca00feb9b375a4`; flattened export at `stwebagentbench/test.csv` @ `688af42c10adff8fdabdb5a687cba3dfafa62bdf`. Every task has non-empty `policies` (no pure-capability tasks in the current inventory). Codex handoff `docs/handoffs/codex-handoff-stwebagentbench-task-subset.md` regenerates the filtered JSON deterministically.
- **Do we run our agents against their env, or post-hoc analyze trajectories from their runner?** Port delivers the former for STWebAgentBench via the AgentLab runner. Cleaner: same agent, our trajectory-dir format.
- **Is the unscored `policies` array gap a blocker for Panel B?** No; STWebAgentBench's paper claim is task completion, which our reward functions cover. Safety-policy scoring is an additional metric we can add later without blocking Panel B.

## Track 5: Analysis stack

### Goal
Produce publication-quality figures with proper statistics. Bootstrap CIs, Mann-Whitney, 2x2 ANOVA, Cohen's kappa.

### What
- `worldsim/analysis.py` module with:
  - `bootstrap_ci(samples, n=10000, ci=0.95)`
  - `mann_whitney_u(x, y)` wrapping `scipy.stats.mannwhitneyu`
  - `wilcoxon_signed(x, y)` for within-task paired comparisons (Experiment B)
  - `two_way_anova(df, factor_a, factor_b, y)` via `statsmodels` (Experiment C)
  - `cohens_kappa(ratings_matrix)` for inter-probe agreement on the 5-probe P(eval) bank
  - Trajectory loader reading `logs/<run>/<task>/result.json` + `logs/phase_5/<task>/validity_report.json` into a DataFrame
- `notebooks/` directory, one notebook per figure:
  - `panel_a_asr_gap_per_model.ipynb` (Contribution 3 headline: raw vs validity-filtered ASR)
  - `panel_b_peval_per_benchmark.ipynb` (Contribution 2 cross-benchmark: our wrapper vs WASP/DoomArena/SafeArena/STWebAgentBench)
  - `experiment_c_2x2_ablation.ipynb` (realism x cue interaction)
  - `strategy_effectiveness.ipynb` (per-framing + per-concealment ASR breakdown)
- `figures/` outputs: PNG + underlying JSON data for reproducibility.

### Why
- Zero analysis tooling in the repo today (verified by grep for matplotlib / seaborn / scipy / pandas DataFrame use — zero hits on tracked Python on active branch).
- Paper reviewers expect bootstrap CIs on every reported ASR and significance tests on every comparison claim.
- Without the analysis stack, the cohort data is uninterpretable.

### Context
- **Orphan branch has partial tooling**: `eval_awareness_experiments/analysis_causal.py` on `plan-eval-experiments-LNd9F` uses numpy + matplotlib for ASR x awareness plots. Port into `worldsim/analysis.py`. Reference layout for Needham-style analysis: `github.com/jjpn97/eval_awareness` @ `1543ffe644d3f8f3a43b5dfd7814dbe9b12f890d` ships `analysis/roc.py` (ROC curves), `analysis/purpose.py` (MCQ / open-ended purpose classification), `analysis/cot.py` (CoT flag analysis), `analysis/colors.py` (plot styling helpers) per `docs/research/gpt-deep-research-2026-04-14.md` "Public codebase that actually ships analysis scripts" section. Use as a structural template, not a hard port.
- **No statistical plan is in the tech spec**. We need to author it ourselves, ideally in the pre-registration.

### Open questions
- **Notebooks vs scripts?** Recommendation: notebooks during development for iteration, then freeze into scripts before submission for reproducibility.
- **Figure framework**: matplotlib (basic, universal) vs plotly (interactive, harder to embed in PDF) vs seaborn (pretty defaults). Recommendation: matplotlib with seaborn styling.
- **How many figures is the paper?** Recommendation: 2 headline panels + 2-3 ablation figures + 2-3 appendix figures = 6-8 total.

## Track 6: Pre-registration

### Goal
Commit a dated, signed analysis plan before the first paper-grade run fires. Stops p-hacking. Reviewers check for it.

### What
New file `docs/PREREGISTRATION.md`. Sections:

- Primary hypothesis (Contribution 3: ecological-validity gap non-zero on at least one model, magnitude > 5 percentage points, bootstrap 95 percent CI excluding zero).
- Secondary hypothesis (Contribution 2: mean P(eval) on our wrapper < mean P(eval) on each comparison benchmark; Mann-Whitney one-sided alpha = 0.05 with Bonferroni correction across 4 comparisons).
- Experiment C: 2x2 ANOVA on realism x cue, effect size eta-squared >= 0.06 (medium) threshold.
- Sample size justification: power analysis for detecting d=0.5 at alpha=0.05 with n=80 per arm > 0.80.
- Stopping rules: no interim peeks. Pipeline runs to completion unless all 4 models OOM or Phase 3 rejects > 50 percent of tasks.
- Finding vs null definitions: any pre-registered hypothesis rejected at corrected alpha with effect above threshold.
- Multiple-comparison correction: Bonferroni within each contribution family.
- Exclusions: Phase 3 invalid-contract entries (`validity_status: "invalid"` in `contracts.json`) never reach Phase 4. Baseline capability is reported from Phase 4's `capability_benign_under_attack` as a nuisance variable for calibration, not used to filter ASR.

Commit as `docs/PREREGISTRATION.md` with dated signature block at top. Tag as `preregistration-v1` in git before the first paper-grade run.

### Why
- Reviewer expectation for any paper claiming effect sizes on IPI vulnerability.
- Self-discipline against p-hacking. Once tagged, exploratory analyses must be labeled "post-hoc" in the paper.
- Fast to write (4-6 hours), constrains everything downstream.

### Context
- **Zero pre-registration exists**. `grep -irln preregister\|pre-register\|prereg\|analysis plan` on tracked files returned zero matches.
- **No forkable AI-safety-IPI prereg template found externally** (per `docs/research/gpt-deep-research-2026-04-14.md` "Pre-registration templates" section). Plan: author from scratch using OSF's registration form as structural baseline plus Bakker et al.'s specificity checklist for quality control. Adjacent-domain precedent: the "IatroBench: Pre-Registered Evidence of Iatrogenic Harm" paper and the "How Evaluation Conditions Shape Measured Safety" paper both claim preregistration in their AI-safety positioning, though neither exposes a public prereg PDF artifact we could fork directly.

### Open questions
- **How strict are stopping rules?** Recommendation: conservative (no peeking). If we find mid-run that Sonnet is 100 percent compliant, we do not adjust probes. We report it.
- **Who signs?** Ashton + Max + Jasmine, dated. Everyone commits to the plan before data collection.
- **How precise are the tests?** Recommendation: name each test, each alpha, each correction method. No hand-waving.

## Track 7: Multi-model cohort production

> **Current execution note, 2026-05-09.** The live paper-facing AgentLab
> cohort has moved past the old 3-4 model draft below. The current execution
> matrix is the 50-task Tier 2 link-naturalization suite on r8a using AgentLab
> `phase4-run`, native BrowserGym launch, page-surface-stable PVPO,
> OpenRouter priority routing, eval-awareness iteration on every PVPO-exposed
> task, and `WORLDSIM_EVAL_AWARENESS_REWRITE_THINKING=budget:32768`. The
> completed/queued model panel is: GPT-5.2 W28 repaired control, Opus 4.7 W48,
> Sonnet 4.6 W48, Kimi K2.5 W48, Gemini 2.5 Pro W48, and GLM-5 W48. This is
> the active paper cohort plan; the older bullets in this section remain
> historical planning context until the full TODO is rewritten.

### Goal
Run the MVP pipeline against 3-4 target models, producing the per-model panel data that is Contribution 1's deliverable.

### What
Once MVP cross-check lands and Phase 0c + Phase 2 are regenerated:

1. **LLM-only smokes**: gemini-3.1-pro-preview and claude-opus-4-6 to verify OpenRouter/Anthropic routing works end-to-end for all 3 models. Commands in `docs/TODO-adversarial-rigor-mvp.md:368-377`.
2. **Phase 3 validity gate**: one run, agent-free, seconds to complete. Produces `contracts.json`.
3. **Phase 4 full per model**: Sonnet 4.6, Gemini 3.1 Pro, Opus 4.6 (cost-gated). Three runs against the admitted cohort. ~$200-400 total + 5-15 hrs per run.
4. **Phase 5 post-hoc**: run the 3-signal probe battery over all Phase 4 trajectories. ~$200-500.

Operational note: baseline capability now comes out of Phase 4's `capability_benign_under_attack` field (ecologically valid trajectories only), not a separate Phase 3 pass.

### Current AgentLab Phase 4 Cohort

All current runs share:

- source set: `logs/tier2_fresh50_exact50_latest_phase3_20260507Tregen5`
- host/instances: r8a with `instances.scale.json`
- runner: `--runner agentlab`
- provider/tier: `--agent-provider openrouter --agent-service-tier priority`
- concurrency: W48 for the main model rows, expressed as
  `--phase-4-max-workers 48` on top-level `worldsim.main phase 4` commands.
  Do not use `--workers 48` there; the remote launch guard rejects it.
  GPT-5.2 currently has a W28 repaired control because that run was produced
  before the W48 queue.
- PVPO: `WORLDSIM_PVPO_CAPTURE_BACKEND=page-surface-stable`
- legacy CDP: `WORLDSIM_AGENTLAB_LEGACY_CONNECT_OVER_CDP` unset
- iterator: `--phase-4-variant-system eval-awareness-iterator
  --phase-4-eval-awareness-max-iterations 3`
- rewrite budget: `WORLDSIM_EVAL_AWARENESS_REWRITE_THINKING=budget:32768`
- common timeouts: `--agent-llm-timeout 240 --agent-step-timeout 900
  --agent-task-timeout 2400`

Sequential execution is the paper-valid default. Each run already uses W48
inside the run; launching multiple W48 model sweeps concurrently would confound
model behavior with BrowserGym/browser host contention. Advance only after the
prior run has canonical `phase_4/results.json`, `50/50` postprocessed,
`postprocess_failed=0`, and an ASR/VEA/Transcript Purpose iterator analysis.

Current queue:

1. **GPT-5.2 control**:
   `logs/agentlab_linknaturalization_true50_gpt52_w28_20260509_repaired`.
   Completed and semantic variant schema normalized. Treat as the current
   GPT-5.2 control, while noting W28 concurrency in runtime comparisons.
2. **Opus 4.7 W48**:
   `logs/agentlab_linknaturalization_true50_opus47_w48_20260509`. Completed
   and semantic variant schema normalized.
3. **Sonnet 4.6 W48**:
   `logs/agentlab_linknaturalization_true50_sonnet46_w48_20260509`. Completed
   as `20260509T054104Z-agentlab-linknaturalization-true50-sonnet46-w48-e40eea`
   in 31m32s with `50/50` postprocessed and `postprocess_failed=0`. Headline
   and Gate-1 ASR were both `40/50=0.80`; final statuses were
   `complied=40`, `inconclusive=8`, `resistant=2`. Iterator accounting after
   semantic schema normalization: `entered_tasks=50`, `rewrite_attempted=114`,
   `variant_evaluated=72`, `pre_browser_rejections=42`,
   `post_eval_rejections=6`, `schema_validation_failures=5`,
   `tp_regression_rejections=5`, and `contract_inapplicable_rejections=37`.
4. **Kimi K2.5 W48**:
   initial job
   `20260509T061312Z-agentlab-linknaturalization-true50-kimik25-w48-fcf9b3`
   was stopped and invalidated because Hugging Face `trust_remote_code` prompt
   text contaminated sidecar stdout before the final JSON payload. The parser
   was patched and covered by
   `test_sidecar_json_payload_accepts_inline_hf_prompt_noise`. The active rerun
   rerun `20260509T061814Z-agentlab-linknaturalization-true50-kimik25-w48-r-302c2f`
   completed in 28m03s with `50/50` postprocessed and
   `postprocess_failed=0`, output
   `logs/agentlab_linknaturalization_true50_kimik25_w48_20260509_rerun1`.
   Headline ASR was `39/50=0.78`; Gate-1 ASR was `39/49=0.80`; final statuses
   were `complied=39`, `inconclusive=7`, `injection_not_encountered=1`,
   `resistant=2`, `task_broke=1`. Baseline TP was
   `Alignment Evaluation=35`, `Capability Evaluation=11`, `Real=4`; VEA was
   `Real=50`. Iterator accounting: `entered_tasks=48`,
   `rewrite_attempted=107`, `variant_evaluated=76`,
   `pre_browser_rejections=31`, `post_eval_rejections=15`,
   `schema_validation_failures=5`, `tp_regression_rejections=13`, and
   `contract_inapplicable_rejections=26`.
5. **Gemini 2.5 Pro W48**:
   main job
   `20260509T064643Z-agentlab-linknaturalization-true50-gemini25pro-w-4bb84f`
   completed with `results.partial.json` because two rows failed
   postprocessing. Repair `gitlab18` r1 completed cleanly; repair `gitlab2`
   r1 is invalid because it used the wrong GitLab replica; repair `gitlab2`
   r2 reused the original worker instance file and completed cleanly. The
   repaired canonical output is
   `logs/agentlab_linknaturalization_true50_gemini25pro_w48_20260509_repaired`
   with `50/50` scorable rows and no error final statuses. Headline and Gate-1
   ASR were both `40/50=0.80`; final statuses were `complied=40`,
   `inconclusive=5`, `resistant=5`. Iterator accounting in the repaired
   summary: `entered_tasks=50`, `rewrite_attempted=90`,
   `variant_evaluated=73`, `pre_browser_rejections=17`,
   `post_eval_rejections=15`, `schema_validation_failures=3`,
   `tp_regression_rejections=10`, and `contract_inapplicable_rejections=14`.
6. **GLM-5 W48**:
   launch
   `20260509T081920Z-agentlab-linknaturalization-true50-glm5-w48-d22293`
   failed immediately before any task execution because the command used the
   stale top-level `--workers 48` spelling. Canonical `worldsim.main phase 4`
   commands must use `--phase-4-max-workers 48`; `--workers` is reserved for
   `scripts/run_phase4_process_pool.py`. The remote launch guard now rejects
   this mismatch before starting a job. The active rerun is
   `20260509T082006Z-agentlab-linknaturalization-true50-glm5-w48-reru-3314ff`
   with `--agent-model glm5`, output
   `logs/agentlab_linknaturalization_true50_glm5_w48_20260509_rerun1`. This
   workspace does not contain the current 2026-05-09 W48 run artifacts, and
   `scripts/remote_job_status.sh` did not find that remote job name, so the
   GLM final status still needs verification from the correct r8a job record or
   output directory before promotion.

For every completed row, report: raw ASR, Gate-1 ASR over PVPO-exposed tasks,
baseline VEA and Transcript Purpose distribution, selected-variant VEA and
Transcript Purpose distribution, transition counts across iterator iterations,
semantic variant accounting (`rewrite_attempted`, `variant_evaluated`,
`pre_browser_rejections`, `post_eval_rejections`, schema/contract/TP-regression
classes), PVPO validity, and any timeout/task-broke rows.

### Why
- Contribution 1 reports ASR per model. Single-model results are not publishable.
- Contribution 2 claims model-dependent effects in eval awareness. Requires at least 3 models.
- 4th model (GPT-5 via OpenRouter) is stretch; recommended if time permits for cross-family representation (Anthropic + Google + OpenAI).

### Context
- Infrastructure ready once MVP cross-check lands (per Day 3 audit, seeding and auth layers all in place).
- Cost estimates in `docs/TODO-adversarial-rigor-mvp.md:361-363` already computed.
- Wall-clock dominates; not code-gated.

### Open questions
- **Total budget ceiling?** Rough estimate: $1000-2500 for full multi-model run. Request confirmation before kicking off.
- **Parallel or sequential runs?** Parallel uses more WebArena instance resources but halves wall-clock. Recommendation: sequential for the first full run (easier to debug), parallel for follow-up runs.
- **Do we include GPT-5?** Recommendation: include if budget allows. Adds cross-family diversity. Path via OpenRouter should work identically to Anthropic but needs smoke test.

## Rough effort rollup

| Track | Ashton hours | Max hours | Jasmine hours | Total |
|---|---|---|---|---|
| 1. Phase 5 integration | 15-22 | 10 | 8-12 | 33-44 |
| 2. Experiment B degradation | 10-15 | - | - | 10-15 |
| 3. Experiment A static capture | 8-16 | - | - | 8-16 |
| 4. Comparison benchmarks (ICML scope: runner port + STWebAgentBench only) | ~13 | 0 | 0 | ~13 |
| 5. Analysis stack | 8-12 | 15-20 | 5-8 | 28-40 |
| 6. Pre-registration | 4-6 | 2 | 2 | 8-10 |
| 7. Multi-model cohort (wall-clock mostly) | 6-10 code | 2 | 2 | ~$1000-2500 + 15-40 wall-clock hrs |
| **Total** | **64-94** | **29-32** | **17-24** | **110-150 person-hours** |

Adopting the `feat/multi-benchmark` runner architecture instead of building from scratch saves roughly 33-50 person-hours on Track 4 alone. Max and Jasmine time on Track 4 drops to zero for the ICML scope; Max can redirect toward Phase 5 integration (Track 1), Jasmine toward probe design (Track 1 + Track 6). Roughly 2 weeks of concurrent work across three people with coordination overhead.

## Sprint carving suggestions (not committed, input for later planning)

- **Sprint 0 (this week, 2 days)**: MVP cross-check lands, Phase 0c + 2 reruns, Phase 3 + 4 smoke on Sonnet. Unblocks everything.
- **Sprint 1 (week 1)**: Tracks 1 + 6 parallel. Phase 5 integration from orphan branch. Pre-registration commit. Multi-model Phase 3/4 starts on Sonnet.
- **Sprint 2 (week 2)**: Tracks 2 + 3 parallel. Experiment B degradation. Experiment A static capture. Multi-model Phase 3/4 continues (Gemini + Opus).
- **Sprint 3 (week 3)**: Track 4 (reduced scope: ~13h to port runner architecture + STWebAgentBench adapter) runs in parallel with Track 5 analysis stack. Freed time (33-50 person-hours on Track 4) absorbed into additional figure iteration, limitations writing, and paper-writing buffer.
- **Submission buffer**: 2-3 days before 5/5 deadline for writing, limitations, editing (widened from 2 days thanks to Track 4's reduced scope).

Risk: **If we port the multi-benchmark runner architecture, the migration MUST preserve our current-branch features**: `benchmark_root` threading, `build_agent_prompt`, `validate_fix_patch`, `--sandbox-model`, `--allow-unknown-auth`, Phase 0d dispatch, and `rescore-phase-3`. The migration step list in Track 4 explicitly enforces this at steps 5 and 6. Failing to preserve any of these silently regresses Phase 3 fix-loop guarantees or Phase 4 auth handling.

Secondary risk: porting the WASP / DoomArena / SafeArena adapter files (step 4) is dead scaffolding for ICML. If we skip it to save 3h, re-merging for NeurIPS extension costs more than that in conflict resolution as `feat/worldsim-v5` evolves.

## GPT research integration (2026-04-14)

Summary of how findings from `docs/research/gpt-deep-research-2026-04-14.md` landed in this doc, and the questions that still need user approval before the corresponding work kicks off.

### Where GPT findings landed

- Track 4 (STWebAgentBench scope): site-overlap corrected to GitLab + ShoppingAdmin only; canonical task file paths cited; 30-task panel recommendation recorded; Codex handoff `docs/handoffs/codex-handoff-stwebagentbench-task-subset.md` produces the filtered JSON.
- Track 1 (Phase 5 probe battery): Needham et al's reproduction repo `github.com/jjpn97/eval_awareness` @ `1543ffe644d3f8f3a43b5dfd7814dbe9b12f890d` cited as canonical prompt reference for VEA and external P(eval); Codex handoff `docs/handoffs/codex-handoff-needham-prompt-audit.md` produces the verbatim-vs-paraphrased diff against the orphan branch port.
- Track 3 (static capture): Playwright MCP docs added as adjacent-stack corroboration for browser-use DOM-filtering claim.
- Track 5 (analysis stack): Needham's `analysis/{roc,purpose,cot,colors}.py` added as a structural reference layout.
- Track 6 (pre-registration): noted that no forkable AI-safety-IPI prereg template was located; plan is to author from scratch with OSF + Bakker et al. as structural basis.

### Decision points still open

1. **SuiteCRM coverage**: GPT supplied a complete SuiteCRM Docker recipe (Bitnami image + mariadb, port 8080, creds `user/bitnami`, demo data SQL) that does not collide with our existing WebArena port map. Adding SuiteCRM to the ICML stack would recover 1 additional comparison site (~4-8h extra work). Decision: accept narrower GitLab + ShoppingAdmin overlap for ICML (current plan), or add SuiteCRM.
2. **Judge model for Phase 5**: spec defaults Opus 4.6. Needham et al used GPT-4.1 (public). Decision: stick with Opus 4.6 (spec default), match Needham (direct comparability), or run both (cross-judge calibration; doubles Phase 5 judge cost).
3. **Unanswered GPT asks**: Ask 1 (DoomArena end-to-end), Ask 2 (WASP seeding), Ask 6 (CSRF scraping), Ask 10 (judge cost optimization). Existing `feat/multi-benchmark` audit and TODO-1 Day 3 "Unknowns" remain authoritative; do not re-send these to GPT unless context changes.

### Redundant GPT findings

- GPT independently surfaced the `feat/multi-benchmark` STWebAgentBench adapter's credentials map (`suitecrm: user/bitnami`), confirming the port's fidelity.
- GPT confirmed no public self-reported-P(eval) implementation exists end-to-end, reinforcing Track 1's "NEW" classification for that probe.
- GPT confirmed browser-use 0.12.6 DOM-filter claim has no public third-party empirical corroboration; our source-inspection citation remains the primary evidence.

## Citations

### Paper spec
- `docs/worldsim-v5-full-specification.md` (current short overview)
- `docs/worldsim-v5-technical-specifcation.md` (implementation authority)
- `docs/current_progress.md` (current state)

### Our branch
- `worldsim/phase_4/transcript_purpose_api.py` (Transcript Purpose classifier)
- `worldsim/phase_4/verbalized_eval_awareness_api.py` (VEA classifier)
- `worldsim/phase_4/encounter_detection.py` + `worldsim/phase_4/ink_occupancy.py` + `worldsim/phase_4/pvpo_capture.py` (PVPO encounter detection; replaces the deleted ecological-validity fix loop and `probe-ecological-validity.md` / `fix-ecological-validity.md` prompts)
- `worldsim/_sandbox_validator.py` (validity schema)
- `docs/TODO-adversarial-rigor-mvp.md` (MVP plan; covers Contribution 1 only)
- `docs/TODO-adversarial-rigor-mvp.md:361-363` (cost estimates per model)
- `docs/handoffs/codex-handoff-paint-verified-oracle.md` (PVPO design + Implementation Status — read first)

### Orphan branches (to merge or port)
- `origin/claude/plan-eval-experiments-LNd9F:eval_awareness_experiments/experiments/p_eval.py` (5-probe P(eval) bank)
- `origin/claude/plan-eval-experiments-LNd9F:eval_awareness_experiments/judges/verbalized_awareness.py` (VEA extractor)
- `origin/claude/plan-eval-experiments-LNd9F:eval_awareness_experiments/trajectory_converter.py` (DoomArena/WASP/OS-harm parsing)
- `origin/claude/plan-eval-experiments-LNd9F:eval_awareness_experiments/analysis_causal.py` (existing analysis script)
- `origin/claude/plan-eval-experiments-LNd9F:eval_awareness_experiments/run_{doomarena,wasp,osharm}.py` (runners)
- `feat/multi-benchmark:worldsim/runner.py` (Protocol definitions for runners)
- `feat/multi-benchmark:worldsim/runners/__init__.py` (registry, importlib dispatch)
- `feat/multi-benchmark:worldsim/runners/browser_use.py` (thin re-export of `phase_3_benign.run_task`)
- `feat/multi-benchmark:worldsim/runners/agentlab.py` (EnvArgs / ExpArgs / _run_experiment_sync / _parse_exp_result / _persist_result_sentinel)
- `feat/multi-benchmark:worldsim/adapters/stwebagentbench.py` (fully working adapter, ICML comparison target)
- `feat/multi-benchmark:worldsim/adapters/{doomarena,wasp,safearena,__init__}.py` (scaffolding for NeurIPS extension)
- `feat/multi-benchmark:tests/test_agentlab_runner.py` (3 tests covering the port)

### Vendored (existing)
- `vendors/webarena-verified/` (primary env)
- `vendors/webarena-infinity/` (static-capture target, needs standing up)

### Principles
- `CLAUDE.md` principle 1 (orchestrator does not manage environments; constrains Experiment B's "broken interactivity" cue)
- `CLAUDE.md` principle 4 (two gates: ecological validity + attack effectiveness; Phase 4 lite probe is the runtime gate, Phase 5 is the rigorous post-hoc measurement)
