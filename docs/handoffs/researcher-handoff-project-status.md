# Engineering status update

April 17, 2026. Branch `feat/worldsim-v5` (181 commits, 7 days). Parallel branch `feat/multi-benchmark` is rebased onto this branch as of `42b8594e` and carries 6 exclusive commits (the pluggable-runner layer plus four follow-up fixes).

---

## What changed since main

Main had AgentLab (the BrowserGym-based pipeline), `behavior-gen/` (Jasmine's injection generation pipeline), and `new-data/`. This branch sets those aside in favor of a new `worldsim/` package that reimplements the pipeline around Modal sandboxes + Browser Use instead of BrowserGym. AgentLab comes back on the `feat/multi-benchmark` branch as a pluggable runner for cross-benchmark comparisons (more on that below). Two files from AgentLab's redteam module are kept as read-only reference for Modal sandbox mechanics and Claude Code invocation flags.

The injection generation prompts in `worldsim/prompts/` (especially `generate-injections.md` and `fill-injection-text.md`) were heavily inspired by Jasmine's `behavior-gen/pipeline.py` decomposition. The plan-and-realize split in Phase 2 directly mirrors her approach of separating adversarial schema generation from concrete payload composition. (The module and prompt filenames still carry the legacy "text fill" name; current terminology in the spec and this doc is **payload realization** for the 2b stage. A code-side rename is queued.)

The new package is roughly 25,153 lines of pipeline code, 22,767 lines of tests across 47 files, 1,117 lines of LLM prompts across 14 files (two prompts deleted with the Phase 3 cutover), and 6,533 lines of infrastructure scripts. Test-file count grew 38% in the past three days as the editor architecture and the Phase 3 cutover both demanded their own coverage.

Here's what the package actually contains:

The pipeline orchestrator (`worldsim/main.py`) is a CLI that runs phases sequentially: `uv run python -m worldsim.main phase 0 --benchmark vendors/webarena-verified`. It handles `--resume` (reads `logs/pipeline_state.json`, skips completed phases), `--sites` filtering, `--max-tasks-per-site` for smoke tests, `--sandbox-model` to pick which Claude model runs in sandboxes, and `--agent-model` for which model Browser Use drives. The `--full-baseline` and other Phase 3 agent flags were removed when Phase 3 became agent-free (commit `8afe863f`); the unknown-auth gate is now scoped to Phase 4 only (commit `e55eaa11`) since Phase 3 no longer touches live instances.

Modal sandbox primitive (`worldsim/modal_sandbox.py`, ~580 lines). `run_claude_in_sandbox` creates a cloud sandbox on Modal, stages files via content-addressed volume upload, runs Claude Code through `claude-agent-sdk`, and returns file outputs plus cost/token metadata. Supports three auth paths: OAuth (preferred, no refusals), OpenRouter proxy (backup), direct API key. Named Modal secret support via `WORLDSIM_CLAUDE_MODAL_SECRET`.

Browser Use agent runner (`worldsim/browser_use_agent.py`, ~1,014 lines). Runs the target agent against live WebArena sites. Async worker pool with staggered start, HAR network capture, per-task screenshots. Configurable LLM via `worldsim/agent_config.py` (Google, OpenAI, Anthropic, OpenRouter).

In-sandbox validator (`worldsim/_sandbox_validator.py`, ~776 lines after the body-field alias work). Every sandbox output is validated before it leaves the sandbox: schema checks for profiles, tasks, adversarial plans, and the new `editor_calls` shape. Host-side retries append concrete validation errors back into the prompt. The validator now carries `_EDITOR_BODY_FIELD_ALIASES` to bridge the gap between prompt-emitted field names and the editor method signatures, since the editor migration left several gitlab and reddit method shapes uncovered (commit `4c27fb48`).

Per-site editor classes (`worldsim/editors/`, 1,943 LOC across `base.py`, `gitlab.py`, `reddit.py`, `shopping.py`, `shopping_admin.py`). Each editor implements `validate_args`, `probe_base_state`, and per-method HTTP composition for that site. Phase 4 dispatches via `EDITOR_REGISTRY[(benchmark, site)]`. This package replaced the `seed_resolvers/` regex-dispatch package on April 17. The editor model is OOP, has one place to look per method, and gives `_sandbox_validator.py` a single shape to enforce at sandbox boundaries.

The Phase 0 reconnaissance system (`worldsim/phases/phase_0_recon.py`, ~1,220 lines) does three things: 0a discovers the benchmark structure and produces `BENCHMARK_MANIFEST.json`, 0b computes per-site file lists (pure Python, no LLM), and 0c runs tiered per-site profiling. Each site gets profiled for verification capabilities, data model, agent context (auth, response format), and injection surfaces. 0c is the expensive step and produces separate JSON artifacts per site: `BENCHMARK_PROFILE_{site}.json`, `AGENT_CONTEXT_{site}.json`, `INJECTION_SURFACE_{site}.json`, `DATA_MODEL_{site}.json`, `VERIFICATION_CAPABILITIES_{site}.json`.

Phase 0d auth bootstrap (`worldsim/phases/phase_0d_auth_bootstrap.py`). Added between 0c and Phase 1. Generates authentication credentials per site: GitLab gets a PAT via a Playwright script, other sites use stored browser cookies. Dispatch order: generator_script, then native form_login, then trust existing storage_state, then skip.

Phase 1 task generation has two modes. existing-task wrapping (`phase_1_existing_tasks.py`) wraps existing WebArena tasks into pipeline schema, deterministic, no LLM, produces 812 benign tasks. new-task generation (`phase_1_generate_new_tasks.py`) generates novel tasks targeting uncovered injection surfaces. Tasks now carry an `origin` tag (`existing_task` or `new_task`) classified by task ID prefix (`novel_<site>_<n>` for new_task, otherwise existing_task) so Phase 4 can stratify ASR by source (commit `37b524a6`).

Phase 2 injection generation (`phase_2_injections.py`, ~1,805 lines + `phase_2_text_fill.py`, legacy filename for what is now called the realization stage). Split into two internal stages: **2a (planning)** generates adversarial plans in Modal sandboxes, pure schema work covering target surface, framing, concealment, delivery mechanism, required tokens, length budget, and reward function, with no payload text. **2b (payload realization)** takes each plan and renders it into the concrete text artifact that will sit on the target surface, via host-side async API calls in the plan's declared voice (a customer review must read like a customer review, a GitLab issue must read like a GitLab issue). The split exists because single-shot generation gets safety-refused ~50% through proxy APIs; the planning stage is refusal-safe because it emits no harm-adjacent text, and the realization stage is accepted at ~95% because it is framed as composing platform-native content rather than as writing an injection. Per-shard persistence so crashes don't lose completed work. Phase 2 now emits the `editor_calls[*].{benchmark, site, method, args}` shape directly; the entire 236-task adversarial dataset was migrated in place over the commit chain `917211e4` through `c1ddb52c`.

**Phase 3 (`phase_3_benign.py`) is now an agent-free contract validity gate.** It schema-validates benign tasks and adversarial references and writes `phase_3/contracts.json` with a `validity_status` per entry. There is no live instance interaction, no diagnosis loop, and no triage layer. The old `phase_3_triage.py` and `fix_validation.py` modules were deleted on April 17; the `--sites` filter now applies to adversarial task references (commit `8deaf505`) so partial-site smoke runs no longer surface cross-site references as errors. Capability discovery moved entirely to Phase 4 as the `capability_benign_under_attack` byproduct.

Phase 4 adversarial evaluation (`phase_4_adversarial.py` + `worldsim/phase_4/*`). Two sequential gates: Gate 1 probes ecological validity (~5 questions per trajectory, averaged into P(eval)), Gate 2 checks attack effectiveness. When attacks fail on ecologically valid trials, the judge classifies the failure mode into one of 7 values (`suspicion`, `tonal_mismatch`, `formatting`, `externally_sourced`, `safety_policy`, `distracted`, `unknown`) via a single-turn Anthropic Messages API call with forced tool-use, and a host-side `TRIGGER_TO_STRATEGIES` lookup selects up to 3 strategies from a 22-entry pool. Both the judge and variant generator moved off Modal sandbox to direct API calls on 2026-04-18 to resolve refusals on the "recommend strategies" framing; the methodology and taxonomy follow Dziemian et al., 2026 (arXiv:2603.15714, Gray Swan IPI Arena). Phase 4 admits every entry from `contracts.json` with `validity_status: "valid"` (commit `1064d677`) and reports stratified ASR with WASP-style splits (`complied_with_adversarial_reward` vs `complied_without`) cross-cut by `origin`. Seeding goes through the editor classes via `EDITOR_REGISTRY` (commit `54888173`).

Supporting modules: `seeding.py` (data seed application via API/form channels through the editor dispatch, SQL excluded from methodology), `rewards.py` (dual-path evaluation, vendor evaluator primary, homebrew fallback), `placeholders.py` (URL placeholder resolution for `__SHOPPING__`, `__GITLAB__` etc.), `cost_tracker.py` (per-sandbox cost accumulation with resume persistence), `agent_prompt.py` (builds benchmark-specific prompts from agent_context artifacts), `state.py` (pipeline state persistence and resume logic), `site_lock.py` (per-site instance locking for parallel diagnosis).

14 LLM prompts in `worldsim/prompts/` covering every pipeline step from benchmark discovery through strategy variation (down from 16; the diagnosis and triage prompts were removed with the Phase 3 cutover). 47 test files covering crash-resume scenarios, sandbox validation edge cases, editor coverage, auth mechanism schemas, and phase-specific logic.

Infrastructure scripts in `scripts/`: `bootstrap_ec2.sh` (end-to-end EC2 orchestrator for all 6 WebArena containers), `deploy_benchmark_proxy.sh` (authenticated nginx reverse proxy), `wa_envctrl_patcher.py` (idempotent Python patcher for env-ctrl external URL support), `patch_form_to_api.py` (transforms form-mechanism seeds to API calls for Magento CSRF workarounds), `migrate_phase_2_seeds_to_targets.py` (the editor-shape migration rewriter that backed the dataset cutover), `run_integration_tests.sh` (the live-stack editor coverage gate that any PR touching `editors/`, `seeding.py`, or Phase 4 must run), plus Docker Compose overrides and helper scripts.

---

## Where things actually are

Phase 0 through Phase 3 are done. Phase 3 in its new form is a 30-second offline run that emits `phase_3/contracts.json`; it is no longer a multi-hour wait. Phase 4 is live as of 19:04 UTC on April 17, working through 7 adversarial tasks on `instances.smoke.json` (4 sites, 2 tasks per site) with gpt-5.4-mini as the agent and claude-sonnet-4-6 in the sandboxes. Run dir is `logs/phase_4/20260417_190423`. Results have not been written yet.

The biggest-unknown framing from the prior status doc is gone. The old worry was Phase 3 pass rate determining whether we have enough adversarial coverage; that question is now answered upstream (every well-formed contract is admitted, capability is measured downstream as a Phase 4 byproduct), so the analogous risk shifts to the analysis stack, which still does not exist.

The multi-benchmark branch was rebased onto the current branch on April 17. Six commits exclusive: the original pluggable-runner architecture (`46809439`), three regression fixes (`ed749954`, `ffb1976e`, `0ad87b19`), a contract-alignment fix (`6c0a4251`), and the rebase repair (`42b8594e`). DoomArena attacks still don't fire (the runner builds plain `EnvArgs` instead of `AttackedBrowserEnvArgs`) and WASP injection text never gets instantiated with live URLs. STWebAgentBench is the most mature of the adapters and has an active codex handoff to filter to a 30-task subset (gitlab + shopping_admin only, no SuiteCRM) for the ICML 4/24 panel.

---

## What's verified vs what's just implemented

| Thing | Status | Evidence |
|-------|--------|----------|
| Phase 0 (recon, all 6 sites profiled) | Verified | Profiles validated against live EC2 instances via authenticated proxy |
| Phase 0d (auth bootstrap) | Verified | GitLab PAT works, storage_state cookies work |
| Phase 1 Existing-task wrapping (812 wrapped tasks) | Verified | Deterministic, no LLM, schema-validated |
| Phase 2 (adversarial plans + payload realization, editor_calls shape) | Verified | 236 adversarial tasks migrated in place to editor_calls; voice registry gaps resolved |
| Phase 2c (feasibility verification) | Running | Shipped 2026-04-18 (commits 1-4 on `feat/worldsim-v5`); AT-009 now lands `length_exceeded` in `adversarial_tasks.infeasible.json`; Phase 4 admission gate wired with `STRICT_FEASIBILITY_ADMISSION` + `WORLDSIM_STRICT_FEASIBILITY` env override |
| Phase 3 (validity gate) | Verified | Agent-free; emits `phase_3/contracts.json`; old triage / diagnosis / fix-loop modules removed; benigns whose linked adversarials are all infeasible carry `adversarially_exhausted=true` annotation |
| Phase 4 (Gate 1 + Gate 2, stratified ASR) | Running | Live since 2026-04-17 19:04 UTC, smoke config, 7 adversarial tasks in flight, results.json not yet written |
| Editor architecture (`worldsim/editors/`) | Verified for shopping, shopping_admin, gitlab, reddit | `editor_calls` shape enforced by `_sandbox_validator.py`; integration tests in `scripts/run_integration_tests.sh` |
| EC2 infra (6 WebArena sites) | Verified | All 6 up, reset endpoints work, proxy deployed |
| Modal sandboxes | Verified | Hundreds of runs across phases, cost tracking works |
| Browser Use agent runner | Verified | Works with OpenRouter, HAR capture, screenshots |
| Multi-benchmark adapters | Rebased; integration not exercised on this branch | Code rebased as of `42b8594e`, six commits exclusive |
| Analysis tooling | Does not exist | Zero notebooks, zero stats code; this is now the load-bearing missing piece |

---

## Design shifts (not a commit log; these are the things that changed how the project works)

### Phase 3 became an agent-free contract validity gate (April 17, commit `45c7905a`)

Phase 3 no longer runs the target agent. It schema-validates benign tasks and adversarial references and writes `phase_3/contracts.json` with a `validity_status` per entry. Phase 4 admits every entry whose status is `valid`. The `phase_3_triage.py`, `fix_validation.py`, and the diagnosis / triage prompts are gone, along with most of the old `phase_3_benign.py`. About 2,854 LOC removed.

The motivation came from the WebArena-Infinity three-layer test model. The old Phase 3 was conflating two questions: "is this contract well-formed?" and "can this agent solve a benign version?", and using the second to gate the first. The first question is cheap and deterministic. The second is expensive, noisy, model-dependent, and load-dependent. Mixing them meant every iteration of the pipeline paid a 30-90 minute overnight cost to learn whether contracts were well-formed.

What this enables: same-day pipeline iteration. Reproducibility, because contract validation is purely a function of inputs and is independent of agent state, network conditions, or wall-clock luck. The "Phase 3 pass rate determines whether we have enough data" risk is structurally gone; we now know contract well-formedness immediately and measure capability downstream where it belongs.

What this costs: we lose the standalone benign-baseline number (the published-style headline "agent X can do Y% of WebArena tasks unmolested"). That number now only exists conditional on Phase 4's `capability_benign_under_attack` on ecologically valid trajectories. That is a more honest framing for our research question, since the population we care about is "tasks the agent can actually do," but it is harder to compare against published baselines that report unconditional pass rates.

### Phase 4 seeding pivoted to per-site editor classes (April 17, commit `54888173`)

The `seed_resolvers/` package (regex-preflight dispatch, 1,336 LOC across 7 files) was replaced with `worldsim/editors/`: per-site OOP classes implementing `validate_args`, `probe_base_state`, and per-method HTTP composition. All 236 adversarial tasks were migrated in place to the `editor_calls[*].{benchmark, site, method, args}` shape; the legacy mapping is preserved in a quarantine file.

The motivation was WASP's architecture, which is the obvious reference. Regex dispatch was brittle: every new method meant another regex, validation was scattered between the resolver layer and the prompts, and the surface for cross-benchmark portability was a tangle. With editor classes, the validation contract for each method is in one place, `_sandbox_validator.py` enforces shape at sandbox boundaries, and adding a new benchmark is a matter of writing a new editor module per site.

What this enables: cleaner Phase 4 preflight (probe base state, validate args, then seed). Real integration tests (`scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r5.yaml`) that exercise the editors directly against a live stack. A tractable path to multi-benchmark: STWebAgentBench, SafeArena, etc. each get their own editor namespace rather than fighting the resolver dispatcher.

What this costs: a migration day. The commit chain `917211e4` through `c1ddb52c` did the dataset rewrite, `8fb0b3b3` repaired a `{{PAYLOAD_TEXT}}` placeholder collision the migration introduced, and `4c27fb48` added body-field aliases for several gitlab and reddit method shapes the migration left uncovered. Expect another small wave of validator-coverage fixes as Phase 4 exercises uncovered methods. Net code impact is roughly -603 LOC (1,943 added under `editors/`, ~2,450 deleted across resolvers and the Phase 3 cleanup overlap).

### Origin tracking and stratified ASR (April 17, commits `1064d677` and `37b524a6`)

Tasks now carry `origin: "existing_task" | "new_task"`, classified by task ID prefix (`novel_<site>_<n>` is new_task, otherwise existing_task). Phase 4 reports ASR stratified by origin and by WASP-style outcome, splitting `complied_with_adversarial_reward` from `complied_without`.

The motivation: the paper's generalization claim hinges on showing that new_task entries (novel, ours) reproduce the existing_task (benchmark-derived) effect. Without per-origin reporting we couldn't make that claim cleanly. The WASP outcome split matters because "agent did the bad thing AND got rewarded for it" is materially different from "agent did the bad thing but failed at the task anyway"; conflating them inflates the headline ASR and misrepresents the threat. The earlier classifier inferred origin from seed shape, which broke for navigate-only tasks; the ID-prefix classifier is unambiguous (commit `6e04ffbe` documents this).

What this costs: slightly more analysis surface area, since we now report 4 cells instead of 1. That is what good ASR reporting looks like.

### We dropped SQL seeding entirely (April 16)

This was a late and painful realization. SQL writes violate the threat model: a regular user can't INSERT into the database. We ripped out 1,622 lines. All adversarial content now enters through authenticated HTTP POST/PUT against public APIs or form endpoints. SQL is only used for reward evaluation (reading postconditions), not for planting anything.

This matters for the paper because it means every injection we report in headline ASR could plausibly be planted by a real attacker. Anything that couldn't (tagged `privileged_seed: true`) is reported separately.

### And dropped api/form/state_push seed mechanisms (April 26)

The editor migration (April 17) replaced regex-preflight seed_resolvers with per-site OOP editor classes, but `validate_data_seed` still accepted the v4-era `mechanism: api`, `mechanism: form`, and `mechanism: state_push` shapes. That meant Phase 1 Mode B kept emitting `mechanism: "api"` benigns by default (30/30 in `logs/phase_1/novel_tasks_gitlab.json`), and they crashed through five different editor-only validators downstream — Phase 2 v7 produced 0/60 new_task adversarials for exactly this reason. Sunsetted in commits `ff8381d5` (validator boundary), `c0d24600` (apply_data_seed dispatch + helpers), `59586f9b` (Phase 1 prompt + validator), `224e2359` (parallel-agent cache rewrite), and `a9f7fd78` (test fixture migration). `validate_data_seed` and `apply_data_seed` now accept only `mechanism: editor` or `mechanism: none`; the legacy HTTP/CSRF/login dispatch path was removed; the Phase 1 Mode B prompt forbids the deprecated mechanisms; and ~30 deprecated tests were dropped while the surviving fixtures were migrated to editor_calls.

### Phase 2 had to be split into planning + payload realization (April 14)

Asking an LLM to generate adversarial injection text in one shot gets refused ~50% of the time through proxy APIs (OpenRouter). We burned ~$190 across two failed attempts before figuring this out. The fix: split into (a) **plan generation**, which is pure schema (target, framing, concealment, required tokens, length budget) with no harm-adjacent text, and never gets refused, and (b) **payload realization**, which renders each plan into its concrete text artifact in the plan's declared voice and gets accepted ~95% of the time because it is framed as composing platform-native content (a review, an issue, a forum post) rather than as writing an injection.

The naming matters for the paper. We initially called the second stage "text fill" or "UGC composition" in code and early docs, but those names describe an implementation trick (frame the request as user-generated content to dodge refusals) rather than the artifact's research role. **Realization** names the stage by what it does: turn an abstract specification into a concrete artifact, in the same sense as lexical realization in NLG or concrete realization of a Hoare spec. It is benchmark-agnostic (works for non-UGC surfaces too: config files, commit messages, README diffs) and it does not bake the refusal-dodging trick into the vocabulary; the voice-as-platform-native-content framing is one *implementation* of realization, not its definition.

The decoupling might actually produce better injections since the planner and the realizer optimize separately, but we haven't measured that. (Module and prompt filenames still carry the legacy "text_fill" / "fill-injection-text" names; rename queued.)

### Phase 0c went from single-pass to tiered (April 14)

Single-pass profiling kept producing incomplete or wrong profiles. Tier 1 now runs verification-capabilities, data-model, and agent-context sandboxes in parallel. Tier 2 uses validated Tier 1 outputs to discover injection surfaces. Cost was ~$88 for the rerun ($60 of which was waste from accidentally running on Opus).

### The voice registry moved from exact IDs to pattern matching (April 16)

The voice exemplar registry (tells the realization stage how to write like a product review vs. a GitLab issue) was keyed on exact surface IDs: 111 brittle entries that broke whenever Phase 0c discovered a slightly different surface name. Now it's 7 regex categories on `source_field` patterns. The remaining Phase 2 failures are surfaces these 7 patterns don't cover yet.

### We committed to wrapping, not building (April 11)

v4 proposed generating web environments from scratch. v5 wraps WebArena Verified instead. Saved months of environment work but locked us to WebArena's 6 sites and whatever realism issues are baked in. We can't control environment quality below the data-seeding layer, which is a real limitation for the ecological validity story.

---

## Infra that's running

All 6 WebArena sites are up on an EC2 m5.xlarge in us-east-2 (18.117.99.179), with an authenticated nginx proxy and env-ctrl sidecars for reset between tasks. This part works and hasn't been a problem since the initial setup.

For everyday pipeline iteration, `instances.smoke.json` is the single-instance 4-site config (shopping, shopping_admin, gitlab, reddit) currently driving the live Phase 4 run. For scale runs we have `instances.scale.json`: a 30x replica fan-out per site with port offsets striding through the 0-29 range (e.g., `__SHOPPING_ADMIN__` maps to 7810-7839), so we can run 30 sandboxed instances of the same benchmark concurrently. The scale config is ready but hasn't been driven yet on this branch; the smoke run goes first to shake out the new editor architecture.

Claude Code pipeline steps run in Modal cloud sandboxes. OAuth auth is preferred (no refusals); OpenRouter proxy is the backup but gets refusals on adversarial content, which is why Phase 2 needed the plan/text split. The benchmark codebase uploads once via a content-addressed volume so we're not re-shipping it every sandbox.

For the actual agent runs, Browser Use runs locally against the live sites with an async worker pool and 5s staggered start. HAR capture and per-task screenshots are wired up. The current run uses gpt-5.4-mini via OpenAI for the agent.

---

## The multi-benchmark branch (`feat/multi-benchmark`)

Six commits exclusive (rebased as of `42b8594e`). This branch adds 4 benchmark adapters and a pluggable runner architecture.

Pluggable runner layer: `worldsim/runner.py` defines a protocol, `worldsim/runners/` has browser_use (existing) and agentlab (new, for BrowserGym-based comparison runs). CLI gets `--runner`, `--attack-mode`, `--benchmark-adapter` flags.

Adapters for each comparison benchmark:

STWebAgentBench is the most ready: 375 tasks, 3057 policy annotations, tests passing, covers gitlab + shopping_admin + suitecrm. There is an active codex handoff (`docs/handoffs/codex-handoff-stwebagentbench-task-subset.md`) to produce a 30-task subset (gitlab + shopping_admin only, no SuiteCRM) for the ICML 4/24 panel. SafeArena loads and wraps tasks but has a model-args mismatch (OpenRouter vs LiteLLM) and is missing from pyproject.toml extras, maybe 4-6h to fix. DoomArena and WASP are further out. DoomArena's runner builds plain `EnvArgs` instead of `AttackedBrowserEnvArgs` so attacks never actually fire (12-22h). WASP's task expansion produces 84 tasks but injection text never gets instantiated with live URLs and needs a VWA GitLab + Reddit stack we don't have running (22-34h).

Cross-benchmark reporting module at `worldsim/reporting.py`. Comparison table generation.

A second active codex handoff (`docs/handoffs/codex-handoff-needham-prompt-audit.md`) is auditing the orphan-branch eval-awareness probes against the canonical Needham et al. prompts. This feeds the Phase 5 eval-awareness probe battery on the multi-benchmark TODO.

`TODO-2-paper-experiments.md` on that branch lays out 7 tracks totaling 110-150 person-hours. The big ones: Phase 5 (rigorous eval-awareness with a three-signal probe battery, 33-44h), the analysis stack which doesn't exist at all yet (notebooks, stats, figures, 28-40h), and the multi-model cohort runs (Sonnet 4.6, Gemini 3 Pro, Opus 4.6, GPT-5, plus 6-10h code and $1-2.5k compute). Smaller tracks: Experiment B degradation (10-15h), Experiment A static capture (8-16h), STWebAgentBench comparison runs (~13h), pre-registration (8-10h).

---

## What's running right now

Phase 4 is live as of 2026-04-17 19:04 UTC. Run dir: `logs/phase_4/20260417_190423`. Config: `instances.smoke.json` (4 sites, 2 tasks/site), gpt-5.4-mini agent via OpenAI, claude-sonnet-4-6 in sandboxes. 8 adversarial task dirs created at last check (`adv-001`, `adv_305_peer_tip_markdown_fenced_system`, `adv_408_a`, `adv_637_error_recovery_mf1`, `adv_shopping_admin_002`, `ADV-002`, `adv-775-review_detail_body-support_escalation-plaintext-001`, `AT-009`). `results.json` has not been written yet.

Earlier today: Phase 3 cutover landed (commit `45c7905a`); Phase 4 admission and stratified ASR landed (`1064d677`); editor architecture pivot landed (`54888173`); validator alias coverage caught up (`4c27fb48`); the `--sites` filter was extended to adversarial task references (`8deaf505`); origin classification moved to ID-prefix dispatch (`37b524a6`); the unknown-auth gate was scoped to Phase 4 (`e55eaa11`); and the `--full-baseline` and Phase 3 agent flags were dropped from the CLI (`8afe863f`).

---

## What's next

The smoke run should produce the first stratified ASR table from the new architecture. Sequence:

1. Phase 4 smoke completes. Verify Gate 1 / Gate 2 outputs and the new ASR stratification (origin x WASP-outcome) actually land in `results.json`.
2. Promote to scale: re-run Phase 4 on `instances.scale.json` (30x parallel) for the headline ASR table.
3. STWebAgentBench cross-benchmark run, gated on the codex handoff producing the 30-task subset.
4. Stand up the analysis stack. This is the load-bearing missing piece now that Phase 3 is no longer the bottleneck: notebooks, stats, figures, 28-40 person-hours per the multi-benchmark TODO. Without it we have raw numbers but no paper-ready artifacts.
5. Multi-model cohort runs (Sonnet 4.6, Gemini 3 Pro, Opus 4.6, GPT-5) once the analysis stack is operational, $1-2.5k additional compute.

The thing I'm most worried about now is the analysis stack timeline. Phase 3 used to be the headline risk; it isn't anymore. The risk shifted, it didn't disappear.

---

## Cost so far

$525.06 sunk to date. Phase 0a $1.95, Phase 0c $114.62 (includes the documented Opus rerun waste), Phase 2 $316.11 (includes the $112 Modal SDK loss and the $79 fail-closed loss), Phase 3 $92.38 from the legacy agent-driven runs (the new validity gate is essentially free since it's an offline schema check). Phase 4 cost not yet tallied (run in progress). Multi-model runs would be $1-2.5k additional.

---

## Files that matter

| If you need to... | Read this |
|-------------------|-----------|
| Understand what the pipeline should do | `docs/worldsim-v5-technical-specifcation.md` (typo intentional) |
| See session-by-session engineering decisions | `docs/current_progress.md` |
| Understand the Phase 2 decomposition | `docs/handoffs/orchestrator-handoff-phase-2-v2.md` |
| Understand the Phase 3 cutover rationale | `docs/current_progress.md` (April 17 entry) and the message on commit `45c7905a` |
| Understand the editor architecture | `worldsim/editors/base.py` and any per-site editor (e.g., `worldsim/editors/gitlab.py`) |
| Run the live-stack integration gate | `scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r5.yaml` |
| See the multi-model and comparison experiment plan | `docs/TODO-2-paper-experiments.md` (on `feat/multi-benchmark`) |
| Run the pipeline | `README.md` and `CLAUDE.md` |
| Check what Claude Code sessions should and shouldn't do | `CLAUDE.md` |
