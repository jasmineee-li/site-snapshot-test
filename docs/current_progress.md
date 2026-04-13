# WorldSim v5 -- Current Progress

Last updated: 2026-04-13

## Branch

`feat/worldsim-v5`, 69 commits ahead of `main`. Latest committed change: `6a5b3cc fix: Chrome --no-sandbox flag, startup retries, temp profile cleanup`.

The current working tree also contains uncommitted but implemented changes for:

- tiered Phase 0c profiling
- benchmark-specific `agent_context` discovery and propagation
- benchmark-specific runtime prompt construction in Phases 3-4
- restoring Claude Code semantics in the Modal sandbox runner

## Current Live Run State

The first run has been archived to:

`logs_run1_no_tiered_site_discovery_and_missing_benchmark_specific_task_template/`

Archive operation:

```bash
mv logs logs_run1_no_tiered_site_discovery_and_missing_benchmark_specific_task_template
mkdir -p logs
cp -r \
  logs_run1_no_tiered_site_discovery_and_missing_benchmark_specific_task_template/phase_0a \
  logs_run1_no_tiered_site_discovery_and_missing_benchmark_specific_task_template/phase_0b \
  logs/
```

After the archive, `logs/` intentionally contains only:

- `phase_0a/`
- `phase_0b/`
- `last_run_state.json`

Why this reset happened:

- The archived run predates the current tiered Phase 0c implementation.
- The archived run also predates the benchmark-specific task-template / `agent_context` fix, so later-phase outputs are no longer canonical for the current code.
- `phase_0a` and `phase_0b` are still valid, unchanged inputs to Phase 0c, so keeping them avoids unnecessary reruns.
- Everything from Phase 0c onward should now be regenerated against the current codebase.

Canonical rerun boundary: start again at **Phase 0c**.

## What's Proven (Archived Run 1)

All items below refer to the archived run at
`logs_run1_no_tiered_site_discovery_and_missing_benchmark_specific_task_template/`.
They ran against real Modal sandboxes and produced artifacts on disk.

- **Phase 0a -- Benchmark Discovery.** Single Modal sandbox analyzed the full WebArena Verified codebase. Produced `BENCHMARK_MANIFEST.json` (6 sites, 813 tasks) and human-readable `.md` summary. Two sandbox runs, 90 total turns, $2.88 spend.
- **Phase 0b -- Sandbox Filesystem Mapping.** Pure local Python, no LLM. Computed per-site file lists from the manifest. Output: `SANDBOX_MAP.json` with keys for all 6 sites (shopping, shopping_admin, gitlab, reddit, wikipedia, map).
- **Legacy Phase 0c -- Per-Site Profiling.** The archived run used the older single-stage per-site profiling flow and produced `BENCHMARK_PROFILE_{site}.json` artifacts for all 6 sites. These artifacts are useful historical evidence, but they have been superseded by the current tiered Phase 0c design and should not be resumed from.
- **Phase 1 -- Task Wrapping (Mode A).** Loaded 813 tasks from the WebArena Verified dataset and wrapped them into benign task bundles for downstream phases. Output: `benign_tasks.json`. No LLM required.
- **Phase 2 -- Injection Generation.** 43 Modal sandbox runs across 6 sites with per-site task sharding. Shopping (182 tasks) and gitlab (160 tasks) were the largest. Produced 671 adversarial tasks total. In-sandbox validation caught schema errors before sandbox exit. Total $82.99 spend across 624 turns.
- **Phase 3 artifact production.** The archived run produced `validated_tasks.json`, `results.json`, and `diagnoses.json` under `phase_3/`, proving the Phase 3 path executed far enough to materialize downstream artifacts.
- **Modal sandbox primitive.** `run_claude_in_sandbox` creates a sandbox, stages files via `add_local_file` / `add_local_dir`, runs Claude Code through `claude-agent-sdk` via `_sdk_runner.py`, and returns file outputs plus a `_summary` key with cost, token usage, session ID, and per-model breakdowns.
- **Three Claude Code auth paths.** OAuth (`CLAUDE_CODE_OAUTH_TOKEN`), OpenRouter (`ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL`), and direct API key (`ANTHROPIC_API_KEY`). Priority: OAuth > OpenRouter > API key. Named Modal secret support via `WORLDSIM_CLAUDE_MODAL_SECRET`.
- **Volume-based benchmark upload.** `upload_to_volume` does one-time content-addressed upload, skips if already populated, and eliminates per-call benchmark hashing overhead.
- **Cost tracking with resume persistence.** `CostTracker` records per-sandbox cost data, persists to `cost_report.json`, and reloads on `--resume` without double-counting.

## What's Implemented (Current Code Complete, Unit-Tested)

Everything below reflects the current working tree.

- **Phase 0c -- Tiered per-site profiling.** Profiling is now split into Tier 1 and Tier 2. Tier 1 runs three sandboxes in parallel per site: verification capabilities, data model, and agent context. Tier 2 then runs injection-surface / task-coverage analysis using validated Tier 1 outputs as inputs.
- **Artifact-specific Phase 0c validation.** `_sandbox_validator.py` now validates `VERIFICATION_CAPABILITIES.json`, `DATA_MODEL.json`, `AGENT_CONTEXT.json`, and `INJECTION_SURFACE.json` independently. Host-side retries append concrete validation errors back into the retry prompt before re-running a tier.
- **`agent_context` as a first-class artifact.** Phase 0c now emits `AGENT_CONTEXT_{site}.json` containing response-format requirements, auth details, discovered vendor prompt templates, and benchmark/site context.
- **Shared benchmark-specific prompt builder.** `worldsim/agent_prompt.py` builds runtime prompts from `agent_context`, appending auth guidance, discovered credentials, response-schema requirements, and per-task format requirements sourced from `instantiation_dict`.
- **Phase 1 Mode A propagation.** Wrapped benchmark tasks now preserve `instantiation_dict` and embed `agent_context` when Phase 0c output is available.
- **Phase 1 Mode B propagation and cache hardening.** Novel-task generation now receives `AGENT_CONTEXT.json`, embeds it into generated tasks, and invalidates stale per-site caches when the sibling `AGENT_CONTEXT_{site}.json` changes or when cached tasks are missing the embedded context.
- **Phase 2 immutable contract update.** Adversarial task generation now treats `agent_context` as an immutable field copied from the benign task and passes `AGENT_CONTEXT.json` into the injection-generation sandbox so attacks are crafted against the real benchmark contract.
- **Phase 3 -- Benign Validation.** Full implementation: reset endpoint call, data seed application, Browser Use agent run, reward function evaluation, failure diagnosis loop (reward bug, data seed issue, impossible task, task too hard, agent limitation). Phase 3 now builds a benchmark-specific `site_prompt` from embedded `agent_context`.
- **Phase 4 -- Adversarial Evaluation.** Full iterative decision tree remains in place, and now uses the same `agent_context`-driven prompt builder as Phase 3 so benign and adversarial runs share the same benchmark contract.
- **Profile cross-reference hardening.** `profile_validation.py` and `_sandbox_validator.py` now reject `source_field` references that use a real entity with a field belonging to some other entity. Field validation is now entity-scoped, not global-name-scoped.
- **Browser Use agent runner.** `BrowserUseAgent` supports `start_urls` and `site_prompt`, preserves task-scoped HAR capture and screenshots, and uses benchmark-specific task text when provided instead of the generic one-line prompt.
- **Sandbox runner correctness.** The Modal sandbox runner now passes the Claude Code `claude_code` preset through the Agent SDK. This is a correctness fix, not just observability: without it, Modal sandboxes were not actually running Claude Code semantics. The runner also caps runs with `max_budget_usd=250.0` and emits better diagnostics for rate limits, stderr, and result subtype.
- **Reward evaluation -- dual path.** Primary: WebArena Verified vendor evaluator (`webarena_verified` package) with proper normalization (NFKC, unidecode, type dispatch). Fallback: homebrew evaluator for when vendor package is unavailable. Custom checker registry for `db_query_match` (Phase 4 injection verification).
- **Configurable agent LLM.** `make_llm` supports google/openai/anthropic providers with auto-detection from model name prefix. Default remains `gemini-3-flash-preview`.
- **Multi-site placeholder resolution.** `placeholders.py` handles `__SHOPPING__`, `__SHOPPING_ADMIN__`, `__GITLAB__`, `__REDDIT__`, `__WIKIPEDIA__`, `__MAP__` tokens. Strict mode raises on unresolved placeholders. `merge_placeholder_maps` composes config-level, instance-level, and task-level sources.
- **Data seeding.** Three mechanisms: sql (MySQL + PostgreSQL), api (HTTP requests), state_push (JSON PUT). SQL statements are validated against a disallowed keyword list. Async wrapper via `asyncio.to_thread`.
- **Eval worker pool.** `run_eval` + `staggered_worker` with `STAGGER_DELAY=5`. Phase-agnostic via `task_runner` callable. Site-aware routing via `run_tasks_by_site`.

## Pipeline Outputs on Disk

Current canonical live run:

```text
logs/
  last_run_state.json        # Marker for the reset live run
  phase_0a/
    BENCHMARK_MANIFEST.json
    BENCHMARK_MANIFEST.md
  phase_0b/
    SANDBOX_MAP.json
```

Archived historical run:

```text
logs_run1_no_tiered_site_discovery_and_missing_benchmark_specific_task_template/
  cost_report.json
  last_run_state.json
  pipeline_state.json
  phase_0a/
    BENCHMARK_MANIFEST.json
    BENCHMARK_MANIFEST.md
  phase_0b/
    SANDBOX_MAP.json
  phase_0c/
    BENCHMARK_PROFILE_gitlab.json
    BENCHMARK_PROFILE_map.json
    BENCHMARK_PROFILE_reddit.json
    BENCHMARK_PROFILE_shopping.json
    BENCHMARK_PROFILE_shopping.md
    BENCHMARK_PROFILE_shopping_admin.json
    BENCHMARK_PROFILE_wikipedia.json
  phase_1/
    benign_tasks.json
  phase_2/
    adversarial_tasks.json
  phase_3/
    diagnoses.json
    results.json
    validated_tasks.json
```

Golden path status:

- **Historical proof:** the archived run reached Phase 3 artifact production and produced real 0a/0b/0c/1/2/3 outputs.
- **Current canonical state:** `logs/` has been intentionally reset to `phase_0a` + `phase_0b` only.
- **Required next step:** rerun Phase 0c and everything downstream on the current code before treating any Phase 0c+ output as canonical again.

## Architecture Highlights

Key design decisions for anyone continuing this work:

- **Tiered Phase 0c profiling.** Per-site profiling is no longer a single monolithic sandbox. Verification capabilities, data model, and agent context are discovered independently first; injection-surface analysis happens only after those artifacts validate.
- **`agent_context` propagation.** Benchmark-specific auth, response-format requirements, and discovered prompt templates are now carried as task data instead of requiring later phases to reopen Phase 0c artifacts.
- **Benchmark-specific runtime prompts.** Phase 3 and Phase 4 no longer rely only on a generic "You are interacting with a web application..." prompt. They can now replay the benchmark's actual response contract and auth workflow.
- **Sandbox runner must use Claude Code semantics.** The Agent SDK path now explicitly requests the `claude_code` preset so Modal sandboxes behave like real Claude Code, not a generic SDK agent.
- **Phase 2 task sharding.** Large sites (shopping: 182 tasks) are split into shards of ~25 tasks. Shards run as independent Modal sandboxes in parallel, then results are merged with deduplication.
- **In-sandbox output validation.** `_sandbox_validator.py` runs inside the Modal sandbox as the last step before exit. This catches schema and contract issues within the same session at effectively zero extra orchestration cost.
- **Per-site instance locking.** `site_lock.py` provides asyncio locks keyed by site name. Diagnosis sandboxes run fully parallel. Agent reruns that mutate shared DB state serialize per site.
- **File routing via inclusion.** Sandboxes are scoped by `add_local_file` / `add_local_dir` calls, not ignore patterns. Phase 0b's `SANDBOX_MAP.json` remains the driver for which files each site's profiling sandbox receives.
- **Per-task resume in Phase 3/4.** `result.json` in each task directory serves as a completion sentinel. On `--resume`, existing completed results are merged with new ones.
- **Run archival boundary.** Because the first run predates the current Phase 0c/task-template fixes, resume is intentionally cut at Phase 0b. This preserves valid reconnaissance outputs while forcing regeneration of semantically stale downstream artifacts.

## WebArena env-ctrl base_url Fix

The original Docker images (`am1n3e/webarena-verified-*`) ship env-ctrl code where `_init()` requires a `base_url` argument, but the HTTP server (`POST /init`) calls `ops.init()` with no args. Some sites lacked an env-var fallback, causing `ValueError("base_url is required")` on every reset.

**Fix (two layers, both required for robustness):**

1. `scripts/webarena-compose-override.yml` -- sets `WA_ENV_CTRL_EXTERNAL_SITE_URL` for shopping, shopping_admin, gitlab, and reddit. Deployed into `vendors/webarena-verified/docker-compose.override.yml` by the patch script. Docker Compose auto-merges it. Survives container recreation.
2. `scripts/patch_webarena_containers.sh` -- patches running containers to add the env-var fallback in the Python `_init()` code (for images that lack it). Also deploys the override file. Idempotent.

Run after `docker compose up`: `./scripts/patch_webarena_containers.sh [HOST_IP]`.

For fresh container starts, only the override file is needed (the env var is sufficient when the Python code has the fallback; the patch script ensures it does).

## Known Issues and Lessons Learned

From code review sweeps, the archived run, and the current refactor:

- **The first run is historically useful but semantically stale.** The archive directory name is literal: that run predates tiered site discovery in Phase 0c and predates the benchmark-specific task-template / `agent_context` fix. It should not be resumed from beyond Phase 0b.
- **Agent SDK invocation is not automatically Claude Code.** When using `claude-agent-sdk`, you must explicitly request the `claude_code` preset. Without it, sandbox behavior diverges from the intended Claude Code contract.
- **Phase 0c needed to be decomposed.** A single profiling sandbox per site was too brittle. Separating data model, verification capabilities, and agent-context discovery prevents one bad inference from contaminating downstream injection-surface analysis.
- **Benchmark-specific task templates matter.** The missing benchmark-specific prompt/template issue was real. Later phases need the benchmark's response contract, auth expectations, and prompt shape, not just the task instruction.
- **Field validation must be entity-scoped.** It is not enough to check whether a field name exists somewhere in the data model. `Product.body` should fail if `body` only belongs to `Review`.
- **Phase 0c sandboxes need 4-hour timeout.** Complex sites (especially shopping / shopping_admin) can take tens of minutes, and the new tiered flow plus retries needs headroom.
- **Phase 2 sharding required for large sites.** Sending all 182 shopping tasks into one sandbox exceeded practical limits. Sharding into ~25-task chunks with parallel execution and merge solved it.
- **In-sandbox validation catches schema errors at zero cost.** Running the validator inside the same sandbox session avoids re-paying for a whole sandbox just to fix malformed JSON.
- **Volume upload for benchmarks matters.** `add_local_dir` on every call re-hashed the entire benchmark tree. `upload_to_volume` removed that per-call overhead.
- **`benign_task_id` from LLM output must be coerced to string.** LLMs sometimes emit integers. All ID matching now normalizes through `str()`.
- **Browser Use 0.12+ teardown uses `session.kill()`.** `.close()` does not exist and leads to hangs.
- **`IS_SANDBOX=1` is required.** Claude Code needs it to accept `bypassPermissions` as root inside Modal.
- **Empty-string env vars cause auth failures.** Set-but-empty vars must be treated as unset during preflight auth checks.

## What's Next

Prioritized execution order from the current canonical state:

1. **Rerun Phase 0c.** Start from the preserved `phase_0a` manifest and `phase_0b` sandbox map. Generate fresh tiered Phase 0c outputs (`BENCHMARK_PROFILE_{site}.json`, `AGENT_CONTEXT_{site}.json`, and tier debug artifacts).
2. **Rerun Phase 1.** Rebuild benign task bundles so Mode A and Mode B outputs embed the new `agent_context` field and preserve per-task formatting metadata.
3. **Rerun Phase 2.** Regenerate adversarial tasks so `agent_context` is preserved as an immutable field and injection prompts are conditioned on the benchmark-specific runtime contract.
4. **Stand up / verify WebArena Docker containers.** Pull the 6 Docker images from `am1n3e/webarena-verified-*`, verify env-ctrl reset behavior, and confirm the override / patch script is in place.
5. **Validate `instances.json`.** Verify site URLs, DB connections, reset endpoints, and URL placeholders for the running containers.
6. **Run Phase 3 (benign validation).** `uv run python -m worldsim.main phase 3 --instances instances.json --agent-model gemini-3-flash-preview`
7. **Run Phase 4 (adversarial evaluation).** `uv run python -m worldsim.main phase 4 --instances instances.json`

## Cost Summary

Historical archived-run summary only. Source:

`logs_run1_no_tiered_site_discovery_and_missing_benchmark_specific_task_template/cost_report.json`

These numbers are useful context, but they are not the final cost profile for the current tiered Phase 0c / `agent_context` design.

| Phase | Sandboxes | Turns | Cost |
|-------|-----------|-------|------|
| Phase 0a | 2 | 90 | $2.88 |
| Legacy Phase 0c | 8 (6 succeeded, 2 retries) | 196 | $13.88 |
| Phase 2 | 43 (sharded across 6 sites) | 624 | $82.99 |
| **Total** | **53** | **910** | **$99.75** |

Models used in the archived run: `claude-opus-4-6` (primary, ~$96.59), `claude-haiku-4-5-20251001` (sub-agent, ~$3.15). The archived run used the OpenRouter auth path.

## Codebase Stats

| Metric | Value |
|--------|-------|
| `worldsim/` Python lines | 10,039 |
| Test files | 14 files, 5,733 lines |
| Tests collected | 246 |
| Prompt files | 14 |

## Key Files

| File | Purpose |
|------|---------|
| `docs/worldsim-v5-technical-specifcation.md` | Source of truth. Updated to match tiered Phase 0c and `agent_context` propagation. |
| `docs/current_progress.md` | This file. Historical proof vs current canonical rerun state. |
| `worldsim/modal_sandbox.py` | The core sandbox primitive, secret wiring, and Modal execution path. |
| `worldsim/_sandbox_runner.py` | Runs inside Modal. Drives `claude-agent-sdk` with the Claude Code preset. |
| `worldsim/_sandbox_validator.py` | In-sandbox output validation for manifests, profiles, tasks, diagnosis, and ecological-validity artifacts. |
| `worldsim/phases/phase_0_recon.py` | Phase 0 orchestration, including the new tiered Phase 0c profiling flow. |
| `worldsim/agent_prompt.py` | Shared builder for benchmark-specific runtime prompts from `agent_context`. |
| `worldsim/phases/phase_1_mode_a.py` | Wraps benchmark tasks and now preserves `instantiation_dict` + embeds `agent_context`. |
| `worldsim/phases/phase_1_mode_b.py` | Novel-task generation, per-site caching, and `agent_context` propagation. |
| `worldsim/phases/phase_2_injections.py` | Task sharding, parallel sandbox execution, immutable-field merge, and `agent_context` preservation. |
| `worldsim/phases/phase_3_benign.py` | Benign validation with diagnosis loop and benchmark-specific runtime prompt injection. |
| `worldsim/phases/phase_4_adversarial.py` | Full adversarial decision tree with the same benchmark-specific runtime prompt builder. |
| `worldsim/browser_use_agent.py` | Browser Use lifecycle, HAR capture, screenshot persistence, and `site_prompt` support. |
| `worldsim/profile_validation.py` | Shared profile loading and host-side validation parity with the in-sandbox validator. |
| `worldsim/prompts/profile-*.md` | Tiered Phase 0c prompt set. |
| `worldsim/prompts/generate-benign-tasks.md` | Mode B prompt, now conditioned on `AGENT_CONTEXT.json`. |
| `worldsim/prompts/generate-injections.md` | Phase 2 prompt, now conditioned on `AGENT_CONTEXT.json`. |
| `tests/test_phase_0_recon.py` | Tests for tiered Phase 0c retries and non-publication of invalid outputs. |
| `tests/test_phase_1_tasks.py` | Tests for `agent_context` propagation, cache invalidation, and `instantiation_dict` preservation. |
| `tests/test_phase_3_benign.py` | Tests for benchmark-specific prompt construction. |
| `tests/test_phase_4_adversarial.py` | Tests for benchmark-specific prompt use in adversarial evaluation. |

## Auth Setup

Current working configuration:

- **Sandbox Claude Code auth:** archived real runs used `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL` (OpenRouter path). OAuth and direct Anthropic API key paths are also supported in current code.
- **Modal:** profile `theta-software-inc`, environment `test`. Token in `~/.modal.toml`.
- **Sandbox runner semantics:** current code explicitly passes the Claude Code `claude_code` preset through the Agent SDK so Modal sandboxes behave like Claude Code rather than a generic SDK agent.
- **Agent LLM (for Phases 3-4):** needs `GOOGLE_API_KEY` for the default `gemini-3-flash-preview`, or `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` if switching providers.

## Prompt Inventory

All 14 prompts in `worldsim/prompts/`:

| Prompt | Used by |
|--------|---------|
| `discover-benchmark.md` | Phase 0a |
| `profile-site.md` | Legacy single-sandbox Phase 0c prompt, retained for reference |
| `profile-verification-capabilities.md` | Phase 0c Tier 1A |
| `profile-data-model.md` | Phase 0c Tier 1B |
| `profile-agent-context.md` | Phase 0c Tier 1C |
| `profile-injection-surface.md` | Phase 0c Tier 2 |
| `generate-benign-tasks.md` | Phase 1 Mode B |
| `generate-injections.md` | Phase 2 |
| `diagnose-benign-failure.md` | Phase 3 |
| `probe-ecological-validity.md` | Phase 4 Gate 1 |
| `judge-adversarial-failure.md` | Phase 4 Gate 2 |
| `fix-ecological-validity.md` | Phase 4 validity-fix loop |
| `fix-injection-placement.md` | Phase 4 placement-fix loop |
| `generate-variant.md` | Phase 4 strategy variation |
