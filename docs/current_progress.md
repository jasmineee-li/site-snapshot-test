# WorldSim v5 -- Current Progress

Last updated: 2026-04-13

## Branch

`feat/worldsim-v5`, 57 commits ahead of `main`. Latest commit: `fix: harden agent DX around resume and CLI contracts`.

## What's Proven (End-to-End Tested)

All items below ran against real Modal sandboxes with real Claude Code invocations and produced artifacts on disk.

- **Phase 0a -- Benchmark Discovery.** Single Modal sandbox analyzed the full WebArena Verified codebase. Produced `BENCHMARK_MANIFEST.json` (6 sites, 813 tasks) and human-readable `.md` summary. Two sandbox runs, 90 total turns, $2.88 spend.
- **Phase 0b -- Sandbox Filesystem Mapping.** Pure local Python, no LLM. Computed per-site file lists from the manifest. Output: `SANDBOX_MAP.json` with keys for all 6 sites (shopping, shopping_admin, gitlab, reddit, wikipedia, map).
- **Phase 0c -- Per-Site Profiling.** 6 parallel Modal sandboxes (one per site) profiled verification capabilities, data model, injection surface, and existing task coverage. All 6 profiles completed. Wall-clock durations ranged from 13 minutes (map) to 57 minutes (shopping). Total $13.88 spend across 196 turns.
- **Phase 1 -- Task Wrapping (Mode A).** Loaded 813 tasks from the WebArena Verified dataset and wrapped them into benign task bundles for downstream phases. Output: `benign_tasks.json`. No LLM required.
- **Phase 2 -- Injection Generation.** 43 Modal sandbox runs across 6 sites with per-site task sharding. Shopping (182 tasks) and gitlab (160 tasks) were the largest. Produced 671 adversarial tasks total. In-sandbox validation caught schema errors before sandbox exit. Total $82.99 spend across 624 turns.
- **Modal sandbox primitive.** `run_claude_in_sandbox` creates a sandbox, stages files via `add_local_file`/`add_local_dir`, runs Claude Code through `claude-agent-sdk` via `_sdk_runner.py` with `IS_SANDBOX=1`. Returns file outputs plus a `_summary` key with cost, token usage, session ID, and per-model breakdowns. NDJSON streaming for live observability of tool calls.
- **Three Claude Code auth paths.** OAuth (`CLAUDE_CODE_OAUTH_TOKEN`), OpenRouter (`ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL`), and direct API key (`ANTHROPIC_API_KEY`). Priority: OAuth > OpenRouter > API key. Named Modal secret support via `WORLDSIM_CLAUDE_MODAL_SECRET`.
- **Volume-based benchmark upload.** `upload_to_volume` does one-time content-addressed upload, skips if already populated. Eliminates per-call file hashing overhead.
- **Cost tracking with resume persistence.** `CostTracker` singleton records per-sandbox cost data, persists to `cost_report.json`, and reloads on `--resume` without double-counting.

## What's Implemented (Code Complete, Tested)

Everything below is coded, has unit tests, and has been reviewed by multiple code review agents.

- **Phase 3 -- Benign Validation.** Full implementation: reset endpoint call, data seed application, Browser Use agent run, reward function evaluation, failure diagnosis loop (reward bug, data seed issue, impossible task, task too hard, agent limitation). Adversarial-paired filtering ensures only tasks with matching adversarial counterparts are run. Parallel diagnosis with per-site instance locking. Blocked on standing up WebArena Docker containers for live runs.
- **Phase 4 -- Adversarial Evaluation.** Full iterative decision tree:
  - Gate 1: ecological validity probing (~5x per trajectory, average P(eval) vs threshold 0.5)
  - Gate 2: attack effectiveness classification (complied, task_broke, refused_or_ignored)
  - Ecological validity fix loop (up to 2 iterations via Modal sandbox)
  - Injection placement fix loop (up to 2 iterations via Modal sandbox)
  - Adaptive strategy variation: judge diagnoses refusal, recommends up to 3 strategies from pool (authority_framing, semantic_coherence, specificity, visual_concealment, verbosity_adjustment), generates variants in parallel Modal sandboxes, evaluates variants, runs each through ecological validity gate again
- **Reward evaluation -- dual path.** Primary: WebArena Verified vendor evaluator (`webarena_verified` package) with proper normalization (NFKC, unidecode, type dispatch). Fallback: homebrew evaluator for when vendor package is unavailable. Custom checker registry for `db_query_match` (Phase 4 injection verification).
- **Browser Use agent runner.** `BrowserUseAgent` with session lifecycle, HAR network trace capture, screenshot saving, network trace redaction (sensitive headers stripped), and `AgentResult` dataclass.
- **Configurable agent LLM.** `make_llm` factory supports google/openai/anthropic providers with auto-detection from model name prefix. Default: `gemini-3-flash-preview`. `make_agent_factory` returns a zero-arg callable for the worker pool.
- **Multi-site placeholder resolution.** `placeholders.py` handles `__SHOPPING__`, `__SHOPPING_ADMIN__`, `__GITLAB__`, `__REDDIT__`, `__WIKIPEDIA__`, `__MAP__` tokens. Strict mode raises on unresolved placeholders. `merge_placeholder_maps` composes config-level, instance-level, and task-level placeholder sources.
- **Data seeding.** Three mechanisms: sql (MySQL + PostgreSQL), api (HTTP requests), state_push (JSON PUT). SQL statements validated against a disallowed keyword list. Async wrapper via `asyncio.to_thread`.
- **SQL safety validation.** Read-only query enforcement in `rewards.py` and seed statement validation in `seeding.py`. Multi-statement queries blocked. Read-only transaction guard set on connections.
- **Eval worker pool.** `run_eval` + `staggered_worker` with `STAGGER_DELAY=5`. Phase-agnostic via `task_runner` callable. Site-aware routing via `run_tasks_by_site`.
- **Phase 1 Mode B (novel task generation).** Scaffolded with reward routing fallback and prompt schema. Not yet exercised end-to-end.

## Pipeline Outputs on Disk

```
logs/
  cost_report.json           # Aggregate cost data, 53 sandbox entries
  pipeline_state.json        # Resume checkpoint: phase_2 complete
  phase_0a/
    BENCHMARK_MANIFEST.json  # 6 sites, 813 tasks, Docker images, ports, services
    BENCHMARK_MANIFEST.md    # Human-readable summary
  phase_0b/
    SANDBOX_MAP.json         # Per-site file routing maps for all 6 sites
  phase_0c/
    BENCHMARK_PROFILE_shopping.json       # verification_capabilities, data_model,
    BENCHMARK_PROFILE_shopping.md         #   injection_surface, existing_task_coverage
    BENCHMARK_PROFILE_shopping_admin.json
    BENCHMARK_PROFILE_gitlab.json
    BENCHMARK_PROFILE_reddit.json
    BENCHMARK_PROFILE_wikipedia.json
    BENCHMARK_PROFILE_map.json
  phase_1/
    benign_tasks.json        # 813 wrapped task bundles
  phase_2/
    adversarial_tasks.json   # 671 adversarial tasks (merged from 6 sites)
```

Golden path status: Phases 0-2 complete with artifacts on disk. Phase 2 produced 671 adversarial tasks across 6 sites (shopping: 182, gitlab: 160, shopping_admin: 118, reddit: 102, map: 93, wikipedia: 16). Phases 3-4 blocked on WebArena container setup.

## Architecture Highlights

Key design decisions for anyone continuing this work:

- **Phase 2 task sharding.** Large sites (shopping, 182 tasks) are split into shards of ~25 tasks each. Shards run as independent Modal sandboxes in parallel, then results are merged with deduplication. Partial success accepted, only fully-failed sites are errors.
- **In-sandbox output validation.** `_sandbox_validator.py` runs inside the Modal sandbox as the last step before exit. Validates JSON schema, data-seed safety, and immutable field integrity. Catches errors within the same session at zero extra cost.
- **Per-site instance locking.** `site_lock.py` provides asyncio locks keyed by site name. Diagnosis sandboxes (Modal, stateless) run fully parallel. Agent reruns (Browser Use, mutate shared DB) serialize per site. Different sites run in parallel.
- **Sandbox observability.** Labels, turn counts, text previews in NDJSON streaming. Distinct log verbs for sandbox exit vs phase completion. Cost tracker entries include session ID, per-model token breakdowns, and wall-clock duration.
- **Shared profile validation.** `profile_validation.py` extracts profile loading and validation into a reusable helper, used by both Phase 2 and Phase 3.
- **SDK-based sandbox invocation with NDJSON streaming.** `_sdk_runner.py` runs inside the sandbox, emits typed NDJSON events (tool_call, text, error, summary) to stdout. The orchestrator streams these for live observability and captures the final summary event for cost tracking.
- **Multi-site placeholder resolution.** WebArena tasks reference sites via `__SHOPPING__` style tokens. `placeholders.py` provides a three-layer merge (config, instance, task) with strict validation. `agent_config.py` handles the full prepare-resolve-route pipeline across `run_tasks_by_site`.
- **Phase 4 iterative decision tree.** Not a flat loop. Each adversarial result flows through: ecological validity gate -> attack classification -> conditional fix loops (ecological validity fix, placement fix) -> adaptive strategy variation with fan-out. Each fix/variant passes back through the validity gate.
- **Configurable agent LLM (default gemini-3-flash-preview).** Provider auto-detected from model prefix. Supported: Google (langchain-google-genai), OpenAI (langchain-openai), Anthropic (langchain-anthropic). The orchestrator (Claude Code in Modal) and the browser agent (Browser Use locally) use different models and auth paths.
- **Network trace redaction.** `browser_use_agent.py` strips sensitive headers (authorization, cookie, session tokens, CSRF, API keys) from HAR traces before writing to disk. Substring matching catches non-standard header names.
- **Async I/O (asyncio.to_thread for DB/HTTP).** Data seeding wraps synchronous DB and HTTP calls via `asyncio.to_thread` to avoid blocking the event loop in the worker pool. Modal sandbox calls are natively async.
- **SQL safety validation.** Both `rewards.py` (read-only reward queries) and `seeding.py` (seed statements) validate SQL before execution. Multi-statement queries blocked, write-capable keywords rejected, read-only transaction guards set on connections.
- **Cost tracking with resume persistence.** `CostTracker` singleton accumulates per-sandbox cost data. `load()` replaces in-memory entries (not appends) to prevent double-counting on re-run. `save()` writes the full report including per-model token breakdowns.
- **File routing via inclusion.** Sandboxes are scoped by `add_local_file`/`add_local_dir` calls, not by ignore patterns. Phase 0b's `SANDBOX_MAP.json` drives which files each site's sandbox receives.
- **Per-task resume in Phase 3/4.** `result.json` in each task directory serves as a completion sentinel (atomic writes via tmpfile + `os.replace`). On `--resume`, `load_completed_results` scans for existing results, skips completed tasks, and merges prior results with new ones. Circuit breaker stops diagnosis if >30% of tasks error.
- **Ruff linting + formatting.** Pre-commit hook runs ruff with auto-fix on staged Python files.

## WebArena env-ctrl base_url Fix

The original Docker images (`am1n3e/webarena-verified-*`) ship env-ctrl code where `_init()` requires a `base_url` argument, but the HTTP server (`POST /init`) calls `ops.init()` with no args. Some sites lacked an env-var fallback, causing `ValueError("base_url is required")` on every reset.

**Fix (two layers, both required for robustness):**

1. `scripts/webarena-compose-override.yml` -- sets `WA_ENV_CTRL_EXTERNAL_SITE_URL` for shopping, shopping_admin, gitlab, and reddit. Deployed into `vendors/webarena-verified/docker-compose.override.yml` by the patch script. Docker Compose auto-merges it. Survives container recreation.
2. `scripts/patch_webarena_containers.sh` -- patches running containers to add the env-var fallback in the Python `_init()` code (for images that lack it). Also deploys the override file. Idempotent.

Run after `docker compose up`: `./scripts/patch_webarena_containers.sh [HOST_IP]`.

For fresh container starts, only the override file is needed (the env var is sufficient when the Python code has the fallback; the patch script ensures it does).

## Known Issues and Lessons Learned

From code review sweeps and real Phase 2 execution:

- **Modal `.aio` confusion.** Modal uses synchronicity wrapping: methods like `Sandbox.create`, `sandbox.terminate`, `sandbox.filesystem.read_text`, `sandbox.filesystem.write_text`, `sandbox.exec`, `claude_ps.wait`, `vol.listdir`, `vol.batch_upload` all need `.aio` for async. Natively async iterators (like `claude_ps.stdout`) do NOT use `.aio`. Getting this wrong causes silent hangs or "coroutine was never awaited" errors.
- **Reward evaluator must use vendor API.** Early implementation had homebrew normalization that missed NFKC, unidecode, trademark stripping, and type dispatch across 17 data types. Switched to `webarena_verified` package as primary path with homebrew as fallback only.
- **Phase 0c sandboxes need 4-hour timeout.** Complex sites (shopping, shopping_admin) took 45-57 minutes. Default 1-hour timeout was too tight with retries. Set to `timeout=14400`.
- **Phase 2 sharding required for large sites.** Sending all 182 shopping tasks into one sandbox exceeded output limits. Sharding into ~25-task chunks with parallel execution and merge solved it.
- **Partial success over all-or-nothing.** Phase 2 initially failed on any site error. Switched to accept partial success, only fully-failed sites are errors. 671 of 813 tasks produced adversarial variants (83% yield).
- **In-sandbox validation catches schema errors at zero cost.** Running the validator inside the same sandbox session, before exit, avoids the cost of re-running a sandbox just to fix a missing field or malformed JSON.
- **Volume upload for benchmarks.** Initial approach used `add_local_dir` per sandbox call, which re-hashed the entire benchmark codebase every time. Switched to `upload_to_volume` for one-time upload with content-addressed skip.
- **`benign_task_id` from LLM must be coerced to str.** Phase 4 receives task IDs from LLM-generated JSON. LLMs sometimes emit integers. All ID comparisons now go through `str()`.
- **cost_tracker.load() must replace, not append.** Original implementation appended loaded entries to existing in-memory entries, causing double-counting when a phase was re-run with `--resume`.
- **Prompt pipe-separator ambiguity.** Early prompts used `|` to delimit enum values in instructions. LLMs interpreted this as "or" rather than literal options. Switched to explicit comma-separated lists.
- **Browser Use 0.12+ teardown.** Uses `session.kill()`, not `.close()` (which does not exist and causes a WebSocket hang).
- **`IS_SANDBOX=1` env var.** Required for Claude Code to accept `bypassPermissions` as root in the Modal sandbox.
- **Immutable fields must be copied programmatically.** Phase 2 LLM output sometimes dropped or mutated fields from the benign task (site, task_id, etc.). Fix: copy immutable fields from the source benign task in code, not in the prompt.
- **Empty-string env vars cause auth failure.** `os.environ.get("KEY")` returns `""` for set-but-empty vars, which passes truthiness checks but fails auth. Preflight checks now validate non-empty values.
- **Sandbox max_turns needs headroom.** Default 100 turns was insufficient for large task sets. Bumped to 300.

## What's Next

Prioritized execution order:

1. **Stand up WebArena Docker containers.** Pull the 6 Docker images from `am1n3e/webarena-verified-*`. Shopping and shopping_admin share a Magento stack. Wikipedia and map require one-time data downloads (ZIM file, OSM tiles). EC2 m5.xlarge in us-east-2 (18.117.99.179) is provisioned.
2. **Configure `instances.json`.** Verify site URLs, DB connections, reset endpoints, and URL placeholders for the running containers. Template already exists at `instances.json`.
3. **Run Phase 3 (benign validation).** `uv run python -m worldsim.main phase 3 --instances instances.json --agent-model gemini-3-flash-preview`. Validates that benign tasks pass with the target agent before adversarial testing.
4. **Run Phase 4 (adversarial evaluation).** `uv run python -m worldsim.main phase 4 --instances instances.json`. Full decision tree with ecological validity gating and adaptive strategy variation.

## Cost Summary

Data from `logs/cost_report.json` (53 sandbox entries):

| Phase | Sandboxes | Turns | Cost |
|-------|-----------|-------|------|
| Phase 0a | 2 | 90 | $2.88 |
| Phase 0c | 8 (6 succeeded, 2 initial failures retried) | 196 | $13.88 |
| Phase 2 | 43 (sharded across 6 sites) | 624 | $82.99 |
| **Total** | **53** | **910** | **$99.75** |

Models used: `claude-opus-4-6` (primary, ~$96.59), `claude-haiku-4-5-20251001` (sub-agent, ~$3.15). All runs used OpenRouter auth path.

## Codebase Stats

| Metric | Value |
|--------|-------|
| `worldsim/` Python lines | 8,746 |
| Test files | 15 files, 4,548 lines |
| Tests collected | 212 |
| Prompt files | 10 |

## Key Files

| File | Purpose |
|------|---------|
| `docs/worldsim-v5-technical-specifcation.md` | Source of truth. Every phase, prompt, schema, code pattern. |
| `worldsim/modal_sandbox.py` | The one primitive every phase calls. SDK-based, NDJSON streaming. |
| `worldsim/_sandbox_runner.py` | Runs inside the Modal sandbox. Drives Claude Agent SDK. |
| `worldsim/_sandbox_validator.py` | In-sandbox output validation. Runs inside Modal, stdlib only. |
| `worldsim/agent_config.py` | LLM factory, agent factory, shared task routing, placeholder resolution. |
| `worldsim/placeholders.py` | Multi-site URL placeholder resolution with strict validation. |
| `worldsim/rewards.py` | Dual-path reward evaluation (vendor + homebrew + custom checkers). |
| `worldsim/browser_use_agent.py` | Browser Use agent lifecycle, HAR capture, trace redaction. |
| `worldsim/phases/phase_2_injections.py` | Task sharding, parallel sandbox execution, result merging. |
| `worldsim/phases/phase_3_benign.py` | Benign validation with diagnosis loop and adversarial filtering. |
| `worldsim/phases/phase_4_adversarial.py` | Largest phase. Full iterative decision tree. |
| `worldsim/site_lock.py` | Per-site asyncio locks for serializing agent reruns. |
| `worldsim/profile_validation.py` | Shared profile loading and validation. |
| `worldsim/seeding.py` | Data seed dispatchers (sql, api, state_push) with safety validation. |
| `worldsim/cost_tracker.py` | Per-sandbox cost accumulation with resume-safe persistence. |
| `worldsim/prompts/*.md` | 10 prompt files covering all phases and sub-tasks. |
| `worldsim/main.py` | CLI entrypoint. `uv run python -m worldsim.main phase {0..4}`. |
| `tests/` | 15 test files, 212 tests. |
| `CLAUDE.md` | Non-negotiable principles, what NOT to do, reference file pointers. |
| `scripts/smoke_modal.py` | Working example of `run_claude_in_sandbox` end-to-end. |
| `scripts/patch_webarena_containers.sh` | Fixes env-ctrl base_url bug in WebArena containers. |
| `scripts/webarena-compose-override.yml` | Docker Compose override setting WA_ENV_CTRL_EXTERNAL_SITE_URL. |

## Auth Setup

Current working configuration:

- **Sandbox Claude Code auth:** `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL` (OpenRouter path). Set in `.env`, loaded via `load_dotenv()` at CLI startup.
- **Modal:** Profile `theta-software-inc`, environment `test`. Token in `~/.modal.toml`.
- **Agent LLM (for Phases 3-4):** Will need `GOOGLE_API_KEY` for the default `gemini-3-flash-preview`, or `OPENAI_API_KEY`/`ANTHROPIC_API_KEY` if switching providers.

## Prompt Inventory

All 10 prompts in `worldsim/prompts/`:

| Prompt | Used by |
|--------|---------|
| `discover-benchmark.md` | Phase 0a |
| `profile-site.md` | Phase 0c |
| `generate-benign-tasks.md` | Phase 1 (Mode B, novel task generation) |
| `generate-injections.md` | Phase 2 |
| `diagnose-benign-failure.md` | Phase 3 |
| `probe-ecological-validity.md` | Phase 4 Gate 1 |
| `judge-adversarial-failure.md` | Phase 4 Gate 2 |
| `fix-ecological-validity.md` | Phase 4 (validity fix loop) |
| `fix-injection-placement.md` | Phase 4 (placement fix loop) |
| `generate-variant.md` | Phase 4 (strategy variation) |
