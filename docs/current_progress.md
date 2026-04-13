# WorldSim v5 -- Current Progress

Last updated: 2026-04-12

## Branch

`feat/worldsim-v5`, 27 commits ahead of `main`. Latest commit: `feat: multi-site placeholders, Phase 4 decision tree, async I/O, SQL hardening, tests`.

## What's Proven (End-to-End Tested)

All items below ran against real Modal sandboxes with real Claude Code invocations and produced artifacts on disk.

- **Phase 0a -- Benchmark Discovery.** Single Modal sandbox analyzed the full WebArena Verified codebase. Produced `BENCHMARK_MANIFEST.json` (6 sites, 813 tasks) and human-readable `.md` summary. Two sandbox runs, 90 total turns, $2.88 spend.
- **Phase 0b -- Sandbox Filesystem Mapping.** Pure local Python, no LLM. Computed per-site file lists from the manifest. Output: `SANDBOX_MAP.json` with keys for all 6 sites (shopping, shopping_admin, gitlab, reddit, wikipedia, map).
- **Phase 0c -- Per-Site Profiling.** 6 parallel Modal sandboxes (one per site) profiled verification capabilities, data model, injection surface, and existing task coverage. All 6 profiles completed. Wall-clock durations ranged from 13 minutes (map) to 57 minutes (shopping). Total $13.88 spend across 196 turns.
- **Phase 1 -- Task Wrapping (Mode A).** Loaded 813 tasks from the WebArena Verified dataset and wrapped them into benign task bundles for downstream phases. Output: `benign_tasks.json`. No LLM required.
- **Modal sandbox primitive.** `run_claude_in_sandbox` creates a sandbox, stages files via `add_local_file`/`add_local_dir`, runs Claude Code through `claude-agent-sdk` via `_sdk_runner.py` with `IS_SANDBOX=1`. Returns file outputs plus a `_summary` key with cost, token usage, session ID, and per-model breakdowns. NDJSON streaming for live observability of tool calls.
- **Three Claude Code auth paths.** OAuth (`CLAUDE_CODE_OAUTH_TOKEN`), OpenRouter (`ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL`), and direct API key (`ANTHROPIC_API_KEY`). Priority: OAuth > OpenRouter > API key. Named Modal secret support via `WORLDSIM_CLAUDE_MODAL_SECRET`.
- **Volume-based benchmark upload.** `upload_to_volume` does one-time content-addressed upload, skips if already populated. Eliminates per-call file hashing overhead.
- **Cost tracking with resume persistence.** `CostTracker` singleton records per-sandbox cost data, persists to `cost_report.json`, and reloads on `--resume` without double-counting.

## What's Implemented (Code Complete, Not Yet E2E Tested)

Everything below is coded, reviewed by 11+ code review agents, and has unit tests where applicable. Blocked on standing up WebArena Docker containers for live runs.

- **Phase 2 -- Injection Generation.** One Modal sandbox per site. Reads Phase 1 benign tasks + Phase 0c profiles. Outputs `adversarial_tasks.json` per site, merged into a combined file. All inputs exist on disk, unblocked for execution.
- **Phase 3 -- Benign Validation.** Full implementation: reset endpoint call, data seed application, Browser Use agent run, reward function evaluation, failure diagnosis loop (reward bug, data seed issue, impossible task, task too hard, agent limitation). Blocked on running WebArena instances and configuring `instances.json`.
- **Phase 4 -- Adversarial Evaluation (875 lines).** Full iterative decision tree:
  - Gate 1: ecological validity probing (~5x per trajectory, average P(eval) vs threshold 0.5)
  - Gate 2: attack effectiveness classification (complied, task_broke, refused_or_ignored)
  - Ecological validity fix loop (up to 2 iterations via Modal sandbox)
  - Injection placement fix loop (up to 2 iterations via Modal sandbox)
  - Adaptive strategy variation: judge diagnoses refusal, recommends up to 3 strategies from pool (authority_framing, semantic_coherence, specificity, visual_concealment, verbosity_adjustment), generates variants in parallel Modal sandboxes, evaluates variants, runs each through ecological validity gate again
- **Reward evaluation -- dual path.** Primary: WebArena Verified vendor evaluator (`webarena_verified` package) with proper normalization (NFKC, unidecode, type dispatch). Fallback: homebrew evaluator for when vendor package is unavailable. Custom checker registry for `db_query_match` (Phase 4 injection verification).
- **Browser Use agent runner (598 lines).** `BrowserUseAgent` with session lifecycle, HAR network trace capture, screenshot saving, network trace redaction (sensitive headers stripped), and `AgentResult` dataclass.
- **Configurable agent LLM.** `make_llm` factory supports google/openai/anthropic providers with auto-detection from model name prefix. Default: `gemini-3-flash-preview`. `make_agent_factory` returns a zero-arg callable for the worker pool.
- **Multi-site placeholder resolution.** `placeholders.py` handles `__SHOPPING__`, `__SHOPPING_ADMIN__`, `__GITLAB__`, `__REDDIT__`, `__WIKIPEDIA__`, `__MAP__` tokens. Strict mode raises on unresolved placeholders. `merge_placeholder_maps` composes config-level, instance-level, and task-level placeholder sources.
- **Data seeding.** Three mechanisms: sql (MySQL + PostgreSQL), api (HTTP requests), state_push (JSON PUT). SQL statements validated against a disallowed keyword list. Async wrapper via `asyncio.to_thread`.
- **SQL safety validation.** Read-only query enforcement in `rewards.py` and seed statement validation in `seeding.py`. Multi-statement queries blocked. Read-only transaction guard set on connections.
- **Eval worker pool.** `run_eval` + `staggered_worker` with `STAGGER_DELAY=5`. Phase-agnostic via `task_runner` callable. Site-aware routing via `run_tasks_by_site`.
- **Unit tests (207 lines across 4 files).** `test_runtime_contracts.py` (91 lines, tests Modal `.aio` method availability, agent config construction, placeholder resolution), `test_sql_safety.py` (54 lines, tests read-only enforcement and keyword blocking), `test_state.py` (15 lines, save/load roundtrip), `test_trace_redaction.py` (39 lines, tests sensitive header stripping).

## Pipeline Outputs on Disk

```
logs/
  cost_report.json           # Aggregate cost data, 10 sandbox entries
  pipeline_state.json        # Resume checkpoint: phase_0c complete
  phase_0a/
    BENCHMARK_MANIFEST.json  # 6 sites, 813 tasks, Docker images, ports, services
    BENCHMARK_MANIFEST.md    # Human-readable summary
  phase_0b/
    SANDBOX_MAP.json         # Per-site file routing maps for all 6 sites
  phase_0c/
    BENCHMARK_PROFILE_shopping.json       # verification_capabilities, data_model,
    BENCHMARK_PROFILE_shopping.md         #   injection_surface, existing_task_coverage
    BENCHMARK_PROFILE_shopping_admin.json
    BENCHMARK_PROFILE_shopping_admin.md
    BENCHMARK_PROFILE_gitlab.json
    BENCHMARK_PROFILE_gitlab.md
    BENCHMARK_PROFILE_reddit.json
    BENCHMARK_PROFILE_reddit.md
    BENCHMARK_PROFILE_wikipedia.json
    BENCHMARK_PROFILE_wikipedia.md
    BENCHMARK_PROFILE_map.json
    BENCHMARK_PROFILE_map.md
  phase_1/
    benign_tasks.json        # 813 wrapped task bundles
```

Golden path status: Phase 0 (all sub-steps) and Phase 1 complete with artifacts. Phase 2 ready to execute. Phases 3-4 blocked on WebArena container setup.

## Architecture Highlights

Key design decisions for anyone continuing this work:

- **SDK-based sandbox invocation with NDJSON streaming.** `_sdk_runner.py` runs inside the sandbox, emits typed NDJSON events (tool_call, text, error, summary) to stdout. The orchestrator streams these for live observability and captures the final summary event for cost tracking.
- **Multi-site placeholder resolution.** WebArena tasks reference sites via `__SHOPPING__` style tokens. `placeholders.py` provides a three-layer merge (config, instance, task) with strict validation. `agent_config.py` handles the full prepare-resolve-route pipeline across `run_tasks_by_site`.
- **Phase 4 iterative decision tree.** Not a flat loop. Each adversarial result flows through: ecological validity gate -> attack classification -> conditional fix loops (ecological validity fix, placement fix) -> adaptive strategy variation with fan-out. Each fix/variant passes back through the validity gate.
- **Configurable agent LLM (default gemini-3-flash-preview).** Provider auto-detected from model prefix. Supported: Google (langchain-google-genai), OpenAI (langchain-openai), Anthropic (langchain-anthropic). The orchestrator (Claude Code in Modal) and the browser agent (Browser Use locally) use different models and auth paths.
- **Network trace redaction.** `browser_use_agent.py` strips sensitive headers (authorization, cookie, session tokens, CSRF, API keys) from HAR traces before writing to disk. Substring matching catches non-standard header names.
- **Async I/O (asyncio.to_thread for DB/HTTP).** Data seeding wraps synchronous DB and HTTP calls via `asyncio.to_thread` to avoid blocking the event loop in the worker pool. Modal sandbox calls are natively async.
- **SQL safety validation.** Both `rewards.py` (read-only reward queries) and `seeding.py` (seed statements) validate SQL before execution. Multi-statement queries blocked, write-capable keywords rejected, read-only transaction guards set on connections.
- **Cost tracking with resume persistence.** `CostTracker` singleton accumulates per-sandbox cost data. `load()` replaces in-memory entries (not appends) to prevent double-counting on re-run. `save()` writes the full report including per-model token breakdowns.
- **File routing via inclusion.** Sandboxes are scoped by `add_local_file`/`add_local_dir` calls, not by ignore patterns. Phase 0b's `SANDBOX_MAP.json` drives which files each site's sandbox receives.
- **Per-task resume in Phase 3/4.** `result.json` in each task directory serves as a completion sentinel (atomic writes via tmpfile + `os.replace`). On `--resume`, `load_completed_results` scans for existing results, skips completed tasks, and merges prior results with new ones. `task_dir_root` is persisted in `pipeline_state.json` so resume reuses the same output directory. Circuit breaker stops diagnosis if >30% of tasks error. No-changes heuristic exits the fix loop early when diagnosis makes no effective changes. Follows the filesystem sentinel pattern from SWE-bench, SWE-agent, and AgentLab.

## Known Issues and Lessons Learned

From 11+ code review agent sweeps:

- **Modal `.aio` confusion.** Modal uses synchronicity wrapping: methods like `Sandbox.create`, `sandbox.terminate`, `sandbox.filesystem.read_text`, `sandbox.filesystem.write_text`, `sandbox.exec`, `claude_ps.wait`, `vol.listdir`, `vol.batch_upload` all need `.aio` for async. Natively async iterators (like `claude_ps.stdout`) do NOT use `.aio`. Getting this wrong causes silent hangs or "coroutine was never awaited" errors.
- **Reward evaluator must use vendor API.** Early implementation had homebrew normalization that missed NFKC, unidecode, trademark stripping, and type dispatch across 17 data types. Switched to `webarena_verified` package as primary path with homebrew as fallback only.
- **Phase 0c sandboxes need 4-hour timeout.** Complex sites (shopping, shopping_admin) took 45-57 minutes. Default 1-hour timeout was too tight with retries. Set to `timeout=14400`.
- **Volume upload for benchmarks.** Initial approach used `add_local_dir` per sandbox call, which re-hashed the entire benchmark codebase every time. Switched to `upload_to_volume` for one-time upload with content-addressed skip.
- **`benign_task_id` from LLM must be coerced to str.** Phase 4 receives task IDs from LLM-generated JSON. LLMs sometimes emit integers. All ID comparisons now go through `str()`.
- **cost_tracker.load() must replace, not append.** Original implementation appended loaded entries to existing in-memory entries, causing double-counting when a phase was re-run with `--resume`.
- **Prompt pipe-separator ambiguity.** Early prompts used `|` to delimit enum values in instructions. LLMs interpreted this as "or" rather than literal options. Switched to explicit comma-separated lists.
- **Browser Use 0.12+ teardown.** Uses `session.kill()`, not `.close()` (which does not exist and causes a WebSocket hang).
- **`IS_SANDBOX=1` env var.** Required for Claude Code to accept `bypassPermissions` as root in the Modal sandbox.

## What's Next

Prioritized execution order:

1. **Phase 2 (injection generation).** All 6 site profiles and 813 benign tasks exist on disk. Run `uv run python -m worldsim.main phase 2`. Unblocked.
2. **Stand up WebArena Docker containers.** Pull the 6 Docker images from `am1n3e/webarena-verified-*`. Shopping and shopping_admin share a Magento stack. Wikipedia and map require one-time data downloads (ZIM file, OSM tiles).
3. **Create `instances.json`.** Configure `BenchmarkConfig` with site URLs, DB connections, reset endpoints, and URL placeholders for the running containers.
4. **Run Phase 3 (benign validation).** `uv run python -m worldsim.main phase 3 --instances instances.json --agent-model gemini-3-flash-preview`. Validates that benign tasks pass with the target agent before adversarial testing.
5. **Run Phase 4 (adversarial evaluation).** `uv run python -m worldsim.main phase 4 --instances instances.json`. Full decision tree with ecological validity gating and adaptive strategy variation.

## Cost Summary

Data from `logs/cost_report.json`:

| Phase | Sandboxes | Turns | Cost |
|-------|-----------|-------|------|
| Phase 0a | 2 | 90 | $2.88 |
| Phase 0c | 8 (6 succeeded, 2 initial failures retried) | 196 | $13.88 |
| **Total** | **10** | **286** | **$16.76** |

Models used: `claude-opus-4-6` (primary, ~$12.93), `claude-haiku-4-5-20251001` (sub-agent, ~$2.86). All runs used OpenRouter auth path.

## Key Files

| File | Purpose |
|------|---------|
| `docs/worldsim-v5-technical-specifcation.md` | Source of truth. Every phase, prompt, schema, code pattern. |
| `worldsim/modal_sandbox.py` | The one primitive every phase calls. SDK-based, NDJSON streaming. |
| `worldsim/_sandbox_runner.py` | Runs inside the Modal sandbox. Drives Claude Agent SDK. |
| `worldsim/agent_config.py` | LLM factory, agent factory, shared task routing, placeholder resolution. |
| `worldsim/placeholders.py` | Multi-site URL placeholder resolution with strict validation. |
| `worldsim/rewards.py` | Dual-path reward evaluation (vendor + homebrew + custom checkers). |
| `worldsim/browser_use_agent.py` | Browser Use agent lifecycle, HAR capture, trace redaction. |
| `worldsim/phases/phase_4_adversarial.py` | Largest phase (875 lines). Full iterative decision tree. |
| `worldsim/seeding.py` | Data seed dispatchers (sql, api, state_push) with safety validation. |
| `worldsim/cost_tracker.py` | Per-sandbox cost accumulation with resume-safe persistence. |
| `worldsim/prompts/*.md` | 10 prompt files covering all phases and sub-tasks. |
| `worldsim/main.py` | CLI entrypoint. `uv run python -m worldsim.main phase {0..4}`. |
| `tests/` | 4 test files, 207 lines. Runtime contracts, SQL safety, state, trace redaction. |
| `CLAUDE.md` | Non-negotiable principles, what NOT to do, reference file pointers. |
| `scripts/smoke_modal.py` | Working example of `run_claude_in_sandbox` end-to-end. |

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
| `generate-benign-tasks.md` | Phase 1 (Mode B, stretch goal) |
| `generate-injections.md` | Phase 2 |
| `diagnose-benign-failure.md` | Phase 3 |
| `probe-ecological-validity.md` | Phase 4 Gate 1 |
| `judge-adversarial-failure.md` | Phase 4 Gate 2 |
| `fix-ecological-validity.md` | Phase 4 (validity fix loop) |
| `fix-injection-placement.md` | Phase 4 (placement fix loop) |
| `generate-variant.md` | Phase 4 (strategy variation) |
