# WorldSim v5 — Current Progress

Last updated: 2026-04-12

## Branch

`feat/worldsim-v5` off `main` — 7 commits ahead.

## What's proven (smoke-tested end-to-end)

- **Modal Sandbox primitive** — `run_claude_in_sandbox` creates a sandbox, stages files via `add_local_file`/`add_local_dir`, runs Claude Code with `IS_SANDBOX=1` (runs as root without the `--dangerously-skip-permissions` root refusal), collects output files. Tested on Modal `test` environment with OpenRouter auth.
- **Three Claude Code auth paths** — OAuth (`CLAUDE_CODE_OAUTH_TOKEN`), OpenRouter (`ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL`), and direct API key (`ANTHROPIC_API_KEY`). Priority: OAuth > OpenRouter > API key. Implemented in `_build_claude_secrets()` with 9 test scenarios passing.
- **`.env` architecture** — `load_dotenv()` at CLI startup, shell exports take precedence, all auth keys in one file.
- **Browser Use + Chromium** — `BrowserSession` starts, navigates, and tears down cleanly. Note: browser-use 0.12+ uses `session.kill()` for teardown, NOT `.close()` (which doesn't exist and causes a WebSocket hang). Playwright is installed but browser-use 0.12 uses `cdp-use` directly.

## What exists (code written, not yet wired to real phases)

| Module | Status | Notes |
|--------|--------|-------|
| `worldsim/modal_sandbox.py` | **Working** | `run_claude_in_sandbox` proven. `IS_SANDBOX=1`, `pty=True`, `--output-format json`. |
| `worldsim/browser_use_agent.py` | **Skeleton** | `BrowserUseAgent` class with `setup/run/teardown`. Not yet called from any phase. `llm` param configured by caller. |
| `worldsim/eval_worker_pool.py` | **Skeleton** | `run_eval` + `staggered_worker` with `STAGGER_DELAY=5`. Phase-agnostic (`task_runner` callable). Not tested. |
| `worldsim/seeding.py` | **Skeleton** | `apply_data_seed` for sql/api/state_push. `execute_sql` supports MySQL. Not tested against a real DB. |
| `worldsim/rewards.py` | **Stubs** | `run_reward_function` dispatcher with four `NotImplementedError` checkers (url_exact_match, html_match, db_query_match, string_match). |
| `worldsim/config.py` | **Done** | Pydantic `BenchmarkConfig` + `BenchmarkInstance` schemas. |
| `worldsim/state.py` | **Done** | `save_state` / `load_state` roundtrip verified. |
| `worldsim/prompt_loading.py` | **Done** | `load_prompt(name)` loads from `worldsim/prompts/`. All 6 prompts load. |
| `worldsim/trajectory.py` | **Done** | `load_trajectory_into_sandbox` populates sandbox file map. |
| `worldsim/main.py` | **Done** | CLI entrypoint with `phase {0..4}` + `resume`. `load_dotenv()` on startup. |
| `worldsim/phases/phase_0_recon.py` | **Stub** | `NotImplementedError` with spec pointers. |
| `worldsim/phases/phase_1_tasks.py` | **Stub** | Mode A wrap + Mode B stub. |
| `worldsim/phases/phase_2_injections.py` | **Stub** | Injection generation stub. |
| `worldsim/phases/phase_3_benign.py` | **Stub** | Benign validation stub. Blocked on WebArena. |
| `worldsim/phases/phase_4_adversarial.py` | **Stub** | Adversarial eval + strategy variation stub. Blocked on WebArena. |

## What's NOT done

1. **Phases 0–4 implementation** — all phase modules are stubs.
2. **WebArena instances** — not running. Required for Phases 3–4. User responsibility per the v5 spec.
3. **Reward function checkers** — four stubs in `rewards.py`.
4. **Browser Use + LLM integration** — `BrowserUseAgent` exists but no LLM has been configured or tested (the `llm` param is set by the caller in `main.py`).
5. **`uv.lock`** — not committed.

## Key files for anyone continuing this work

| File | What to read it for |
|------|-------------------|
| `docs/worldsim-v5-technical-specifcation.md` | **Source of truth.** Every phase, prompt, schema, and code pattern. |
| `worldsim/modal_sandbox.py` | The one primitive every phase calls. |
| `worldsim/prompts/*.md` | Ready-to-use prompts for each phase. |
| `CLAUDE.md` | Non-negotiable principles, what NOT to do, reference file pointers. |
| `scripts/smoke_modal.py` | Working example of `run_claude_in_sandbox` end-to-end. |
| `AgentLab/src/agentlab/benchmarks/redteam/execution.py` | Light reference for Modal mechanics (read-only, never import). |

## Auth setup (current .env)

- `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL` — OpenRouter path, working.
- Modal profile: `theta-software-inc`, environment: `test`.
- Modal token: written to `~/.modal.toml`.

## Dependency versions

- Python 3.12
- modal >= 1.0 (installed: 1.4.1)
- browser-use >= 0.12.6
- playwright 1.58.0 + Chromium
- Claude Code 2.1.104 (inside Modal sandbox)
