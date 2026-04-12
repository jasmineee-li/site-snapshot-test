# CLAUDE.md

Guidance for Claude Code sessions working in this repository.

## What this is

**WorldSim v5** — a Modal-Sandbox + Browser-Use adversarial evaluation pipeline for browser agents. The core research question is whether browser agents resist or comply with indirect prompt injections embedded in realistic web environments (WebArena).

**The authoritative technical spec is [`docs/worldsim-v5-technical-specifcation.md`](docs/worldsim-v5-technical-specifcation.md). Read it first.** Every module in `worldsim/` implements a section of that spec. When behavior diverges from the spec, the spec is right — fix the code.

## Architectural principles

1. **The v5 spec is the source of truth.** If the spec gives you literal code (`run_claude_in_sandbox`, `BrowserUseAgent`, `apply_data_seed`, `save_state`, `run_eval`, `run_strategy_variation`), use it. If the spec is silent on a mechanical detail, use your judgment and add a short comment explaining the decision.
2. **The orchestrator does not manage environments.** It connects to pre-running benchmark instances the user supplies (`{site_url, db_connection, reset_endpoint}`). It never starts, stops, or provisions them. Data seeding is per-task only, via `apply_data_seed`.
3. **File routing via inclusion, not ignore.** Modal sandboxes are scoped by *which files you add*, not by which files you ignore. `run_claude_in_sandbox` builds the image per call using `image.add_local_dir`. Never rely on ignore-file patterns for isolation.
4. **Trajectories flow through disk.** Per-task directories under `logs/<run>/<task_id>/` contain `history.json`, `result.json`, `screenshots/`, and `conversations/`. Diagnosis/judge sandboxes read `history.json` first and selectively load screenshots only when the text log is ambiguous.
5. **Two gates in Phase 4.** Gate 1 is ecological validity (probe the trajectory ~5× and average P(eval)); Gate 2 is attack effectiveness. Every trajectory passes through Gate 1 before we draw any conclusions from Gate 2.

## Where to read

| Topic | File |
|-------|------|
| Canonical spec (read first) | `docs/worldsim-v5-technical-specifcation.md` |
| Modal primitive + base image | `worldsim/modal_sandbox.py` |
| Browser Use runner | `worldsim/browser_use_agent.py` |
| Worker pool | `worldsim/eval_worker_pool.py` |
| State / resume | `worldsim/state.py` |
| Data seeding (sql/api/state_push) | `worldsim/seeding.py` |
| Reward dispatching | `worldsim/rewards.py` |
| Trajectory loader | `worldsim/trajectory.py` |
| Full prompts (verbatim from spec) | `worldsim/prompts/*.md` |
| Phase entrypoints | `worldsim/phases/phase_{0..4}_*.py` |

## Light-reference files (do NOT import)

Two files from the predecessor AgentLab pipeline survive on this branch as read-only reference material for specific Modal / Claude Code auth mechanics:

- `AgentLab/src/agentlab/benchmarks/redteam/execution.py` — Modal image setup, secret wiring, sandbox lifecycle
- `AgentLab/src/agentlab/benchmarks/redteam/claude_code.py` — Claude Code invocation flags

**Rules:**

- Read them when stuck on a specific mechanic. Understand, retype the equivalent in `worldsim/`.
- Never `import` from `AgentLab/`. The new package has zero runtime dependency on it.
- These two files could be deleted at a later cleanup pass without breaking `worldsim/`.
- Don't reach for any other file under `AgentLab/` — only those two survived commit 1 of this branch deliberately.

## Prerequisites before running anything

See `README.md`. Short version:

- Modal account + `modal token new`
- `anthropic-secret` Modal secret created
- `vendors/webarena-infinity/` cloned (or your own benchmark codebase)
- **For Phases 3 and 4 only:** a running WebArena instance with a known `{site_url, db_connection, reset_endpoint}`

## Working patterns

- `worldsim/main.py` is a **lazy CLI** — modify and run, not elaborate argparse
- Claude Code steps always run in Modal sandboxes via `run_claude_in_sandbox`
- Agent evaluation runs locally via Browser Use in an async worker pool with staggered start (`STAGGER_DELAY = 5`)
- Results default to `./logs/`; override with `WORLDSIM_STATE_DIR`
- `--resume` reads `logs/pipeline_state.json` and skips completed phases

## What NOT to do

- Do not generate or serve web applications. v5 connects to external environments.
- Do not import from `AgentLab/`.
- Do not reintroduce BrowserGym or the `agent-browser` CLI — those were the predecessor runner and are gone.
- Do not modify Phase 3's reward functions or tasks during Phase 4. Phase 4 only changes adversarial strategy; the benign task and reward function are invariant across variants.
- Do not manage benchmark environment lifecycles (starting, stopping, snapshotting). Calling `reset_endpoint` between tasks is fine; anything beyond that is out of scope.
