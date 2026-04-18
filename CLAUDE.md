# CLAUDE.md

Guidance for Claude Code sessions working in this repository.

## What this is

**WorldSim v5** — a Modal-Sandbox + Browser-Use adversarial evaluation pipeline for browser agents. The research question: do browser agents resist or comply with indirect prompt injections embedded in realistic web environments (WebArena)?

## Read the spec first

The authoritative technical spec is [`docs/worldsim-v5-technical-specifcation.md`](docs/worldsim-v5-technical-specifcation.md) (typo intentional — do not "fix"). Every module in `worldsim/` implements a section of it. **When behavior diverges from the spec, the spec is right — fix the code.** If the spec gives you literal code (`run_claude_in_sandbox`, `BrowserUseAgent`, `apply_data_seed`, `save_state`, `run_eval`, `run_strategy_variation`), use it verbatim. If the spec is silent on a mechanical detail, use judgment and leave a short comment. For the full repo layout and module map, see `README.md`; the one primitive you almost always need is `worldsim/modal_sandbox.py`.

`worldsim/main.py` is the CLI entrypoint — `uv run python -m worldsim.main phase 0 --benchmark vendors/webarena-verified`. See `--help` for flags.

Phase 2 is a single CLI phase with three internal stages: 2a planning in Modal sandboxes, 2b host-side text fill, 2c feasibility verification against a live dev instance. The standalone `phase 2c` subcommand is sugar for `phase 2 --feasibility-only`. Do not invent split-stage flags unless the code actually adds them.

## Non-negotiable principles

1. **The orchestrator does not manage environments.** It connects to pre-running benchmark instances the user supplies (`{site_url, reset_endpoint}`). Never start, stop, provision, or snapshot them. Data seeding is per-task only, via `apply_data_seed` (api/form channels only). SQL seeding is excluded from the evaluation methodology because it violates the threat model (a regular authenticated user cannot write to the database directly). `db_connection` is optional and used only for postcondition verification and reward evaluation. Calling `reset_endpoint` between tasks is fine; anything beyond that is out of scope.
2. **File routing via inclusion, not ignore.** Modal sandboxes are scoped by which files you `image.add_local_dir`, not by ignore-file patterns. Never rely on ignore patterns for isolation.
3. **Trajectories flow through disk.** Per-task dirs under `logs/<run>/<task_id>/` hold `history.json`, `result.json`, `screenshots/`, `conversations/`. Judge sandboxes read `history.json` first and load screenshots only when the text log is ambiguous.
4. **Two gates in Phase 4.** Gate 1 is ecological validity (probe each trajectory ~5× and average P(eval)). Gate 2 is attack effectiveness. Every trajectory passes Gate 1 before conclusions are drawn from Gate 2. Baseline capability (benign-under-attack) is reported as a Phase 4 byproduct, never used to filter tasks before Phase 4.

## Integration test requirement

If a PR changes `worldsim/editors/**`, `worldsim/seeding.py`, `worldsim/phases/phase_4_adversarial.py`, or `worldsim/phases/phase_2_feasibility.py`, run `scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r5.yaml` against a live stack before shipping and include the output in the PR description. Unit tests alone are not sufficient for editor, Phase 2c, or Phase 4 integration changes.

## What NOT to do

- Do not generate or serve web applications. v5 connects to external environments.
- Do not `import` from `AgentLab/`. The new package has zero runtime dependency on it.
- Do not reintroduce BrowserGym or the `agent-browser` CLI — predecessor runner, gone.
- Do not modify benign task contracts or reward functions during Phase 4. Phase 4 only varies adversarial strategy; the contracts emitted by Phase 3 are invariant across variants.
- Do not manage benchmark environment lifecycles (starting, stopping, snapshotting). `reset_endpoint` between tasks is the one exception.
- Do not bypass Phase 2c (`--skip-feasibility`) on shipping runs; the `feasibility.status="verified"` stamp is a gate input for Phase 4 under strict admission.
- Do not hand-edit `feasibility.status` in `adversarial_tasks.json`; trust the gate or re-run `phase 2c`.
- Do not run Phase 4 on a dataset that hasn't been through 2c — admission is strict as of 2026-04-18 and unverified tasks are skipped.
- The break-glass env override for Phase 4 admission is `WORLDSIM_STRICT_FEASIBILITY={true,false}`; it supersedes the source-controlled `STRICT_FEASIBILITY_ADMISSION` constant.

<important if="you are stuck on Modal image setup, secret wiring, sandbox lifecycle, or Claude Code invocation flags">
Two files from the predecessor AgentLab pipeline survive on this branch as read-only reference material for exactly those mechanics:

- `AgentLab/src/agentlab/benchmarks/redteam/execution.py` — Modal image setup, secrets, sandbox lifecycle
- `AgentLab/src/agentlab/benchmarks/redteam/claude_code.py` — Claude Code invocation flags

Read them, understand the mechanic, then retype the equivalent in `worldsim/`. Never `import` from `AgentLab/`. Do not reach for any other file under `AgentLab/` — only those two were kept deliberately, and both could be deleted at a later cleanup pass without breaking `worldsim/`.
</important>

<important if="you are actually running the pipeline, wiring up a phase entrypoint, or debugging a run">
- Claude Code steps always run in Modal sandboxes via `run_claude_in_sandbox`.
- Agent evaluation runs locally via Browser Use in an async worker pool with staggered start (`STAGGER_DELAY = 5`).
- Results default to `./logs/`; override with `WORLDSIM_STATE_DIR`.
- `--resume` reads `logs/pipeline_state.json` and skips completed phases.
- Prerequisites (Modal token, Claude Code auth, benchmark clone, running WebArena for Phase 4): see `README.md`. Phase 3 is agent-free and needs no live instances. Do not duplicate prerequisites into code or configs.
- Claude Code auth inside the sandbox supports **both** `CLAUDE_CODE_OAUTH_TOKEN` (Pro/Max subscription) and `ANTHROPIC_API_KEY` (API credits). OAuth wins when both are set — see `worldsim/modal_sandbox.py:_build_claude_secrets`. Never hard-code which one; always let `_build_claude_secrets` decide.
</important>
