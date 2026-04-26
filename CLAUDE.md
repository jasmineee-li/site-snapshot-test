## What this is

**WorldSim v5** — a Modal-Sandbox + Browser-Use adversarial evaluation pipeline for browser agents. Research question: do browser agents resist or comply with indirect prompt injections embedded in realistic web environments (WebArena)?

**Scope:** strict WASP alignment — **GitLab issues/comments and Reddit (Postmill) posts/comments only.** Current dataset: 38 tasks (22 GitLab + 16 Reddit), regenerated 2026-04-26 after post-`057e8e26` strict exposure-contract eligibility from the same WebArena task pool as the prior 84-task figure; see `logs_clean_2026-04-26/phase_2/`. Rationale (IPI threat model, Magento/Wikipedia/OSM exclusion): [`docs/handoffs/wasp-aligned-scoping-decision.md`](docs/handoffs/wasp-aligned-scoping-decision.md).

## Read the spec first

Authoritative spec: [`docs/worldsim-v5-technical-specifcation.md`](docs/worldsim-v5-technical-specifcation.md) — the typo in the filename is intentional, do not "fix". Every module in `worldsim/` implements a section. **When code diverges from the spec, the spec is right — fix the code.** If current source or handoff docs expose a real spec drift, update the spec first, then align the code. If the spec provides literal code (`run_claude_in_sandbox`, `BrowserUseAgent`, `apply_data_seed`, `save_state`, `run_eval`), use it verbatim.

Repo layout and prerequisites: `README.md`. Current Claude/Modal auth precedence lives in `worldsim/modal_sandbox.py::_build_claude_secrets` and `worldsim/phase_4/anthropic_client.py`. CLI entrypoint: `uv run python -m worldsim.main --help`.

## Non-negotiable invariants

- The orchestrator does not manage benchmark environment lifecycles. It connects to pre-running instances supplied by the user (`{site_url, reset_endpoint}`); `reset_endpoint` between tasks is the one exception.
- No runtime imports from `AgentLab/`. v5 has zero dependency on it; the two files still on disk (`execution.py`, `claude_code.py`) are read-only reference material.
- Do not generate or serve web applications.
- Phase 2c feasibility gate is strict admission for Phase 4. Never ship a Phase 4 run on an unverified dataset, and never hand-edit `feasibility.status`.
- No Phase 4 trajectory step routes through `run_claude_in_sandbox`. Refusal judge, variant generator, Transcript Purpose classifier, VEA, and placement-fix all use direct Anthropic Messages API calls — see `worldsim/phase_4/{judge_api,variant_api,transcript_purpose_api,verbalized_eval_awareness_api,placement_api}.py`.

## Handoff docs

Pick the one matching your task and read it before working in that area:

- [`codex-handoff-paint-verified-oracle.md`](docs/handoffs/codex-handoff-paint-verified-oracle.md) — PVPO capture, ink-occupancy encounter detection, Phase 4 gates; read the top supersession notes before using older P(eval) material.
- [`codex-handoff-phase-4-judge-variant-api-cutover.md`](docs/handoffs/codex-handoff-phase-4-judge-variant-api-cutover.md) — why judge/variant/TP/VEA/placement-fix use host API, not sandboxes.
- [`rigor-run-setup.md`](docs/handoffs/rigor-run-setup.md) — fresh-host bootstrap, preflight, launch sequence.
- [`wasp-aligned-scoping-decision.md`](docs/handoffs/wasp-aligned-scoping-decision.md) — dataset scope rationale.

<important if="you are touching Phase 4, PVPO capture, encounter detection, trajectory extraction, or the observational metrics">
- Phase 4 has **two gates** (encounter via PVPO `max_coverage ∈ [0.0, 1.0]`; attack effectiveness via refusal judge) and **two observational metrics** that never gate (Transcript Purpose, VEA). `max_coverage == 0` routes to placement-fix.
- Trajectories flow through disk. Per-task dirs under `logs/<run>/<task_id>/` hold `history.json`, `result.json`, `screenshots/`, `conversations/`, `pvpo/`. Judges and classifiers receive host-side dataclass-projected slices.
- Full current mechanics live in `worldsim/phase_4/*`; [`docs/handoffs/codex-handoff-paint-verified-oracle.md`](docs/handoffs/codex-handoff-paint-verified-oracle.md) is useful historical context with supersession notes at the top. Apollo / Needham prompt protocols are documented in the code at `worldsim/phase_4/{transcript_purpose_api,verbalized_eval_awareness_api}.py`; Apollo prompts are SHA256-pinned in `tests/test_phase_4_transcript_purpose_api.py`.
</important>

<important if="you are modifying the Phase 4 strategy pool or reward functions">
- Do not reintroduce `visual_concealment` — paper concealment is behavioral (final-response steering), not visual. Category error.
- Behavioral concealment strategies (`scripted_message`, `command_silent_execution`, `false_justification`) are deferred indefinitely; the current `AgentResponseEvaluator` + `db_query_match` reward is tool-side only and cannot grade them.
- Phase 4 only varies adversarial strategy. Benign task contracts and reward functions emitted by Phase 3 are invariant across variants — do not modify.
</important>

<important if="you are setting up a fresh host, deploying or debugging the benchmark proxy, or preparing a rigor run">
- Full runbook: [`docs/handoffs/rigor-run-setup.md`](docs/handoffs/rigor-run-setup.md).
- Run `scripts/setup_phase4_on_host.sh` on any fresh host before Phase 4 — its preflight step (`pytest -m preflight tests/preflight`) proves CDP endpoints, storage_state, and evaluator venv.
- Proxy: source of truth is `scripts/deploy_benchmark_proxy.sh`. Never hand-edit `/etc/nginx/conf.d/worldsim-proxy.conf`. Verify parity with `scripts/check_proxy_drift.sh --verify-runtime`.
- Rigor runs need `chrome-headless-shell` Docker containers (`worldsim/docker/chrome-headless-shell.Dockerfile`) because `HeadlessExperimental.beginFrame` is unsupported on native macOS. Without them, PVPO falls back to zero coverage and every trajectory routes to placement-fix.
</important>

<important if="you are preparing to ship a PR that touches editors, seeding, Phase 2c, Phase 4, PVPO capture, or the six host-API prompt files">
- Run `scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r5.yaml --quiet` against a live stack. Unit tests alone are insufficient for editor, Phase 2c, PVPO, or Phase 4 integration changes. `--quiet` swallows passing output; only failures surface to stderr. Paste the surfaced failure output (if any) into the PR description.
- The six host-API prompt files tracked here: `judge-adversarial-failure.md`, `generate-variant.md`, `transcript-purpose-guess.md`, `transcript-purpose-classify.md`, `verbalized-eval-awareness.md`, `placement-fix.md` — they feed single-turn API calls with no sandbox error boundary.
- Phase 2c feasibility is strict admission. Do not pass `--skip-feasibility` on shipping runs. Break-glass override is `WORLDSIM_STRICT_FEASIBILITY={true,false}`.
</important>

<important if="you are stuck on Modal image setup, secret wiring, sandbox lifecycle, Claude Code invocation flags, or file routing into a sandbox">
- Modal sandboxes are scoped by which files you `image.add_local_dir`, not by ignore-file patterns. Never rely on ignore patterns for isolation.
- Two predecessor files on this branch are kept as read-only reference for exactly these mechanics:
  - `AgentLab/src/agentlab/benchmarks/redteam/execution.py` — Modal image, secrets, sandbox lifecycle.
  - `AgentLab/src/agentlab/benchmarks/redteam/claude_code.py` — Claude Code invocation flags.
- Read them, retype the equivalent in `worldsim/`. Never `import` from `AgentLab/`. No other file under `AgentLab/` is in scope.
</important>

<important if="you are debugging a live run, choosing auth, or wiring a phase entrypoint">
- Agent evaluation runs locally via Browser Use in an async worker pool with staggered start (`STAGGER_DELAY = 5`).
- Results default to `./logs/`; override with `WORLDSIM_STATE_DIR`. `--resume` reads `logs/pipeline_state.json` and skips completed phases.
- Claude auth precedence is decided by `worldsim/modal_sandbox.py::_build_claude_secrets` (sandbox) and `worldsim/phase_4/anthropic_client.py` (host API). Three modes supported (`CLAUDE_CODE_OAUTH_TOKEN`, `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL`). Never hard-code which one; let the helper decide.
</important>
