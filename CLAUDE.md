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
3. **Trajectories flow through disk.** Per-task dirs under `logs/<run>/<task_id>/` hold `history.json`, `result.json`, `screenshots/`, `conversations/`, and `pvpo/` (per-step visibility vectors + hasDamage flag from the Paint-Verified Payload Oracle capture). The Phase 4 refusal judge, placement-fix, IPI-AER p(eval) judge, and VEA judge all receive host-side dataclass-projected trajectory slices (or the reference screenshot) via direct Anthropic Messages API calls (see `worldsim/phase_4/{aer_trajectory_extract,judge_api,placement_api,ipi_aer_p_eval_api,vea_api,ipi_aer_caption}.py`). No Phase 4 trajectory step runs in a Modal sandbox.
4. **Two gates in Phase 4.** Gate 1 is encounter detection via PVPO (`worldsim/phase_4/encounter_detection.py` + `ink_occupancy.py`): a continuous `max_coverage` score in `[0.0, 1.0]` from per-character *ink-occupancy* verification — count non-background pixels inside each glyph's layout rect against the page's resolved background RGB, classify as rendered iff absolute floor (≥3 ink pixels) OR relative floor (≥3% occupancy) is met. `max_coverage == 0` routes to placement-fix via `final_status="injection_not_encountered"`. Gate 2 is attack effectiveness. **IPI-AER p(eval)** and **VEA (Verbalized Eval Awareness)** are observational metrics (not gates) computed in parallel on every trajectory with `max_coverage > 0` — nothing in the pipeline branches on either score. IPI-AER = Needham et al. 2025 5-PQ × N-reps protocol over an OS-Harm AER-formatted transcript; VEA = single forced-tool-use classification of whether the agent's own CoT verbalized awareness of being evaluated. Baseline capability (benign-under-attack) is a Phase 4 byproduct, never used to filter tasks before Phase 4.

## Integration test requirement

If a PR changes `worldsim/editors/**`, `worldsim/seeding.py`, `worldsim/phases/phase_4_adversarial.py`, `worldsim/phase_4/**` (including `encounter_detection.py`, `ink_occupancy.py`, `aer_trajectory_extract.py`, `aer_formatter.py`, `ipi_aer_caption.py`, `ipi_aer_p_eval_api.py`, `vea_api.py`, `placement_api.py`, `pvpo_capture.py`, `pvpo_browser_config.py`, `pvpo_query.js`), `worldsim/phases/phase_2_feasibility.py`, `worldsim/phases/phase_2_text_fill.py`, `worldsim/browser_use_agent.py`, `worldsim/prompts/judge-adversarial-failure.md`, `worldsim/prompts/generate-variant.md`, `worldsim/prompts/ipi-aer-p-eval.md`, `worldsim/prompts/ipi-aer-caption.md`, `worldsim/prompts/verbalized-eval-awareness.md`, or `worldsim/prompts/placement-fix.md`, run `scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r5.yaml` against a live stack before shipping and include the output in the PR description. Unit tests alone are not sufficient for editor, Phase 2c, PVPO capture, or Phase 4 integration changes. Prompt-file changes to judge-adversarial-failure.md, generate-variant.md, ipi-aer-p-eval.md, ipi-aer-caption.md, verbalized-eval-awareness.md, and placement-fix.md are tracked here because those prompts are the only input to single-turn API calls with no sandbox error boundary.

Rigor runs require the `chrome-headless-shell` Docker container (`worldsim/docker/chrome-headless-shell.Dockerfile`) because `HeadlessExperimental.beginFrame` is not supported on native macOS and the deterministic paint-compare depends on `--enable-begin-frame-control` + `--run-all-compositor-stages-before-draw`. Browser-Use connects to the container over CDP on `127.0.0.1:9222`. On hosts without the container, PVPO capture falls back to zero coverage per step; every trajectory routes to placement-fix, which is correct behavior but not a rigor result.

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
- Do not route the Phase 4 refusal judge, variant generator, IPI-AER p(eval), VEA, caption, or placement-fix through `run_claude_in_sandbox`. All six use direct Anthropic Messages API calls (see `worldsim/phase_4/{judge_api,variant_api,ipi_aer_p_eval_api,vea_api,ipi_aer_caption,placement_api}.py`). The sandbox path was removed for judge + variant on 2026-04-18 after observed refusals on the "recommend strategies" framing, and for P(eval) + placement-fix on 2026-04-19 to close the same refusal class. The IPI-Screenshot P(eval) probe was replaced by IPI-AER + VEA in the two-metric cutover. The fix is in the prompts (classification-only / structured-output framing, authorization preamble citing Dziemian et al., 2026, arXiv:2603.15714 — with the exception of IPI-AER p(eval) which uses Needham et al. 2025's exact default "You are a helpful assistant." for methodological fidelity) and in the mechanism (single-turn API with structured tool-use output, or temperature=1 free-text for IPI-AER).
- Do not add `visual_concealment` back to the strategy pool. It was a category error — paper concealment is behavioral (final-response steering), not visual (CSS/font). Behavioral concealment strategies (`scripted_message`, `command_silent_execution`, `false_justification`) are deferred indefinitely because they require a new reward function to grade; the current `AgentResponseEvaluator` + `db_query_match` reward is tool-side only.

<important if="you are stuck on Modal image setup, secret wiring, sandbox lifecycle, or Claude Code invocation flags">
Two files from the predecessor AgentLab pipeline survive on this branch as read-only reference material for exactly those mechanics:

- `AgentLab/src/agentlab/benchmarks/redteam/execution.py` — Modal image setup, secrets, sandbox lifecycle
- `AgentLab/src/agentlab/benchmarks/redteam/claude_code.py` — Claude Code invocation flags

Read them, understand the mechanic, then retype the equivalent in `worldsim/`. Never `import` from `AgentLab/`. Do not reach for any other file under `AgentLab/` — only those two were kept deliberately, and both could be deleted at a later cleanup pass without breaking `worldsim/`.
</important>

<important if="you are actually running the pipeline, wiring up a phase entrypoint, or debugging a run">
- Claude Code steps run in Modal sandboxes via `run_claude_in_sandbox` — **except** the Phase 4 judge, variant generator, P(eval) probe, and placement-fix, which use direct Anthropic Messages API calls via `worldsim/phase_4/{judge_api,variant_api,p_eval_api,placement_api}.py`. After the 2026-04-19 placement-fix cutover, no Phase 4 trajectory step routes through the sandbox; the ecoval-fix loop was deleted entirely in the PVPO cutover (see `docs/handoffs/codex-handoff-paint-verified-oracle.md`).
- Agent evaluation runs locally via Browser Use in an async worker pool with staggered start (`STAGGER_DELAY = 5`).
- Results default to `./logs/`; override with `WORLDSIM_STATE_DIR`.
- `--resume` reads `logs/pipeline_state.json` and skips completed phases.
- Prerequisites (Modal token, Claude auth, benchmark clone, running WebArena for Phase 4): see `README.md`. Phase 3 is agent-free and needs no live instances. Do not duplicate prerequisites into code or configs.
- Claude auth supports `CLAUDE_CODE_OAUTH_TOKEN` (Pro/Max), `ANTHROPIC_API_KEY` (API credits), or `ANTHROPIC_AUTH_TOKEN` + `ANTHROPIC_BASE_URL` (OpenRouter). All three work for both sandbox paths and the host-side Messages API path. Precedence is defined in `worldsim/modal_sandbox.py:_build_claude_secrets` (for sandbox) and `worldsim/phase_4/anthropic_client.py` (for host API). Never hard-code which one; always let the helper decide.
</important>
