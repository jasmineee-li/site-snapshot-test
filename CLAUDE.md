# WorldSim v5 Agent Guide

## Scope
WorldSim v5 is a controlled AI safety benchmark for browser-agent robustness in
local/sandboxed WebArena-style environments. Work only on repo code, generated
logs/traces, and configured benchmark infrastructure; do not target real
services, credentials, users, money, or production systems.

Allowed work includes benchmark debugging, trace/log audits, scoring and
readback validation, iterator failure analysis, and measurement-validity fixes.
Do not weaken admission, visibility, scoring, or readback checks. Do not
hand-edit generated logs. Do not design real-world exploit instructions or
optimize for evading safeguards.

## WHAT
- Python 3.12 `uv` project for Modal Sandbox plus Browser Use and AgentLab adversarial evaluation of browser agents.
- Strict WASP scope: GitLab issues/comments and Reddit/Postmill posts/comments only.
- Main map: `worldsim/` core package, `scripts/` host/run tooling, `tests/` unit/live/preflight suites, `packages/worldsim-webarena-verified/` isolated evaluator adapter, `vendors/` gitignored benchmark clones.
- `AgentLab/src/agentlab/benchmarks/redteam/{execution.py,claude_code.py}` are read-only references only; never import from `AgentLab/`.

## WHY
- Research question: do browser agents resist or comply with indirect prompt injections embedded in realistic WebArena environments?
- The orchestrator connects to pre-running benchmark instances supplied by the user (`site_url`, `reset_endpoint`); it does not manage benchmark lifecycles except reset between tasks.
- `docs/worldsim-v5-technical-specifcation.md` is authoritative; the filename typo is intentional. If code and spec diverge, update the spec first when needed, then align code.

## HOW
- Before non-trivial work, read the relevant spec section and one companion doc. Do not preload every `agent_docs/` file.
- `agent_docs/verification.md` — test/lint/live-gate selection and quiet output wrappers.
- `agent_docs/domain-invariants.md` — Phase 2/4, auth, sandbox, prompt, and WASP invariants.
- `agent_docs/code-organization.md` — feature/domain module ownership, compatibility-wrapper policy, and readiness debt sequencing.
- `agent_docs/remote-runs.md` — r5, proxy, fresh-host, and long-run discipline.
- `agent_docs/trace-inspection.md` — Phase 4 trace/result debugging. When asked why tasks complied, resisted, were unaware, or why iterator contrasts failed, start with `uv run python -m worldsim.main trace ...` or `scripts/remote_trace_inspect.sh` before ad hoc JSON dumps.
- Work loop: Research -> Plan -> Implement -> Validate. Keep each context small; use `rg`; delegate focused exploration/verification to sub-agents when the harness supports it.
- Prefer existing patterns and helpers over new abstractions. Keep edits scoped; do not rewrite generated logs or hand-edit `feasibility.status`.
- Treat `logs/` as runtime output; use `agent_docs/artifacts.md` before deleting, restoring, or promoting generated artifacts.
- Main commands: `uv sync --extra dev`, `bash scripts/verify_fast.sh`, `bash scripts/verify_default.sh`, `uv run python -m worldsim.main --help`, `uv run pytest <paths> -q`, `uv run ruff check <paths>`, `uv run ruff format <paths>`.
- Use `scripts/lib/run_silent.sh` or repo quiet wrappers; for live gates use `scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r5.yaml --quiet`. Never truncate verbose output yourself.
- Claude/Modal auth precedence lives in `worldsim/modal_sandbox.py::_build_claude_secrets` and `worldsim/phase_4/anthropic_client.py`; never hard-code one auth mode.
- On resume or context compaction, preserve completed actions, active assumptions, artifact paths/IDs, tool outcomes, unresolved blockers, and the next concrete goal.
- Keep final responses concise by default; use detailed sections only for reviews, handoffs, or when the user asks for depth.
- If an instruction here caused a wrong turn or was missing during a real failure, propose the smallest `AGENTS.md`/`agent_docs/` fix after handling the task.
- Do not generate or serve web applications from this repo.

<important if="you are changing tests, lint, CI hooks, or verification flow">
- Default pytest excludes `integration`, `feasibility`, `preflight`, `live_l3`, and `crash_resume`; opt into those markers deliberately.
- `.claude/hooks/ruff-autofix.sh` formats edited Python files; `.claude/hooks/stop-verify.sh` checks changed tracked Python files only.
- For changes touching editors, seeding, Phase 2c, Phase 4, PVPO, or host-API prompts, unit tests are insufficient; run the live integration wrapper when a stack is available.
</important>

<important if="you are touching Phase 2/4, eligibility, exposure contracts, feasibility, PVPO, judges, TP, VEA, placement-fix, strategy, or rewards">
- Phase 2c is strict admission for Phase 4. Do not ship Phase 4 on an unverified dataset and do not pass `--skip-feasibility` for shipping runs.
- Current scope is GitLab and Reddit/Postmill UGC surfaces only; Magento, Wikipedia, OSM, shopping, and classifieds are out of WASP scope unless the spec changes.
- Phase 4 has two gates: PVPO encounter (`max_coverage` in `[0,1]`; `0` routes to placement-fix) and attack effectiveness.
- Transcript Purpose and VEA are observational in baseline trajectory scoring. The explicit `eval-awareness-iterator` variant system may branch on Transcript Purpose only for bounded rewrite control; VEA remains report-only and must never gate admission, scoring, or branching.
- Refusal judge, variant generator, Transcript Purpose, VEA, placement-fix, eval-awareness cue diagnosis, and eval-awareness rewrite use direct host Anthropic Messages API calls, not `run_claude_in_sandbox`.
- Do not reintroduce `visual_concealment`; paper concealment is behavioral final-response steering, not visual.
- Phase 4 keeps benign task contracts and reward functions emitted by Phase 3 invariant across variants. The default post-resistance path is `eval-awareness-iterator`; legacy strategy variation is opt-in via `--phase-4-variant-system strategy-variation`.
</important>

<important if="you are changing Modal sandbox setup, secret wiring, sandbox lifecycle, or file routing">
- Modal sandboxes are scoped by explicit `image.add_local_file` / `image.add_local_dir` inclusion, not ignore patterns.
- Read `agent_docs/secrets.md` before changing tracked credentials or instance/proxy token handling.
- Read the two AgentLab reference files if needed, then retype equivalent behavior in `worldsim/`; runtime imports from `AgentLab/` are forbidden.
</important>

<important if="you are setting up hosts, deploying the proxy, or preparing rigor runs">
- Read `agent_docs/remote-runs.md` and `docs/handoffs/rigor-run-setup.md` first.
- On r5, instance files are execution-locality contracts: Modal Phase 0/0c uses externally reachable/proxied `instances.smoke.json`, while on-host Phase 2c/4 uses `instances.scale.json`. Do not generalize one file across both localities.
- Use `scripts/setup_phase4_on_host.sh` on fresh hosts; its preflight proves storage state, evaluator venv, and benchmark connectivity.
- Proxy source of truth is `scripts/deploy_benchmark_proxy.sh`; never hand-edit `/etc/nginx/conf.d/worldsim-proxy.conf`.
- Rigor PVPO uses `page-surface-stable` capture on the runner-owned browser; dedicated beginFrame Chrome containers are legacy/parity infrastructure, not the default run path.
</important>
