# Verification Guide

Use this when selecting tests, lint, live gates, or compact command output.
Unless a command says otherwise, run it from `packages/warp-taskgen/`.

## Output and local checks

Use `scripts/lib/run_silent.sh` or an existing quiet wrapper. It prints one
success line and full failure output; preserve the failure instead of manually
truncating it. Prefer fail-fast focused tests before broad suites:

```bash
scripts/lib/run_silent.sh "focused tests" \
  "uv run pytest -x <relevant-test-files> -q"
uv run ruff check <changed-python-files>
uv run ruff format <changed-python-files>
```

The fast baseline is `bash scripts/verify_fast.sh`; normal local shipping is
`bash scripts/verify_default.sh`. The latter runs the fast readiness checks and
the default pytest suite with xdist. `--skip-collect` is for wrappers that run
pytest immediately afterward.

Default pytest uses `--strict-markers` and excludes
`integration`, `feasibility`, `preflight`, `live_l3`, and `crash_resume`. Add
new markers to `pyproject.toml` and opt into excluded suites deliberately.
`logs/` is runtime output; apply `agent_docs/artifacts.md` before treating it as
source or a fixture.

## Live gate

Changes to editors, seeding, Phase 2c, Phase 4, PVPO capture, or host-API
prompts use a live stack when available:

```bash
scripts/run_integration_tests.sh \
  --host-config configs/benchmark_hosts/r8a.local.yaml --quiet
```

Docs-only guidance changes do not require a live stack. They do require a
truth check against the current spec, source, and command help. The setup
preflight command is `uv run pytest -m preflight tests/preflight -q`; its
evidence is storage-state presence and evaluator-venv resolution, not live task
connectivity.

## Phase 4 evidence

For `eval-awareness-iterator`, verify the run summary covers every default-path
PVPO-valid baseline, including compliant rows, with an iterator envelope or an
explicit skip reason. Record the configured sequential cap (default 3), TP
trigger, cue diagnosis/rewrite status, host contract-QA rejection buckets,
browser-evaluated/PVPO-valid/TP-real/VEA-aware/compliant counts, selected
iteration, stop reason, and unchanged benign task/reward contracts. Use
`agent_docs/phase4-contracts.md` for gate and branch semantics and
`phase4-reporting-metrics.md` for labels.

For legacy `strategy-variation`, record the `3+3+1` budget, per-round
generated/host-rejected/browser-evaluated/PVPO-valid/compliant counts, lineage,
termination reason, and unchanged benign contracts. Do not fold observational
iterator or action fields into final ASR.

## Specialized checks

- CLI: `uv run warp-taskgen --help`
- Readiness: `uv run python scripts/readiness_audit.py --json`
- Fresh-host gate: `uv run pytest -m preflight tests/preflight -q`
- Crash/resume: `uv run pytest -m crash_resume tests/test_crash_resume_*.py -q`
- WebArena adapter: `uv sync --directory packages/warp-taskgen-webarena-verified --locked`
- AgentLab sidecar: `uv sync --directory packages/worldsim-agentlab-runner --locked`
- Adapter smoke: `uv run pytest tests/test_agentlab_runner.py tests/phase_4/test_process_pool.py tests/rewards/test_vendor_webarena.py -q`
- Trace observability: `uv run pytest tests/test_agentlab_runner.py tests/phase_4/test_process_pool.py tests/test_phase_4_trace_inspection.py -q`
- Phase 0c audit: `uv run python scripts/audit_phase_0c_profiles.py logs/<run>/phase_0c`
- Phase 4 summary: `uv run python scripts/summarize_phase_4_results.py logs/<run>`
- Variant QA: `uv run python scripts/audit_phase_4_variants.py logs/<run>`
- Paired runs: `uv run python scripts/compare_phase_4_runs.py logs/<baseline> logs/<candidate>`

## Hooks and completion

Claude sessions use `.claude/hooks/ruff-autofix.sh` after Python edits and
`.claude/hooks/stop-verify.sh` for changed tracked Python. Other agents use
quiet wrappers to provide equivalent evidence.

Completion means the narrowest relevant checks passed, excluded/live suites
were opted into when required, and the handoff names commands, evidence paths,
and any unresolved infrastructure blocker.
