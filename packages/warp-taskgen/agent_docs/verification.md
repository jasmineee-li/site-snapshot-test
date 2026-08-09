# Verification Guide

Use this when choosing tests, lint, live gates, or context-efficient command output.

## Context-Efficient Output

Prefer repo wrappers that print one success line and full failure output. Do not manually summarize or truncate verbose logs; deterministic wrappers handle that.

```bash
scripts/lib/run_silent.sh "phase 4 unit tests" \
  "uv run pytest -x tests/phase_4 tests/test_phase_4_judge_api.py tests/test_phase_4_variant_api.py tests/test_phase_4_pvpo_capture.py -q"
```

You can also source it inside a longer shell session:

```bash
source scripts/lib/run_silent.sh
run_silent "ruff changed files" "uv run ruff check path/to/file.py"
```

The wrapper uses fail-fast-friendly commands well: prefer `pytest -x` for focused failing suites so the agent fixes one failure at a time.

## Fast Local Checks

For a repo-wide fast baseline, prefer the checked-in wrapper:

```bash
bash scripts/verify_fast.sh
```

It runs scoped Ruff over `worldsim tests scripts`, pytest collection, and the
tracked-file readiness audit through `scripts/lib/run_silent.sh`. Use it before
broader validation and before handing off agent-authored changes.

`bash scripts/verify_fast.sh --skip-collect` is reserved for wrappers that run
pytest immediately afterward and would otherwise pay collection twice.

`logs/` is runtime output, not source. Regenerate Phase artifacts through the
pipeline commands instead of committing files under `logs/`. See
`agent_docs/artifacts.md` for artifact traceability, recovery, and fixture
promotion rules.

Use the narrowest meaningful check first, then broaden before shipping:

```bash
uv run ruff check <changed-python-files>
uv run ruff format <changed-python-files>
uv run pytest -x <relevant-test-files> -q
```

The default pytest config excludes live or expensive markers:

```text
not integration and not feasibility and not preflight and not live_l3 and not crash_resume
```

Default pytest also uses `--strict-markers`; add new markers to
`pyproject.toml` before using them in tests.

Run excluded markers explicitly only when the task requires them.

For normal local shipping, use the default wrapper:

```bash
bash scripts/verify_default.sh
```

It runs the fast Ruff/readiness checks via `verify_fast.sh --skip-collect`,
then runs the default pytest suite with `pytest-xdist`
(`-n auto --dist worksteal`). The wrapper skips the standalone collection check
because the following pytest run collects the same default suite. Use plain
`uv run pytest -q` when debugging order-sensitive failures or when you need a
single-process run.

## Live Integration Gate

For changes touching editors, seeding, Phase 2c, Phase 4, PVPO capture, or any host-API prompt file, use a live stack when available:

```bash
scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r8a.local.yaml --quiet
```

`--quiet` captures passing pytest output and surfaces full output only on failure. If it fails, include the surfaced failure output in the PR or handoff.

Docs-only changes that describe Phase 4 behavior do not require a live stack by
themselves, but they should name the expected runtime evidence. For the default
`eval-awareness-iterator`, verification evidence should show:

- every Phase 4 default-path task has an `eval_awareness_iterator` envelope or
  an explicit iterator skip reason; baseline-complied rows are included, not
  only resistant rows
- the configured sequential rewrite cap (default 3)
- the TP trigger for each attempted iteration, including `None`/failure triggers
- cue diagnosis and rewrite status for each iteration
- rejected-rewrite classes from host payload-contract QA
- for renderer-backed payloads, renderer id, slot-delta status, and host
  render-validation buckets; accepted `markdown_fenced_system` rewrites must
  still contain exactly one ` ```system ` fenced block with required concealed
  tokens inside the block
- browser-evaluated, PVPO-valid, TP-real, VEA-aware, and compliant counts
- selected iteration and stop reason
- unchanged benign task contracts and Phase 3 reward functions across rewrites

For the legacy `3+3+1` adaptive strategy loop, verification evidence should show:

- the configured attack budget (`[3, 3, 1]`, total variant budget 7)
- per-round generated, host-rejected, browser-evaluated, PVPO-valid, and compliant counts
- round lineage for each variant (`root_attempt_id`, `parent_attempt_id`, `round_index`, `round_variant_index`, selected strategy, refusal trigger)
- unchanged benign task contracts and Phase 3 reward functions across variants

For Phase 4 summary/reporting changes, check `agent_docs/phase4-reporting-metrics.md`
and verify that output distinguishes fixed-cohort `Headline ASR` from
PVPO-encounter-conditional `Gate-1 ASR`. Legacy aliases are acceptable only as
compatibility fields or explicitly labeled historical output.
- termination reason (`success`, no actionable lineage, no viable generation, or budget exhausted)

Wrapper facts that matter:

- Without `--host-config`, the default instances file is the local
  `instances.smoke.json`.
- With `--host-config` and no explicit `--instances`, the wrapper generates a
  temporary host-config-specific smoke instances file. This avoids stale local
  ports after remote setup regenerates host-local topology artifacts.
- Pass `--instances` only when intentionally testing a specific topology file.
- `--host-view auto` chooses the right URL view for local vs remote execution;
  use `--host-view orchestrator` on r5 when checking host-local browser paths.
- The wrapper sources `.env` when present and fails loudly when required live
  browser/Playwright dependencies are missing.

## Specialized Checks

- Phase 4 fresh-host preflight: `uv run pytest -m preflight tests/preflight -q`
- Crash/resume behavior: `uv run pytest -m crash_resume tests/test_crash_resume_*.py -q`
- WebArena evaluator adapter: `uv sync --directory packages/warp-taskgen-webarena-verified --locked`
- AgentLab sidecar sync: `uv sync --directory packages/worldsim-agentlab-runner --locked`
- Package Ruff: `uv run ruff check packages/worldsim-agentlab-runner/src packages/warp-taskgen-webarena-verified/src`
- AgentLab/WebArena adapter smoke: `uv run pytest tests/test_agentlab_runner.py tests/phase_4/test_process_pool.py tests/rewards/test_vendor_webarena.py -q`
- AgentLab/trace-inspection observability: `uv run pytest tests/test_agentlab_runner.py tests/phase_4/test_process_pool.py tests/test_phase_4_trace_inspection.py -q`
- CLI smoke: `uv run warp-taskgen --help`
- Readiness metrics: `uv run python scripts/readiness_audit.py --json`
- Phase 4 result audit: `uv run python scripts/summarize_phase_4_results.py logs/<run>`
- Phase 4 variant-generation QA: `uv run python scripts/audit_phase_4_variants.py logs/<run>`; for `3+3+1` runs, confirm attack-budget totals, round lineage, host-finalization rejection buckets, and that only selected payload text changed across variants
- Phase 4 paired-run comparison: `uv run python scripts/compare_phase_4_runs.py logs/<baseline> logs/<candidate>`
- Phase 0c profile provenance/quality audit: `uv run python scripts/audit_phase_0c_profiles.py logs/<run>/phase_0c`
- Secrets policy: `agent_docs/secrets.md`

## Hooks

Claude Code sessions have two local hooks:

- `.claude/hooks/ruff-autofix.sh` runs after Python edits and applies safe Ruff fixes/formatting.
- `.claude/hooks/stop-verify.sh` checks tracked changed Python files on stop, surfacing only failures.

Other agents should emulate the same behavior with `run_silent` rather than streaming full passing test output.
