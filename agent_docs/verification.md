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

Run excluded markers explicitly only when the task requires them.

For normal local shipping, use the default wrapper:

```bash
bash scripts/verify_default.sh
```

It runs the fast baseline and then the default pytest suite.

## Live Integration Gate

For changes touching editors, seeding, Phase 2c, Phase 4, PVPO capture, or any host-API prompt file, use a live stack when available:

```bash
scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r5.yaml --quiet
```

`--quiet` captures passing pytest output and surfaces full output only on failure. If it fails, include the surfaced failure output in the PR or handoff.

Docs-only changes that describe Phase 4 behavior do not require a live stack by
themselves, but they should name the expected runtime evidence. For the default
`eval-awareness-iterator`, verification evidence should show:

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
- termination reason (`success`, no actionable lineage, no viable generation, or budget exhausted)

Wrapper facts that matter:

- Default instances file is `instances.smoke.json`; pass `--instances` when
  testing a different topology.
- `--host-view auto` chooses the right URL view for local vs remote execution;
  use `--host-view orchestrator` on r5 when checking host-local browser paths.
- The wrapper sources `.env` when present and fails loudly when required live
  browser/Playwright dependencies are missing.

## Specialized Checks

- Phase 4 fresh-host preflight: `uv run pytest -m preflight tests/preflight -q`
- Crash/resume behavior: `uv run pytest -m crash_resume tests/test_crash_resume_*.py -q`
- WebArena evaluator adapter: `uv sync --directory packages/worldsim-webarena-verified`
- CLI smoke: `uv run python -m worldsim.main --help`
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
