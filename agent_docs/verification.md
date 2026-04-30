# Verification Guide

Use this when choosing tests, lint, live gates, or context-efficient command output.

## Context-Efficient Output

Prefer repo wrappers that print one success line and full failure output. Do not manually summarize or truncate verbose logs; deterministic wrappers handle that.

```bash
scripts/lib/run_silent.sh "phase 4 unit tests" \
  "uv run pytest -x tests/test_phase_4_adversarial.py tests/test_phase_4_pvpo_capture.py -q"
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

## Specialized Checks

- Phase 4 fresh-host preflight: `uv run pytest -m preflight tests/preflight -q`
- Crash/resume behavior: `uv run pytest -m crash_resume tests/test_crash_resume_*.py -q`
- WebArena evaluator adapter: `uv sync --directory packages/worldsim-webarena-verified`
- CLI smoke: `uv run python -m worldsim.main --help`
- Readiness metrics: `uv run python scripts/readiness_audit.py --json`
- Phase 4 result audit: `uv run python scripts/summarize_phase_4_results.py logs/<run>`
- Phase 4 variant-generation QA: `uv run python scripts/audit_phase_4_variants.py logs/<run>`
- Phase 4 paired-run comparison: `uv run python scripts/compare_phase_4_runs.py logs/<baseline> logs/<candidate>`
- Secrets policy: `agent_docs/secrets.md`

## Hooks

Claude Code sessions have two local hooks:

- `.claude/hooks/ruff-autofix.sh` runs after Python edits and applies safe Ruff fixes/formatting.
- `.claude/hooks/stop-verify.sh` checks tracked changed Python files on stop, surfacing only failures.

Other agents should emulate the same behavior with `run_silent` rather than streaming full passing test output.
