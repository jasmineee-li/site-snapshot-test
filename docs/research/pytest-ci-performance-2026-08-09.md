# Pytest and CI test-speed research — 2026-08-09

## Scope and baseline

This note evaluates how to reduce WARP Taskgen feedback time without reducing
the release gate's coverage. The current boundary is intentionally broad:

- [`verify_default.sh`](../../packages/warp-taskgen/scripts/verify_default.sh)
  runs the fast lint/readiness checks and then the complete default pytest
  selection with `-n auto --dist worksteal`.
- [`pyproject.toml`](../../packages/warp-taskgen/pyproject.toml) excludes only
  the explicitly opted-out marks (`integration`, `feasibility`, `preflight`,
  `live_l3`, and `crash_resume`) from that default selection.
- [`accept_taskgen.sh`](../../scripts/accept_taskgen.sh) additionally builds a
  wheel, installs it into a fresh virtual environment, and exercises the
  installed CLI. That packaging smoke is a separate correctness boundary, not
  test-suite overhead that can be removed safely.

The observed acceptance run (GitHub Actions run
[31351035328](https://github.com/jasmineee-li/warp/actions/runs/31351035328),
job 93341775691) collected 3,552 tests, passed 3,548, skipped 4, and took
about 61 seconds in pytest within a 1m39s job. The same pytest command took
about 24 seconds locally; the fast non-pytest check took about 6 seconds. The
large gap is runner capacity and process/subprocess overhead, not dependency
resolution: the CI locked sync was about 3.4 seconds.

## Findings from primary documentation

### 1. Measure before changing the scheduler

Pytest exposes `--durations=N` and `--durations-min=N` to report the slowest
setup/test durations ([pytest API reference](https://docs.pytest.org/en/stable/reference/reference.html#cmdoption-durations)).
Capture both serial and xdist runs, for example:

```text
uv run pytest -q -n 0 --durations=0
uv run pytest -q -n 2 --dist worksteal --durations=20
uv run pytest -q -n 4 --dist worksteal --durations=20
uv run pytest -q -n auto --dist worksteal --durations=20
```

Use several CI runs (median wall time and failure/flake count), not one local
run, before changing a default. A slow-test list is a prioritization signal;
xdist workers overlap those durations, so they must not be summed.

### 2. Benchmark worker count and distribution explicitly

`pytest-xdist`'s `-n auto` uses the machine's physical CPU count; an explicit
number, `PYTEST_XDIST_AUTO_NUM_WORKERS`, or `--maxprocesses` can cap it
([xdist distribution](https://pytest-xdist.readthedocs.io/en/stable/distribution.html#running-tests-across-multiple-cpus)).
Its `worksteal` scheduler is intended for suites with uneven test durations,
while `loadscope`/`loadfile` keep module/file groups together when expensive
fixtures make process reuse valuable (same source).

Therefore benchmark `-n 2`, `-n 4`, and `-n auto` on the actual
`ubuntu-latest` runner before touching the checked-in command. Do not assume
that more workers are faster: this repository has many subprocess-heavy tests,
and each worker pays Python/import and fixture setup costs. Keep
`worksteal` unless measured data shows a different scheduler wins without
introducing ordering or fixture coupling.

### 3. Use marks and focused selection for development, not as a coverage cut

Pytest supports selecting by marker (`-m`) or name expression (`-k`), and
`--lf` reruns the last-failed tests ([pytest selection reference](https://docs.pytest.org/en/stable/reference/reference.html#test-selection)).
The repository already has a safe default mark expression. Use focused node
IDs/`-k` and `--lf` for an edit loop, then run the full default gate before
handoff. Do not make `--lf`, a changed-file heuristic, or a new skip the
required check: those modes can silently omit a regression.

### 4. Keep dependency caching, but do not cache the isolation boundary

The workflow already enables uv's cache and keys it from
`packages/warp-taskgen/uv.lock` ([`taskgen-acceptance.yml`](../../.github/workflows/taskgen-acceptance.yml)).
GitHub recommends caching reusable dependencies or intermediate outputs while
keeping jobs able to regenerate them, and warns never to cache credentials
([dependency-caching reference](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching);
[artifacts versus caching](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflow-artifacts)).
The measured sync is already a small fraction of the job. Do not cache the
fresh acceptance virtualenv or package-local secrets: the fresh install is the
point of the wheel smoke, and cached executable state would weaken that proof.

### 5. Tiering and sharding need a stable required-check contract

If profiling shows a material wall-time win from parallel CI shards, GitHub
Actions matrix jobs can create one job per shard and limit concurrency with
`max-parallel` ([matrix strategies](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/run-job-variations)).
Start with worker-count tuning first: sharding repeats checkout/setup and makes
failure diagnosis more complex. If sharding is later justified, retain one
stable aggregate job named `taskgen-acceptance` and require that aggregate, not
ad-hoc matrix names.

GitHub requires a required check to report on the latest commit and warns that
workflow-level path filters leave required checks pending; a conditionally
skipped job reports success ([required-check troubleshooting](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks)).
The current workflow therefore correctly runs on every pull request while the
root wrapper performs its safe internal no-op routing. Preserve that shape.

## Repository-specific optimization order

1. **P0 — establish a repeatable benchmark.** Record CI medians for
   `-n 2/4/auto`, `worksteal` versus `load`, total test count, and flaky/failing
   node IDs. Upload the duration report as a diagnostic artifact only; do not
   make diagnostics a required second check.
2. **P1 — remove duplicated cheap work.**
   `tests/test_phase_compat_wrappers.py` calls the full readiness audit while
   `verify_fast.sh` has just run the same audit. Share a pure audit result or
   narrow the test to its compatibility-specific assertion without dropping
   the standalone audit.
3. **P1 — reduce process churn at the test seam.**
   `tests/test_remote_job_scripts.py` launches shell/Python/fake-SSH processes
   repeatedly (about 68 subprocess calls in that file); the start script also
   polls process-group readiness in 100 ms increments. Keep a small number of
   end-to-end shell tests, but unit-test argument/metadata logic in-process and
   make fake polling deterministic. Preserve at least one real subprocess test
   for each safety-critical boundary.
4. **P2 — isolate CLI startup cost.** Keep one installed-entrypoint smoke and
   test command serialization/dispatch through an in-process seam where the
   behavior permits it. Do not replace the packaging smoke with unit tests.
5. **P3 — consider matrix sharding only after the above.** Re-measure setup
   duplication, cache pressure, log usability, and required-check behavior
   before adopting it.

## Coverage-preserving acceptance criteria

Any speed change should prove, on the same commit:

- the collected test count is unchanged (or every intentional mark change is
  reviewed explicitly);
- the full default command still passes, including the fresh wheel install and
  CLI smoke;
- serial and parallel runs agree on outcomes for a representative sample;
- no live/infrastructure marker is accidentally promoted into the default
  suite; and
- the single required `taskgen-acceptance` check remains reported on the latest
  commit.

## Sources

- [pytest API reference: duration reporting and test selection](https://docs.pytest.org/en/stable/reference/reference.html)
- [pytest-xdist: running tests across multiple CPUs](https://pytest-xdist.readthedocs.io/en/stable/distribution.html)
- [GitHub dependency caching reference](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching)
- [GitHub artifacts versus caching](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflow-artifacts)
- [GitHub matrix strategies](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/run-job-variations)
- [GitHub required-check troubleshooting](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks)
