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

## Larger-gains addendum: CI-level parallelism

The largest remaining wall-time lever is running independent test partitions
on independent runners. `pytest-xdist` parallelizes workers inside one VM;
GitHub Actions matrix jobs parallelize across VMs, without requiring a larger
runner. A larger runner is the other option, but it changes the billing and
runner-availability trade-off described below. The xdist documentation
confirms that `-n auto` uses the physical cores visible to the machine and that
`worksteal` is useful for uneven test durations
([xdist distribution](https://pytest-xdist.readthedocs.io/en/stable/distribution.html#running-tests-across-multiple-cpus)).

### Controlled benchmark evidence and corrected projection

The controlled local comparison for the proposed two-shard shape is:

| Command shape | Wall time |
| --- | ---: |
| Full default suite, `-n 4 --dist worksteal` | **29.86s** |
| Core suite excluding `tests/test_remote_job_scripts.py`, `-n 4 --dist worksteal` | **16.93s** |
| `tests/test_remote_job_scripts.py` alone, `-n 4 --dist worksteal` | **17.80s** |
| Same remote-job file, `-n 4 --dist load` | **16.33s** |

The reproducible forms were:

```text
uv run pytest -q -n 4 --dist worksteal
uv run pytest -q -n 4 --dist worksteal --ignore tests/test_remote_job_scripts.py
uv run pytest -q -n 4 --dist worksteal tests/test_remote_job_scripts.py
uv run pytest -q -n 4 --dist load tests/test_remote_job_scripts.py
```

These are wall times from the same checkout and environment; xdist worker
durations must not be summed. A two-shard CI matrix keeps `-n 4` inside each
independent runner, so its critical pytest path is the slower of the two
partitions, approximately **17.80s** in this controlled local shape. That is
about 40% below the 29.86-second unsplit run. The earlier 31-second `n2`
estimate was the wrong model for CI sharding: it treated the shards as two
workers on one machine, rather than two four-worker runners.

Scaling the observed GitHub pytest phase (about 61s) by that local ratio gives
approximately 36s; allowing for runner differences and imperfect balancing,
use **36–42s as a CI pytest projection**, not a guarantee. With fresh-runner
checkout/setup, the package smoke, and the aggregate job, the full required
check projects to roughly **55–65s versus 99s observed**. Validate that range
with at least five CI runs and report median/p95 before changing the default.

Three shards have no controlled three-way partition evidence yet. They may
reduce the test critical path further, but another checkout/sync, collection,
queue, and imbalance surface can erase the theoretical gain. Do not promise a
40–55s total until a real three-way duration-balanced run demonstrates it.
The `load` result for the remote-job file is useful follow-up evidence, but it
is not enough to replace the repository-wide `worksteal` default; compare
multiple CI runs and fixture behavior before changing the scheduler.

| Shape | Evidence-backed test critical path | Main trade-off | Full-check projection |
| --- | ---: | --- | ---: |
| One standard runner, current CI job | 61s observed in CI | No setup duplication | 99s observed |
| Two standard runners, each `-n 4` | 17.80s local shape; 36–42s CI projection | Two checkouts/syncs and balancing | **55–65s projected** |
| Three standard runners, each `-n 4` | Not measured | More setup, queue, and imbalance risk | Not estimated |
| One 8-core larger runner | At best about 31s by arithmetic only | Metered runner and xdist/process overhead | Requires a real 8-core benchmark |

### Minimal architecture that keeps one required check

Use one matrix job for test shards, one independent job for the package smoke,
and one final aggregate job whose displayed name remains exactly
`taskgen-acceptance`:

```text
taskgen-tests (matrix: shard 0/1[, 2])  ─┐
                                           ├─> taskgen-acceptance (required)
taskgen-package-smoke                    ─┘
```

The core test job should run the default selection with the remote-job file
excluded and `--dist worksteal`; the remote-job job should run that file with
`--dist load`. The package job should run the lint/readiness and
fresh wheel-install/CLI smoke once, but not rerun pytest. Keep the local
`accept_taskgen.sh` default as the sequential full boundary, and factor small
test-only/package-only entry points for CI rather than teaching every matrix
step to duplicate the wrapper's shell logic. This keeps local behavior and the
fresh-install proof unchanged while allowing the two CI branches to overlap.

The aggregate job must use `if: ${{ always() }}` and inspect the matrix and
package results, failing if any required branch failed or was cancelled. GitHub
documents that `needs` normally skips dependants after a failure and that an
`always()` conditional is required when a dependent status must still be
reported ([using jobs](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-jobs),
[required-check troubleshooting](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks)).
Only this aggregate check should be selected as the branch-protection
requirement; matrix names are implementation details that can change when the
shard count changes.

Set matrix `fail-fast: true` so a known failure cancels
remaining shards and shortens feedback. Set `max-parallel` to the chosen shard
count so the intended concurrency is explicit. GitHub's matrix strategy starts
one job per combination, maximizes parallel jobs by default, and supports both
`fail-fast` and `max-parallel` ([matrix strategies](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/run-job-variations),
[workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax#jobsjob_idstrategymatrix)).

### Preserve the current path no-op

Do not add a workflow-level `paths:` filter. The current workflow intentionally
starts on every pull request and lets the root acceptance router no-op when no
canonical Taskgen path changed. GitHub warns that a workflow skipped by path
filtering leaves its required check pending and can block a pull request; a
conditionally skipped job, by contrast, reports success ([required-check
troubleshooting](https://docs.github.com/en/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks)).

For the matrix design, run the existing router immediately after checkout in
each shard/package job, before Python/uv setup. A non-Taskgen change should
return success without syncing dependencies; a Taskgen change should continue
to the selected test or package mode. This preserves the current no-op
contract without introducing a serial route job or relying on a fragile matrix
output. The aggregate job then sees successful no-op branches and still emits
the stable `taskgen-acceptance` check.

### Shard selection and coverage proof

There is no built-in pytest option that partitions a suite across separate
machines. The measured suite already has a natural exact partition: the core
lane runs the default selection with `tests/test_remote_job_scripts.py`
ignored, while the remote-job lane runs exactly that file. Their union is the
default suite and their intersection is empty by construction, without a
selector plugin or duration manifest. Wrapper contract tests pin both commands
so a later edit cannot silently overlap or omit the remote-job file. The
baseline research run selected 3,552 tests; after adding the reviewed tests in
this implementation, validation selected 3,569 tests (3,525 core plus 44
remote-job), with the same 41 marked-out tests. Keep the full parallel default
command available for local verification; the matrix is a distribution
mechanism, not a new test-selection policy.

Avoid uploading one artifact per passing shard merely to aggregate status.
GitHub's artifacts are intended to persist or transfer files between jobs, and
jobs that consume an artifact must wait for the producer and perform the
upload/download ([workflow artifacts](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflow-artifacts)).
Use normal job logs and `GITHUB_STEP_SUMMARY` for pass/fail and duration
information ([workflow commands — job summaries](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands#adding-a-job-summary)).
Upload a small JUnit/duration artifact only on failure or when running a
diagnostic benchmark; keep it out of the required critical path.

### Runner-size alternative

The standard `ubuntu-latest` runner has four CPUs and 16 GB RAM for public
repositories; private-repository standard runners have two CPUs and 8 GB
([GitHub-hosted runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)).
That explains why a larger runner can improve the existing single-job shape:
the same xdist command can use more workers without repeating checkout and
dependency setup. GitHub's larger Linux runners offer 8, 16, and larger CPU
sizes, but they are metered and have possible queue-to-assign delay
([larger-runner reference](https://docs.github.com/en/actions/reference/runners/larger-runners),
[runner pricing](https://docs.github.com/en/enterprise-cloud@latest/billing/reference/actions-runner-pricing)).
For a public repository, standard runners are free while larger runners are
still billed; for a private repository, compare the billed whole-minute cost of
one larger job with the multiplied matrix-job minutes. Do not switch runner
size based on the local Mac result alone—measure `-n 4` on the current runner,
then an 8-core candidate, using the same commit and at least five runs.

### Recommendation

The largest low-risk gain is a **two-shard CI matrix plus a parallel package
smoke and a stable `taskgen-acceptance` aggregate**. It should cut the current
99-second critical path toward the 50–65-second range without weakening the
fresh-wheel boundary. After the test seams in the first optimization order are
fixed, re-measure two versus three shards. Consider an 8-core larger runner
only if the organization already accepts its metered cost; it is simpler YAML,
but it cannot beat well-balanced free matrix jobs by enough to justify paying
for it in this small suite. A 16-core runner or more than three shards is not a
minimal change: process startup, collection, and the slowest shard will become
the limiting factors.

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
- [GitHub Actions jobs and `needs`](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-jobs)
- [GitHub-hosted runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
- [GitHub larger runners reference](https://docs.github.com/en/actions/reference/runners/larger-runners)
- [GitHub Actions runner pricing](https://docs.github.com/en/enterprise-cloud@latest/billing/reference/actions-runner-pricing)
- [GitHub workflow commands and job summaries](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-commands#adding-a-job-summary)
