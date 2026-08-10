# Test-suite speed deep dive — 2026-08-10

This report records the evidence gathered for the `warp-taskgen` suite in the
`test-suite-speed-research` Codex worktree at commit `3449fc9a`. Three Luna Max
research agents independently measured the suite, audited test value, and
checked recommendations against primary sources. No test or product code was
changed.

## Executive answer

The suite is not mainly slow because it has 3,570 tests. Most of those tests
are cheap. It is slow because a small number of tests perform work that unit
tests should not pay for:

1. the default acceptance path scans and AST-parses the whole repository in
   `verify_fast.sh`, then a pytest test repeats the same scan;
2. fake retry tests wait for real production backoffs (3 s, 1 s, and 1 s);
3. shell/CLI variants repeatedly cross process boundaries—129 subprocess call
   sites across the tests, with the densest concentration in the 2,435-line
   remote-job test module; and
4. xdist repeats full collection and worker startup, so extra workers quickly
   stop helping.

The first controlled A/B result is already large. With four workers and
`--dist worksteal`, the full default selection reported 20.76 s. Deselecting
only the duplicate scan and three tests that pay real sleeps reported 15.25 s:
**5.51 s faster, 26.5% less pytest time, or 1.36x throughput**. The optimized
version should keep three behavioral tests and inject a fake sleep; only the
duplicate full-repository audit should leave the default pytest lane. Because
this is one local A/B pair and the second run had a warm filesystem/import
cache, it is a strong prioritization result rather than a release promise.

The recommended end state is:

- a 12–16 s full local pytest target on the current four-worker machine;
- a 1–3 s normal edit loop using focused node/file selection;
- one readiness scan per acceptance run, never two;
- a small, explicit set of subprocess/browser/package boundary tests; and
- in-process tests for the much larger argument, metadata, routing, and state
  matrices.

Do not mass-delete cheap unit tests. Delete the proven exact duplicates and
source-shape assertions, rewrite artificial waits, and deepen the process seam
before reducing the number of shell end-to-end cases.

## Suite boundaries: there is no healthy monorepo-wide `pytest`

The repository contains more than the canonical Taskgen suite:

- **Taskgen** is the owned shipping boundary discussed in this report. Its
  canonical command is `scripts/accept_taskgen.sh`, and the package guide
  explicitly owns its 3,574-test default selection.
- **AgentLab** is a separate upstream/reference tree with its own
  `AgentLab/pytest.ini` and Make target (`pytest -n 5 -m "not pricy" tests/`).
  The Taskgen guide treats this tree as read-only reference material.
- **Evaluation-awareness experiments** contain research scripts and a few test
  modules, but are not part of Taskgen acceptance.

A root recursive pytest invocation is therefore both undefined and currently
unsafe. AgentLab discovery attempted 87 selected tests but failed during
collection because `AgentLab/tests/test_openrouter.py` performs a live
OpenRouter request at module import, while a tutorial `redteam_test.py` is
accidentally discoverable outside `tests/`. Even after explicitly restricting
to `AgentLab/tests/` and ignoring the live script, collection took about
17.05 s because importing Ray/BrowserGym is expensive.

Do not fold AgentLab into Taskgen just to create one impressive “all tests”
number. Define explicit root orchestration commands for Taskgen, AgentLab, and
evaluation research. In a dedicated AgentLab change, rename the OpenRouter file
to a smoke/example module or guard it behind a `pricy`/live test function so
collection never performs network I/O; restrict discovery to `tests/`; and
rename tutorial files that are examples rather than tests.

## What is in scope

The package is Python 3.12 + `pytest`/`pytest-asyncio`/`pytest-xdist`, with
Playwright used by Phase 2c code and a large number of subprocess-oriented
shell/CLI tests. The package config in
[`packages/warp-taskgen/pyproject.toml`](../../packages/warp-taskgen/pyproject.toml)
sets `asyncio_mode = "auto"`, discovers `tests`, and excludes only the
`integration`, `feasibility`, `preflight`, `live_l3`, and `crash_resume` marks
from the default invocation.

At this checkout:

- `find tests -name 'test_*.py'` reports 207 test modules (220 Python files
  including helpers). Pytest collected 3,615 tests, deselected 41 by marker,
  and ran 3,574: 3,570 passed and 4 skipped.
- Serial trials reported 54.75–74.06 s as other work varied on the machine; a
  detailed run reported 60.22 s. Collection itself reported 1.62–2.34 s
  (roughly 2.5–3.3 s process wall time).
- Existing repository measurements show the default xdist path is already much
  faster (the previous baseline was 26.28 s and the optimized path was
  17.60 s). The current fresh-worktree four-worker run reported 20.76 s; use
  repeated medians rather than treating either result as permanent.

Scheduler trials reinforce that conclusion rather than selecting a new
scheduler:

| Shape | Pytest-reported time | Interpretation |
| --- | ---: | --- |
| `-n 1 --dist load` | 54.75 s | Little parallelism; useful as an aggregate-work diagnostic. |
| `-n 2 --dist load` | 43.77 s | Some gain, still process/wait heavy. |
| `-n 4 --dist load` | 31.78 s in one busy trial; 12.41 s core-only when idle | Highly sensitive to contention and suite skew. |
| `-n 8 --dist load` | 23.10 s | More workers did not beat four-worker `worksteal`. |
| `-n 4 --dist worksteal` | 20.76 s full; 10.86–12.79 s core-only | Current default remains defensible. |
| `-n auto --dist worksteal` | 23.31 s plus one flaky timeout assertion | More processes increased risk without a demonstrated win. |

The current CI-style split is already effective locally: the 3,530-test core
lane reported 10.86–12.79 s of pytest time on repeated idle/contended runs,
while the 44-test remote-job lane reported 4.07–4.47 s. The core lane is now
the critical path. Do not add more shards until its internal waste is removed.

## Measured local hotspots and likely causes

The detailed serial run attributes most of the aggregate work to seven files:

| File | Aggregate serial call time | Main cost |
| --- | ---: | --- |
| `test_remote_job_scripts.py` (44 tests) | 16.97 s | Shell, Python, and fake-SSH process churn. |
| `test_phase_2_feasibility.py` (90 tests) | 9.52 s | Real retry sleeps and orchestration. |
| `test_r8a_control_plane_iac.py` (13 tests) | 4.91 s | Subprocess AWS shims. |
| `test_host_lifecycle_scripts.py` (10 tests) | 3.46 s | Subprocess AWS/shell contracts. |
| `test_generate_compose_scale.py` (14 tests) | 2.21 s | Generator subprocesses. |
| `test_auth_mechanism.py` | 2.08 s | Broad auth orchestration matrix. |
| `test_phase_0c_verify_http.py` (3 tests) | 2.00 s | Real local HTTP process/server boundary. |

Those seven files consume about 41.15 s of serial call time. The slowest
individual entries explain the easiest first change:

| Node | Call time | Why it matters |
| --- | ---: | --- |
| `tests/test_phase_compat_wrappers.py::test_tracked_source_has_no_legacy_phase_imports` | 2.73–4.74 s | Calls `scripts.readiness_audit.build_audit()`, which runs `git ls-files`, reads 814 tracked files, and AST-parses 593 code files. `verify_fast.sh` already runs the same audit before the suite. |
| `tests/test_phase_2_feasibility.py::test_render_check_failing_marks_render_unverified` | 3.00 s | The production retry path sleeps for `_RENDER_UNVERIFIED_RETRY_DELAY_S == 3.0` before a second render check. The fake-behavior test does not replace that sleep. |
| `tests/test_phase_2_feasibility.py::test_case_04_retry_after_request_failed_succeeds` | 1.16 s | Pays a real one-second backoff even though the editor is fake. |
| `tests/test_phase_2_feasibility.py::test_case_05_retry_exhausted_yields_request_failed` | 1.16 s | Pays the same real backoff before asserting retry exhaustion. |

The suite has 68 `subprocess.run`/`Popen`/`check_*` call sites in
`tests/test_remote_job_scripts.py`, 14 in `test_generate_compose_scale.py`, 11
each in `test_r8a_control_plane_iac.py` and `test_host_bootstrap_scripts.py`,
10 in `test_host_lifecycle_scripts.py`, and 9 in
`test_reclassify_phase_4_results_cli.py`. These are valuable process-boundary
contracts, but running every argument/error variant through a fresh shell or
Python process is a high-cost way to test pure argument and metadata logic.
Keep a small representative subprocess smoke for each safety-critical
boundary, and move serialization, validation, routing, and state-transition
cases behind an in-process function seam.

The tests contain 211 explicit `asyncio.run()` call sites. Repeated loop
creation may contribute, but it did not appear among the dominant measured
costs. Do not mechanically rewrite them. Prefer native `pytest-asyncio` tests
when the behavior under test is already async, while keeping explicit
loop-boundary tests for synchronous public entry points.

The shared test fixtures are mostly lazy and deliberately fake external
providers. The `patched_anthropic_client` fixture in
[`tests/conftest.py`](../../packages/warp-taskgen/tests/conftest.py) dynamically
imports every `worldsim.phase_4` module that binds `get_client` the first time a
worker needs the fixture. This is a good safety guard against accidental live
API calls, but each xdist worker performs its own imports. Keep the guard, but
consider a static module list or a package-level discovery cache if profiling
shows import discovery in the worker startup budget. Do not make this fixture
autouse: it is currently requested only by tests that exercise the provider
seam.

## Primary-source evidence: pytest and pytest-xdist

### Measure before changing selection or scheduling

Pytest documents `--durations=N` and `--durations-min` for reporting slowest
test durations, and supports node IDs, `-k`, and marker expressions for focused
runs. The command-line usage page is the owning source:
[`How to invoke pytest`](https://docs.pytest.org/en/stable/how-to/usage.html#profiling-test-execution-duration).
Use a repeatable benchmark matrix, for example:

```text
uv run --extra dev pytest -q -n 0 --durations=80
uv run --extra dev pytest -q -n 2 --dist worksteal --durations=20
uv run --extra dev pytest -q -n 4 --dist worksteal --durations=20
uv run --extra dev pytest -q -n auto --dist worksteal --durations=20
```

Record median and p95 wall time across several CI runs, plus failures and
flake/retry counts. A serial slow-test sum is not a parallel wall-time model.

### Worker count, scheduler, and fixture reuse

The xdist distribution reference states that `-n auto` uses the machine's
physical CPU count, that `--maxprocesses` or
`PYTEST_XDIST_AUTO_NUM_WORKERS` can cap it, and that `-n 0` disables xdist. It
also defines the scheduler trade-offs:

- `loadscope` keeps module/class groups together, which is useful when an
  expensive module/class fixture can be reused.
- `loadfile` keeps a whole file in one worker.
- `worksteal` handles significantly different test durations while retaining
  similar or better fixture reuse than `load`.

Source: [`pytest-xdist distribution`](https://pytest-xdist.readthedocs.io/en/stable/distribution.html).

The current default `--dist worksteal` is therefore a defensible choice for
this uneven suite. Benchmark `loadscope`/`loadfile` only for files with proven
expensive module fixtures; grouping all files can reduce load balance. Benchmark
worker counts on the actual CI runner before increasing `auto`: each worker is
an OS process with Python import and fixture startup costs.

xdist's own execution model explains that every worker performs a *full test
collection* and then runs a subset of tests. Source:
[`How it works`](https://pytest-xdist.readthedocs.io/en/stable/how-it-works.html).
This is the key caveat for “more workers”: repository-wide collection,
`conftest` imports, and provider fixture discovery are repeated per worker.
Reducing import/collection work, or avoiding unnecessary workers on a
small/mostly-fast shard, can beat blindly raising `-n`.

The xdist how-to page further states that high-scope fixtures are executed once
per worker, not once globally, and shows a lock-file pattern when an expensive
session fixture truly must be produced once. Source:
[`Making session-scoped fixtures execute only once`](https://pytest-xdist.readthedocs.io/en/stable/how-to.html#making-session-scoped-fixtures-execute-only-once).
Use this only for immutable, shareable artifacts (for example a generated JSON
fixture). Do not share mutable browser/auth/database state across workers;
derive a worker/run-specific path instead.

The xdist limitations page requires all workers to collect the same test order
and amount, especially for parametrization. Source:
[`Known limitations`](https://pytest-xdist.readthedocs.io/en/stable/known-limitations.html#order-and-amount-of-test-must-be-consistent).
Any custom changed-file selector, generated shard manifest, or set-based
parametrization must be deterministic and have a union/intersection coverage
test. A selector that silently drops tests is not an acceptable speed fix.

### Fixture scope and autouse boundaries

Pytest's fixture explanation says fixtures are activated by explicit names and
can be reused at function, class, module, package, or session scope; it also
advises minimizing unnecessary fixture dependencies. Source:
[`About fixtures`](https://docs.pytest.org/en/latest/explanation/fixtures.html).
The fixture reference explains that higher scopes run before lower scopes and
that autouse fixtures are applied to every test in their visibility scope.
Source:
[`Fixture instantiation order`](https://docs.pytest.org/en/7.1.x/reference/fixtures.html#fixture-instantiation-order).

Application to this repository:

1. Keep the default suite's provider/browser fakes explicit and lazy. A fixture
   that only one test needs should not become autouse in a root `conftest.py`.
2. Promote immutable, expensive setup to module/session scope only after proving
   it is safe to share. Under xdist, remember that the scope is per worker.
3. For live/integration tests, keep browser/database setup in a separate marked
   lane. The integration `conftest.py` already has a session proxy fixture and
   an autouse token-cache clear; because those tests are excluded by default,
   they should not be paid by unit workers.

### Development selection is not a coverage cut

Pytest owns `-k`, `-m`, node IDs, `--lf`/`--last-failed`, and `--ff`/`--failed-first`.
Sources: [`How to invoke pytest`](https://docs.pytest.org/en/stable/how-to/usage.html#specifying-which-tests-to-run)
and [`Rerun failed tests`](https://docs.pytest.org/en/7.1.x/how-to/cache.html).
Use these for the edit loop and keep the full default gate for handoff. Do not
replace the required check with `--lf`, a changed-file heuristic, or a
permanent `-k` expression: all can omit an unrelated regression.

### Assertion rewriting is a measured, opt-in trade-off

Pytest documents that the default `rewrite` mode rewrites test asserts at import
time for failure introspection and that `--assert=plain` disables that behavior.
Source: [`Assertion introspection`](https://docs.pytest.org/en/7.1.x/how-to/assert.html#assertion-introspection-details).
Benchmarking `--assert=plain` can quantify import/collection savings, but do not
make it the default without a failure-diagnostics review. It gives up the rich
assertion explanation that makes a large suite maintainable; any gain is likely
smaller than the process/fixture and CI-sharding opportunities above.

## Primary-source evidence: Python process and async boundaries

Python's `subprocess` reference states that `Popen` can use `os.posix_spawn()`
in some cases “for better performance,” but process creation is still a real
OS boundary. It also documents that `shell=True` explicitly invokes a shell and
that `communicate()` is the safe way to drain pipes. Source:
[`subprocess`](https://docs.python.org/3/library/subprocess.html#popen-objects).
Use the fastest safe form for tests (`Popen` with an argument list, no shell,
and one representative process-boundary assertion), but the bigger gain here
is reducing the number of process launches rather than micro-optimizing each
launch.

The async subprocess reference says `create_subprocess_exec`/`shell` create
real child processes and that `asyncio.gather` can monitor multiple child
processes concurrently; it also warns that direct stream reads can deadlock
and recommends `communicate()`. Source:
[`asyncio subprocesses`](https://docs.python.org/3/library/asyncio-subprocess.html).
If a production path genuinely launches independent commands, test bounded
concurrency with one async integration test; unit-test command construction and
state transitions without launching a child for every case.

Python's multiprocessing reference records that `spawn` starts a fresh Python
interpreter and is slower than `fork`/`forkserver`, while `forkserver` can preload
modules to avoid repeated import work. Source:
[`multiprocessing contexts and start methods`](https://docs.python.org/3.13/library/multiprocessing.html#contexts-and-start-methods).
This matters to `phase_4` process-pool tests: do not change the start method just
for speed without checking Playwright/thread safety and cross-platform behavior.
If profiling proves process startup dominates, reuse a long-lived pool or preload
safe, immutable modules; preserve a smoke test for the selected start method.

For measurement, Python documents `perf_counter()` as the highest-resolution
clock for short durations and notes that it includes sleep; `process_time()`
excludes sleep. Source: [`time.perf_counter`](https://docs.python.org/3.13/library/time.html#time.perf_counter).
Use wall-clock `perf_counter` for suite/test latency and process CPU time only to
separate scheduler/IO waits from Python CPU work.

## Primary-source evidence: browser reuse and isolation

The official Playwright isolation guide says browser contexts are isolated,
“fast and cheap to create,” and can run independently inside one browser.
Source: [`Playwright browser contexts`](https://playwright.dev/docs/browser-contexts).
The Python Browser API documents `browser.new_context()` as creating an isolated
context and explicitly shows closing the context before closing the browser.
Source: [`Playwright Python Browser.new_context`](https://playwright.dev/python/docs/api/class-browser#browser-new-context).

Application to Phase 2c: the current implementation launches a Playwright/
Chromium pair per verification task for fault containment. That is a reasonable
production isolation decision, but it is the dominant cost of any live
feasibility lane. Do not share a browser across tests merely to speed unit tests;
the unit suite should keep fake browser objects. For a separately marked live
benchmark lane, measure a worker-scoped browser plus per-task fresh contexts
against per-task browser launches. The safer candidate is one browser per worker
with a new context/page per task, explicit context cleanup, and a worker-specific
failure/restart policy—not a reused page or mutable context.

The Playwright test-runner documentation gives the same scope pattern: browser
fixtures are shared to optimize resources, contexts/pages are isolated per test,
and worker-scoped fixtures are created once per worker. Sources:
[`Playwright fixtures`](https://playwright.dev/docs/test-fixtures) and
[`Playwright parallelism`](https://playwright.dev/docs/test-parallel).
Those docs also warn that parallel tests cannot share global state and recommend
worker-specific data or IDs. Preserve that isolation in any browser reuse.

## Primary-source evidence: what to retain versus cut

Google's first-party Testing Blog defines small tests as isolated code/module
tests, medium tests as neighboring-component interaction tests, and large tests
as real user scenarios. Source: [`How Google Tests Software — Part Five`](https://testing.googleblog.com/2011/03/how-google-tests-software-part-five.html).
Its later qualification guidance recommends a solid unit base, integration tests
with fewer dependencies, and end-to-end tests for Critical User Journeys (CUJs),
not exhaustive end-to-end duplication. Source: [`How Much Testing is Enough?`](https://testing.googleblog.com/2021/06/how-much-testing-is-enough.html).

Use that as a review rubric, not a mandatory 70/20/10 quota:

- Keep behavior tests for pure validators, serializers, state machines, and
  error taxonomies. They are cheap and local.
- Keep one contract/integration test at each genuine seam: installed CLI/wheel
  invocation, process-group lifecycle, a representative proxy/browser
  interaction, and the critical Phase 2c render/reachability path.
- Convert redundant variants that only assert private helper call order,
  exact internal mock call counts, or duplicated compatibility wrappers into
  parameterized behavior tests or delete them after checking mutation/coverage
  evidence. A mock call count is worth retaining only when the count itself is a
  product/resource contract (for example “must not make a second paid API call”).
- Never delete a test solely because it is slow. First name the behavior it
  protects, identify the lower-level test that covers the same behavior, and
  retain a boundary test if removing it would leave an untested integration.

## Test-value audit: cut narrowly, not indiscriminately

### Delete or remove from the default lane now

These have high-confidence overlap or no production-behavior signal:

- Remove `test_tracked_source_has_no_legacy_phase_imports` from default pytest.
  The acceptance wrapper already runs the complete audit, while focused scanner
  fixture tests remain in `test_readiness_audit.py`.
- Delete one of the byte-for-byte empty-array tests at
  `test_sandbox_validator.py:1821-1823` and `:2480-2482`.
- Delete or differentiate one of the byte-for-byte Phase 0c retry tests at
  `test_phase_0_recon.py:1437-1469` and `:1517-1549`.
- Delete the callable-import smoke at
  `phase_2/target_resolution/test_reconstruction.py:215-218`; the following
  tests import and invoke the same entry point.
- Delete `test_sandbox_validator.py:3073-3079`, which compares two identical
  hard-coded tuples and calls no production code.
- Merge/remove the identity-to-constant assertions at
  `test_phase_4_options.py:16-18`; retain parser valid/invalid choice behavior.

These deletions are small in count. That is intentional: deleting hundreds of
cheap behavior tests would buy little and create broad regression risk.

### Rewrite behavior, then delete implementation-shape assertions

- `test_phase_2_render_check.py:43-47` inspects function source for two
  JavaScript substrings. Replace it with one JavaScript/browser behavior test
  that proves a `comment_` anchor is found, then remove the source assertion.
- `test_phase_4_pvpo_gate.py:282-294` inspects a private function's source and
  signature. The placement-loop suite already exercises outcomes; express any
  missing trigger rule as input/output behavior, then delete introspection.
- `test_generate_compose_scale.py:403-406` pins a literal shell command.
  Exercise the wrapper with a fake generator or rely on the generated output
  contract instead.
- `test_readiness_audit.py:131-136` reads `verify_fast.sh` and asserts exact
  flags. The acceptance gate itself is the meaningful behavior; exact command
  spelling is not.
- `test_sandbox_validator.py:3024-3048` manually recreates one side of a merge
  and therefore cannot detect sandbox drift. Extract a shared immutable-field
  seam and test both real callers, or delete the false parity claim.
- Collapse the duplicated PVPO generated-JavaScript shape checks at
  `test_phase_4_pvpo_capture.py:54-110` into one generator contract plus one
  executed JavaScript/browser behavior test.
- Remove the standalone exact retry-constant test at
  `test_phase_2_feasibility.py:3777-3779` after the behavioral fake-sleeper
  test asserts the requested delay. The test should protect retry behavior, not
  merely restate a private constant.

### Rewrite for speed while keeping protection

- Inject a sleeper/clock into the render retry and generic retry helper. Keep
  success, final-miss, retry-success, and retry-exhaustion assertions; assert
  requested delays without actually waiting.
- Extract remote-job command validation, metadata construction, and rejection
  taxonomy into an importable module. Keep real shell tests for secret stdin,
  sync exclusions, one start/resume path, status/tail, process-group/zombie
  handling, and child-launch failure. Move the remaining command-form matrix
  in-process.
- Do the same for R8a/host lifecycle shared guards and compose generation.
  `test_generate_compose_scale.py` launches a new Python process for nearly
  every output case even though most assertions concern JSON/YAML, not the CLI.
- Parameterize identical assertion shapes in the 22 suffix/site/budget cases in
  `test_phase_2_text_fill.py:1567-1765` and the status/URL mapping clusters.
  Keep every distinct input boundary; this is mainly maintainability and
  collection cleanup, not a headline runtime win.

### Keep

Keep the fast process-group/zombie unit tests, Needham golden/upstream byte
compatibility, outcome-taxonomy matrices, auth/security contracts, and one real
AWS/shell/browser/package test per boundary. A slow unique safety test belongs
in an explicit marked lane if necessary; it should not be deleted for speed.

The decision rule is: a test earns its cost when it protects an externally
observable behavior, security invariant, schema, compatibility boundary, or
critical seam. Private source text, identical constants, and duplicated setup
without a distinct outcome do not.

## Implementation plan and forecast

### Phase 0 — make the benchmark trustworthy (half day)

Add one quiet benchmark command that records commit, selected count, serial
`--durations`, four-worker wall time, collection time, failures, and flakes.
Run five repetitions on an otherwise idle machine and report median/p95. Keep
the 3,574 selected count (minus explicitly reviewed deletions) as a guard.

Why first: the measurements here demonstrate real machine contention, cold
cache, and scheduler variance. A single fastest run is not a baseline.

### Phase 1 — remove duplicate work and fake time (half to one day)

1. Run the repository-wide readiness audit exactly once in acceptance.
2. Replace the 3 s render wait and two 1 s retry backoffs with an injected fake
   sleeper, while asserting the delay requests.
3. Apply the high-confidence duplicate/source-shape removals above.
4. Re-run the exact A/B matrix and flake check.

Evidence-backed target: approximately 15–16 s pytest time with four workers.
The controlled deselection experiment already reported 15.25 s versus 20.76 s.
Only one behavioral test needs to leave the lane; the delayed tests remain.

### Phase 2 — deepen the process seams (two to four days)

Start with the 16.97 s serial remote-job file, then the 4.91 s R8a, 3.46 s host
lifecycle, and 2.21 s compose-generator files. Make shell scripts thin adapters
over deterministic validation/serialization functions. For every seam:

1. enumerate unique externally visible predicates;
2. retain one real process test for each safety-critical boundary;
3. move combinatorial input variants in-process;
4. prove shell and pure seams agree on representative inputs; and
5. compare test count, behavior coverage, wall time, and flakes before merging.

Forecast, not promise: 10–13 s for the full four-worker Taskgen pytest suite is
plausible after Phase 2. A sub-10-second **full** gate on four cores is not yet
evidence-backed; it likely requires deeper import/process work or more hardware.

### Phase 3 — optimize collection only after call-time waste (one to two days)

Collection is a real 2–3 s floor and xdist repeats it per worker, but import
profiling found no single dominant dependency. Keep root fixtures lazy and
non-autouse, consolidate obvious table-shaped tests, and benchmark any
`--assert=plain` or import-mode experiment against the loss in diagnostics.
Do not prioritize assertion rewriting or a wholesale asyncio conversion.

Retest `-n 2`, `-n 4`, and `-n auto` with `worksteal` and `load` using
medians. Keep `worksteal` unless repeated data wins elsewhere; one `auto`
trial already exposed a timeout flake with no speed gain.

### Phase 4 — make the normal edit loop 1–3 seconds

Use node/file selection, `-k`, and `--lf` while editing, with feature-local
commands documented in the nearest guide. Changed-test mapping or testmon-style
impact analysis may be advisory, but never the shipping gate: dynamic imports,
configuration, prompts, shell scripts, and data contracts make an unsound
selector especially risky here.

The full default suite remains the pre-handoff gate. The already-implemented
CI split and package-proof lane remain the release boundary.

### Phase 5 — stop adding CI machinery

The remote lane is now about 4–5 s locally while core is about 11–13 s. Another
matrix shard would duplicate collection/setup for a small theoretical gain.
Revisit sharding only after Phases 1–3 produce new measured balance. Preserve
the stable aggregate required check and exact shard-union proof documented in
the earlier performance report.

## Acceptance criteria for any speed PR

- Default selection changes only for the named, reviewed deletions.
- Every removed test has an explicit overlapping behavior test or is proven to
  call no production behavior.
- Serial and parallel outcomes agree; five repeated four-worker runs report
  median/p95 and flakes.
- No live/infrastructure marker is silently promoted or excluded.
- The full readiness, fresh wheel install, CLI smoke, and required aggregate
  check still pass.
- The report records actual before/after wall time, not a projection.

## Implemented result

The completed vertical slice kept the default marker boundary and removed nine
selected pytest nodes after value review. The removed nodes were one duplicate
full-repository readiness scan, one private retry-constant assertion, five
manual/duplicate sandbox-validator tests, one exact Phase 0c duplicate, and one
import-only smoke already covered by the following behavioral test. The new
remote-job decision matrix replaces the same number of repeated shell variants,
so it does not reduce coverage through node-count hiding.

Direct four-worker A/B runs on the same checkout and machine were:

- untouched `main`: 3,570 passed, 4 skipped in 17.78 s;
- optimized worktree (initial three-run sample): 3,561 passed, 4 skipped in
  13.37, 13.62, and 14.87 s;
- optimized median: 13.62 s, 4.16 s or 23.4% faster than the direct baseline.

The final repeatability sample used
`./.venv/bin/pytest -q -n 4 --dist worksteal` five times. Every run reported
3,562 passed and 4 skipped, with wall times of 12.42, 12.18, 11.64, 11.78, and
12.10 s and zero failures or flakes. Sorted, the median is 12.10 s. Linear
interpolation at rank `1 + (n - 1) * 0.95` gives a p95 of 12.37 s. A serial
parity run with `./.venv/bin/pytest -q -n 0` reported the same selected
outcomes—3,562 passed, 4 skipped, and 41 deselected—in 32.41 s. The one-node
increase from the initial sample is the added regression for the plain-topology
remote-job decision boundary.

The earlier 20.76 s baseline remains useful as the original symptom, but it was
captured under different resource contention and is not used for the final
percentage. On the optimized tree, `-n auto` took 15.69 s; the measured
four-worker median was 13.62 s, so the shipping wrappers now use four workers.

Focused results explain the wall-time change:

- Phase 2c feasibility: 8.96 s to 4.06 s by recording requested retry delays
  instead of sleeping;
- compatibility/readiness pytest: roughly 3–5 s to 0.08 s, with the full scan
  retained in `verify_fast.sh`;
- Phase 0c HTTP boundaries: 2.06 s to 0.10 s by reducing test-server shutdown
  polling;
- sandbox validator: 1.17 s to 0.22 s after removing false/manual parity and
  duplicate coverage;
- target reconstruction: 1.35 s to 0.08 s after moving placeholders to the
  target-resolution owner instead of importing the whole Phase 2 runner;
- Phase 0d auth: 1.95 s to 0.37 s through branch-local imports;
- site-lock concurrency: 0.34 s to 0.03 s with events instead of timed sleeps;
- compose generation: about 2.21 s to 0.32 s by calling `main(argv)` in-process
  for 13 matrix cases while keeping one CLI subprocess smoke;
- remote-job serial A/B: 13.15 s to 12.29 s; repeated four-worker results were
  effectively neutral, so no large parallel speedup is claimed.

R8a and host-lifecycle extraction experiments were fully reverted. They added
interface surface and measured slower than the original shell boundaries. This
is the stopping rule in practice: keep a seam only when it improves locality or
speed without weakening the contract, and require a measured runtime gain when
speed is its reason to exist.
