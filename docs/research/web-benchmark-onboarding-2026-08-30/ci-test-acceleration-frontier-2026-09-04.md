# CI/test acceleration frontier — 2026-09-04

**Research date:** 2026-09-04 (America/New_York)
**Repository under review:** WARP, checkout `docs/ci-agent-devex` in the
`web-benchmark-onboarding-2026-08-30` worktree
**Scope:** read-only CI and test-architecture research. No source, workflow,
dependency, paper, or infrastructure files were changed.

This note combines measurements already recorded for WARP Taskgen with current
first-party documentation. “WARP evidence” means a measurement or behavior
observed in the repository and linked to the local note or source. “General
guidance” means a recommendation supported by the cited first-party source but
not yet measured on WARP. Vendor documentation is used for mechanics and
trade-offs, not as a performance claim about WARP.

## Executive decision

The current shape has the right safety architecture: a route-aware workflow,
separate test and package-proof work, and one stable required-check aggregator.
Its present performance is not close to the older local baseline, however.
Seven consecutive Taskgen pull-request runs on 2026-09-04 had a median 154 s
workflow duration; the core pytest step alone had a median 126 s and is the
current critical path. The high-leverage sequence is therefore: expose current
test durations once, remove duplicated repository scans only if the trace shows
they matter, then compare the current core lane with one duration-balanced
two-way core split. This is a
bounded diagnosis, not an open-ended matrix experiment.

| Decision | Evidence | Recommendation |
| --- | --- | --- |
| Preserve the remote lane and aggregate; measure one two-way split of the current core lane | **Current WARP evidence:** across seven consecutive 2026-09-04 PR runs, median steps were 126 s core, 16 s remote, and 37 s package proof. The older 10.86–12.79 s local core result predates substantial suite growth and is historical, not the current baseline. | Add duration visibility, remove duplicate scans only when the trace shows material time, then compare exactly one explicit feature-oriented split with the current lane. Keep it only if the required-check median moves materially and selected-test parity holds. |
| Keep `-n 4 --dist worksteal` inside each candidate core job | **Historical WARP evidence plus current runner fact:** it beat `-n 8` and `-n auto` in the recorded comparison, and this public repository's `ubuntu-latest` runner is documented as four CPU. [Study, lines 100–107](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/docs/research/test-suite-speed-deep-dive-2026-08-10.md:100) [GitHub-hosted runners](https://docs.github.com/en/actions/reference/runners/github-hosted-runners) | Do not spend runs re-testing a large scheduler grid unless current duration data contradicts the earlier result. |
| Keep lock-keyed uv caching and `uv sync --locked` | **WARP evidence:** both acceptance lanes use `astral-sh/setup-uv@v6` with the Taskgen lockfile and the runner uses locked sync. [Workflow, lines 43–52](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/.github/workflows/taskgen-acceptance.yml:43) [Acceptance script, lines 114–121](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/scripts/accept_taskgen.sh:114) | Do not turn the environment, credentials, browser state, or generated evidence into a cache. |
| Reuse a wheel only through an explicit artifact when there are multiple consumers | **WARP evidence:** package proof currently builds and installs a fresh wheel/sdist once per package-proof job. [Acceptance script, lines 123–170](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/scripts/accept_taskgen.sh:123) | Build once and upload/download only when build duplication is measured; keep the fresh installed-package proof. |
| Preserve real boundary/replay tests | **WARP evidence:** package proof exercises installed CLI, resources, sdist omission, and upgrade behavior. [Acceptance script, lines 130–195](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/scripts/accept_taskgen.sh:130) | Move combinatorial logic in-process or behind deterministic replay; do not replace output-contract boundaries with a second mock implementation. |

## What is in the current WARP gate

The Taskgen workflow triggers on pull requests and pushes to `main`, uses a
ref-scoped concurrency group with `cancel-in-progress: true`, and fans out a
matrix containing `core-tests` and `remote-job-tests` with `max-parallel: 2` and
`fail-fast: true`. A separate `taskgen-package-proof` job checks the built
distribution, and `taskgen-acceptance` is an `always()` aggregate that requires
both jobs to report success. [Taskgen workflow, lines 1–105](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/.github/workflows/taskgen-acceptance.yml:1)

Routing happens after checkout and before Python/uv setup. The route is based on
the changed-file set and skips only when there is no canonical Taskgen change;
an unknown base revision falls back to running the full acceptance boundary.
[Acceptance router, lines 75–121](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/scripts/accept_taskgen.sh:75)
The root workflow intentionally has no top-level `paths:` filter. [Root gate,
lines 1–17](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/.github/workflows/check-root.yml:1)

The package's default pytest configuration discovers `tests`, runs with strict
markers, and deselects the `integration`, `feasibility`, `preflight`, `live_l3`,
and `crash_resume` marker groups. The lock currently records pytest 9.0.3,
pytest-asyncio 1.3.0, and pytest-xdist 3.8.0;
these versions, rather than an unconstrained fresh resolve, define a comparable
benchmark. [Project configuration, lines 128–138](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/pyproject.toml:128)
[Locked versions](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/packages/warp-taskgen/uv.lock:5207)

## WARP measurements and hypotheses

### Current hosted baseline — 2026-09-04

The seven consecutive Taskgen pull-request runs for PRs #248–#254 are the
current baseline. Their median workflow duration was 154 s. Median measured
step durations were 126 s for `core-tests`, 16 s for `remote-job-tests`, and
37 s for `package-proof`; the core range was 91–129 s. Over the corresponding
seven runs, the separate Root gates workflow had about a 111 s median elapsed
time and its `check-root` job had a 106 s median. It is not the present critical
path, but it will become one if core pytest falls below it.

One representative core log separates the 123 s lane into about 1.3 s for
locked uv synchronization and about 122 s for pytest. Checkout cost was about
10 s. Dependency caching and environment creation are therefore not credible
primary explanations for current latency. The workflow's `run_silent` wrapper
discards successful pytest output, so exact current node durations and expanded
test counts are not available from these runs. That is an explicit measurement
gap, not permission to guess a slow file.

Static inspection at `origin/main` `d7d9a33e` finds 297 `test_*.py` modules
containing about 139,752 lines; including Python test-support files gives 333
files and about 143,769 lines. An AST inventory finds 4,380 syntactic test
definitions before parameter expansion and 125 syntactic `subprocess.run`,
`Popen`, or `call` launch sites. This inventory does not show how many execute
in the default lane or make them a measured bottleneck. Two default-lane tests
each call the full `readiness_audit.build_audit()`. The
package-proof lane already runs the same canonical audit with all required
failure categories. Those two pytest repository walks are concrete duplicate
work; their semantic assertions remain required, and exact current savings
still require the duration-bearing run.

The Phase 2 retry sleepers identified in the August study have already been
replaced by the feature-owned `phase_2c_retry_sleep` seam in current source and
the tests inject a zero-time recorder. They are no longer an implementation
target. This is why the dated study cannot be copied into a September plan
without rechecking source.

### Historical comparison, not current baseline

The August local study collected 3,615 tests, deselected 41 by marker, and ran
3,574. Its four-worker `worksteal` run was 20.76 s. It also reported that
`-n 4 --dist worksteal` beat `-n 8` and `-n auto`, with an `auto` timeout.
Those results still support avoiding an unbounded worker experiment, but the
suite has since grown materially and the timings do not describe current CI.
[Study, lines 84–107](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/docs/research/test-suite-speed-deep-dive-2026-08-10.md:84)

An older CI study observed a 61 s pytest phase inside a 1m39s acceptance job,
versus about 24 s locally; locked sync was only about 3.4 s. That is useful
runner/critical-path evidence, but it is tied to its recorded run and should not
be presented as today's baseline. [CI study, lines 19–25](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/docs/research/pytest-ci-performance-2026-08-09.md:19)

### Benchmark protocol before a CI change

**WARP evidence + general guidance:** use the seven existing hosted runs for
job-level variance. On the first CI source slice, expose current collection,
selected/passed/skipped counts, pytest wall time, and the slowest nodes from the
ordinary core command. This requires the CI lane to bypass or narrowly extend
`run_silent`, because that wrapper currently discards successful output. Do not
add a second test runner. Then compare only the current configuration and one
candidate partition:

```text
uv run pytest -q -n 4 --dist worksteal --durations=20
```

Pytest documents `--durations=N` and `--durations-min` for finding slow tests,
and node IDs, `-k`, markers, and `--collect-only` for controlled selection.
[Pytest usage — profiling and selection](https://docs.pytest.org/en/stable/how-to/usage.html)
One diagnostic run is enough to locate candidates. Two focused candidate runs
are enough to reject a clear non-winner; final confidence can accumulate over
the next ordinary PRs rather than blocking development on a 25-run scheduler
grid. Report median and range until enough organic runs exist for a meaningful
p95. Keep pytest-reported worker time separate from job wall time: xdist worker
durations are not additive, and the matrix critical path is the slowest
concurrent job plus setup/aggregation.

For visibility without making logs noisy, publish a short selected-count and
duration table to `$GITHUB_STEP_SUMMARY`, and upload full logs/JUnit only on
failure. GitHub documents job summaries via `$GITHUB_STEP_SUMMARY` and their
per-step size limit. [Workflow commands](https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/workflow-commands-for-github-actions)

## Recommendations

### 1. Fan out only when the saved critical path pays for duplicated setup

**General guidance, applied to a WARP hypothesis.** GitHub's matrix strategy
creates one job per matrix combination; matrix jobs run independently, while
`max-parallel` caps concurrency and `fail-fast` can cancel queued/in-progress
siblings when one fails. [Workflow syntax](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)

The current two-lane matrix duplicates checkout, route evaluation, Python setup,
and uv setup. In the measured run those costs were roughly 12 s while core
pytest was roughly 122 s, so a shared “setup” job would optimize the wrong
component and serialize the matrix behind an artifact transfer. A second core
job can be worthwhile because it exchanges about 12 s of parallel setup for a
chance to cut the 126 s median core step nearly in half. Keep the existing
remote lane separate.

Use this decision rule:

1. Expose current slow-node and selected-count output from the real core lane.
2. If the trace shows the two duplicate default-lane repository walks consume
   material time, replace them with focused inputs to the same semantic
   assertions. Otherwise retain them. Do not delete either assertion set; keep
   the standalone production readiness gate and its focused unit tests.
3. Define one explicit, feature-oriented core partition from measured node
   durations. Do not hash node IDs, introduce a duration manifest, or add a
   sharding dependency.
4. Model the critical path as `max(lane wall times) + aggregate`, not the sum,
   and compare selected-test union/intersection as part of the route tests.
5. Keep the stable `taskgen-acceptance` aggregate name and explicit result checks.

The acceptance threshold should be material: after the split, target a median
`taskgen-acceptance` workflow below 90 s, with no candidate core shard more than
20% slower than its sibling and no selected-test loss. If the explicit split
cannot meet that after two focused repairs, keep the simpler lane and use the
duration data to remove the actual slow boundaries instead.

### 2. Cache dependencies; do not cache a mutable test environment

**General guidance, with WARP evidence.** The workflow already enables uv's
cache and keys it from `packages/warp-taskgen/uv.lock`. [Taskgen workflow,
lines 43–48](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/.github/workflows/taskgen-acceptance.yml:43)
Keep `uv sync --locked` as the correctness boundary: uv says `--locked` refuses
to modify or silently refresh a stale lock, while normal `uv run` may lock/sync
automatically. [uv project sync](https://docs.astral.sh/uv/concepts/projects/sync/)

GitHub's dependency-cache contract is exact-key restore followed by optional
partial `restore-keys`; a successful job creates a new immutable key. [Dependency
caching](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching)
If a manual key is introduced, include OS/architecture, Python ABI, and the
lockfile hash, and avoid one unique key per shard unless the measured hit rate
justifies it. Do not cache `.venv`, credentials, browser profiles, live-service
state, generated benchmark outputs, or failure logs as if they were dependency
inputs.

uv documents that its cache contains prebuilt and source-built wheels and that
CI caching is common. It also notes that source-built wheel reuse is often more
valuable than retaining prebuilt wheels, which can be downloaded again. [uv
cache](https://docs.astral.sh/uv/concepts/cache/)
If CI invokes `uv cache prune --ci`, make that policy explicit and measure cold
and warm runs. The `setup-uv` release page records v9.0.0 (2026-07-21) changing
the default `prune-cache` behavior to reduce PyPI load; pin the action and set
the intended behavior rather than inheriting a moving default. [setup-uv
releases](https://github.com/astral-sh/setup-uv/releases)

The GitHub Actions cache service now rate-limits cache uploads to 200 per minute
per repository (downloads are unaffected). [Actions cache rate limit changelog,
2026-01-16](https://github.blog/changelog/2026-01-16-rate-limiting-for-actions-cache-entries/)
Avoid a cache-key explosion if the matrix grows. A cache miss or a rate-limited
upload must still leave a correct locked run.

#### Prebuilt versus locked environments

**General guidance, with WARP evidence:** use the hosted runner's supported
Python tool installation plus the committed uv lockfile as the portable
environment contract. GitHub recommends `actions/setup-python` for a consistent
Python version and notes that hosted runners include a tool cache. [Building and
testing Python](https://docs.github.com/en/actions/tutorials/build-and-test-code/python)
uv's `--locked` mode then makes resolution failure explicit instead of silently
repairing the environment. [uv project sync](https://docs.astral.sh/uv/concepts/projects/sync/)

Treat a prebuilt virtualenv as an optimization experiment, not as the source of
truth. A venv can encode runner paths, Python ABI, native-library state, and
stale editable installs; restoring it across images or Python versions can make
tests pass for the wrong reason. If a benchmark demonstrates that a same-image,
same-ABI venv artifact is worthwhile, key it by OS/architecture/Python/lock hash,
verify the interpreter and package metadata, and retain a cold locked-sync lane
periodically. The safer default is a lockfile plus uv's dependency/wheel cache,
with a wheel artifact only for a package that has multiple downstream
consumers.

### 3. Route work internally while preserving required checks

**WARP evidence:** the Taskgen workflow runs on every pull request and uses a
route-only step before dependency setup; the root workflow deliberately has no
`paths:` filter. [Taskgen workflow, lines 28–52](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/.github/workflows/taskgen-acceptance.yml:28)
[Root gate, lines 1–6](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/.github/workflows/check-root.yml:1)

**General guidance:** GitHub warns that a workflow skipped by a branch/path
filter leaves its associated check pending; a required check can then block a
pull request. [Workflow syntax — path filters](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)
Keep top-level triggers broad and make the route decision inside the workflow.
For a no-op change, every lane should exit successfully after route-only, and a
stable aggregate should still run and report success. For an unknown base SHA,
the safe fallback is the current full run.

Do not add a selection manifest for this work. Route tests can use a readable
fixture list of changed paths and assert `run`/`skip` results. Never let each
xdist worker independently invent a different subset: xdist requires every
worker to collect the same test IDs in the same order, and the controller
verifies that invariant. [xdist how it works](https://pytest-xdist.readthedocs.io/en/stable/how-it-works.html)
[xdist known limitations](https://pytest-xdist.readthedocs.io/en/stable/known-limitations.html)

### 4. Keep the current xdist scheduler unless the duration trace disproves it

**WARP evidence:** the recorded `-n4 --dist worksteal` result beat `-n8` and
`-n auto`; the remote-job-only lane was measured separately and uses `--dist
load`. Keep those choices unless the current duration trace reveals a concrete
fixture- or scheduler-shaped long tail.
[Study, lines 100–112](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/docs/research/test-suite-speed-deep-dive-2026-08-10.md:100)

**General guidance:** xdist's `load` scheduler assigns work as workers finish;
`worksteal` can rebalance substantially different test durations. `loadscope`
and `loadfile` keep module/class or file groups together, which can reduce
expensive fixture duplication but can increase skew. [xdist distribution](https://pytest-xdist.readthedocs.io/en/stable/distribution.html)
Every worker still performs collection and startup, so more workers can lose to
process/fixture overhead. If current evidence implicates scheduling, compare the
present command with one scheduler chosen for that measured shape; do not run a
general scheduler grid.

Do not infer a win from serial duration sums. Use wall time, p95, fixture setup
cost, selected-count parity, and flake rate. xdist 3.8.0 (released 2025-06-30)
also changed loadscope reordering controls; a lockfile update should be treated
as a benchmark input, not an incidental dependency refresh. [xdist changelog](https://pytest-xdist.readthedocs.io/en/latest/changelog.html)

### 5. Split slow suites by measured workload, not file count

**Current WARP evidence:** the existing split isolates remote-job process tests,
but the remaining core step now has a 126 s median. Current source has already
injected the old feasibility retry clock, while two whole-repository audit
calls remain in the default pytest selection. The first optimization is those
known duplicates; everything else requires current `--durations` evidence.

Preserve process-boundary cases that protect shell, package, browser, import
order, or safety behavior. If a duration report identifies repeated process
variants, retain one real boundary case and move only pure
argument/metadata/serialization combinations behind an existing in-process
owner. Do not replace Python import-order checks or installed-wheel checks with
a test-only simulator.

The historical study's two-shard arithmetic is a useful hypothesis, not a target:
it estimated a 17.80 s local critical path and 36–42 s CI pytest projection, but
had no controlled three-way measurement. [CI study, lines 155–173](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/docs/research/pytest-ci-performance-2026-08-09.md:155)
Do not add three-way fanout until a real matrix demonstrates a lower p95 after
setup and queue time.

### 6. Reuse a built wheel through an artifact, not a cache

**General guidance, applied to the current package proof.** Python packaging
distinguishes source distributions from wheels; wheels contain install-ready
files and avoid a build step at install time. [PyPA packaging flow (last updated
2026-09-01)](https://packaging.python.org/en/latest/flow/)

PyPA's GitHub Actions publishing guide deliberately separates build and publish
jobs: the build job stores distributions as artifacts, and later jobs download
those exact artifacts instead of rebuilding. [PyPA publishing guide (last
updated 2026-09-01)](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/)
GitHub artifacts are persisted job outputs intended for sharing files between
jobs and are distinct from dependency caches. [Workflow artifacts](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflow-artifacts)

If future WARP jobs consume the same wheel, use a dedicated build step that:

1. builds from the checked-out commit;
2. records the commit, package version, filename, and SHA-256;
3. uploads the wheel (and sdist if a consumer needs it) as an immutable artifact;
4. downloads and verifies it in consumers; and
5. keeps at least one fresh isolated install/CLI/resource/upgrade proof.

Artifact v4 is documented as immutable and faster than prior artifact versions;
uploads with the same name should be merged deliberately rather than silently
overwritten. [Artifact v4 changelog, 2023-12-14](https://github.blog/changelog/2023-12-14-github-actions-artifacts-v4-is-now-generally-available/)
Do not use the uv dependency cache as evidence that a wheel was built from this
commit. Current package proof already performs the necessary fresh wheel, sdist,
sidecar, and upgrade checks; reuse is an option only when multiple consumers
make its transfer cost worthwhile.

### 7. Keep fail-fast and cancellation scoped to safe work

**WARP evidence:** the acceptance matrix uses `fail-fast: true`, and the
workflow uses ref-scoped `cancel-in-progress: true`; the final aggregate is
`always()` and checks both result values. [Taskgen workflow, lines 9–11 and
84–105](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/.github/workflows/taskgen-acceptance.yml:9)

**General guidance:** fail-fast is appropriate for independent PR validation
where a sibling result cannot rescue a failed check. Concurrency cancellation is
appropriate for superseded commits, because GitHub cancels the running and
queued work in the group. [Concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency)
Do not copy that policy to a job that creates external resources, performs a
reset-sensitive migration, or must always emit teardown evidence. WARP's normal
default lane excludes live markers; live/sandbox jobs need their own lifecycle
and cleanup decision.

On failures, retain bounded source-test diagnostics without weakening the gate:
the ordinary pytest output or JUnit and package metadata are sufficient. Do not
add a selected-node manifest. The aggregate continues to fail on any
non-success lane; summaries are for humans, while its status remains the
machine-readable contract.

## Output-contract validation: test the real production boundary

This is a correctness boundary, not merely a speed preference.

**General guidance:** pytest's monkeypatch documentation shows patching network
clients and other external dependencies as a way to isolate a unit, while
warning that broad patches can interfere with pytest or third-party internals.
[pytest monkeypatch](https://docs.pytest.org/en/stable/how-to/monkeypatch.html)
Use explicit dependency injection or a narrow transport fake for failure and
retry branches. Do not implement a “test parser,” “test compiler,” or alternate
serializer that reproduces the production output and then compare it with
itself; such tests can pass while the real output contract drifts.

Phase 1 novel-task generation currently has two production generation
boundaries. The direct contract-bound route calls the configured client,
receives the provider SDK's tool-use content, and sends it through
`_extract_slots` and the real compiler. The sandbox-backed route consumes the
returned
`/workspace/output/benign_tasks.json`, parses it, applies the family compiler,
and runs generic validation before caching. Representative GitLab and
Rocket.Chat Phase 1 pipeline tests replace only `run_claude_in_sandbox`; they
still execute the production JSON parsing, family compiler, validator,
metadata, and cache behavior. That is a narrow transport substitute, not a
second implementation of the product. Other lower-level tests use fabricated
provider-shaped objects or patch narrower compiler/validator seams, so none of
them establishes the genuine provider SDK boundary.

No committed Phase 1 `emit_action_task_slots` response or sandbox provider
envelope is suitable for these two boundaries. A tracked Phase 4 judge fixture
is provider-shaped, but it owns a different contract. The retained Run 15
artifact contains compiled tasks and metadata, not a raw Phase 1 provider
envelope, so manufacturing a response from it would not prove the missing
boundary. A new valid/invalid response pair for every family and a new
no-network CLI would mostly duplicate already-covered feature behavior while
creating seven new test-owned contracts.

The smallest honest ladder is therefore:

1. **Keep the feature tests:** retain the current narrow transport substitution
   and production extraction/compiler/validator calls for deterministic branch
   and family behavior.
2. **Do not add a no-provider rehearsal command or seven response fixtures.**
   They do not establish SDK or provider compatibility and add another surface
   to maintain.
3. **When the exact frozen provider route is available, run two one-row
   production micro-canaries with no external Site or browser mutation:** call
   `generate_new_tasks_for_site` twice with card slicing disabled. Use a
   host-action-only card and capability profile that require the direct backend
   for one call; this reaches `generate_contract_bound_action_tasks_api` and
   then the top-level final validation/cache path. Use a model-owned,
   non-host-action-only card for the other so it must reach
   `run_claude_in_sandbox`. Use a disposable output/state root; these calls do
   incur provider usage, create generation files, and may create sandbox work.
   Before dispatch, assert and record only the non-secret effective route:
   expected OpenRouter base host, expected frozen model, required auth variable
   present, and higher-precedence OAuth variable absent. Never print the token.
   Retain only the current compiled output, cost summary, and telemetry where
   the owning backend already produces it, under the existing run policy. Do
   not claim that current entry points retain a raw provider envelope.
4. **Promote one sanitized genuine replay only after a recurring boundary
   regression demonstrates the need.** Capturing that response would require a
   deliberately reviewed, sanitized recorder at the owning provider seam. The
   replay must instantiate the actual locked SDK/sandbox response shape and
   feed the untouched production extractor, compiler, and validator; it must
   not add a test parser, compiler, service, or generic fixture hierarchy.

A separate Anthropic credential can cheaply test that provider's SDK and
transport shape, but it cannot establish compatibility with the frozen
OpenRouter route. Label it as transport evidence if used; do not substitute it
for the exact-route canary. Neither micro-canary proves generated-task quality,
admission, Site behavior, exposure, or grading. Those claims remain with their
existing later phases and sandbox checks.

OpenAI's deployment-simulation report describes the changing-state problem with
real external tools and uses recorded tool-call/response pairs to create a
high-fidelity, repeatable environment; it frames simulation as complementary to
adversarial evaluation, not a universal replacement. [Deployment simulation](https://openai.com/index/deployment-simulation/)
OpenAI's EVMbench similarly uses isolated local contracts and deterministic
transaction replay for grading, while acknowledging that mocks can sometimes be
necessary. [EVMbench](https://openai.com/index/introducing-evmbench/)

That guidance matches the WARP evidence: package proof invokes the installed
wheel and CLI, checks package resources, validates sdist omission behavior, and
tests the upgrade path from the historical namespace. [Acceptance script, lines
130–239](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/scripts/accept_taskgen.sh:130)
The test-value study therefore recommends retaining a representative real
boundary for each safety-critical seam while moving redundant argument,
metadata, and mock-call-order variants in-process. [Study, lines 334–357](/Users/ashtonchew/projects/warp/.codex-worktrees/web-benchmark-onboarding-2026-08-30/docs/research/test-suite-speed-deep-dive-2026-08-10.md:334)

## Suggested rollout and acceptance criteria

1. **Visibility and known duplicate work, one PR:** use the seven retained runs
   as the job-level baseline. On the PR's first commit, expose standard pytest
   selected/pass/skip counts, wall time, and `--durations=20` from the real core
   command by bypassing or narrowly extending `run_silent` in CI. Use that trace
   to measure the two full readiness-audit calls. If they consume material wall
   time, replace only their redundant whole-repository walks with focused inputs
   to the same assertions; otherwise retain them. Do not delete either assertion
   set. Keep the standalone production audit, its focused tests, and its
   required failure categories unchanged.
2. **Measured two-way core split, one dependent PR:** define two explicit,
   feature-oriented selections from the duration trace. Preserve `-n 4
   --dist worksteal`, the remote lane, package proof, and the stable aggregate.
   Run the two core lanes and remote lane with three-way matrix capacity so a
   short remote job cannot queue a core shard; measure any hosted-runner queue
   delay separately. Add a small test that the two selections are disjoint and
   jointly cover the former default selection. Keep the split only if the
   `taskgen-acceptance` median approaches 90 s, neither sibling is more than 20%
   slower than the other, and no selected tests disappear.
3. **Disjoint root routing, independently developable PR:** keep the root
   workflow and required job visible, but return successful no-op for a
   Taskgen-only diff. Root/config/mixed/workflow changes and any unresolved base
   run the full root gate. Known-positive and fail-safe path cases are enough;
   do not add a routing framework or manifest.
4. **Slow-boundary repair only if the trace names one:** preserve at least one
   real import/process/package boundary. Move only repeated pure argument or
   serialization cases behind the existing owner. Make at most two focused
   repair attempts; if neither moves wall time materially, keep the boundary
   and stop optimizing it.
5. **Provider contract, outside ordinary CI:** when the exact route is restored,
   run the two production micro-canaries above from a disposable run root. Do
   not make network credentials or provider availability a PR-gate dependency.

The measured hosted `taskgen-acceptance` baseline is 154 s median. An ideal
two-way core split alone eventually exposes the roughly 111 s Root gates
workflow, so the combined split, three-way Taskgen matrix capacity, and safe
Taskgen-only root no-op is the first plan capable of a roughly 80--90 s overall
merge-gate critical path, or about a 45--55% reduction. This is a projection to
falsify in CI, not a claimed result. Success also requires unchanged
selected-node parity (except explicitly reviewed duplicate repository walks),
no new flakes, a fresh isolated package proof, and required checks that report
success/failure rather than Pending.

## Explicit non-goals

- Do not add a top-level `paths:` filter to a required workflow.
- Do not remove or weaken the `taskgen-acceptance` aggregate or required lanes.
- Do not add a third shard, run a scheduler grid, use `-n auto`, or buy a larger
  runner as the first response. The four-CPU public runner and current critical
  path provide a cheaper, falsifiable two-way split first.
- Do not cache `.venv`, credentials, browser/auth state, generated outputs,
  logs, or mutable live-service state.
- Do not treat a dependency cache as a reproducible package artifact or as proof
  that a wheel came from the current commit.
- Do not replace real output-contract, process, installed-package, browser, or
  safety-boundary tests with a mock reimplementation merely because they are
  slow.
- Do not add seven hand-authored provider-response fixtures, a mock provider
  service, or a no-provider output-contract CLI. Add a genuine replay only when
  an observed boundary failure justifies one.
- Do not run authenticated, private, destructive, or account-affecting
  services in ordinary CI; use approved sandbox/replay fixtures and explicit
  marked live gates where required.
- Do not claim a vendor's reported speedup as a WARP result.
- Do not edit source, workflows, dependencies, paper, or infrastructure as part
  of this research note.

## Source-date notes

All web pages were retrieved on 2026-09-04. Dynamic GitHub Actions, uv, pytest,
and xdist documentation pages did not expose a stable “last updated” date in the
rendered content, so the retrieval date is recorded above. Dates called out in
the note are taken from first-party changelogs/releases: GitHub artifact v4
(2023-12-14), GitHub cache upload rate limiting (2026-01-16), GitHub larger
concurrency queues (2026-05-07), xdist 3.8.0 (2025-06-30), setup-uv v9.0.0
(2026-07-21), and PyPA packaging guides last updated 2026-09-01.
