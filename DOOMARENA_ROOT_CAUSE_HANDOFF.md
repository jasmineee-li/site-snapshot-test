# DoomArena Performance Root-Cause Handoff

This handoff is for the `data-import` branch worktree at:

```text
/Users/ashtonchew/projects/browser-sim/.codex-worktrees/data-import
```

The current branch here remains `feat/worldsim-v5`. The comparison below uses our working WorldSim v5 architecture as the reference point.

## Handoff Metadata

This document is written as an operational handoff, not just an analysis note. It follows the runbook pattern of owner/context, symptoms, deterministic repro, ordered remediation, rollback, and verification criteria. Keep it version-controlled and update the "last verified" fields whenever someone runs the commands below.

```text
Owner: DoomArena/data-import evaluation owner
Reviewed from: browser-sim feat/worldsim-v5
Compared worktree: .codex-worktrees/data-import
data-import commit reviewed: 85223c68
Last verified locally: 2026-04-27
Primary failing benchmark: DoomArena browser track
Primary symptoms: shopping_admin and gitlab cells time out or produce zero task summaries; shopping produces near-zero; reddit is comparatively healthy.
Risk level: high for full-matrix launches; low for single-cell diagnostic runs.
```

Before handing this to an agent or a teammate, the one-sentence task is:

```text
Make every DoomArena browser run own its mutable resources deterministically before scaling the causal matrix.
```

## Definition Of Done

Do not call the fix complete because a full launch "does not hang." Call it complete only when these are true for a representative `tasks_per_split=20` DoomArena cell:

- The cell has exactly 4 expected splits: `reddit`, `shopping`, `shopping_admin`, and `gitlab`.
- The cell can account for `4 x 20 = 80` expected task attempts, with each split reporting completed, partial-timeout, or failed explicitly.
- Every concurrent runner logs a unique `(arm, model, split, site_url, report_port, results_dir)` tuple.
- No two live subprocesses share a `report_port`.
- No process discovers or judges another process's `study_*` directory.
- `--skip-existing` does not skip cells unless the expected split/task counts are satisfied.
- A timeout still leaves a machine-readable manifest for completed partial trajectories.
- Heavy-site capacity is enforced by configuration, not by hope or comments.

The invariant to prove in logs is:

```text
Every mutable resource has exactly one owner for the lifetime of a task.
```

Mutable resources include app container, database volume, browser context, popup report port, AgentLab study directory, auth/session state, reset endpoint, model/API quota budget, and output manifest.

## Do Not Do

- Do not run the 24-cell DoomArena model-parallel matrix until unique report ports and namespaced result ownership are fixed.
- Do not trust `--skip-existing` on partial DoomArena cells until expected split/task counts are part of the completeness check.
- Do not assume the per-model Docker stacks are active just because `setup_doomarena_per_model_dockers.sh up` succeeded.
- Do not rely on HTTP health alone as proof of clean benchmark state.
- Do not rely on `WA_*` environment variables only; log and verify the actual unprefixed `GITLAB`, `REDDIT`, `SHOPPING`, and `SHOPPING_ADMIN` URLs seen by BrowserGym/WebArena.
- Do not treat Reddit's success as evidence that GitLab, shopping, or shopping_admin can tolerate the same concurrency.

## Thirty-Minute Repro

Use this to reproduce and inspect the architecture bug without launching the full matrix.

1. In the `data-import` worktree, dry-run the launcher and confirm DoomArena is still hardcoded to the original ports:

```bash
cd /Users/ashtonchew/projects/browser-sim/.codex-worktrees/data-import
./eval_awareness_experiments/launch_pilot.sh --dry-run
```

Expected finding: the script has only one DoomArena env override block, using `9002/8081/8082/8083`, not model-specific ports from `DOCKER_PORTS_MULTI.md`.

2. Run one small DoomArena cell with one model and one arm, then inspect the emitted subprocess commands:

```bash
python -m eval_awareness_experiments.run_safety_pipeline \
  --benchmark doomarena \
  --model-name z-ai/glm-5:thinking \
  --extra-instructions-preset none \
  --system-prompt-frame none \
  --tasks-per-split 1 \
  --stage run-only \
  --browser-stage1-timeout 600
```

Expected finding: `_stage1_browser_parallel_splits()` spawns four split subprocesses in parallel, and unless fixed all four share the default `--report-port 1234`.

3. Inspect result ownership:

```bash
ls -td results/browsergym/study_* | head
```

Expected finding: outputs are in the global AgentLab `study_*` namespace, not a cell-owned path that encodes `(arm, model, split)`.

If any of those expected findings are absent because the branch has changed, update this handoff before continuing.

## Executive Summary

The root cause is not "DoomArena needs more duplicate Dockers" in the abstract. The root cause is that their current architecture does not create isolated execution lanes. It duplicates some site containers, but the launcher, routing, report-port allocation, reset semantics, result discovery, and concurrency control still treat shared mutable resources as if they were isolated.

The most concrete issues are:

1. The active launcher does not use the new per-model Docker stacks. DoomArena is still pinned to `GITLAB=:9002`, `REDDIT=:8081`, `SHOPPING=:8082`, `SHOPPING_ADMIN=:8083`.
2. Even the documented per-model plan would still put all 4 arms for a model onto the same per-model site container. That preserves the "4 concurrent sessions per heavy site container" failure mode.
3. Every parallel DoomArena split uses the same `--report-port` default, `1234`, so the popup attack success target is shared across concurrently running splits and cells.
4. Study/result discovery uses a global `results/browsergym/study_*` namespace and matches by split name only. Concurrent cells with the same split names can pick up each other's study directories.
5. DoomArena runs with `skip_reset_and_massage=True` and `relaunch=False`, while the setup script restarts existing containers without recreating them. Container state accumulates across runs.
6. The Linux `timeout(1)` wrapper bounds the Playwright `env.close()` hang, but partial recovery is implicit and race-prone. A timed-out split can leave useful task summaries behind without a trustworthy split status or completeness check.
7. Shopping, shopping_admin, and GitLab are materially heavier than Reddit. Four concurrent agents per container is already too much for those sites; scaling models without controlling arm/site concurrency worsens the problem.

WorldSim v5 got GitLab and Reddit stable by treating each execution lane as a bundle of resources:

```text
lane = app container + reset endpoint + auth state + browser/PVPO endpoint
       + result namespace + capacity budget + deterministic ownership
```

Their current DoomArena setup duplicates only part of that lane.

## What They Have Today

Relevant files:

- `.codex-worktrees/data-import/eval_awareness_experiments/DOOMARENA_PERFORMANCE.md`
- `.codex-worktrees/data-import/eval_awareness_experiments/launch_pilot.sh`
- `.codex-worktrees/data-import/eval_awareness_experiments/run_causal_experiment.py`
- `.codex-worktrees/data-import/eval_awareness_experiments/run_safety_pipeline.py`
- `.codex-worktrees/data-import/eval_awareness_experiments/run_doomarena_glm5_popup.py`
- `.codex-worktrees/data-import/eval_awareness_experiments/DOCKER_PORTS.md`
- `.codex-worktrees/data-import/eval_awareness_experiments/DOCKER_PORTS_MULTI.md`
- `.codex-worktrees/data-import/scripts/setup_doomarena_per_model_dockers.sh`

Their performance note says three optimizations landed:

- subprocess timeout around browser stage 1
- parallel splits within a cell
- per-model Docker stacks

The observed result is the important part:

```text
Sites that produced data per-cell, out of 24 cells per arm:
  reddit:         24/24
  shopping:        4/24
  shopping_admin:  0/24
  gitlab:          0/24

Tasks finished in retry:
  reddit:         288
  shopping:         1
  shopping_admin:   0
  gitlab:           0
```

That pattern is exactly what you expect if the light site can tolerate shared concurrency and the heavy sites cannot.

## Actual Process Topology

`launch_pilot.sh` starts one stream for each `(arm, benchmark)` pair. For DoomArena, that means 4 arm streams:

```text
bare x doomarena
xml_safety x doomarena
xml_scenario x doomarena
native x doomarena
```

Inside each stream, it passes all 6 models to `run_causal_experiment.py`:

```bash
--models "${MODELS[@]}"
--benchmarks "$benchmark"
--extra-instructions-presets "$preset"
--system-prompt-frames "$frame"
```

`run_causal_experiment.py` then loops over the product of benchmark, condition, model, preset, and frame, and calls `subprocess.run(cmd)` synchronously for each cell. The comments in that file are accurate: each stream owns one arm and one benchmark, then runs all 6 models sequentially.

For a single model at a time, the 4 arm streams all enter DoomArena together. Each DoomArena cell calls `_stage1_browser_parallel_splits()`, which launches the 4 site splits in parallel:

```text
reddit
shopping
shopping_admin
gitlab
```

So today's active topology is:

```text
4 arm streams x 1 current model x 4 parallel splits = 16 browser subprocesses

Per shared site container:
  reddit/forum:       4 concurrent sessions
  shopping:           4 concurrent sessions
  shopping_admin:     4 concurrent sessions
  gitlab:             4 concurrent sessions
```

That explains why Reddit survives and the heavy sites time out.

## Root Cause 1: The Per-Model Dockers Are Not Used

The per-model stacks exist in docs and setup:

- `scripts/setup_doomarena_per_model_dockers.sh`
- `eval_awareness_experiments/DOCKER_PORTS_MULTI.md`

But `launch_pilot.sh` still routes every DoomArena stream to the original DoomArena ports:

```bash
GITLAB=http://localhost:9002
REDDIT=http://localhost:8081
SHOPPING=http://localhost:8082
SHOPPING_ADMIN=http://localhost:8083
```

It does not route Sonnet to `9012/8091/8092/8093`, Opus to `9022/8101/8102/8103`, and so on.

This means commit `46bdfa0d` created capacity that the active launcher does not consume.

## Root Cause 2: Per-Model Isolation Is Still Not Enough

`DOCKER_PORTS_MULTI.md` catches this itself:

```text
Wait, that's not right - each stack only has ONE of each site, so 4
arms hitting the same model's gitlab = 4 sessions on `gitlab_<model>`
container. Same as before per container, just spread across 6
containers per site instead of 1.
```

That is the deeper architectural bug. Their desired full launch is 4 arms x 6 models. If each model has one GitLab container, then all 4 arms for that model still collide on that one GitLab. Same for shopping and shopping_admin.

The right unit of isolation is not "model". It is either:

- one lane per `(model, arm)` cell, or
- a scheduler that limits how many `(model, arm)` cells may use a site container at once.

For heavy sites, the cap is probably 1 or 2 per container, not 4.

## Root Cause 3: Shared DoomArena Report Port

`run_safety_pipeline.py` defaults to:

```text
--report-port 1234
```

It passes that unchanged into every split. `run_doomarena_glm5_popup.py` then builds the attack target from that port:

```python
report_url = f"localhost:{report_port}"
success_filter=TargetUrl(port=report_port, target_urls=[report_url])
```

The older `run_doomarena_pipeline.py` knew this would be a problem and had a per-site map:

```python
port_map = {"reddit": 1234, "shopping": 1235, "shopping_admin": 1236, "gitlab": 1237}
```

That protection was lost in the unified pipeline.

With 4 splits in one cell, 4 arm streams, and eventually 24 model/arm cells, `localhost:1234` is a shared mutable target. Even if it does not fully deadlock, it can corrupt attack success measurement and cross-wire popup success between cells.

Minimum fix: derive a unique report port for every `(model, arm, split)` runner.

Example scheme:

```text
base 12000
+ model_index * 100
+ arm_index * 10
+ split_index
```

That gives stable, non-overlapping ports such as:

```text
glm5/bare/reddit          12000
glm5/bare/shopping        12001
glm5/native/reddit        12030
sonnet/bare/reddit        12100
opus/xml_safety/gitlab    12213
```

## Root Cause 4: Global Study Directory Race

`_stage1_browser_parallel_splits()` discovers AgentLab outputs by taking:

```python
before = set(results/browsergym/study_*)
...
after = set(results/browsergym/study_*)
new_dirs = after - before
```

Then it maps study dirs back to splits by inspecting the inner child directory name for `-reddit-`, `-shopping-`, `-shopping_admin-`, or `-gitlab-`.

This is barely safe inside one process. It is not safe with 4 arm streams running the same split names at the same time. Each process can see study dirs created by sibling arm processes during its own run. Since the matching key is just the split name, not `(arm, model, split)`, a process can attach the wrong study dir to its output manifest.

In the future 24-process model-parallel launcher, this gets much worse.

Minimum fix:

- pass a unique run/cell ID into the DoomArena runner
- force AgentLab study output under a cell-specific directory, or post-move study dirs immediately under a lock
- match outputs by explicit metadata written by the child, not by global directory diff plus split-name substring

Target invariant:

```text
No process should discover another process's trajectories by scanning a shared global namespace.
```

## Root Cause 5: No Clean Reset Boundary

`run_doomarena_glm5_popup.py` invokes BrowserGym with:

```python
run_bgym_experiment(
    relaunch=False,
    n_jobs=args.n_jobs,
    skip_reset_and_massage=True,
)
```

The setup script also starts existing containers when they already exist:

```bash
docker start "$name"
```

That preserves writable state. It is convenient, but it means:

- tasks can leave state behind
- failed/hung browser sessions can leave app/browser artifacts behind
- repeated runs are not clean experiments
- "duplicate stack" does not mean "fresh stack"

WorldSim's mental model is different. A configured instance has a `reset_endpoint`, and tasks bind to concrete instances. Phase 4 resets every endpoint the task may touch when needed, and it marks a bound instance dirty if the seed, browser trace, or cleanup suggests mutation.

For DoomArena they need an equivalent lane reset policy. Options:

1. Recreate containers before each cell or before each arm/model batch.
2. Wire an env-ctrl reset endpoint per site lane and call it between tasks/cells.
3. Restore Docker volumes from a known baseline snapshot.
4. If reset is too expensive for GitLab, serialize tasks per lane and recreate after a bounded number of tasks.

The important point: "container exists and responds to HTTP" is not the same as "container is clean and ready for an experiment".

## Root Cause 6: Timeout Bounds Hangs But Partial Recovery Is Not First-Class

The Playwright hang is real. AgentLab's cleanup does:

```python
self.save_summary_info(...)
self._run_judge_evaluation(...)
env.close()
```

The `env.close()` call can block. Their `timeout(1)` wrapper is a reasonable emergency guard because it kills the whole Chromium subtree.

But the timeout is placed around an entire split batch. If task 1 through task 12 completed and task 13 hangs in cleanup, the child may leave useful `summary_info.json` files on disk while the parent sees exit code `124` or `137`.

The current code can sometimes discover matched study dirs after a timeout because `_stage1_browser_parallel_splits()` scans `results/browsergym/study_*` after all subprocesses exit. That is not a reliable recovery contract. It is opportunistic, not explicitly statused as `partial_timeout`, and it still depends on global study-dir matching that can race with sibling arm/model processes.

Minimum fix:

- On exit code 124/137, scan the child-owned study dir for completed `summary_info.json` files and judge those partials.
- Write an explicit split status such as `partial_timeout` with counts: `expected_tasks`, `completed_task_summaries`, `judged_trajectories`, and `timeout_exit_code`.
- Better fix: move the timeout inside AgentLab per task, around `env.close()` or task execution, so one cleanup hang does not fail the whole split batch.

Their own doc hints at the right patch:

```text
await asyncio.wait_for(close(), timeout=30)
```

or a sync equivalent around `env.close()` in AgentLab.

## Root Cause 7: Heavy Sites Need Capacity Budgets

Reddit/Postmill is lightweight. It handled 4 concurrent sessions per container well enough to produce partial data in all cells.

Shopping/Magento is heavy:

- large image
- slow PHP/Magento backend
- product/review state can be transactional
- historically brittle for IPI because reviews default to pending and do not render

Shopping admin is likely worse for browser agents:

- admin UI pages can have large DOM/accessibility trees
- auth/session state is more brittle
- tasks can be deeper and less cache-friendly

GitLab is robust in WorldSim, but not because GitLab is cheap. We made it robust by giving it many lanes and capping load. In our current `instances.scale.json`, GitLab has 21 replicas and Reddit has 10. Phase 2c caps GitLab at 10 in-flight verifications per replica and globally caps browser probes at 8.

Their DoomArena setup gives GitLab 1 active container today, or 6 containers if the per-model stacks are actually used. Four arms can still hit the same GitLab container at once. That is too much for their 30-minute split timeout and model latency profile.

## Root Cause 8: Completeness Is Not Strict Enough For `--skip-existing`

`launch_pilot.sh` passes `--skip-existing` to `run_causal_experiment.py`. That code scans the result manifest and skips any cell whose status starts with `complete`.

The manifest scanner infers completion from judge outputs:

```text
if n_5pq > 0: status = "complete (with 5pq)"
elif n_vea > 0: status = "complete (vea only)"
```

That is too weak for DoomArena. A cell with one judged Reddit trajectory and zero GitLab/shopping_admin trajectories can look "complete" to `--skip-existing`, even though the expected cell shape is 4 splits x N tasks.

Minimum fix:

- Teach the manifest expected cardinality per benchmark.
- For DoomArena, require all expected splits to exist.
- Require each split to have either `completed == expected_tasks` or an explicit terminal status such as `partial_timeout`, `run_failed`, or `skipped_by_capacity_policy`.
- Make `--skip-existing` skip only cells whose manifest passes this strict completeness check.

Until then, disable `--skip-existing` for DoomArena recovery runs or manually delete partial cell dirs before rerun.

## Root Cause 9: Reachability And Env Routing Are Easy To Misread

`run_doomarena_glm5_popup.py` defaults:

```text
--online-sites reddit
```

But `_build_browser_cmd()` in `run_safety_pipeline.py` does not pass `--online-sites` for the single-site split it is launching. That means non-Reddit splits can run with reachability checks scoped as if only Reddit is online. Depending on BrowserGym/DoomArena internals, this may hide misconfigured shopping/admin/gitlab endpoints instead of failing early.

Minimum fix:

```text
for DoomArena split S:
  pass --online-sites S
```

or pass all four online sites explicitly only after every configured site URL has passed a preflight.

There is also an env-var trap. DoomArena's helper sets `WA_*` variables from `DOOMARENA_WEBARENA_BASE_URL`, then the runner sets unprefixed names with `setdefault()`:

```text
REDDIT
SHOPPING
SHOPPING_ADMIN
GITLAB
```

The launcher should set and log the unprefixed names, then the child should print the final resolved URLs before building the benchmark. If they later move to model-specific ports and only set `WA_*`, routing may silently fall back to the wrong base URL.

## Root Cause 10: The Setup Script Has Footguns

`scripts/setup_doomarena_per_model_dockers.sh` advertises:

```bash
./scripts/setup_doomarena_per_model_dockers.sh --stop
./scripts/setup_doomarena_per_model_dockers.sh --rm
```

But the `case` accepts:

```text
stop|down
rm|remove
```

So the documented dashed flags do not match the script. Fix the docs or accept both forms. This matters because stale containers are already part of the failure mode; cleanup commands must be boring and correct.

Also preflight every duplicated GitLab for external URL correctness. `DOCKER_PORTS.md` says `gitlab_doom` did not need an `external_url` rewrite, but the safe rule is: every GitLab duplicate must prove links/forms resolve to its advertised host port before it enters the run matrix.

## WorldSim v5 Reference Architecture

WorldSim v5 currently keeps only GitLab and Reddit for the main v5 experiment:

```text
instances.json:       gitlab 1, reddit 1
instances.smoke.json: gitlab 1, reddit 1
instances.scale.json: gitlab 21, reddit 10
```

Each scale instance includes:

- `site_name`
- `site_url`
- `replica_index`
- `replica_name`
- `db_connection` where needed for evaluation
- `reset_endpoint`
- per-site auth
- `agent_auth`
- `pvpo_cdp_url`

The architecture is not "many Docker containers" alone. It is:

```text
same-site replica pool
+ deterministic or load-aware instance selection
+ reset endpoint per instance
+ auth state tied to the selected instance
+ dedicated browser/PVPO endpoint per worker
+ result/resume fingerprinting
+ per-replica and global concurrency caps
+ preflight that fails before a bad run starts
```

### Phase 2c: load-aware verification

Phase 2c is stateless seed/probe/cleanup, so it uses power-of-two-choices load balancing. It samples two replicas and routes to the less loaded one. It also has explicit bulkheads:

```text
GitLab per-replica cap: 10
Reddit per-replica cap: 8
Global browser probe cap: 8
```

This matters because GitLab can return quickly from the backend while the browser renderer still starves CPU. We fixed that by capping total Chromium probes, not by blindly increasing container count.

### Phase 4: deterministic execution lanes

Phase 4 needs reproducibility, so it uses deterministic task-to-instance routing. A task should land on the same replica across resume/retry paths. The worker pool pins workers to benchmark instances and staggers startup.

Each task carries runtime metadata with:

- bound instance
- bound same-site/cross-site instances
- reset endpoints
- URL placeholders for the actual selected instance

### PVPO/browser isolation

For Phase 4 rigor runs, every execution instance needs a unique `pvpo_cdp_url`. Preflight rejects duplicate CDP endpoints. After each task, the managed `pvpo-chrome-<port>` container is hard-restarted. Soft browser cleanup was not enough because beginFrame-controlled renderers can survive and burn CPU after a task ends.

DoomArena does not use PVPO in the same way, but the lesson transfers: browser resources are part of the execution lane. Shared browser/report/listener resources are not safe by default.

### Reset and dirtiness

WorldSim has `TaskResetCache`. It tracks whether a bound instance needs reset. It marks dirty when:

- a task has a seed
- the network trace contains mutating HTTP methods
- cleanup fails
- execution errors during a stateful path

DoomArena currently does not have this equivalent. It mostly relies on runtime injection and long-lived containers. That is acceptable for a small smoke test, not for 96 parallel browser sessions.

## Why We Dropped Shopping, Shopping Admin, Map, and Wikipedia

This was not only an infrastructure decision.

The turning point was the 2026-04-21 WASP-aligned cutover:

- `ce9c4d9e`: documented the WASP-aligned scoping decision
- `8d674ce8`: dropped Magento/shopping plus map from adversarial dataset and instances
- `10441858`: deleted Magento-specific editor and health-check code
- `6f44ff25`: dropped Wikipedia for full WASP alignment
- `057e8e26`: later required verified exposure contracts for Phase 4 admission

Shopping/Magento had a real bug: Magento reviews defaulted to pending (`status_id=2`), so seeded payloads did not render on the storefront. We built a three-layer defense:

- force approval (`status_id=1`)
- Phase 2c render verification
- pre-Phase-4 pending-review backstop

That made the bug fixable as engineering, but it exposed the methodological problem: Magento is transactional/admin commerce software, not a clean trusted-domain/untrusted-UGC threat model like GitLab issues/comments or Reddit posts/comments.

Map/OpenStreetMap and Wikipedia/kiwix were dropped for WASP scope alignment. Wikipedia only affected a few GitLab tasks as an auxiliary lookup surface. It was not the main robustness bottleneck in the final architecture.

For their DoomArena work, this means:

- Reddit is expected to be easiest.
- GitLab is viable, but only with real lane isolation and load control.
- Shopping/shopping_admin are both heavy and less aligned with the IPI threat model.
- If they must keep shopping/admin for DoomArena comparability, they should run them with stricter per-container concurrency than Reddit.

## Fix Plan

Apply fixes in this order. Do not reorder them for speed: later changes depend on earlier ownership guarantees.

```text
0. Add metadata logging and strict manifests.
1. Fix unique report ports.
2. Fix study/result ownership.
3. Fix `--skip-existing` completeness.
4. Fix online-sites/env routing preflight.
5. Route the launcher to per-model stacks.
6. Add per-site concurrency scheduling or arm-level isolation.
7. Add reset/cleanup policy.
8. Improve timeout/partial recovery.
9. Run capacity sweep, then scale.
```

The first three are correctness gates. Per-model Docker routing before those gates can make the run faster while preserving silent corruption.

### Step 1: Stop claiming per-model stacks are active until the launcher uses them

Update or replace `launch_pilot.sh` so each DoomArena subprocess gets the right env vars for its model.

The current script does this for every model:

```bash
GITLAB=http://localhost:9002
REDDIT=http://localhost:8081
SHOPPING=http://localhost:8082
SHOPPING_ADMIN=http://localhost:8083
```

It needs model-specific routing:

```bash
glm5:   9002 / 8081 / 8082 / 8083
sonnet: 9012 / 8091 / 8092 / 8093
opus:   9022 / 8101 / 8102 / 8103
gpt:    9032 / 8111 / 8112 / 8113
flash:  9042 / 8121 / 8122 / 8123
pro:    9052 / 8131 / 8132 / 8133
```

This will reduce cross-model contention, but it does not solve cross-arm contention.

### Step 2: Allocate unique report ports

Add arguments to `run_safety_pipeline.py`:

```text
--report-port-base
--cell-id or --arm-index/--model-index
```

Then derive split ports internally:

```python
split_offsets = {
    "reddit": 0,
    "shopping": 1,
    "shopping_admin": 2,
    "gitlab": 3,
}
report_port = report_port_base + split_offsets[split]
```

For full model/arm parallelism, make `report_port_base` unique per cell.

Do not share `localhost:1234` across parallel DoomArena processes.

### Step 3: Make result ownership explicit

Stop using global `results/browsergym/study_*` diffing as the ownership mechanism.

A robust design:

```text
results/browsergym_cells/
  doomarena/
    <arm>/
      <model_slug>/
        <split>/
          study_...
          run_meta.json
```

If AgentLab cannot write to that layout directly, immediately move or symlink the produced study dir under a file lock, then write a manifest:

```json
{
  "benchmark": "doomarena",
  "arm": "xml_safety",
  "model": "anthropic/claude-sonnet-4.6:thinking",
  "split": "gitlab",
  "study_dir": "...",
  "site_urls": {
    "GITLAB": "http://localhost:9012"
  },
  "report_port": 12113
}
```

Never infer ownership from `after - before` while sibling processes are writing to the same directory.

### Step 4: Fix strict completeness before trusting `--skip-existing`

Extend `run_manifest.py` so a DoomArena cell is complete only when the expected split records exist. A suggested cell manifest shape:

```json
{
  "benchmark": "doomarena",
  "arm": "bare",
  "model_slug": "z-ai_glm-5_thinking",
  "expected_splits": ["reddit", "shopping", "shopping_admin", "gitlab"],
  "tasks_per_split": 20,
  "splits": {
    "reddit": {"status": "complete", "completed": 20, "expected": 20},
    "shopping": {"status": "partial_timeout", "completed": 7, "expected": 20},
    "shopping_admin": {"status": "run_failed", "completed": 0, "expected": 20},
    "gitlab": {"status": "complete", "completed": 20, "expected": 20}
  },
  "cell_status": "partial_timeout"
}
```

Only `cell_status == "complete"` should be skipped on rerun. A cell with any partial or failed split should be eligible for targeted rerun of only those missing splits.

### Step 5: Fix online-sites and env preflight

For each DoomArena split subprocess, pass:

```text
--online-sites <split>
```

or pass all four sites only after a preflight verifies each one.

At child startup, log:

```text
arm=<arm>
model=<model>
split=<split>
GITLAB=<final resolved URL>
REDDIT=<final resolved URL>
SHOPPING=<final resolved URL>
SHOPPING_ADMIN=<final resolved URL>
report_port=<port>
results_dir=<dir>
```

Then do a cheap HTTP preflight against the exact URL for the split. Fail before launching AgentLab if the URL is not the intended model/arm lane.

### Step 6: Choose either arm isolation or arm throttling

Option A: true isolation.

Create one site stack per `(model, arm)` cell:

```text
6 models x 4 arms x 4 sites = 96 site containers
```

This is the closest match to their desired "24 simultaneous cells" run. It is expensive but conceptually clean.

Option B: scheduler with caps.

Keep 6 model stacks, but enforce per-site-container caps:

```text
reddit:         4 sessions per container may be OK
gitlab:         start at 1, maybe 2 after measuring
shopping:       start at 1
shopping_admin: start at 1
```

Under this model, 24 cells can be queued, but heavy site splits run only when their lane has capacity. This is slower than the optimistic 1-hour target but much more likely to complete.

Recommendation: use Option B first. It gives evidence about real per-site capacity before multiplying containers.

### Step 7: Add a real lane reset policy

Pick one:

- recreate containers before each cell
- recreate containers after each cell
- env-ctrl reset between tasks
- volume snapshot restore
- bounded reuse, such as recreate every N tasks or on any timeout

For the short term, the simplest reliable policy is:

```text
before each heavy-site cell:
  docker rm -f lane containers
  docker run fresh containers from populated images
  wait for health
  run cell
  docker rm -f lane containers on timeout/failure
```

For GitLab, boot time is expensive, so a per-model or per-arm long-lived lane can be reused, but only with dirty-state tracking and a reset endpoint.

### Step 8: Recover partial trajectories after timeout

Change timeout handling:

- if a split exits 124/137, scan for completed task dirs
- collect every task dir with `summary_info.json`
- run Stage 2 judges on those partials
- mark the split as `partial_timeout`, not just `run_failed`

Then patch AgentLab cleanup:

```text
save summary
run judge evaluation
try env.close() with timeout
if close timeout: log and continue process exit
```

This converts a cleanup hang from "lose the cell" to "lose only cleanup".

### Step 9: Measure site capacity independently

Before another full matrix run, run a capacity sweep:

```text
for site in reddit shopping shopping_admin gitlab:
  for concurrency in 1 2 4:
    run one model, one arm, fixed N tasks
    record p50/p95 task latency
    record success count
    record timeout count
    record container CPU/RSS
```

Expected:

- Reddit tolerates 4.
- GitLab likely needs 1 or 2 unless it has many replicas.
- Shopping likely needs 1.
- Shopping admin likely needs 1 and maybe higher timeout.

This prevents guessing.

## Verification Checklist

Run these checks before any full matrix launch.

### Static Checks

- `launch_pilot.sh --dry-run` prints model-specific DoomArena site URLs, not only `9002/8081/8082/8083`.
- Every generated DoomArena child command includes a unique `--report-port`.
- Every generated DoomArena child command includes an explicit `--online-sites`.
- Setup script cleanup docs match accepted arguments (`stop`, `down`, `rm`, `remove`, or dashed aliases if added).
- `run_manifest.py --check` fails partial DoomArena cells whose judged count is nonzero but expected split/task counts are incomplete.

### Runtime Checks

For one diagnostic cell:

```bash
python -m eval_awareness_experiments.run_safety_pipeline \
  --benchmark doomarena \
  --model-name z-ai/glm-5:thinking \
  --extra-instructions-preset none \
  --system-prompt-frame none \
  --tasks-per-split 2 \
  --stage all \
  --browser-stage1-timeout 900
```

Verify:

- one manifest exists for the cell
- all four split records exist
- each split logs the intended site URL
- all report ports are unique
- completed task summary count is `<= 8` and missing tasks have explicit statuses
- rerunning with skip-existing does not skip a partial cell
- no sibling process writes into the cell's results directory

### Full-Matrix Gate

Only launch 24 model/arm cells after:

- `reddit` passes at intended high concurrency
- `gitlab` passes at its chosen per-container cap
- `shopping` passes at concurrency 1 or is intentionally excluded/throttled
- `shopping_admin` passes at concurrency 1 or is intentionally excluded/throttled
- OpenRouter/model quota behavior has been measured for Opus separately from Docker contention

## Rollback Plan

If a new launcher or scheduler behaves oddly:

1. Stop new submissions.
2. Kill only the launcher processes, not all Docker containers:

```bash
pkill -f "eval_awareness_experiments.run_causal_experiment"
pkill -f "eval_awareness_experiments.run_safety_pipeline"
```

3. Preserve results for forensics:

```bash
artifact_dir="debug_artifacts/doomarena_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$artifact_dir"
cp -R logs/causal_pilot "$artifact_dir"/ 2>/dev/null || true
cp -R eval_awareness_experiments/results/causal_pilot "$artifact_dir"/ 2>/dev/null || true
```

4. Restart from the last known-good mode: one model, one arm, four splits, unique report ports, no `--skip-existing`.
5. Recreate or reset any lane that saw timeout, wrong URL routing, or cross-owned study dirs.

Do not delete partial results until they have been copied or their manifests have been inspected.

## Recommended Target Architecture for Them

Define a first-class `DoomArenaLane` concept:

```python
class DoomArenaLane:
    lane_id: str
    arm: str
    model: str
    site_urls: dict[str, str]
    report_ports: dict[str, int]
    results_root: Path
    max_concurrent_by_site: dict[str, int]
    reset_policy: str
```

Then the launcher becomes a scheduler:

```text
build all cells: 4 arms x 6 models
for each cell:
  assign model stack or arm/model stack
  assign report ports
  assign results namespace
  submit four site splits subject to per-site caps
  collect completed and partial trajectories
  judge whatever completed
```

The important invariant:

```text
Every mutable resource has exactly one owner for the lifetime of a task.
```

Mutable resources include:

- app container
- database volume
- browser context
- popup report port
- AgentLab study directory
- auth/session state
- reset endpoint
- log/output manifest

## Agent Handoff Prompt

If you hand this to an implementation agent, give it a bounded task. Do not ask it to "make DoomArena faster" in one pass.

Recommended prompt:

```text
You are working in the data-import branch of browser-sim. Do not modify WorldSim v5.

Goal: make DoomArena browser-track runs deterministic before scaling.

Implement only the first correctness slice:
1. Give every DoomArena split subprocess a unique report port derived from arm/model/split.
2. Pass explicit --online-sites for the split.
3. Log final resolved GITLAB/REDDIT/SHOPPING/SHOPPING_ADMIN URLs, report_port, arm, model, split, and output dir at child startup.
4. Add a strict manifest field that records expected_splits and tasks_per_split.
5. Make --skip-existing refuse to skip incomplete DoomArena cells.

Do not launch the full 24-cell matrix. Validate with one model, one arm, tasks_per_split=1 or 2.
Return the exact commands run and the manifest output.
```

Once that lands, give a second prompt for result namespacing and scheduler/capacity policy. Keeping the prompts small matters because the failure mode here is architectural coupling; broad agents tend to add another launcher without fixing ownership.

## Practical Next Run Recommendation

Do not jump directly to 96 simultaneous browser sessions.

Run this sequence:

1. Fix report-port uniqueness.
2. Fix study-dir ownership.
3. Route the launcher to the per-model stacks.
4. Run one model x one arm x all four splits with concurrency 1 per site.
5. Run one model x four arms, but serialize GitLab/shopping/shopping_admin and allow Reddit parallelism.
6. Only then try multi-model parallelism.

For a fast pilot, split by site:

```text
Track A: reddit full matrix, high parallelism
Track B: gitlab full matrix, low/medium parallelism
Track C: shopping/shopping_admin diagnostic only, concurrency 1
```

This gives useful data quickly without letting Magento/admin instability sink the whole matrix.

## What To Tell Them

The issue is not that their Docker duplication failed. The issue is that they duplicated containers without making them the unit of scheduling and ownership.

Reddit works because it is light enough to survive the accidental sharing. GitLab, shopping, and shopping_admin expose the real architecture bug: shared mutable lanes, shared report ports, shared result discovery, no reset boundary, and no per-site capacity budget.

The WorldSim lesson is:

```text
Scale reliable browser evaluations by creating execution lanes, not by merely
starting more containers.
```

Once each lane owns its app state, browser/report resources, reset path, result namespace, and concurrency budget, GitLab and Reddit become robust. Shopping and shopping_admin can be made more reliable operationally, but they remain heavier and less methodologically clean for IPI than GitLab/Reddit UGC surfaces.
