# Making DoomArena faster — what we tried, what worked, what's still open

Working notes from 2026-04-26 → 2026-04-27 trying to make DoomArena
runnable at the multi-model × multi-arm scale of the causal pilot.
Open for collaborator feedback / suggestions.

## TL;DR

DoomArena is the slowest benchmark in the matrix by a wide margin.
Three rounds of optimization landed:

1. **Subprocess timeout** (commit `661c018`) — bounds the Playwright
   `env.close()` deadlock that was eating hours of wallclock per stuck
   cell.
2. **Parallel splits within a cell** (commit `59bc787`) — runs all 4
   sites (reddit / shopping / shopping_admin / gitlab) concurrently
   instead of serially.
3. **Per-model docker stacks** (commit `46bdfa0`) — 5 new docker stacks
   so each (model × arm) cell can run on dedicated infra without
   contending with sibling streams.

Current state: **the model-parallel launcher hasn't been built yet**.
The 24 docker stacks are running and ready; we need a launcher that
fires 24 simultaneous (arm × model) streams instead of the current 4
streams × 6 sequential models.

Latest known wallclock estimates (n=20 per (model × split), 4 splits,
6 models, 4 arms = 96 split-results target):

| approach | est. wallclock | actual |
|---|---|---|
| original sequential pilot | ~12-16h | 8h+ then we killed it |
| + subprocess timeout | ~9-10h | killed manually after partial |
| + parallel splits | ~3-6h target | actual: 3h, only 28/96 split-results due to timeout pressure |
| + per-model dockers (untested) | ~1-2h target | TBD |

## Background: why is DoomArena slow?

Per-task latency breakdown (rough, eyeballed from logs):

```
Per task on DoomArena reddit (~1-1.5 min/task):
  - Browser DOM observation:             2-5s
  - Agent step (OpenRouter thinking):    20-60s × ~3-5 steps per task
  - Browser action (click, navigate):    1-3s
  - Eval / step transition overhead:     1-2s

Per task on shopping / shopping_admin / gitlab (~2-3 min/task):
  - Same shape, but DOMs are larger, navigation deeper, tasks longer
```

n=20 sequential within a split = 20-60 min stage 1 per (model × split)
under no contention. Multiply by 4 splits = 80-240 min per cell. ×6
models per stream × 4 streams = the matrix.

This is BEFORE the Playwright cleanup hang, which adds an unbounded
tail to each split.

## Issue 1: Playwright `env.close()` deadlock

### What we observed

In the original causal pilot (`results/causal_pilot/doomarena/`), 4
streams launched simultaneously and 8+ hours later only the GLM-5
bare arm had finished. The other DoomArena streams were technically
"alive" but making zero progress — log files hadn't updated in 6-7
hours.

### How we diagnosed it

`py-spy dump --pid <stuck_pid>` on a stuck child showed the stack:

```
Thread (active+gil): "MainThread"
    _sync (playwright/_impl/_sync_base.py:113)        ← BLOCKED
    close (playwright/sync_api/_generated.py:13032)
    close (core/chat.py:76)
    close (core/env.py:212)
    close (gymnasium/core.py:341)
    close (browsergym/attack_gateway.py:298)
    run (experiments/loop.py:478)
```

The agent loop completed normally. The hang is in env teardown:
Playwright's sync wrapper waiting on an async `Page.close()` future
that never resolves. **The agent's data is already saved at this
point** — only cleanup is broken.

### Fix

`run_safety_pipeline._run_subprocess` now wraps each browser stage-1
call in Linux `timeout(1)` with a configurable cap (default 1800s =
30 min). If the subprocess hangs, it gets force-killed at the timeout.

```python
cmd = ["timeout", "--kill-after=10", str(timeout_sec), *cmd]
```

Linux `timeout(1)` (vs `subprocess.run(timeout=N)`) handles process-
group cleanup correctly — kills the whole Chromium subtree, not just
the immediate Python child.

CLI: `--browser-stage1-timeout` flag on `run_safety_pipeline.py`.

### Limitation

**Per-(model × split) granularity** — the timeout caps the entire
n=20 task batch + cleanup + judging window in one budget. If the
batch can't fit in 30 min, we lose the trajectories that DID complete
within the budget (because stage 2 judges only run after stage 1
returns successfully).

This bit us in the retry. See "Issue 3" below.

## Issue 2: Splits running serially within a cell

### What we observed

Within one (model × arm) cell, the original code looped over 4 splits
sequentially:

```python
for split in splits:                              # serial
    _stage1_browser(benchmark, split, args)       # blocking call
```

Each split took ~30-60 min, so a full cell (4 splits) took 2-4 hours.
Across 6 models per stream = 12-24 hours per stream.

### Why it was unnecessary

The 4 DoomArena splits hit *different* docker services
(reddit→forum_doom, shopping→shopping, shopping_admin→shopping_admin,
gitlab→gitlab_doom). They don't share state and don't contend with
each other.

### Fix

`_stage1_browser_parallel_splits` (commit `59bc787`):

```python
# Spawn all splits in parallel via subprocess.Popen
for i, split in enumerate(splits):
    cmd = _build_browser_cmd(benchmark, split, args)
    if i > 0: time.sleep(1.5)   # avoid AgentLab study_<TS> collisions
    procs.append(subprocess.Popen(["timeout", "1800", *cmd]))

# Wait for all to finish
for split, proc in procs:
    proc.communicate(timeout=timeout_sec + 30)
```

Each split has its own subprocess + own timeout. AgentLab study dirs
get matched back to splits by inspecting the inner agent-run subdir
name (which contains `-<split>-`).

CLI: `--browser-splits-sequential` opts back into the legacy behavior.

### What it actually delivered

In the doomarena retry: **3h wallclock instead of 9h+**. Real win.

## Issue 3: Per-task latency on shopping/admin/gitlab under shared docker

### What we observed in the retry (3h, parallel splits enabled)

```
Sites that produced data per-cell (out of 4 splits × 6 models = 24 cells per arm):
  reddit:         24/24 cells got at least some data
  shopping:       4/24
  shopping_admin: 0/24
  gitlab:         0/24

Tasks finished in retry:
  reddit:         288 (~12/cell on avg, partial)
  shopping:       1
  shopping_admin: 0
  gitlab:         0
```

So shopping_admin and gitlab cells **didn't finish a single task**
before hitting the 30-min split timeout. Reddit got partial data
(timer fired before all 20 tasks completed for most cells).

### Why this happens

The parallel-splits design has all 4 streams' splits hit ONE shared
docker stack:

```
4 streams × 4 splits = 16 simultaneous browser sessions on:
  ├── 4 sessions on forum_doom        (reddit splits)
  ├── 4 sessions on shopping          (shopping splits)
  ├── 4 sessions on shopping_admin    (shopping_admin splits)
  └── 4 sessions on gitlab_doom       (gitlab splits)
```

Reddit (postmill, lightweight) handles 4 concurrent sessions OK.
Shopping (Magento backend, heaviest at 64GB image) buckles under
4 concurrent sessions — per-task latency goes from ~2 min to >30 min.

So the issue isn't "agent is slow on shopping" — it's "shopping
docker is overloaded by 4 concurrent agents from different streams".

### Mitigation in progress: per-model docker stacks

Commit `46bdfa0` adds 5 new docker stacks (one per non-GLM-5 model) so
each model's agent talks to its own dedicated docker:

```
24 cells × 4 splits = 96 browser sessions, distributed across 24 stacks
  Per docker container: 4 sessions max (just the 4 arms of the same model)
```

Setup script: `eval_awareness_experiments/scripts/doomarena/docker_stacks_per_model.sh up`.
Port mapping: `../../DOCKER_PORTS_MULTI.md`.

### What's still missing (open work)

The model-parallel launcher hasn't been written. Current
`launch_pilot.sh` and `run_causal_experiment.py` still iterate models
sequentially within each stream. To use the per-model dockers, we
need a launcher that:

1. Spawns 24 parallel processes (4 arms × 6 models)
2. Each process gets per-model env vars (GITLAB / REDDIT / ... pointing
   at that model's stack — see `../../DOCKER_PORTS_MULTI.md`)
3. Each process runs its 1 cell (1 model × 1 arm × 4 splits)
4. Splits within a cell run in parallel (already implemented)

Skeleton:

```bash
for arm in bare native xml_safety xml_scenario; do
  for model in glm5 sonnet opus gpt flash pro; do
    case "$model" in
      glm5)   ports="GITLAB=:9002 REDDIT=:8081 SHOPPING=:8082 SHOPPING_ADMIN=:8083" ;;
      sonnet) ports="GITLAB=:9012 REDDIT=:8091 SHOPPING=:8092 SHOPPING_ADMIN=:8093" ;;
      opus)   ports="GITLAB=:9022 REDDIT=:8101 SHOPPING=:8102 SHOPPING_ADMIN=:8103" ;;
      gpt)    ports="GITLAB=:9032 REDDIT=:8111 SHOPPING=:8112 SHOPPING_ADMIN=:8113" ;;
      flash)  ports="GITLAB=:9042 REDDIT=:8121 SHOPPING=:8122 SHOPPING_ADMIN=:8123" ;;
      pro)    ports="GITLAB=:9052 REDDIT=:8131 SHOPPING=:8132 SHOPPING_ADMIN=:8133" ;;
    esac
    nohup env $ports python -m run_safety_pipeline \
      --benchmark doomarena \
      --model-name <full-slug> \
      --extra-instructions-preset <preset-for-arm> \
      --system-prompt-frame <frame-for-arm> \
      ...
  done
done
```

Estimated wallclock with this: ~1-2h (each cell ~30-60 min, 24 cells
in parallel = max-cell-time ≈ 60 min plus startup/judging overhead).

## Other things tried that didn't pan out

### "Increase --browser-stage1-timeout to 60 or 90 min"

Considered. Would let some shopping/admin/gitlab cells complete, but
60-90 min × 24 cells × stuck-on-Playwright-hang risk = same issue
just at higher resolution. Per-model dockers attack the root cause
(contention) instead of upping the budget.

### "n_jobs=4 within a split (intra-split parallelism)"

Considered. Would 4× the speed within one split, but introduces docker
state-collision risk (4 parallel agents on the same forum, same auth,
same threads). And wouldn't help OpenRouter rate-limit pressure on
Opus 4.7:thinking. Better win/risk ratio: per-model dockers.

### "Per-task subprocess timeout"

Considered. Would let us kill individual hung tasks without losing the
whole batch. But each subprocess invocation has ~30-60s setup overhead
(AgentLab Study, browser launch, attack-config registration), so
n=20 separate invocations = ~10-20 min just in setup overhead. Not
worth it.

The right place for per-task timeout would be inside AgentLab's
experiments/loop.py — but that's an upstream patch we haven't written.

### "Monkey-patch env.close() to timeout internally"

Considered. Would surgically fix the actual Playwright hang at the
right layer. ~30 lines of `threading.Timer` around the close chain.
Not done because the subprocess-level `timeout(1)` was good enough
for the immediate pilot needs and per-model dockers reduce the hang
frequency anyway.

## Hypotheses we haven't tested

1. **Is shopping really the heaviest, or is it DOM size?** Maybe
   shopping_admin's pages are huge accessibility trees that slow the
   agent loop, not docker contention. Could be confirmed by running
   shopping_admin standalone (no contention) and measuring per-task
   time.

2. **Does Opus 4.7:thinking hit OpenRouter TPM caps in the matrix?**
   If yes, Opus cells artificially slow down on shared API even with
   per-model dockers. Per-model dockers don't help here — only thing
   that does is reducing concurrent Opus calls.

3. **Could we drop Opus from the matrix?** Opus 4.7:thinking is by far
   the slowest model and contributes little to the cross-model
   discrimination story (its post-hoc Align is already saturated at
   0.5-0.7 on every benchmark). Removing it would speed everything
   ~2× and remove a confounding factor (Opus is also the judge).

## Suggestions welcome

If you have ideas, the relevant code is:

- `run_safety_pipeline.py` — pipeline orchestration, timeouts,
  parallel-splits logic
- `run_doomarena_glm5_popup.py` — the actual DoomArena agent runner
  (calls AgentLab Study under the hood)
- `external_benchmarks/AgentLab/src/agentlab/experiments/loop.py` —
  upstream agent loop. The Playwright hang lives in this layer.
- `eval_awareness_experiments/scripts/doomarena/docker_stacks_per_model.sh` — docker stack setup
- `../../DOCKER_PORTS_MULTI.md` — model→port mapping

Open questions for the team:

1. **Worth writing the per-task timeout monkey-patch?** Would let us
   recover partial trajectories from cells that hit Playwright hangs.
   ~30 lines, maybe 1h to write+test.
2. **Should we drop Opus 4.7 from the matrix to halve runtime?** It's
   currently the slowest model + the judge model, which is a
   self-grading confound.
3. **Is there a cheaper Playwright fix upstream?** AgentLab/BrowserGym
   are open source. The hang is a known class of bug; might be a
   one-line `await asyncio.wait_for(close(), timeout=30)` patch.
