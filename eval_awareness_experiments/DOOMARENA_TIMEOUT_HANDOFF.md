# DoomArena timeout handoff — 2026-04-29

## TL;DR

When we run the DoomArena causal pilot on **gitlab**, 6 of 8 model cells
produce **zero** trajectories. The trace shows it's **not** the agent
running long — the per-(model × split) subprocess spends ~25-30 minutes
in *startup / WebArena warm-up* before the agent ever takes its first
step, then our outer 1800s timeout fires and kills the process group.

The cells that do succeed only succeed because they happened to be
fast-per-step models that squeezed in a handful of trajectories in the
~2-3 minutes left after warm-up.

We need a second pair of eyes to:
1. Confirm the diagnosis (or find what we missed).
2. Decide between the candidate fixes (raise timeout? skip warm-up?
   change cell topology?).
3. Implement.

## How the timeout architecture currently works

### Per-cell wallclock timeout (the one biting us)

`run_safety_pipeline._stage1_browser_parallel_splits()` launches one
subprocess per `(model × split)` pair, each wrapped in Linux
`timeout(1)` so a Playwright `env.close()` deadlock can't stall the
stream forever. Code at
`eval_awareness_experiments/run_safety_pipeline.py:395-450`:

```python
timeout_sec = getattr(args, "browser_stage1_timeout", None) or 1800
...
cmd = ["timeout", "--kill-after=10", str(timeout_sec), *cmd]
proc = subprocess.Popen(cmd, ...)
...
if proc.returncode in (124, 137):
    logger.error(
        f"  [{split}] subprocess TIMED OUT after {timeout_sec}s "
        f"(exit={proc.returncode}). Likely Playwright env.close() "
        f"hang — partial trajectories may be saved."
    )
```

The 1800s default is in `--browser-stage1-timeout` (declared at
`run_safety_pipeline.py:731`). Each cell gets the same budget; reddit
and gitlab subprocesses run in parallel for a given cell.

### Per-step timeout (the one that's NOT involved)

`run_doomarena_glm5_popup` exposes `--avg-step-timeout` (default 60s)
which becomes `max_steps × avg_step_timeout` per task. With
`--max-steps 15` that's a 15-min ceiling per task. None of the failing
cells appear to be hitting this — they don't even reach a first step.

### Why we use Linux `timeout(1)` instead of `subprocess.run(timeout=)`

Playwright's headless-Chromium child tree doesn't get cleaned up by
Python's `Popen.kill()` — we need the process-group SIGKILL that
`timeout(1)` does (with `--kill-after=10`). Comment in the code says
this was added to work around a Playwright `env.close()` deadlock seen
in a py-spy stack trace.

## The actual symptom

After the most recent full run (per-model launcher,
`logs/causal_pilot_doom_per_model/*.log` from 2026-04-28), here's the
trajectory count per cell (n=20 requested):

| arm           | model        | reddit | gitlab |
|---------------|--------------|--------|--------|
| bare          | glm-5        |    20  |    13  |
| bare          | sonnet-4.6   |     4  |     0  |
| bare          | opus-4.7     |     7  |     0  |
| bare          | gpt-5.2      |     4  |     0  |
| bare          | gemini-3-flash|    7  |     0  |
| bare          | gemini-3.1-pro|    6  |     0  |
| bare          | gemini-2.5-pro|   20  |     5  |
| bare          | kimi-k2.5    |    10  |     0  |
| xml_safety    | (8 models)   |  4-16  | **all 0** |
| xml_scenario  | (8 models)   |  4-20  | 5 / rest 0|
| native        | (8 models)   |  4-20  | 0–20      |

**Reddit is fine** for every model (4-20 trajectories). **Gitlab is
catastrophically empty for 6/8 models.** The 2 models that get gitlab
data (glm and gemini-2.5-pro) tend to be fast-per-step.

Every single empty cell's parent log shows the same line:

```
ERROR __main__:   [gitlab] subprocess TIMED OUT after 1800s (exit=124).
Likely Playwright env.close() hang — partial trajectories may be saved.
```

But here's the thing: when I dig into the trajectory dirs that *did*
get created for the empty cells, I find experiment.log files that show
**the agent never took a step**. Example from
`bare/anthropic_claude-opus-4.7_thinking/_browser_runs/gitlab/.../webarena.801_0/experiment.log`
(full file is 9 lines):

```
2026-04-28 05:37:52,086 - WARNING - Model anthropic/claude-opus-4.7:thinking not found in pricing
2026-04-28 05:37:52,168 - browsergym.experiments.loop - DEBUG - Agent created.
2026-04-28 05:37:52,178 - werkzeug - INFO - WARNING: This is a development server.
 * Running on http://localhost:56495
2026-04-28 05:37:52,178 - werkzeug - INFO - Press CTRL+C to quit
[end of log]
```

The subprocess started at **05:07:52** (per parent log) and the
timeout fires at **05:37:52** — exactly 1800s later. But the AgentLab
study dir is named `2026-04-28_05-37-51_...`, meaning AgentLab only
*finished its setup phase and started writing per-task log files*
**1 second before** the parent timeout fired. So the entire 1800s
budget was burned in the gitlab subprocess's startup phase, not in
agent execution.

For comparison, **gemini-2.5-pro on gitlab succeeded with 5 tasks** —
its study dir is named `2026-04-28_05-20-57_...` (started writing at
05:20:57, ~13 minutes after subprocess start). Then the first
per-task log entry appears at 05:34:55 (another ~14 min later — Backends
preparing). Then 5 tasks complete in ~3 min (05:34:55 → 05:37:52)
before the timeout. Each task takes ~30s for gemini-2.5-pro on gitlab
when it actually runs.

So the pattern is:

```
[ 25-30 min: gitlab subprocess startup / WebArena warm-up ] [ 1-3 min agent runs ] [ TIMEOUT ]
```

Reddit doesn't have this problem — the same model+arm on reddit gets
its first agent step within ~1 min and uses the full 30 min for tasks.

## What's burning the 25-30 min on gitlab?

I haven't pinned this exactly. The captured stdout (last 30 lines, due
to our `[-30:]` log-truncation) for a failed gitlab cell shows these
messages in order:

```
INFO:doomarena.browsergym.scripts.run_bgym_experiment:Saved attack configs to ...
INFO:agentlab.experiments.study:Launching study ... - trial 1 / 3
INFO:agentlab.experiments.study:Preparing backends...
INFO:browsergym.experiments.benchmark.base:Preparing webarena backend...
ERROR:browsergym.webarena.instance:Environment variable WA_FULL_RESET is missing
WARNING:browsergym.webarena.instance:Skipping automated reset.
INFO:root:Initiating WebArena instance warm-up. Some tasks will be pre-loaded
        (massaged) to trigger some caching mechanisms and make the server more responsive.
INFO:browsergym.experiments.benchmark.base:webarena backend ready
INFO:agentlab.experiments.study:Backends ready.
INFO:root:Saving experiments to .../2026-04-28_05-37-51_genericagent-...-gitlab-...
WARNING - Model anthropic/claude-opus-4.7:thinking not found in pricing
```

We can't tell from these timestamps (all roughly the same — they're
the last 30 lines flushed when the subprocess died) which step took
the most wallclock. But the strongest signal is "WebArena instance
warm-up" — for gitlab, the warm-up "massages" tasks against the gitlab
docker container, and gitlab is famously slow to respond (each gitlab
HTTP request can take seconds).

What I have *not* yet done:
- Run a single gitlab cell **without** the timeout wrapper to measure
  exactly how long the warm-up takes.
- Look at where in browsergym the warm-up sleep / massage code lives
  to see what it actually does.
- Check whether warm-up is per-`Study` (so it runs once per cell) or
  per-task (so it runs 20× per cell).
- Check whether all 8 model streams hitting their respective gitlab
  containers in parallel saturates the host (1.9 TiB RAM, but maybe
  CPU-bound during warm-up?).

## Why this isn't the documented `env.close()` deadlock

The error message in our code says "Likely Playwright env.close()
hang", but that diagnosis is wrong for these failures. The
`env.close()` deadlock only matters once an agent has finished a task
and is trying to tear down the browser. For the gitlab failures, the
agent never even finished a *first* task. The error message just
defaulted to that explanation because exit=124 from `timeout(1)` is
ambiguous about *what* the subprocess was doing.

Reddit cells presumably *do* sometimes hit `env.close()` deadlock at
the very end (after task N completes), but they at least run task 1.

## Prior history: shopping_admin had the same disease in the old pilot

Worth flagging — the gitlab-empty pattern we're seeing now is **not
new behavior**. The previous pilot (2026-04-27, `logs/causal_pilot/*.log`,
sequential single-stack runner before per-model isolation) showed the
*same* failure mode on a *different* split: `shopping_admin`.

Counts from the old pilot (GLM-5 + sometimes sonnet/opus/gpt, splits
ran sequentially, no `timeout(1)` wrapper yet):

| split          | old-pilot outcome                                      |
|----------------|--------------------------------------------------------|
| reddit         | worked — 20+ trajectories per cell                     |
| shopping       | worked — 20-24 trajectories per cell                   |
| **shopping_admin** | **FAILED — 7 separate retries across arms, every one logged `Skipping shopping_admin: no trajectory root (None)`. Subprocess always reached "selected 20 tasks with sites == ['shopping_admin']" but never produced a study_dir.** |
| gitlab         | worked — 20 trajectories per cell                      |

Sample of the consistent failure across old logs:

```
06:30:27 [bare/glm]      === Stage 1: doomarena/shopping_admin ===
06:35:54 [bare/glm]      [stdout] selected 20 tasks ... ['shopping_admin']
07:20:13 [bare/glm]      Skipping shopping_admin: no trajectory root (None)
                         (50 min elapsed, no study_dir produced)

07:36:54 [xml_safety/glm] selected 20 tasks ... ['shopping_admin']
09:00:51 [xml_safety/glm] Skipping shopping_admin: no trajectory root (None)
                         (84 min elapsed, no study_dir produced)
```

Update after re-checking the old logs: the parent summary above hides
a more specific failure. The shopping_admin subprocesses did create
AgentLab study dirs, but all 20 tasks errored at `env.reset()` before
the agent acted:

```
playwright._impl._errors.TimeoutError: Locator.fill: Timeout 10000ms exceeded.
Call log:
waiting for get_by_label("Username")
```

Then DoomArena's `collect_results()` crashed with:

```
KeyError: "['success_filter_key', 'success_filter_value',
 'stats.cum_input_tokens', 'stats.cum_output_tokens'] not in index"
```

So the old shopping_admin issue is adjacent but not identical to the
new gitlab pipe/timeout issue. It was likely a combination of
shopping_admin login/container-state trouble under the old shared-stack
topology plus a DoomArena result-collection bug for all-error runs.
The result-collection bug has been patched; the remaining admin risk is
whether the per-model Magento containers are configured with the right
host-port base URLs.

The archived results dir confirms: every other split has trajectories
or at minimum a `trajectory_awareness_results.jsonl` from the judge,
but every `shopping_admin/` dir is **empty** —
`eval_awareness_experiments/results/_archive/doomarena_pre_2026-04-28_0505/*/*/doomarena/shopping_admin/`
contains nothing.

That's why the user said "(1) preserve it but rerun from scratch
(2) yeah should not do multiple on same container" and we scoped the
new pilot to just reddit + gitlab. We dropped shopping_admin (which
had been broken) and *kept* shopping (which had worked).

But now in the new pilot, **gitlab is the new shopping_admin** — same
"subprocess starts, never produces useful output, eats whole budget"
shape. Two different splits now, in different pilots, hitting what
looks like the same class of bug. The fact that the failing split
*moved* between pilots is itself a clue: it's not "shopping_admin's
docker is broken" — it's likely something about WebArena's per-split
warm-up + the cells / containers that happen to be assigned to it.

### Hypotheses, given both data points

- **The slow-warm-up theory:** WebArena warm-up "massages" the target
  docker container by pre-loading tasks. For some splits that's slow
  (>30 min). In the old pilot it was shopping_admin (slow on the
  shared GLM stack). In the new pilot it's gitlab (slow on the
  per-model isolation stacks). The slow split *changed* because we
  changed cell topology — something about per-model containers is
  making gitlab warm-up slower than it was on the original shared
  gitlab_doom container, where 13 trajectories DID land for glm.
- **The host-contention theory:** With 8 streams running concurrently,
  16 gitlab subprocesses warming up in parallel saturate either CPU,
  network, or docker-engine resources. Any single split going
  parallel-heavy at the same wallclock minute will starve. In the old
  pilot only 4 streams were parallel and only one shopping_admin
  warm-up at a time would run, but it still failed — so contention
  alone can't be the whole story, but it could compound the warm-up
  slowness.
- **The "wrong split hits a bug in benchmark code" theory:** The
  `webarena.shopping_admin` task list might trigger some specific
  warm-up path that hangs — and in the new pilot, the way we set
  `--online-sites gitlab` plus `--single-site gitlab` might trigger
  the same path. If so, this is in `browsergym.webarena.instance` or
  the warm-up "massaging" logic.

The fact that gitlab worked in the old pilot but fails now suggests
the issue is not "gitlab is intrinsically slow" — something about the
per-model docker stack OR our new code paths
(`--results-dir`, `--online-sites`, port 9022 instead of 9002)
slowed it down. Across all 4 arms × 2 splits in the new run, the only
gitlab cells that landed any trajectories are:

| arm           | model        | port  | n  |
|---------------|--------------|-------|----|
| bare          | glm-5        | 9002  | 13 |
| bare          | gemini-2.5-pro | 9062 |  5 |
| xml_scenario  | glm-5        | 9002  |  5 |
| native        | glm-5        | 9002  |  3 |
| native        | gemini-2.5-pro | 9062 | 20 |

Every other (model, arm, gitlab) cell is n=0. The successful cells
are concentrated on ports 9002 and 9062 — but it's not a clean
pattern: gemini-2.5-pro succeeds inconsistently (5 / 0 / 0 / 20
across 4 arms). Most likely a mix of warm-up jitter + concurrent-load
contention rather than a clean per-port "this stack is broken" story.
Worth swapping the model→port mapping in one re-run to see if the
failure follows the model or the port.

## Cell topology context

The per-model launcher
(`eval_awareness_experiments/launch_doomarena_per_model.sh`) gives
each model its own dedicated docker stack — 8 stacks total, one
gitlab + forum + shopping + admin per stack — to enforce
"no docker container ever has 2 concurrent agent sessions" isolation.
All 8 stacks run in parallel. Within each stack:

- 8 streams (one per model) × 4 arms (sequential within stream) ×
  2 splits (parallel within arm) = up to 16 concurrent gitlab
  subprocesses across the host at any moment, each hitting a
  *different* gitlab docker container.

All gitlab docker containers report `Up X hours (healthy)` when we
check `docker ps`. So it's not a "container is dead" problem.

## Files involved

| File | What's there |
|---|---|
| `eval_awareness_experiments/run_safety_pipeline.py` | `_stage1_browser_parallel_splits` (the timeout wrapper); `--browser-stage1-timeout` CLI arg |
| `eval_awareness_experiments/run_doomarena_glm5_popup.py` | DoomArena runner; `--avg-step-timeout`, `--max-steps`, `--results-dir` |
| `eval_awareness_experiments/launch_doomarena_per_model.sh` | Top-level launcher; calls `run_causal_experiment` for each (model × arm) |
| `eval_awareness_experiments/results/causal_pilot/doomarena/<arm>/<model>/_browser_runs/<split>/` | Where each cell's AgentLab `exp_root` lives (set via `--results-dir`) |
| `logs/causal_pilot_doom_per_model/<stack>.log` | Parent-side stdout per stream — has the TIMED OUT lines + first 30 lines of subprocess stdout |
| `AgentLab/src/agentlab/benchmarks/redteam.py` | `RedteamBenchmark` (max_steps default 20) |
| `AgentLab/src/agentlab/experiments/loop.py:542,640` | The `env.close()` calls that *might* deadlock |

## Candidate fixes (rank-order them and pick one)

1. **Raise `--browser-stage1-timeout` to 5400s (90 min)** for the
   gitlab split only. Cheap. Buys the slow models another 60 min on
   top of warm-up. Risk: ties up streams longer, but 8 streams ×
   90min = 12 hours total instead of 4 hours, which we can afford.
   Doesn't fix the underlying "warm-up is slow" issue.

2. **Find and disable / shorten the WebArena warm-up massaging** for
   gitlab. The "Initiating WebArena instance warm-up" line traces to
   `browsergym.webarena` somewhere — if we can pass `WA_FULL_RESET=0`
   *and* skip the massage, gitlab subprocesses might start in seconds.
   Risk: the warm-up exists for caching reasons; first agent task
   might be slower.

3. **Run gitlab subprocesses sequentially per stack instead of
   parallel-with-reddit.** If the contention is host-level (CPU during
   warm-up), running them serially might let warm-up complete in a
   few minutes each. Doubles total wallclock per cell but might be
   faster than fighting the timeout.

4. **Pre-warm the gitlab containers before launching the matrix.**
   Run a one-shot agent task per gitlab container ahead of time to
   trigger whatever caching the warm-up step does. Then either skip
   warm-up entirely (#2) or just let the second warm-up be a no-op.

5. **Skip gitlab entirely** for the slow models and just report on
   reddit. The user explicitly wants gitlab data, so this is a
   last resort.

## What we want from the second agent

- Validate or refute the diagnosis (subprocess startup eating the
  budget, *not* slow agent steps).
- Find where in `browsergym` / `agentlab` the gitlab warm-up
  actually lives — is it `browsergym.webarena.instance.WebArenaInstance`?
  How many requests does it make to the gitlab container, and what's
  expected wallclock?
- Pick a fix from the candidate list above (or propose a better one)
  and implement it.

If you're going to instrument the warm-up, the cheapest thing is to
add a `time.time()` print around each phase in the runner subprocess
and re-launch one gitlab cell with `--browser-stage1-timeout 5400`
to get a real measurement.
