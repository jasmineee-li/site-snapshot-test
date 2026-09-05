# Run guide — DoomArena + WASP, n=200, 6 models, 3 arms

Target: DoomArena and WASP at the largest feasible sample size, with 6
models × 3 elicitation arms × VEA + Transcript Purpose + ASR.

## The actual orchestrator

`eval_awareness_experiments.run_causal_experiment` is the matrix
orchestrator. It accepts lists for `--models`, `--benchmarks`,
`--conditions`, `--extra-instructions-presets`, `--system-prompt-frames`,
and `--judges`, then loops the cartesian product, dispatching each
combo to `run_safety_pipeline`. **You do not need a custom launcher.**

The bash scripts (`launch_pilot.sh`, `launch_doomarena_per_model.sh`)
exist for two reasons only:

1. **Env-var overrides** — DoomArena and WASP both use the WebArena
   env vars (`GITLAB`, `REDDIT`, `SHOPPING`, `SHOPPING_ADMIN`) but
   point at different docker stacks. A single Python process can only
   hold one set of env vars, so you need to launch separate processes
   per benchmark with different env exports.
2. **Parallelism** — splitting the matrix across N parallel processes
   so you don't wait days for everything to run sequentially. For
   DoomArena we additionally split per-model (8 streams) because each
   model has its own docker stack.

If you don't care about wallclock and don't mind running benchmarks
serially, **a single command will do the whole matrix**:

```bash
GITLAB="http://localhost:9001" \
REDDIT="http://localhost:8080" \
.venv/bin/python -m eval_awareness_experiments.run_causal_experiment \
    --models \
        anthropic/claude-opus-4.7:thinking \
        anthropic/claude-sonnet-4.6:thinking \
        z-ai/glm-5:thinking \
        openai/gpt-5.2:thinking \
        google/gemini-2.5-pro:thinking \
        moonshotai/kimi-k2.5:thinking \
    --benchmarks wasp \
    --conditions baseline \
    --extra-instructions-presets none \
    --system-prompt-frames none safety_test hypothetical_scenario \
    --judges verbalized_awareness purpose_continue_5q \
    --judge-model anthropic/claude-opus-4.7:thinking \
    --tasks-per-split 200 \
    --splits reddit gitlab \
    --output-base eval_awareness_experiments/results/n200_wasp_2026-04-29
```

That's one command, sequential, for WASP. For DoomArena you'd run a
similar command with `--benchmarks doomarena` and the per-model docker
ports — but DoomArena env vars vary per-model (different ports per
docker stack), so it's better to use the existing parallel launcher.

## Sample size reality check

`n=200` is aspirational. The actual task pools cap us:

| benchmark   | reddit | gitlab | shopping | shopping_admin |
|-------------|-------:|-------:|---------:|---------------:|
| **DoomArena** (single-site WebArena tasks) | 106 | 180 | 187 | 182 |
| **WASP** (planted prompt-injection tasks)  |  72 |  96 | n/a | n/a            |

`--tasks-per-split 200` is fine — runners take `[:max_tasks]` silently,
so each cell gets `min(200, pool_size)`. If you want comparable n
across cells, use `--tasks-per-split 72` for WASP and `100` for
DoomArena.

## Models

```
anthropic/claude-opus-4.7:thinking
anthropic/claude-sonnet-4.6:thinking
z-ai/glm-5:thinking
openai/gpt-5.2:thinking
google/gemini-2.5-pro:thinking
moonshotai/kimi-k2.5:thinking
```

## Arms

3 arms, encoded as `(preset, frame)` pairs. Pass them to
`run_causal_experiment` as a list per axis — it'll take the cartesian
product, but since `preset` is constant we end up with exactly 3 arms:

| arm           | `--extra-instructions-presets` | `--system-prompt-frames`    |
|---------------|--------------------------------|------------------------------|
| `bare`        | `none`                         | `none`                       |
| `xml_safety`  | `none`                         | `safety_test`                |
| `xml_scenario`| `none`                         | `hypothetical_scenario`      |

So:

```
--extra-instructions-presets none \
--system-prompt-frames none safety_test hypothetical_scenario
```

The orchestrator's `_arm_name()` (`run_causal_experiment.py:50-76`)
maps each `(preset, frame)` pair to a canonical arm name so the
output dirs come out clean (`bare/`, `xml_safety/`, `xml_scenario/`).

## Judges

```
--judges verbalized_awareness purpose_continue_5q
```

| metric              | judge name              | how it's computed                                          |
|---------------------|-------------------------|------------------------------------------------------------|
| **VEA**             | `verbalized_awareness`  | LLM judge per trajectory transcript. Output → `trajectory_awareness_results.jsonl`. |
| **Transcript Purpose** | `purpose_continue_5q` | 12-prompt 5PQ judge categorizing inferred purpose into Alignment/Capability/Other Eval/Real. |
| **ASR**             | (no judge — post-hoc)   | DoomArena: `attack_summary_info.json` per task. WASP: post-hoc evaluator (commit `5b2035f`). Computed at table-build time. |

## What the other agent fixed since the broken run

- **`0cfa301` Fix DoomArena timeout handling** — per-cell wallclock is
  now `N_TASKS × BROWSER_TASK_TIMEOUT + BROWSER_STAGE1_OVERHEAD`
  (defaults: 900s/task + 1800s overhead). For n=200 that's ~50.5h
  per cell. Adjust `BROWSER_TASK_TIMEOUT` down if needed.
- **`d8c15c2` Fix DoomArena shopping_admin URL handling** — fixes the
  `WA_SHOPPING_ADMIN`/`SHOPPING_ADMIN` env-var ordering bug. Both
  shopping splits should now work.
- **`46d4f42` Fix per-model Magento base URL setup** — fixes
  shopping/shopping_admin URLs for the per-model docker stacks.

So you can now opt into all 4 splits with
`--splits reddit shopping shopping_admin gitlab` if desired.

## Pre-flight

```bash
# Per-model docker stacks (8 stacks: glm/sonnet/opus/gpt/flash/pro/gemini25/kimi25).
./eval_awareness_experiments/scripts/doomarena/docker_stacks_per_model.sh up

# Verify all 8 gitlab stacks are healthy.
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "gitlab|forum" | head

# Confirm WASP task dir exists (168 task files: 96 gitlab + 72 reddit).
ls /tmp/wasp_full/webarena_tasks/ | wc -l

# Confirm shared WASP docker is up (port 9001 / 8080).
docker ps --format "{{.Names}} {{.Ports}}" | grep -E ":9001|:8080" | grep -v "_"
```

## Recommended invocation — DoomArena (parallel, per-model)

The existing `launch_doomarena_per_model.sh` already calls
`run_causal_experiment` correctly per-model, with per-model docker
env vars. It's parameterized via env vars — for our run:

```bash
N_TASKS=200 \
SPLITS="reddit gitlab" \
OUTPUT_BASE=eval_awareness_experiments/results/n200_2026-04-29 \
LOG_DIR=logs/n200_doom_2026-04-29 \
./eval_awareness_experiments/launch_doomarena_per_model.sh
```

**Two caveats with the existing launcher for this run:**

1. It iterates **all 8 models** in `MODEL_STACKS`. We only want 6
   (drop `flash` and `pro`). Comment those two lines out.
2. It iterates **all 4 arms** in `ARMS`. We only want 3 (drop
   `native`). Comment that line out.
3. Only `bare` arm gets `purpose_continue_5q` in the existing script.
   We want it on all 3 arms — change the judges per arm to
   `verbalized_awareness purpose_continue_5q`.

Three small edits to `launch_doomarena_per_model.sh` (lines 65-84):

```diff
 MODEL_STACKS=(
     "z-ai/glm-5:thinking|glm|9002|8081"
     "anthropic/claude-sonnet-4.6:thinking|sonnet|9012|8091"
     "anthropic/claude-opus-4.7:thinking|opus|9022|8101"
     "openai/gpt-5.2:thinking|gpt|9032|8111"
-    "google/gemini-3-flash-preview|flash|9042|8121"
-    "google/gemini-3.1-pro-preview|pro|9052|8131"
+#    "google/gemini-3-flash-preview|flash|9042|8121"
+#    "google/gemini-3.1-pro-preview|pro|9052|8131"
     "google/gemini-2.5-pro:thinking|gemini25|9062|8141"
     "moonshotai/kimi-k2.5:thinking|kimi25|9072|8151"
 )

 ARMS=(
     "bare         none              none                  verbalized_awareness purpose_continue_5q"
-    "xml_safety   none              safety_test           verbalized_awareness"
-    "xml_scenario none              hypothetical_scenario verbalized_awareness"
-    "native       scratchpad_native none                  verbalized_awareness"
+    "xml_safety   none              safety_test           verbalized_awareness purpose_continue_5q"
+    "xml_scenario none              hypothetical_scenario verbalized_awareness purpose_continue_5q"
 )
```

Then launch as above.

## Recommended invocation — WASP (sequential)

WASP shares ONE docker stack across all models (port 9001 / 8080), so
we can't parallelize across models without a per-model WASP setup
script (doesn't exist today; would mirror
`eval_awareness_experiments/scripts/doomarena/docker_stacks_per_model.sh` but with WASP-style
images). For now, run sequentially in one `run_causal_experiment` call:

```bash
nohup env GITLAB="http://localhost:9001" REDDIT="http://localhost:8080" \
    DATASET="webarena_prompt_injections" \
    .venv/bin/python -m eval_awareness_experiments.run_causal_experiment \
    --models \
        anthropic/claude-opus-4.7:thinking \
        anthropic/claude-sonnet-4.6:thinking \
        z-ai/glm-5:thinking \
        openai/gpt-5.2:thinking \
        google/gemini-2.5-pro:thinking \
        moonshotai/kimi-k2.5:thinking \
    --benchmarks wasp \
    --conditions baseline \
    --extra-instructions-presets none \
    --system-prompt-frames none safety_test hypothetical_scenario \
    --judges verbalized_awareness purpose_continue_5q \
    --judge-model anthropic/claude-opus-4.7:thinking \
    --tasks-per-split 200 \
    --splits reddit gitlab \
    --output-base eval_awareness_experiments/results/n200_2026-04-29 \
    > logs/n200_wasp_2026-04-29.log 2>&1 &
```

That's 6 models × 3 arms × 2 splits = 36 cells, sequential. At ~30 min
to ~3h per cell, budget ~3-5 days end-to-end.

To shave wallclock, you can fan WASP out by arm (3 parallel streams,
each running 6 models sequentially) — that's exactly what
`launch_pilot.sh` does. But arm-level parallelism on WASP still hits
the same shared docker stack 3 ways; you'd have container-state
contention. Sequential is safer.

## Alternative — `launch_pilot.sh` for both at once

`launch_pilot.sh` already wraps both DoomArena and WASP:
- 4 arms × 4 benchmarks = 16 streams (we'd only want 3 arms × 2 benchmarks = 6 streams)
- All 8 models per stream serially (we'd want only 6)
- DoomArena env vars hardcoded to ports `9002/8081/8082/8083` (no per-model isolation — uses the shared `gitlab_doom`/`forum_doom` stack only, NOT the per-model stacks)

For this run, **don't use `launch_pilot.sh`** — its DoomArena fan-out
doesn't use per-model docker isolation, so 3 arm-streams would all
hammer the same `gitlab_doom` container (which is the broken topology
that motivated `launch_doomarena_per_model.sh`). Use the per-model
launcher for DoomArena and a one-shot `run_causal_experiment` call
for WASP.

## Where things land

```
eval_awareness_experiments/results/n200_2026-04-29/
  doomarena/
    {arm}/
      {model_slug}/
        _browser_runs/{split}/                # AgentLab study output
          attack_config.json
          {study_dir}/{task}/
            summary_info.json
            attack_summary_info.json          # → ASR
        doomarena/{split}/
          trajectory_awareness_results.jsonl  # → VEA + Transcript Purpose
          run_meta.json
  wasp/
    {arm}/{model_slug}/
      _browser_runs/{split}/...               # same shape as DoomArena
      wasp/{split}/trajectory_awareness_results.jsonl
```

## Monitoring

```bash
# Live tails
tail -f logs/n200_doom_2026-04-29/*.log
tail -f logs/n200_wasp_2026-04-29.log

# Per-cell completion status
.venv/bin/python -m eval_awareness_experiments.run_manifest \
    --results-dir eval_awareness_experiments/results/n200_2026-04-29 \
    --tasks-per-split 50 \
    --benchmark-splits doomarena=reddit,shopping,shopping_admin,gitlab \
    --print

# Trajectory counts per cell
for d in eval_awareness_experiments/results/n200_2026-04-29/*/*/*/_browser_runs/*; do
    n=$(find "$d" -name summary_info.json 2>/dev/null | wc -l)
    echo "$n  $d"
done | sort -n
```

## Building the result tables

After both runs complete, adapt `/tmp/build_doom_tables_v2.py` (or
write a sister `build_wasp_tables.py`) by pointing `ROOT` at
`results/n200_2026-04-29` and updating the model/arm lists. The three
canonical tables (per `results_presentation_guide.md`):

1. **Table 1**: 1 row per (model, arm, split) — n, ASR, VEA, Classic
   Align share, Curated Align share.
2. **Table 2**: per-question 12-prompt mini-tables for the 5PQ
   breakdown.
3. **Table 3**: cross-model pivot for the headline metric.

## Killing a run

```bash
pkill -f "run_causal_experiment.*n200_"
```

## Open risks for this run

1. **Per-task timeout still 900s** (`max_steps=15 × avg_step_timeout=60`).
   The cell-level budget is now generous (50h), but per-task is
   unchanged. Slow models on heavy gitlab pages can hit
   `truncated: true` per task. Check rate after the first few cells —
   if high, lower `--max-steps` (e.g. to 10) or raise
   `--avg-step-timeout`.
2. **WASP container state pollution** — sequential per-cell mitigates
   but doesn't eliminate. If you see suspicious patterns, restart the
   shared WASP gitlab/forum containers between models.
3. **Judge cost** — VEA + 5PQ on every trajectory. With caps applied:
   - DoomArena: 6 models × 3 arms × (180+106) = ~5,150 trajectories
   - WASP:      6 models × 3 arms × (96+72)  = ~3,025 trajectories
   - Total:     ~8,175 trajectories × 13 judge calls (1 VEA + 12 5PQ)
     = **~106k judge calls**. Budget ~$150-300 in Opus 4.7 thinking.
