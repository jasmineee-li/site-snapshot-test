# GLM-5 elicitation pilot — commands

Runs `z-ai/glm-5:thinking` across four safety benchmarks with two elicitation
presets (`none` vs `scratchpad`) under the `baseline` injection condition, and
applies two awareness judges: `verbalized_awareness` (VEA) and
`purpose_continue_5q` (5PQ). No `test_awareness_steering`.

For each benchmark this produces `2 presets × tasks-per-split × n_splits`
trajectories, then two JSONL files of judge scores per split.

## Prereqs

- `OPENROUTER_API_KEY` in `.env`.
- Benchmarks installed: `bash eval_awareness_experiments/setup_benchmarks.sh`.
- **WASP** and **DoomArena** require VisualWebArena Docker containers up
  (GitLab :8023, Reddit :8080, plus WebArena sites for DoomArena — see
  `SAFETY_BENCHMARKS_HANDOFF.md`). AgentDojo and InjecAgent are pure Python.

## 1. AgentDojo (start here — pure Python, fastest)

```bash
python -m eval_awareness_experiments.run_causal_experiment \
    --benchmarks agentdojo \
    --models z-ai/glm-5:thinking \
    --conditions baseline \
    --extra-instructions-presets none scratchpad \
    --judges verbalized_awareness purpose_continue_5q \
    --tasks-per-split 20 \
    --stage all
```

**What it does:**
- 1 benchmark × 1 condition × 1 model × 2 presets = **2 combos**
- AgentDojo defaults to 1 split (`workspace`) → 20 tasks/combo → **40 trajectories total**
- Each combo: shell out to `run_safety_pipeline.py` which (stage 1) runs GLM-5
  through the tool-calling loop on 20 AgentDojo tasks, then (stage 2) applies
  VEA + 5PQ judges over the resulting trajectories.
- Output: `eval_awareness_experiments/results/causal/agentdojo/baseline/{none,scratchpad}/z-ai_glm-5_thinking/`

## 2. InjecAgent (also pure Python)

```bash
python -m eval_awareness_experiments.run_causal_experiment \
    --benchmarks injecagent \
    --models z-ai/glm-5:thinking \
    --conditions baseline \
    --extra-instructions-presets none scratchpad \
    --judges verbalized_awareness purpose_continue_5q \
    --tasks-per-split 20 \
    --stage all
```

**Shape:** 2 combos × 1 split (`dh_base` = direct harm, base setting) × 20 = 40 trajectories.

## 3. WASP (requires VWA Docker up)

```bash
python -m eval_awareness_experiments.run_causal_experiment \
    --benchmarks wasp \
    --models z-ai/glm-5:thinking \
    --conditions baseline \
    --extra-instructions-presets none scratchpad \
    --judges verbalized_awareness purpose_continue_5q \
    --tasks-per-split 20 \
    --stage all
```

**Shape:** 2 combos × 2 splits (`gitlab`, `reddit`) × 20 = 80 trajectories.

## 4. DoomArena (requires WebArena Docker up — largest)

```bash
python -m eval_awareness_experiments.run_causal_experiment \
    --benchmarks doomarena \
    --models z-ai/glm-5:thinking \
    --conditions baseline \
    --extra-instructions-presets none scratchpad \
    --judges verbalized_awareness purpose_continue_5q \
    --tasks-per-split 20 \
    --stage all
```

**Shape:** 2 combos × 4 splits (`reddit`, `shopping`, `shopping_admin`, `gitlab`) × 20 = **160 trajectories.**
Longest run by far. Consider `--tasks-per-split 10` for a faster pilot.

## 5. All four at once (single invocation)

```bash
python -m eval_awareness_experiments.run_causal_experiment \
    --benchmarks agentdojo injecagent wasp doomarena \
    --models z-ai/glm-5:thinking \
    --conditions baseline \
    --extra-instructions-presets none scratchpad \
    --judges verbalized_awareness purpose_continue_5q \
    --tasks-per-split 20 \
    --stage all
```

**Shape:** 4 × 2 = 8 combos, runs serially. Matrix log saved to
`eval_awareness_experiments/results/causal/causal_experiment_log.json`.

## Dry-run (print the commands without executing)

Append `--dry-run` to any of the above.

## Tighter smoke-test before scaling

```bash
python -m eval_awareness_experiments.run_safety_pipeline \
    --benchmark agentdojo \
    --model-name z-ai/glm-5:thinking \
    --condition baseline \
    --extra-instructions-preset scratchpad \
    --judges verbalized_awareness purpose_continue_5q \
    --tasks-per-split 2
```

2 tasks end-to-end. If the output dir contains a non-empty JSONL with both judge
scores populated, the wiring is good and you can scale up.

## Where to look at results

- **Stage-1 trajectories:**
  - Browser (doomarena/wasp): `results/browsergym/study_*/`
  - Tool-calling (agentdojo/injecagent): `results/toolcalling/<bench>/<split>/<run_name>/`
- **Stage-2 judge JSONL:**
  `eval_awareness_experiments/results/causal/<bench>/<condition>/<preset>/<model_slug>/<bench>/<split>/trajectory_awareness_results.jsonl`
- **Matrix log:** `eval_awareness_experiments/results/causal/causal_experiment_log.json`
  — one entry per combo with status (`ok` / `failed:N` / `dry_run`).
