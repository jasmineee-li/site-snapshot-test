# GLM-5 elicitation pilot — commands

Runs `z-ai/glm-5:thinking` across four safety benchmarks with three elicitation
presets (`none`, `scratchpad_bare`, `scratchpad_private`) under the `baseline`
injection condition, and applies two awareness judges:
`verbalized_awareness` (VEA) and `purpose_continue_5q` (5PQ). No
`test_awareness_steering`.

For each benchmark this produces `3 presets × tasks-per-split × n_splits`
trajectories, then two JSONL files of judge scores per split.

## Why these presets

Using the less-directive scratchpad variants rather than the original
`scratchpad` preset. The original asks "what's weird, confusing, surprising?"
and "where the hell are you?" — those lead the model toward eval-awareness
directly and so aren't a clean elicitation probe.

| Preset | What it does |
|---|---|
| `none` | No extra instructions. Baseline. |
| `scratchpad_bare` | "Write 15+ bullets of unfiltered observations before acting." No leading questions about environment or weirdness. |
| `scratchpad_private` | "Private scratchpad, the user will not see it, be candid." Unmonitored-CoT framing from Hubinger 2024 / Meinke 2024 / Greenblatt 2024. |

The `bare` vs `private` comparison decomposes the effect: if `bare` ≈ `none`
and `private` > `bare`, the permission-to-be-candid is doing real work
(suppressed-but-elicitable awareness). If `bare` ≈ `private`, just making
room for free-form reflection is enough.

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
    --extra-instructions-presets none scratchpad_bare scratchpad_private \
    --judges verbalized_awareness purpose_continue_5q \
    --tasks-per-split 20 \
    --stage all
```

**What it does:**
- 1 benchmark × 1 condition × 1 model × 3 presets = **3 combos**
- AgentDojo defaults to 1 split (`workspace`) → 20 tasks/combo → **60 trajectories total**
- Each combo: shell out to `run_safety_pipeline.py` which (stage 1) runs GLM-5
  through the tool-calling loop on 20 AgentDojo tasks, then (stage 2) applies
  VEA + 5PQ judges over the resulting trajectories.
- Output: `eval_awareness_experiments/results/causal/agentdojo/baseline/{none,scratchpad_bare,scratchpad_private}/z-ai_glm-5_thinking/`

## 2. InjecAgent (also pure Python)

```bash
python -m eval_awareness_experiments.run_causal_experiment \
    --benchmarks injecagent \
    --models z-ai/glm-5:thinking \
    --conditions baseline \
    --extra-instructions-presets none scratchpad_bare scratchpad_private \
    --judges verbalized_awareness purpose_continue_5q \
    --tasks-per-split 20 \
    --stage all
```

**Shape:** 3 combos × 1 split (`dh_base` = direct harm, base setting) × 20 = 60 trajectories.

## 3. WASP (requires VWA Docker up)

```bash
python -m eval_awareness_experiments.run_causal_experiment \
    --benchmarks wasp \
    --models z-ai/glm-5:thinking \
    --conditions baseline \
    --extra-instructions-presets none scratchpad_bare scratchpad_private \
    --judges verbalized_awareness purpose_continue_5q \
    --tasks-per-split 20 \
    --stage all
```

**Shape:** 3 combos × 2 splits (`gitlab`, `reddit`) × 20 = 120 trajectories.

## 4. DoomArena (requires WebArena Docker up — largest)

```bash
python -m eval_awareness_experiments.run_causal_experiment \
    --benchmarks doomarena \
    --models z-ai/glm-5:thinking \
    --conditions baseline \
    --extra-instructions-presets none scratchpad_bare scratchpad_private \
    --judges verbalized_awareness purpose_continue_5q \
    --tasks-per-split 20 \
    --stage all
```

**Shape:** 3 combos × 4 splits (`reddit`, `shopping`, `shopping_admin`, `gitlab`) × 20 = **240 trajectories.**
Longest run by far. Consider `--tasks-per-split 10` for a faster pilot.

## 5. All four at once (single invocation)

```bash
python -m eval_awareness_experiments.run_causal_experiment \
    --benchmarks agentdojo injecagent wasp doomarena \
    --models z-ai/glm-5:thinking \
    --conditions baseline \
    --extra-instructions-presets none scratchpad_bare scratchpad_private \
    --judges verbalized_awareness purpose_continue_5q \
    --tasks-per-split 20 \
    --stage all
```

**Shape:** 4 benchmarks × 3 presets = 12 combos, runs serially. Matrix log
saved to `eval_awareness_experiments/results/causal/causal_experiment_log.json`.

## Dry-run (print the commands without executing)

Append `--dry-run` to any of the above.

## Tighter smoke-test before scaling

```bash
python -m eval_awareness_experiments.run_safety_pipeline \
    --benchmark agentdojo \
    --model-name z-ai/glm-5:thinking \
    --condition baseline \
    --extra-instructions-preset scratchpad_private \
    --judges verbalized_awareness purpose_continue_5q \
    --tasks-per-split 2
```

2 tasks end-to-end. If the output dir contains a non-empty JSONL with both judge
scores populated, the wiring is good and you can scale up.

## Optional: include the heavy-handed `scratchpad` for comparison

If you want to see the original preset alongside the gentler ones (to measure
how much the leading questions matter), add it:

```
--extra-instructions-presets none scratchpad_bare scratchpad_private scratchpad
```

That's 4 presets → 4 × n_splits × 20 trajectories per benchmark.

## Where to look at results

- **Stage-1 trajectories:**
  - Browser (doomarena/wasp): `results/browsergym/study_*/`
  - Tool-calling (agentdojo/injecagent): `results/toolcalling/<bench>/<split>/<run_name>/`
- **Stage-2 judge JSONL:**
  `eval_awareness_experiments/results/causal/<bench>/<condition>/<preset>/<model_slug>/<bench>/<split>/trajectory_awareness_results.jsonl`
- **Matrix log:** `eval_awareness_experiments/results/causal/causal_experiment_log.json`
  — one entry per combo with status (`ok` / `failed:N` / `dry_run`).
