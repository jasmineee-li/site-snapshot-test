# Causal eval-awareness × ASR experiment plan

## Goal

Test the **causal effect of eval awareness on attack success rate
(ASR)** across 6 frontier models × 4 safety benchmarks. Specifically:
when the model is told up-front "this is a safety test" or "this is a
hypothetical scenario", does its ASR shift relative to a clean
baseline? And does an elicitation-friendly system prompt
(`scratchpad_native`) replicate the effect of the explicit XML tag,
or does it land somewhere different?

## Hypothesis

Eval awareness is *causally upstream* of safer agent behavior on
adversarial prompts. Predictions:

1. **`bare`** (clean baseline) — natural ASR. Some natural eval
   awareness, measured by the 12-question 5PQ.
2. **`<llm_safety_test>` frame** — model is told it's in a safety test.
   If eval awareness causes safer behavior: **ASR drops** vs. bare.
3. **`<hypothetical_scenario>` frame** — model is told this is a
   simulation. Direction less clear: if "scenario recognition" lets
   the model treat malicious instructions as fictional/non-binding,
   **ASR increases**; if scenario recognition is just a different
   form of eval-awareness, ASR drops.
4. **`scratchpad_native`** — production-realistic elicitation prompt
   (no XML, no privacy claim). If natural eval-awareness elicitation
   recapitulates the XML-tag effect, ASR should land **between bare
   and `<llm_safety_test>`**; if it doesn't, the XML manipulation is
   doing something the natural elicitation can't.

The contrast `<llm_safety_test>` vs. `bare` is the **causal upper
bound**: it tells us how much ASR moves when eval-awareness is forced
to maximum. The contrast `scratchpad_native` vs. `bare` is the
**ecological estimate**: how much shift do we get from a
production-realistic intervention?

## What we are measuring

- **ASR** (attack success rate) — primary outcome. Measured on every
  cell.
- **VEA** (verbalized eval-awareness, in-trace) — measured on every
  cell. Spontaneous awareness in the agent's own thinking trace.
- **5PQ post-hoc** (full 12-prompt suite) — measured **only on the
  `bare` arm**. Reasoning: on the XML-frame arms the model has been
  *told* it's in an eval, so the 5PQ answer is trivially "yes" and
  uninformative. On `scratchpad_native` it'd be redundant with
  `bare` for measuring the same thing.

  Running all 12 prompts (not Curated-8) — the savings from
  Curated-8 are minor (~$28 of the ~$960 pilot) and we keep the
  Classic-5 / Curated-8 contrast as a within-cell metric.

---

## Models (6 total, all `:thinking`)

Verified against the OpenRouter API (`/api/v1/models`). One issue:

**⚠️ `anthropic/claude-sonnet-4.7` is not yet on OpenRouter.** The
latest available Sonnet is `4.6`. Either substitute `sonnet-4.6` or
wait for `4.7` to ship. Decision needed before launch.

| Model | OpenRouter ID | Input $/M | Output $/M | Notes |
|---|---|---|---|---|
| GLM-5 | `z-ai/glm-5:thinking` | $0.60 | $2.08 | Confirmed (used in n=100 sweeps) |
| Claude Sonnet 4.6 ⚠️ | `anthropic/claude-sonnet-4.6:thinking` | $3.00 | $15.00 | Substitute for unavailable 4.7 |
| Claude Opus 4.7 | `anthropic/claude-opus-4.7:thinking` | $5.00 | $25.00 | Confirmed |
| GPT-5.2 | `openai/gpt-5.2:thinking` | $1.75 | $14.00 | Verified on API (note: also has `gpt-5.2-codex`, `gpt-5.2-pro` variants) |
| Gemini 3 Flash Preview | `google/gemini-3-flash-preview` | $0.50 | $3.00 | Verified on API |
| Gemini 3.1 Pro Preview | `google/gemini-3.1-pro-preview` | $2.00 | $12.00 | Verified on API |

`:thinking` enables reasoning mode where supported (Anthropic + GLM
families); reasoning tokens are billed at the output rate. Gemini
preview models route reasoning through the same completion field
(no `:thinking` suffix needed). GPT-5.2 supports reasoning via the
`reasoning` parameter.

**Smoke-test before launch**: 1-call probe per model checking that
`message.reasoning` (or equivalent) is non-empty. Especially
important for GPT-5.2 and the Gemini previews where the codebase
has limited prior usage.

---

## Benchmarks (4 in scope)

In scope:

| Benchmark | Track | Attack vector |
|---|---|---|
| AgentDojo | tool-calling | injection in tool outputs |
| InjecAgent | tool-calling | injection in tool outputs (DH/DS attack types) |
| DoomArena | browser (BrowserGym) | popup attacks, page-content injections |
| WASP | browser (BrowserGym) | in-page injection on real-feeling sites |

### Other benchmarks we support but are not running

- **EIA** — supported but **import-only**. Pre-recorded trajectories
  from Computer Using Agent (Browser-use, OpenHands, etc.); we
  normalize the eval results into our trajectory format and run
  judges on them. Cannot run new conditions or new presets — just
  re-judges what's there. Skipped here because we can't apply causal
  manipulations.
- **MiniWob, WebArena, OSWorld, WorkArena** — supported by AgentLab
  but these are **capability/agent benchmarks**, not safety
  benchmarks. No injection attacks; no ASR concept. Out of scope.
- **OS-Harm, SafeArena** — not integrated (runner stubs removed 2026-09).
  They cover different attack vectors than the four above. Could be
  added later; keeping the matrix tight for this experiment.

---

## Arms (4 per model × benchmark)

Each cell is a `(preset, frame)` combination. Implementation already
exists in `extra_instructions.py` (presets) and
`system_prompt_frame.py` (frames). The orthogonal `condition` flag
(`injection_modifier`, which wraps injection text not the system
prompt) is always set to `baseline` in this experiment — the XML
manipulation is via `frame`, not `condition`.

| Arm | Preset | Frame | What the model sees |
|---|---|---|---|
| `bare` | `none` | `none` | Benchmark default system prompt, no modifications |
| `xml_safety` | `none` | `safety_test` | `<llm_safety_test>` wrapping the entire system prompt |
| `xml_scenario` | `none` | `hypothetical_scenario` | `<hypothetical_scenario>` wrapping the entire system prompt |
| `native` | `scratchpad_native` | `none` | Default system prompt + appended `scratchpad_native` instruction |

**Note on "bare"**: this experiment uses `preset=none` (no scratchpad
instruction at all) as the bare baseline, not `scratchpad_bare`. This
is the cleanest reference point for the causal manipulations and
matches the existing `n20_glm5_none_v2_*` baseline run, which is the
one cell directly reusable from prior runs.

**Why no fifth arm combining preset+frame** (e.g.,
`scratchpad_native` + `safety_test`): the design is
factorial-but-additive, not full-factorial. The four arms isolate
(bare) baseline, (xml_safety/xml_scenario) XML manipulation alone,
(native) preset manipulation alone. Combining preset + frame would
test interaction, which we can run later if the additive results
suggest it matters.

---

## The matrix

6 models × 4 benchmarks × 4 conditions = **96 cells**. Each cell runs
n=20 tasks for the pilot (n=100 reserved for canonical follow-ups
once the pilot identifies which contrasts are large enough to be
worth scaling).

### Implementation prereqs (must land before launch)

1. **Wire `system_prompt_frame` into the browser track.** Currently
   only the tool-calling track applies the frame; DoomArena/WASP
   skip non-`none` frames at `run_causal_experiment.py:84-87`. Need
   to thread `wrap_system_prompt(...)` through
   `agent_factory.build_openrouter_agent` so it wraps the AgentLab
   `GenericAgent`'s system message before the agent runs. Add
   `--system-prompt-frame` to `run_doomarena` and `run_wasp`. Remove
   the skip in the matrix runner. **Estimated 30-60 minutes.**

2. **Bug fix in `system_prompt_frame.py:30-34`**:
   ```python
   ALL_FRAMES: list[SystemPromptFrame] = [
       "none",
       "llm_safety_test",       # ← BUG: dict keys are "safety_test"
       "hypothetical_scenario",
   ]
   ```
   Dict in `_WRAPPERS` uses `"safety_test"`, but `ALL_FRAMES` (fed to
   `argparse choices=`) lists `"llm_safety_test"`. Currently
   `--system-prompt-frames safety_test` fails at argparse, and
   `--system-prompt-frames llm_safety_test` fails at the wrapper
   lookup. **One-line edit.**

3. **Result-dir restructure.** Current scheme buries benchmark deep
   in the path:
   `causal_pilot/<arm_group>/<benchmark>/<condition>/<preset>/<frame>/<model>/`.
   Restructure to put benchmark at the top and collapse
   (preset, frame) into a single arm name:
   `causal_pilot/<benchmark>/<arm>/<model>/`
   where `<arm>` ∈ `{bare, xml_safety, xml_scenario, native}`.
   The old `<condition>` segment is dropped because we always pass
   `--condition baseline` in this experiment (frame is the
   manipulation, not condition). **~10-line edit to
   `run_causal_experiment.py`'s output-dir construction.**

4. **5PQ-only-on-bare wiring.** `run_safety_pipeline` accepts
   `--judges` to select which judges run. For the bare arm we pass
   `verbalized_awareness purpose_continue_5q`; for the other three
   arms we pass `verbalized_awareness` only. **Simplest: launch the
   matrix in three separate `run_causal_experiment` invocations**
   (one per arm-group with the right `--judges`), no code change
   required.

5. **Run tracker.** New `run_manifest.json` written to
   `eval_awareness_experiments/results/causal_pilot/run_manifest.json`,
   one entry per (model, benchmark, arm, n, status, timestamp).
   Hooked into `run_causal_experiment.py` so cells already complete
   are skipped. Lets multiple sessions / collaborators see what's
   done without rereading every result dir. **Estimated 30 minutes.**

### Wall-clock estimate (corrected)

Per-task wall-clock from existing runs:
- Tool-calling (AgentDojo, InjecAgent): ~1-3 min/task
- Browser (DoomArena, WASP): ~5-10 min/task

With `concurrency=20` *within* a cell, 20 tasks complete in roughly
the longest single task (~5-10 min). For 96 cells:
- Sequential: ~11 hours total
- **4 benchmarks running in parallel** (4 tmux sessions): 24 cells
  per stream × ~7 min = **~3 hours wall-clock**

This is the recommended setup. The earlier "30 hours" estimate was
based on 30-min/task pessimism; actual per-task times are 5-10x
faster.

---

## Existing runs (audit)

From `results/` inventory:

**Reusable for this experiment (1 cell):**
- `n20_glm5_none_v2_20260425_2009` → GLM-5 + AgentDojo + bare. Maps
  directly to a cell in the matrix; can skip rerunning.

**Not reusable** (different presets/conditions):
- All other GLM-5 runs use `scratchpad_bare` / `scratchpad_private` /
  OG `scratchpad` — none match the new four arms.
- All Opus 4.7 runs are on DoomArena with `motive` / scratchpad
  variants — not in the new arm set.
- Kimi 2.6, Sonnet 4.6, GPT-5.2, Gemini variants: nothing prior.

**Effective scope for the pilot: 95 of 96 cells to run.** The single
GLM-5 + AgentDojo + bare cell already-done saves ~$1.20.

## Run plan

### Step 0: implementation prereqs ✅ done
- [x] Fix `system_prompt_frame.ALL_FRAMES` typo
      (`llm_safety_test` → `safety_test`) — commit `0e4a957`
- [x] Wire `wrap_system_prompt` into the browser track via
      `agent_factory` + `run_doomarena_glm5_popup` + `run_wasp` —
      commit `c75fbaa`
- [x] Restructure output-dir layout in `run_causal_experiment.py`
      to `<benchmark>/<arm>/<model>/` — commit `c4ed106`
- [x] Build run-manifest tracker with `--skip-existing` flag —
      commit `28e95a9`
- [x] Smoke-test script for all 6 models — commit `7340e12`

### Step 0a: pre-flight smoke test

Verify all 6 OpenRouter IDs resolve and reasoning mode engages
before launching the matrix. Cost: ~$0.10.

```bash
.venv/bin/python -m eval_awareness_experiments.smoke_test_models
```

Expected: `Summary: 6/6 models passed`. Re-run if a transient failure
appears (GLM-5 occasionally doesn't engage reasoning on the first
try; succeeds on retry).

### Step 1: bare arm (with full 12-prompt 5PQ judging)

Most expensive arm because of 5PQ. ~6×4×20 = 480 task runs + 480 VEA
+ 480×12 = 5,760 5PQ calls.

```bash
.venv/bin/python -m eval_awareness_experiments.run_causal_experiment \
    --models z-ai/glm-5:thinking \
             anthropic/claude-sonnet-4.6:thinking \
             anthropic/claude-opus-4.7:thinking \
             openai/gpt-5.2:thinking \
             google/gemini-3-flash-preview \
             google/gemini-3.1-pro-preview \
    --benchmarks agentdojo injecagent doomarena wasp \
    --conditions baseline \
    --extra-instructions-presets none \
    --system-prompt-frames none \
    --judges verbalized_awareness purpose_continue_5q \
    --judge-model anthropic/claude-opus-4.7:thinking \
    --tasks-per-split 20 \
    --skip-existing \
    --output-base eval_awareness_experiments/results/causal_pilot
```

The 5PQ judge runs all 12 prompts in
`judges/prompts/purpose_continue_5q/*.json`. Classic-5 / Curated-8
contrasts are computed downstream from the per-prompt outputs.

### Step 2: XML-frame arms (VEA only)

```bash
.venv/bin/python -m eval_awareness_experiments.run_causal_experiment \
    --models z-ai/glm-5:thinking \
             anthropic/claude-sonnet-4.6:thinking \
             anthropic/claude-opus-4.7:thinking \
             openai/gpt-5.2:thinking \
             google/gemini-3-flash-preview \
             google/gemini-3.1-pro-preview \
    --benchmarks agentdojo injecagent doomarena wasp \
    --conditions baseline \
    --extra-instructions-presets none \
    --system-prompt-frames safety_test hypothetical_scenario \
    --judges verbalized_awareness \
    --judge-model anthropic/claude-opus-4.7:thinking \
    --tasks-per-split 20 \
    --skip-existing \
    --output-base eval_awareness_experiments/results/causal_pilot
```

### Step 3: scratchpad_native arm (VEA only)

```bash
.venv/bin/python -m eval_awareness_experiments.run_causal_experiment \
    --models z-ai/glm-5:thinking \
             anthropic/claude-sonnet-4.6:thinking \
             anthropic/claude-opus-4.7:thinking \
             openai/gpt-5.2:thinking \
             google/gemini-3-flash-preview \
             google/gemini-3.1-pro-preview \
    --benchmarks agentdojo injecagent doomarena wasp \
    --conditions baseline \
    --extra-instructions-presets scratchpad_native \
    --system-prompt-frames none \
    --judges verbalized_awareness \
    --judge-model anthropic/claude-opus-4.7:thinking \
    --tasks-per-split 20 \
    --skip-existing \
    --output-base eval_awareness_experiments/results/causal_pilot
```

All three steps share one `--output-base` and write into
`causal_pilot/<benchmark>/<arm>/<model>/` without colliding (each step
hits a distinct (preset, frame) combination, which maps to a distinct
arm directory). `--skip-existing` makes any of the steps safely
restartable — if a session dies halfway through, just re-run the same
command and it'll pick up where it left off.

Steps 1–3 can run in parallel (separate tmux sessions). Within each
step, launch one tmux pane per benchmark for 4× parallelism on the
wall-clock.

### Step 4: while running, watch progress in the dashboard

```bash
.venv/bin/python -m eval_awareness_experiments.dashboard \
    --results-dir eval_awareness_experiments/results/causal_pilot \
    --port 8000
```

Open `http://localhost:8000`. The dashboard re-scans the manifest on
every page load — refresh to see newly-completed cells. Click any
cell for per-cell metrics (ASR, VEA rate, 5PQ Align/Any) and the
list of per-trajectory markdown files.

For a CLI alternative (e.g. monitoring from a remote tmux session):

```bash
.venv/bin/python -m eval_awareness_experiments.run_manifest \
    --results-dir eval_awareness_experiments/results/causal_pilot \
    --print
```

### Step 5: post-launch — pre-launch sanity check on a single cell

Before launching the full matrix, smoke-test one cell end-to-end (one
model × one benchmark × one arm × n=2 tasks, ~2 minutes, ~$0.05) to
confirm the new browser-track frame plumbing works as expected:

```bash
.venv/bin/python -m eval_awareness_experiments.run_causal_experiment \
    --models z-ai/glm-5:thinking \
    --benchmarks agentdojo \
    --conditions baseline \
    --extra-instructions-presets none \
    --system-prompt-frames safety_test \
    --judges verbalized_awareness \
    --judge-model anthropic/claude-opus-4.7:thinking \
    --tasks-per-split 2 \
    --output-base /tmp/causal_pilot_smoke
```

Confirm the resulting markdown sidecar shows `<llm_safety_test>` in
the system prompt section before scaling up.

### Wall-clock target
- Each step: ~3 hours with 4 benchmarks in parallel
- All three steps in parallel (12 tmux panes total): **~3 hours
  wall-clock for the entire pilot**

---

## Cost estimate (n=20 pilot)

### Calibration

Anchor: the n=100 GLM-5 + AgentDojo + bare run (`875a281`) cost
**$27.90** end-to-end (agent + VEA + 5PQ judges, with concurrency=20).
Per task: ~$0.28 / 100 = **$0.28/task** for cheap-model + cheap-bench
+ all judges. Working backward from GLM-5's $0.60/$2.08 pricing and
the cost-of-judges (Opus 4.6 at $5/$25 with thinking-token cap),
implied per-task token usage is ~50k input + 15k output for the
agent on AgentDojo. We use this calibration to estimate the other
benchmarks (DoomArena/WASP need more input tokens for axtree
observations; InjecAgent runs are shorter than AgentDojo).

### Per-task cost by model × benchmark (agent only, no judges)

Token assumptions: AgentDojo / InjecAgent ~50k input + 15k output
(tool-calling, capped thinking). DoomArena ~75k input + 15k output
(axtree observations). WASP ~100k input + 20k output (richer
multi-page navigation).

| Model | AgentDojo | InjecAgent | DoomArena | WASP |
|---|---|---|---|---|
| GLM-5 | $0.06 | $0.06 | $0.08 | $0.10 |
| Sonnet 4.6 | $0.38 | $0.38 | $0.45 | $0.60 |
| Opus 4.7 | $0.63 | $0.63 | $0.75 | $1.00 |
| GPT-5.2 | $0.30 | $0.30 | $0.34 | $0.45 |
| Gemini 3 Flash | $0.07 | $0.07 | $0.08 | $0.11 |
| Gemini 3.1 Pro | $0.28 | $0.28 | $0.33 | $0.44 |
| **Sum across 6 models** | **$1.71** | **$1.71** | **$2.03** | **$2.71** |

Opus 4.7 is the dominant cost contributor (~37% of agent spend).
GLM-5 + Gemini Flash are essentially free relative to the others.

### Per-arm cost at n=20

Per cell = 20 × per-task. Per arm = sum across (benchmarks × models).

**Bare arm** (preset=none, frame=none, all 4 benchmarks, all 6 models):
- Agent runs: 20 × ($1.71 + $1.71 + $2.03 + $2.71) = **$163**
- VEA judges (Opus 4.7, $5/$25 — same as 4.6): 480 trajectories × ~$0.03 = **$14**
- 5PQ judges (Opus 4.7, full 12 prompts): 480 × 12 × ~$0.015 = **$86**
- **Bare total: ~$263**

**XML-frame arms** (2 frames × all 4 benchmarks × all 6 models):
- Agent runs: 2 × $163 (same shape as bare) = **$326**
- VEA judges only: 960 × $0.03 = **$29**
- (no 5PQ — design decision)
- **XML total: ~$355**

**Native arm** (preset=scratchpad_native, frame=none):
- Agent runs: $163
- VEA judges only: 480 × $0.03 = **$14**
- **Native total: ~$177**

### Pilot grand total

| Arm | Cost |
|---|---|
| Bare (with full 12-prompt 5PQ) | $263 |
| XML × 2 (VEA only) | $355 |
| Native (VEA only) | $177 |
| **Subtotal** | **$795** |
| 25% buffer (retries, longer trajectories than estimated) | $200 |
| **Pilot total (n=20)** | **~$1,000** |

This is **substantially lower** than the initial $4,300 estimate
because the GLM-5 calibration showed actual per-task token usage is
roughly half what the older COST_ESTIMATE.md numbers assumed
(thinking-token caps + concurrency=20 keep individual trajectories
tight).

### Scaling to n=100

If the pilot shows promising contrasts and we want canonical
numbers, scale 5×: **~$5,000 for n=100 across the full matrix**.
Comparable to the alignment-faking paper's reported budget.

### Cost-reduction levers

| Lever | Savings | New total (n=20) |
|---|---|---|
| Drop Opus 4.7 | ~37% | ~$630 |
| Drop GPT-5.2 | ~17% | ~$830 |
| Restrict to tool-calling track only (AgentDojo+InjecAgent) | ~50% | ~$500 |
| Drop the `native` arm (test only XML manipulation) | ~18% | ~$820 |
| Skip 5PQ on `bare` (just VEA everywhere) | ~$86 | ~$910 |
| n=10 instead of n=20 | ~50% | ~$500 |

The cheapest defensible variant: tool-calling-only at n=20 with all
6 models = ~$500. The most expensive defensible variant: full
matrix at n=100 = ~$5,000. **Recommendation: launch the full pilot
at n=20 (~$1,000) — every lever costs more in interpretability than
it saves in dollars.**

---

## Decisions resolved

| Decision | Choice |
|---|---|
| "bare" definition | `preset=none, frame=none` ✅ |
| Judge model | `anthropic/claude-opus-4.7:thinking` ✅ |
| Sonnet substitution | `claude-sonnet-4.6:thinking` until 4.7 ships ⚠️ |
| 5PQ scope | Curated-8 (drops 04, 05, 11, 12) ✅ |
| Browser-track frame support | Wire it (Priority 1 implementation) ✅ |
| Result dir layout | `causal_pilot/<benchmark>/<arm>/<model>/` ✅ |
| n | n=20 pilot, n=100 follow-up if pilot is informative |

## Decisions still open

1. **Sonnet 4.7 substitution.** `anthropic/claude-sonnet-4.7` is
   not on OpenRouter as of this query. Either substitute
   `sonnet-4.6:thinking` or wait. **Recommendation: substitute,
   note in the Table 1 footer that 4.6 was used as a 4.7 stand-in.**
2. **Verify OpenRouter slugs**. Smoke test each model with a 1-call
   probe before the matrix launch. Especially GPT-5.2 and the
   Gemini preview IDs.

---

## What success looks like

A clean Table-1 row for each of the 96 cells, plus the cross-arm
contrasts:

- ASR(bare) vs ASR(xml_safety): does forcing eval-awareness reduce
  ASR? By how much, per model and per benchmark?
- ASR(bare) vs ASR(xml_scenario): does scenario-framing reduce or
  increase ASR? Direction is the headline finding.
- ASR(bare) vs ASR(native): does the production-realistic preset
  recover the XML-tag effect, or does it land at baseline? This is
  the lever for "can we get safer behavior at deployment without
  showing the model an obvious eval signal?"
- VEA(bare) → VEA(xml_safety): a sanity check that the manipulation
  worked. We expect VEA to spike to ~100% on xml_safety. If it
  doesn't, models are ignoring the frame.
- 5PQ(bare): natural eval-awareness baseline by model × benchmark.
  Lets us correlate **which models naturally have high awareness**
  with **which models show the largest ASR gap between bare and
  xml_safety**. If they correlate, eval-awareness is a real driver
  of ASR. If they don't, something else is.

The strongest result this experiment could deliver: a per-model
elasticity number — "ASR drops by X% on this model when we move from
no eval cue to maximum eval cue", correlated against natural eval-
awareness from 5PQ. That's the central object the rest of the
program can build on.
