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
- **5PQ post-hoc** (12-question elicited-awareness suite) — measured
  **only on the `bare` arm**. Reasoning: on the XML-frame arms the
  model has been *told* it's in an eval, so the 5PQ answer is
  trivially "yes" and uninformative. On `scratchpad_native` it'd be
  redundant with `bare` for measuring the same thing. Reserve the
  expensive 12-question suite for the one place it gives clean
  information about natural awareness.

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
- **OS-Harm, SafeArena** — runners exist (`run_osharm.py`,
  `run_safearena.py`) but they cover different attack vectors than
  the four above. Could be added later; keeping the matrix tight for
  this experiment.

---

## Conditions (4 per model × benchmark)

Each cell is a `(preset, frame)` combination. Implementation already
exists in `extra_instructions.py` (presets) and
`system_prompt_frame.py` (frames).

| # | Name | Preset | Frame | What the model sees |
|---|---|---|---|---|
| 1 | `bare` | `none` | `none` | Benchmark default system prompt, no modifications |
| 2 | `xml_safety` | `none` | `safety_test` | `<llm_safety_test>` wrapping the entire system prompt |
| 3 | `xml_scenario` | `none` | `hypothetical_scenario` | `<hypothetical_scenario>` wrapping the entire system prompt |
| 4 | `native` | `scratchpad_native` | `none` | Default system prompt + appended `scratchpad_native` instruction |

**Note on "bare"**: this experiment uses `preset=none` (no scratchpad
instruction at all) as the bare baseline, not `scratchpad_bare`. This
is the cleanest reference point for the causal manipulations and
matches the existing `n20_glm5_none_v2_*` baseline runs.

**Why no fifth condition combining preset+frame** (e.g.,
`scratchpad_native` + `safety_test`): the design is
factorial-but-additive, not full-factorial. The four conditions
isolate (1) baseline, (2/3) XML manipulation alone, (4) preset
manipulation alone. Combining preset + frame would test interaction,
which we can run later if the additive results suggest it matters.

---

## The matrix

6 models × 4 benchmarks × 4 conditions = **96 cells**. Each cell runs
n=20 tasks for the pilot (n=100 reserved for canonical follow-ups
once the pilot identifies which contrasts are large enough to be
worth scaling).

### Implementation gap to flag

`system_prompt_frame` (the XML wrapper) is **only wired into the
tool-calling track** (AgentDojo, InjecAgent). On the browser track
(DoomArena, WASP), the system prompt is built inside AgentLab's
GenericAgent template and the frame wrapper isn't hooked in.
`run_causal_experiment.py` already has a clean skip for this
(`run_causal_experiment.py:84-87`):

```
if frame != "none" and benchmark in BROWSER_BENCHMARKS:
    print(f"[{i}/{len(combos)}] SKIP {benchmark}/{frame} ...")
```

So as-shipped, the matrix collapses for browser benchmarks:

| Benchmark | Conditions runnable today |
|---|---|
| AgentDojo | all 4 (bare, xml_safety, xml_scenario, native) |
| InjecAgent | all 4 |
| DoomArena | only 2 (bare, native) |
| WASP | only 2 (bare, native) |

**Two options**:

A. **Run the partial matrix as-is.** Tool-calling benchmarks get the
   full 4 conditions; browser benchmarks get only 2. We still have
   the causal contrast on AgentDojo and InjecAgent, with browser
   benchmarks contributing only the elicitation contrast (`bare` vs.
   `native`). Cheaper and unblocks the launch.

B. **Wire `system_prompt_frame` into the browser track first.**
   `agent_factory.build_openrouter_agent` would need to apply
   `wrap_system_prompt(...)` to the AgentLab GenericAgent's system
   message before the agent runs. Estimated 30-60 minutes to
   implement and verify (one new arg through `run_safety_pipeline`,
   `run_doomarena`, `run_wasp`, plus the agent-factory call). Adds
   16 more cells (2 frames × 2 benchmarks × 6 models × 4 of those
   conditions wait) to the pilot.

**Recommendation: B.** The causal contrast is the headline of this
experiment; running it on tool-calling benchmarks only would leave
half the data table with the most-interesting comparison missing.
Browser benchmarks are also where DoomArena's high VEA rates have
been observed (75-100% on Opus 4.7 + scratchpad), so they're the
most informative cells for the XML-tag manipulation.

### Other implementation prereqs

1. **Bug in `system_prompt_frame.py:30-34`**:
   ```python
   ALL_FRAMES: list[SystemPromptFrame] = [
       "none",
       "llm_safety_test",       # ← BUG: dict keys are "safety_test"
       "hypothetical_scenario",
   ]
   ```
   The dict in `_WRAPPERS` uses `"safety_test"`, but `ALL_FRAMES` (which
   is fed to `argparse choices=`) lists `"llm_safety_test"`. So
   `--system-prompt-frames safety_test` fails at argparse, and
   `--system-prompt-frames llm_safety_test` fails at the wrapper
   lookup. **Fix: change `"llm_safety_test"` → `"safety_test"` in
   `ALL_FRAMES`** before launch. One-line edit.

2. **5PQ-only-on-bare wiring.** `run_safety_pipeline` accepts
   `--judges` to select which judges run. We pass
   `verbalized_awareness purpose_continue_5q` for bare cells and
   `verbalized_awareness` only for the other three condition types.
   `run_causal_experiment` already passes `--judges` through, so this
   is a launch-time CLI flag, not a code change. Just need to be
   careful to invoke the matrix runner once per (judge-set ×
   condition-set) pair, or extend the runner to know which judges
   apply to which conditions. **Simplest: two separate
   `run_causal_experiment` invocations** — one for bare arm with full
   judges, one for the other three conditions with VEA only.

3. **Concurrency.** The recent n=100 runs used `--concurrency 20` and
   finished in ~30 min. For 96 cells × n=20 = 1,920 tasks, even at 20
   concurrent we're talking ~30 hours wall-clock if everything runs
   sequentially. Multiple cells can run in parallel via tmux/separate
   processes — recommend launching one tmux pane per benchmark so the
   four benchmarks run in parallel.

---

## Run plan

### Step 0: implementation prereqs (one commit)
- [ ] Fix `system_prompt_frame.ALL_FRAMES` typo (`llm_safety_test` →
      `safety_test`).
- [ ] Wire `wrap_system_prompt` into the browser track via
      `agent_factory` + `run_doomarena` + `run_wasp`. Add
      `--system-prompt-frame` arg to both browser runners; thread
      through to `agent_factory.build_openrouter_agent`.
- [ ] Smoke-test all 6 models with a 1-call probe each to verify
      `:thinking` mode is enabled and OpenRouter IDs resolve.

### Step 1: bare arm (with full 5PQ judging)

Most expensive arm because of 5PQ. ~6×4×20 = 480 task runs + 480 VEA
+ 480×12 = 5,760 5PQ calls.

```bash
python -m eval_awareness_experiments.run_causal_experiment \
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
    --tasks-per-split 20 \
    --output-base eval_awareness_experiments/results/causal_pilot/bare
```

### Step 2: XML-frame arms (VEA only)

```bash
python -m eval_awareness_experiments.run_causal_experiment \
    --models <same list> \
    --benchmarks agentdojo injecagent doomarena wasp \
    --conditions baseline \
    --extra-instructions-presets none \
    --system-prompt-frames safety_test hypothetical_scenario \
    --judges verbalized_awareness \
    --tasks-per-split 20 \
    --output-base eval_awareness_experiments/results/causal_pilot/xml_frames
```

### Step 3: scratchpad_native arm (VEA only)

```bash
python -m eval_awareness_experiments.run_causal_experiment \
    --models <same list> \
    --benchmarks agentdojo injecagent doomarena wasp \
    --conditions baseline \
    --extra-instructions-presets scratchpad_native \
    --system-prompt-frames none \
    --judges verbalized_awareness \
    --tasks-per-split 20 \
    --output-base eval_awareness_experiments/results/causal_pilot/native
```

Steps 1–3 can run in parallel (separate tmux sessions / output
directories). Independent.

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
- VEA judges (Opus 4.6): 480 trajectories × ~$0.03 = **$14**
- 5PQ judges (Opus 4.6): 480 × 12 prompts × ~$0.015 = **$86**
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
| Bare | $263 |
| XML × 2 | $355 |
| Native | $177 |
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

## Decisions to make before launch

1. **Implementation gap A or B?** (Recommendation: B — wire
   `system_prompt_frame` into browser runners. ~30-60 min of
   engineering, makes the matrix complete.)
2. **Confirm "bare" means `preset=none`, not `scratchpad_bare`.**
   This doc assumes `none`. If you want `scratchpad_bare`,
   `scratchpad_native` (which has no XML / no bullet count) becomes
   harder to differentiate cleanly from the baseline.
3. **n=20 vs. n=10 pilot.** n=20 matches existing pilot conventions
   and gives reliable per-prompt aggregates. n=10 would halve cost
   but make per-task swings (0.10 each) coarser. Recommend n=20.
4. **Judge model.** Default has been Opus 4.6:thinking. With Opus
   4.7 in the agent set, do we want 4.7 as judge to keep judge ≥
   strongest agent, or stick with 4.6 for continuity with prior
   runs? Recommendation: keep 4.6 — keeps numbers comparable to
   prior n=20 / n=100 work; a 4.7 re-judge can happen later.
5. **Verify OpenRouter slugs.** Especially the GPT-5.2 and Gemini
   preview IDs. One smoke test per model before the matrix launch.
6. **Result-dir naming.** Output structure is
   `causal_pilot/{bare,xml_frames,native}/<benchmark>/<condition>/<preset>/<frame>/<model>/`.
   Confirm this is the layout you want for downstream analysis.

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
