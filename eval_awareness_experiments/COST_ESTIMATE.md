# Cost Estimate: Safety Benchmark Eval-Awareness Experiments

## Scope

Running **n=100 tasks per benchmark** across 3 benchmarks (DoomArena, WASP, SafeArena), up to 6 models, up to 3 injection conditions, plus judging. Used for budgeting the experiment suite.

## Pricing Assumptions (USD per 1M tokens)

Rates as of April 2026 (user-confirmed). Verify current OpenRouter / provider pricing before launch.

| Model | Input $/M | Output $/M | Reasoning? | Notes |
|-------|-----------|-----------|------------|-------|
| `anthropic/claude-opus-4.6` | 5.00 | 25.00 | yes (thinking) | Flagship |
| `anthropic/claude-sonnet-4.6` | 3.00 | 15.00 | yes (thinking) | Workhorse |
| `openai/gpt-4.1` | 2.00 | 8.00 | no | (input estimated) |
| `google/gemini-2.5-pro` | 1.25 | 10.00 | yes | Reasoning tokens billed at output rate |
| `anthropic/claude-3.7-sonnet` | 3.00 | 15.00 | yes (thinking) | Same as 4.6-sonnet pricing |
| `openai/o4-mini` | 1.10 | 4.40 | yes (heavy reasoning) | Cheap but emits lots of reasoning tokens |

## Token Usage Per Task (Agent Runs)

Estimated per single agent task trajectory on a web benchmark:

| Benchmark | Avg steps | Input tokens/task | Output tokens/task (non-reasoning) | Output incl. reasoning |
|-----------|-----------|--------------------|----------------------------------|------------------------|
| DoomArena | 15 | ~150k | ~6k | ~30k |
| WASP | 20 | ~200k | ~8k | ~40k |
| SafeArena | 12 | ~120k | ~5k | ~25k |

Per-step size is dominated by the accessibility tree observation (~8-15k tokens/step text-only; screenshots add ~1-2k tokens as image input). Reasoning models emit ~5x more output tokens.

---

## Experiment A only: baseline Opus 4.6, n=100 per benchmark

This is the minimum viable run to answer "do models have high eval awareness on IPI benchmarks?" — no causal conditions, just baseline trajectories on 3 safety benchmarks with Opus 4.6, plus judging.

### Agent runs (Opus 4.6 with reasoning)

| Benchmark | Input | Output | Cost/task | n=100 |
|-----------|-------|--------|-----------|-------|
| DoomArena | 150k × $5/M = $0.75 | 30k × $25/M = $0.75 | $1.50 | **$150** |
| WASP | 200k × $5/M = $1.00 | 40k × $25/M = $1.00 | $2.00 | **$200** |
| SafeArena | 120k × $5/M = $0.60 | 25k × $25/M = $0.625 | $1.23 | **$122** |
| | | | **Subtotal** | **$472** |

### Judging

**Trajectories to judge**: 300 new + ~96 existing (WAI + TAC) = **~396 trajectories**.

Per-trajectory token usage depends on how many judges you run:

| Judge set | Input/traj | Output/traj | Cost/traj (Opus) | Cost/traj (Sonnet) | Cost/traj (Gemini) |
|-----------|-----------|-------------|------------------|--------------------|---------------------|
| TAS + verbalized (2 judges) | 30k | 1k | $0.175 | $0.105 | $0.048 |
| All 5 judges | 160k | 3.8k | $0.895 | $0.537 | $0.238 |

Totals for 396 trajectories:

| Judge config | Cost |
|--------------|------|
| Opus self-judge, 2 judges | $69 |
| Opus self-judge, all 5 judges | $354 |
| Sonnet judge, 2 judges | $42 |
| Sonnet judge, all 5 judges | $213 |
| **Gemini judge, 2 judges** ← recommended | **$19** |
| Gemini judge, all 5 judges | $94 |

### P(eval) on safety-benchmark environments

~90 new pages × 10 prompts × 3 formats = 2,700 calls (~5k in + 50 out each).
(Existing WAI/TAC P(eval) results already in `results/p_eval_all/` — no re-run needed.)

| Judge | Cost |
|-------|------|
| Opus | $70 |
| Sonnet | $42 |
| **Gemini** ← recommended | **$18** |

### Experiment A totals

| Configuration | Agent | Judge | P(eval) | **Total** |
|---------------|-------|-------|---------|-----------|
| Opus agent, Gemini judge, 2 judges + P(eval) ← **recommended** | $472 | $19 | $18 | **~$509** |
| Opus agent, Gemini judge, all 5 judges + P(eval) | $472 | $94 | $18 | ~$584 |
| Opus agent, Opus self-judge, 2 judges + P(eval) | $472 | $69 | $70 | ~$611 |
| Opus agent, Opus self-judge, all 5 judges + P(eval) | $472 | $354 | $70 | ~$896 |
| Opus agent, 2 judges, skip P(eval) | $472 | $69 | — | ~$541 |

Add 20% buffer for retries/failures → **$600–$1,100 range** for Experiment A.

---

## Full multi-model, multi-condition experiment

### Cost per 100 tasks per model per benchmark (with reasoning where supported)

| Model | DoomArena | WASP | SafeArena | Per model |
|-------|-----------|------|-----------|-----------|
| opus-4.6 :thinking | $150 | $200 | $122 | **$472** |
| sonnet-4.6 :thinking | $81 | $108 | $65 | **$254** |
| gpt-4.1 (no reasoning) | $35 | $46 | $28 | **$109** |
| gemini-2.5-pro :thinking | $45 | $60 | $36 | **$141** |
| claude-3.7-sonnet :thinking | $81 | $108 | $65 | **$254** |
| o4-mini :thinking | $50 | $70 | $42 | **$162** |

### Total agent-run cost (6 models × 3 benchmarks × 100 tasks × 3 conditions)

Multiply "per model" by 3 conditions:

| Model | Total (3 conditions) |
|-------|---------------------|
| opus-4.6 | $1,416 |
| sonnet-4.6 | $762 |
| gpt-4.1 | $327 |
| gemini-2.5-pro | $423 |
| claude-3.7-sonnet | $762 |
| o4-mini | $486 |
| **Subtotal (all 6 models)** | **~$4,176** |

### Judging cost (full experiment)

5,400 trajectories (6 models × 3 benchmarks × 100 tasks × 3 conditions):

| Judge | All 5 judges | 2 judges (TAS + verbalized) |
|-------|--------------|------------------------------|
| Opus | $4,833 | $945 |
| Sonnet | $2,900 | $567 |
| **Gemini** ← recommended | **$1,285** | **$259** |

### P(eval) on environments (all 6 models)

~90 new pages × 10 prompts × 3 formats × 6 models = 16,200 calls × ~$0.007/call (Gemini) = **~$113**.

(At Opus: **~$405**.)

### Full experiment grand total

Recommended stack: Opus/Sonnet/Gemini/etc agents with reasoning, **Gemini 2.5 Pro as judge**, TAS + verbalized only.

| Component | Cost |
|-----------|------|
| Agent runs (6 models, reasoning) | $4,176 |
| Judging (5,400 trajectories × 2 judges, Gemini) | $259 |
| P(eval) on environments (Gemini) | $113 |
| Buffer (20%) | $910 |
| **Total** | **~$5,460** |

Upgrade to all 5 judges: **~$6,600**.
Upgrade to Opus as judge (full 5 judges): **~$10,200**.

### Cost-reduced alternatives

| Strategy | Savings | New total |
|----------|---------|-----------|
| Drop reasoning mode everywhere | ~30% | ~$3,800 |
| Skip opus-4.6 | ~$1,400 | ~$4,000 |
| n=50 instead of n=100 | ~50% | ~$2,700 |
| Run only baseline + eval_cue (skip scenario_cue) | ~33% | ~$3,650 |
| Only opus-4.6 + sonnet-4.6 + gpt-4.1 first pass | skips 3 models | ~$3,000 |

---

## Recommended staged approach

**Phase 1 — Experiment A pilot** (~$500): Opus 4.6 baseline on n=100 per safety benchmark, Gemini as judge with 2-judge set + P(eval). Validate the pipeline and get initial awareness rates.

**Phase 2 — Multi-model breadth** (~$1,800): Add sonnet-4.6, gpt-4.1, gemini-2.5-pro baselines at n=100 to show Experiment A generalizes across models. Gemini judging.

**Phase 3 — Causal experiment** (~$3,200): Opus 4.6 + Sonnet 4.6 + GPT-4.1 on 3 conditions × 3 benchmarks × n=100. Produces the core causal result for Experiment C.

**Phase 4 — Full breadth** (~$1,500): Add claude-3.7-sonnet + o4-mini to whichever conditions are most informative from Phase 3.

**Staged total**: ~$7,000 over 4 rounds with go/no-go decisions between phases.

---

## Key caveats

1. **Step counts are rough estimates** — adversarial tasks may need more steps than benign ones. Expect ±30% variance.
2. **Reasoning token budgets** vary wildly between models. Opus "high" effort emits more tokens than o4-mini "medium". Calibrate on n=5 per benchmark first.
3. **Failed/timed-out runs** don't produce usable trajectories but still cost money. Budget 20% buffer.
4. **Judge cost scales linearly** with trajectory length. If agents hit the 30-step limit on DoomArena, judging costs can double.
5. **P(eval) on screenshots** requires vision-capable judges and is more expensive than text-only formats.
6. **With the corrected Opus 4.6 pricing ($5/$25)**, it's only ~1.7x more expensive than Sonnet on input and ~1.7x on output — no longer a prohibitive cost tier. Using Opus as a flagship agent is affordable.
