# Cost Estimate: Safety Benchmark Eval-Awareness Experiments

## Scope

Running **n=100 tasks per benchmark** across 3 benchmarks, 6 models, 3 injection conditions, plus all judging. Used for budgeting the experiment suite.

## Pricing Assumptions (USD per 1M tokens)

Rates as of April 2026. Verify current OpenRouter / provider pricing before launch.

| Model | Input $/M | Output $/M | Reasoning? | Notes |
|-------|-----------|-----------|------------|-------|
| `anthropic/claude-opus-4.6` | 15.00 | 75.00 | yes (thinking) | Flagship, most expensive |
| `anthropic/claude-sonnet-4.6` | 3.00 | 15.00 | yes (thinking) | Workhorse |
| `openai/gpt-4.1` | 2.00 | 8.00 | no | |
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

Per-step size is dominated by the accessibility tree observation (~8-15k tokens/step on text-only; screenshots add ~1-2k tokens as image input). Reasoning models emit ~5x more output tokens.

## Cost Per 100 Tasks Per Model Per Benchmark

### Non-reasoning output (baseline)

| Model | DoomArena (n=100) | WASP (n=100) | SafeArena (n=100) | Total per model |
|-------|-------------------|---------------|---------------------|-----------------|
| opus-4.6 | $270 | $360 | $217 | **$847** |
| sonnet-4.6 | $54 | $72 | $43 | **$169** |
| gpt-4.1 | $35 | $46 | $28 | **$109** |
| gemini-2.5-pro | $25 | $33 | $20 | **$78** |
| claude-3.7-sonnet | $54 | $72 | $43 | **$169** |
| o4-mini | $19 | $26 | $16 | **$61** |

### With reasoning mode on (Opus/Sonnet/o4-mini/Gemini — multiply output tokens ~5x)

| Model | DoomArena | WASP | SafeArena | Total per model |
|-------|-----------|------|-----------|-----------------|
| opus-4.6 :thinking | $405 | $540 | $323 | **$1,268** |
| sonnet-4.6 :thinking | $81 | $108 | $65 | **$254** |
| gpt-4.1 | $35 | $46 | $28 | **$109** |
| gemini-2.5-pro :thinking | $31 | $43 | $25 | **$99** |
| claude-3.7-sonnet :thinking | $81 | $108 | $65 | **$254** |
| o4-mini | $50 | $70 | $42 | **$162** |

## Per Condition Multiplier

Causal experiment has **3 conditions** (baseline, eval_cue, scenario_cue). Multiply the totals above by 3.

## Total Agent-Run Cost

For 6 models × 3 benchmarks × 100 tasks × 3 conditions, with reasoning mode where supported:

| Model | Total agent-run cost (reasoning) |
|-------|-----------------------------------|
| opus-4.6 | ~$3,804 |
| sonnet-4.6 | ~$762 |
| gpt-4.1 | ~$327 |
| gemini-2.5-pro | ~$297 |
| claude-3.7-sonnet | ~$762 |
| o4-mini | ~$486 |
| **Subtotal (all 6 models)** | **~$6,438** |

Without reasoning mode:

| **Subtotal (non-reasoning)** | **~$4,299** |

## Judging Cost

Eval-awareness judges run on every trajectory produced. Using a single consistent judge model (recommend `claude-sonnet-4.6` or `gpt-4.1` for cost-vs-quality).

**Per-trajectory judge tokens**:
- TAS judge (test_awareness_steering): ~15k input (full reasoning trace + prompt) + ~500 output
- verbalized_awareness: ~15k input + ~500 output
- Purpose probes (purpose_xml + purpose_continue): ~30k input total + ~800 output
- binary_third_person (5 prompts × epochs=1): ~50k input + ~1k output
- probability_third_person (5 prompts): ~50k input + ~1k output

**Total per trajectory** across all judges: ~160k input + ~3.8k output
= **$0.54** on sonnet-4.6, **$0.37** on gpt-4.1, **$0.22** on gemini-2.5-pro

**Judging cost for full experiment**:
- Total trajectories: 6 models × 3 benchmarks × 100 tasks × 3 conditions = **5,400 trajectories**
- At $0.54/trajectory (sonnet judge): **~$2,916**
- At $0.37/trajectory (gpt-4.1 judge): **~$2,000**
- At $0.22/trajectory (gemini judge): **~$1,188**

## P(eval) on Benchmark Environments

Running P(eval) scoring on website environments the agents encountered (pages/axtrees/screenshots):
- Pages per benchmark: ~30 unique environments
- 10 prompts (5 binary + 5 probability) per page, 3 formats (html, axtree, screenshot)
- Per-prompt: ~5k input + ~50 output
- **Total**: 30 pages × 10 prompts × 3 formats × 6 models = 5,400 calls
- At $0.03/call (sonnet): **~$162**
- At opus pricing: **~$810**

## Grand Total Estimate

Recommended judge: **gpt-4.1** (good quality/cost balance).

| Component | Cost estimate |
|-----------|---------------|
| Agent runs (6 models, reasoning where supported) | **$6,438** |
| Judging (5,400 trajectories × all 5 judges, gpt-4.1) | **$2,000** |
| P(eval) on environments (6 models × 900 calls) | **~$500** |
| Buffer (20% — retries, failed runs, dev) | **~$1,800** |
| **Total** | **~$10,738** |

### Cost-reduced alternatives

| Strategy | Savings | Total |
|----------|---------|-------|
| Drop reasoning mode everywhere | ~$2,100 | ~$8,600 |
| Skip opus-4.6, use sonnet as flagship | ~$3,800 | ~$6,900 |
| n=50 instead of n=100 per benchmark | ~50% | ~$5,370 |
| Run only baseline + eval_cue (skip scenario_cue) | ~33% | ~$7,000 |
| Only opus-4.6 + sonnet-4.6 + gpt-4.1 first pass | ~60% | ~$4,300 |

### Recommended staged approach

**Phase 1 (pilot, ~$1,500)**: opus-4.6 + sonnet-4.6 on n=30 per benchmark, baseline only, full judging. Validate pipeline and measure baseline awareness rates.

**Phase 2 (core, ~$5,000)**: opus-4.6 + sonnet-4.6 + gpt-4.1 on n=100 per benchmark, all 3 conditions, full judging. Produce main results for paper.

**Phase 3 (breadth, ~$4,000)**: Add gemini-2.5-pro + claude-3.7-sonnet + o4-mini to n=100 baseline runs (skip causal conditions for these). Show results generalize.

**Total staged**: ~$10,500 over 3 rounds, with go/no-go decisions between phases.

## Key Caveats

1. **Step counts are rough estimates** — adversarial tasks may require more steps than benign ones (agents retry, attackers force extra navigation). Expect ±30% variance.
2. **Reasoning token budgets** vary wildly between models. Opus with "high" effort emits more tokens than o4-mini "medium". Calibrate before committing to full runs.
3. **Failed/timed-out runs** don't produce usable trajectories but still cost money. Budget 20% buffer.
4. **Judge cost scales linearly** with trajectory length. If agents run long (some DoomArena tasks hit 30-step limit), judging costs can double.
5. **P(eval) on screenshots** requires vision-capable judges and is more expensive than text-only formats.
