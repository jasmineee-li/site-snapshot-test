# Run the causal eval-awareness pilot (collaborator quickstart)

One-page runbook. For experimental design, hypothesis, cost estimate,
and full per-arm interpretation, see
[`docs/CAUSAL_EVAL_AWARENESS_EXPERIMENT.md`](docs/CAUSAL_EVAL_AWARENESS_EXPERIMENT.md).

## Prerequisites

- Python 3.12 + [`uv`](https://github.com/astral-sh/uv) installed
- A repo venv at `./.venv` (run `uv sync` once if missing)
- An `OPENROUTER_API_KEY` set in a `./.env` file at the repo root
- Disk space:
  - **Tool-calling only (AgentDojo + InjecAgent)**: ~2 GB free
  - **Full pilot (+ DoomArena + WASP)**: ~30–500 GB free for WebArena Docker
    images (DoomArena and WASP both rely on running WebArena hosts; this
    is by far the heaviest prerequisite). If you don't have it, just
    drop browser benchmarks — see "Skipping browser benchmarks" below.

## Two commands

```bash
# 1. One-time dependency setup (idempotent — safe to re-run)
./eval_awareness_experiments/setup_pilot_deps.sh

# 2. Launch all 16 streams in parallel (4 arms × 4 benchmarks)
./eval_awareness_experiments/launch_pilot.sh
```

Wall-clock target: **~30–45 minutes**. Cost target: **~$1,000** at
n=20 per cell. Tool-calling-only variant is **~$450**.

## What it runs

6 models × 4 benchmarks × 4 arms = 96 cells, each at n=20 tasks:

| Models (all `:thinking`) | Benchmarks | Arms |
|---|---|---|
| `z-ai/glm-5` | `agentdojo` | `bare` (preset=none, frame=none) |
| `anthropic/claude-sonnet-4.6`* | `injecagent` | `xml_safety` (frame=`safety_test`) |
| `anthropic/claude-opus-4.7` | `doomarena` | `xml_scenario` (frame=`hypothetical_scenario`) |
| `openai/gpt-5.2` | `wasp` | `native` (preset=`scratchpad_native`) |
| `google/gemini-3-flash-preview` | | |
| `google/gemini-3.1-pro-preview` | | |

\* Sonnet 4.7 isn't on OpenRouter as of 2026-04-26; using 4.6 as
substitute.

The two XML-frame arms wrap the entire system prompt in
`<llm_safety_test>...</llm_safety_test>` or
`<hypothetical_scenario>...</hypothetical_scenario>` — the **causal
manipulation** the experiment is testing. `bare` is the clean
baseline, `native` is the production-realistic elicitation arm
(no XML, distilled from the alignment-faking paper).

## What gets created

```
eval_awareness_experiments/results/causal_pilot/
└── <benchmark>/
    └── <arm>/
        └── <model_slug>/
            └── <benchmark>/<split>/
                ├── *.json              ← raw trajectories
                ├── trajectory_awareness_results.jsonl  ← judge results
                └── markdown/*.md       ← human-readable sidecars

logs/causal_pilot/<arm>__<benchmark>.log    (16 files, one per stream)
```

## Monitoring

```bash
# Web dashboard (recommended) — refresh as cells complete
.venv/bin/python -m eval_awareness_experiments.dashboard \
    --results-dir eval_awareness_experiments/results/causal_pilot --port 8000
# → http://localhost:8000

# CLI alternative
watch -n 30 '.venv/bin/python -m eval_awareness_experiments.run_manifest \
    --results-dir eval_awareness_experiments/results/causal_pilot --print'

# Tail individual stream logs
tail -f logs/causal_pilot/*.log
```

## Killing / restarting

```bash
# Kill all running streams
pkill -f "run_causal_experiment.*causal_pilot"

# Then verify nothing's still alive
ps aux | grep -E "run_safety_pipeline|run_agentdojo|run_wasp" | grep -v grep

# Restart — --skip-existing on every stream means already-complete
# cells get skipped, only missing ones re-run
./eval_awareness_experiments/launch_pilot.sh
```

## Skipping browser benchmarks

DoomArena and WASP need running **WebArena Docker hosts** (separate
Dockerized services serving GitLab/Reddit/Shopping clones). If you
don't have those configured, those cells will fail. To skip them and
run only the tool-calling track (AgentDojo + InjecAgent):

Edit `launch_pilot.sh` and change:

```bash
BENCHMARKS=(agentdojo injecagent doomarena wasp)
```

to

```bash
BENCHMARKS=(agentdojo injecagent)
```

This drops 8 of 16 streams and ~50% of the cost. The XML-frame causal
contrast is fully testable on tool-calling benchmarks alone — the
headline result of the experiment doesn't require browser benchmarks.

## Smoke test before launch (recommended)

Verifies all 6 model IDs resolve on OpenRouter and `:thinking` mode
engages. ~$0.10, ~30 seconds.

```bash
.venv/bin/python -m eval_awareness_experiments.smoke_test_models
```

GLM-5 occasionally fails on the first call (transient — doesn't engage
reasoning). Re-run if it does. Other failures usually mean an
OpenRouter ID doesn't exist (verify on
https://openrouter.ai/api/v1/models).

## Reproducing existing GLM-5 + AgentDojo + bare

We have one prior run that maps directly into this matrix:
`results/n20_glm5_none_v2_20260425_2009`. It's the same `(model,
benchmark, arm)` cell as the new pilot. `--skip-existing` won't pick
it up (different output dir layout) but the headline numbers should
match — useful as a sanity check.

## After the pilot

Once cells start landing, look at the dashboard or run `run_manifest`.
Key contrasts to compute downstream:

- **ASR(bare) vs ASR(xml_safety)** — does forcing eval-awareness
  reduce attack success? Per model, per benchmark.
- **VEA(bare) → VEA(xml_safety)** — sanity check that the manipulation
  worked. We expect VEA to spike to ~100% on xml_safety arms.
- **5PQ(bare)** — natural eval-awareness baseline by model.
  Correlate with the ASR delta to test "natural awareness predicts
  manipulation effectiveness".

For the canonical 3-table presentation format used by all eval-awareness
runs, see [`results_presentation_guide.md`](results_presentation_guide.md).

## Where to ask questions

- Methodology / hypothesis: `docs/CAUSAL_EVAL_AWARENESS_EXPERIMENT.md`
- Existing GLM-5 findings (background): `docs/GLM5_SCRATCHPAD_ANALYSIS.md`
- Code: each module has a docstring; `run_causal_experiment.py` is the
  matrix runner entry point.
