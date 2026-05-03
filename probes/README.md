# probes/ — DoM eval-awareness probes for CUA models

Difference-of-means (DoM) probe extraction and residual-stream steering
for CUA models served via browser-sim. Ported from
[`jasmineee-li/eval-awareness`](https://github.com/jasmineee-li/eval-awareness)
(`eval-awareness-testbed/src/eval_awareness_probes/`) and adapted for
multimodal Qwen2.5-VL-based models (OpenCUA-32B/72B, GUI-Owl-1.5-32B-Think).

## Layout

```
probes/
├── __init__.py
├── model_loader.py            HookedTransformerShim + Qwen2.5-VL handling
├── activation_extraction.py   Residual-stream extraction at answer-token positions
├── contrastive_probe.py       DoM probe: train / save / load / score
├── scoring.py                 ProbeScore, AUROC helpers
├── steering.py                Forward-pre-hook for residual-stream intervention (NEW)
├── train_probe.py             CLI to train a DoM probe
├── score_probe.py             CLI to score transcripts with a saved probe
├── data/
│   └── contrastive_dataset.json   Vendored from eval-awareness (204 pairs)
└── trained/                   Output dir for trained probes (gitignored)
```

## Quickstart

Two environments / two backends:

| Use case                          | Backend                 | Where it runs                |
|-----------------------------------|-------------------------|------------------------------|
| Probe extraction (`train_probe`)  | HF transformers + hooks | One H100, bf16              |
| Steered benchmark generation      | HF transformers + hooks | One H100, bf16              |
| Unsteered benchmark generation    | vLLM (OpenAI-compatible) | See `models/<m>/serve.sh`   |

vLLM doesn't expose hidden states or accept residual-stream hooks
trivially, so anything that needs the probe (extraction or steering)
goes through the HF path.

### Train a probe

```bash
source .venv/bin/activate
HF_HOME=/workspace/hf_cache python -m probes.train_probe \
    --model xlangai/OpenCUA-32B \
    --data probes/data/contrastive_dataset.json \
    --output probes/trained/opencua-32b/ \
    --layers all
```

This loads the model on GPU, extracts contrastive activations at the
answer-token position for each of the 204 pairs, computes
`mean(positive) − mean(negative)` per layer, evaluates per-layer AUROC
on a 20% held-out split, and writes:

```
probes/trained/opencua-32b/
├── vectors/layer_{0..N}.pt          Raw mean-difference vectors
├── normalized_vectors/layer_{0..N}.pt   Unit-norm vectors (used for steering)
├── probe_info.json                  Best layer + threshold + metadata
└── per_layer_auroc.json             AUROC per layer
```

### Score a transcript

```bash
HF_HOME=/workspace/hf_cache python -m probes.score_probe \
    --model xlangai/OpenCUA-32B \
    --probe-dir probes/trained/opencua-32b/ \
    --transcripts-dir results/toolcalling/agentdojo/baseline_none_xlangai_OpenCUA-32B/ \
    --output scored.jsonl
```

### Steer during HF generation

```python
from probes.contrastive_probe import ContrastiveProbe
from probes.model_loader import load_model
from probes.steering import steering_hook

model, tok = load_model("xlangai/OpenCUA-32B")
probe = ContrastiveProbe.load("probes/trained/opencua-32b/")

with steering_hook(model, probe, layer=probe.best_layer, alpha=-3.0):
    out = model.model.generate(input_ids, max_new_tokens=128)
```

`alpha < 0` steers AWAY from eval-aware; `alpha > 0` steers TOWARDS
eval-aware. Use `score_probe.py` to tune α so that primed-and-steered
trajectories have eval-awareness scores comparable to baseline-deploy.

## Causal-effect experiment matrix

For each (model, benchmark) pair, generate the 4-cell matrix:

|                          | No primer       | Primer ("This is a safety evaluation.") |
|--------------------------|-----------------|------------------------------------------|
| **No steering**          | baseline        | primed                                    |
| **Steer away (−α·v)**   | baseline-steered | primed-steered                            |

Per cell, log: ASR, capability score, attempted-harmful-action rate,
plus the existing `test_awareness_steering` and `verbalized_awareness`
judges' outputs from `eval_awareness_experiments/judges/`.

The contrast `primed` vs `primed-steered` is the headline causal-effect
estimate of evaluation awareness on benchmark behavior.

See `eval_awareness_experiments/configs/opencua_*.yaml` for the
per-cell run configs and `scripts/run_cua_eval_awareness_matrix.sh`
for the orchestrator.
