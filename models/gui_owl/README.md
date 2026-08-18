# GUI-Owl

Adapter for `mPLUG/GUI-Owl-1.5-32B-Think`.

| short id              | HF repo                       | port | hardware (bf16) |
|-----------------------|-------------------------------|------|-----------------|
| `gui-owl-32b-think`   | mPLUG/GUI-Owl-1.5-32B-Think | 8003 | 1× H100 (TP=1)  |

The Think variant emits explicit reasoning traces, which makes it
particularly useful for eval-awareness experiments — we can probe at
the end-of-reasoning token specifically, and the trace also gives us a
behavioral signal of awareness ("verbalized awareness") to compare
against the probe's representational signal.

## Serve

```bash
bash models/gui_owl/serve_32b.sh
```

## Reach from a runner

```bash
python -m eval_awareness_experiments.run_safety_pipeline \
    --benchmark agentdojo \
    --model-name local/gui-owl-32b-think \
    --condition baseline
```

## Train a probe

```bash
HF_HOME=/workspace/hf_cache python -m probes.train_probe \
    --model mPLUG/GUI-Owl-1.5-32B-Think \
    --output probes/trained/gui-owl-32b-think/
```
