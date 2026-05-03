# OpenCUA

Adapters for `xlangai/OpenCUA-32B` and `xlangai/OpenCUA-72B`.

| short id        | HF repo                  | port | hardware (bf16)              |
|-----------------|--------------------------|------|------------------------------|
| `opencua-32b`   | xlangai/OpenCUA-32B      | 8001 | 1× H100 (TP=1)               |
| `opencua-72b`   | xlangai/OpenCUA-72B      | 8002 | 2× H100 (TP=2), vLLM-only    |

## Serve

```bash
# 32B (single GPU)
bash models/opencua/serve_32b.sh

# 72B (TP=2)
bash models/opencua/serve_72b.sh

# Override port + GPUs
PORT=8011 TENSOR_PARALLEL_SIZE=2 bash models/opencua/serve_32b.sh
```

## Reach from a runner

Use the model id `local/opencua-32b` (or `local/opencua-72b`):

```bash
python -m eval_awareness_experiments.run_safety_pipeline \
    --benchmark agentdojo \
    --model-name local/opencua-32b \
    --condition baseline \
    --extra-instructions-preset none
```

The runners read the URL from the registry default
(`http://localhost:8001/v1` for 32B); override per-host with
`LOCAL_OPENAI_BASE_URL_OPENCUA_32B=http://other-host:8001/v1`.

## Train a probe

```bash
HF_HOME=/workspace/hf_cache python -m probes.train_probe \
    --model xlangai/OpenCUA-32B \
    --output probes/trained/opencua-32b/
```

72B probe training is gated on TP-aware HF loading or 4-bit
quantization — out of scope for the first pass.
