# models/

Local CUA model adapters. Each subpackage covers one model family and
exposes a vLLM serve script plus shared HF-loading + OpenAI-client
factories via `models.common`.

```
models/
├── common/
│   ├── registry.py      LocalModelSpec + LOCAL_MODELS dict
│   ├── vllm_client.py   AsyncOpenAI factory + AgentLab BaseModelArgs factory
│   └── hf_loader.py     HookedTransformerShim wrapper for HF probing
├── opencua/
│   ├── serve_32b.sh     vLLM server for xlangai/OpenCUA-32B (port 8001)
│   └── serve_72b.sh     vLLM server for xlangai/OpenCUA-72B (port 8002, TP>=2)
└── gui_owl/
    └── serve_32b.sh     vLLM server for mPLUG/GUI-Owl-1.5-32B-Think (port 8003)
```

## Registered short ids

| short id              | HF repo                       | port | benchmarks | probing |
|-----------------------|-------------------------------|------|------------|---------|
| `opencua-32b`         | xlangai/OpenCUA-32B           | 8001 | yes        | yes     |
| `opencua-72b`         | xlangai/OpenCUA-72B           | 8002 | yes        | no (1st pass) |
| `gui-owl-32b-think`   | mPLUG/GUI-Owl-1.5-32B-Think | 8003 | yes        | yes     |

Reach a model from a runner via `local/<short-id>`:

```bash
python -m eval_awareness_experiments.run_safety_pipeline \
    --benchmark agentdojo \
    --model-name local/opencua-32b
```

## Adding a new model

1. Add a `LocalModelSpec(short_id="...", hf_repo="...", default_url="...")`
   to `LOCAL_MODELS` in `models/common/registry.py`.
2. Drop a `models/<family>/serve_*.sh` that runs `vllm serve <repo> --port <port>`.
3. Confirm a `local/<short-id>` invocation lands on the new server.

`probes/model_loader._load_vl()` selects the loader class. It reads
`model_type` from the HF config and handles two values today:

| `model_type` | loader class | models |
|---|---|---|
| `qwen2_5_vl` | `Qwen2_5_VLForConditionalGeneration` | OpenCUA-32B/72B |
| `qwen3_vl` | `Qwen3VLForConditionalGeneration` | GUI-Owl-1.5-32B-Think |

Unknown values fall back to `AutoModelForVision2Seq`, then to `AutoModel`.
To add a third architecture, add a branch to `_load_vl()`.
