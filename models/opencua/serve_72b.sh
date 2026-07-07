#!/usr/bin/env bash
# Spin up vLLM's OpenAI-compatible server for xlangai/OpenCUA-72B.
#
# Listens on :8002 by default; runners reach it as `local/opencua-72b`.
# Requires TENSOR_PARALLEL_SIZE>=2 in bf16 (won't fit on a single H100).
#
# 72B is benchmark-only (HF probing/steering on a single host needs
# either TP>=2 or 4/8-bit quantization — out of scope for the first pass).

set -euo pipefail

MODEL=${MODEL:-xlangai/OpenCUA-72B}
PORT=${PORT:-8002}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-2}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
DTYPE=${DTYPE:-bfloat16}

export HF_HOME=${HF_HOME:-/workspace/hf_cache}

echo "[serve_72b] model=$MODEL port=$PORT tp=$TENSOR_PARALLEL_SIZE max_model_len=$MAX_MODEL_LEN"
echo "[serve_72b] HF_HOME=$HF_HOME"

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --host 0.0.0.0 \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-model-len "$MAX_MODEL_LEN" \
    --dtype "$DTYPE" \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser hermes
