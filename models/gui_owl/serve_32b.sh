#!/usr/bin/env bash
# Spin up vLLM's OpenAI-compatible server for xlangai/GUI-Owl-1.5-32B-Think.
#
# Listens on :8003 by default; runners reach it as `local/gui-owl-32b-think`.
# Single H100 in bf16 with TP=1 fits comfortably.

set -euo pipefail

MODEL=${MODEL:-xlangai/GUI-Owl-1.5-32B-Think}
PORT=${PORT:-8003}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
DTYPE=${DTYPE:-bfloat16}

export HF_HOME=${HF_HOME:-/workspace/hf_cache}

echo "[serve_32b_think] model=$MODEL port=$PORT tp=$TENSOR_PARALLEL_SIZE max_model_len=$MAX_MODEL_LEN"
echo "[serve_32b_think] HF_HOME=$HF_HOME"

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
