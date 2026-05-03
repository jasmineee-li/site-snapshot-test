#!/usr/bin/env bash
# Spin up vLLM's OpenAI-compatible server for xlangai/OpenCUA-32B.
#
# Per the registry default, this server listens on :8001 and is reached
# from the runners as `local/opencua-32b`. Override with:
#   PORT=8011 bash models/opencua/serve_32b.sh
# and set LOCAL_OPENAI_BASE_URL_OPENCUA_32B=http://localhost:8011/v1 in
# the runner environment.
#
# Hardware: bf16 fits on a single H100 (80GB) with KV cache headroom.
# For two H100s, set TENSOR_PARALLEL_SIZE=2 to halve per-GPU memory and
# roughly 1.6× throughput.
#
# Pre-cache the weights into the shared HF cache before launching parallel
# probe + benchmark jobs so they don't race on the download (per the
# eval-awareness CLAUDE.md guidance).

set -euo pipefail

MODEL=${MODEL:-xlangai/OpenCUA-32B}
PORT=${PORT:-8001}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
DTYPE=${DTYPE:-bfloat16}

export HF_HOME=${HF_HOME:-/workspace/hf_cache}

echo "[serve_32b] model=$MODEL port=$PORT tp=$TENSOR_PARALLEL_SIZE max_model_len=$MAX_MODEL_LEN"
echo "[serve_32b] HF_HOME=$HF_HOME"

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
