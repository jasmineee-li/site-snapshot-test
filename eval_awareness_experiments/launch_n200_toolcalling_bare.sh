#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {agentdojo|injecagent}" >&2
  exit 2
fi

BENCHMARK="$1"
if [[ "$BENCHMARK" != "agentdojo" && "$BENCHMARK" != "injecagent" ]]; then
  echo "unknown benchmark: $BENCHMARK" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
OUTPUT_BASE="$REPO_ROOT/eval_awareness_experiments/results/n200_2026-04-29"
LOG_DIR="$OUTPUT_BASE/logs"
mkdir -p "$LOG_DIR"

LOG_PATH="$LOG_DIR/${BENCHMARK}_n200_bare_$(date -u +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_PATH") 2>&1

cd "$REPO_ROOT"

echo "started_at=$(date -u --iso-8601=seconds)"
echo "benchmark=$BENCHMARK"
echo "output_base=$OUTPUT_BASE"
echo "log_path=$LOG_PATH"

.venv/bin/python -m eval_awareness_experiments.run_causal_experiment \
  --models \
    anthropic/claude-opus-4.7:thinking \
    anthropic/claude-sonnet-4.6:thinking \
    z-ai/glm-5:thinking \
    openai/gpt-5.2:thinking \
    google/gemini-2.5-pro:thinking \
    moonshotai/kimi-k2.5:thinking \
  --benchmarks "$BENCHMARK" \
  --conditions baseline \
  --extra-instructions-presets none \
  --system-prompt-frames none \
  --judges verbalized_awareness purpose_continue_5q \
  --judge-model anthropic/claude-opus-4.7:thinking \
  --tasks-per-split 200 \
  --stage all \
  --skip-existing \
  --output-base "$OUTPUT_BASE"

echo "finished_at=$(date -u --iso-8601=seconds)"
