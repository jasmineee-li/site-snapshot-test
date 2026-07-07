#!/usr/bin/env bash
set -euo pipefail

# Recompute only verbalized eval awareness (VEA) with the structured-output
# Sonnet 4.6 judge. Results go to a fresh tree; source n=200 outputs are not
# modified.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-$REPO_ROOT/eval_awareness_experiments/results/n200_2026-04-29}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/eval_awareness_experiments/results/n200_2026-04-29_vea_structured_sonnet46_20260509}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs/vea_structured_sonnet46_20260509}"

JUDGE_MODEL="${JUDGE_MODEL:-anthropic/claude-sonnet-4.6}"

# Per benchmark, the existing launcher runs MODEL_WORKERS cells concurrently,
# each with JUDGE_CONCURRENCY calls. Defaults: 4 * 4 = 16 per benchmark.
MODEL_WORKERS="${MODEL_WORKERS:-4}"
JUDGE_CONCURRENCY="${JUDGE_CONCURRENCY:-4}"
JUDGE_RETRIES="${JUDGE_RETRIES:-7}"

BENCHMARKS=("$@")
if [ "${#BENCHMARKS[@]}" -eq 0 ]; then
  BENCHMARKS=(wasp doomarena agentdojo injecagent)
fi

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

echo "source_root=$SOURCE_ROOT"
echo "output_root=$OUTPUT_ROOT"
echo "log_dir=$LOG_DIR"
echo "judge_model=$JUDGE_MODEL"
echo "benchmarks=${BENCHMARKS[*]}"
echo "per_benchmark_concurrency=$((MODEL_WORKERS * JUDGE_CONCURRENCY))"
echo "estimated_total_concurrency=$(( ${#BENCHMARKS[@]} * MODEL_WORKERS * JUDGE_CONCURRENCY ))"

pids=()
for benchmark in "${BENCHMARKS[@]}"; do
  log_path="$LOG_DIR/${benchmark}.log"
  echo "[launch] $benchmark -> $log_path"
  (
    cd "$REPO_ROOT"
    "$PYTHON_BIN" -m eval_awareness_experiments.rejudge_agent_purpose_5q_n200 \
      --benchmarks "$benchmark" \
      --arms bare xml_control xml_safety xml_scenario \
      --source-root "$SOURCE_ROOT" \
      --output-root "$OUTPUT_ROOT" \
      --judges verbalized_awareness \
      --judge-model "$JUDGE_MODEL" \
      --judge-concurrency "$JUDGE_CONCURRENCY" \
      --judge-retries "$JUDGE_RETRIES" \
      --model-workers "$MODEL_WORKERS"
  ) >"$log_path" 2>&1 &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    rc=1
  fi
done

if [ "$rc" -eq 0 ]; then
  echo "[done] all VEA rejudge benchmarks completed"
else
  echo "[done] one or more VEA rejudge benchmarks failed; see $LOG_DIR" >&2
fi

exit "$rc"
