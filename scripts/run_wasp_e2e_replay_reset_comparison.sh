#!/usr/bin/env bash
# Compare WASP e2e replay with and without hard per-cell resets.
#
# This is a replay-only comparison: it never queries target models. It first
# prepares a clean planted starting state for the stacks used by the saved n200
# run, then runs:
#   1. RESET_STACK_BEFORE_CELL=0  (dirty cross-cell replay)
#   2. RESET_STACK_BEFORE_CELL=1  (hard recreate + replant before each cell)

set -euo pipefail

cd "$(dirname "$0")/.."

RESULTS_DIR="${RESULTS_DIR:-eval_awareness_experiments/results/n200_2026-04-29}"
OUT_BASE="${OUT_BASE:-$RESULTS_DIR/wasp_e2e_replay_20260509_compare}"
LOG_DIR="${LOG_DIR:-logs/wasp_e2e_replay_compare_20260509}"
PREPARE_STACKS="${PREPARE_STACKS:-default gpt gemini25 kimi25}"
MAX_TASKS="${MAX_TASKS:-}"
PHASES="${PHASES:-no_reset reset_per_cell}"

mkdir -p "$LOG_DIR"

stack_ports() {
    case "$1" in
        default) echo "default 9001 8080" ;;
        gpt) echo "gpt 9231 8231" ;;
        gemini25) echo "gemini25 9241 8241" ;;
        kimi25) echo "kimi25 9251 8251" ;;
        glm) echo "glm 9201 8201" ;;
        sonnet) echo "sonnet 9211 8211" ;;
        opus) echo "opus 9221 8221" ;;
        *)
            echo "unknown WASP stack: $1" >&2
            return 1
            ;;
    esac
}

run_replay() {
    local label=$1
    local reset_before_cell=$2
    local out_dir="$OUT_BASE/$label"

    echo "=== replay label=$label reset_stack_before_cell=$reset_before_cell out_dir=$out_dir ==="
    if [ -n "$MAX_TASKS" ]; then
        OUT_DIR="$out_dir" RESET_STACK_BEFORE_CELL="$reset_before_cell" MAX_TASKS="$MAX_TASKS" \
            ./scripts/run_wasp_e2e_replay_n200.sh
    else
        OUT_DIR="$out_dir" RESET_STACK_BEFORE_CELL="$reset_before_cell" \
            ./scripts/run_wasp_e2e_replay_n200.sh
    fi
}

echo "results_dir=$RESULTS_DIR"
echo "out_base=$OUT_BASE"
echo "log_dir=$LOG_DIR"
echo "prepare_stacks=$PREPARE_STACKS"
echo "max_tasks=${MAX_TASKS:-all}"
echo "phases=$PHASES"

echo
echo "=== preparing clean one-time planted starting state ==="
for stack in $PREPARE_STACKS; do
    read -r slug gitlab_port reddit_port <<<"$(stack_ports "$stack")"
    echo "=== prepare stack=$slug gitlab_port=$gitlab_port reddit_port=$reddit_port ==="
    ./scripts/wasp_reset_stack.sh "$slug" "$gitlab_port" "$reddit_port"
done

echo
if [[ " $PHASES " == *" no_reset "* ]]; then
    run_replay "no_reset" 0
else
    echo "=== skipping no_reset phase ==="
fi

echo
if [[ " $PHASES " == *" reset_per_cell "* ]]; then
    run_replay "reset_per_cell" 1
else
    echo "=== skipping reset_per_cell phase ==="
fi

echo
echo "=== summary ==="
OUT_BASE="$OUT_BASE" .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

out_base = Path(os.environ["OUT_BASE"])
for label in ("no_reset", "reset_per_cell"):
    root = out_base / label
    print(f"[{label}] {root}")
    summaries = []
    for path in sorted(root.glob("*.jsonl")):
        summary = None
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get("type") == "summary":
                    summary = row
        if summary:
            summaries.append((path.name, summary))
            asr = summary.get("asr")
            asr_str = "NA" if asr is None else f"{100 * asr:.2f}%"
            print(
                f"  {path.name}: n={summary.get('n_scored')} "
                f"success={summary.get('n_success')} asr={asr_str} "
                f"errors={summary.get('n_errors')}"
            )
    scored = sum(s.get("n_scored", 0) for _, s in summaries)
    success = sum(s.get("n_success", 0) for _, s in summaries)
    asr = success / scored if scored else None
    asr_str = "NA" if asr is None else f"{100 * asr:.2f}%"
    print(f"  TOTAL: n={scored} success={success} asr={asr_str}")
PY
