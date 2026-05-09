#!/usr/bin/env bash
# Replay saved WASP n200 trajectories and compute attacker-side e2e ASR.
#
# Defaults to all n200 WASP cells under eval_awareness_experiments/results.
# Restrict scope with:
#   ARMS=bare ./scripts/run_wasp_e2e_replay_n200.sh
#   ARMS="bare xml_safety" MODELS="openai_gpt-5.2_thinking" ./scripts/run_wasp_e2e_replay_n200.sh
#   MAX_TASKS=10 ./scripts/run_wasp_e2e_replay_n200.sh
#   RESET_STACK_BEFORE_CELL=1 ./scripts/run_wasp_e2e_replay_n200.sh
#
# Browser replay needs Playwright/Chromium access, so run outside Codex's
# sandbox, or via an approved escalated tmux command.

set -euo pipefail

REPO="${REPO:-/local_data/temp/max/browser-sim}"
RESULTS_DIR="${RESULTS_DIR:-$REPO/eval_awareness_experiments/results/n200_2026-04-29}"
WASP_DIR="${WASP_DIR:-$RESULTS_DIR/wasp}"
OUT_DIR="${OUT_DIR:-$RESULTS_DIR/wasp_e2e_replay}"
TASK_DIR_DEFAULT="${TASK_DIR_DEFAULT:-/tmp/wasp_full}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"

ARMS="${ARMS:-bare xml_safety xml_scenario xml_control}"
MODELS="${MODELS:-anthropic_claude-opus-4.7_thinking anthropic_claude-sonnet-4.6_thinking google_gemini-2.5-pro_thinking moonshotai_kimi-k2.5_thinking openai_gpt-5.2_thinking z-ai_glm-5_thinking}"
MAX_TASKS="${MAX_TASKS:-}"
DRY_RUN="${DRY_RUN:-0}"
SINGLE_SITE="${SINGLE_SITE:-}"
RESET_STACK_BEFORE_CELL="${RESET_STACK_BEFORE_CELL:-0}"

mkdir -p "$OUT_DIR" "$OUT_DIR/logs"

contains_word() {
    local needle=$1 haystack=$2
    for word in $haystack; do
        if [ "$word" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

task_dir_for_cell() {
    local cell_dir=$1
    local manifest="$cell_dir/pipeline_manifest_wasp.json"
    if [ -f "$manifest" ]; then
        local task_dir
        task_dir="$(jq -r '.config.wasp_task_dir // empty' "$manifest")"
        if [ -n "$task_dir" ]; then
            echo "$task_dir"
            return
        fi
    fi
    echo "$TASK_DIR_DEFAULT"
}

stack_info_for_task_dir() {
    local task_dir=$1
    case "$task_dir" in
        /tmp/wasp_full|/tmp/wasp_full/)
            echo "default 9001 8080"
            ;;
        /tmp/wasp_full_glm|/tmp/wasp_full_glm/)
            echo "glm 9201 8201"
            ;;
        /tmp/wasp_full_sonnet|/tmp/wasp_full_sonnet/)
            echo "sonnet 9211 8211"
            ;;
        /tmp/wasp_full_opus|/tmp/wasp_full_opus/)
            echo "opus 9221 8221"
            ;;
        /tmp/wasp_full_gpt|/tmp/wasp_full_gpt/)
            echo "gpt 9231 8231"
            ;;
        /tmp/wasp_full_gemini25|/tmp/wasp_full_gemini25/)
            echo "gemini25 9241 8241"
            ;;
        /tmp/wasp_full_kimi25|/tmp/wasp_full_kimi25/)
            echo "kimi25 9251 8251"
            ;;
        *)
            echo "unknown task_dir for reset: $task_dir" >&2
            return 1
            ;;
    esac
}

reset_stack_for_task_dir() {
    local task_dir=$1
    local reset_log=$2
    local stack gitlab_port reddit_port stack_info

    stack_info="$(stack_info_for_task_dir "$task_dir")"
    read -r stack gitlab_port reddit_port <<<"$stack_info"

    echo "=== reset stack=$stack gitlab_port=$gitlab_port reddit_port=$reddit_port ==="
    echo "reset_log=$reset_log"
    {
        echo "=== reset/recreate/plant $(date -Is) ==="
        "$REPO/scripts/wasp_reset_stack.sh" "$stack" "$gitlab_port" "$reddit_port"
    } 2>&1 | tee "$reset_log"
}

extra_args=()
if [ -n "$MAX_TASKS" ]; then
    extra_args+=(--max-tasks "$MAX_TASKS")
fi
if [ "$DRY_RUN" = "1" ]; then
    extra_args+=(--dry-run)
fi
if [ -n "$SINGLE_SITE" ]; then
    extra_args+=(--single-site "$SINGLE_SITE")
fi

echo "results_dir=$RESULTS_DIR"
echo "out_dir=$OUT_DIR"
echo "arms=$ARMS"
echo "models=$MODELS"
echo "max_tasks=${MAX_TASKS:-all}"
echo "dry_run=$DRY_RUN"
echo "reset_stack_before_cell=$RESET_STACK_BEFORE_CELL"

for arm in $ARMS; do
    for model in $MODELS; do
        cell_dir="$WASP_DIR/$arm/$model"
        if [ ! -d "$cell_dir" ]; then
            echo "skip missing $arm/$model"
            continue
        fi
        if ! contains_word "$arm" "$ARMS" || ! contains_word "$model" "$MODELS"; then
            continue
        fi

        task_dir="$(task_dir_for_cell "$cell_dir")"
        output_jsonl="$OUT_DIR/${arm}__${model}.jsonl"
        log_file="$OUT_DIR/logs/${arm}__${model}.log"
        reset_log="$OUT_DIR/logs/${arm}__${model}.reset.log"
        if [ "$DRY_RUN" = "1" ]; then
            mkdir -p "$OUT_DIR/dry_run"
            output_jsonl="$OUT_DIR/dry_run/${arm}__${model}.jsonl"
            log_file="$OUT_DIR/logs/${arm}__${model}.dry_run.log"
        fi

        echo "=== $arm/$model ==="
        echo "cell_dir=$cell_dir"
        echo "task_dir=$task_dir"
        echo "output=$output_jsonl"

        if [ "$RESET_STACK_BEFORE_CELL" = "1" ] && [ "$DRY_RUN" != "1" ]; then
            reset_stack_for_task_dir "$task_dir" "$reset_log"
        fi

        "$PYTHON" -m eval_awareness_experiments.wasp_replay_e2e_evaluator \
            --cell-dir "$cell_dir" \
            --task-dir "$task_dir" \
            --output-jsonl "$output_jsonl" \
            "${extra_args[@]}" \
            2>&1 | tee "$log_file"
    done
done

echo "done"
