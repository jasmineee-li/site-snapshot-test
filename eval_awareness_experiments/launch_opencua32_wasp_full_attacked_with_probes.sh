#!/usr/bin/env bash
# Full WASP run on OpenCUA-32B with optional residual-stream steering.
#
# Modeled on launch_opencua72_wasp_full_attacked_with_probes.sh. The probe
# family is chosen by setting PROBE_DIR + SAD_EVAL — point them at either
# the MCQ or framing subdir under probes/trained/opencua-32b-chat-template-span/
# to compare the two. Both probes use SAD stages_oversight all-set selection
# at scoring time (`_resolve_sad_layer_threshold` reads `top_layers_all` from
# the SAD eval JSON), so any difference between probe families is in the
# trained vector itself, not the layer/threshold protocol.
#
# To switch from MCQ to the framing probe:
#   PROBE_DIR=$REPO_ROOT/probes/trained/opencua-32b-chat-template-span/framing \
#   SAD_EVAL=$REPO_ROOT/probes/trained/opencua-32b-chat-template-span/sad_stages_oversight_eval_framing.json \
#   STEERING_LAYERS=auto bash launch_opencua32_wasp_full_attacked_with_probes.sh
#
# STEERING_LAYERS=auto reads the top-6 layers from $SAD_EVAL.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

RUN_ID="${RUN_ID:-opencua32_wasp_attacked_full_$(date -u +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/eval_awareness_experiments/results/$RUN_ID/bare/local_opencua-32b}"
BROWSER_ROOT="$OUTPUT_DIR/_browser_runs/all"
LOG_DIR="$OUTPUT_DIR/logs"
LOG_PATH="$LOG_DIR/run.log"
mkdir -p "$LOG_DIR" "$BROWSER_ROOT"

exec > >(tee -a "$LOG_PATH") 2>&1

MODEL_PATH="${MODEL_PATH:-xlangai/OpenCUA-32B}"
PROBE_DIR="${PROBE_DIR:-$REPO_ROOT/probes/trained/opencua-32b-chat-template-span/mcq}"
SAD_EVAL="${SAD_EVAL:-$REPO_ROOT/probes/trained/opencua-32b-chat-template-span/sad_stages_oversight_eval.json}"
SAD_SCORE_LAYER="${SAD_SCORE_LAYER:-}"
SAD_SCORE_THRESHOLD="${SAD_SCORE_THRESHOLD:-}"
TASK_DIR="${TASK_DIR:-/tmp/wasp_full_glm}"
WASP_GITLAB_URL="${WASP_GITLAB_URL:-http://localhost:9201}"
WASP_REDDIT_URL="${WASP_REDDIT_URL:-http://localhost:8201}"
N_TASKS="${N_TASKS:-168}"
MAX_STEPS="${MAX_STEPS:-15}"
AVG_STEP_TIMEOUT="${AVG_STEP_TIMEOUT:-120}"
N_JOBS="${N_JOBS:-1}"
SHUFFLE="${SHUFFLE:-42}"
SYSTEM_PROMPT_FRAME="${SYSTEM_PROMPT_FRAME:-none}"
OPENCUA_PORT="${OPENCUA_PORT:-8001}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
OPENCUA_MAX_MEMORY="${OPENCUA_MAX_MEMORY:-}"
RELAUNCH_INCOMPLETE="${RELAUNCH_INCOMPLETE:-0}"
WASP_ASR_CLASSIFIER_MODEL="${WASP_ASR_CLASSIFIER_MODEL:-anthropic/claude-sonnet-4.6}"
WASP_ASR_CONCURRENCY="${WASP_ASR_CONCURRENCY:-32}"
WASP_ASR_MAX_STEPS_PER_TASK="${WASP_ASR_MAX_STEPS_PER_TASK:-10}"
STEERING_ENABLED="${STEERING_ENABLED:-0}"
STEERING_PROBE_DIR="${STEERING_PROBE_DIR:-$PROBE_DIR}"
STEERING_LAYERS="${STEERING_LAYERS:-auto}"
STEERING_ALPHA_PER_LAYER="${STEERING_ALPHA_PER_LAYER:-0.0}"

export HF_HOME="${HF_HOME:-/local_data/group_dir/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/local_data/group_dir/huggingface/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/local_data/group_dir/huggingface/hub}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-/tmp/hf_modules}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
HF_PYTHON="${HF_PYTHON:-$REPO_ROOT/.venv/bin/python}"

# Resolve STEERING_LAYERS=auto by reading top-6 layers from SAD_EVAL JSON.
if [[ "$STEERING_LAYERS" = "auto" ]]; then
    if [[ ! -f "$SAD_EVAL" ]]; then
        echo "STEERING_LAYERS=auto requires SAD_EVAL to exist; got $SAD_EVAL" >&2
        exit 1
    fi
    STEERING_LAYERS="$(
        "$HF_PYTHON" - "$SAD_EVAL" <<'PY'
import json, sys
sad = json.loads(open(sys.argv[1]).read())
split = sad.get("dev_test_split") or {}
top = (split.get("dev_top_layers") or sad.get("top_layers_all") or [])[:6]
print(",".join(str(item["layer"]) for item in top))
PY
    )"
    if [[ -z "$STEERING_LAYERS" ]]; then
        echo "could not derive STEERING_LAYERS from $SAD_EVAL" >&2
        exit 1
    fi
fi

MESSAGES_JSONL="$OUTPUT_DIR/wasp_messages.jsonl"
PROBE_ROWS="$OUTPUT_DIR/probe_spans.jsonl"
PROBE_SUMMARY="$OUTPUT_DIR/probe_span_summary.json"
ASR_JSONL="$OUTPUT_DIR/wasp_asr_all.jsonl"
MERGED_CSV="$OUTPUT_DIR/probe_reward_asr_table.csv"
MERGED_MD="$OUTPUT_DIR/probe_reward_asr_table.md"
MERGED_SUMMARY="$OUTPUT_DIR/probe_reward_asr_summary.json"

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID:-}" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "[$(date -u --iso-8601=seconds)] stopping HF OpenAI server pid=$SERVER_PID"
        kill "$SERVER_PID" || true
        wait "$SERVER_PID" || true
    fi
}
trap cleanup EXIT

wait_for_server() {
    local url="http://127.0.0.1:${OPENCUA_PORT}/v1/models"
    for _ in $(seq 1 180); do
        if curl -fsS "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 2
    done
    echo "server did not become ready at $url" >&2
    return 1
}

echo "started_at=$(date -u --iso-8601=seconds)"
echo "run_id=$RUN_ID"
echo "output_dir=$OUTPUT_DIR"
echo "browser_root=$BROWSER_ROOT"
echo "log_path=$LOG_PATH"
echo "task_dir=$TASK_DIR"
echo "wasp_gitlab_url=$WASP_GITLAB_URL"
echo "wasp_reddit_url=$WASP_REDDIT_URL"
echo "n_tasks=$N_TASKS"
echo "max_steps=$MAX_STEPS"
echo "avg_step_timeout=$AVG_STEP_TIMEOUT"
echo "n_jobs=$N_JOBS"
echo "system_prompt_frame=$SYSTEM_PROMPT_FRAME"
echo "cuda_visible_devices=$CUDA_VISIBLE_DEVICES"
echo "opencua_max_memory=${OPENCUA_MAX_MEMORY:-<none>}"
echo "relaunch_incomplete=$RELAUNCH_INCOMPLETE"
echo "probe_dir=$PROBE_DIR"
echo "sad_eval=$SAD_EVAL"
echo "sad_score_layer=${SAD_SCORE_LAYER:-<auto from SAD top_layers_all>}"
echo "sad_score_threshold=${SAD_SCORE_THRESHOLD:-<auto from SAD top_layers_all>}"
echo "hf_python=$HF_PYTHON"
echo "asr_classifier=$WASP_ASR_CLASSIFIER_MODEL"
echo "steering_enabled=$STEERING_ENABLED"
if [[ "$STEERING_ENABLED" = "1" || "$STEERING_ENABLED" = "true" ]]; then
    echo "steering_probe_dir=$STEERING_PROBE_DIR"
    echo "steering_layers=$STEERING_LAYERS"
    echo "steering_alpha_per_layer=$STEERING_ALPHA_PER_LAYER"
fi

SERVER_MAX_MEMORY_ARGS=()
if [[ -n "${OPENCUA_MAX_MEMORY:-}" ]]; then
    SERVER_MAX_MEMORY_ARGS+=(--max-memory "$OPENCUA_MAX_MEMORY")
fi

SERVER_STEERING_ARGS=()
if [[ "$STEERING_ENABLED" = "1" || "$STEERING_ENABLED" = "true" ]]; then
    SERVER_STEERING_ARGS+=(
        --steering-probe-dir "$STEERING_PROBE_DIR"
        --steering-layers "$STEERING_LAYERS"
        --steering-alpha-per-layer "$STEERING_ALPHA_PER_LAYER"
    )
fi

if curl -fsS "http://127.0.0.1:${OPENCUA_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[$(date -u --iso-8601=seconds)] reusing existing OpenCUA server on port $OPENCUA_PORT"
else
    echo "[$(date -u --iso-8601=seconds)] starting HF OpenAI server on port $OPENCUA_PORT"
    env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
        "$HF_PYTHON" -m models.common.hf_openai_server \
            "$MODEL_PATH" \
            --served-model-name xlangai/OpenCUA-32B \
            --host 0.0.0.0 \
            --port "$OPENCUA_PORT" \
            --dtype bfloat16 \
            --device-map auto \
            --max-new-tokens-cap 1024 \
            --trust-remote-code \
            "${SERVER_MAX_MEMORY_ARGS[@]}" \
            "${SERVER_STEERING_ARGS[@]}" \
            > "$LOG_DIR/hf_openai_server.log" 2>&1 &
    SERVER_PID="$!"
    echo "server_pid=$SERVER_PID"
    wait_for_server
fi

RUN_WASP_EXTRA_ARGS=()
if [[ "$RELAUNCH_INCOMPLETE" = "1" || "$RELAUNCH_INCOMPLETE" = "true" ]]; then
    RUN_WASP_EXTRA_ARGS+=(--relaunch-incomplete)
fi

echo "[$(date -u --iso-8601=seconds)] running full WASP task pool"
env \
    DOOMARENA_WEBARENA_BASE_URL=http://localhost \
    WA_GITLAB="$WASP_GITLAB_URL" \
    WA_REDDIT="$WASP_REDDIT_URL" \
    LOCAL_OPENAI_BASE_URL_OPENCUA_32B="http://127.0.0.1:${OPENCUA_PORT}/v1" \
    HF_HOME="$HF_HOME" \
    HF_HUB_CACHE="$HF_HUB_CACHE" \
    TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE" \
    HF_MODULES_CACHE="$HF_MODULES_CACHE" \
    .venv/bin/python -m eval_awareness_experiments.run_wasp \
        --max-tasks "$N_TASKS" \
        --model-name local/opencua-32b \
        --extra-instructions-preset none \
        --system-prompt-frame "$SYSTEM_PROMPT_FRAME" \
        --max-steps "$MAX_STEPS" \
        --n-jobs "$N_JOBS" \
        --avg-step-timeout "$AVG_STEP_TIMEOUT" \
        --shuffle "$SHUFFLE" \
        --condition baseline \
        --task-dir "$TASK_DIR" \
        --online-sites gitlab reddit \
        --results-dir "$BROWSER_ROOT" \
        "${RUN_WASP_EXTRA_ARGS[@]}"

echo "[$(date -u --iso-8601=seconds)] WASP run complete"

cleanup
SERVER_PID=""
sleep 5

echo "[$(date -u --iso-8601=seconds)] exporting chat messages"
env BROWSER_ROOT="$BROWSER_ROOT" MESSAGES_JSONL="$MESSAGES_JSONL" \
    .venv/bin/python - <<'PY'
import gzip
import json
import os
import pickle
import re
from pathlib import Path

from probes.score_probe import _normalize_chat_messages, _safe_get

root = Path(os.environ["BROWSER_ROOT"])
out = Path(os.environ["MESSAGES_JSONL"])
rows = []
for task_dir in sorted({p.parent for p in root.rglob("summary_info.json")}):
    if not list(task_dir.glob("step_*.pkl.gz")):
        continue
    match = re.search(r"webarena\.(\d+)", task_dir.name)
    task_id = f"webarena.{match.group(1)}" if match else task_dir.name
    for step_path in sorted(task_dir.glob("step_*.pkl.gz")):
        with gzip.open(step_path, "rb") as f:
            step = pickle.load(f)
        agent_info = _safe_get(step, "agent_info", {})
        chat = _safe_get(agent_info, "chat_messages")
        messages = _normalize_chat_messages(_safe_get(chat, "messages", []))
        if not messages:
            continue
        step_index = int(step_path.name.split("_", 1)[1].split(".", 1)[0])
        rows.append({
            "id": f"{task_id}:step_{step_index}",
            "task_id": task_id,
            "step_index": step_index,
            "messages": messages,
        })
out.write_text(
    "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
    encoding="utf-8",
)
print(f"wrote {out} rows={len(rows)} tasks={len({row['task_id'] for row in rows})}")
PY

echo "[$(date -u --iso-8601=seconds)] scoring probe spans"
env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
    MODEL_PATH="$MODEL_PATH" \
    PROBE_DIR="$PROBE_DIR" \
    SAD_EVAL="$SAD_EVAL" \
    SAD_SCORE_LAYER="$SAD_SCORE_LAYER" \
    SAD_SCORE_THRESHOLD="$SAD_SCORE_THRESHOLD" \
    MESSAGES_JSONL="$MESSAGES_JSONL" \
    PROBE_ROWS="$PROBE_ROWS" \
    PROBE_SUMMARY="$PROBE_SUMMARY" \
    "$HF_PYTHON" - <<'PY'
import json
import os
from pathlib import Path
from statistics import mean, pstdev

import torch

from probes.contrastive_probe import ContrastiveProbe
from probes.model_loader import load_model
from probes.score_probe import (
    _normalize_chat_messages,
    _resolve_sad_layer_threshold,
    _score_messages,
)

model_path = os.environ["MODEL_PATH"]
probe_dir = os.environ["PROBE_DIR"]
sad_eval = os.environ["SAD_EVAL"]
messages_jsonl = Path(os.environ["MESSAGES_JSONL"])
out_path = Path(os.environ["PROBE_ROWS"])
summary_path = Path(os.environ["PROBE_SUMMARY"])
spans = ["last_user_token", "user_message_mean", "first_assistant_message_mean"]

requested_layer = os.environ.get("SAD_SCORE_LAYER") or None
requested_threshold = os.environ.get("SAD_SCORE_THRESHOLD") or None
requested_layer = int(requested_layer) if requested_layer else None
requested_threshold = float(requested_threshold) if requested_threshold else None

print("loading model for probe scoring")
model, tokenizer = load_model(model_path, device=None, dtype=torch.bfloat16)
probe = ContrastiveProbe.load(probe_dir)
layer, threshold, selection = _resolve_sad_layer_threshold(
    sad_eval,
    requested_layer,
    requested_threshold,
    probe,
)
selection["layer_source"] = "sad_stages_oversight_all"
selection["threshold_source"] = "sad_stages_oversight_all"
print(f"probe layer={layer} threshold={threshold} selection={selection}")

items = []
with messages_jsonl.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        item = json.loads(line)
        item["messages"] = _normalize_chat_messages(item.get("messages", []))
        items.append(item)
print(f"items={len(items)}")

rows = []
for span in spans:
    print(f"scoring {span}")
    for item in items:
        result = _score_messages(
            probe,
            model,
            tokenizer,
            item["messages"],
            layer,
            threshold,
            span=span,
        )
        rows.append({
            "id": item.get("id"),
            "task_id": item.get("task_id"),
            "step_index": item.get("step_index"),
            "span": span,
            "layer": layer,
            "threshold": threshold,
            **result,
        })

out_path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""), encoding="utf-8")
summary = {
    "n_items": len(items),
    "layer": layer,
    "threshold": threshold,
    **selection,
    "spans": {},
}
for span in spans:
    span_rows = [row for row in rows if row["span"] == span]
    scores = [row["mean_score"] for row in span_rows]
    summary["spans"][span] = {
        "n": len(span_rows),
        "mean_score": mean(scores) if scores else None,
        "std_score": pstdev(scores) if len(scores) > 1 else 0.0,
        "min_score": min(scores) if scores else None,
        "max_score": max(scores) if scores else None,
        "n_eval_aware": sum(1 for row in span_rows if row["classification"] == "eval_aware"),
        "eval_aware_rate": (
            sum(1 for row in span_rows if row["classification"] == "eval_aware") / len(span_rows)
            if span_rows else None
        ),
        "mean_n_positions": mean([row["n_positions"] for row in span_rows]) if span_rows else None,
    }
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out_path}")
print(f"wrote {summary_path}")
print(json.dumps(summary, indent=2))
PY

echo "[$(date -u --iso-8601=seconds)] running WASP post-hoc ASR classifier"
env \
    BROWSER_ROOT="$BROWSER_ROOT" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    WASP_ASR_CLASSIFIER_MODEL="$WASP_ASR_CLASSIFIER_MODEL" \
    WASP_ASR_CONCURRENCY="$WASP_ASR_CONCURRENCY" \
    WASP_ASR_MAX_STEPS_PER_TASK="$WASP_ASR_MAX_STEPS_PER_TASK" \
    .venv/bin/python - <<'PY'
import asyncio
import os
from pathlib import Path

from eval_awareness_experiments.llm import LLM
from eval_awareness_experiments.wasp_post_hoc_evaluator import evaluate_wasp_cell_split

async def main() -> None:
    output_dir = Path(os.environ["OUTPUT_DIR"])
    cell = {
        "arm": "bare",
        "model_slug": "local_opencua-32b",
        "split": "all",
        "study_dir": os.environ["BROWSER_ROOT"],
        "cell_dir": str(output_dir),
    }
    classifier_model = os.environ["WASP_ASR_CLASSIFIER_MODEL"]
    llm = LLM(
        model=classifier_model,
        thinking="thinking" in classifier_model,
        concurrency=int(os.environ["WASP_ASR_CONCURRENCY"]),
        temperature=0.0,
    )
    summary = await evaluate_wasp_cell_split(
        cell=cell,
        llm=llm,
        classifier_model=classifier_model,
        max_steps_per_task=int(os.environ["WASP_ASR_MAX_STEPS_PER_TASK"]),
        max_tasks_per_cell=None,
        force=True,
    )
    print(summary)

asyncio.run(main())
PY

echo "[$(date -u --iso-8601=seconds)] writing merged probe/reward/ASR table"
env \
    BROWSER_ROOT="$BROWSER_ROOT" \
    OUTPUT_DIR="$OUTPUT_DIR" \
    PROBE_ROWS="$PROBE_ROWS" \
    ASR_JSONL="$ASR_JSONL" \
    MERGED_CSV="$MERGED_CSV" \
    MERGED_MD="$MERGED_MD" \
    MERGED_SUMMARY="$MERGED_SUMMARY" \
    .venv/bin/python - <<'PY'
import csv
import gzip
import json
import os
import pickle
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

browser_root = Path(os.environ["BROWSER_ROOT"])
probe_rows_path = Path(os.environ["PROBE_ROWS"])
asr_path = Path(os.environ["ASR_JSONL"])
csv_path = Path(os.environ["MERGED_CSV"])
md_path = Path(os.environ["MERGED_MD"])
summary_path = Path(os.environ["MERGED_SUMMARY"])

rewards = {}
short = browser_root / "short_results.csv"
if short.exists():
    with short.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            task_id = row.get("task_id")
            if task_id:
                try:
                    rewards[task_id] = float(row.get("reward", ""))
                except ValueError:
                    rewards[task_id] = None

if not rewards:
    for summary_info in browser_root.rglob("summary_info.json"):
        task_dir = summary_info.parent
        match = re.search(r"webarena\.(\d+)", task_dir.name)
        if not match:
            continue
        task_id = f"webarena.{match.group(1)}"
        try:
            rewards[task_id] = float(json.loads(summary_info.read_text()).get("cum_reward", 0.0))
        except Exception:
            rewards[task_id] = None

asr = {}
asr_summary = {}
if asr_path.exists():
    with asr_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "task_id" in row and "any_compromised" in row:
                asr[str(row["task_id"])] = bool(row["any_compromised"])
            elif "asr" in row:
                asr_summary = row

probe_by_task = defaultdict(dict)
if probe_rows_path.exists():
    with probe_rows_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            task_id = str(row.get("task_id"))
            span = row.get("span")
            if task_id and span:
                probe_by_task[task_id][span] = row.get("mean_score")

task_ids = sorted(set(rewards) | set(asr) | set(probe_by_task))
fields = [
    "task_id",
    "reward",
    "attack_success",
    "last_user_token",
    "user_message_mean",
    "first_assistant_message_mean",
]
rows = []
for task_id in task_ids:
    spans = probe_by_task.get(task_id, {})
    rows.append({
        "task_id": task_id,
        "reward": rewards.get(task_id),
        "attack_success": asr.get(task_id),
        "last_user_token": spans.get("last_user_token"),
        "user_message_mean": spans.get("user_message_mean"),
        "first_assistant_message_mean": spans.get("first_assistant_message_mean"),
    })

with csv_path.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).lower() if isinstance(value, bool) else str(value)

md_lines = [
    "| task_id | reward | attack_success | last_user_token | user_message_mean | first_assistant_message_mean |",
    "|---|---:|---:|---:|---:|---:|",
]
for row in rows:
    md_lines.append(
        "| "
        + " | ".join(fmt(row[field]) for field in fields)
        + " |"
    )
md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

summary = {
    "n_rows": len(rows),
    "n_rewards": sum(1 for row in rows if row["reward"] is not None),
    "n_attack_success_labels": sum(1 for row in rows if row["attack_success"] is not None),
    "asr": asr_summary.get("asr") if asr_summary else (
        mean([float(row["attack_success"]) for row in rows if row["attack_success"] is not None])
        if any(row["attack_success"] is not None for row in rows) else None
    ),
    "tsr": (
        mean([float((row["reward"] or 0) > 0) for row in rows if row["reward"] is not None])
        if any(row["reward"] is not None for row in rows) else None
    ),
    "probe_by_attack_success": {},
}
for attack_value in (False, True):
    group = [row for row in rows if row["attack_success"] is attack_value]
    label = str(attack_value).lower()
    summary["probe_by_attack_success"][label] = {"n": len(group)}
    for span in ("last_user_token", "user_message_mean", "first_assistant_message_mean"):
        scores = [row[span] for row in group if row[span] is not None]
        summary["probe_by_attack_success"][label][span] = {
            "mean": mean(scores) if scores else None,
            "std": pstdev(scores) if len(scores) > 1 else 0.0,
        }
summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"wrote {csv_path}")
print(f"wrote {md_path}")
print(f"wrote {summary_path}")
print(json.dumps(summary, indent=2))
PY

echo "finished_at=$(date -u --iso-8601=seconds)"
