#!/usr/bin/env bash
# Plant the full WASP prompt-injection task pool into one GitLab/forum stack.
#
# Usage:
#   STACK=glm GITLAB_PORT=9201 REDDIT_PORT=8201 ./scripts/wasp_plant_full_stack.sh
#   ./scripts/wasp_plant_full_stack.sh glm 9201 8201
#
# Output defaults:
#   /tmp/wasp_full_plant_<stack>/
#   /tmp/wasp_full_<stack>/

set -euo pipefail

STACK="${STACK:-${1:-}}"
GITLAB_PORT="${GITLAB_PORT:-${2:-}}"
REDDIT_PORT="${REDDIT_PORT:-${3:-}}"

if [ -z "$STACK" ] || [ -z "$GITLAB_PORT" ] || [ -z "$REDDIT_PORT" ]; then
    echo "usage: STACK=<slug> GITLAB_PORT=<port> REDDIT_PORT=<port> $0"
    echo "   or: $0 <slug> <gitlab_port> <reddit_port>"
    exit 1
fi

REPO="${REPO:-/local_data/temp/max/browser-sim}"
WASP="${WASP:-$REPO/external_benchmarks/wasp/webarena_prompt_injections}"
PLANT_ROOT="${PLANT_ROOT:-/tmp/wasp_full_plant_${STACK}}"
FINAL="${FINAL:-/tmp/wasp_full_${STACK}}"
RAW_CONFIG="${RAW_CONFIG:-$WASP/configs/experiment_config.raw.json}"
GITLAB_URL="http://localhost:${GITLAB_PORT}"
REDDIT_URL="http://localhost:${REDDIT_PORT}"

rm -rf "$PLANT_ROOT" "$FINAL"
mkdir -p "$PLANT_ROOT" "$FINAL/webarena_tasks"

cd "$WASP"

FORMATS=(
    generic_plain_text
    generic_url_injection
    goal_hijacking_plain_text
    goal_hijacking_url_injection
)
USER_GOALS=(0 1)

echo "=== planting WASP stack=$STACK gitlab=$GITLAB_URL reddit=$REDDIT_URL ==="
echo "    plant_root=$PLANT_ROOT"
echo "    final=$FINAL"

i=0
for ug in "${USER_GOALS[@]}"; do
    for fmt in "${FORMATS[@]}"; do
        out="$PLANT_ROOT/u${ug}_${fmt}"
        echo "=== plant $i: user_goal=$ug format=$fmt -> $out ==="
        GITLAB="$GITLAB_URL" REDDIT="$REDDIT_URL" DATASET=webarena_prompt_injections \
        PYTHONPATH=. "$REPO/.venv/bin/python" prompt_injector.py \
            --config "$RAW_CONFIG" \
            --gitlab-domain "$GITLAB_URL" --reddit-domain "$REDDIT_URL" \
            --model gpt-4o --system_prompt configs/system_prompts/wa_p_som_cot_id_actree_3s.json \
            --output-dir "$out" --user_goal_idx "$ug" \
            --injection_format "$fmt" --output-format webarena 2>&1 | tail -4
        i=$((i + 1))
    done
done

echo "=== merging into $FINAL with task_id renumbering ==="
PLANT_ROOT="$PLANT_ROOT" FINAL="$FINAL" "$REPO/.venv/bin/python" - <<'PY'
import json
import os
from pathlib import Path

plant_root = Path(os.environ["PLANT_ROOT"])
final = Path(os.environ["FINAL"])
combined_inst = []
new_id = 1000

for sub in sorted(plant_root.iterdir()):
    inst_path = sub / "instantiated_prompt_injections_config.json"
    if inst_path.exists():
        combined_inst.extend(json.loads(inst_path.read_text()))
    tasks_dir = sub / "webarena_tasks"
    if not tasks_dir.is_dir():
        continue
    for f in sorted(tasks_dir.glob("*.json"), key=lambda p: int(p.stem)):
        data = json.loads(f.read_text())
        data["task_id"] = new_id
        (final / "webarena_tasks" / f"{new_id}.json").write_text(json.dumps(data, indent=2))
        new_id += 1

(final / "instantiated_prompt_injections_config.json").write_text(
    json.dumps(combined_inst, indent=2)
)
print(f"merged: {new_id - 1000} tasks -> {final}/webarena_tasks/")
print(f"merged: {len(combined_inst)} attacker entries -> {final}/instantiated_prompt_injections_config.json")
PY

echo "=== plant + merge done for stack=$STACK ==="
ls "$FINAL/webarena_tasks/" | wc -l
echo "tasks at: $FINAL/webarena_tasks/"
