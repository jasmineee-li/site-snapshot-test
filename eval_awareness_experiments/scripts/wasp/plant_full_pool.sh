#!/usr/bin/env bash
# Plant full WASP cross-product (gitlab + reddit, all 21 attackers × 2 user_goals × 4 formats).
# Output: /tmp/wasp_full_plant/u{0,1}_{format}/  (8 subdirs, each with webarena_tasks/*.json + instantiated_prompt_injections_config.json)
# Then merges into /tmp/wasp_full/webarena_tasks/  with task_ids renumbered to be globally unique.
#
# Total: ~168 task JSONs (21 × 2 × 4). Wall time ~8-12 minutes.
set -euo pipefail

REPO="${REPO:-/local_data/temp/max/browser-sim}"
WASP=$REPO/external_benchmarks/wasp/webarena_prompt_injections
PLANT_ROOT=/tmp/wasp_full_plant
FINAL=/tmp/wasp_full
RAW_CONFIG=$WASP/configs/experiment_config.raw.json

rm -rf "$PLANT_ROOT" "$FINAL"
mkdir -p "$PLANT_ROOT" "$FINAL/webarena_tasks"

cd "$WASP"

FORMATS=(generic_plain_text generic_url_injection goal_hijacking_plain_text goal_hijacking_url_injection)
USER_GOALS=(0 1)

i=0
for ug in "${USER_GOALS[@]}"; do
  for fmt in "${FORMATS[@]}"; do
    out="$PLANT_ROOT/u${ug}_${fmt}"
    echo "=== plant $i: user_goal=$ug, format=$fmt → $out ==="
    GITLAB=http://localhost:9001 REDDIT=http://localhost:8080 DATASET=webarena_prompt_injections \
    PYTHONPATH=. "$REPO/.venv/bin/python" prompt_injector.py \
      --config "$RAW_CONFIG" \
      --gitlab-domain http://localhost:9001 --reddit-domain http://localhost:8080 \
      --model gpt-4o --system_prompt configs/system_prompts/wa_p_som_cot_id_actree_3s.json \
      --output-dir "$out" --user_goal_idx "$ug" \
      --injection_format "$fmt" --output-format webarena 2>&1 | tail -4
    i=$((i+1))
  done
done

echo "=== merging into $FINAL with task_id renumbering ==="
"$REPO/.venv/bin/python" - <<'PY'
import json, shutil
from pathlib import Path
PLANT_ROOT = Path("/tmp/wasp_full_plant")
FINAL = Path("/tmp/wasp_full")
combined_inst = []
new_id = 1000
for sub in sorted(PLANT_ROOT.iterdir()):
    inst_path = sub / "instantiated_prompt_injections_config.json"
    if inst_path.exists():
        combined_inst.extend(json.loads(inst_path.read_text()))
    tasks_dir = sub / "webarena_tasks"
    if not tasks_dir.is_dir(): continue
    for f in sorted(tasks_dir.glob("*.json"), key=lambda p: int(p.stem)):
        data = json.loads(f.read_text())
        data["task_id"] = new_id
        (FINAL / "webarena_tasks" / f"{new_id}.json").write_text(json.dumps(data, indent=2))
        new_id += 1
(FINAL / "instantiated_prompt_injections_config.json").write_text(json.dumps(combined_inst, indent=2))
print(f"merged: {new_id - 1000} tasks → {FINAL}/webarena_tasks/")
print(f"merged: {len(combined_inst)} attacker entries → {FINAL}/instantiated_prompt_injections_config.json")
PY

echo "=== plant + merge done ==="
ls "$FINAL/webarena_tasks/" | wc -l
echo "tasks at: $FINAL/webarena_tasks/"
