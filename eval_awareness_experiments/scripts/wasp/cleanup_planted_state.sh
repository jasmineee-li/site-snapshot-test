#!/usr/bin/env bash
# Cleanup all WASP planted state from /tmp/wasp_full plant runs.
# Iterates each subdir's instantiated_prompt_injections_config.json + the merged one.
set -uo pipefail

REPO="${REPO:-/local_data/temp/max/browser-sim}"
WASP=$REPO/external_benchmarks/wasp/webarena_prompt_injections
PLANT_ROOT=/tmp/wasp_full_plant
FINAL=/tmp/wasp_full

cd "$WASP"

# Cleanup the merged config (covers everything in one pass, since each entry's
# cleanup_fn idempotently undoes its planted state).
INST=$FINAL/instantiated_prompt_injections_config.json
if [ ! -f "$INST" ]; then
  echo "no merged config at $INST — falling back to per-subdir"
else
  echo "=== cleanup using merged config ($(wc -l < $INST) lines) ==="
  GITLAB=http://localhost:9001 REDDIT=http://localhost:8080 \
  PYTHONPATH=. "$REPO/.venv/bin/python" environment_cleanup.py \
    --prompt-injection-config-path "$INST" \
    --gitlab-domain http://localhost:9001 --reddit-domain http://localhost:8080 \
    --delete-attacker-account 2>&1 | tail -20
fi

# Belt-and-braces: also iterate per-subdir configs, since the merged file may
# omit entries that failed to plant (and per-subdir cleanup retries those).
for sub in "$PLANT_ROOT"/*/; do
  inst="$sub/instantiated_prompt_injections_config.json"
  [ -f "$inst" ] || continue
  echo "=== cleanup subdir: $(basename "$sub") ==="
  GITLAB=http://localhost:9001 REDDIT=http://localhost:8080 \
  PYTHONPATH=. "$REPO/.venv/bin/python" environment_cleanup.py \
    --prompt-injection-config-path "$inst" \
    --gitlab-domain http://localhost:9001 --reddit-domain http://localhost:8080 2>&1 | tail -5
done

echo "=== all cleanup done ==="
