#!/bin/bash
# Overnight DoomArena pipeline: 20 tasks/site × 4 sites + trajectory awareness judges
#
# Run in tmux:
#   tmux new-session -d -s doomarena_overnight 'bash run_doomarena_overnight.sh 2>&1 | tee doomarena_overnight_$(date +%Y%m%d_%H%M).log'
set -e

echo "=== DoomArena overnight pipeline started $(date) ==="

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate world-sim
set -a
source /local_data/temp/max/browser-sim/.env
set +a
cd /local_data/temp/max/browser-sim

# Verify containers are up
echo "Checking WebArena containers..."
for port_site in "8080:reddit" "8082:shopping" "8083:shopping_admin" "9001:gitlab"; do
    port=${port_site%%:*}
    site=${port_site##*:}
    status=$(curl -sI -o /dev/null -w "%{http_code}" http://localhost:$port/ 2>/dev/null || echo "FAIL")
    echo "  $site (localhost:$port): $status"
    if [ "$status" != "200" ] && [ "$status" != "302" ]; then
        echo "  WARNING: $site not reachable!"
    fi
done

echo ""
echo "Running pipeline: 4 sites × 20 tasks × glm-5:thinking + trajectory judges"
echo ""

python -m eval_awareness_experiments.run_doomarena_pipeline \
    --sites reddit shopping shopping_admin gitlab \
    --tasks-per-site 20 \
    --agent-model "z-ai/glm-5:thinking" \
    --judge-model "anthropic/claude-opus-4.6:thinking" \
    --judges test_awareness_steering verbalized_awareness \
    --max-steps 15

echo ""
echo "=== DoomArena overnight pipeline complete $(date) ==="
