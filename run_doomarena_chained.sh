#!/bin/bash
# Wait for current glm-5 runs to finish, then launch glm-5.1
# Run this in its own tmux session:
#   tmux new-session -d -s doom_chain 'bash run_doomarena_chained.sh 2>&1 | tee doom_chain.log'

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate world-sim
set -a && source /local_data/temp/max/browser-sim/.env && set +a
cd /local_data/temp/max/browser-sim

SITES="reddit shopping shopping_admin gitlab"
PORTS="1234 1235 1236 1237"
TASKS=25
MAX_STEPS=15
JUDGE_MODEL="anthropic/claude-opus-4.6:thinking"

echo "=== Waiting for glm-5 runs to finish $(date) ==="

# Poll until all doom_* sessions are gone
while true; do
    alive=$(tmux ls 2>/dev/null | grep -c "^doom_\(reddit\|shopping\|gitlab\)" || true)
    if [ "$alive" -eq 0 ]; then
        break
    fi
    echo "  $alive doom sessions still running... $(date)"
    sleep 60
done

echo "=== glm-5 runs complete $(date) ==="
echo ""
echo "=== Launching glm-5.1 runs $(date) ==="

MODEL="z-ai/glm-5.1:thinking"
OUTPUT_DIR="eval_awareness_experiments/results/doomarena_overnight_25_glm51"

i=0
for site in $SITES; do
    port=$(echo $PORTS | cut -d' ' -f$((i+1)))
    session="doom_${site}"

    tmux new-session -d -s "$session" "bash -c '
        source /opt/miniconda3/etc/profile.d/conda.sh && conda activate world-sim
        set -a && source /local_data/temp/max/browser-sim/.env && set +a
        cd /local_data/temp/max/browser-sim
        echo \"=== ${site} glm-5.1 started \$(date) ===\"
        python -m eval_awareness_experiments.run_doomarena_glm5_popup \
            --single-site ${site} \
            --max-tasks ${TASKS} \
            --online-sites reddit shopping shopping_admin gitlab \
            --model-name ${MODEL} \
            --max-steps ${MAX_STEPS} \
            --report-port ${port} \
            --shuffle 42
        echo \"=== ${site} DoomArena done \$(date) ===\"
        STUDY_DIR=\$(ls -td results/browsergym/study_* | head -1)
        echo \"Study dir: \$STUDY_DIR\"
        echo \"=== ${site} running judges \$(date) ===\"
        python -m eval_awareness_experiments.run_doomarena_pipeline \
            --skip-doomarena \
            --study-dirs \"${site}:\$STUDY_DIR\" \
            --sites ${site} \
            --judge-model ${JUDGE_MODEL} \
            --judges test_awareness_steering verbalized_awareness \
            --output-dir ${OUTPUT_DIR}
        echo \"=== ${site} ALL DONE \$(date) ===\"
    ' 2>&1 | tee doomarena_parallel_glm51_${site}_\$(date +%Y%m%d_%H%M).log"

    echo "  Launched $session (glm-5.1, port $port)"
    i=$((i+1))
    sleep 2
done

echo ""
echo "=== All glm-5.1 sessions launched $(date) ==="
