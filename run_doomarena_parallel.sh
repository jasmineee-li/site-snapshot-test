#!/bin/bash
# Run DoomArena 25 tasks/site × 4 sites in parallel tmux sessions,
# then run trajectory judges on each site's results.
#
# Usage: bash run_doomarena_parallel.sh
set -e

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate world-sim
set -a
source /local_data/temp/max/browser-sim/.env
set +a
cd /local_data/temp/max/browser-sim

SITES="reddit shopping shopping_admin gitlab"
PORTS="1234 1235 1236 1237"
TASKS=25
MODEL="z-ai/glm-5:thinking"
JUDGE_MODEL="anthropic/claude-opus-4.6:thinking"
MAX_STEPS=15

echo "=== Launching 4 DoomArena sites in parallel $(date) ==="

# Launch each site in its own tmux session
i=0
for site in $SITES; do
    port=$(echo $PORTS | cut -d' ' -f$((i+1)))
    session="doom_${site}"

    tmux new-session -d -s "$session" "bash -c '
        source /opt/miniconda3/etc/profile.d/conda.sh && conda activate world-sim
        set -a && source /local_data/temp/max/browser-sim/.env && set +a
        cd /local_data/temp/max/browser-sim
        echo \"=== ${site} started \$(date) ===\"
        python -m eval_awareness_experiments.run_doomarena_glm5_popup \
            --single-site ${site} \
            --max-tasks ${TASKS} \
            --online-sites reddit shopping shopping_admin gitlab \
            --model-name ${MODEL} \
            --max-steps ${MAX_STEPS} \
            --report-port ${port} \
            --shuffle 42
        echo \"=== ${site} DoomArena done \$(date) ===\"

        # Find the study dir just created
        STUDY_DIR=\$(ls -td results/browsergym/study_* | head -1)
        echo \"Study dir: \$STUDY_DIR\"

        # Run trajectory judges
        echo \"=== ${site} running judges \$(date) ===\"
        python -m eval_awareness_experiments.run_doomarena_pipeline \
            --skip-doomarena \
            --study-dirs \"${site}:\$STUDY_DIR\" \
            --sites ${site} \
            --judge-model ${JUDGE_MODEL} \
            --judges test_awareness_steering verbalized_awareness \
            --output-dir eval_awareness_experiments/results/doomarena_overnight_25
        echo \"=== ${site} ALL DONE \$(date) ===\"
    ' 2>&1 | tee doomarena_parallel_${site}_\$(date +%Y%m%d_%H%M).log"

    echo "  Launched $session (port $port)"
    i=$((i+1))
    sleep 2  # stagger launches slightly
done

echo ""
echo "All 4 sessions launched. Monitor with:"
echo "  tmux ls"
echo "  tmux capture-pane -t doom_reddit -p | tail -20"
echo "  tail -f doomarena_parallel_*.log"
