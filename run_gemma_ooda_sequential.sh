#!/bin/bash
set -e
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate world-sim
set -a && source /local_data/temp/max/browser-sim/.env && set +a
cd /local_data/temp/max/browser-sim

OUTPUT_DIR="eval_awareness_experiments/results/doomarena_gemma4_31b_ooda"

for site in reddit shopping shopping_admin gitlab; do
    echo "=== ${site} gemma OODA started $(date) ==="
    python -m eval_awareness_experiments.run_doomarena_glm5_popup \
        --single-site ${site} \
        --max-tasks 2 \
        --online-sites reddit shopping shopping_admin gitlab \
        --model-name "google/gemma-4-31b-it" \
        --max-steps 15 \
        --report-port 1234 \
        --shuffle 42 \
        --extra-instructions-preset scratchpad_green
    echo "=== ${site} DoomArena done $(date) ==="

    STUDY_DIR=$(ls -td results/browsergym/study_* | head -1)
    echo "Study dir: $STUDY_DIR"

    python -m eval_awareness_experiments.run_doomarena_pipeline \
        --skip-doomarena \
        --study-dirs "${site}:$STUDY_DIR" \
        --sites ${site} \
        --judge-model anthropic/claude-opus-4.6:thinking \
        --judges test_awareness_steering verbalized_awareness \
        --output-dir ${OUTPUT_DIR}
    echo "=== ${site} ALL DONE $(date) ==="
    echo
done

echo "=== All sites done $(date) ==="
