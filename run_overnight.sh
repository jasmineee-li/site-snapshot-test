#!/bin/bash
# Overnight eval awareness experiments
# Models: claude-opus-4.6, gpt-5.4-mini
# ~940 API calls total
set -e

echo "=== Starting overnight experiments $(date) ==="

echo ""
echo "=== [1/2] Comparative — gitlab+gmail+xero, 2pp/side, html, 2 models (~640 calls) ==="
python -m eval_awareness_experiments.run \
    --config eval_awareness_experiments/configs/overnight_comparative.yaml

echo ""
echo "=== [2/2] Realism critique — 25 samples, 3 formats, informed_eval+lightly_primed (~300 calls) ==="
python -m eval_awareness_experiments.experiments.realism_critique \
    --models "anthropic/claude-opus-4.6:thinking" "openai/gpt-5.4-mini:thinking" \
    --formats screenshot html axtree \
    --prompts informed_eval lightly_primed \
    --max-samples 25 --seed 42

echo ""
echo "=== All experiments complete $(date) ==="
