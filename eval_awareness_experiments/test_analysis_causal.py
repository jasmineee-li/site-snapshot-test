from __future__ import annotations

from eval_awareness_experiments.analysis_causal import (
    compute_awareness_rates,
    compute_causal_stats,
)


def test_causal_analysis_excludes_wasp_intermediate_asr_metadata() -> None:
    results = [
        {
            "model": "model-a",
            "metadata": {
                "benchmark": "wasp",
                "injection_condition": "bare",
                "attack_success": True,
                "attack_success_source": "wasp_post_hoc_action_classifier",
            },
        },
        {
            "model": "model-a",
            "metadata": {
                "benchmark": "wasp",
                "injection_condition": "bare",
                "attack_success": False,
            },
        },
    ]

    awareness = compute_awareness_rates(results)
    causal = compute_causal_stats(results)

    assert awareness["wasp"]["asr_total"] == 1
    assert awareness["wasp"]["asr_count"] == 0
    assert awareness["wasp"]["asr"] == 0.0
    assert causal["wasp"]["bare"]["model-a"]["asr"] == 0.0
