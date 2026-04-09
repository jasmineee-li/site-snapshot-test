from __future__ import annotations

from agentlab.benchmarks.redteam.analysis import analyze_results


def _result(
    *,
    behavior_id: str,
    agent_name: str,
    task_seed: int,
    variation_seed: int,
    test_mode: str,
    success: bool,
) -> dict:
    return {
        "behavior_id": behavior_id,
        "agent_name": agent_name,
        "task_seed": task_seed,
        "variation_seed": variation_seed,
        "test_mode": test_mode,
        "judge_result_full": {"overall_success": success, "individual_results": []},
    }


def test_analyze_results_pairs_by_agent_and_task_seed() -> None:
    results = [
        _result(
            behavior_id="demo",
            agent_name="agent-a",
            task_seed=1,
            variation_seed=0,
            test_mode="benign",
            success=True,
        ),
        _result(
            behavior_id="demo",
            agent_name="agent-a",
            task_seed=1,
            variation_seed=0,
            test_mode="adversarial",
            success=False,
        ),
        _result(
            behavior_id="demo",
            agent_name="agent-b",
            task_seed=2,
            variation_seed=0,
            test_mode="benign",
            success=False,
        ),
        _result(
            behavior_id="demo",
            agent_name="agent-b",
            task_seed=2,
            variation_seed=0,
            test_mode="adversarial",
            success=True,
        ),
    ]

    metrics = analyze_results(results, verbose=False)

    assert metrics["n_paired"] == 2
    assert metrics["avg_utility_degradation"] == 0.0
    assert {
        (pair["agent_name"], pair["task_seed"], pair["variation_seed"])
        for pair in metrics["pairs"]
    } == {
        ("agent-a", 1, 0),
        ("agent-b", 2, 0),
    }


def test_analyze_results_supports_flattened_env_args_keys() -> None:
    results = [
        {
            "env_args.behavior_id": "demo",
            "agent_args.agent_name": "agent-a",
            "env_args.task_seed": 7,
            "env_args.variation_seed": 3,
            "env_args.test_mode": "benign",
            "judge_result_full": {"overall_success": True, "individual_results": []},
        },
        {
            "env_args.behavior_id": "demo",
            "agent_args.agent_name": "agent-a",
            "env_args.task_seed": 7,
            "env_args.variation_seed": 3,
            "env_args.test_mode": "adversarial",
            "judge_result_full": {"overall_success": False, "individual_results": []},
        },
    ]

    metrics = analyze_results(results, verbose=False)

    assert metrics["n_paired"] == 1
    assert metrics["pairs"][0]["agent_name"] == "agent-a"
    assert metrics["pairs"][0]["task_seed"] == 7
    assert metrics["pairs"][0]["variation_seed"] == 3
