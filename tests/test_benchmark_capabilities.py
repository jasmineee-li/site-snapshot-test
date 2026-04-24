import pytest

from worldsim.benchmark_capabilities import (
    get_benchmark_capabilities,
    infer_benchmark_name,
    normalize_benchmark_name,
)
from worldsim.phases import phase_2_injections


def test_normalize_benchmark_aliases():
    assert normalize_benchmark_name("WebArena Verified") == "webarena_verified"
    assert normalize_benchmark_name("st-webagentbench") == "stwebagentbench"
    assert normalize_benchmark_name("Doom Arena") == "doomarena"


def test_known_comparison_benchmarks_do_not_support_phase_2c():
    for name in ("wasp", "stwebagentbench", "doomarena"):
        capabilities = get_benchmark_capabilities(name)
        assert capabilities.phase_2_supported is False
        assert capabilities.phase_2_feasibility_supported is False


def test_infer_benchmark_rejects_unknown():
    with pytest.raises(ValueError, match="unknown benchmark"):
        infer_benchmark_name(["new-benchmark"])


def test_infer_benchmark_rejects_mixed_metadata():
    with pytest.raises(ValueError, match="mixed benchmark metadata"):
        infer_benchmark_name(["webarena_verified", "wasp"])


def test_phase_2c_gate_accepts_webarena():
    benchmark = phase_2_injections._gate_phase_2c_benchmark(
        task_records=[{"id": "task-1", "benchmark": "WebArena Verified"}],
        raw_instances={
            "benchmark_name": "WebArena Verified",
            "instances": [{"site_name": "gitlab", "site_url": "http://gitlab.test"}],
        },
        instances=[{"site_name": "gitlab", "site_url": "http://gitlab.test"}],
    )

    assert benchmark == "webarena_verified"


def test_phase_2c_gate_accepts_instances_top_level_benchmark():
    benchmark = phase_2_injections._gate_phase_2c_benchmark(
        task_records=[{"id": "task-1", "benchmark": "WebArena Verified"}],
        raw_instances={
            "benchmark": "WebArena Verified",
            "instances": [{"site_name": "gitlab", "site_url": "http://gitlab.test"}],
        },
        instances=[{"site_name": "gitlab", "site_url": "http://gitlab.test"}],
    )

    assert benchmark == "webarena_verified"


def test_phase_2c_gate_rejects_unsupported_benchmark():
    with pytest.raises(ValueError, match="does not support WorldSim v5 Phase 2c"):
        phase_2_injections._gate_phase_2c_benchmark(
            task_records=[{"id": "task-1", "benchmark": "wasp"}],
            raw_instances={"benchmark_name": "wasp", "instances": [{"site_name": "gitlab"}]},
            instances=[{"site_name": "gitlab", "benchmark": "wasp"}],
        )


def test_phase_2c_gate_rejects_missing_metadata():
    with pytest.raises(ValueError, match="missing benchmark metadata"):
        phase_2_injections._gate_phase_2c_benchmark(
            task_records=[{"id": "task-1"}],
            raw_instances={"instances": [{"site_name": "gitlab"}]},
            instances=[{"site_name": "gitlab"}],
        )


def test_phase_2c_gate_rejects_task_instance_mismatch():
    with pytest.raises(ValueError, match="mixed benchmark metadata"):
        phase_2_injections._gate_phase_2c_benchmark(
            task_records=[{"id": "task-1", "benchmark": "webarena_verified"}],
            raw_instances={"benchmark_name": "wasp", "instances": [{"site_name": "gitlab"}]},
            instances=[{"site_name": "gitlab", "benchmark": "wasp"}],
        )
