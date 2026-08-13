import pytest

from worldsim.benchmark_capabilities import (
    DEFAULT_BENCHMARK_CATALOG,
    BenchmarkCapabilities,
    BenchmarkCatalog,
    get_benchmark_capabilities,
    infer_benchmark_name,
    normalize_benchmark_name,
    resolve_evaluator_authority,
)
from worldsim.benchmark_contracts import (
    BenchmarkCapabilities as ContractBenchmarkCapabilities,
)
from worldsim.benchmark_contracts import (
    BenchmarkCatalog as ContractBenchmarkCatalog,
)
from worldsim.phase_2.phase_2c.config import _gate_phase_2c_benchmark


def test_normalize_benchmark_aliases():
    assert normalize_benchmark_name("WebArena Verified") == "webarena_verified"
    assert normalize_benchmark_name(" webarena   verified ") == "webarena_verified"
    assert normalize_benchmark_name("st-webagentbench") == "stwebagentbench"
    assert normalize_benchmark_name("Doom Arena") == "doomarena"


def test_capability_facade_reexports_contract_types_by_identity():
    assert BenchmarkCapabilities is ContractBenchmarkCapabilities
    assert BenchmarkCatalog is ContractBenchmarkCatalog


def test_benchmark_catalog_is_immutable_and_explicit():
    with pytest.raises(TypeError):
        DEFAULT_BENCHMARK_CATALOG.entries["unknown"] = get_benchmark_capabilities("wasp")

    webarena = DEFAULT_BENCHMARK_CATALOG.resolve("WebArena Verified")
    assert webarena.warp_phase_admission == (
        "phase_1_generation",
        "phase_2_generation",
        "phase_2_feasibility",
        "phase_4_execution",
    )
    assert webarena.supports("phase_2_feasibility") is True
    assert webarena.supports_runner("browser-use") is True
    assert webarena.comparison_only_ingestion_supported is False


def test_legacy_capability_constructor_translates_to_exact_capabilities():
    legacy = BenchmarkCapabilities(
        "legacy_webarena",
        "browser_use",
        ("browser_use",),
        True,
        True,
        True,
        "worldsim_v5",
        None,
        True,
    )

    assert legacy.warp_phase_admission == (
        "phase_1_generation",
        "phase_2_generation",
        "phase_2_feasibility",
        "phase_4_execution",
    )
    assert legacy.evaluator_authorities == (
        "canonical_vendor_task_id",
        "warp_local_task_idless",
    )

    legacy_comparison = BenchmarkCapabilities(
        canonical_name="legacy_comparison",
        default_runner="agentlab",
        supported_runners=("agentlab",),
        phase_1_supported=True,
        phase_2_supported=False,
        phase_2_feasibility_supported=False,
        phase_4_mode="comparison_runner",
        comparison_outcome_mode="capability",
    )
    assert legacy_comparison.capabilities == frozenset({"comparison_ingestion"})
    assert legacy_comparison.phase_1_supported is False


def _custom_capability(name: str) -> BenchmarkCapabilities:
    return BenchmarkCapabilities(
        canonical_name=name,
        default_runner="browser_use",
        supported_runners=("browser_use",),
        capabilities=frozenset({"warp_evaluation"}),
        evaluator_authorities=("canonical_vendor_task_id", "warp_local_task_idless"),
    )


def _contract(**overrides: object) -> BenchmarkCapabilities:
    values: dict[str, object] = {
        "canonical_name": "demo",
        "default_runner": "browser_use",
        "supported_runners": ("browser_use",),
        "capabilities": frozenset({"warp_evaluation"}),
        "phase_4_mode": "unsupported",
        "evaluator_authorities": (
            "canonical_vendor_task_id",
            "warp_local_task_idless",
        ),
    }
    values.update(overrides)
    return BenchmarkCapabilities(**values)  # type: ignore[arg-type]


def test_benchmark_catalog_rejects_normalized_duplicates_and_conflicts():
    with pytest.raises(ValueError, match="duplicate benchmark catalog entry"):
        BenchmarkCatalog(
            {
                "demo-benchmark": _custom_capability("demo_benchmark"),
                "demo_benchmark": _custom_capability("demo_benchmark"),
            }
        )

    with pytest.raises(ValueError, match="conflicting benchmark alias"):
        BenchmarkCatalog(
            {"demo": _custom_capability("demo"), "other": _custom_capability("other")},
            {"demo-alias": "demo", "demo alias": "other"},
        )
    with pytest.raises(ValueError, match="conflicts with a canonical entry"):
        BenchmarkCatalog(
            {"demo": _custom_capability("demo"), "other": _custom_capability("other")},
            {"demo": "other"},
        )


def test_benchmark_catalog_rejects_malformed_mappings_and_runner_types():
    with pytest.raises(ValueError, match="entries must be a mapping"):
        BenchmarkCatalog([])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="aliases must be a mapping"):
        BenchmarkCatalog({}, [])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="supported runners"):
        BenchmarkCapabilities(
            canonical_name="demo",
            default_runner="browser_use",
            supported_runners=(object(),),  # type: ignore[arg-type]
            capabilities=frozenset({"warp_evaluation"}),
            evaluator_authorities=("warp_local_task_idless",),
        )
    with pytest.raises(ValueError, match="string default runner"):
        BenchmarkCapabilities(
            canonical_name="demo",
            default_runner=object(),  # type: ignore[arg-type]
            supported_runners=("browser_use",),
            capabilities=frozenset({"warp_evaluation"}),
            evaluator_authorities=("warp_local_task_idless",),
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("supported_runners", "iterable supported runners"),
        ("capabilities", "capabilities must be iterable"),
        ("evaluator_authorities", "evaluator authorities must be iterable"),
    ],
)
def test_benchmark_capability_rejects_non_iterable_fields(field: str, message: str):
    with pytest.raises(ValueError, match=message):
        _contract(**{field: None})


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"capabilities": frozenset({"phase_2_feasibility"})},
            "Phase 2 feasibility requires",
        ),
        (
            {"capabilities": frozenset({"phase_4_execution_typo"})},
            "unknown benchmark capabilities",
        ),
        (
            {"capabilities": frozenset({"phase_4_execution"})},
            "phase_4_execution capability must match",
        ),
        (
            {"phase_4_mode": "worldsim_v5"},
            "phase_4_execution capability must match",
        ),
        (
            {
                "capabilities": frozenset({"warp_evaluation", "comparison_ingestion"}),
            },
            "cannot coexist",
        ),
        (
            {
                "capabilities": frozenset({"comparison_ingestion"}),
                "evaluator_authorities": ("warp_local_task_idless",),
            },
            "only comparison_runner",
        ),
        (
            {"evaluator_authorities": ("comparison_runner",)},
            "cannot use comparison_runner",
        ),
        (
            {
                "capabilities": frozenset({"phase_4_execution"}),
                "phase_4_mode": "worldsim_v5",
            },
            "Phase 4 execution requires WARP evaluation",
        ),
        (
            {"comparison_outcome_mode": "resistance"},
            "comparison_outcome_mode requires",
        ),
        (
            {"comparison_outcome_mode": "not-an-outcome"},
            "unknown comparison outcome mode",
        ),
        (
            {"requires_host_api_preflight": "yes"},
            "host API preflight declaration must be boolean",
        ),
    ],
)
def test_benchmark_capability_rejects_inconsistent_contracts(
    overrides: dict[str, object],
    message: str,
):
    with pytest.raises(ValueError, match=message):
        _contract(**overrides)


def test_comparison_registration_does_not_grant_warp_phase_admission():
    for name in ("wasp", "stwebagentbench", "doomarena"):
        capabilities = get_benchmark_capabilities(name)
        assert capabilities.is_comparison_only is True
        assert capabilities.warp_phase_admission == ()
        assert capabilities.supports("comparison_ingestion") is True
        for phase in (
            "phase_1_generation",
            "phase_2_generation",
            "phase_2_feasibility",
            "phase_4_execution",
            "warp_evaluation",
        ):
            assert capabilities.supports(phase) is False
        with pytest.raises(ValueError, match="does not support capability"):
            capabilities.require("phase_4_execution")


def test_evaluator_authority_is_selected_by_task_id_presence():
    assert resolve_evaluator_authority("WebArena Verified", task_id=123) == (
        "canonical_vendor_task_id"
    )
    assert resolve_evaluator_authority("WebArena Verified", task_id=None) == (
        "warp_local_task_idless"
    )
    for task_id in (None, 123):
        with pytest.raises(ValueError, match="no evaluator authority"):
            resolve_evaluator_authority("wasp", task_id=task_id)


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
    benchmark = _gate_phase_2c_benchmark(
        task_records=[{"id": "task-1", "benchmark": "WebArena Verified"}],
        raw_instances={
            "benchmark_name": "WebArena Verified",
            "instances": [{"site_name": "gitlab", "site_url": "http://gitlab.test"}],
        },
        instances=[{"site_name": "gitlab", "site_url": "http://gitlab.test"}],
    )

    assert benchmark == "webarena_verified"


def test_phase_2c_gate_accepts_instances_top_level_benchmark():
    benchmark = _gate_phase_2c_benchmark(
        task_records=[{"id": "task-1", "benchmark": "WebArena Verified"}],
        raw_instances={
            "benchmark": "WebArena Verified",
            "instances": [{"site_name": "gitlab", "site_url": "http://gitlab.test"}],
        },
        instances=[{"site_name": "gitlab", "site_url": "http://gitlab.test"}],
    )

    assert benchmark == "webarena_verified"


def test_phase_2c_gate_rejects_unsupported_benchmark():
    with pytest.raises(ValueError, match="does not support WARP Taskgen Phase 2c"):
        _gate_phase_2c_benchmark(
            task_records=[{"id": "task-1", "benchmark": "wasp"}],
            raw_instances={"benchmark_name": "wasp", "instances": [{"site_name": "gitlab"}]},
            instances=[{"site_name": "gitlab", "benchmark": "wasp"}],
        )


def test_phase_2c_gate_rejects_missing_metadata():
    with pytest.raises(ValueError, match="missing benchmark metadata"):
        _gate_phase_2c_benchmark(
            task_records=[{"id": "task-1"}],
            raw_instances={"instances": [{"site_name": "gitlab"}]},
            instances=[{"site_name": "gitlab"}],
        )


def test_phase_2c_gate_rejects_task_instance_mismatch():
    with pytest.raises(ValueError, match="mixed benchmark metadata"):
        _gate_phase_2c_benchmark(
            task_records=[{"id": "task-1", "benchmark": "webarena_verified"}],
            raw_instances={"benchmark_name": "wasp", "instances": [{"site_name": "gitlab"}]},
            instances=[{"site_name": "gitlab", "benchmark": "wasp"}],
        )
