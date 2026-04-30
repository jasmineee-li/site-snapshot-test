from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldsim.config import BenchmarkInstance, VerificationProxy
from worldsim.phases import phase_0_recon

VERIFICATION_OUTPUT = "/workspace/output/VERIFICATION_CAPABILITIES.json"
DATA_MODEL_OUTPUT = "/workspace/output/DATA_MODEL.json"
AGENT_CONTEXT_OUTPUT = "/workspace/output/AGENT_CONTEXT.json"
INJECTION_OUTPUT = "/workspace/output/INJECTION_SURFACE.json"


class _FakeVolume:
    def __init__(self, name: str = "fake-volume") -> None:
        self.name = name

    def read_only(self):
        return self


@pytest.fixture(autouse=True)
def _stub_phase_0c_volume_upload(monkeypatch):
    async def fake_upload_to_volume(path):
        return _FakeVolume(str(path))

    monkeypatch.setattr(phase_0_recon, "upload_to_volume", fake_upload_to_volume)


def _valid_verification_capabilities() -> list[dict]:
    return [
        {
            "eval_type": "NetworkEventEvaluator",
            "description": "Checks the outbound network request.",
            "interface": {"params": {"expected": "object"}, "returns": "boolean"},
            "constraints": "",
            "examples": [],
        }
    ]


def _valid_data_model() -> list[dict]:
    return [
        {
            "entity": "products",
            "storage": "products",
            "fields": [
                {"name": "title", "type": "string", "mutable_via_ui": False},
                {"name": "description", "type": "string", "mutable_via_ui": True},
            ],
            "relationships": [],
            "sample_values": [{"title": "Desk lamp"}],
        }
    ]


def _valid_agent_context() -> dict:
    return {
        "response_format": {
            "requires_structured_output": False,
            "output_schema": None,
            "per_task_format_field": None,
            "description": "Evaluation inspects browser state directly.",
        },
        "authentication": {
            "pre_authenticated": False,
            "credentials": {"username": "demo", "password": "secret"},
            "description": "Log in with the seeded shopping test account.",
        },
        "auth_mechanism": {
            "type": "form_login",
            "form_login": {
                "login_url": "/login",
                "username_selector": "#email",
                "password_selector": "#password",
                "submit_selector": "button[type='submit']",
                "success_url_substring": "/account",
            },
        },
        "agent_prompt_template": None,
        "site_context": {
            "platform_name": "Shopping",
            "description": "An ecommerce storefront with order history and product pages.",
        },
    }


def _valid_injection_surface(*, source_field: str = "products.description") -> dict:
    return {
        "injection_surface": [
            {
                "id": "surface-1",
                "location_page": "/orders/1",
                "source_field": source_field,
                "rendering_format": "plaintext",
                "visibility": "always_shown",
                "controllable_by_tier": "any_user",
                "controllability_justification": "Authenticated shoppers can submit this content.",
                "delivery_channels": [
                    {
                        "mechanism": "form",
                        "privileged_seed": False,
                        "path_template": "/orders/{id}",
                        "method": "POST",
                        "body_field": "detail",
                        "table": None,
                        "column": None,
                        "postcondition": {
                            "type": "db_row_value",
                            "table": "orders",
                            "value_column": "detail",
                            "where": {"id": {"path_param": "id"}},
                        },
                    },
                ],
                "rendering_context": "Order detail page body copy.",
                "compatible_concealments": ["plaintext"],
                "attacker_realism": "medium",
                "realism_justification": "Customer-visible free-form text field.",
            }
        ],
        "existing_task_coverage": {
            "injection_surfaces_with_task_coverage": ["surface-1"],
            "injection_surfaces_without_task_coverage": [],
        },
    }


def _sandbox_json(output_path: str, payload: object) -> dict[str, str | None]:
    return {
        output_path: json.dumps(payload),
        "_summary": None,
    }


def _instances_config(benchmark_root: Path, *, site_url: str = "http://shopping.test") -> dict:
    return {
        "benchmark_name": "WebArena Verified",
        "benchmark_codebase": str(benchmark_root),
        "instances": [
            {
                "site_name": "shopping",
                "site_url": site_url,
            }
        ],
    }


def _benchmark_setup(tmp_path):
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    source_file = benchmark_root / "shopping.txt"
    source_file.write_text("demo")
    return benchmark_root, source_file


@pytest.mark.asyncio
async def test_run_phase_0c_fails_when_any_tier_output_is_missing(monkeypatch, tmp_path):
    benchmark_root, source_file = _benchmark_setup(tmp_path)

    async def fake_run_claude_in_sandbox(*args, **kwargs):
        label = kwargs["label"]
        if "-A-verify" in label:
            return {"_summary": None}
        if "-B-data" in label:
            return _sandbox_json(DATA_MODEL_OUTPUT, _valid_data_model())
        if "-C-context" in label:
            return _sandbox_json(AGENT_CONTEXT_OUTPUT, _valid_agent_context())
        if "-DE-inject" in label:
            return _sandbox_json(INJECTION_OUTPUT, _valid_injection_surface())
        raise AssertionError(f"unexpected sandbox label: {label}")

    monkeypatch.setattr(phase_0_recon, "run_claude_in_sandbox", fake_run_claude_in_sandbox)

    with pytest.raises(RuntimeError, match="did not complete all required site profiles"):
        await phase_0_recon.run_phase_0c(
            manifest={"evaluation": {"eval_types": ["NetworkEventEvaluator"]}},
            sandbox_map={"shopping": [str(source_file)]},
            benchmark_root=benchmark_root,
            output_dir=tmp_path / "out",
        )

    assert not (tmp_path / "out" / "BENCHMARK_PROFILE_shopping.json").exists()
    assert not (tmp_path / "out" / "AGENT_CONTEXT_shopping.json").exists()


@pytest.mark.asyncio
async def test_run_phase_0a_passes_explicit_sandbox_model(monkeypatch, tmp_path):
    benchmark_root, _ = _benchmark_setup(tmp_path)
    captured = {}

    async def fake_upload_to_volume(path):
        assert path == benchmark_root
        return object()

    async def fake_run_claude_in_sandbox(*args, **kwargs):
        captured["model"] = kwargs.get("model")
        return {
            "/workspace/output/BENCHMARK_MANIFEST.json": json.dumps(
                {
                    "sites": [],
                    "evaluation": {"eval_types": []},
                    "benchmark_codebase": str(benchmark_root),
                }
            ),
            "/workspace/output/BENCHMARK_MANIFEST.md": "ok",
            "_summary": None,
        }

    monkeypatch.setattr(phase_0_recon, "upload_to_volume", fake_upload_to_volume)
    monkeypatch.setattr(phase_0_recon, "run_claude_in_sandbox", fake_run_claude_in_sandbox)

    await phase_0_recon.run_phase_0a(
        benchmark_root,
        tmp_path / "out",
        sandbox_model="claude-opus-4-6",
    )

    assert captured["model"] == "claude-opus-4-6"


@pytest.mark.asyncio
async def test_run_phase_0c_mounts_staged_site_subset_via_read_only_volume(monkeypatch, tmp_path):
    benchmark_root, source_file = _benchmark_setup(tmp_path)
    captured_volumes: list[dict[str, object] | None] = []

    async def fake_run_claude_in_sandbox(*args, **kwargs):
        captured_volumes.append(kwargs.get("volumes"))
        label = kwargs["label"]
        if "-A-verify" in label:
            return _sandbox_json(VERIFICATION_OUTPUT, _valid_verification_capabilities())
        if "-B-data" in label:
            return _sandbox_json(DATA_MODEL_OUTPUT, _valid_data_model())
        if "-C-context" in label:
            return _sandbox_json(AGENT_CONTEXT_OUTPUT, _valid_agent_context())
        if "-DE-inject" in label:
            return _sandbox_json(INJECTION_OUTPUT, _valid_injection_surface())
        raise AssertionError(f"unexpected sandbox label: {label}")

    monkeypatch.setattr(phase_0_recon, "run_claude_in_sandbox", fake_run_claude_in_sandbox)

    await phase_0_recon.run_phase_0c(
        manifest={"evaluation": {"eval_types": ["NetworkEventEvaluator"]}},
        sandbox_map={"shopping": [str(source_file)]},
        benchmark_root=benchmark_root,
        output_dir=tmp_path / "out",
    )

    assert captured_volumes
    for volumes in captured_volumes:
        assert volumes is not None
        assert list(volumes) == ["/workspace/benchmark"]
        assert isinstance(volumes["/workspace/benchmark"], _FakeVolume)


@pytest.mark.asyncio
async def test_run_phase_0c_honors_site_filter(monkeypatch, tmp_path):
    profiled_sites: list[str] = []
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    gitlab_file = benchmark_root / "gitlab.py"
    reddit_file = benchmark_root / "reddit.py"
    shopping_file = benchmark_root / "shopping.py"
    for path in (gitlab_file, reddit_file, shopping_file):
        path.write_text("# fixture", encoding="utf-8")

    async def fake_profile_one_site_tiered(**kwargs):
        profiled_sites.append(kwargs["site_name"])
        return kwargs["site_name"], {"profile": {}, "agent_context": {}}

    monkeypatch.setattr(phase_0_recon, "_profile_one_site_tiered", fake_profile_one_site_tiered)

    result = await phase_0_recon.run_phase_0c(
        {"evaluation": {"eval_types": ["NetworkEventEvaluator"]}},
        {
            "gitlab": [str(gitlab_file)],
            "reddit": [str(reddit_file)],
            "shopping": [str(shopping_file)],
        },
        benchmark_root,
        tmp_path / "phase_0c",
        site_filter={"gitlab", "reddit"},
    )

    assert sorted(profiled_sites) == ["gitlab", "reddit"]
    assert sorted(result) == ["gitlab", "reddit"]


@pytest.mark.asyncio
async def test_run_phase_0c_does_not_publish_invalid_profiles(monkeypatch, tmp_path):
    benchmark_root, source_file = _benchmark_setup(tmp_path)
    de_attempts = 0

    async def fake_run_claude_in_sandbox(*args, **kwargs):
        nonlocal de_attempts
        label = kwargs["label"]
        if "-A-verify" in label:
            return _sandbox_json(VERIFICATION_OUTPUT, _valid_verification_capabilities())
        if "-B-data" in label:
            return _sandbox_json(DATA_MODEL_OUTPUT, _valid_data_model())
        if "-C-context" in label:
            return _sandbox_json(AGENT_CONTEXT_OUTPUT, _valid_agent_context())
        if "-DE-inject" in label:
            de_attempts += 1
            return _sandbox_json(
                INJECTION_OUTPUT, _valid_injection_surface(source_field="posts.body")
            )
        raise AssertionError(f"unexpected sandbox label: {label}")

    monkeypatch.setattr(phase_0_recon, "run_claude_in_sandbox", fake_run_claude_in_sandbox)

    with pytest.raises(RuntimeError, match="failed validation"):
        await phase_0_recon.run_phase_0c(
            manifest={"evaluation": {"eval_types": ["NetworkEventEvaluator"]}},
            sandbox_map={"shopping": [str(source_file)]},
            benchmark_root=benchmark_root,
            output_dir=tmp_path / "out",
        )

    assert de_attempts == 1 + phase_0_recon.PROFILE_FIX_MAX_ITERATIONS
    assert not (tmp_path / "out" / "BENCHMARK_PROFILE_shopping.json").exists()
    assert not (tmp_path / "out" / "AGENT_CONTEXT_shopping.json").exists()


@pytest.mark.asyncio
async def test_correction_loop_fixes_invalid_tier_output(monkeypatch, tmp_path):
    benchmark_root, source_file = _benchmark_setup(tmp_path)
    de_attempts = 0
    de_prompts: list[str] = []

    async def fake_run_claude_in_sandbox(*args, **kwargs):
        nonlocal de_attempts
        label = kwargs["label"]
        if "-A-verify" in label:
            return _sandbox_json(VERIFICATION_OUTPUT, _valid_verification_capabilities())
        if "-B-data" in label:
            return _sandbox_json(DATA_MODEL_OUTPUT, _valid_data_model())
        if "-C-context" in label:
            return _sandbox_json(AGENT_CONTEXT_OUTPUT, _valid_agent_context())
        if "-DE-inject" in label:
            de_attempts += 1
            de_prompts.append(kwargs["prompt"])
            source_field = "posts.body" if de_attempts == 1 else "products.description"
            return _sandbox_json(
                INJECTION_OUTPUT, _valid_injection_surface(source_field=source_field)
            )
        raise AssertionError(f"unexpected sandbox label: {label}")

    monkeypatch.setattr(phase_0_recon, "run_claude_in_sandbox", fake_run_claude_in_sandbox)

    result = await phase_0_recon.run_phase_0c(
        manifest={"evaluation": {"eval_types": ["NetworkEventEvaluator"]}},
        sandbox_map={"shopping": [str(source_file)]},
        benchmark_root=benchmark_root,
        output_dir=tmp_path / "out",
    )

    assert "shopping" in result
    assert de_attempts == 2
    assert "<validation_feedback>" in de_prompts[1]
    assert '"artifact": "INJECTION_SURFACE.json"' in de_prompts[1]
    assert "entity.field format" in de_prompts[1]
    assert (tmp_path / "out" / "BENCHMARK_PROFILE_shopping.json").exists()
    assert (tmp_path / "out" / "AGENT_CONTEXT_shopping.json").exists()
    profile = json.loads((tmp_path / "out" / "BENCHMARK_PROFILE_shopping.json").read_text())
    assert profile["agent_context"] == _valid_agent_context()


@pytest.mark.asyncio
async def test_correction_loop_hard_fails_after_max_retries(monkeypatch, tmp_path):
    benchmark_root, source_file = _benchmark_setup(tmp_path)
    de_attempts = 0

    async def fake_run_claude_in_sandbox(*args, **kwargs):
        nonlocal de_attempts
        label = kwargs["label"]
        if "-A-verify" in label:
            return _sandbox_json(VERIFICATION_OUTPUT, _valid_verification_capabilities())
        if "-B-data" in label:
            return _sandbox_json(DATA_MODEL_OUTPUT, _valid_data_model())
        if "-C-context" in label:
            return _sandbox_json(AGENT_CONTEXT_OUTPUT, _valid_agent_context())
        if "-DE-inject" in label:
            de_attempts += 1
            return _sandbox_json(
                INJECTION_OUTPUT, _valid_injection_surface(source_field="posts.body")
            )
        raise AssertionError(f"unexpected sandbox label: {label}")

    monkeypatch.setattr(phase_0_recon, "run_claude_in_sandbox", fake_run_claude_in_sandbox)

    with pytest.raises(RuntimeError, match="failed validation"):
        await phase_0_recon.run_phase_0c(
            manifest={"evaluation": {"eval_types": ["NetworkEventEvaluator"]}},
            sandbox_map={"shopping": [str(source_file)]},
            benchmark_root=benchmark_root,
            output_dir=tmp_path / "out",
        )

    assert de_attempts == 1 + phase_0_recon.PROFILE_FIX_MAX_ITERATIONS
    assert not (tmp_path / "out" / "BENCHMARK_PROFILE_shopping.json").exists()
    assert not (tmp_path / "out" / "AGENT_CONTEXT_shopping.json").exists()


@pytest.mark.asyncio
async def test_run_phase_0c_threads_explicit_sandbox_model(monkeypatch, tmp_path):
    benchmark_root, source_file = _benchmark_setup(tmp_path)
    seen_models: list[str] = []

    async def fake_run_claude_in_sandbox(*args, **kwargs):
        seen_models.append(kwargs["model"])
        label = kwargs["label"]
        if "-A-verify" in label:
            return _sandbox_json(VERIFICATION_OUTPUT, _valid_verification_capabilities())
        if "-B-data" in label:
            return _sandbox_json(DATA_MODEL_OUTPUT, _valid_data_model())
        if "-C-context" in label:
            return _sandbox_json(AGENT_CONTEXT_OUTPUT, _valid_agent_context())
        if "-DE-inject" in label:
            return _sandbox_json(INJECTION_OUTPUT, _valid_injection_surface())
        raise AssertionError(f"unexpected sandbox label: {label}")

    monkeypatch.setattr(phase_0_recon, "run_claude_in_sandbox", fake_run_claude_in_sandbox)

    await phase_0_recon.run_phase_0c(
        manifest={"evaluation": {"eval_types": ["NetworkEventEvaluator"]}},
        sandbox_map={"shopping": [str(source_file)]},
        benchmark_root=benchmark_root,
        output_dir=tmp_path / "out",
        sandbox_model="claude-opus-4-6",
    )

    assert seen_models
    assert set(seen_models) == {"claude-opus-4-6"}


@pytest.mark.asyncio
async def test_run_phase_0c_reprofiles_when_existing_outputs_are_stale(monkeypatch, tmp_path):
    benchmark_root, source_file = _benchmark_setup(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    site_name = "shopping"

    stale_profile = {
        "site_name": site_name,
        "verification_capabilities": _valid_verification_capabilities(),
        "data_model": _valid_data_model(),
        "agent_context": _valid_agent_context(),
        "injection_surface": _valid_injection_surface()["injection_surface"],
        "existing_task_coverage": _valid_injection_surface()["existing_task_coverage"],
    }
    (output_dir / f"BENCHMARK_PROFILE_{site_name}.json").write_text(
        json.dumps(stale_profile), encoding="utf-8"
    )
    (output_dir / f"AGENT_CONTEXT_{site_name}.json").write_text(
        json.dumps(_valid_agent_context()), encoding="utf-8"
    )
    (output_dir / f"PROFILE_METADATA_{site_name}.json").write_text(
        json.dumps(
            {
                "site_name": site_name,
                "benchmark_root": str(benchmark_root),
                "sandbox_model": "claude-opus-4-6",
            }
        ),
        encoding="utf-8",
    )

    calls: list[str] = []

    async def fake_run_claude_in_sandbox(*args, **kwargs):
        calls.append(kwargs["label"])
        label = kwargs["label"]
        if "-A-verify" in label:
            return _sandbox_json(VERIFICATION_OUTPUT, _valid_verification_capabilities())
        if "-B-data" in label:
            return _sandbox_json(DATA_MODEL_OUTPUT, _valid_data_model())
        if "-C-context" in label:
            context = _valid_agent_context()
            context["site_context"]["description"] = "Fresh profile"
            return _sandbox_json(AGENT_CONTEXT_OUTPUT, context)
        if "-DE-inject" in label:
            return _sandbox_json(INJECTION_OUTPUT, _valid_injection_surface())
        raise AssertionError(f"unexpected sandbox label: {label}")

    monkeypatch.setattr(phase_0_recon, "run_claude_in_sandbox", fake_run_claude_in_sandbox)

    result = await phase_0_recon.run_phase_0c(
        manifest={"evaluation": {"eval_types": ["NetworkEventEvaluator"]}},
        sandbox_map={site_name: [str(source_file)]},
        benchmark_root=benchmark_root,
        output_dir=output_dir,
    )

    assert site_name in result
    assert calls, "stale outputs must be re-profiled instead of silently skipped"
    profile = json.loads((output_dir / f"BENCHMARK_PROFILE_{site_name}.json").read_text())
    assert profile["agent_context"]["site_context"]["description"] == "Fresh profile"


@pytest.mark.asyncio
async def test_run_phase_0c_reprofiles_when_instance_site_url_changes(monkeypatch, tmp_path):
    benchmark_root, source_file = _benchmark_setup(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    site_name = "shopping"

    cached_profile = {
        "site_name": site_name,
        "verification_capabilities": _valid_verification_capabilities(),
        "data_model": _valid_data_model(),
        "agent_context": _valid_agent_context(),
        "injection_surface": _valid_injection_surface()["injection_surface"],
        "existing_task_coverage": _valid_injection_surface()["existing_task_coverage"],
    }
    (output_dir / f"BENCHMARK_PROFILE_{site_name}.json").write_text(
        json.dumps(cached_profile), encoding="utf-8"
    )
    (output_dir / f"AGENT_CONTEXT_{site_name}.json").write_text(
        json.dumps(_valid_agent_context()), encoding="utf-8"
    )
    (output_dir / f"PROFILE_METADATA_{site_name}.json").write_text(
        json.dumps(
            {
                "site_name": site_name,
                "benchmark_root": str(benchmark_root),
                "sandbox_model": "claude-sonnet-4-6",
                "instance_site_url": "http://shopping-v1.test",
            }
        ),
        encoding="utf-8",
    )

    calls: list[str] = []

    async def fake_run_claude_in_sandbox(*args, **kwargs):
        calls.append(kwargs["label"])
        label = kwargs["label"]
        if "-A-verify" in label:
            return _sandbox_json(VERIFICATION_OUTPUT, _valid_verification_capabilities())
        if "-B-data" in label:
            return _sandbox_json(DATA_MODEL_OUTPUT, _valid_data_model())
        if "-C-context" in label:
            context = _valid_agent_context()
            context["site_context"]["description"] = "Fresh instance URL profile"
            return _sandbox_json(AGENT_CONTEXT_OUTPUT, context)
        if "-DE-inject" in label:
            return _sandbox_json(INJECTION_OUTPUT, _valid_injection_surface())
        raise AssertionError(f"unexpected sandbox label: {label}")

    monkeypatch.setattr(phase_0_recon, "run_claude_in_sandbox", fake_run_claude_in_sandbox)

    result = await phase_0_recon.run_phase_0c(
        manifest={"evaluation": {"eval_types": ["NetworkEventEvaluator"]}},
        sandbox_map={site_name: [str(source_file)]},
        benchmark_root=benchmark_root,
        output_dir=output_dir,
        instances=[BenchmarkInstance(site_name=site_name, site_url="http://shopping-v2.test")],
    )

    assert site_name in result
    assert calls, "changing the representative instance URL must invalidate reuse"
    refreshed = json.loads((output_dir / f"BENCHMARK_PROFILE_{site_name}.json").read_text())
    assert refreshed["agent_context"]["site_context"]["description"] == "Fresh instance URL profile"
    metadata = json.loads((output_dir / f"PROFILE_METADATA_{site_name}.json").read_text())
    assert metadata["instance_site_url"] == "http://shopping-v2.test"


@pytest.mark.asyncio
async def test_run_phase_0c_stages_sanitized_instance_connectivity(monkeypatch, tmp_path):
    benchmark_root, source_file = _benchmark_setup(tmp_path)
    captured: dict[str, object] = {}

    async def fake_run_claude_in_sandbox(*args, **kwargs):
        label = kwargs["label"]
        if "-A-verify" in label:
            return _sandbox_json(VERIFICATION_OUTPUT, _valid_verification_capabilities())
        if "-B-data" in label:
            return _sandbox_json(DATA_MODEL_OUTPUT, _valid_data_model())
        if "-C-context" in label:
            return _sandbox_json(AGENT_CONTEXT_OUTPUT, _valid_agent_context())
        if "-DE-inject" in label:
            connectivity_path = kwargs["site_files"]["/workspace/inputs/INSTANCE_CONNECTIVITY.json"]
            captured["connectivity"] = json.loads(Path(connectivity_path).read_text())
            return _sandbox_json(INJECTION_OUTPUT, _valid_injection_surface())
        raise AssertionError(f"unexpected sandbox label: {label}")

    monkeypatch.setattr(phase_0_recon, "run_claude_in_sandbox", fake_run_claude_in_sandbox)

    await phase_0_recon.run_phase_0c(
        manifest={"evaluation": {"eval_types": ["NetworkEventEvaluator"]}},
        sandbox_map={"shopping": [str(source_file)]},
        benchmark_root=benchmark_root,
        output_dir=tmp_path / "out",
        instances=[BenchmarkInstance(site_name="SHOPPING", site_url="http://shopping.test/")],
    )

    assert captured["connectivity"] == {
        "site_name": "shopping",
        "site_url": "http://shopping.test",
    }


def test_load_phase_0c_instances_rejects_invalid_config_shape(tmp_path):
    instances_path = tmp_path / "instances.json"
    instances_path.write_text("[]", encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid instances config"):
        phase_0_recon._load_phase_0c_config(instances_path)


def test_build_instance_site_url_map_rejects_sensitive_site_urls():
    with pytest.raises(ValueError, match="must not include credentials"):
        BenchmarkInstance(site_name="shopping", site_url="http://user:pass@shopping.test")


def test_sanitize_instance_site_url_canonicalizes_equivalent_origins():
    assert (
        phase_0_recon._sanitize_instance_site_url(
            "HTTP://SHOPPING.test:80/",
            site_name="shopping",
        )
        == "http://shopping.test"
    )
    assert (
        phase_0_recon._sanitize_instance_site_url(
            "https://SHOPPING.test:443/base/",
            site_name="shopping",
        )
        == "https://shopping.test/base"
    )


@pytest.mark.asyncio
async def test_run_persists_instances_path_across_phase_0_state(monkeypatch, tmp_path):
    benchmark_root, source_file = _benchmark_setup(tmp_path)
    instances_path = tmp_path / "instances.json"
    instances_path.write_text(json.dumps(_instances_config(benchmark_root)), encoding="utf-8")

    state_calls: list[tuple[str, dict]] = []
    phase_0c_kwargs: dict[str, object] = {}

    async def fake_run_phase_0a(*args, **kwargs):
        return {
            "sites": [{"name": "shopping"}],
            "evaluation": {"eval_types": ["NetworkEventEvaluator"]},
            "benchmark_codebase": str(benchmark_root),
        }

    async def fake_run_phase_0c(*args, **kwargs):
        phase_0c_kwargs.update(kwargs)
        return {"shopping": {"profile": {}, "agent_context": {}}}

    async def fake_preflight():
        return None

    monkeypatch.setattr(phase_0_recon, "preflight_sandbox_environment", fake_preflight)
    monkeypatch.setattr(phase_0_recon, "run_phase_0a", fake_run_phase_0a)
    monkeypatch.setattr(
        phase_0_recon,
        "compute_sandbox_maps",
        lambda manifest, benchmark: {"shopping": [str(source_file)]},
    )
    monkeypatch.setattr(phase_0_recon, "run_phase_0c", fake_run_phase_0c)
    monkeypatch.setattr(phase_0_recon, "get_state_dir", lambda: tmp_path / "state")
    monkeypatch.setattr(
        phase_0_recon,
        "save_state",
        lambda step, iteration=0, **metadata: state_calls.append((step, metadata)),
    )
    monkeypatch.setattr(
        phase_0_recon.cost_tracker, "log_phase_summary", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(phase_0_recon.cost_tracker, "save", lambda *args, **kwargs: None)

    rc = await phase_0_recon.run(
        benchmark=benchmark_root,
        sub="0",
        instances_path=instances_path,
        site_filter={"shopping"},
    )

    assert rc == 0
    phase_calls = [call for call in state_calls if call[0].startswith("phase_0")]
    assert phase_calls
    assert all(call[1]["instances_path"] == str(instances_path) for call in phase_calls)
    assert phase_0c_kwargs["site_filter"] == {"shopping"}


def test_profile_injection_surface_prompt_includes_delivery_feasibility_probing():
    prompt_path = (
        Path(phase_0_recon.__file__).resolve().parents[1]
        / "prompts"
        / "profile-injection-surface.md"
    )
    prompt = prompt_path.read_text(encoding="utf-8")

    # After 1d32012, the prompt allows minimal POST probing for delivery
    # feasibility instead of prohibiting all write methods during profiling.
    assert "ONE minimal POST per form channel" in prompt
    assert "Live instance verification" in prompt


@pytest.mark.asyncio
async def test_run_phase_0c_reprofiles_when_profile_metadata_is_malformed(monkeypatch, tmp_path):
    benchmark_root, source_file = _benchmark_setup(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    site_name = "shopping"

    profile = {
        "site_name": site_name,
        "verification_capabilities": _valid_verification_capabilities(),
        "data_model": _valid_data_model(),
        "agent_context": _valid_agent_context(),
        "injection_surface": _valid_injection_surface()["injection_surface"],
        "existing_task_coverage": _valid_injection_surface()["existing_task_coverage"],
    }
    (output_dir / f"BENCHMARK_PROFILE_{site_name}.json").write_text(
        json.dumps(profile), encoding="utf-8"
    )
    (output_dir / f"AGENT_CONTEXT_{site_name}.json").write_text(
        json.dumps(_valid_agent_context()), encoding="utf-8"
    )
    (output_dir / f"PROFILE_METADATA_{site_name}.json").write_text(
        "{not-json",
        encoding="utf-8",
    )

    calls: list[str] = []

    async def fake_run_claude_in_sandbox(*args, **kwargs):
        calls.append(kwargs["label"])
        label = kwargs["label"]
        if "-A-verify" in label:
            return _sandbox_json(VERIFICATION_OUTPUT, _valid_verification_capabilities())
        if "-B-data" in label:
            return _sandbox_json(DATA_MODEL_OUTPUT, _valid_data_model())
        if "-C-context" in label:
            context = _valid_agent_context()
            context["site_context"]["description"] = "Fresh malformed metadata profile"
            return _sandbox_json(AGENT_CONTEXT_OUTPUT, context)
        if "-DE-inject" in label:
            return _sandbox_json(INJECTION_OUTPUT, _valid_injection_surface())
        raise AssertionError(f"unexpected sandbox label: {label}")

    monkeypatch.setattr(phase_0_recon, "run_claude_in_sandbox", fake_run_claude_in_sandbox)

    result = await phase_0_recon.run_phase_0c(
        manifest={"evaluation": {"eval_types": ["NetworkEventEvaluator"]}},
        sandbox_map={site_name: [str(source_file)]},
        benchmark_root=benchmark_root,
        output_dir=output_dir,
    )

    assert site_name in result
    assert calls, "malformed metadata must trigger re-profiling"
    refreshed = json.loads((output_dir / f"BENCHMARK_PROFILE_{site_name}.json").read_text())
    assert refreshed["agent_context"]["site_context"]["description"] == (
        "Fresh malformed metadata profile"
    )


@pytest.mark.asyncio
async def test_run_phase_0c_reprofiles_when_proxy_metadata_changes(monkeypatch, tmp_path):
    benchmark_root, source_file = _benchmark_setup(tmp_path)
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    site_name = "shopping"

    profile = {
        "site_name": site_name,
        "verification_capabilities": _valid_verification_capabilities(),
        "data_model": _valid_data_model(),
        "agent_context": _valid_agent_context(),
        "injection_surface": _valid_injection_surface()["injection_surface"],
        "existing_task_coverage": _valid_injection_surface()["existing_task_coverage"],
    }
    (output_dir / f"BENCHMARK_PROFILE_{site_name}.json").write_text(
        json.dumps(profile), encoding="utf-8"
    )
    (output_dir / f"AGENT_CONTEXT_{site_name}.json").write_text(
        json.dumps(_valid_agent_context()), encoding="utf-8"
    )
    (output_dir / f"PROFILE_METADATA_{site_name}.json").write_text(
        json.dumps(
            {
                "site_name": site_name,
                "benchmark_root": str(benchmark_root),
                "sandbox_model": "claude-sonnet-4-6",
                "instance_site_url": "http://shopping.test",
                "verification_proxy": {
                    "scheme": "http",
                    "port_offset": 10000,
                    "token_sha256": "stale",
                },
            }
        ),
        encoding="utf-8",
    )

    calls: list[str] = []

    async def fake_run_claude_in_sandbox(*args, **kwargs):
        calls.append(kwargs["label"])
        label = kwargs["label"]
        if "-A-verify" in label:
            return _sandbox_json(VERIFICATION_OUTPUT, _valid_verification_capabilities())
        if "-B-data" in label:
            return _sandbox_json(DATA_MODEL_OUTPUT, _valid_data_model())
        if "-C-context" in label:
            context = _valid_agent_context()
            context["site_context"]["description"] = "Fresh proxy metadata profile"
            return _sandbox_json(AGENT_CONTEXT_OUTPUT, context)
        if "-DE-inject" in label:
            return _sandbox_json(INJECTION_OUTPUT, _valid_injection_surface())
        raise AssertionError(f"unexpected sandbox label: {label}")

    monkeypatch.setattr(phase_0_recon, "run_claude_in_sandbox", fake_run_claude_in_sandbox)

    result = await phase_0_recon.run_phase_0c(
        manifest={"evaluation": {"eval_types": ["NetworkEventEvaluator"]}},
        sandbox_map={site_name: [str(source_file)]},
        benchmark_root=benchmark_root,
        output_dir=output_dir,
        instances=[BenchmarkInstance(site_name="shopping", site_url="http://shopping.test")],
        verification_proxy=VerificationProxy(
            token="fresh-token", port_offset=10000, scheme="https"
        ),
    )

    assert site_name in result
    assert calls, "proxy metadata change must trigger re-profiling"
    refreshed = json.loads((output_dir / f"BENCHMARK_PROFILE_{site_name}.json").read_text())
    assert refreshed["agent_context"]["site_context"]["description"] == (
        "Fresh proxy metadata profile"
    )
    metadata = json.loads((output_dir / f"PROFILE_METADATA_{site_name}.json").read_text())
    assert metadata["verification_proxy"]["scheme"] == "https"
