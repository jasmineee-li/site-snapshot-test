from __future__ import annotations

import json

import pytest

from worldsim.phases import phase_0_recon

VERIFICATION_OUTPUT = "/workspace/output/VERIFICATION_CAPABILITIES.json"
DATA_MODEL_OUTPUT = "/workspace/output/DATA_MODEL.json"
AGENT_CONTEXT_OUTPUT = "/workspace/output/AGENT_CONTEXT.json"
INJECTION_OUTPUT = "/workspace/output/INJECTION_SURFACE.json"


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
                    {
                        "mechanism": "sql",
                        "privileged_seed": False,
                        "path_template": None,
                        "method": None,
                        "body_field": None,
                        "table": "orders",
                        "column": "detail",
                        "postcondition": None,
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
            return _sandbox_json(INJECTION_OUTPUT, _valid_injection_surface(source_field="posts.body"))
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
            return _sandbox_json(INJECTION_OUTPUT, _valid_injection_surface(source_field=source_field))
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
    assert "CORRECTION NEEDED" in de_prompts[1]
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
            return _sandbox_json(INJECTION_OUTPUT, _valid_injection_surface(source_field="posts.body"))
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
