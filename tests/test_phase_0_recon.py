from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from worldsim.phases import phase_0_recon


@pytest.mark.asyncio
async def test_run_phase_0c_fails_when_any_profile_is_missing(monkeypatch, tmp_path):
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    source_file = benchmark_root / "shopping.txt"
    source_file.write_text("demo")

    async def fake_run_claude_in_sandbox(*args, **kwargs):
        return {
            "/workspace/output/BENCHMARK_PROFILE.md": "# profile",
            "_summary": None,
        }

    monkeypatch.setattr(phase_0_recon, "run_claude_in_sandbox", fake_run_claude_in_sandbox)

    with pytest.raises(RuntimeError, match="did not complete all required site profiles"):
        await phase_0_recon.run_phase_0c(
            manifest={"evaluation": {"eval_types": []}},
            sandbox_map={"shopping": [str(source_file)]},
            benchmark_root=benchmark_root,
            output_dir=tmp_path / "out",
        )


@pytest.mark.asyncio
async def test_run_phase_0c_does_not_publish_invalid_profiles(monkeypatch, tmp_path):
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    source_file = benchmark_root / "shopping.txt"
    source_file.write_text("demo")

    async def fake_run_claude_in_sandbox(*args, **kwargs):
        return {
            "/workspace/output/BENCHMARK_PROFILE.json": (
                '{"data_model":[{"fields":[{"name":"title"}]}],'
                '"injection_surface":[{"id":"surface-1","source_field":"posts.body"}],'
                '"verification_capabilities":[]}'
            ),
            "/workspace/output/BENCHMARK_PROFILE.md": "# invalid profile",
            "_summary": None,
        }

    monkeypatch.setattr(phase_0_recon, "run_claude_in_sandbox", fake_run_claude_in_sandbox)

    with pytest.raises(RuntimeError, match="failed validation"):
        await phase_0_recon.run_phase_0c(
            manifest={"evaluation": {"eval_types": []}},
            sandbox_map={"shopping": [str(source_file)]},
            benchmark_root=benchmark_root,
            output_dir=tmp_path / "out",
        )

    assert not (tmp_path / "out" / "BENCHMARK_PROFILE_shopping.json").exists()
    assert not (tmp_path / "out" / "BENCHMARK_PROFILE_shopping.md").exists()


@pytest.mark.asyncio
async def test_correction_loop_fixes_invalid_profile(monkeypatch, tmp_path):
    """Sandbox returns invalid profile first, valid profile on retry -- should succeed."""
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    source_file = benchmark_root / "shopping.txt"
    source_file.write_text("demo")

    # First call: entity "posts" not in data_model (triggers entity-level validation error)
    invalid_profile = (
        '{"site_name":"shopping",'
        '"data_model":[{"entity":"products","fields":[{"name":"title"}]}],'
        '"injection_surface":[{"id":"surface-1","source_field":"posts.body"}],'
        '"verification_capabilities":[]}'
    )
    # Second call: corrected -- entity "products" matches source_field
    valid_profile = (
        '{"site_name":"shopping",'
        '"data_model":[{"entity":"products","fields":[{"name":"title"},{"name":"description"}]}],'
        '"injection_surface":[{"id":"surface-1","source_field":"products.description"}],'
        '"verification_capabilities":[]}'
    )

    mock_sandbox = AsyncMock(
        side_effect=[
            {
                "/workspace/output/BENCHMARK_PROFILE.json": invalid_profile,
                "/workspace/output/BENCHMARK_PROFILE.md": "# invalid",
                "_summary": None,
            },
            {
                "/workspace/output/BENCHMARK_PROFILE.json": valid_profile,
                "/workspace/output/BENCHMARK_PROFILE.md": "# valid",
                "_summary": None,
            },
        ]
    )
    monkeypatch.setattr(phase_0_recon, "run_claude_in_sandbox", mock_sandbox)

    result = await phase_0_recon.run_phase_0c(
        manifest={"evaluation": {"eval_types": []}},
        sandbox_map={"shopping": [str(source_file)]},
        benchmark_root=benchmark_root,
        output_dir=tmp_path / "out",
    )

    assert "shopping" in result
    assert mock_sandbox.call_count == 2  # initial + 1 retry
    assert (tmp_path / "out" / "BENCHMARK_PROFILE_shopping.json").exists()


@pytest.mark.asyncio
async def test_correction_loop_hard_fails_after_max_retries(monkeypatch, tmp_path):
    """Sandbox always returns invalid profile -- should fail after 3 attempts."""
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    source_file = benchmark_root / "shopping.txt"
    source_file.write_text("demo")

    # Always invalid: entity "posts" not in data_model
    invalid_profile = (
        '{"site_name":"shopping",'
        '"data_model":[{"entity":"products","fields":[{"name":"title"}]}],'
        '"injection_surface":[{"id":"surface-1","source_field":"posts.body"}],'
        '"verification_capabilities":[]}'
    )

    mock_sandbox = AsyncMock(
        return_value={
            "/workspace/output/BENCHMARK_PROFILE.json": invalid_profile,
            "/workspace/output/BENCHMARK_PROFILE.md": "# invalid",
            "_summary": None,
        }
    )
    monkeypatch.setattr(phase_0_recon, "run_claude_in_sandbox", mock_sandbox)

    with pytest.raises(RuntimeError, match="failed validation"):
        await phase_0_recon.run_phase_0c(
            manifest={"evaluation": {"eval_types": []}},
            sandbox_map={"shopping": [str(source_file)]},
            benchmark_root=benchmark_root,
            output_dir=tmp_path / "out",
        )

    # initial + PROFILE_FIX_MAX_ITERATIONS retries = 3 total
    assert mock_sandbox.call_count == 3
