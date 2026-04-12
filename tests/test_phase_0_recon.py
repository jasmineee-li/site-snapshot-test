from __future__ import annotations

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
