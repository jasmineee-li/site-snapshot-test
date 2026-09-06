"""Novel-task cache reads and the site and shared-input fingerprints."""

from __future__ import annotations

import json

from warp_taskgen.phase_1 import novel_task_cache
from warp_taskgen.phases import phase_1_generate_new_tasks

from ._fixtures import (  # noqa: F401
    _agent_context,
    _manifest,
    _novel_task_list,
    _profile,
    _site_cache_metadata,
    _stub_generate_new_tasks_sandbox_preflight,
)


def test_load_cached_novel_tasks_rejects_missing_embedded_agent_context(tmp_path):
    output_dir = tmp_path / "phase_1"
    output_dir.mkdir()
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    profile_path = tmp_path / "phase_0c" / "BENCHMARK_PROFILE_shopping.json"
    profile_path.parent.mkdir()
    profile_path.write_text(json.dumps(_profile(uncovered=["surface-1"])))
    agent_context = _agent_context()
    (profile_path.parent / "AGENT_CONTEXT_shopping.json").write_text(json.dumps(agent_context))

    site = phase_1_generate_new_tasks.EligibleSiteProfile(
        site_name="shopping",
        profile_path=profile_path,
        profile=_profile(uncovered=["surface-1"]),
    )
    cached_tasks = _novel_task_list()
    intermediate_path = output_dir / "novel_tasks_shopping.json"
    intermediate_path.write_text(json.dumps(cached_tasks))
    metadata = _site_cache_metadata(
        benchmark_root=benchmark_root,
        manifest=_manifest(benchmark_root),
        site=site,
    )
    (output_dir / "novel_tasks_shopping.json.metadata.json").write_text(json.dumps(metadata))

    cached = phase_1_generate_new_tasks.load_cached_novel_tasks(
        intermediate_path=intermediate_path,
        site_name="shopping",
        profile=site.profile,
        cache_fingerprint=metadata["fingerprint"],
        expected_agent_context=agent_context,
    )

    assert cached is None


def test_compute_site_cache_fingerprint_changes_when_agent_context_changes(tmp_path):
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    manifest = _manifest(benchmark_root)

    profile_path = tmp_path / "phase_0c" / "BENCHMARK_PROFILE_shopping.json"
    profile_path.parent.mkdir()
    profile_path.write_text(json.dumps(_profile(uncovered=["surface-1"])))
    site = phase_1_generate_new_tasks.EligibleSiteProfile(
        site_name="shopping",
        profile_path=profile_path,
        profile=_profile(uncovered=["surface-1"]),
    )

    shared_inputs_fingerprint = (
        phase_1_generate_new_tasks.compute_generate_new_tasks_shared_inputs_fingerprint(
            benchmark_root=benchmark_root,
            manifest=manifest,
        )
    )
    agent_context_path = profile_path.parent / "AGENT_CONTEXT_shopping.json"
    agent_context_path.write_text(
        json.dumps({"response_format": {"requires_structured_output": False}})
    )
    first = phase_1_generate_new_tasks.compute_site_cache_fingerprint(
        shared_inputs_fingerprint=shared_inputs_fingerprint,
        site=site,
    )

    agent_context_path.write_text(
        json.dumps({"response_format": {"requires_structured_output": True}})
    )
    second = phase_1_generate_new_tasks.compute_site_cache_fingerprint(
        shared_inputs_fingerprint=shared_inputs_fingerprint,
        site=site,
    )

    assert first != second


def test_compute_site_cache_fingerprint_changes_when_task_count_changes(tmp_path):
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    manifest = _manifest(benchmark_root)
    site = phase_1_generate_new_tasks.EligibleSiteProfile(
        site_name="shopping",
        profile_path=tmp_path / "BENCHMARK_PROFILE_shopping.json",
        profile=_profile(uncovered=["surface-1"]),
    )

    shared_inputs_fingerprint = (
        phase_1_generate_new_tasks.compute_generate_new_tasks_shared_inputs_fingerprint(
            benchmark_root=benchmark_root,
            manifest=manifest,
        )
    )
    first = phase_1_generate_new_tasks.compute_site_cache_fingerprint(
        shared_inputs_fingerprint=shared_inputs_fingerprint,
        site=site,
        novel_tasks_per_site=30,
    )
    second = phase_1_generate_new_tasks.compute_site_cache_fingerprint(
        shared_inputs_fingerprint=shared_inputs_fingerprint,
        site=site,
        novel_tasks_per_site=50,
    )

    assert first != second


def test_compute_generate_new_tasks_shared_inputs_fingerprint_changes_when_sandbox_model_changes(
    tmp_path,
):
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    manifest = _manifest(benchmark_root)

    first = phase_1_generate_new_tasks.compute_generate_new_tasks_shared_inputs_fingerprint(
        benchmark_root=benchmark_root,
        manifest=manifest,
        sandbox_model="claude-opus-4-6",
    )
    second = phase_1_generate_new_tasks.compute_generate_new_tasks_shared_inputs_fingerprint(
        benchmark_root=benchmark_root,
        manifest=manifest,
        sandbox_model="claude-sonnet-4-6",
    )

    assert first != second


def test_compute_generate_new_tasks_shared_inputs_fingerprint_changes_when_prompt_changes(
    monkeypatch, tmp_path
):
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    manifest = _manifest(benchmark_root)

    first = phase_1_generate_new_tasks.compute_generate_new_tasks_shared_inputs_fingerprint(
        benchmark_root=benchmark_root,
        manifest=manifest,
    )
    original_load_prompt = novel_task_cache.load_prompt

    def fake_load_prompt(*args, **kwargs):
        return original_load_prompt(*args, **kwargs) + "\nchanged"

    monkeypatch.setattr(novel_task_cache, "load_prompt", fake_load_prompt)
    second = phase_1_generate_new_tasks.compute_generate_new_tasks_shared_inputs_fingerprint(
        benchmark_root=benchmark_root,
        manifest=manifest,
    )

    assert first != second


def test_compute_generate_new_tasks_shared_inputs_fingerprint_changes_when_task_card_plan_changes(
    tmp_path,
):
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    manifest = _manifest(benchmark_root)

    first = phase_1_generate_new_tasks.compute_generate_new_tasks_shared_inputs_fingerprint(
        benchmark_root=benchmark_root,
        manifest=manifest,
        task_card_plan={"schema_version": 1, "task_cards": [{"id": "a", "site": "gitlab"}]},
    )
    second = phase_1_generate_new_tasks.compute_generate_new_tasks_shared_inputs_fingerprint(
        benchmark_root=benchmark_root,
        manifest=manifest,
        task_card_plan={"schema_version": 1, "task_cards": [{"id": "b", "site": "gitlab"}]},
    )

    assert first != second
