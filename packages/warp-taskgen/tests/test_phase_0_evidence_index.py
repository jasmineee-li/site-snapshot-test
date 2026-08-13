from __future__ import annotations

import json
from pathlib import Path

from warp_taskgen.phases.phase_0_evidence_index import build_phase_0c_evidence_indexes


def test_phase_0c_evidence_indexes_record_files_routes_and_task_candidates(tmp_path: Path):
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    app = benchmark_root / "app.py"
    app.write_text('ROUTE = "/issues/{id}/notes"\n', encoding="utf-8")
    tasks = benchmark_root / "tasks.json"
    tasks.write_text(
        json.dumps(
            [
                {
                    "task_id": "task-1",
                    "intent": "Find the issue",
                    "start_url": "/issues/1",
                    "eval_type": "NetworkEventEvaluator",
                }
            ]
        ),
        encoding="utf-8",
    )

    routed = build_phase_0c_evidence_indexes(
        file_list=[str(app), str(tasks)],
        benchmark_root=benchmark_root,
        manifest={
            "sites": [{"name": "gitlab", "source_path": "."}],
            "evaluation": {
                "eval_types": ["NetworkEventEvaluator"],
                "task_definition_paths": ["tasks.json"],
            },
        },
        site_name="gitlab",
        output_dir=tmp_path / "indexes",
    )

    assert sorted(routed) == [
        "/workspace/inputs/FILES_INDEX.json",
        "/workspace/inputs/MANIFEST_SLICE.json",
        "/workspace/inputs/ROUTES_INDEX.json",
        "/workspace/inputs/TASKS_INDEX.json",
    ]
    routes_path = Path(routed["/workspace/inputs/ROUTES_INDEX.json"])
    routes = json.loads(routes_path.read_text(encoding="utf-8"))
    assert {"file": "app.py", "line": 1, "literal": "/issues/{id}/notes"} in routes[
        "route_candidates"
    ]
    task_index = json.loads(
        Path(routed["/workspace/inputs/TASKS_INDEX.json"]).read_text(encoding="utf-8")
    )
    assert any(
        record.get("fields", {}).get("task_id") == "task-1"
        for record in task_index["task_candidates"]
    )


def test_phase_0c_evidence_index_skips_large_json_and_caps_jsonl_tasks(tmp_path: Path):
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    large_json = benchmark_root / "large_tasks.json"
    large_json.write_text(
        json.dumps({"task_id": "large", "padding": "x" * (2 * 1024 * 1024)}),
        encoding="utf-8",
    )
    jsonl = benchmark_root / "tasks.jsonl"
    jsonl.write_text(
        "\n".join(
            json.dumps({"task_id": f"task-{index}", "intent": "check"}) for index in range(1200)
        ),
        encoding="utf-8",
    )

    routed = build_phase_0c_evidence_indexes(
        file_list=[str(large_json), str(jsonl)],
        benchmark_root=benchmark_root,
        manifest={"evaluation": {"eval_types": ["NetworkEventEvaluator"]}},
        site_name="gitlab",
        output_dir=tmp_path / "indexes",
    )

    task_index = json.loads(
        Path(routed["/workspace/inputs/TASKS_INDEX.json"]).read_text(encoding="utf-8")
    )
    candidates = task_index["task_candidates"]
    assert len(candidates) == 1000
    assert any(
        candidate.get("skipped") == "json_file_too_large_for_task_index" for candidate in candidates
    )


def test_phase_0c_evidence_index_bounds_routes_and_sparse_jsonl(tmp_path: Path):
    benchmark_root = tmp_path / "benchmark"
    benchmark_root.mkdir()
    large_routes = benchmark_root / "routes.py"
    large_routes.write_text(
        'ROUTE = "/issues/1"\n' + ("#" * (2 * 1024 * 1024)),
        encoding="utf-8",
    )
    large_jsonl = benchmark_root / "sparse.jsonl"
    large_jsonl.write_text(
        "\n".join(json.dumps({"noise": "x" * 5000}) for _ in range(600)),
        encoding="utf-8",
    )
    long_line_jsonl = benchmark_root / "long_line.jsonl"
    long_line_jsonl.write_text(
        json.dumps({"task_id": "too-large", "intent": "x" * (70 * 1024)}),
        encoding="utf-8",
    )

    routed = build_phase_0c_evidence_indexes(
        file_list=[str(large_routes), str(large_jsonl), str(long_line_jsonl)],
        benchmark_root=benchmark_root,
        manifest={"evaluation": {"eval_types": ["NetworkEventEvaluator"]}},
        site_name="gitlab",
        output_dir=tmp_path / "indexes",
    )

    route_index = json.loads(
        Path(routed["/workspace/inputs/ROUTES_INDEX.json"]).read_text(encoding="utf-8")
    )
    assert any(
        candidate.get("skipped") == "text_file_too_large_for_route_index"
        for candidate in route_index["route_candidates"]
    )
    task_index = json.loads(
        Path(routed["/workspace/inputs/TASKS_INDEX.json"]).read_text(encoding="utf-8")
    )
    candidates = task_index["task_candidates"]
    assert any(
        candidate.get("skipped") == "jsonl_file_too_large_for_task_index"
        for candidate in candidates
    )
    assert any(
        candidate.get("skipped") == "jsonl_line_too_large_for_task_index"
        for candidate in candidates
    )
