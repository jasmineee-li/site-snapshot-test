from __future__ import annotations

import json
from pathlib import Path

from worldsim.phases.phase_0_evidence_index import build_phase_0c_evidence_indexes


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
