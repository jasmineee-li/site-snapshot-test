from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from agentlab.analyze.agent_xray import (
    _resolve_website_viewer_html_files,
    get_behavior_comparison_df,
    on_select_variant,
)


def test_agent_xray_tracks_namespaced_adversarial_variants() -> None:
    result_df = pd.DataFrame(
        [
            {
                "_behavior_id": "demo",
                "_test_mode": "benign",
                "_row_index": 11,
                "judge.overall_success": True,
                "env.task_name": "redteam.demo.benign",
            },
            {
                "_behavior_id": "demo",
                "_test_mode": "adversarial",
                "_row_index": 30,
                "judge.overall_success": np.nan,
                "env.variant_name": "adversarial_demo_v10",
                "env.task_name": "redteam.demo.adversarial_demo_v10",
            },
            {
                "_behavior_id": "demo",
                "_test_mode": "adversarial",
                "_row_index": 20,
                "judge.overall_success": True,
                "env.variant_name": "adversarial_demo_v2",
                "env.task_name": "redteam.demo.adversarial_demo_v2",
            },
            {
                "_behavior_id": "demo",
                "_test_mode": "adversarial",
                "_row_index": 21,
                "judge.overall_success": np.bool_(False),
                "env.variant_name": "adversarial_demo_v1",
                "env.task_name": "redteam.demo.adversarial_demo_v1",
            },
        ]
    )

    comparison_df = get_behavior_comparison_df(result_df)

    assert comparison_df.loc[0, "adv_0_name"] == "adversarial_demo_v1"
    assert comparison_df.loc[0, "adv_0"] == "✗"
    assert comparison_df.loc[0, "adv_1_name"] == "adversarial_demo_v2"
    assert comparison_df.loc[0, "adv_1"] == "✓"
    assert comparison_df.loc[0, "adv_2_name"] == "adversarial_demo_v10"
    assert comparison_df.loc[0, "adv_2"] == "-"

    episode_id = on_select_variant(
        "adversarial_demo_v10",
        {
            "behavior": "demo",
            "benign_idx": 11,
            "adv_0_name": "adversarial_demo_v1",
            "adv_0_idx": 21,
            "adv_1_name": "adversarial_demo_v2",
            "adv_1_idx": 20,
            "adv_2_name": "adversarial_demo_v10",
            "adv_2_idx": 30,
        },
        "agent-1",
    )

    assert episode_id.task_name == "demo.adversarial_demo_v10"
    assert episode_id.row_index == 30


def test_resolve_website_viewer_html_files_prefers_archived_runtime(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    runtime_dir = exp_dir / "app_runtime"
    variation_dir = exp_dir / "variation_0"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    variation_dir.mkdir(parents=True, exist_ok=True)
    (runtime_dir / "index.html").write_text("<html>runtime</html>\n", encoding="utf-8")
    (variation_dir / "index.html").write_text("<html>legacy</html>\n", encoding="utf-8")
    exp_result = SimpleNamespace(exp_dir=exp_dir)

    html_files, message = _resolve_website_viewer_html_files(exp_result)

    assert message == ""
    assert [path.name for path in html_files] == ["index.html"]
    assert html_files[0].parent == runtime_dir


def test_resolve_website_viewer_html_files_rejects_symlink_escape(tmp_path: Path) -> None:
    exp_dir = tmp_path / "exp"
    runtime_dir = exp_dir / "app_runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    outside_html = tmp_path / "outside.html"
    outside_html.write_text("<html>outside</html>\n", encoding="utf-8")
    (runtime_dir / "index.html").symlink_to(outside_html)
    exp_result = SimpleNamespace(exp_dir=exp_dir)

    html_files, message = _resolve_website_viewer_html_files(exp_result)

    assert html_files == []
    assert "No archived HTML files found" in message
