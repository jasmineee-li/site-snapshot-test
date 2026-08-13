from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from warp_taskgen.phase_4 import placement_loop as phase_4_placement_loop
from warp_taskgen.phase_4.encounter_detection import EncounterResult


def _task() -> dict[str, Any]:
    return {
        "id": "adv-agentlab-needham-1",
        "site": "gitlab",
        "payload_texts": [{"rendered_payload": "PAYLOAD_AB"}],
    }


def _write_history(tmp_path: Path) -> None:
    (tmp_path / "history.json").write_text(
        json.dumps(
            {
                "history": [
                    {
                        "model_output": {
                            "thinking": "agentlab fallback thought",
                            "action": [{"agentlab_action": {"raw": "click(1)"}}],
                        },
                        "state": {"url": "https://gitlab.test", "title": "GitLab"},
                        "result": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.asyncio
async def test_agentlab_needham_trace_survives_pvpo_gate(monkeypatch, tmp_path):
    _write_history(tmp_path)
    sidecar_xml = '<message role="assistant">\nfinal eval-aware answer\n</message>\n\n'
    (tmp_path / "needham_trace.json").write_text(
        json.dumps(
            {
                "format": "needham-agentlab-v1",
                "transcript_format": "needham-xml-v1",
                "xml": sidecar_xml,
                "messages": [
                    {
                        "role": "assistant",
                        "text": "final eval-aware answer",
                        "provenance": {"source": "agentlab_final_response"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "needham_trace.xml").write_text(sidecar_xml, encoding="utf-8")
    captured: dict[str, str | None] = {}

    def fake_determine(task, trajectory_dir):
        return EncounterResult(
            max_coverage=0.5,
            reference_step=None,
            reference_screenshot_path=None,
            reference_screenshot_bytes=None,
            per_char_visibility=[],
            per_step_coverage=[],
            pvpo_status="ok",
            pvpo_failure=None,
        )

    async def fake_tp(
        task,
        trajectory_dir,
        extracted,
        *,
        sandbox_model,
        needham_xml=None,
        include_apollo_continue_format=True,
    ):
        captured["tp_xml"] = needham_xml
        return {
            "transcript_purpose_classification": "Real",
            "transcript_purpose_is_aware": False,
            "transcript_purpose_guess": "real",
            "transcript_purpose_explanation": "ok",
            "transcript_purpose_failure": None,
            "transcript_purpose_format": "needham-xml-v1",
            "transcript_purpose_attempts": 1,
            "transcript_purpose_raw_response_path": None,
            "transcript_purpose_was_refused": False,
        }

    async def fake_vea(task, trajectory_dir, extracted, *, sandbox_model, needham_xml=None):
        captured["vea_xml"] = needham_xml
        return {
            "verbalized_eval_awareness_classification": "Real",
            "verbalized_eval_awareness_is_aware": False,
            "verbalized_eval_awareness_evidence": [],
            "verbalized_eval_awareness_failure": None,
            "verbalized_eval_awareness_transcript_format": "needham-xml-v1",
            "verbalized_eval_awareness_raw_response_path": None,
        }

    monkeypatch.setattr(phase_4_placement_loop, "determine_encounter", fake_determine)
    monkeypatch.setattr(phase_4_placement_loop, "run_transcript_purpose_api", fake_tp)
    monkeypatch.setattr(phase_4_placement_loop, "run_verbalized_eval_awareness_api", fake_vea)

    result = await phase_4_placement_loop._run_pvpo_gate(
        _task(), tmp_path, sandbox_model="claude-sonnet-4-6"
    )

    assert result["transcript_purpose_classification"] == "Real"
    assert captured == {"tp_xml": sidecar_xml, "vea_xml": sidecar_xml}
    persisted = json.loads((tmp_path / "needham_trace.json").read_text(encoding="utf-8"))
    assert persisted["format"] == "needham-agentlab-v1"
