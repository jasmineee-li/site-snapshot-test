"""Smoke tests for scripts/validate_phase_2_gates.py Gate 3 (Phase 4 exposure)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_gates(artifact: Path) -> int:
    repo = Path(__file__).resolve().parents[1]
    return subprocess.call(
        [
            sys.executable,
            str(repo / "scripts" / "validate_phase_2_gates.py"),
            "--artifact",
            str(artifact),
        ],
        cwd=repo,
    )


def _verified_task(*, bad_verification: bool) -> dict:
    contract_id = "gitlab:benign-1:abc"
    base_url = "http://gitlab.test/issues"
    ver = {"url": base_url, "mode": "direct_detail", "witness_policy": "dom_text"}
    if bad_verification:
        ver = {
            "url": "http://gitlab.test/other",
            "mode": "direct_detail",
            "witness_policy": "dom_text",
        }
    return {
        "id": "adv-1",
        "site": "gitlab",
        "feasibility": {
            "status": "verified",
            "reachability": "reachable_direct",
            "exposure": {
                "contract_id": contract_id,
                "reachable": True,
                "verification": ver,
            },
        },
        "exposure_contract": {
            "contract_id": contract_id,
            "eligibility": {"status": "eligible"},
            "verification": {
                "url": base_url,
                "mode": "direct_detail",
                "witness_policy": "dom_text",
            },
            "phase4_exposure": {
                "schema_version": 1,
                "admissible": True,
            },
        },
    }


def test_gate_3_passes_when_exposure_aligns(tmp_path: Path) -> None:
    p = tmp_path / "adversarial_tasks.json"
    p.write_text(json.dumps([_verified_task(bad_verification=False)]))
    assert _run_gates(p) == 0


def test_gate_3_fails_on_verification_mismatch(tmp_path: Path) -> None:
    p = tmp_path / "adversarial_tasks.json"
    p.write_text(json.dumps([_verified_task(bad_verification=True)]))
    assert _run_gates(p) == 3
