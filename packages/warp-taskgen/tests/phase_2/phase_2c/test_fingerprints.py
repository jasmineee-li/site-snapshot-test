"""Phase 2c fingerprints: task content hash, sync stamp, idempotency decision."""

from __future__ import annotations

import json
from datetime import UTC
from pathlib import Path

from warp_taskgen.phase_2.phase_2c import fingerprints

from ._fixtures import (
    _bypass_preflight,  # noqa: F401
    _stable_git_fingerprint,  # noqa: F401
    _task,
)


def test_task_content_hash_includes_exposure_contract_verification():
    editor_calls = _task()["adversarial_data_seed"]["editor_calls"]
    contract = {
        "contract_id": "gitlab:task:test",
        "site": "gitlab",
        "kind": "gitlab_issue",
        "mode": "direct_detail",
        "editor_method": "create_issue_note",
        "target_surface_id": "note_on_issue",
        "payload_arg": "body",
        "editor_args_template": {"body": "{{PAYLOAD_TEXT}}"},
        "verification": {
            "url": "https://gitlab.local/acme/demo/-/issues/7",
            "mode": "direct_detail",
            "witness_policy": "signature_plus_second_witness",
        },
        "eligibility": {"status": "eligible"},
    }
    changed = {
        **contract,
        "verification": {
            **contract["verification"],
            "url": "https://gitlab.local/search?search=theme&scope=issues",
        },
    }

    assert fingerprints._task_content_hash(editor_calls, exposure_contract=contract) != (
        fingerprints._task_content_hash(editor_calls, exposure_contract=changed)
    )


def test_sync_stamp_commit_uses_deployed_local_sha(tmp_path):
    (tmp_path / ".worldsim_sync_stamp.json").write_text(
        json.dumps(
            {
                "local_git": {
                    "sha": "87de6788d9a44a8aba2c5269e39d12cfda685865",
                    "branch": "feat/worldsim-v5",
                },
                "remote_git": {
                    "sha": "07919d7ea67a0000000000000000000000000000",
                    "branch": "HEAD",
                },
            }
        )
    )

    assert fingerprints._sync_stamp_commit(tmp_path) == "87de6788d9a4"


def test_sync_stamp_commit_ignores_missing_or_invalid_stamp(tmp_path):
    assert fingerprints._sync_stamp_commit(tmp_path) is None

    (tmp_path / ".worldsim_sync_stamp.json").write_text("{not json")
    assert fingerprints._sync_stamp_commit(tmp_path) is None


def test_git_head_short_preserves_sync_stamp_lookup(monkeypatch, tmp_path):
    observed: list[Path] = []

    def fake_sync_stamp_commit(repo_root: Path) -> str | None:
        observed.append(repo_root)
        return "stamp12345678"

    monkeypatch.delenv("WORLDSIM_EDITOR_COMMIT_OVERRIDE", raising=False)
    monkeypatch.setattr(fingerprints, "_sync_stamp_commit", fake_sync_stamp_commit)

    assert fingerprints._git_head_short() == "stamp12345678"
    # ``parents[3]`` is the package root from tests/phase_2/phase_2c/; it was
    # ``parents[1]`` while this test lived in tests/test_phase_2_feasibility.py.
    assert observed == [Path(__file__).resolve().parents[3] / "warp_taskgen"]


# ---------------------------------------------------------------------------
# Idempotency decision unit tests
# ---------------------------------------------------------------------------


def test_idempotency_decision_truth_table():
    fp = {
        "host_config": "a",
        "instances_digest": "aa11bb22cc33",
        "editor_commit": "b",
        "dataset_commit": "c",
        "task_content_hash": "d",
    }
    drift = {**fp, "task_content_hash": "other"}

    def _decide(existing, *, ttl=None, force=False):
        return fingerprints._idempotency_decision(
            existing, current_fingerprint=fp, ttl_hours=ttl, force_reverify=force
        )

    # missing → verify
    assert _decide(None) == ("verify", None)
    # verified + match → skip (reason=fingerprint_match)
    assert _decide({"status": "verified", "host_fingerprint": fp}) == (
        "skip",
        "fingerprint_match",
    )
    # verified + drift → re-verify
    assert _decide({"status": "verified", "host_fingerprint": drift}) == ("verify", None)
    # verified + drift + TTL covers it → skip (reason=ttl_hours)
    from datetime import datetime

    recent = datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert _decide(
        {"status": "verified", "host_fingerprint": drift, "verified_at": recent},
        ttl=24.0,
    ) == ("skip", "ttl_hours")
    # infeasible → always re-verify
    assert _decide({"status": "infeasible", "host_fingerprint": fp}) == ("verify", None)
    # unverified (skip flag) → verify
    assert _decide({"status": "unverified"}) == ("verify", None)
    # force overrides skip
    assert _decide({"status": "verified", "host_fingerprint": fp}, force=True) == (
        "verify",
        None,
    )
