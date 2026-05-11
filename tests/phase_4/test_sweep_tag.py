from __future__ import annotations

import subprocess

import pytest

from worldsim.phase_4 import sweep_tag


def test_sweep_tag_skips_when_disabled_env(monkeypatch, caplog) -> None:
    calls: list[list[str]] = []
    monkeypatch.setenv("WORLDSIM_DISABLE_SWEEP_TAG", "1")
    monkeypatch.setattr(sweep_tag, "_run_aws", lambda args: calls.append(args) or True)

    with sweep_tag.sweep_in_progress():
        pass

    assert calls == []


def test_sweep_tag_skips_when_disabled_flag(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(sweep_tag, "_run_aws", lambda args: calls.append(args) or True)

    with sweep_tag.sweep_in_progress(disabled=True):
        pass

    assert calls == []


def test_sweep_tag_skips_when_off_ec2(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.delenv("WORLDSIM_DISABLE_SWEEP_TAG", raising=False)
    monkeypatch.delenv("WORLDSIM_SWEEP_TAG_INSTANCE_ID", raising=False)
    monkeypatch.delenv("WORLDSIM_SWEEP_TAG_REGION", raising=False)
    monkeypatch.setattr(sweep_tag, "_imdsv2_token", lambda: None)
    monkeypatch.setattr(sweep_tag, "_run_aws", lambda args: calls.append(args) or True)

    with sweep_tag.sweep_in_progress():
        pass

    assert calls == []


def test_sweep_tag_sets_and_clears_via_env_override(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.delenv("WORLDSIM_DISABLE_SWEEP_TAG", raising=False)
    monkeypatch.setenv("WORLDSIM_SWEEP_TAG_INSTANCE_ID", "i-0bf197c9d4e41d500")
    monkeypatch.setenv("WORLDSIM_SWEEP_TAG_REGION", "us-east-2")
    monkeypatch.setattr(sweep_tag, "_run_aws", lambda args: calls.append(args) or True)

    with sweep_tag.sweep_in_progress():
        pass

    assert len(calls) == 2
    assert calls[0][:3] == ["aws", "ec2", "create-tags"]
    assert "i-0bf197c9d4e41d500" in calls[0]
    assert "us-east-2" in calls[0]
    assert f"Key={sweep_tag.SWEEP_TAG_KEY},Value={sweep_tag.SWEEP_TAG_VALUE}" in calls[0]
    assert calls[1][:3] == ["aws", "ec2", "delete-tags"]
    assert "i-0bf197c9d4e41d500" in calls[1]
    assert f"Key={sweep_tag.SWEEP_TAG_KEY}" in calls[1]


def test_sweep_tag_clears_even_on_exception(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setenv("WORLDSIM_SWEEP_TAG_INSTANCE_ID", "i-0bf197c9d4e41d500")
    monkeypatch.setenv("WORLDSIM_SWEEP_TAG_REGION", "us-east-2")
    monkeypatch.setattr(sweep_tag, "_run_aws", lambda args: calls.append(args) or True)

    with pytest.raises(RuntimeError, match="boom"):
        with sweep_tag.sweep_in_progress():
            raise RuntimeError("boom")

    # set called once, then delete called even after the exception.
    assert len(calls) == 2
    assert calls[0][2] == "create-tags"
    assert calls[1][2] == "delete-tags"


def test_sweep_tag_does_not_clear_when_set_failed(monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run_aws(args: list[str]) -> bool:
        calls.append(args)
        # First call (create-tags) fails; subsequent calls should not happen.
        return False

    monkeypatch.setenv("WORLDSIM_SWEEP_TAG_INSTANCE_ID", "i-0bf197c9d4e41d500")
    monkeypatch.setenv("WORLDSIM_SWEEP_TAG_REGION", "us-east-2")
    monkeypatch.setattr(sweep_tag, "_run_aws", fake_run_aws)

    with sweep_tag.sweep_in_progress():
        pass

    assert len(calls) == 1
    assert calls[0][2] == "create-tags"


def test_run_aws_handles_missing_cli(monkeypatch) -> None:
    def raise_fnfe(*_a, **_k):
        raise FileNotFoundError("aws")

    monkeypatch.setattr(subprocess, "run", raise_fnfe)
    assert sweep_tag._run_aws(["aws", "ec2", "create-tags"]) is False


def test_run_aws_handles_called_process_error(monkeypatch) -> None:
    def raise_cpe(*_a, **_k):
        raise subprocess.CalledProcessError(255, ["aws"], stderr=b"AccessDenied")

    monkeypatch.setattr(subprocess, "run", raise_cpe)
    assert sweep_tag._run_aws(["aws", "ec2", "create-tags"]) is False


def test_run_aws_handles_timeout(monkeypatch) -> None:
    def raise_timeout(*_a, **_k):
        raise subprocess.TimeoutExpired(["aws"], 30)

    monkeypatch.setattr(subprocess, "run", raise_timeout)
    assert sweep_tag._run_aws(["aws", "ec2", "create-tags"]) is False
