from __future__ import annotations

import importlib
import os

import dotenv
import pytest


def test_explicit_remote_state_aliases_survive_dotenv_without_changing_other_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from warp_taskgen.cli import _impl as cli_impl

    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", "explicit-state")
    monkeypatch.setenv("WORLDSIM_STATE_DIR", "explicit-state")
    monkeypatch.setenv("WARP_TASKGEN_REMOTE_STATE_DIR_EXPLICIT", "1")
    monkeypatch.setenv("WARP_TEST_DOTENV_SENTINEL", "shell-value")

    def fake_load_dotenv(*args, **kwargs):
        assert kwargs.get("override") is True
        os.environ["WARP_TASKGEN_STATE_DIR"] = "dotenv-canonical"
        os.environ["WORLDSIM_STATE_DIR"] = "dotenv-legacy"
        os.environ["WARP_TEST_DOTENV_SENTINEL"] = "dotenv-value"
        return True

    monkeypatch.setattr(dotenv, "load_dotenv", fake_load_dotenv)
    try:
        importlib.reload(cli_impl)
        assert os.environ["WARP_TASKGEN_STATE_DIR"] == "explicit-state"
        assert os.environ["WORLDSIM_STATE_DIR"] == "explicit-state"
        assert os.environ["WARP_TEST_DOTENV_SENTINEL"] == "dotenv-value"
    finally:
        monkeypatch.undo()
        importlib.reload(cli_impl)


def test_without_remote_marker_dotenv_aliases_keep_existing_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from warp_taskgen.cli import _impl as cli_impl

    monkeypatch.setenv("WARP_TASKGEN_STATE_DIR", "shell-canonical")
    monkeypatch.setenv("WORLDSIM_STATE_DIR", "shell-legacy")
    monkeypatch.delenv("WARP_TASKGEN_REMOTE_STATE_DIR_EXPLICIT", raising=False)

    def fake_load_dotenv(*args, **kwargs):
        assert kwargs.get("override") is True
        os.environ["WARP_TASKGEN_STATE_DIR"] = "dotenv-canonical"
        os.environ["WORLDSIM_STATE_DIR"] = "dotenv-legacy"
        return True

    monkeypatch.setattr(dotenv, "load_dotenv", fake_load_dotenv)
    try:
        importlib.reload(cli_impl)
        assert os.environ["WARP_TASKGEN_STATE_DIR"] == "dotenv-canonical"
        assert os.environ["WORLDSIM_STATE_DIR"] == "dotenv-legacy"
    finally:
        monkeypatch.undo()
        importlib.reload(cli_impl)
