"""Shared test fixtures for benchmark tests."""

import os

import pytest


@pytest.fixture(autouse=True)
def _skip_pipeline_config_validation(monkeypatch):
    """Skip fail-fast config validation in tests (no API keys in CI)."""
    monkeypatch.setenv("AGENTLAB_REDTEAM_SKIP_CONFIG_VALIDATION", "1")
