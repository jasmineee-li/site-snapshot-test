"""Phase 4 setup preflight — runs via ``pytest -m preflight``.

The setup script ``scripts/setup_phase4_on_host.sh`` drives these tests as
the final gate after bootstrapping uv/playwright/docker/artifacts. A green
preflight proves: the PVPO CDP endpoint is reachable, every Magento
replica has the correct base_url, the gitlab Phase 0d artifact exists,
and the WebArena-Verified evaluator venv resolves.

Inputs (setup script exports these):
- ``WORLDSIM_PREFLIGHT_INSTANCES`` — path to instances.json
- ``WORLDSIM_PREFLIGHT_HOST_CONFIG`` — optional host YAML

Each test fails loudly with a clear remediation if its precondition is
unmet. Exit code from pytest is the gate the bash orchestrator checks.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
from pathlib import Path
from urllib.request import urlopen

import pytest

pytestmark = pytest.mark.preflight


PVPO_CDP_URL_DEFAULT = "http://127.0.0.1:9222"


def _pvpo_cdp_url() -> str:
    return os.environ.get("WORLDSIM_PVPO_CDP_URL", PVPO_CDP_URL_DEFAULT)


def _instances_path() -> Path | None:
    raw = os.environ.get("WORLDSIM_PREFLIGHT_INSTANCES", "").strip()
    if not raw:
        return None
    return Path(raw)


def _load_instances_config() -> dict | None:
    path = _instances_path()
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def test_pvpo_cdp_endpoint_reachable() -> None:
    """GET /json/version on the chrome-headless-shell CDP port must return 200."""
    url = _pvpo_cdp_url()
    host = url.split("://", 1)[-1].split(":", 1)[0]
    port = int(url.rsplit(":", 1)[-1].strip("/"))
    try:
        with socket.create_connection((host, port), timeout=2):
            pass
    except OSError as exc:
        pytest.fail(
            f"pvpo-chrome CDP endpoint not reachable at {url}: {exc}. "
            f"Rerun setup_phase4_on_host.sh step 3 to (re)start the container."
        )
    try:
        with urlopen(f"{url}/json/version", timeout=3) as resp:
            assert resp.status == 200
    except Exception as exc:
        pytest.fail(f"pvpo-chrome /json/version did not return 200: {exc}")


def test_magento_base_urls_resolved() -> None:
    """Every shopping* instance must render the expected proxy-origin base_url."""
    config = _load_instances_config()
    if config is None:
        pytest.skip("WORLDSIM_PREFLIGHT_INSTANCES unset or missing")
    try:
        from worldsim.config import BenchmarkConfig
        from worldsim.phase_4.magento_health import check_magento_instances
    except ImportError as exc:
        pytest.skip(f"worldsim modules unavailable: {exc}")
    parsed = BenchmarkConfig.model_validate(config)
    errors = check_magento_instances(parsed)
    assert not errors, (
        "Magento base_url drift — run scripts/sync_magento_base_urls.py "
        "to repair. Errors:\n  " + "\n  ".join(str(e) for e in errors)
    )


def test_gitlab_storage_state_present() -> None:
    """Phase 0d gitlab storage_state must exist and contain cookies."""
    state_dir = Path(os.environ.get("WORLDSIM_STATE_DIR", "logs"))
    artifact = state_dir / "phase_0d" / "gitlab" / "storage_state.json"
    if not artifact.exists():
        pytest.fail(
            f"missing gitlab Phase 0d storage_state at {artifact}. "
            f"Rerun setup_phase4_on_host.sh step 5 (or scripts/login_gitlab_r5.py)."
        )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    cookies = payload.get("cookies")
    assert isinstance(cookies, list) and cookies, (
        f"gitlab storage_state at {artifact} has no cookies; re-mint via step 5"
    )


def test_eval_venv_reachable() -> None:
    """The worldsim-webarena-verified evaluator venv must resolve + import."""
    try:
        from worldsim.rewards import WEBARENA_EVAL_PYTHON_ENV, _default_eval_python
    except ImportError as exc:
        pytest.skip(f"worldsim.rewards unavailable: {exc}")
    python_exe = os.environ.get(WEBARENA_EVAL_PYTHON_ENV, "").strip() or _default_eval_python()
    if not python_exe:
        pytest.fail(
            "no evaluator Python resolved. Run `cd packages/worldsim-webarena-verified "
            "&& uv sync --locked` (setup_phase4_on_host.sh step 1)."
        )
    completed = subprocess.run(
        [python_exe, "-c", "import webarena_verified; print(webarena_verified.__version__)"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, (
        f"evaluator venv python {python_exe!r} cannot import webarena_verified: "
        f"rc={completed.returncode} stderr={completed.stderr[:300]}"
    )
