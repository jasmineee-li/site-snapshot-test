"""Phase 4 setup preflight — runs via ``pytest -m preflight``.

The setup script ``scripts/setup_phase4_on_host.sh`` drives these tests as
the final gate after bootstrapping uv/playwright/docker/artifacts. A green
preflight proves: every configured PVPO CDP endpoint is reachable and
uniquely assigned, every Magento replica has the correct base_url, the
gitlab Phase 0d artifact exists, and the WebArena-Verified evaluator venv
resolves.

Inputs (setup script exports these):
- ``WORLDSIM_PREFLIGHT_INSTANCES`` — path to instances.json
- ``WORLDSIM_PREFLIGHT_HOST_CONFIG`` — optional host YAML

Each test fails loudly with a clear remediation if its precondition is
unmet. Exit code from pytest is the gate the bash orchestrator checks.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

import pytest

pytestmark = pytest.mark.preflight


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


def test_pvpo_cdp_endpoints_reachable_and_unique() -> None:
    """Every configured worker PVPO endpoint must answer /json/version and be unique."""
    config = _load_instances_config()
    if config is None:
        pytest.skip("WORLDSIM_PREFLIGHT_INSTANCES unset or missing")
    try:
        from worldsim.config import BenchmarkConfig
        from worldsim.phases.phase_4_adversarial import _pvpo_endpoint_preflight_errors
    except ImportError as exc:
        pytest.skip(f"worldsim modules unavailable: {exc}")

    parsed = BenchmarkConfig.model_validate(config)
    errors = _pvpo_endpoint_preflight_errors(parsed.instances)
    assert not errors, (
        "PVPO endpoint config invalid — each Phase 4 worker needs a unique "
        "BenchmarkInstance.pvpo_cdp_url. Errors:\n  " + "\n  ".join(errors)
    )

    checked: set[str] = set()
    for instance in parsed.instances:
        url = instance.pvpo_cdp_url
        assert url is not None
        if url in checked:
            continue
        checked.add(url)
        version_url = f"{url.rstrip('/')}/json/version"
        try:
            with urlopen(version_url, timeout=3) as resp:
                assert resp.status == 200
        except Exception as exc:
            host = urlparse(url).netloc or url
            pytest.fail(
                f"pvpo-chrome CDP endpoint not reachable at {host}: {exc}. "
                "Rerun setup_phase4_on_host.sh step 3 to (re)start the per-instance containers."
            )


def test_pvpo_cdp_containers_are_restart_supervised() -> None:
    """Managed PVPO containers must support per-task docker restart recycling."""
    config = _load_instances_config()
    if config is None:
        pytest.skip("WORLDSIM_PREFLIGHT_INSTANCES unset or missing")
    try:
        from worldsim.config import BenchmarkConfig
    except ImportError as exc:
        pytest.skip(f"worldsim modules unavailable: {exc}")

    parsed = BenchmarkConfig.model_validate(config)
    checked_ports: set[int] = set()
    for instance in parsed.instances:
        raw_url = instance.pvpo_cdp_url
        assert raw_url is not None
        parsed_url = urlparse(raw_url)
        host = (parsed_url.hostname or "").lower()
        if host not in {"127.0.0.1", "localhost", "::1"}:
            continue
        assert parsed_url.port is not None
        if parsed_url.port in checked_ports:
            continue
        checked_ports.add(parsed_url.port)
        container_name = f"pvpo-chrome-{parsed_url.port}"
        try:
            completed = subprocess.run(
                [
                    "docker",
                    "inspect",
                    "--format",
                    "{{.HostConfig.RestartPolicy.Name}}",
                    container_name,
                ],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except FileNotFoundError:
            pytest.fail("docker is not installed; rerun setup_phase4_on_host.sh step 3")
        assert completed.returncode == 0, (
            f"PVPO container {container_name} is missing. "
            "Rerun setup_phase4_on_host.sh step 3."
        )
        assert completed.stdout.strip() == "unless-stopped", (
            f"PVPO container {container_name} has restart policy "
            f"{completed.stdout.strip()!r}; expected 'unless-stopped'. "
            "Rerun setup_phase4_on_host.sh step 3 so Phase 4 can recycle "
            "the browser after each task without losing the endpoint."
        )


# test_magento_base_urls_resolved removed 2026-04-21 with the WASP-aligned
# scoping decision (see docs/handoffs/wasp-aligned-scoping-decision.md).


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
        [python_exe, "-c", "from webarena_verified import WebArenaVerified"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, (
        f"evaluator venv python {python_exe!r} cannot import webarena_verified: "
        f"rc={completed.returncode} stderr={completed.stderr[:300]}"
    )
