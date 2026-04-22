from __future__ import annotations

import json
import os
import urllib.parse
import uuid
from pathlib import Path

import pytest

from worldsim.auth_tokens import clear_run_token_cache
from worldsim.config import BenchmarkConfig
from worldsim.http_proxy import install_proxy


def _replace_url_host(url: str, host: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or not parsed.netloc:
        return url
    netloc = host
    if parsed.port is not None:
        netloc = f"{host}:{parsed.port}"
    return urllib.parse.urlunparse(parsed._replace(netloc=netloc))


def _replace_db_host(db_connection: str, host: str) -> str:
    parsed = urllib.parse.urlparse(db_connection)
    if not parsed.scheme or not parsed.netloc:
        return db_connection
    auth = ""
    if parsed.username:
        auth = urllib.parse.quote(parsed.username)
        if parsed.password:
            auth += f":{urllib.parse.quote(parsed.password)}"
        auth += "@"
    netloc = f"{auth}{host}"
    if parsed.port is not None:
        netloc = f"{netloc}:{parsed.port}"
    return urllib.parse.urlunparse(parsed._replace(netloc=netloc))


def _override_instance_host(instance: dict[str, object], host: str) -> dict[str, object]:
    payload = json.loads(json.dumps(instance))
    if not host:
        return payload
    for field in ("site_url", "reset_endpoint"):
        value = payload.get(field)
        if isinstance(value, str) and value.strip():
            payload[field] = _replace_url_host(value, host)
    db_connection = payload.get("db_connection")
    if isinstance(db_connection, str) and db_connection.strip():
        payload["db_connection"] = _replace_db_host(db_connection, host)
    placeholders = payload.get("url_placeholders")
    if isinstance(placeholders, dict):
        payload["url_placeholders"] = {
            key: _replace_url_host(value, host)
            if isinstance(value, str) and value.strip()
            else value
            for key, value in placeholders.items()
        }
    return payload


# Default allowlist of real (non-proxy) site ports used by WebArena Verified.
# The proxy rewrites these to ``port + port_offset`` on send. Ports outside
# the set pass through unchanged — notably the ``reset_endpoint`` ports at
# ``site_port + 1``, which the current nginx config does not front.
_DEFAULT_SITE_PORTS: frozenset[int] = frozenset(
    {
        7770,  # shopping
        7780,  # shopping_admin
        8023,  # gitlab
        9999,  # reddit (legacy single-replica topology)
        # Scale-topology reddit replicas (see scripts/proxy_ports.conf:
        # reddit_0..reddit_9 on 9900,9910,...,9990). Included here so the
        # test-time ProxyingHTTPAdapter rewrites hits against any reddit
        # replica port to its nginx proxy port regardless of which
        # topology the live host runs.
        9900,
        9910,
        9920,
        9930,
        9940,
        9950,
        9960,
        9970,
        9980,
        9990,
        8888,  # wikipedia
        3030,  # map
    }
)


@pytest.fixture(scope="session", autouse=True)
def _install_verification_proxy(request: pytest.FixtureRequest):
    """Install the token-gated proxy adapter on ``requests.Session``.

    When ``LIVE_INSTANCES_FILE`` points at an instances file with a
    populated ``verification_proxy`` block, every ``requests.Session()``
    created during the test session rewrites outbound site-port URLs to
    the proxy port and attaches the ``X-Worldsim-Token`` header. This
    is necessary because ``apply_data_seed``, ``acquire_tokens_for_instances``,
    and each editor's ``probe_base_state`` construct their own session
    internally and cannot be reached by a per-test session fixture.
    """
    instances_file = os.getenv("LIVE_INSTANCES_FILE", "").strip()
    if not instances_file:
        yield
        return
    path = Path(instances_file)
    if not path.exists():
        yield
        return
    try:
        config = BenchmarkConfig.model_validate_json(path.read_text())
    except Exception:
        yield
        return
    proxy = config.verification_proxy
    if proxy is None or not proxy.token.strip():
        yield
        return
    uninstall = install_proxy(
        token=proxy.token,
        port_offset=proxy.port_offset,
        site_ports=_DEFAULT_SITE_PORTS,
    )
    try:
        yield
    finally:
        uninstall()


@pytest.fixture(autouse=True)
def _clear_runtime_token_cache():
    clear_run_token_cache()
    yield
    clear_run_token_cache()


@pytest.fixture(scope="session")
def live_config() -> BenchmarkConfig:
    instances_file = os.getenv("LIVE_INSTANCES_FILE", "").strip()
    if not instances_file:
        pytest.skip("LIVE_INSTANCES_FILE is not set")
    path = Path(instances_file)
    if not path.exists():
        pytest.skip(f"LIVE_INSTANCES_FILE does not exist: {path}")
    config = BenchmarkConfig.model_validate_json(path.read_text())
    host = os.getenv("LIVE_HOST_IP", "").strip()
    payload = config.model_dump()
    payload["instances"] = [
        _override_instance_host(instance, host) for instance in payload["instances"]
    ]
    placeholders = payload.get("url_placeholders")
    if host and isinstance(placeholders, dict):
        payload["url_placeholders"] = {
            key: _replace_url_host(value, host)
            if isinstance(value, str) and value.strip()
            else value
            for key, value in placeholders.items()
        }
    return BenchmarkConfig.model_validate(payload)


@pytest.fixture
def live_instance(live_config: BenchmarkConfig):
    by_site = {instance.site_name: instance.model_dump() for instance in live_config.instances}

    def _get(site_name: str) -> dict[str, object]:
        payload = by_site.get(site_name)
        if payload is None:
            pytest.skip(f"live instances file does not define site {site_name!r}")
        return json.loads(json.dumps(payload))

    return _get


@pytest.fixture(scope="session")
def phase_2_tasks() -> dict[str, dict[str, object]]:
    artifact_path = Path(os.getenv("LIVE_PHASE2_ARTIFACT", "logs/phase_2/adversarial_tasks.json"))
    if not artifact_path.exists():
        pytest.skip(f"phase 2 artifact does not exist: {artifact_path}")
    payload = json.loads(artifact_path.read_text())
    return {str(task["id"]): task for task in payload if isinstance(task, dict) and "id" in task}


@pytest.fixture
def phase_2_task(phase_2_tasks: dict[str, dict[str, object]]):
    def _get(task_id: str) -> dict[str, object]:
        payload = phase_2_tasks.get(task_id)
        if payload is None:
            pytest.skip(f"phase 2 artifact does not contain task {task_id!r}")
        return json.loads(json.dumps(payload))

    return _get


@pytest.fixture
def unique_suffix() -> str:
    return uuid.uuid4().hex[:12]
