from __future__ import annotations

import ipaddress
import json
import os
import urllib.parse
import uuid
from pathlib import Path

import pytest

from warp_taskgen.auth_tokens import clear_run_token_cache
from warp_taskgen.config import BenchmarkConfig, load_benchmark_config
from warp_taskgen.http_proxy import install_proxy


def _is_loopback_hostname(hostname: str | None) -> bool:
    if hostname is None:
        return False
    normalized = hostname.strip().lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


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
def _build_default_site_ports() -> frozenset[int]:
    """Build the ProxyingHTTPAdapter's ``site_ports`` allowlist.

    Includes both web (site_url) and envctrl (reset_endpoint at
    ``real_web+1``) ports for every legacy-topology site and every
    scale-topology replica. Without envctrl ports in the set, the
    reset_endpoint hits pass through direct to docker-loopback and
    every feasibility test times out.

    Matches the generator in ``scripts/generate_compose_scale.py::
    build_proxy_ports`` which emits a listener per web + envctrl port.
    """
    legacy_web = [
        7770,  # shopping
        7780,  # shopping_admin
        8023,  # gitlab
        9999,  # reddit (legacy single-replica topology)
        8888,  # wikipedia
        3030,  # map
    ]
    scale_reddit_web = list(range(9900, 10000, 10))  # reddit_0..reddit_9
    scale_gitlab_web = list(range(8023, 8224, 10))  # gitlab_0..gitlab_20
    all_web = set(legacy_web) | set(scale_reddit_web) | set(scale_gitlab_web)
    # envctrl = web + 1 (per scripts/generate_compose_scale.py:14).
    all_envctrl = {p + 1 for p in all_web}
    return frozenset(all_web | all_envctrl)


_DEFAULT_SITE_PORTS: frozenset[int] = _build_default_site_ports()


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
        config = load_benchmark_config(path)
    except Exception:
        yield
        return
    proxy = config.verification_proxy
    if proxy is None or not proxy.token.strip():
        yield
        return
    if all(
        _is_loopback_hostname(urllib.parse.urlparse(instance.site_url).hostname)
        for instance in config.instances
    ):
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
    config = load_benchmark_config(path)
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
    by_site = {}
    for instance in live_config.instances:
        payload = instance.model_dump()
        payload["benchmark"] = live_config.benchmark_name or "webarena_verified"
        by_site[instance.site_name] = payload

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
