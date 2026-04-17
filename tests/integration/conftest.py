from __future__ import annotations

import json
import os
import urllib.parse
import uuid
from pathlib import Path

import pytest

from worldsim.auth_tokens import clear_run_token_cache
from worldsim.config import BenchmarkConfig


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
            key: _replace_url_host(value, host) if isinstance(value, str) and value.strip() else value
            for key, value in placeholders.items()
        }
    return payload


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
    payload["instances"] = [_override_instance_host(instance, host) for instance in payload["instances"]]
    placeholders = payload.get("url_placeholders")
    if host and isinstance(placeholders, dict):
        payload["url_placeholders"] = {
            key: _replace_url_host(value, host) if isinstance(value, str) and value.strip() else value
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
    artifact_path = Path(
        os.getenv("LIVE_PHASE2_ARTIFACT", "logs/phase_2/adversarial_tasks.json")
    )
    if not artifact_path.exists():
        pytest.skip(f"phase 2 artifact does not exist: {artifact_path}")
    payload = json.loads(artifact_path.read_text())
    return {
        str(task["id"]): task
        for task in payload
        if isinstance(task, dict) and "id" in task
    }


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
