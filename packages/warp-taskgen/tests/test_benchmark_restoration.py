from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import threading
import time
import uuid
from pathlib import Path

import pytest

from scripts.benchmark_restoration.daemon import RestoreDaemon
from scripts.benchmark_restoration.errors import RestorationError
from warp_taskgen.host_restoration import HostRestorationClient

IMAGE_ID = "sha256:" + "a" * 64
SITE_URL = "http://127.0.0.1:18080"
INSTANCE_ID = "reddit_restore_0"
SCOPE_ID = "00000000-0000-4000-8000-000000000001"
OPERATION_ID = "operation-00000000-0000-4000-8000-000000000001"
SECOND_OPERATION_ID = "warp-reset-00000000-0000-4000-8000-000000000002"
_ACTIVE_DAEMONS: list[RestoreDaemon] = []


@pytest.fixture(autouse=True)
def _close_test_daemons(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "scripts.benchmark_restoration.daemon._OWNER_LOCK_ROOT", tmp_path / "owner-locks"
    )
    yield
    while _ACTIVE_DAEMONS:
        _ACTIVE_DAEMONS.pop().close()


def _service(*, image: str = IMAGE_ID, volumes: list[str] | None = None) -> dict[str, object]:
    return {
        "image": image,
        "pull_policy": "never",
        "network_mode": "bridge",
        "environment": {"WA_ENV_CTRL_EXTERNAL_SITE_URL": SITE_URL},
        "ports": ["127.0.0.1:18080:80"],
        "volumes": volumes or [],
        "labels": {"warp.restore.owner": "test"},
    }


def _instances(path: Path, *, site_url: str = SITE_URL, container_name: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "instances": [
                    {
                        "site_name": "reddit",
                        "site_url": site_url,
                        "replica_name": INSTANCE_ID,
                        "replica_index": 0,
                        **({"container_name": container_name} if container_name else {}),
                    }
                ]
            }
        )
    )


class FakeDocker:
    def __init__(self, service: dict[str, object]) -> None:
        self.services = {INSTANCE_ID: service}
        self.before_id = "container-before"
        self.after_id = "container-after"
        self.recreate_count = 0
        self.fail_recreate: BaseException | None = None
        self.container = self._container(self.before_id)
        self.commands: list[tuple[str, ...]] = []

    def _container(self, container_id: str) -> dict[str, object]:
        service = self.services[INSTANCE_ID]
        mounts: list[dict[str, object]] = []
        for raw in service.get("volumes", []) or []:
            parts = str(raw).split(":", 2)
            source, destination = parts[:2]
            mode = parts[2] if len(parts) == 3 else "rw"
            mounts.append(
                {
                    "Source": source,
                    "Destination": destination,
                    "RW": mode != "ro",
                }
            )
        return {
            "Id": container_id,
            "Image": IMAGE_ID,
            "Config": {
                "Image": service["image"],
                "Labels": service.get("labels", {}),
                "Env": [f"{key}={value}" for key, value in service.get("environment", {}).items()],
            },
            "HostConfig": {"NetworkMode": "bridge"},
            "Mounts": mounts,
            "NetworkSettings": {
                "Networks": {"bridge": {}},
                "Ports": {
                    "80/tcp": [{"HostIp": "127.0.0.1", "HostPort": "18080"}],
                },
            },
        }

    def compose_service(self, service: str) -> dict[str, object]:
        return json.loads(json.dumps(self.services[service]))

    def inspect_container(self, name: str) -> dict[str, object]:
        expected_name = self.services[INSTANCE_ID].get("container_name", INSTANCE_ID)
        assert name == expected_name
        return json.loads(json.dumps(self.container))

    def inspect_image(self, image: str) -> dict[str, object]:
        return {"Id": IMAGE_ID, "RepoDigests": ["example/reddit@" + "b" * 64]}

    def recreate(self, service: str) -> None:
        self.commands.append(("recreate", service))
        self.recreate_count += 1
        if self.fail_recreate is not None:
            raise self.fail_recreate
        self.container = self._container(f"container-after-{self.recreate_count}")


def _daemon(
    tmp_path: Path,
    docker: FakeDocker,
    *,
    readiness=lambda: True,
    socket_path: Path | None = None,
    container_name: str | None = None,
) -> RestoreDaemon:
    instances = tmp_path / "instances.json"
    _instances(instances, container_name=container_name)
    daemon = RestoreDaemon(
        instances_path=instances,
        compose_path=tmp_path / "compose.yml",
        instance_id=INSTANCE_ID,
        socket_path=socket_path or tmp_path / "run" / "restore.sock",
        state_dir=tmp_path / "state",
        docker=docker,
        readiness_probe=readiness,
    )
    _ACTIVE_DAEMONS.append(daemon)
    return daemon


def _request(daemon: RestoreDaemon, *, action: str, scope_id: str = SCOPE_ID, **extra: str):
    payload = {
        "action": action,
        "instance_id": INSTANCE_ID,
        "site_url": SITE_URL,
        "scope_id": scope_id,
        **extra,
    }
    return daemon.handle_request(payload)


def _require_unix_socket(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        probe.bind(str(path))
    except PermissionError:
        pytest.skip("sandbox does not permit AF_UNIX bind")
    finally:
        probe.close()
        path.unlink(missing_ok=True)
        shutil.rmtree(path.parent, ignore_errors=True)


def test_acquire_rejects_wrong_target_without_mutation(tmp_path: Path) -> None:
    docker = FakeDocker(_service())
    daemon = _daemon(tmp_path, docker)

    response = daemon.handle_request(
        {
            "action": "acquire",
            "instance_id": "other_instance",
            "site_url": SITE_URL,
            "scope_id": SCOPE_ID,
        }
    )

    assert response == {"status": "error", "reason": "wrong_target"}
    assert docker.recreate_count == 0


def test_acquire_rejects_mutable_image_and_db_mount(tmp_path: Path) -> None:
    mutable = FakeDocker(_service(image="example/reddit:latest"))
    mutable_daemon = _daemon(tmp_path / "mutable", mutable)
    assert _request(mutable_daemon, action="acquire")["reason"] == ("mutable_image")
    mutable_daemon.close()

    mounted = FakeDocker(_service(volumes=["db:/var/lib/postgresql/data"]))
    assert _request(_daemon(tmp_path / "mount", mounted), action="acquire")["reason"] == (
        "unexpected_db_mount"
    )


@pytest.mark.parametrize(
    "volume", ["config:/etc/gitlab/gitlab.rb:rw", "unapproved:/application/data:ro"]
)
def test_only_readonly_allowlisted_config_mount_is_accepted(tmp_path: Path, volume: str) -> None:
    daemon = _daemon(tmp_path, FakeDocker(_service(volumes=[volume])))
    response = _request(daemon, action="acquire")
    assert response["status"] == "error"
    assert response["reason"] == "unexpected_mount"


def test_conflicting_lease_is_fail_closed(tmp_path: Path) -> None:
    daemon = _daemon(tmp_path, FakeDocker(_service()))
    first = _request(daemon, action="acquire")
    assert first["status"] == "acquired"

    second = _request(daemon, action="acquire", scope_id="00000000-0000-4000-8000-000000000002")
    assert second == {"status": "error", "reason": "lease_conflict"}


def test_config_drift_blocks_restore_before_recreate(tmp_path: Path) -> None:
    docker = FakeDocker(_service())
    daemon = _daemon(tmp_path, docker)
    acquired = _request(daemon, action="acquire")
    assert acquired["status"] == "acquired"
    docker.services[INSTANCE_ID]["ports"] = ["127.0.0.1:18081:80"]

    response = _request(
        daemon,
        action="restore",
        lease_token=acquired["lease_token"],
        operation_id=OPERATION_ID,
    )

    assert response == {"status": "error", "reason": "config_drift"}
    assert docker.recreate_count == 0


@pytest.mark.parametrize("drift", ["environment", "network_mode", "extra_network"])
def test_actual_routing_drift_blocks_restore(tmp_path: Path, drift: str) -> None:
    docker = FakeDocker(_service())
    daemon = _daemon(tmp_path, docker)
    acquired = _request(daemon, action="acquire")
    assert acquired["status"] == "acquired"
    if drift == "environment":
        docker.container["Config"]["Env"] = ["WA_ENV_CTRL_EXTERNAL_SITE_URL=http://127.0.0.1:18081"]
    elif drift == "network_mode":
        docker.container["HostConfig"]["NetworkMode"] = "host"
    else:
        docker.container["NetworkSettings"]["Networks"]["unrelated-network"] = {}
    response = _request(
        daemon, action="restore", lease_token=acquired["lease_token"], operation_id=OPERATION_ID
    )
    assert response["status"] == "error"
    assert response["reason"] == "config_drift"
    assert docker.recreate_count == 0


def test_unimplemented_compose_networking_fails_closed(tmp_path: Path) -> None:
    service = _service()
    service.pop("network_mode")
    service["networks"] = {"custom": {"ipv4_address": "192.0.2.10"}}
    daemon = _daemon(tmp_path, FakeDocker(service))
    response = _request(daemon, action="acquire")
    assert response["status"] == "error"
    assert response["reason"] == "network_mode_unsupported"


def test_duplicate_operation_returns_terminal_result_without_second_recreate(
    tmp_path: Path,
) -> None:
    docker = FakeDocker(_service())
    daemon = _daemon(tmp_path, docker)
    acquired = _request(daemon, action="acquire")
    token = acquired["lease_token"]

    first = _request(daemon, action="restore", lease_token=token, operation_id=OPERATION_ID)
    second = _request(daemon, action="restore", lease_token=token, operation_id=OPERATION_ID)

    assert first["status"] == "restored"
    assert second == first
    assert docker.recreate_count == 1

    third = _request(
        daemon,
        action="restore",
        lease_token=token,
        operation_id=SECOND_OPERATION_ID,
    )
    assert third["status"] == "restored"
    assert third["before_container_id"] == first["after_container_id"]
    assert docker.recreate_count == 2

    released = _request(
        daemon, action="release", lease_token=token, operation_id=SECOND_OPERATION_ID
    )
    assert released["status"] == "released"
    journal = json.loads((tmp_path / "state" / "lease.json").read_text())
    assert "service" not in journal["before"]
    assert token not in (tmp_path / "state" / "lease.json").read_text()


def test_read_only_config_mount_and_compose_container_name_are_read_back(tmp_path: Path) -> None:
    config_source = str(tmp_path / "gitlab.rb")
    Path(config_source).write_text("external_url 'http://127.0.0.1:18080'\n")
    service = _service(
        volumes=[f"{config_source}:/etc/gitlab/gitlab.rb:ro"],
    )
    service["container_name"] = "restore-container-0"
    docker = FakeDocker(service)
    daemon = _daemon(tmp_path, docker, container_name="restore-container-0")

    acquired = _request(daemon, action="acquire")

    assert acquired["status"] == "acquired"
    Path(config_source).write_text("external_url 'http://127.0.0.1:18081'\n")
    response = _request(
        daemon,
        action="restore",
        lease_token=acquired["lease_token"],
        operation_id=OPERATION_ID,
    )
    assert response == {"status": "error", "reason": "config_drift"}
    assert docker.recreate_count == 0


@pytest.mark.parametrize("contents", ["{", "{}", '{"status":"released"}'])
def test_corrupt_state_is_not_treated_as_an_empty_lease(tmp_path: Path, contents: str) -> None:
    daemon = _daemon(tmp_path, FakeDocker(_service()))
    (tmp_path / "state" / "lease.json").write_text(contents)

    response = _request(daemon, action="acquire")

    assert response == {"status": "error", "reason": "state_corrupt"}


def test_unpublished_image_port_does_not_count_as_host_exposure(tmp_path: Path) -> None:
    docker = FakeDocker(_service())
    docker.container["NetworkSettings"]["Ports"]["5432/tcp"] = None
    daemon = _daemon(tmp_path, docker)
    assert _request(daemon, action="acquire")["status"] == "acquired"


def test_different_instance_alias_cannot_own_the_same_container(tmp_path: Path) -> None:
    service = _service()
    service["container_name"] = "shared-restore-container"
    first = _daemon(tmp_path / "first", FakeDocker(service))
    alternate = "reddit_alternate_0"
    instances = tmp_path / "second" / "instances.json"
    _instances(instances)
    config = json.loads(instances.read_text())
    config["instances"][0]["replica_name"] = alternate
    instances.write_text(json.dumps(config))
    docker = FakeDocker(service)
    docker.services[alternate] = service
    with pytest.raises(RestorationError) as raised:
        RestoreDaemon(
            instances_path=instances,
            compose_path=tmp_path / "compose.yml",
            instance_id=alternate,
            socket_path=tmp_path / "second" / "run" / "restore.sock",
            state_dir=tmp_path / "second" / "state",
            docker=docker,
        )
    assert raised.value.reason == "daemon_locked"
    assert docker.recreate_count == 0
    first.close()


def test_second_daemon_for_same_instance_is_rejected_even_with_new_state(tmp_path: Path) -> None:
    first = _daemon(tmp_path / "first", FakeDocker(_service()))

    with pytest.raises(RestorationError) as raised:
        _daemon(tmp_path / "second", FakeDocker(_service()))

    assert raised.value.reason == "daemon_locked"
    first.close()


def test_operation_from_old_scope_cannot_be_reused(tmp_path: Path) -> None:
    docker = FakeDocker(_service())
    first = _daemon(tmp_path, docker)
    acquired = _request(first, action="acquire")
    restored = _request(
        first, action="restore", lease_token=acquired["lease_token"], operation_id=OPERATION_ID
    )
    assert restored["status"] == "restored"
    released = _request(
        first, action="release", lease_token=acquired["lease_token"], operation_id=OPERATION_ID
    )
    assert released["status"] == "released"
    first.close()

    second = _daemon(tmp_path, FakeDocker(_service()))
    new_scope = _request(second, action="acquire", scope_id="00000000-0000-4000-8000-000000000002")
    response = _request(
        second,
        action="restore",
        scope_id=new_scope["scope_id"],
        lease_token=new_scope["lease_token"],
        operation_id=OPERATION_ID,
    )

    assert response == {"status": "error", "reason": "operation_conflict"}


def test_timeout_quarantines_scope_and_never_retries_mutation(tmp_path: Path) -> None:
    docker = FakeDocker(_service())
    docker.fail_recreate = TimeoutError()
    daemon = _daemon(tmp_path, docker)
    acquired = _request(daemon, action="acquire")
    token = acquired["lease_token"]

    timed_out = _request(daemon, action="restore", lease_token=token, operation_id=OPERATION_ID)
    retry = _request(daemon, action="restore", lease_token=token, operation_id="operation-2")

    assert timed_out == {"status": "error", "reason": "timeout_quarantined"}
    assert retry == {"status": "error", "reason": "scope_quarantined"}
    assert docker.recreate_count == 1


def test_socket_permissions_are_narrow(tmp_path: Path) -> None:
    socket_parent = Path("/tmp") / f"warp-r-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    _require_unix_socket(socket_parent / "probe.sock")
    daemon = _daemon(tmp_path, FakeDocker(_service()), socket_path=socket_parent / "s")
    sock = daemon.prepare_socket()
    try:
        assert stat.S_IMODE(daemon.socket_path.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(daemon.socket_path.stat().st_mode) == 0o600
    finally:
        sock.close()
        daemon.close_socket()
        daemon.close()
        shutil.rmtree(socket_parent, ignore_errors=True)


def test_client_round_trip_uses_the_daemon_socket_protocol(tmp_path: Path) -> None:
    socket_parent = Path("/tmp") / f"warp-r-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    _require_unix_socket(socket_parent / "probe.sock")
    docker = FakeDocker(_service())
    daemon = _daemon(tmp_path, docker, socket_path=socket_parent / "s")
    server = threading.Thread(target=daemon.serve_forever, kwargs={"max_requests": 3}, daemon=True)
    server.start()
    deadline = time.monotonic() + 2
    while not daemon.socket_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert daemon.socket_path.exists()
    client = HostRestorationClient(
        daemon.socket_path,
        instance_id=INSTANCE_ID,
        site_url=SITE_URL,
        timeout_s=5,
        transport_retries=0,
    )
    scope = client.acquire(SCOPE_ID)
    restored = client.restore(
        scope_id=SCOPE_ID,
        lease_token=scope["lease_token"],
        operation_id=OPERATION_ID,
    )
    released = client.release(
        scope_id=SCOPE_ID,
        lease_token=scope["lease_token"],
        operation_id=OPERATION_ID,
    )
    server.join(timeout=5)
    try:
        assert scope["status"] == "acquired"
        assert restored["status"] == "restored"
        assert released["status"] == "released"
        assert docker.recreate_count == 1
    finally:
        daemon.close()
        shutil.rmtree(socket_parent, ignore_errors=True)
