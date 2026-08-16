from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

from warp_taskgen.classifieds_canary import (
    CLASSIFIEDS_DB_MANIFEST_DIGEST,
    CLASSIFIEDS_SOURCE_COMMIT,
    CLASSIFIEDS_WEB_MANIFEST_DIGEST,
    ClassifiedsCanaryConfig,
)
from warp_taskgen.host_config import BenchmarkHostConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "run_classifieds_canary.py"


def _load_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_classifieds_canary", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(tmp_path: Path) -> ClassifiedsCanaryConfig:
    host = BenchmarkHostConfig(
        name="r8a",
        access_mode="host_local",
        advertise_host="127.0.0.1",
        bind_host="127.0.0.1",
        db_bind_host="127.0.0.1",
        compose_dir_remote="/srv/warp-taskgen",
        region="us-east-2",
        instance_id="i-0123456789abcdef0",
    )
    return ClassifiedsCanaryConfig(
        host_config=tmp_path / "r8a.local.yaml",
        host=host,
        site_url="http://127.0.0.1:18080",
        listing_id="12085",
        instances_path="/srv/warp-taskgen/instances.classifieds-canary.json",
        writer_storage_state="/srv/warp-taskgen/secrets/classifieds-writer.json",
        app_env_file="/srv/warp-taskgen/secrets/classifieds-app.env",
        web_image="ghcr.io/bgrins/vwa_classifieds_web",
        web_manifest_digest=CLASSIFIEDS_WEB_MANIFEST_DIGEST,
        db_image="ghcr.io/bgrins/vwa_classifieds_db",
        db_manifest_digest=CLASSIFIEDS_DB_MANIFEST_DIGEST,
        source_commit=CLASSIFIEDS_SOURCE_COMMIT,
        network="zoo-network",
        web_port=18080,
        remote_dir="/srv/warp-taskgen",
        overlay_path="/srv/warp-taskgen/canaries/classifieds-canary.compose.yaml",
    )


class _Runner:
    def __init__(
        self,
        *,
        statuses: list[dict[str, object]],
        aws_code: int = 0,
        availability_code: int = 0,
        stop_codes: list[int] | None = None,
    ) -> None:
        self.statuses = list(statuses)
        self.aws_code = aws_code
        self.availability_code = availability_code
        self.sweep_owner: str | None = None
        self.lifecycle_owner: str | None = None
        self.stop_codes = list(stop_codes or [0])
        self.calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def __call__(self, args, **kwargs):  # type: ignore[no-untyped-def]
        command = tuple(str(item) for item in args)
        self.calls.append((command, dict(kwargs)))
        script = Path(command[0]).name
        if script == "host_resume.sh":
            return self._result(command)
        if script == "remote_job_start.sh":
            return self._result(command, stdout="job_id=canary-1\n")
        if script == "remote_job_status.sh":
            payload = self.statuses.pop(0) if self.statuses else {"status": "running"}
            return self._result(command, stdout=json.dumps(payload) + "\n")
        if script == "remote_job_list.sh":
            return self._result(command, stdout='{"jobs": []}\n')
        if command[0] == "aws":
            if "describe-instances" in command:
                return self._result(
                    command,
                    returncode=self.availability_code,
                    stdout=json.dumps(
                        {
                            "state": "stopped",
                            "sweep": self.sweep_owner,
                            "owner": self.lifecycle_owner,
                        }
                    )
                    + "\n",
                )
            if "create-tags" in command:
                tags = command[command.index("--tags") + 1 :]
                for tag in tags:
                    key, value = tag.removeprefix("Key=").split(",Value=", 1)
                    if key == "worldsim:sweep-in-progress":
                        self.sweep_owner = value
                    if key == "warp:classifieds-canary-owner":
                        self.lifecycle_owner = value
            if "delete-tags" in command:
                tag = command[command.index("--tags") + 1]
                key, expected = tag.removeprefix("Key=").split(",Value=", 1)
                if key == "worldsim:sweep-in-progress" and self.sweep_owner == expected:
                    self.sweep_owner = None
                if key == "warp:classifieds-canary-owner" and self.lifecycle_owner == expected:
                    self.lifecycle_owner = None
            return self._result(command, returncode=self.aws_code)
        if script == "remote_job_stop.sh":
            code = self.stop_codes.pop(0) if self.stop_codes else 0
            return self._result(command, returncode=code)
        if script == "host_park.sh":
            return self._result(command)
        raise AssertionError(f"unexpected command: {command}")

    @staticmethod
    def _result(command, *, returncode: int = 0, stdout: str = ""):  # type: ignore[no-untyped-def]
        import subprocess

        return subprocess.CompletedProcess(command, returncode, stdout, "")


def _loader(config: ClassifiedsCanaryConfig):
    def load(_path, **_kwargs):  # type: ignore[no-untyped-def]
        return config

    return load


def _names(runner: _Runner) -> list[str]:
    return [
        Path(command[0]).name if command[0].endswith(".sh") else command[0]
        for command, _ in runner.calls
    ]


def test_success_polls_json_and_always_clears_tag_before_parking(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(tmp_path)
    runner = _Runner(statuses=[{"status": "running"}, {"status": "exited", "returncode": 0}])

    assert (
        module.run_canary(
            config.host_config,
            run_dir="logs/classifieds-canary/test-success",
            timeout_seconds=10,
            poll_interval_seconds=0,
            runner=runner,
            sleeper=lambda _seconds: None,
            clock=lambda: 0,
            config_loader=_loader(config),
        )
        == 0
    )

    names = _names(runner)
    assert names[0] == "aws"
    assert names.index("host_resume.sh") < names.index("remote_job_start.sh")
    assert names.count("remote_job_status.sh") == 2
    assert names[-5:] == ["aws", "aws", "aws", "host_park.sh", "aws"]
    assert names[:1] == [
        "aws",
    ]
    clear = runner.calls[-4][0]
    assert clear[:3] == ("aws", "ec2", "delete-tags")
    tag = clear[clear.index("--tags") + 1]
    assert tag == "Key=worldsim:sweep-in-progress,Value=true"
    start = next(
        command for command, _ in runner.calls if Path(command[0]).name == "remote_job_start.sh"
    )
    remote = start[start.index("--") + 1 :]
    assert "--host-config" not in remote
    assert str(config.host_config) not in remote


def test_resume_failure_after_exclusive_acquisition_cleans_and_parks(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(tmp_path)
    runner = _Runner(statuses=[])
    original = runner.__call__

    def fail_resume(args, **kwargs):  # type: ignore[no-untyped-def]
        if Path(str(args[0])).name == "host_resume.sh":
            runner.calls.append((tuple(str(item) for item in args), dict(kwargs)))
            raise subprocess.CalledProcessError(1, args)
        return original(args, **kwargs)

    with pytest.raises(subprocess.CalledProcessError):
        module.run_canary(
            config.host_config,
            runner=fail_resume,
            clock=lambda: 0,
            config_loader=_loader(config),
        )

    names = _names(runner)
    assert "host_resume.sh" in names
    assert "remote_job_start.sh" not in names
    assert names[-5:] == ["aws", "aws", "aws", "host_park.sh", "aws"]


def test_running_or_owned_host_is_rejected_before_resume(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(tmp_path)
    runner = _Runner(statuses=[])
    original = runner.__call__

    def occupied(args, **kwargs):  # type: ignore[no-untyped-def]
        command = tuple(str(item) for item in args)
        if command[:3] == ("aws", "ec2", "describe-instances"):
            runner.calls.append((command, dict(kwargs)))
            return subprocess.CompletedProcess(
                command,
                0,
                '{"state":"running","sweep":"other-run"}\n',
                "",
            )
        return original(args, **kwargs)

    with pytest.raises(RuntimeError, match="exclusive stopped host"):
        module.run_canary(
            config.host_config,
            runner=occupied,
            clock=lambda: 0,
            config_loader=_loader(config),
        )

    assert _names(runner) == ["aws"]


def test_concurrent_owner_token_wins_before_host_resume(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(tmp_path)
    runner = _Runner(statuses=[])
    original = runner.__call__

    def competing_claim(args, **kwargs):  # type: ignore[no-untyped-def]
        result = original(args, **kwargs)
        command = tuple(str(item) for item in args)
        if command[:3] == ("aws", "ec2", "create-tags"):
            runner.lifecycle_owner = "other-operator"
        return result

    with pytest.raises(RuntimeError, match="lost its exclusive host ownership"):
        module.run_canary(
            config.host_config,
            runner=competing_claim,
            clock=lambda: 0,
            config_loader=_loader(config),
        )

    assert "host_resume.sh" not in _names(runner)
    assert "host_park.sh" not in _names(runner)


def test_claim_verification_failure_cleans_applied_exact_tags(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(tmp_path)
    runner = _Runner(statuses=[])
    original = runner.__call__
    describes = 0

    def fail_claim_verification(args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal describes
        command = tuple(str(item) for item in args)
        if command[:3] == ("aws", "ec2", "describe-instances"):
            describes += 1
            if describes == 2:
                runner.calls.append((command, dict(kwargs)))
                raise subprocess.CalledProcessError(1, command)
        return original(args, **kwargs)

    with pytest.raises(subprocess.CalledProcessError):
        module.run_canary(
            config.host_config,
            runner=fail_claim_verification,
            clock=lambda: 0,
            config_loader=_loader(config),
        )

    assert "host_resume.sh" not in _names(runner)
    assert "host_park.sh" in _names(runner)
    assert runner.sweep_owner is None
    assert runner.lifecycle_owner is None


def test_signal_during_start_waits_for_job_id_then_stops_exact_job(tmp_path: Path) -> None:
    import signal

    module = _load_module()
    config = _config(tmp_path)
    runner = _Runner(statuses=[])
    original = runner.__call__

    def signal_during_start(args, **kwargs):  # type: ignore[no-untyped-def]
        command = tuple(str(item) for item in args)
        if Path(command[0]).name == "remote_job_start.sh":
            assert kwargs.get("start_new_session") is True
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)
        return original(args, **kwargs)

    with pytest.raises(module.CanaryInterrupted):
        module.run_canary(
            config.host_config,
            runner=signal_during_start,
            clock=lambda: 0,
            config_loader=_loader(config),
        )

    names = _names(runner)
    assert "remote_job_status.sh" not in names
    stop = next(
        command for command, _ in runner.calls if Path(command[0]).name == "remote_job_stop.sh"
    )
    assert stop[stop.index("--job-id") + 1] == "canary-1"


def test_ambiguous_start_timeout_recovers_and_stops_detached_job(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(tmp_path)
    runner = _Runner(statuses=[])
    original = runner.__call__
    run_dir = "logs/classifieds-canary/recover-start"

    def ambiguous_start(args, **kwargs):  # type: ignore[no-untyped-def]
        command = tuple(str(item) for item in args)
        script = Path(command[0]).name
        if script == "remote_job_start.sh":
            runner.calls.append((command, dict(kwargs)))
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if script == "remote_job_list.sh":
            runner.calls.append((command, dict(kwargs)))
            return subprocess.CompletedProcess(
                command,
                0,
                (
                    '{"jobs":[{"job_id":"recovered-1",'
                    '"name":"classifieds-listing-reply-canary",'
                    f'"state_dir":"{run_dir}","status":"running"}}]}}\n'
                ),
                "",
            )
        return original(args, **kwargs)

    with pytest.raises(subprocess.TimeoutExpired):
        module.run_canary(
            config.host_config,
            run_dir=run_dir,
            runner=ambiguous_start,
            clock=lambda: 0,
            config_loader=_loader(config),
        )

    stop = next(
        command for command, _ in runner.calls if Path(command[0]).name == "remote_job_stop.sh"
    )
    assert stop[stop.index("--job-id") + 1] == "recovered-1"


def test_unrecoverable_start_timeout_contains_by_parking_owned_host(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(tmp_path)
    runner = _Runner(statuses=[])
    original = runner.__call__

    def ambiguous_start(args, **kwargs):  # type: ignore[no-untyped-def]
        command = tuple(str(item) for item in args)
        if Path(command[0]).name == "remote_job_start.sh":
            runner.calls.append((command, dict(kwargs)))
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return original(args, **kwargs)

    with pytest.raises(subprocess.TimeoutExpired):
        module.run_canary(
            config.host_config,
            run_dir="logs/classifieds-canary/unrecoverable-start",
            runner=ambiguous_start,
            clock=lambda: 0,
            config_loader=_loader(config),
        )

    names = _names(runner)
    assert names.count("remote_job_list.sh") == 3
    assert "remote_job_stop.sh" not in names
    assert "host_park.sh" in names
    assert runner.lifecycle_owner is None


def test_timeout_stops_remote_job_then_cleans_host(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(tmp_path)
    runner = _Runner(statuses=[{"status": "running"}])
    ticks = iter([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])

    with pytest.raises(TimeoutError, match="timed out"):
        module.run_canary(
            config.host_config,
            timeout_seconds=1,
            poll_interval_seconds=0,
            runner=runner,
            sleeper=lambda _seconds: None,
            clock=lambda: next(ticks),
            config_loader=_loader(config),
        )

    names = _names(runner)
    assert names.index("remote_job_stop.sh") < names.index("host_park.sh")
    assert names[-1] == "aws"
    stop = next(
        command for command, _ in runner.calls if Path(command[0]).name == "remote_job_stop.sh"
    )
    assert "--job-id" in stop and "canary-1" in stop
    assert "--graceful" in stop
    assert ("--pause-timeout", "300") == tuple(
        stop[stop.index("--pause-timeout") : stop.index("--pause-timeout") + 2]
    )
    start = next(
        command for command, _ in runner.calls if Path(command[0]).name == "remote_job_start.sh"
    )
    remote = start[start.index("--") + 1 :]
    selected = remote[remote.index("--run-dir") + 1]
    assert selected.startswith(f"{config.run_root}/")
    assert selected != config.run_root


def test_status_command_timeout_stops_job_and_cleans_host(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(tmp_path)
    runner = _Runner(statuses=[])
    original_call = runner.__call__

    def timeout_status(args, **kwargs):  # type: ignore[no-untyped-def]
        command = tuple(str(item) for item in args)
        if Path(command[0]).name == "remote_job_status.sh":
            runner.calls.append((command, dict(kwargs)))
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return original_call(args, **kwargs)

    with pytest.raises(subprocess.TimeoutExpired):
        module.run_canary(
            config.host_config,
            timeout_seconds=10,
            poll_interval_seconds=0,
            runner=timeout_status,
            sleeper=lambda _seconds: None,
            clock=lambda: 0,
            config_loader=_loader(config),
        )

    names = _names(runner)
    assert names.index("remote_job_stop.sh") < names.index("host_park.sh")
    assert names[-1] == "aws"
    assert all(call_kwargs.get("timeout", 0) > 0 for _, call_kwargs in runner.calls)


def test_invalid_run_dir_fails_before_host_resume(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(tmp_path)
    runner = _Runner(statuses=[])

    with pytest.raises(ValueError, match="run_dir"):
        module.run_canary(
            config.host_config,
            run_dir="../../unsafe",
            runner=runner,
            config_loader=_loader(config),
        )

    assert runner.calls == []


def test_local_operator_lock_rejects_concurrent_wrapper_before_aws(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(tmp_path)
    runner = _Runner(statuses=[])
    assert config.host.instance_id is not None

    with module._exclusive_operator_lock(config.host.instance_id):
        with pytest.raises(RuntimeError, match="operator already owns"):
            module.run_canary(
                config.host_config,
                runner=runner,
                config_loader=_loader(config),
            )

    assert runner.calls == []


def test_poll_sleep_is_capped_by_remaining_deadline(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(tmp_path)
    runner = _Runner(statuses=[{"status": "running"}])
    ticks = iter([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 4.0])
    sleeps: list[float] = []

    def capture_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        raise module.CanaryInterrupted(15)

    with pytest.raises(module.CanaryInterrupted):
        module.run_canary(
            config.host_config,
            timeout_seconds=10,
            poll_interval_seconds=60,
            runner=runner,
            sleeper=capture_sleep,
            clock=lambda: next(ticks),
            config_loader=_loader(config),
        )

    assert sleeps == [6.0]


def test_interrupt_stops_remote_job_and_runs_both_cleanup_steps(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(tmp_path)
    runner = _Runner(statuses=[{"status": "running"}])

    def interrupt(_seconds: float) -> None:
        raise module.CanaryInterrupted(15)

    with pytest.raises(module.CanaryInterrupted):
        module.run_canary(
            config.host_config,
            timeout_seconds=10,
            poll_interval_seconds=1,
            runner=runner,
            sleeper=interrupt,
            clock=lambda: 0,
            config_loader=_loader(config),
        )

    names = _names(runner)
    assert "remote_job_stop.sh" in names
    assert names[-5:] == ["aws", "aws", "aws", "host_park.sh", "aws"]


def test_failed_graceful_stop_uses_explicit_abrupt_fallback_before_parking(
    tmp_path: Path,
) -> None:
    module = _load_module()
    config = _config(tmp_path)
    runner = _Runner(statuses=[{"status": "running"}], stop_codes=[1, 0])
    ticks = iter([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0])

    with pytest.raises(TimeoutError):
        module.run_canary(
            config.host_config,
            timeout_seconds=1,
            poll_interval_seconds=0,
            runner=runner,
            sleeper=lambda _seconds: None,
            clock=lambda: next(ticks),
            config_loader=_loader(config),
        )

    stops = [
        command for command, _ in runner.calls if Path(command[0]).name == "remote_job_stop.sh"
    ]
    assert len(stops) == 2
    assert "--graceful" in stops[0] and "--force" not in stops[0]
    assert "--force" in stops[1] and "--graceful" not in stops[1]
    assert _names(runner)[-5:] == ["aws", "aws", "aws", "host_park.sh", "aws"]


def test_cleanup_refuses_parking_when_sweep_tag_clear_fails(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(tmp_path)
    runner = _Runner(statuses=[{"status": "exited", "returncode": 0}], aws_code=1)

    with pytest.raises(module.CanaryCleanupError, match="clear sweep tag"):
        module.run_canary(
            config.host_config,
            timeout_seconds=10,
            poll_interval_seconds=0,
            runner=runner,
            sleeper=lambda _seconds: None,
            clock=lambda: 0,
            config_loader=_loader(config),
        )
    assert "host_park.sh" not in _names(runner)


def test_park_exception_retains_owner_fence(tmp_path: Path) -> None:
    module = _load_module()
    config = _config(tmp_path)
    runner = _Runner(statuses=[{"status": "exited", "returncode": 0}])
    original = runner.__call__

    def fail_park(args, **kwargs):  # type: ignore[no-untyped-def]
        command = tuple(str(item) for item in args)
        if Path(command[0]).name == "host_park.sh":
            runner.calls.append((command, dict(kwargs)))
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        return original(args, **kwargs)

    with pytest.raises(module.CanaryCleanupError, match="park host"):
        module.run_canary(
            config.host_config,
            runner=fail_park,
            clock=lambda: 0,
            config_loader=_loader(config),
        )

    assert runner.sweep_owner is None
    assert runner.lifecycle_owner is not None
