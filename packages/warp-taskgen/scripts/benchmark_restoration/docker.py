from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .errors import DockerOperationError, DockerTimeoutError


class DockerBackend:
    """Small bounded Docker/Compose adapter used by the fixed-instance daemon."""

    def __init__(self, compose_path: Path, *, command_timeout: float = 600.0) -> None:
        if command_timeout <= 0 or command_timeout > 900:
            raise ValueError("command_timeout must be between 0 and 900 seconds")
        self.compose_path = compose_path
        self.command_timeout = float(command_timeout)

    def _run(self, args: list[str], *, timeout: float | None = None) -> str:
        try:
            result = subprocess.run(
                args,
                capture_output=True,
                text=True,
                check=False,
                timeout=self.command_timeout if timeout is None else timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise DockerTimeoutError from exc
        except OSError as exc:
            raise DockerOperationError("docker_unavailable") from exc
        if result.returncode != 0:
            raise DockerOperationError("docker_command_failed")
        return result.stdout

    def compose_service(self, service: str) -> dict[str, Any]:
        raw = self._run(
            [
                "docker",
                "compose",
                "-f",
                str(self.compose_path),
                "config",
                "--format",
                "json",
            ]
        )
        try:
            document = json.loads(raw)
            services = document["services"]
            value = services[service]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise DockerOperationError("config_drift") from exc
        if not isinstance(value, dict):
            raise DockerOperationError("config_drift")
        return value

    def inspect_container(self, name: str) -> dict[str, Any]:
        raw = self._run(["docker", "inspect", name])
        try:
            rows = json.loads(raw)
            value = rows[0]
        except (IndexError, TypeError, json.JSONDecodeError) as exc:
            raise DockerOperationError("container_missing") from exc
        if not isinstance(value, dict):
            raise DockerOperationError("container_missing")
        return value

    def inspect_image(self, image: str) -> dict[str, Any]:
        raw = self._run(["docker", "image", "inspect", image])
        try:
            rows = json.loads(raw)
            value = rows[0]
        except (IndexError, TypeError, json.JSONDecodeError) as exc:
            raise DockerOperationError("image_missing") from exc
        if not isinstance(value, dict):
            raise DockerOperationError("image_missing")
        return value

    def recreate(self, service: str) -> None:
        self._run(
            [
                "docker",
                "compose",
                "-f",
                str(self.compose_path),
                "up",
                "-d",
                "--no-deps",
                "--no-build",
                "--force-recreate",
                "--pull",
                "never",
                service,
            ]
        )
