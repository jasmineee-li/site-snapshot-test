from __future__ import annotations


class RestorationError(Exception):
    """Expected fail-closed restoration error with a safe reason code."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class DockerOperationError(RestorationError):
    """A bounded Docker/Compose operation failed."""


class DockerTimeoutError(DockerOperationError):
    """A Docker/Compose operation reached its deadline."""

    def __init__(self) -> None:
        super().__init__("docker_timeout")
