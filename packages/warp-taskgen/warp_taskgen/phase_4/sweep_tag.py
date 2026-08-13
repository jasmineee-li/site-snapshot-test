"""Best-effort EC2 instance tag for the lifecycle auto-stop gates.

Phase 4 runners on EC2 hosts mark themselves with
``worldsim:sweep-in-progress=true`` so the EventBridge and CloudWatch
auto-stop layers know not to fire mid-run. The tag is best-effort: missing
``aws`` CLI, missing IAM, or running off-EC2 all log and continue instead
of failing the run.

Env knobs (for tests and local debugging):
  WORLDSIM_DISABLE_SWEEP_TAG=1       opt out entirely
  WORLDSIM_SWEEP_TAG_INSTANCE_ID=... override IMDS discovery
  WORLDSIM_SWEEP_TAG_REGION=...      override IMDS discovery
"""

from __future__ import annotations

import logging
import os
import subprocess
import urllib.request
from collections.abc import Iterator
from contextlib import contextmanager

LOG = logging.getLogger(__name__)

SWEEP_TAG_KEY = "worldsim:sweep-in-progress"
SWEEP_TAG_VALUE = "true"

_IMDS_BASE = "http://169.254.169.254/latest"
_IMDS_TIMEOUT_S = 1.0
_AWS_CLI_TIMEOUT_S = 30.0


def _imdsv2_token() -> str | None:
    try:
        req = urllib.request.Request(
            f"{_IMDS_BASE}/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        with urllib.request.urlopen(req, timeout=_IMDS_TIMEOUT_S) as resp:
            return resp.read().decode("ascii").strip()
    except Exception:
        return None


def _imds_get(path: str, token: str) -> str | None:
    try:
        req = urllib.request.Request(
            f"{_IMDS_BASE}/{path}",
            headers={"X-aws-ec2-metadata-token": token},
        )
        with urllib.request.urlopen(req, timeout=_IMDS_TIMEOUT_S) as resp:
            return resp.read().decode("ascii").strip()
    except Exception:
        return None


def _discover_instance_and_region() -> tuple[str, str] | None:
    env_id = os.environ.get("WORLDSIM_SWEEP_TAG_INSTANCE_ID")
    env_region = os.environ.get("WORLDSIM_SWEEP_TAG_REGION")
    if env_id and env_region:
        return env_id, env_region
    token = _imdsv2_token()
    if token is None:
        return None
    instance_id = _imds_get("meta-data/instance-id", token)
    region = _imds_get("meta-data/placement/region", token)
    if not instance_id or not region:
        return None
    return instance_id, region


def _run_aws(args: list[str]) -> bool:
    try:
        subprocess.run(
            args,
            check=True,
            capture_output=True,
            timeout=_AWS_CLI_TIMEOUT_S,
        )
        return True
    except FileNotFoundError:
        LOG.warning("sweep_tag: aws CLI not on PATH; tag operation skipped")
        return False
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", "replace").strip()
        LOG.warning("sweep_tag: aws %s failed (rc=%s): %s", args[1:3], exc.returncode, stderr)
        return False
    except subprocess.TimeoutExpired:
        LOG.warning("sweep_tag: aws %s timed out", args[1:3])
        return False


def _set_tag(instance_id: str, region: str) -> bool:
    return _run_aws(
        [
            "aws",
            "ec2",
            "create-tags",
            "--region",
            region,
            "--resources",
            instance_id,
            "--tags",
            f"Key={SWEEP_TAG_KEY},Value={SWEEP_TAG_VALUE}",
        ]
    )


def _clear_tag(instance_id: str, region: str) -> bool:
    return _run_aws(
        [
            "aws",
            "ec2",
            "delete-tags",
            "--region",
            region,
            "--resources",
            instance_id,
            "--tags",
            f"Key={SWEEP_TAG_KEY}",
        ]
    )


@contextmanager
def sweep_in_progress(*, disabled: bool = False) -> Iterator[None]:
    """Mark this EC2 instance as in-sweep for the duration of the context.

    No-ops cleanly when off EC2, when the aws CLI is missing, when IAM lacks
    ec2:CreateTags, or when ``WORLDSIM_DISABLE_SWEEP_TAG=1`` is set.
    """
    if disabled or os.environ.get("WORLDSIM_DISABLE_SWEEP_TAG") == "1":
        LOG.debug("sweep_in_progress: disabled by config; skipping tag")
        yield
        return

    target = _discover_instance_and_region()
    if target is None:
        LOG.debug("sweep_in_progress: not on EC2 or IMDS unreachable; skipping tag")
        yield
        return

    instance_id, region = target
    set_ok = _set_tag(instance_id, region)
    if set_ok:
        LOG.info("sweep_tag: set %s=%s on %s", SWEEP_TAG_KEY, SWEEP_TAG_VALUE, instance_id)
    try:
        yield
    finally:
        if set_ok:
            if _clear_tag(instance_id, region):
                LOG.info("sweep_tag: cleared %s on %s", SWEEP_TAG_KEY, instance_id)
