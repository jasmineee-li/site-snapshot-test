"""Sync Magento base_url across every shopping* replica in an instances.json.

Supersedes ``scripts/fix_magento_base_url.sh`` for scale deployments.
Differences:

1. **Loops through every replica.** The bash script hardcodes
   ``webarena-verified-shopping`` / ``webarena-verified-shopping_admin``;
   under scale compose the real containers are named
   ``webarena-verified-shopping_0..N``. Deriving the container name from
   ``instances.json:replica_name`` guarantees every replica is covered.

2. **Uses ``config:set --lock-env`` (defense in depth).** Writes to
   ``app/etc/env.php``, which sits at the TOP of Magento's precedence chain
   (``env.php > config.php > core_config_data``). Beats any subsequent
   ``setup:store-config:set`` from env-ctrl's ``_init()``; beats
   ``app:config:import`` and ``cache:flush``. Belt-and-suspenders to the
   root-cause fix in ``scripts/generate_compose_scale.py:96``
   (``WA_ENV_CTRL_EXTERNAL_SITE_URL`` now bakes the proxy port, so
   ``_init()`` is idempotent with the proxy origin).

3. **Structured JSON summary on exit.** Per-replica timing + final state so
   any residual drift is investigable offline without re-running the script.

Usage:
    uv run python scripts/sync_magento_base_urls.py \\
        --instances instances.scale.json --verify-after --retry-on-revert 2
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("sync_magento_base_urls")

SHOPPING_SITES = frozenset({"shopping", "shopping_admin"})


@dataclass
class ReplicaResult:
    container: str
    site: str
    desired: str
    applied_lock_env: bool = False
    lock_env_error: str | None = None
    applied_sql_fallback: bool = False
    http_probe_ok: bool = False
    http_probe_value: str | None = None
    reverted_after_sleep: bool = False
    reverted_value: str | None = None
    attempts: int = 0
    elapsed_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)


def _expected_base_url(instance: dict[str, Any], proxy_port_offset: int) -> str:
    """Return ``http://<host>:<port>/`` where port = raw + proxy_port_offset."""
    site_url = str(instance.get("site_url", "")).rstrip("/")
    if not site_url:
        raise ValueError(f"instance {instance.get('replica_name')!r} has no site_url")
    parts = urlsplit(site_url)
    host = parts.hostname or "127.0.0.1"
    real_port = parts.port
    if real_port is None:
        raise ValueError(f"site_url {site_url!r} has no port")
    proxy_port = real_port + proxy_port_offset if proxy_port_offset else real_port
    scheme = parts.scheme or "http"
    return f"{scheme}://{host}:{proxy_port}/"


def _container_name(instance: dict[str, Any]) -> str:
    # Prefer the explicit replica_name (generator writes e.g. "shopping_0");
    # fall back to site_name + replica_index when replica_name is absent.
    replica_name = str(instance.get("replica_name", "")).strip()
    if replica_name:
        return f"webarena-verified-{replica_name}"
    site = str(instance.get("site_name", "")).strip()
    idx = instance.get("replica_index")
    if site and isinstance(idx, int):
        return f"webarena-verified-{site}_{idx}"
    raise ValueError(f"cannot derive container name from instance {instance!r}")


def _docker_exec(container: str, cmd: list[str], *, dry_run: bool) -> tuple[int, str]:
    """Run ``docker exec <container> <cmd>`` and return (rc, combined_output)."""
    full = ["docker", "exec", container, *cmd]
    if dry_run:
        logger.info("DRY-RUN %s", " ".join(full))
        return 0, ""
    try:
        completed = subprocess.run(
            full,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except FileNotFoundError:
        return 127, "docker not found on PATH"
    output = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode, output.strip()


def _apply_lock_env(
    container: str,
    desired: str,
    *,
    dry_run: bool,
    result: ReplicaResult,
) -> bool:
    """Run ``config:set --lock-env`` for both unsecure + secure base_url."""
    paths = ["web/unsecure/base_url", "web/secure/base_url"]
    for path in paths:
        rc, out = _docker_exec(
            container,
            ["php", "bin/magento", "config:set", "--lock-env", path, desired],
            dry_run=dry_run,
        )
        if rc != 0:
            result.lock_env_error = f"{path}: rc={rc} out={out[:200]}"
            return False
    rc, out = _docker_exec(container, ["php", "bin/magento", "cache:flush"], dry_run=dry_run)
    if rc != 0:
        # cache:flush failure is advisory, not fatal; env.php still wins.
        result.notes.append(f"cache:flush returned {rc}: {out[:200]}")
    result.applied_lock_env = True
    return True


def _apply_sql_fallback(
    container: str,
    desired: str,
    *,
    mysql_user: str,
    mysql_pass: str,
    mysql_db: str,
    dry_run: bool,
    result: ReplicaResult,
) -> bool:
    """Last-resort SQL UPDATE (pre-2017 Magento lacking --lock-env)."""
    sql = (
        "UPDATE core_config_data SET value=%s WHERE path IN "
        "('web/unsecure/base_url','web/secure/base_url');"
    ).replace("%s", f"'{desired}'")
    rc, out = _docker_exec(
        container,
        ["mysql", f"-u{mysql_user}", f"-p{mysql_pass}", "-D", mysql_db, "-e", sql],
        dry_run=dry_run,
    )
    if rc != 0:
        result.notes.append(f"sql fallback failed: rc={rc} out={out[:200]}")
        return False
    _docker_exec(container, ["php", "bin/magento", "cache:flush"], dry_run=dry_run)
    result.applied_sql_fallback = True
    return True


def _http_probe(desired: str, instance: dict[str, Any]) -> tuple[bool, str | None]:
    """Fetch the shopping home and compare Magento's rendered BASE_URL."""
    try:
        from worldsim.phase_4.magento_health import probe_base_url
    except ImportError as exc:
        return False, f"probe_base_url import failed: {exc}"
    site_url = str(instance.get("site_url", ""))
    try:
        actual = probe_base_url(site_url)
    except Exception as exc:
        return False, f"probe_base_url raised: {exc}"
    if actual is None:
        return False, None
    # Compare after trailing-slash normalization.
    return (actual.rstrip("/") == desired.rstrip("/")), actual


def _process_instance(
    instance: dict[str, Any],
    *,
    proxy_port_offset: int,
    mysql_user: str,
    mysql_pass: str,
    mysql_db: str,
    verify_after: bool,
    dry_run: bool,
) -> ReplicaResult:
    container = _container_name(instance)
    desired = _expected_base_url(instance, proxy_port_offset)
    result = ReplicaResult(
        container=container,
        site=str(instance.get("site_name", "")),
        desired=desired,
    )
    start = time.monotonic()
    result.attempts = 1
    if not _apply_lock_env(container, desired, dry_run=dry_run, result=result):
        _apply_sql_fallback(
            container,
            desired,
            mysql_user=mysql_user,
            mysql_pass=mysql_pass,
            mysql_db=mysql_db,
            dry_run=dry_run,
            result=result,
        )
    if verify_after and not dry_run:
        ok, actual = _http_probe(desired, instance)
        result.http_probe_ok = ok
        result.http_probe_value = actual
    result.elapsed_seconds = round(time.monotonic() - start, 3)
    return result


def _retry_on_revert(
    instances: list[dict[str, Any]],
    results: list[ReplicaResult],
    *,
    sleep_seconds: int,
    attempts: int,
    proxy_port_offset: int,
    mysql_user: str,
    mysql_pass: str,
    mysql_db: str,
    dry_run: bool,
) -> None:
    """Sleep then re-probe; re-apply on any replica that drifted back."""
    for attempt in range(1, attempts + 1):
        logger.info(
            "retry-on-revert attempt %d/%d (sleeping %ds)", attempt, attempts, sleep_seconds
        )
        if not dry_run:
            time.sleep(sleep_seconds)
        any_reverted = False
        for instance, result in zip(instances, results, strict=True):
            ok, actual = _http_probe(result.desired, instance)
            if ok:
                continue
            any_reverted = True
            result.reverted_after_sleep = True
            result.reverted_value = actual
            result.attempts += 1
            # Re-apply.
            _apply_lock_env(result.container, result.desired, dry_run=dry_run, result=result)
            ok2, actual2 = _http_probe(result.desired, instance)
            result.http_probe_ok = ok2
            result.http_probe_value = actual2
        if not any_reverted:
            logger.info("no revert detected; retry loop done")
            return


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", required=True, help="path to instances.json")
    ap.add_argument(
        "--proxy-port-offset",
        type=int,
        default=None,
        help="override verification_proxy.port_offset from instances.json",
    )
    ap.add_argument("--mysql-user", default="magentouser")
    ap.add_argument("--mysql-pass", default="MyPassword")
    ap.add_argument("--mysql-db", default="magentodb")
    ap.add_argument(
        "--verify-after",
        action="store_true",
        help="run HTTP probe after each apply",
    )
    ap.add_argument(
        "--retry-on-revert",
        type=int,
        default=0,
        metavar="N",
        help="after applying, sleep 30s and re-probe N times; re-apply on drift",
    )
    ap.add_argument(
        "--revert-sleep-seconds",
        type=int,
        default=30,
        help="seconds to wait between retry-on-revert probes",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--summary-out",
        default=None,
        help="optional path to write structured JSON summary",
    )
    args = ap.parse_args()

    instances_path = Path(args.instances)
    if not instances_path.exists():
        logger.error("instances file not found: %s", instances_path)
        return 2
    payload = json.loads(instances_path.read_text(encoding="utf-8"))
    all_instances = payload.get("instances", []) if isinstance(payload, dict) else []
    shopping = [
        inst
        for inst in all_instances
        if isinstance(inst, dict) and str(inst.get("site_name", "")) in SHOPPING_SITES
    ]
    if not shopping:
        logger.warning("no shopping* instances in %s; nothing to do", instances_path)
        return 0

    proxy_port_offset = args.proxy_port_offset
    if proxy_port_offset is None:
        proxy_cfg = payload.get("verification_proxy") if isinstance(payload, dict) else None
        if isinstance(proxy_cfg, dict):
            proxy_port_offset = int(proxy_cfg.get("port_offset", 0))
        else:
            proxy_port_offset = 0
    logger.info(
        "processing %d shopping replicas (proxy_port_offset=%d)",
        len(shopping),
        proxy_port_offset,
    )
    if not args.dry_run and shutil.which("docker") is None:
        logger.error("docker not found on PATH; rerun with --dry-run or install docker")
        return 2

    results: list[ReplicaResult] = []
    for instance in shopping:
        result = _process_instance(
            instance,
            proxy_port_offset=proxy_port_offset,
            mysql_user=args.mysql_user,
            mysql_pass=args.mysql_pass,
            mysql_db=args.mysql_db,
            verify_after=args.verify_after,
            dry_run=args.dry_run,
        )
        results.append(result)
        status = "OK" if (result.http_probe_ok or not args.verify_after) else "FAIL"
        logger.info(
            "%s %s (%s) -> %s in %.2fs%s",
            status,
            result.container,
            result.site,
            result.desired,
            result.elapsed_seconds,
            " (lock-env)" if result.applied_lock_env else " (sql-fallback)",
        )

    if args.retry_on_revert > 0 and args.verify_after:
        _retry_on_revert(
            shopping,
            results,
            sleep_seconds=args.revert_sleep_seconds,
            attempts=args.retry_on_revert,
            proxy_port_offset=proxy_port_offset,
            mysql_user=args.mysql_user,
            mysql_pass=args.mysql_pass,
            mysql_db=args.mysql_db,
            dry_run=args.dry_run,
        )

    summary = {
        "proxy_port_offset": proxy_port_offset,
        "replicas": [asdict(r) for r in results],
    }
    if args.summary_out:
        Path(args.summary_out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("wrote summary to %s", args.summary_out)
    else:
        print(json.dumps(summary, indent=2))

    failed = [r for r in results if args.verify_after and not r.http_probe_ok]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
