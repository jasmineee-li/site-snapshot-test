#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST_CONFIG=""
INSTANCES_FILE="${REPO_ROOT}/instances.smoke.json"
VERIFY_READ_SURFACE_URLS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host-config)
            HOST_CONFIG="$2"
            shift 2
            ;;
        --instances)
            INSTANCES_FILE="$2"
            shift 2
            ;;
        --verify-read-surface-urls)
            VERIFY_READ_SURFACE_URLS=1
            shift
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if [[ -f "${REPO_ROOT}/.env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
    set +a
fi

LIVE_HOST_IP="${LIVE_HOST_IP:-}"
if [[ -n "$HOST_CONFIG" ]]; then
    while IFS='=' read -r key quoted_value; do
        eval "value=$quoted_value"
        case "$key" in
            HOST_IP)
                LIVE_HOST_IP="$value"
                ;;
        esac
    done < <(cd "$REPO_ROOT" && uv run python scripts/export_host_config_env.py --host-config "$HOST_CONFIG")
fi

eval "$(
    cd "$REPO_ROOT" && LIVE_HOST_IP_VALUE="$LIVE_HOST_IP" INSTANCES_FILE_VALUE="$INSTANCES_FILE" uv run python - <<'PY'
import json
import os
import shlex
import urllib.parse
from pathlib import Path

from worldsim.config import BenchmarkConfig


def replace_url_host(url: str, host: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if not parsed.scheme or not parsed.netloc or not host:
        return url
    netloc = host
    if parsed.port is not None:
        netloc = f"{host}:{parsed.port}"
    return urllib.parse.urlunparse(parsed._replace(netloc=netloc))


def replace_db_host(db_connection: str, host: str) -> str:
    parsed = urllib.parse.urlparse(db_connection)
    if not parsed.scheme or not parsed.netloc or not host:
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


instances_path = Path(os.environ["INSTANCES_FILE_VALUE"]).resolve()
host = os.environ.get("LIVE_HOST_IP_VALUE", "").strip()
config = BenchmarkConfig.model_validate_json(instances_path.read_text())

exports = {
    "LIVE_INSTANCES_FILE": str(instances_path),
    "LIVE_PHASE2_ARTIFACT": str((instances_path.parent / "logs/phase_2/adversarial_tasks.json").resolve()),
    "LIVE_PHASE2C_ARTIFACT": str((instances_path.parent / "logs/phase_2/feasibility_report.json").resolve()),
    "LIVE_HOST_IP": host,
}

for instance in config.instances:
    site = instance.site_name.upper()
    exports[f"LIVE_{site}_URL"] = replace_url_host(instance.site_url, host)
    if instance.db_connection:
        exports[f"LIVE_{site}_DB_CONNECTION"] = replace_db_host(instance.db_connection, host)

for key, value in exports.items():
    print(f"export {key}={shlex.quote(value)}")
PY
)"

if [[ ! -f "$LIVE_INSTANCES_FILE" ]]; then
    echo "ERROR: LIVE_INSTANCES_FILE does not exist: $LIVE_INSTANCES_FILE" >&2
    exit 1
fi

printf '==> Live integration test config\n'
printf '    LIVE_INSTANCES_FILE = %s\n' "$LIVE_INSTANCES_FILE"
printf '    LIVE_HOST_IP = %s\n' "${LIVE_HOST_IP:-<unchanged>}"
printf '    LIVE_GITLAB_URL = %s\n' "${LIVE_GITLAB_URL:-}"
printf '    LIVE_SHOPPING_URL = %s\n' "${LIVE_SHOPPING_URL:-}"
printf '    LIVE_SHOPPING_ADMIN_URL = %s\n' "${LIVE_SHOPPING_ADMIN_URL:-}"
printf '    LIVE_REDDIT_URL = %s\n' "${LIVE_REDDIT_URL:-}"

cd "$REPO_ROOT"
if [[ -n "$VERIFY_READ_SURFACE_URLS" ]]; then
    export PYTEST_VERIFY_READ_SURFACE_URLS=1
    printf '    PYTEST_VERIFY_READ_SURFACE_URLS = 1\n'
fi
uv run pytest -m "integration or feasibility" tests/integration "$@"
