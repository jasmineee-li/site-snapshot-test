#!/usr/bin/env bash
# Push selected local env vars into the remote checkout .env without logging values.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=scripts/lib/remote_jobs.sh
source "$REPO_ROOT/scripts/lib/remote_jobs.sh"

HOST_CONFIG=""
REMOTE_DIR=""
SSH_KEY_ARG=""
ENV_FILE="$REPO_ROOT/.env"
KEYS=()
DRY_RUN=0

usage() {
    cat <<'USAGE'
remote_env_push.sh

Options:
  --host-config <path>      benchmark host YAML (required)
  --remote-dir <path>       remote checkout dir (default: <compose_dir_remote>/browser-sim)
  --ssh-key <path>          SSH private key (default: $SSH_KEY or ~/.ssh/webarena-key.pem)
  --env-file <path>         local env file (default: .env)
  --key <name>              env var name to push; repeatable (required)
  --dry-run                 validate and print key names only
  -h, --help                show this help
USAGE
}

while (($#)); do
    case "$1" in
        --host-config) HOST_CONFIG="$2"; shift 2 ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        --ssh-key) SSH_KEY_ARG="$2"; shift 2 ;;
        --env-file) ENV_FILE="$2"; shift 2 ;;
        --key) KEYS+=("$2"); shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) rj_die "unknown arg: $1" ;;
    esac
done

[[ -n "$HOST_CONFIG" ]] || { usage >&2; rj_die "--host-config required"; }
((${#KEYS[@]} > 0)) || { usage >&2; rj_die "--key required"; }
rj_prepare_connection "$HOST_CONFIG" "$SSH_KEY_ARG"
REMOTE_DIR="${REMOTE_DIR:-$(rj_default_remote_dir)}"

payload_file="$(mktemp "${TMPDIR:-/tmp}/worldsim-remote-env.XXXXXX")"
remote_program_file="$(mktemp "${TMPDIR:-/tmp}/worldsim-remote-env-program.XXXXXX")"
trap 'rm -f "$payload_file" "$remote_program_file"' EXIT
chmod 600 "$payload_file"
chmod 600 "$remote_program_file"

python3 - "$ENV_FILE" "${KEYS[@]}" >"$payload_file" <<'PY'
import json
import os
import re
import sys
from pathlib import Path

env_file = Path(sys.argv[1])
keys = sys.argv[2:]
valid = re.compile(r"^[A-Z_][A-Z0-9_]*$")
for key in keys:
    if not valid.match(key):
        raise SystemExit(f"invalid env key: {key}")

values: dict[str, str] = {}
if env_file.exists():
    for raw in env_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in values:
            values[key] = value

selected: dict[str, str] = {}
for key in keys:
    value = os.environ.get(key) or values.get(key) or ""
    if not value.strip():
        raise SystemExit(f"missing non-empty value for {key} in environment or {env_file}")
    selected[key] = value

print(json.dumps(selected, sort_keys=True))
PY

if [[ "$DRY_RUN" -eq 1 ]]; then
    python3 - "$payload_file" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print("would push env keys: " + ", ".join(sorted(payload)))
PY
    exit 0
fi

python3 - "$payload_file" >"$remote_program_file" <<'PY'
import json
import sys
from pathlib import Path

payload_json = Path(sys.argv[1]).read_text(encoding="utf-8")
json.loads(payload_json)
print(f"PAYLOAD_JSON = {payload_json!r}")
print(r'''
import json
import os
import sys
from pathlib import Path

remote_dir = Path(sys.argv[1])
payload = json.loads(PAYLOAD_JSON)
env_path = remote_dir / ".env"
lines = []
seen = set()
if env_path.exists():
    lines = env_path.read_text(encoding="utf-8").splitlines()

out = []
for line in lines:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in line:
        out.append(line)
        continue
    key = line.split("=", 1)[0].strip()
    if key in payload:
        out.append(f"{key}={payload[key]}")
        seen.add(key)
    else:
        out.append(line)

for key in sorted(payload):
    if key not in seen:
        out.append(f"{key}={payload[key]}")

tmp = env_path.with_suffix(".env.tmp")
tmp.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")
os.chmod(tmp, 0o600)
tmp.replace(env_path)
os.chmod(env_path, 0o600)
print("updated remote .env keys: " + ", ".join(sorted(payload)))
''')
PY

rj_ssh python3 - "$REMOTE_DIR" <"$remote_program_file"
