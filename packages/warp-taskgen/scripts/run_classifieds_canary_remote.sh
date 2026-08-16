#!/usr/bin/env bash
# Execute the bounded Classifieds canary on the benchmark host.
#
# The local launcher validates the operator host YAML.  This remote script
# receives only sanitized values and references, never the local YAML or its
# secrets.  Every Taskgen phase shares one state root through the environment.

set -Eeuo pipefail

# The canary provider route is owned by the one-shot file below. Prevent the
# CLI's normal developer convenience loader from discovering an ancestor .env
# and overriding that bounded route after the file has been consumed.
export PYTHON_DOTENV_DISABLED=1

PROVIDER_ENV_FILE="/home/ubuntu/warp-taskgen-private/classifieds-provider.env"
cleanup_provider_env() {
    rm -f -- "$PROVIDER_ENV_FILE"
}
trap cleanup_provider_env EXIT
UV_BIN="$HOME/.local/bin/uv"
if [[ ! -x "$UV_BIN" ]]; then
    echo "remote uv executable is missing" >&2
    exit 2
fi

RUN_DIR=""
SITE_URL=""
LISTING_ID=""
OVERLAY_PATH=""
PROJECT_NAME=""
NETWORK=""
WEB_PORT=""
INSTANCES=""
WRITER_STORAGE_STATE=""
APP_ENV_FILE=""
WEB_IMAGE_REF=""
DB_IMAGE_REF=""
SOURCE_COMMIT=""

usage() {
    cat >&2 <<'USAGE'
run_classifieds_canary_remote.sh

Required values are sanitized paths/refs supplied by classifieds_canary.py.
The operator host YAML is intentionally not accepted on the remote host.
USAGE
}

while (($#)); do
    case "$1" in
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --site-url) SITE_URL="$2"; shift 2 ;;
        --listing-id) LISTING_ID="$2"; shift 2 ;;
        --overlay-path) OVERLAY_PATH="$2"; shift 2 ;;
        --project-name) PROJECT_NAME="$2"; shift 2 ;;
        --network) NETWORK="$2"; shift 2 ;;
        --web-port) WEB_PORT="$2"; shift 2 ;;
        --instances) INSTANCES="$2"; shift 2 ;;
        --writer-storage-state) WRITER_STORAGE_STATE="$2"; shift 2 ;;
        --app-env-file) APP_ENV_FILE="$2"; shift 2 ;;
        --web-image-ref) WEB_IMAGE_REF="$2"; shift 2 ;;
        --db-image-ref) DB_IMAGE_REF="$2"; shift 2 ;;
        --source-commit) SOURCE_COMMIT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) usage; echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

for required in RUN_DIR SITE_URL LISTING_ID OVERLAY_PATH PROJECT_NAME NETWORK WEB_PORT INSTANCES WRITER_STORAGE_STATE APP_ENV_FILE WEB_IMAGE_REF DB_IMAGE_REF SOURCE_COMMIT; do
    if [[ -z "${!required}" ]]; then
        echo "missing required value: $required" >&2
        exit 2
    fi
done
if [[ ! "$RUN_DIR" =~ ^logs/classifieds-canary/[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
    echo "run directory must be a safe relative Classifieds canary Run root" >&2
    exit 2
fi
if [[ "$PROJECT_NAME" != "warp-classifieds-canary" ]]; then
    echo "project name must be the dedicated Classifieds canary project" >&2
    exit 2
fi

if [[ ! -s "$WRITER_STORAGE_STATE" ]]; then
    echo "writer storage-state reference is missing or empty" >&2
    exit 2
fi
if [[ ! -s "$APP_ENV_FILE" ]]; then
    echo "Classifieds application environment reference is missing or empty" >&2
    exit 2
fi
# Provider credentials use one canary-only file beside the other externally
# owned secrets. Consume and delete it before any benchmark mutation; exported
# values remain available to this bounded process and its children.
if [[ ! -f "$PROVIDER_ENV_FILE" || -L "$PROVIDER_ENV_FILE" || ! -s "$PROVIDER_ENV_FILE" ]]; then
    echo "Classifieds provider environment reference is missing or empty" >&2
    exit 2
fi
PROVIDER_ENV_STAT="$(stat -c '%u:%a' -- "$PROVIDER_ENV_FILE")"
if [[ "$PROVIDER_ENV_STAT" != "$(id -u):600" ]]; then
    echo "Classifieds provider environment must be owned by this user with mode 0600" >&2
    exit 2
fi
set -a
# shellcheck disable=SC1090
source "$PROVIDER_ENV_FILE"
set +a
cleanup_provider_env
trap - EXIT
if [[ -z "${ANTHROPIC_API_KEY:-}" ]] && \
   { [[ -z "${ANTHROPIC_AUTH_TOKEN:-}" ]] || [[ -z "${ANTHROPIC_BASE_URL:-}" ]]; }; then
    echo "Anthropic provider route is not present" >&2
    exit 2
fi
if [[ -e "$RUN_DIR" ]]; then
    echo "run directory already exists; use a fresh canary Run root" >&2
    exit 2
fi

export WARP_TASKGEN_STATE_DIR="$RUN_DIR"
export WORLDSIM_STATE_DIR="$RUN_DIR"
export WARP_TASKGEN_RESUME_POINTER="$RUN_DIR/last_run_state.json"
export WORLDSIM_RESUME_POINTER="$RUN_DIR/last_run_state.json"

# Establish every result-affecting input on the first dispatch and repeat it
# unchanged at each later phase. The Run Definition gate rejects drift instead
# of silently mutating the canary in place.
RUN_DEFINITION_ARGS=(
    --instances "$INSTANCES"
    --feasibility-instances "$INSTANCES"
    --sites classifieds
    --runtime-composition classifieds_listing_reply_poc
    --task-capability-profile classifieds_listing_reply_poc
    --max-tasks-per-site 1
    --feasibility-retry-count 0
    --feasibility-only
    --force-reverify
    --phase-4-variant-system eval-awareness-iterator
    --phase-4-eval-awareness-max-iterations 1
    --phase-4-task-id classifieds_listing_reply_canary_semantic
    --runner browser_use
    --agent-provider anthropic
    --agent-model claude-sonnet-4-6
    --sandbox-model claude-sonnet-4-6
    --agent-llm-timeout 240
    --agent-step-timeout 300
    --agent-task-timeout 900
    --skip-intermediate-asr
)

compose() {
    docker compose --project-name "$PROJECT_NAME" --file "$OVERLAY_PATH" "$@"
}

wait_for_classifieds() {
    local attempt
    for attempt in {1..45}; do
        if curl --fail --silent --max-time 5 \
            "$SITE_URL/index.php?page=login" \
            | grep --fixed-strings --quiet 'login_post'; then
            return 0
        fi
        sleep 2
    done
    echo "Classifieds login surface did not become ready" >&2
    return 1
}

probe() {
    local mode="$1"
    local evidence="$RUN_DIR/classifieds_probe.json"
    local body
    body="$("$UV_BIN" run python -c 'from warp_taskgen.classifieds_canary import canary_body_for_run; import sys; print(canary_body_for_run(sys.argv[1]))' "$RUN_DIR")"
    "$UV_BIN" run python scripts/classifieds_canary_probe.py "$mode" \
        --site-url "$SITE_URL" --listing-id "$LISTING_ID" \
        --writer-storage-state "$WRITER_STORAGE_STATE" --evidence "$evidence" \
        --body "$body"
}

on_error() {
    local status=$?
    compose down --volumes --remove-orphans >/dev/null 2>&1 || true
    mkdir -p "$RUN_DIR"
cat > "$RUN_DIR/operator-next-steps.txt" <<'EOF'
Classifieds canary containers and volumes were removed after failure. When
launched through scripts/run_classifieds_canary.py, the local wrapper clears
the sweep tag and parks the host after this job exits. If this script was
invoked directly, clear the tag and run host_park.sh before inspection. Do not
rerun Phase 4 until the failed run is tagged.
EOF
    exit "$status"
}
trap on_error ERR

on_signal() {
    local status="$1"
    trap - ERR INT TERM
    compose down --volumes --remove-orphans >/dev/null 2>&1 || true
    mkdir -p "$RUN_DIR"
cat > "$RUN_DIR/operator-next-steps.txt" <<'EOF'
Classifieds canary received an operator stop and removed its canary containers
and volumes. The local run_classifieds_canary.py wrapper clears the sweep tag
and parks the host in its finally block; direct invocations must do that
manually before inspecting evidence.
EOF
    exit "$status"
}
trap 'on_signal 130' INT
trap 'on_signal 143' TERM

mkdir -p "$RUN_DIR"
"$UV_BIN" run python scripts/prepare_classifieds_canary.py \
    --site-url "$SITE_URL" --listing-id "$LISTING_ID" --run-dir "$RUN_DIR" \
    --overlay-path "$OVERLAY_PATH" --project-name "$PROJECT_NAME" \
    --network "$NETWORK" --web-port "$WEB_PORT" \
    --web-image-ref "$WEB_IMAGE_REF" --db-image-ref "$DB_IMAGE_REF" \
    --app-env-file "$APP_ENV_FILE" \
    --instances-path "$INSTANCES" --writer-storage-state "$WRITER_STORAGE_STATE" \
    --source-commit "$SOURCE_COMMIT"

docker pull --platform linux/amd64 "$WEB_IMAGE_REF"
docker pull --platform linux/amd64 "$DB_IMAGE_REF"
"$UV_BIN" run python scripts/record_classifieds_canary_images.py \
    --web-image-ref "$WEB_IMAGE_REF" --db-image-ref "$DB_IMAGE_REF" \
    --output "$RUN_DIR/images.json"

compose config --quiet
# EBS persists across host parks. Remove any residue from an interrupted prior
# canary before creating this run's first golden DB state.
compose down --volumes --remove-orphans
compose up --detach --no-build --pull never
wait_for_classifieds
probe precondition
probe write-read

compose down --volumes --remove-orphans
compose up --detach --no-build --pull never
wait_for_classifieds
probe absence

"$UV_BIN" run warp-taskgen phase 2c "${RUN_DEFINITION_ARGS[@]}" --feasibility-concurrency 1
"$UV_BIN" run warp-taskgen phase 3 "${RUN_DEFINITION_ARGS[@]}"

# Phase 2c writes its own UGC row. Recreate the golden DB before Phase 4 so
# the agent sees only the intended seeded baseline and a fresh anonymous reader
# can prove the canary row is absent.
compose down --volumes --remove-orphans
compose up --detach --no-build --pull never
wait_for_classifieds
probe absence

# This is the last gate before an agent is allowed to run.  It checks the
# identified Run/Definition, exact one-instance auth split, verified Phase 2c
# task, Phase 3 contract, pinned image/source provenance, reset absence, and
# the bounded Anthropic command contract without recording any secret value.
"$UV_BIN" run python scripts/preflight_classifieds_canary.py \
    --run-dir "$RUN_DIR" --instances "$INSTANCES" \
    --site-url "$SITE_URL" --listing-id "$LISTING_ID" \
    --writer-storage-state "$WRITER_STORAGE_STATE" \
    --overlay "$OVERLAY_PATH" --project-name "$PROJECT_NAME" \
    --network "$NETWORK" --web-port "$WEB_PORT" --app-env-file "$APP_ENV_FILE" \
    --expected-task-id classifieds_listing_reply_canary_semantic \
    --expected-benign-task-id classifieds_listing_reply_canary \
    --web-image-ref "$WEB_IMAGE_REF" --db-image-ref "$DB_IMAGE_REF" \
    --source-commit "$SOURCE_COMMIT" \
    --task-count 1 --worker-count 1 --max-iterations 1 \
    --variant-system eval-awareness-iterator --runner browser_use \
    --agent-provider anthropic --agent-model claude-sonnet-4-6 \
    --sandbox-model claude-sonnet-4-6

"$UV_BIN" run warp-taskgen phase 4 "${RUN_DEFINITION_ARGS[@]}" --phase-4-max-workers 1

"$UV_BIN" run warp-taskgen status "$RUN_DIR" --json > "$RUN_DIR/status.json"
"$UV_BIN" run warp-taskgen resume --plan --json > "$RUN_DIR/resume-plan.json"

# Create one exact post-Phase-4 reset witness. Its saved reply ID replaces the
# earlier probe ID, so the final absence proof cannot pass on stale evidence.
probe write-read

# Capture final absence proof before removing the canary pair. Parking the
# cloud host is intentionally a local operator action, not a remote command.
compose down --volumes --remove-orphans
compose up --detach --no-build --pull never
wait_for_classifieds
probe absence
# The terminal artifact is written only after the canary pair and volumes are
# removed successfully. A teardown failure must not leave completion.json.
compose down --volumes --remove-orphans
"$UV_BIN" run python scripts/verify_classifieds_canary_completion.py \
    --run-dir "$RUN_DIR" \
    --expected-task-id classifieds_listing_reply_canary_semantic
cat > "$RUN_DIR/operator-next-steps.txt" <<'EOF'
Classifieds canary completed. The local run_classifieds_canary.py wrapper has
cleared the sweep tag and parked the host. If this script was invoked directly,
clear the tag and run scripts/host_park.sh --host-config <ignored-config>.
Resume only with the recorded Run/Definition identities.
EOF
