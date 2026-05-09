#!/usr/bin/env bash
# Recreate one WASP GitLab/forum stack from the populated docker images, then
# plant the full WASP task pool into that fresh state.
#
# Usage:
#   ./scripts/wasp_reset_stack.sh gpt 9231 8231
#   ./scripts/wasp_reset_stack.sh default 9001 8080

set -euo pipefail

STACK="${STACK:-${1:-}}"
GITLAB_PORT="${GITLAB_PORT:-${2:-}}"
REDDIT_PORT="${REDDIT_PORT:-${3:-}}"

if [ -z "$STACK" ] || [ -z "$GITLAB_PORT" ] || [ -z "$REDDIT_PORT" ]; then
    echo "usage: STACK=<slug> GITLAB_PORT=<port> REDDIT_PORT=<port> $0"
    echo "   or: $0 <slug> <gitlab_port> <reddit_port>"
    exit 1
fi

if [ "$STACK" = "default" ]; then
    GITLAB_CONTAINER="gitlab"
    FORUM_CONTAINER="forum"
else
    GITLAB_CONTAINER="gitlab_wasp_${STACK}"
    FORUM_CONTAINER="forum_wasp_${STACK}"
fi
GITLAB_URL="http://localhost:${GITLAB_PORT}"
REDDIT_URL="http://localhost:${REDDIT_PORT}"
RESET_LOCK="${WASP_DOCKER_RESET_LOCK:-/tmp/wasp_docker_reset.lock}"
GITLAB_READY_EXTRA_SLEEP="${GITLAB_READY_EXTRA_SLEEP:-30}"

wait_for_http() {
    local label=$1
    local url=$2
    local max_attempts=${3:-120}
    local sleep_seconds=${4:-5}
    local code

    for attempt in $(seq 1 "$max_attempts"); do
        code="$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "$url" || echo "000")"
        if [[ "$code" =~ ^(200|301|302|401|403)$ ]]; then
            echo "  [ok] $label ready at $url -> HTTP $code"
            return 0
        fi
        if [ "$attempt" -eq 1 ] || [ $((attempt % 12)) -eq 0 ]; then
            echo "  [wait] $label not ready yet at $url -> HTTP $code (attempt $attempt/$max_attempts)"
        fi
        sleep "$sleep_seconds"
    done

    echo "  [error] $label did not become ready at $url" >&2
    return 1
}

wait_for_http_body() {
    local label=$1
    local url=$2
    local pattern=$3
    local max_attempts=${4:-120}
    local sleep_seconds=${5:-5}
    local body

    for attempt in $(seq 1 "$max_attempts"); do
        body="$(curl -fsSL --max-time 15 "$url" 2>/dev/null || true)"
        if printf '%s' "$body" | grep -q "$pattern"; then
            echo "  [ok] $label ready at $url"
            return 0
        fi
        if [ "$attempt" -eq 1 ] || [ $((attempt % 12)) -eq 0 ]; then
            echo "  [wait] $label content not ready yet at $url (attempt $attempt/$max_attempts)"
        fi
        sleep "$sleep_seconds"
    done

    echo "  [error] $label did not expose expected content at $url" >&2
    return 1
}

disable_forum_rate_limits() {
    local container=$1
    echo "=== disabling forum rate limits in $container ==="
    docker exec "$container" sh -lc '
        set -e
        cd /var/www/html
        sed -i \
            -e '\''s/@RateLimit(period="5 minutes", max=15/@RateLimit(period="1 second", max=10000/'\'' \
            -e '\''s/@RateLimit(period="1 hour", max=3/@RateLimit(period="1 second", max=10000/'\'' \
            src/DataObject/SubmissionData.php
        sed -i \
            -e '\''s/@RateLimit(period="5 minutes", max=10/@RateLimit(period="1 second", max=10000/'\'' \
            src/DataObject/CommentData.php
        sed -i \
            -e '\''s/@RateLimit(entityClass="App\\Entity\\User", max="3", period="1 hour"/@RateLimit(entityClass="App\\Entity\\User", max="10000", period="1 second"/'\'' \
            src/DataObject/UserData.php
        rm -rf var/cache/dev var/cache/prod
    '
}

remove_container_if_exists() {
    local container=$1

    if ! docker ps -a --format '{{.Names}}' | grep -qx "$container"; then
        return 0
    fi

    for attempt in 1 2 3; do
        echo "=== removing existing $container (attempt $attempt/3) ==="
        if docker rm -f "$container"; then
            return 0
        fi
        echo "=== docker did not remove $container; waiting before retry ==="
        sleep $((attempt * 20))
    done

    echo "=== final remove attempt for $container ==="
    docker rm -f "$container"
}

recreate_containers() {
    echo "=== acquiring docker reset lock: $RESET_LOCK ==="
    exec 9>"$RESET_LOCK"
    flock 9

    remove_container_if_exists "$GITLAB_CONTAINER"
    remove_container_if_exists "$FORUM_CONTAINER"

    echo "=== starting fresh $GITLAB_CONTAINER ==="
    docker run -d --name "$GITLAB_CONTAINER" -p "${GITLAB_PORT}:8023" --hostname localhost \
        gitlab-populated-final-port8023:latest \
        /opt/gitlab/embedded/bin/runsvdir-start >/dev/null

    echo "=== starting fresh $FORUM_CONTAINER ==="
    docker run -d --name "$FORUM_CONTAINER" -p "${REDDIT_PORT}:80" \
        -e RATELIMIT_WHITELIST=0.0.0.0/0,::/0 \
        postmill-populated-exposed-withimg:latest >/dev/null

    disable_forum_rate_limits "$FORUM_CONTAINER"

    flock -u 9
    exec 9>&-
    echo "=== released docker reset lock ==="
}

echo "=== reset WASP stack=$STACK gitlab=$GITLAB_URL reddit=$REDDIT_URL ==="

recreate_containers
wait_for_http "forum" "${REDDIT_URL}/"
wait_for_http "gitlab" "${GITLAB_URL}/help" 180 5
wait_for_http_body "gitlab sign-in form" "${GITLAB_URL}/users/sign_in" "user_login" 120 5

if [ "$GITLAB_READY_EXTRA_SLEEP" != "0" ]; then
    echo "=== waiting ${GITLAB_READY_EXTRA_SLEEP}s after GitLab readiness before WASP planting ==="
    sleep "$GITLAB_READY_EXTRA_SLEEP"
fi

echo "=== planting WASP task pool into fresh stack=$STACK ==="
"$(dirname "$0")/wasp_plant_full_stack.sh" "$STACK" "$GITLAB_PORT" "$REDDIT_PORT"

echo "=== reset + plant done for stack=$STACK ==="
