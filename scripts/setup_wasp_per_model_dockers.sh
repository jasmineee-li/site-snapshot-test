#!/usr/bin/env bash
# Spin up WASP-only docker stacks so WASP can run multiple isolated cells
# without sharing GitLab/forum state with the original WASP stack or the
# active DoomArena stacks. The first six stacks are named after models;
# w01-w18 are generic extra workers.
#
# This script does not touch:
#   - original WASP containers: gitlab (:9001), forum (:8080)
#   - DoomArena containers: gitlab_doom/forum_doom and per-model Doom stacks
#
# Usage:
#   ./scripts/setup_wasp_per_model_dockers.sh up
#   ./scripts/setup_wasp_per_model_dockers.sh health
#   ./scripts/setup_wasp_per_model_dockers.sh stop
#   ./scripts/setup_wasp_per_model_dockers.sh rm
#   WASP_STACK_FILTER="w01 w02" ./scripts/setup_wasp_per_model_dockers.sh up

set -euo pipefail

ACTION="${1:-up}"
WASP_STACK_FILTER="${WASP_STACK_FILTER:-all}"

# stack | gitlab_port | forum_port
STACKS=(
    "glm       9201 8201"
    "sonnet    9211 8211"
    "opus      9221 8221"
    "gpt       9231 8231"
    "gemini25  9241 8241"
    "kimi25    9251 8251"
    "w01       9301 8301"
    "w02       9311 8311"
    "w03       9321 8321"
    "w04       9331 8331"
    "w05       9341 8341"
    "w06       9351 8351"
    "w07       9361 8361"
    "w08       9371 8371"
    "w09       9381 8381"
    "w10       9391 8391"
    "w11       9401 8401"
    "w12       9411 8411"
    "w13       9421 8421"
    "w14       9431 8431"
    "w15       9441 8441"
    "w16       9451 8451"
    "w17       9461 8461"
    "w18       9471 8471"
)

contains_word() {
    local needle=$1 haystack=$2
    if [ "$haystack" = "all" ]; then
        return 0
    fi
    for word in $haystack; do
        if [ "$word" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

container_name() {
    local svc=$1 slug=$2
    echo "${svc}_wasp_${slug}"
}

disable_forum_rate_limits() {
    local container=$1
    echo "  disabling benchmark-blocking forum rate limits in $container"
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

up_stack() {
    local slug=$1 gl_port=$2 fo_port=$3
    echo "[$slug] starting WASP stack on ports gitlab=:$gl_port forum=:$fo_port"

    local gl_name
    gl_name=$(container_name gitlab "$slug")
    if docker ps -a --format '{{.Names}}' | grep -qx "$gl_name"; then
        echo "  $gl_name already exists, starting if stopped"
        docker start "$gl_name" >/dev/null
    else
        docker run -d --name "$gl_name" -p "${gl_port}:8023" --hostname localhost \
            gitlab-populated-final-port8023:latest \
            /opt/gitlab/embedded/bin/runsvdir-start >/dev/null
        echo "  $gl_name created"
    fi

    local fo_name
    fo_name=$(container_name forum "$slug")
    if docker ps -a --format '{{.Names}}' | grep -qx "$fo_name"; then
        echo "  $fo_name already exists, starting if stopped"
        docker start "$fo_name" >/dev/null
    else
        docker run -d --name "$fo_name" -p "${fo_port}:80" \
            postmill-populated-exposed-withimg:latest >/dev/null
        echo "  $fo_name created"
    fi
    disable_forum_rate_limits "$fo_name"
}

health_stack() {
    local slug=$1 gl_port=$2 fo_port=$3
    local status=0

    # GitLab can return 502 while still booting. Treat that as "not ready";
    # callers can rerun health after another minute.
    local gl_code
    gl_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
        "http://localhost:${gl_port}/help" || echo "000")
    if [[ "$gl_code" =~ ^(200|301|302|401|403)$ ]]; then
        echo "  [ok]  gitlab_wasp_${slug} (:${gl_port}) -> HTTP $gl_code"
    else
        echo "  [WARN] gitlab_wasp_${slug} (:${gl_port}) not ready (HTTP $gl_code)"
        status=1
    fi

    local fo_code
    fo_code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 \
        "http://localhost:${fo_port}/" || echo "000")
    if [[ "$fo_code" =~ ^(200|301|302|401|403)$ ]]; then
        echo "  [ok]  forum_wasp_${slug} (:${fo_port}) -> HTTP $fo_code"
    else
        echo "  [WARN] forum_wasp_${slug} (:${fo_port}) not ready (HTTP $fo_code)"
        status=1
    fi

    return "$status"
}

stop_stack() {
    local slug=$1
    for svc in gitlab forum; do
        local name
        name=$(container_name "$svc" "$slug")
        if docker ps -q --filter "name=^${name}$" | grep -q .; then
            docker stop "$name" >/dev/null && echo "  stopped $name"
        fi
    done
}

rm_stack() {
    local slug=$1
    stop_stack "$slug"
    for svc in gitlab forum; do
        local name
        name=$(container_name "$svc" "$slug")
        if docker ps -aq --filter "name=^${name}$" | grep -q .; then
            docker rm "$name" >/dev/null && echo "  removed $name"
        fi
    done
}

case "$ACTION" in
    up|"")
        echo "=== bringing up WASP per-model docker stacks ==="
        echo "filter=$WASP_STACK_FILTER"
        for stack in "${STACKS[@]}"; do
            read -r slug _ <<< "$stack"
            contains_word "$slug" "$WASP_STACK_FILTER" || continue
            up_stack $stack
        done
        echo
        echo "Containers started. GitLab usually needs 2-4 minutes before planting."
        echo "Run: ./scripts/setup_wasp_per_model_dockers.sh health"
        ;;
    health)
        echo "=== WASP per-model docker health ==="
        echo "filter=$WASP_STACK_FILTER"
        status=0
        for stack in "${STACKS[@]}"; do
            read -r slug _ <<< "$stack"
            contains_word "$slug" "$WASP_STACK_FILTER" || continue
            if ! health_stack $stack; then
                status=1
            fi
        done
        exit "$status"
        ;;
    stop|down)
        echo "=== stopping WASP per-model docker stacks ==="
        for stack in "${STACKS[@]}"; do
            slug="${stack%% *}"
            contains_word "$slug" "$WASP_STACK_FILTER" || continue
            echo "[$slug]"
            stop_stack "$slug"
        done
        ;;
    rm|remove)
        echo "=== removing WASP per-model docker stacks ==="
        for stack in "${STACKS[@]}"; do
            slug="${stack%% *}"
            contains_word "$slug" "$WASP_STACK_FILTER" || continue
            echo "[$slug]"
            rm_stack "$slug"
        done
        ;;
    *)
        echo "usage: $0 [up|health|stop|rm]"
        exit 1
        ;;
esac
