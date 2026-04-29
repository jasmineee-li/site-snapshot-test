#!/usr/bin/env bash
# Spin up additional DoomArena docker stacks (one per non-GLM model) so
# all models can run in parallel within each arm. Each stack has its
# own gitlab, forum, shopping, shopping_admin containers on a unique port
# offset.
#
# Stack 1 (existing, for GLM-5): gitlab_doom:9002, forum_doom:8081,
#                                shopping:8082, shopping_admin:8083
# Additional stacks: created here, port offsets +10/+20/+30/...
#
# Disk: fresh containers share image layers, so only the writable layer
# costs disk (~0 GB initial, ~500MB-2GB per container over a multi-hour
# run). 20 new containers = ~10-40 GB during use.
#
# Memory: gitlab is ~10GB RAM each, others smaller. 5 new gitlabs +
# 15 other containers ≈ 60-80GB RAM. Host has 1.9 TiB free.
#
# Usage:
#   ./scripts/setup_doomarena_per_model_dockers.sh
#   ./scripts/setup_doomarena_per_model_dockers.sh --stop    # stop the new ones
#   ./scripts/setup_doomarena_per_model_dockers.sh --rm      # stop + remove
#
# After running: see eval_awareness_experiments/DOCKER_PORTS_MULTI.md
# for the canonical model→port mapping.

set -euo pipefail

ACTION="${1:-up}"

# Stack 1 = existing containers, owned by GLM-5
# Stacks 2-6 = new containers, one per non-GLM model
# (model_slug | gitlab_port | forum_port | shopping_port | shopping_admin_port)
STACKS=(
    "sonnet     9012 8091 8092 8093"
    "opus       9022 8101 8102 8103"
    "gpt        9032 8111 8112 8113"
    "flash      9042 8121 8122 8123"
    "pro        9052 8131 8132 8133"
    "gemini25   9062 8141 8142 8143"
    "kimi25     9072 8151 8152 8153"
)

container_name() {
    local svc=$1 slug=$2
    echo "${svc}_${slug}"
}

configure_magento_base_url() {
    local container=$1 port=$2
    local base_url="http://localhost:${port}"

    echo "  configuring $container Magento base URL -> $base_url"
    for attempt in $(seq 1 12); do
        if docker exec "$container" /var/www/magento2/bin/magento \
                setup:store-config:set --base-url="$base_url" >/dev/null \
            && docker exec "$container" /var/www/magento2/bin/magento \
                setup:store-config:set --base-url-secure="$base_url" >/dev/null \
            && docker exec "$container" /var/www/magento2/bin/magento \
                cache:flush >/dev/null; then
            return 0
        fi
        echo "    $container Magento CLI not ready yet ($attempt/12); retrying..."
        sleep 10
    done
    echo "  [WARN] failed to configure $container Magento base URL"
    return 0
}

up_stack() {
    local slug=$1 gl_port=$2 fo_port=$3 sh_port=$4 sa_port=$5
    echo "[$slug] starting stack on ports gl=:$gl_port forum=:$fo_port shop=:$sh_port admin=:$sa_port"

    # gitlab
    local gl_name=$(container_name gitlab "$slug")
    if docker ps -a --format '{{.Names}}' | grep -qx "$gl_name"; then
        echo "  $gl_name already exists, starting if stopped"
        docker start "$gl_name" >/dev/null
    else
        docker run -d --name "$gl_name" -p "${gl_port}:8023" --hostname localhost \
            gitlab-populated-final-port8023:latest \
            /opt/gitlab/embedded/bin/runsvdir-start >/dev/null
        echo "  $gl_name created"
    fi

    # forum (postmill / reddit)
    local fo_name=$(container_name forum "$slug")
    if docker ps -a --format '{{.Names}}' | grep -qx "$fo_name"; then
        docker start "$fo_name" >/dev/null
    else
        docker run -d --name "$fo_name" -p "${fo_port}:80" \
            postmill-populated-exposed-withimg:latest >/dev/null
        echo "  $fo_name created"
    fi

    # shopping
    local sh_name=$(container_name shopping "$slug")
    if docker ps -a --format '{{.Names}}' | grep -qx "$sh_name"; then
        docker start "$sh_name" >/dev/null
    else
        docker run -d --name "$sh_name" -p "${sh_port}:80" \
            shopping_final_0712:latest >/dev/null
        echo "  $sh_name created"
    fi
    configure_magento_base_url "$sh_name" "$sh_port"

    # shopping_admin
    local sa_name=$(container_name shopping_admin "$slug")
    if docker ps -a --format '{{.Names}}' | grep -qx "$sa_name"; then
        docker start "$sa_name" >/dev/null
    else
        docker run -d --name "$sa_name" -p "${sa_port}:80" \
            shopping_admin_final_0719:latest >/dev/null
        echo "  $sa_name created"
    fi
    configure_magento_base_url "$sa_name" "$sa_port"
}

stop_stack() {
    local slug=$1
    for svc in gitlab forum shopping shopping_admin; do
        local name=$(container_name $svc "$slug")
        if docker ps -q --filter "name=^${name}$" | grep -q .; then
            docker stop "$name" >/dev/null && echo "  stopped $name"
        fi
    done
}

rm_stack() {
    local slug=$1
    stop_stack "$slug"
    for svc in gitlab forum shopping shopping_admin; do
        local name=$(container_name $svc "$slug")
        if docker ps -aq --filter "name=^${name}$" | grep -q .; then
            docker rm "$name" >/dev/null && echo "  removed $name"
        fi
    done
}

case "$ACTION" in
    up|"")
        echo "=== bringing up additional DoomArena docker stacks ==="
        for stack in "${STACKS[@]}"; do
            up_stack $stack
        done
        echo
        echo "=== waiting 30s for containers to settle ==="
        sleep 30
        echo
        echo "=== health check (wait for HTTP responses) ==="
        all_ok=1
        for stack in "${STACKS[@]}"; do
            read -r slug gl_port fo_port sh_port sa_port <<< "$stack"
            for spec in "gitlab:${gl_port}/help" "forum:${fo_port}/" "shopping:${sh_port}/" "shopping_admin:${sa_port}/"; do
                svc=${spec%%:*}
                rest=${spec#*:}
                # gitlab takes a long time to wake up; tolerate up to 4 min on it
                max_attempts=$([ "$svc" = "gitlab" ] && echo 24 || echo 6)
                ok=0
                for attempt in $(seq 1 $max_attempts); do
                    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 10 "http://localhost:${rest}" || echo "000")
                    if [[ "$code" =~ ^(200|301|302|401|403|502)$ ]]; then
                        # 502 from gitlab during boot is expected; counts as "alive but waking"
                        if [ "$code" != "502" ]; then
                            ok=1
                            echo "  [ok]  ${svc}_${slug} (:${rest%/*}) → HTTP $code"
                            break
                        fi
                    fi
                    sleep 10
                done
                if [ "$ok" -ne 1 ]; then
                    echo "  [WARN] ${svc}_${slug} (:${rest%/*}) not responding after $max_attempts attempts (last: $code)"
                    all_ok=0
                fi
            done
        done
        echo
        if [ "$all_ok" -eq 1 ]; then
            echo "All 5 stacks healthy ✓"
        else
            echo "Some stacks didn't respond — gitlab usually needs 2-4 min more to boot."
            echo "Re-run health check: docker ps | grep -E 'gitlab|forum|shopping'"
        fi
        ;;
    stop|down)
        echo "=== stopping additional DoomArena stacks ==="
        for stack in "${STACKS[@]}"; do
            slug="${stack%% *}"
            echo "[$slug]"
            stop_stack "$slug"
        done
        ;;
    rm|remove)
        echo "=== removing additional DoomArena stacks (will need to re-create on next up) ==="
        for stack in "${STACKS[@]}"; do
            slug="${stack%% *}"
            echo "[$slug]"
            rm_stack "$slug"
        done
        ;;
    *)
        echo "usage: $0 [up|stop|rm]"
        exit 1
        ;;
esac
