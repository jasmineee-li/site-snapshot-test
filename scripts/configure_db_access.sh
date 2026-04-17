#!/usr/bin/env bash
# configure_db_access.sh -- Grant remote database access inside WebArena
# Verified containers after they have been (re)created.
#
# Container recreation (docker compose up -d) discards in-container
# config changes. This script re-applies:
#
#   1. Shopping / Shopping Admin: MySQL remote-user grant for magentouser@%
#   2. GitLab: PostgreSQL listen_addresses='*' + pg_hba allow-all rule,
#      then restart the postgresql service so it binds to TCP.
#
# Reddit and Map already accept TCP from any host via their base images,
# so no configuration is needed for those.
#
# Wikipedia has no database (Kiwix ZIM), so it is skipped.
#
# Usage (from the repo root, on your workstation):
#   ./scripts/configure_db_access.sh --host-config configs/benchmark_hosts/r5.yaml
#
#   # Optional overrides:
#   HOST_IP=1.2.3.4 SSH_KEY=~/.ssh/other-key.pem ./scripts/configure_db_access.sh
#   GITLAB_DB_ALLOWED_CIDRS=127.0.0.1/32,172.20.0.0/20 ./scripts/configure_db_access.sh

set -euo pipefail

HOST_CONFIG=""
HOST_IP="${HOST_IP:-}"
SSH_KEY_RAW="${SSH_KEY:-$HOME/.ssh/webarena-key.pem}"
SSH_KEY="${SSH_KEY_RAW/#\~/$HOME}"
SSH_USER="${SSH_USER:-ubuntu}"
GITLAB_DB_ALLOWED_CIDRS="${GITLAB_DB_ALLOWED_CIDRS:-127.0.0.1/32,172.20.0.0/20}"
WORLDSIM_TRUSTED_OPERATOR_CIDRS="${WORLDSIM_TRUSTED_OPERATOR_CIDRS:-}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --host-config)
            HOST_CONFIG="$2"
            shift 2
            ;;
        --host)
            HOST_IP="$2"
            shift 2
            ;;
        --ssh-key)
            SSH_KEY="$2"
            shift 2
            ;;
        --ssh-user)
            SSH_USER="$2"
            shift 2
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

if [[ -n "$HOST_CONFIG" ]]; then
    while IFS='=' read -r key quoted_value; do
        eval "value=$quoted_value"
        case "$key" in
            HOST_IP)
                [[ -n "$HOST_IP" ]] || HOST_IP="$value"
                ;;
            SSH_USER)
                [[ "$SSH_USER" != "ubuntu" ]] || SSH_USER="$value"
                ;;
            WORLDSIM_TRUSTED_OPERATOR_CIDRS)
                [[ -n "$WORLDSIM_TRUSTED_OPERATOR_CIDRS" ]] || WORLDSIM_TRUSTED_OPERATOR_CIDRS="$value"
                ;;
        esac
    done < <(uv run python "$REPO_ROOT/scripts/export_host_config_env.py" --host-config "$HOST_CONFIG")
fi

if [[ -z "$HOST_IP" ]]; then
    echo "ERROR: missing benchmark host. Pass --host-config or set HOST_IP/--host." >&2
    exit 1
fi
if [[ "$GITLAB_DB_ALLOWED_CIDRS" == "127.0.0.1/32,172.20.0.0/20" && -n "$WORLDSIM_TRUSTED_OPERATOR_CIDRS" ]]; then
    GITLAB_DB_ALLOWED_CIDRS="${GITLAB_DB_ALLOWED_CIDRS},${WORLDSIM_TRUSTED_OPERATOR_CIDRS}"
fi
SSH_KEY="${SSH_KEY/#\~/$HOME}"

SSH_OPTS=(
    -i "$SSH_KEY"
    -o StrictHostKeyChecking=accept-new
    -o ServerAliveInterval=30
    -o ConnectTimeout=15
)

log() { printf '\n==> %s\n' "$*"; }

ssh_host() { ssh "${SSH_OPTS[@]}" "$SSH_USER@$HOST_IP" "$@"; }

list_matching_containers() {
    local pattern="$1"
    ssh_host "docker ps --format '{{.Names}}' | grep -E \"$pattern\" || true"
}

for_each_container() {
    local pattern="$1"
    local callback="$2"
    local found=0
    while IFS= read -r container; do
        [[ -z "$container" ]] && continue
        found=1
        "$callback" "$container"
    done < <(list_matching_containers "$pattern")
    if [[ "$found" -eq 0 ]]; then
        echo "    WARN: no containers matched pattern $pattern"
    fi
}

host_port_for() {
    local container="$1"
    local container_port="$2"
    ssh_host "docker port $container ${container_port}/tcp 2>/dev/null | head -n1 | awk -F: '{print \$NF}'"
}

# ---------------------------------------------------------------------------
# MySQL remote-user grant (shopping + shopping_admin)
# ---------------------------------------------------------------------------
#
# The base image creates magentouser@localhost only. We insert a
# magentouser@% row with the same password hash and grant ALL on
# magentodb. The root password is 1234567890 (set by the image's
# docker-entrypoint.sh via MYSQL_ROOT_PASSWORD default).

MYSQL_GRANT_SQL='
INSERT IGNORE INTO global_priv (Host, User, Priv)
  VALUES ("%", "magentouser",
    "{\"access\":0,\"version_id\":100612,\"plugin\":\"mysql_native_password\",\"authentication_string\":\"*ACE88B25517D0F22E5C3C643944B027D56345D2D\",\"password_last_changed\":1681054041}");
INSERT IGNORE INTO db
  (Host, Db, User,
   Select_priv, Insert_priv, Update_priv, Delete_priv,
   Create_priv, Drop_priv, Grant_priv, References_priv,
   Index_priv, Alter_priv, Create_tmp_table_priv, Lock_tables_priv,
   Create_view_priv, Show_view_priv, Create_routine_priv, Alter_routine_priv,
   Execute_priv, Event_priv, Trigger_priv)
  VALUES ("%", "magentodb", "magentouser",
   "Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y","Y");
FLUSH PRIVILEGES;
'

grant_mysql_remote() {
    local container="$1"
    log "$container: granting magentouser@% remote access"
    ssh_host "docker exec $container bash -c 'mysql -u root -p1234567890 mysql -e \"$MYSQL_GRANT_SQL\"'" \
        && echo "    OK" \
        || echo "    WARN: grant failed (container may still be starting)"
}

# ---------------------------------------------------------------------------
# GitLab PostgreSQL: enable TCP listen + pg_hba allow-all
# ---------------------------------------------------------------------------

configure_gitlab_postgres() {
    local container="$1"
    log "$container: configuring PostgreSQL for TCP access"
    ssh_host "docker exec $container bash -lc '
ALLOWED_CIDRS=\"${GITLAB_DB_ALLOWED_CIDRS}\"
IFS=, read -ra cidrs <<< \"\$ALLOWED_CIDRS\"

# Set listen_addresses to all interfaces
sed -i \"s/listen_addresses = '\'''\''.*$/listen_addresses = '\''*'\''    # modified for external DB access/\" /var/opt/gitlab/postgresql/data/postgresql.conf

# Append scoped TCP host entries if not already present.
for cidr in \"\${cidrs[@]}\"; do
  cidr=\$(echo \"\$cidr\" | xargs)
  [[ -z \"\$cidr\" ]] && continue
  if ! grep -q \"[[:space:]]\$cidr[[:space:]]\" /var/opt/gitlab/postgresql/data/pg_hba.conf; then
    printf \"\\nhost    all    all    %s    trust\\n\" \"\$cidr\" >> /var/opt/gitlab/postgresql/data/pg_hba.conf
  fi
done

# Restart postgresql (listen_addresses requires full restart, not reload)
gitlab-ctl stop postgresql 2>&1
sleep 2
gitlab-ctl start postgresql 2>&1
sleep 3

# Verify TCP connectivity
su - gitlab-psql -s /bin/sh -c \"/opt/gitlab/embedded/bin/psql -h 127.0.0.1 -U gitlab -d gitlabhq_production -c \\\"SELECT 1 as test\\\"\" 2>&1
' " && echo "    OK" || echo "    WARN: GitLab PostgreSQL config failed"
}

# ---------------------------------------------------------------------------
# Verify all DB connections from within containers
# ---------------------------------------------------------------------------

verify_mysql_connection() {
    local container="$1"
    local host_port
    host_port="$(host_port_for "$container" 3306)"
    echo "  ${container} MySQL (${host_port:-unknown}):"
    ssh_host "docker exec $container bash -c 'mysql -u magentouser -pMyPassword -h 127.0.0.1 magentodb -e \"SELECT 1\" 2>&1'" \
        && echo "    OK" || echo "    FAIL"
}

verify_gitlab_connection() {
    local container="$1"
    local host_port
    host_port="$(host_port_for "$container" 5432)"
    echo "  ${container} PostgreSQL (${host_port:-unknown}):"
    ssh_host "docker exec $container bash -c 'su - gitlab-psql -s /bin/sh -c \"/opt/gitlab/embedded/bin/psql -h 127.0.0.1 -U gitlab -d gitlabhq_production -c \\\"SELECT 1\\\"\" 2>&1'" \
        && echo "    OK" || echo "    FAIL"
}

verify_postgres_connection() {
    local container="$1"
    local user="$2"
    local password="$3"
    local database="$4"
    local host_port
    host_port="$(host_port_for "$container" 5432)"
    echo "  ${container} PostgreSQL (${host_port:-unknown}):"
    ssh_host "docker exec $container bash -c 'PGPASSWORD=${password} psql -h 127.0.0.1 -U ${user} -d ${database} -c \"SELECT 1\" 2>&1'" \
        && echo "    OK" || echo "    FAIL"
}

verify_connections() {
    log "Verifying database connections via TCP"
    for_each_container '^webarena-verified-shopping(_[0-9]+)?$' verify_mysql_connection
    for_each_container '^webarena-verified-shopping_admin(_[0-9]+)?$' verify_mysql_connection
    for_each_container '^webarena-verified-gitlab(_[0-9]+)?$' verify_gitlab_connection
    while IFS= read -r container; do
        [[ -z "$container" ]] && continue
        verify_postgres_connection "$container" "postmill" "postmill" "postmill"
    done < <(list_matching_containers '^webarena-verified-reddit(_[0-9]+)?$')
    while IFS= read -r container; do
        [[ -z "$container" ]] && continue
        verify_postgres_connection "$container" "renderer" "renderer" "gis"
    done < <(list_matching_containers '^webarena-verified-map(_[0-9]+)?$')
}

print_host_ports() {
    log "Host DB port summary"
    while IFS= read -r container; do
        [[ -z "$container" ]] && continue
        local port
        if [[ "$container" =~ shopping ]] && [[ ! "$container" =~ gitlab|reddit|map ]]; then
            port="$(host_port_for "$container" 3306)"
        else
            port="$(host_port_for "$container" 5432)"
        fi
        printf '    %-32s %s:%s\n' "$container" "$HOST_IP" "${port:-unknown}"
    done < <(
        {
            list_matching_containers '^webarena-verified-shopping(_[0-9]+)?$'
            list_matching_containers '^webarena-verified-shopping_admin(_[0-9]+)?$'
            list_matching_containers '^webarena-verified-gitlab(_[0-9]+)?$'
            list_matching_containers '^webarena-verified-reddit(_[0-9]+)?$'
            list_matching_containers '^webarena-verified-map(_[0-9]+)?$'
        } | sort
    )
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main() {
    log "configure_db_access.sh -- post-recreate DB configuration"
    printf '    HOST_IP = %s\n' "$HOST_IP"
    printf '    GITLAB_DB_ALLOWED_CIDRS = %s\n' "$GITLAB_DB_ALLOWED_CIDRS"

    for_each_container '^webarena-verified-shopping(_[0-9]+)?$' grant_mysql_remote
    for_each_container '^webarena-verified-shopping_admin(_[0-9]+)?$' grant_mysql_remote
    for_each_container '^webarena-verified-gitlab(_[0-9]+)?$' configure_gitlab_postgres
    verify_connections
    print_host_ports

    echo ""
    log "Done."
}

main "$@"
