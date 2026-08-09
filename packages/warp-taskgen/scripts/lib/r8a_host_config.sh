#!/usr/bin/env bash
# Shared r8a wrapper guard. Source this file; do not execute it directly.

r8a_resolve_host_config_path() {
    local repo_root="$1"
    local requested="$2"
    if [[ "$requested" != /* ]]; then
        requested="$repo_root/$requested"
    fi
    printf '%s\n' "$requested"
}

r8a_require_ignored_local_config() {
    local repo_root="$1"
    local host_config="$2"
    local config_rel

    [[ -f "$host_config" ]] || {
        printf 'ERROR: host config not found: %s\n' "$host_config" >&2
        return 2
    }

    if [[ "$host_config" == "$repo_root/configs/benchmark_hosts/r8a.yaml" ]]; then
        printf '%s\n' \
            "ERROR: r8a wrapper refuses the tracked public template; pass configs/benchmark_hosts/r8a.local.yaml" >&2
        return 2
    fi

    config_rel="${host_config#"$repo_root/"}"
    if [[ "$config_rel" == "$host_config" || "$config_rel" != *.local.yaml ]]; then
        printf '%s\n' \
            "ERROR: r8a wrappers require an ignored .local.yaml host config" >&2
        return 2
    fi

    command -v git >/dev/null 2>&1 || {
        printf 'ERROR: required command not found: git\n' >&2
        return 2
    }
    if ! git -C "$repo_root" check-ignore -q -- "$config_rel"; then
        printf 'ERROR: host config is not gitignored: %s\n' "$config_rel" >&2
        return 2
    fi
}
