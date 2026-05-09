#!/usr/bin/env bash
# Context-efficient command wrapper for agent sessions.
# Success emits one line. Failure emits the full captured output.

run_silent() {
    local description="$1"
    local command="$2"
    local tmp_file
    tmp_file=$(mktemp -t worldsim_run_silent.XXXXXX) || return 2

    if eval "$command" >"$tmp_file" 2>&1; then
        printf "  ✓ %s\n" "$description"
        rm -f "$tmp_file"
        return 0
    else
        local exit_code=$?
        printf "  ✗ %s\n" "$description"
        cat "$tmp_file"
        rm -f "$tmp_file"
        return "$exit_code"
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    if [[ $# -lt 2 ]]; then
        echo "usage: $0 <description> <command>" >&2
        exit 2
    fi

    description="$1"
    shift
    run_silent "$description" "$*"
fi
