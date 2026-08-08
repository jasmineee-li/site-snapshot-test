#!/usr/bin/env bash
# The root-facing, fresh-checkout acceptance boundary for WARP Taskgen.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_DIR="$ROOT_DIR/packages/warp-taskgen"
ROUTER="$PACKAGE_DIR/scripts/taskgen_acceptance_router.py"

usage() {
    cat <<'EOF'
Usage: scripts/accept_taskgen.sh

Run the canonical WARP Taskgen acceptance boundary. In GitHub Actions, the
wrapper skips dependency and package work when the pull request has no changes
to the canonical package, this command, or its workflow. A local invocation
always runs unless TASKGEN_ACCEPTANCE_CHANGED_FILES is supplied.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
if [[ $# -ne 0 ]]; then
    usage >&2
    exit 2
fi

route_known=0
route_args=()

if [[ "${TASKGEN_ACCEPTANCE_FORCE:-0}" == "1" ]]; then
    route_known=0
elif [[ -n "${TASKGEN_ACCEPTANCE_CHANGED_FILES+x}" ]]; then
    route_known=1
    while IFS= read -r path; do
        [[ -n "$path" ]] && route_args+=(--path "$path")
    done <<< "$TASKGEN_ACCEPTANCE_CHANGED_FILES"
elif [[ -n "${GITHUB_BASE_REF:-}" ]]; then
    base_ref="${GITHUB_BASE_SHA:-origin/$GITHUB_BASE_REF}"
    if git -C "$ROOT_DIR" rev-parse --verify "$base_ref" >/dev/null 2>&1; then
        route_known=1
        while IFS= read -r path; do
            [[ -n "$path" ]] && route_args+=(--path "$path")
        done < <(git -C "$ROOT_DIR" diff --name-only "$base_ref...HEAD")
    else
        printf 'Taskgen acceptance: unable to resolve %s; running full acceptance\n' "$base_ref"
    fi
fi

if [[ "$route_known" -eq 1 ]]; then
    decision="$(python3 "$ROUTER" "${route_args[@]}")"
    if [[ "$decision" == "skip" ]]; then
        printf 'Taskgen acceptance: skip (no canonical Taskgen changes)\n'
        exit 0
    fi
fi

printf 'Taskgen acceptance: run (locked sync, package verification, wheel smoke)\n'
cd "$ROOT_DIR"

uv sync --directory "$PACKAGE_DIR" --extra dev --locked
bash "$PACKAGE_DIR/scripts/verify_default.sh"

temporary_root="$(mktemp -d "${TMPDIR:-/tmp}/warp-taskgen-acceptance.XXXXXX")"
trap 'rm -rf "$temporary_root"' EXIT
build_dir="$temporary_root/dist"
isolated_env="$temporary_root/venv"
mkdir -p "$build_dir"

uv build "$PACKAGE_DIR" --out-dir "$build_dir"

shopt -s nullglob
wheels=("$build_dir"/*.whl)
shopt -u nullglob
if [[ "${#wheels[@]}" -ne 1 ]]; then
    printf 'error: expected exactly one wheel in %s, found %s\n' "$build_dir" "${#wheels[@]}" >&2
    exit 1
fi

python_version="$(<"$PACKAGE_DIR/.python-version")"
uv venv --python "$python_version" "$isolated_env"
uv pip install --python "$isolated_env/bin/python" "${wheels[0]}"
PYTHON_DOTENV_DISABLED=1 "$isolated_env/bin/warp-taskgen" --help

printf 'Taskgen acceptance: passed\n'
