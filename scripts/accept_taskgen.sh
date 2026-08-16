#!/usr/bin/env bash
# The root-facing, fresh-checkout acceptance boundary for WARP Taskgen.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PACKAGE_DIR="$ROOT_DIR/packages/warp-taskgen"
ROUTER="$PACKAGE_DIR/scripts/taskgen_acceptance_router.py"

LANE_FULL="full"
LANE_PACKAGE_PROOF="package-proof"
LANE_CORE_TESTS="core-tests"
LANE_REMOTE_JOB_TESTS="remote-job-tests"

usage() {
    cat <<'EOF'
Usage: scripts/accept_taskgen.sh [--lane LANE] [--route-only]

Run the canonical WARP Taskgen acceptance boundary. With no lane, run the
complete local gate. In GitHub Actions, each lane skips dependency and package
work when the pull request has no changes to the canonical package, this
command, or its workflow. A local invocation always runs unless
TASKGEN_ACCEPTANCE_CHANGED_FILES is supplied.

Use --route-only to print exactly "run" or "skip" without installing or
running anything. CI uses this before Python and uv setup.

Lanes:
  full             Complete local gate (the default).
  package-proof    Ruff, readiness, wheel build, and installed CLI smoke.
  core-tests       Pytest excluding tests/test_remote_job_scripts.py.
  remote-job-tests Pytest for tests/test_remote_job_scripts.py.
EOF
}

lane="$LANE_FULL"
route_only=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h)
            usage
            exit 0
            ;;
        --lane)
            if [[ $# -lt 2 ]]; then
                printf 'error: --lane requires a value\n' >&2
                usage >&2
                exit 2
            fi
            lane="$2"
            shift 2
            ;;
        --route-only)
            route_only=1
            shift
            ;;
        *)
            printf 'error: unknown argument %s\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

case "$lane" in
    "$LANE_FULL"|"$LANE_PACKAGE_PROOF"|"$LANE_CORE_TESTS"|"$LANE_REMOTE_JOB_TESTS")
        ;;
    *)
        printf 'error: unknown acceptance lane %s\n' "$lane" >&2
        usage >&2
        exit 2
        ;;
esac

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
        printf 'Taskgen acceptance: unable to resolve %s; running full acceptance\n' "$base_ref" >&2
    fi
fi

if [[ "$route_known" -eq 1 ]]; then
    decision="$(python3 "$ROUTER" "${route_args[@]}")"
    if [[ "$decision" == "skip" ]]; then
        if [[ "$route_only" -eq 1 ]]; then
            printf 'skip\n'
            exit 0
        fi
        printf 'Taskgen acceptance: skip (no canonical Taskgen changes)\n'
        exit 0
    fi
fi

if [[ "$route_only" -eq 1 ]]; then
    printf 'run\n'
    exit 0
fi

if [[ "$lane" == "$LANE_FULL" ]]; then
    printf 'Taskgen acceptance: run (locked sync, package verification, wheel smoke)\n'
else
    printf 'Taskgen acceptance: run lane %s (locked sync)\n' "$lane"
fi
cd "$ROOT_DIR"

uv sync --directory "$PACKAGE_DIR" --extra dev --locked

build_and_smoke_package() {
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
    PYTHON_DOTENV_DISABLED=1 "$isolated_env/bin/python" -m warp_taskgen.main --help
    PYTHON_DOTENV_DISABLED=1 "$isolated_env/bin/warp-taskgen" \
        site doctor gitlab --benchmark webarena_verified \
        --use-case phase_2_feasibility --json
    PYTHON_DOTENV_DISABLED=1 "$isolated_env/bin/warp-taskgen" \
        site doctor classifieds --benchmark visualwebarena \
        --use-case ugc_reply --json
    PYTHON_DOTENV_DISABLED=1 "$isolated_env/bin/python" -c \
        "import importlib.resources; from warp_taskgen.site_composition import default_site_definitions; assert {item.site for item in default_site_definitions()} == {'classifieds', 'gitlab', 'reddit'}; assert importlib.resources.files('warp_taskgen').joinpath('site_composition.py').is_file()"
    PYTHON_DOTENV_DISABLED=1 "$isolated_env/bin/python" \
        "$PACKAGE_DIR/scripts/compatibility_wheel_matrix.py" \
        --python "$isolated_env/bin/python" --package-root "$PACKAGE_DIR"

    shopt -s nullglob
    sdists=("$build_dir"/*.tar.gz)
    shopt -u nullglob
    if [[ "${#sdists[@]}" -ne 1 ]]; then
        printf 'error: expected exactly one sdist in %s, found %s\n' \
            "$build_dir" "${#sdists[@]}" >&2
        exit 1
    fi
    sdist_env="$temporary_root/sdist-venv"
    uv venv --python "$python_version" "$sdist_env"
    uv pip install --python "$sdist_env/bin/python" "${sdists[0]}"
    PYTHON_DOTENV_DISABLED=1 "$sdist_env/bin/warp-taskgen" \
        site doctor reddit --benchmark webarena_verified \
        --use-case phase_2_feasibility --json
    PYTHON_DOTENV_DISABLED=1 "$sdist_env/bin/warp-taskgen" \
        site doctor classifieds --benchmark visualwebarena \
        --use-case ugc_reply --json

    sidecar_build_dir="$temporary_root/sidecar-dist"
    mkdir -p "$sidecar_build_dir"
    uv build "$PACKAGE_DIR/packages/worldsim-agentlab-runner" --out-dir "$sidecar_build_dir"
    shopt -s nullglob
    sidecar_wheels=("$sidecar_build_dir"/*.whl)
    shopt -u nullglob
    if [[ "${#sidecar_wheels[@]}" -ne 1 ]]; then
        printf 'error: expected exactly one AgentLab sidecar wheel in %s, found %s\n' \
            "$sidecar_build_dir" "${#sidecar_wheels[@]}" >&2
        exit 1
    fi
    # The sidecar's --help path is intentionally dependency-light; install
    # without dependencies so this smoke checks both historical console names
    # without starting AgentLab or a browser.
    uv pip install --python "$isolated_env/bin/python" --no-deps "${sidecar_wheels[0]}"
    PYTHON_DOTENV_DISABLED=1 "$isolated_env/bin/warp-taskgen-agentlab-runner" --help
    PYTHON_DOTENV_DISABLED=1 "$isolated_env/bin/worldsim-agentlab-runner" --help
    PYTHON_DOTENV_DISABLED=1 "$isolated_env/bin/python" -c "import worldsim_agentlab_runner.sync_pvpo; import warp_taskgen.phase_4.pvpo_capture"

    # Prove the real upgrade path, not only a fresh install. Version 0.1.0
    # shipped the temporary worldsim package and console. Ordinary pip must
    # recognize 0.1.1 as newer, uninstall the old wheel's RECORD entries, and
    # leave only the canonical package without requiring --force-reinstall.
    adapter_fixture="$PACKAGE_DIR/tests/fixtures/namespace_compatibility/adapter_wheel_0_1_0"
    adapter_build_dir="$temporary_root/adapter-dist"
    upgrade_env="$temporary_root/upgrade-venv"
    mkdir -p "$adapter_build_dir"
    uv build --wheel "$adapter_fixture" --out-dir "$adapter_build_dir"
    shopt -s nullglob
    adapter_wheels=("$adapter_build_dir"/*.whl)
    shopt -u nullglob
    if [[ "${#adapter_wheels[@]}" -ne 1 ]]; then
        printf 'error: expected exactly one adapter fixture wheel in %s, found %s\n' \
            "$adapter_build_dir" "${#adapter_wheels[@]}" >&2
        exit 1
    fi
    uv venv --seed --python "$python_version" "$upgrade_env"
    "$upgrade_env/bin/python" -m pip install --no-deps "${adapter_wheels[0]}"
    "$upgrade_env/bin/python" -c \
        "import importlib.util, pathlib, worldsim; assert importlib.util.find_spec('worldsim') is not None; assert pathlib.Path('$upgrade_env/bin/worldsim').is_file()"
    "$upgrade_env/bin/python" -m pip install --upgrade --no-deps "${wheels[0]}"
    "$upgrade_env/bin/python" -c \
        "import importlib.metadata, importlib.resources, importlib.util, pathlib, warp_taskgen; assert importlib.metadata.version('warp-taskgen') == '0.1.1'; assert warp_taskgen.__version__ == '0.1.1'; assert importlib.util.find_spec('worldsim') is None; assert not pathlib.Path('$upgrade_env/bin/worldsim').exists(); assert importlib.resources.files('warp_taskgen').joinpath('prompts/profile-site.md').is_file()"
}

run_package_proof() {
    bash "$PACKAGE_DIR/scripts/verify_fast.sh" --skip-collect
    build_and_smoke_package
}

case "$lane" in
    "$LANE_FULL")
        bash "$PACKAGE_DIR/scripts/verify_default.sh"
        build_and_smoke_package
        ;;
    "$LANE_PACKAGE_PROOF")
        run_package_proof
        ;;
    "$LANE_CORE_TESTS")
        (
            cd "$PACKAGE_DIR"
            "$PACKAGE_DIR/scripts/lib/run_silent.sh" \
                "core pytest parallel" \
                "uv run pytest -q -n 4 --dist worksteal --ignore tests/test_remote_job_scripts.py --ignore tests/test_remote_job_decisions.py"
        )
        ;;
    "$LANE_REMOTE_JOB_TESTS")
        (
            cd "$PACKAGE_DIR"
            "$PACKAGE_DIR/scripts/lib/run_silent.sh" \
                "remote-job pytest parallel" \
                "uv run pytest -q -n 4 --dist load tests/test_remote_job_scripts.py tests/test_remote_job_decisions.py"
        )
        ;;
esac

printf 'Taskgen acceptance: passed\n'
