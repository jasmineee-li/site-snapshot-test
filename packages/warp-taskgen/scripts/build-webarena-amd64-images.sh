#!/usr/bin/env bash
# build-webarena-amd64-images.sh — Build amd64 WebArena images whose env-ctrl
# site implementations already include the WA_ENV_CTRL_EXTERNAL_SITE_URL
# fallback in git.
#
# This replaces the old bring-up path that patched running containers on every
# host boot. Active WASP scope is GitLab + Reddit/Postmill only, so we build
# local amd64 images only for those tags from the vendored webarena-verified
# source tree in this repo. The vendored site files already include the env-var
# fallback, so the fix survives `docker compose up --force-recreate`.
#
# Usage (ON the EC2 host):
#   ./build-webarena-amd64-images.sh
#   ./build-webarena-amd64-images.sh --force

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WEBARENA_SOURCE_DIR="${WEBARENA_SOURCE_DIR:-}"
FINGERPRINT_LABEL="warp_taskgen.source_fingerprint"

SITES=(
    "gitlab"
    "reddit"
)

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "x86_64" && "$ARCH" != "amd64" ]]; then
    echo "ERROR: expected to run on x86_64/amd64, got $ARCH"
    exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
    echo "ERROR: docker not found on PATH"
    exit 1
fi
if ! docker buildx version >/dev/null 2>&1; then
    echo "ERROR: docker buildx not available"
    exit 1
fi
CONTEXT_DIR=""
for candidate in \
    "$WEBARENA_SOURCE_DIR" \
    "$SCRIPT_DIR/vendors/webarena-verified" \
    "$SCRIPT_DIR/../vendors/webarena-verified" \
    "$HOME/vendors/webarena-verified" \
    "$HOME/webarena-verified"; do
    [[ -z "$candidate" ]] && continue
    if [[ -f "$candidate/dev/environments/docker/sites/gitlab/Dockerfile" ]]; then
        CONTEXT_DIR="$candidate"
        break
    fi
done

if [[ -z "$CONTEXT_DIR" ]]; then
    echo "ERROR: no pinned webarena-verified source checkout found."
    echo "Set WEBARENA_SOURCE_DIR or place the vendored checkout at:"
    echo "  $HOME/vendors/webarena-verified"
    echo "  $HOME/webarena-verified"
    exit 1
fi

echo "==> Using pinned source checkout at $CONTEXT_DIR"

compute_source_fingerprint() {
    python3 - "$@" <<'PY'
import hashlib
import pathlib
import sys

digest = hashlib.sha256()
for raw_path in sys.argv[1:]:
    path = pathlib.Path(raw_path)
    digest.update(path.read_bytes())
    digest.update(b"\0")
print(digest.hexdigest())
PY
}

for site in "${SITES[@]}"; do
    dockerfile="$CONTEXT_DIR/dev/environments/docker/sites/$site/Dockerfile"
    site_file="$CONTEXT_DIR/packages/environment_control/environment_control/ops/sites/${site}.py"
    tag="worldsim/webarena-verified-${site}:amd64"

    if [[ ! -f "$dockerfile" ]]; then
        echo "ERROR: missing Dockerfile for $site at $dockerfile"
        exit 1
    fi
    if [[ ! -f "$site_file" ]]; then
        echo "ERROR: missing site implementation for $site at $site_file"
        exit 1
    fi

    if ! grep -q 'WA_ENV_CTRL_EXTERNAL_SITE_URL' "$site_file"; then
        echo "ERROR: expected env-ctrl fallback missing from pinned source $site_file"
        exit 1
    fi
    source_fingerprint="$(compute_source_fingerprint "$dockerfile" "$site_file")"

    if [[ "$FORCE" != "1" ]] && docker image inspect "$tag" >/dev/null 2>&1; then
        image_fingerprint="$(docker image inspect --format "{{ index .Config.Labels \"$FINGERPRINT_LABEL\" }}" "$tag" 2>/dev/null || true)"
        if [[ "$image_fingerprint" == "$source_fingerprint" ]]; then
            echo "==> $tag already matches pinned source fingerprint. Skipping."
            continue
        fi
        echo "==> $tag exists but fingerprint differs; rebuilding."
    fi

    echo "==> Building $tag from $dockerfile"
    docker buildx build \
        --platform linux/amd64 \
        --file "$dockerfile" \
        --label "$FINGERPRINT_LABEL=$source_fingerprint" \
        --tag "$tag" \
        --load \
        "$CONTEXT_DIR"

    image_arch="$(docker image inspect --format '{{.Architecture}}' "$tag")"
    if [[ "$image_arch" != "amd64" ]]; then
        echo "ERROR: built image $tag reports Architecture=$image_arch, expected amd64"
        exit 1
    fi
done

echo ""
echo "==> Done. Local amd64 images are ready:"
for site in "${SITES[@]}"; do
    echo "    - worldsim/webarena-verified-${site}:amd64"
done
