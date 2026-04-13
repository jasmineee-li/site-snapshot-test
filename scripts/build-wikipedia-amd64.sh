#!/usr/bin/env bash
# build-wikipedia-amd64.sh — Build an amd64 variant of the WebArena Verified
# Wikipedia container image on the EC2 host.
#
# Problem: am1n3e/webarena-verified-wikipedia:latest on Docker Hub is published
# as arm64-only (no amd64 variant; confirmed with `docker manifest inspect`).
# The other five am1n3e/webarena-verified-* images have working amd64 manifests.
# On our x86_64 EC2 host the arm64 image crash-loops with
#   "exec /entrypoint.sh: exec format error".
#
# Workaround: rebuild locally from the upstream Dockerfile against an amd64
# base, tag as worldsim/webarena-verified-wikipedia:amd64, and point the
# compose override at the local tag. Nothing is pushed anywhere.
#
# Usage (ON the EC2 host, like setup-map-robust.sh):
#   ./build-wikipedia-amd64.sh           # skip if tag already exists
#   ./build-wikipedia-amd64.sh --force   # rebuild unconditionally
#
# Idempotent: safe to run multiple times.

set -euo pipefail

# --- Config -----------------------------------------------------------------

IMAGE_TAG="worldsim/webarena-verified-wikipedia:amd64"
UPSTREAM_REPO="https://github.com/ServiceNow/webarena-verified"
UPSTREAM_DOCKERFILE_REL="dev/environments/docker/sites/wikipedia/Dockerfile"

# Where to put the build context when we need to clone upstream. Kept inside
# the script's directory so repeated runs reuse the same checkout.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/.build/webarena-verified"

FORCE=0
if [[ "${1:-}" == "--force" ]]; then
    FORCE=1
fi

# --- Sanity checks ----------------------------------------------------------

# This script is intended to run on the x86_64 EC2 host. The point of the
# whole exercise is to produce an amd64 image, so building on arm64 would
# defeat the purpose (and require emulation we don't want to depend on).
ARCH="$(uname -m)"
if [[ "$ARCH" != "x86_64" && "$ARCH" != "amd64" ]]; then
    echo "ERROR: expected to run on x86_64, got $ARCH."
    echo "This script is meant to run on the EC2 host. Aborting."
    exit 1
fi

if ! command -v docker &>/dev/null; then
    echo "ERROR: docker not found on PATH."
    exit 1
fi

if ! docker buildx version &>/dev/null; then
    echo "ERROR: docker buildx not available. Install a recent Docker Engine."
    exit 1
fi

# --- Idempotency check ------------------------------------------------------

if [[ "$FORCE" != "1" ]] && docker image inspect "$IMAGE_TAG" &>/dev/null; then
    echo "==> $IMAGE_TAG already exists locally. Skipping build."
    echo "    (Pass --force to rebuild.)"
    exit 0
fi

# --- Locate build context ---------------------------------------------------
# The Dockerfile references paths from the repo root (packages/..., dev/...),
# so the build context must be the webarena-verified repo root.
#
# Preference order:
#   1. A sibling checkout at ../vendors/webarena-verified relative to this
#      script (useful when running on a dev box).
#   2. $HOME/webarena-verified (common EC2 layout).
#   3. Fresh clone into $BUILD_DIR.

CONTEXT_DIR=""
for candidate in \
    "$SCRIPT_DIR/../vendors/webarena-verified" \
    "$HOME/webarena-verified" \
    "$BUILD_DIR"; do
    if [[ -f "$candidate/$UPSTREAM_DOCKERFILE_REL" ]]; then
        CONTEXT_DIR="$candidate"
        break
    fi
done

if [[ -z "$CONTEXT_DIR" ]]; then
    echo "==> No local checkout found. Cloning $UPSTREAM_REPO into $BUILD_DIR"
    mkdir -p "$(dirname "$BUILD_DIR")"
    git clone --depth=1 "$UPSTREAM_REPO" "$BUILD_DIR"
    CONTEXT_DIR="$BUILD_DIR"
elif [[ "$CONTEXT_DIR" == "$BUILD_DIR" ]]; then
    echo "==> Refreshing existing checkout at $BUILD_DIR"
    git -C "$BUILD_DIR" fetch --depth=1 origin main >/dev/null 2>&1 || true
    git -C "$BUILD_DIR" reset --hard origin/main >/dev/null 2>&1 || true
fi

if [[ ! -f "$CONTEXT_DIR/$UPSTREAM_DOCKERFILE_REL" ]]; then
    echo "ERROR: Dockerfile not found at $CONTEXT_DIR/$UPSTREAM_DOCKERFILE_REL"
    echo "Repo layout may have changed upstream."
    exit 1
fi

echo "==> Using build context: $CONTEXT_DIR"
echo "==> Using Dockerfile:    $UPSTREAM_DOCKERFILE_REL"

# --- Build ------------------------------------------------------------------
# The upstream Dockerfile FROMs ghcr.io/kiwix/kiwix-serve:3.3.0 which is
# published multi-arch; forcing --platform linux/amd64 selects the correct
# base layer. We --load instead of --push so the image stays local.

echo "==> Building $IMAGE_TAG (linux/amd64)"
docker buildx build \
    --platform linux/amd64 \
    --file "$CONTEXT_DIR/$UPSTREAM_DOCKERFILE_REL" \
    --tag "$IMAGE_TAG" \
    --load \
    "$CONTEXT_DIR"

# --- Self-test --------------------------------------------------------------
# Confirm the image actually runs on this host (no "exec format error").
# We run a trivial `true`; the whole point is that the shell starts at all.

echo "==> Self-test: running /bin/sh -c true inside $IMAGE_TAG"
if ! docker run --rm --entrypoint /bin/sh "$IMAGE_TAG" -c "true"; then
    echo "ERROR: self-test failed. The built image did not execute."
    echo "       This usually means the wrong platform landed in the manifest."
    exit 1
fi

# Also verify architecture reported by the image matches amd64.
IMAGE_ARCH="$(docker image inspect --format '{{.Architecture}}' "$IMAGE_TAG")"
if [[ "$IMAGE_ARCH" != "amd64" ]]; then
    echo "ERROR: built image reports Architecture=$IMAGE_ARCH, expected amd64."
    exit 1
fi

echo ""
echo "==> Done. Built $IMAGE_TAG (Architecture=$IMAGE_ARCH)."
echo ""
echo "Next steps:"
echo "  1. Ensure scripts/webarena-compose-override.yml has the wikipedia"
echo "     service override pointing at $IMAGE_TAG."
echo "  2. Redeploy the stack:"
echo "       cd ~/webarena-verified || cd ~"
echo "       docker compose up -d wikipedia"
