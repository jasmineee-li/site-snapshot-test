#!/usr/bin/env bash
# setup-wikipedia-robust.sh — Robust replacement for the vendor
# setup_wikipedia.sh flow. Downloads the WebArena Verified wikipedia ZIM
# file to /home/ubuntu/downloads/, verifies size + ZIM magic bytes, then
# copies (replace-in-place) the file into the docker-compose-prefixed
# volume `webarena-verified-envs_webarena-verified-wikipedia-data` if the
# existing volume copy is missing, wrong size, or corrupt.
#
# Why this exists:
#
#   * The vendor setup_wikipedia.sh only pulls from the CMU mirror
#     (http://metis.lti.cs.cmu.edu/webarena-images/...), which throttles
#     to 2-5 MiB/s. At 95 GiB that's hours of wall clock. archive.org's
#     mirror is the same file (Content-Length 95199730590) and can be
#     fetched in parallel for ~2-3x throughput.
#   * The vendor script doesn't handle the "volume already contains a
#     stale ZIM" case. If a prior download was corrupt or truncated,
#     re-running the vendor script does nothing because its test is just
#     "does the file exist in the volume".
#   * Running on the host (not in an alpine container) means a Docker
#     daemon hiccup can't SIGKILL the downloader mid-flight.
#
# Fix:
#
#   1. Resume-first download via aria2c. Both CMU and archive.org URLs
#      are passed on the same command line so aria2 parallelizes across
#      mirrors.
#   2. Verify Content-Length (95199730590) + ZIM magic bytes ("ZIM\x04" at
#      start of file). Only stamp the .verified sentinel on success.
#   3. Replace-in-place into the volume via a short alpine container:
#         cp <zim> <vol>/<name>.new && mv <vol>/<name>.new <vol>/<name>
#      The atomic rename guarantees the file is never half-written under
#      a running wikipedia container.
#   4. Start the wikipedia service via docker compose. If it's already
#      running and the ZIM didn't change, do nothing. If the ZIM did
#      change, restart wikipedia so kiwix re-opens the file.
#
# Idempotent: safe to run N times.
#
# Usage (ON the EC2 host):
#   bash /home/ubuntu/setup-wikipedia-robust.sh
#
# Prerequisites on EC2 host:
#   * /home/ubuntu/docker-compose.yml with `wikipedia` service defined
#   * /home/ubuntu/docker-compose.override.yml (copied by bootstrap_ec2.sh)
#     which pins wikipedia to worldsim/webarena-verified-wikipedia:amd64
#     (built by build-wikipedia-amd64.sh)

# NOTE: intentionally NO `set -euo pipefail` at top level; per-step
# failures should not cascade. The main() function returns non-zero if
# the overall flow didn't reach a running wikipedia container.

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DOWNLOADS_DIR="/home/ubuntu/downloads"
COMPOSE_DIR="/home/ubuntu"
COMPOSE_FILE="$COMPOSE_DIR/docker-compose.yml"
OVERRIDE_FILE="$COMPOSE_DIR/docker-compose.override.yml"

ZIM_NAME="wikipedia_en_all_maxi_2022-05.zim"
ZIM_PATH="$DOWNLOADS_DIR/$ZIM_NAME"
ZIM_SENTINEL="$ZIM_PATH.verified"
ZIM_EXPECTED_BYTES=95199730590

# Docker Compose prefixes volumes with the project name when the
# top-level `name:` is set. The wikipedia volume is left prefixed
# (unlike the map volumes) because the vendor compose file declares
# wikipedia-data without a `name:` override. Actual name on the host
# is `webarena-verified-envs_webarena-verified-wikipedia-data`.
WIKI_VOLUME="webarena-verified-envs_webarena-verified-wikipedia-data"

# Mirrors. aria2 can consume both on one command line and will pull
# from them in parallel, rebalancing across mirrors if one is slow.
URL_CMU="http://metis.lti.cs.cmu.edu/webarena-images/$ZIM_NAME"
URL_ARCHIVE_ORG="https://archive.org/download/wikipedia_en_all_maxi_2022-05/$ZIM_NAME"

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

sanity_check_host() {
    if [[ ! -d "$DOWNLOADS_DIR" ]]; then
        echo "==> $DOWNLOADS_DIR missing; creating"
        mkdir -p "$DOWNLOADS_DIR"
    fi
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        echo "ERROR: $COMPOSE_FILE missing. This script must run on the EC2"
        echo "webarena host after docker-compose.yml has been placed."
        return 1
    fi
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: docker not found on PATH."
        return 1
    fi
    if ! command -v aria2c >/dev/null 2>&1; then
        echo "==> aria2c not installed; installing via apt-get"
        sudo apt-get update -y >/dev/null 2>&1
        sudo apt-get install -y aria2 || {
            echo "ERROR: could not install aria2; please install manually"
            return 1
        }
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

download_zim() {
    if [[ -f "$ZIM_SENTINEL" ]]; then
        echo "    - $ZIM_NAME already verified (sentinel present); skipping download"
        return 0
    fi

    echo "    - downloading $ZIM_NAME via aria2 (parallel mirrors)"
    echo "      CMU:         $URL_CMU"
    echo "      archive.org: $URL_ARCHIVE_ORG"
    echo "      target:      $ZIM_PATH"

    # -x 16 -s 16: 16 connections total, 16 per-server max
    # --split 16: split file into 16 chunks
    # --min-split-size 100M: keep chunks big enough that CMU's rate
    #   limit doesn't dominate wallclock time
    # --continue=true --file-allocation=none: resume on re-run
    # --max-tries=0 --retry-wait=10: retry transient errors forever
    # Multiple URLs on one command line: aria2 parallelizes across mirrors.
    if ! sudo aria2c \
        --continue=true \
        --file-allocation=none \
        -x 16 -s 16 \
        --split 16 \
        --min-split-size=100M \
        --max-tries=0 \
        --retry-wait=10 \
        --summary-interval=30 \
        -d "$DOWNLOADS_DIR" \
        -o "$ZIM_NAME" \
        "$URL_CMU" "$URL_ARCHIVE_ORG"; then
        echo "      WARN: aria2c returned non-zero for $ZIM_NAME"
        return 1
    fi

    return 0
}

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

verify_zim() {
    if [[ -f "$ZIM_SENTINEL" ]]; then
        echo "    - verify sentinel already present; skipping"
        return 0
    fi

    if [[ ! -f "$ZIM_PATH" ]]; then
        echo "    - ERROR: $ZIM_PATH missing"
        return 1
    fi

    local actual
    actual=$(stat -c '%s' "$ZIM_PATH" 2>/dev/null)

    if [[ "$actual" != "$ZIM_EXPECTED_BYTES" ]]; then
        echo "    - size mismatch: expected=$ZIM_EXPECTED_BYTES actual=$actual"
        echo "      (an .aria2 control file likely still exists; re-run to resume)"
        return 1
    fi
    echo "    - size OK ($actual bytes)"

    # ZIM magic: first 4 bytes are 'Z' 'I' 'M' 0x04. We read the first
    # four bytes with dd and compare via hex. Pure bash / coreutils — no
    # Python required.
    local magic_hex
    magic_hex=$(sudo dd if="$ZIM_PATH" bs=1 count=4 status=none 2>/dev/null | od -An -tx1 | tr -d ' \n')
    if [[ "$magic_hex" != "5a494d04" ]]; then
        echo "    - ERROR: ZIM magic bytes mismatch (got 0x$magic_hex, expected 0x5a494d04 = 'ZIM\\x04')"
        return 1
    fi
    echo "    - ZIM magic bytes OK (0x$magic_hex)"

    sudo touch "$ZIM_SENTINEL"
    echo "    - wrote sentinel $ZIM_SENTINEL"
    return 0
}

# ---------------------------------------------------------------------------
# Volume-resident copy check
# ---------------------------------------------------------------------------

volume_zim_size() {
    # Prints the size (bytes) of the ZIM inside the wiki volume, or nothing
    # if the file is absent.
    docker run --rm -v "$WIKI_VOLUME":/dst alpine \
        sh -c "[ -f /dst/$ZIM_NAME ] && stat -c '%s' /dst/$ZIM_NAME" 2>/dev/null
}

volume_zim_magic_hex() {
    # Prints the first 4 bytes of the volume-resident ZIM as lowercase
    # hex, or nothing if the file is absent.
    docker run --rm -v "$WIKI_VOLUME":/dst alpine \
        sh -c "[ -f /dst/$ZIM_NAME ] && dd if=/dst/$ZIM_NAME bs=1 count=4 2>/dev/null | od -An -tx1 | tr -d ' \n'" 2>/dev/null
}

ensure_volume_exists() {
    if docker volume inspect "$WIKI_VOLUME" >/dev/null 2>&1; then
        return 0
    fi
    echo "    - volume $WIKI_VOLUME missing; creating"
    docker volume create "$WIKI_VOLUME" >/dev/null
}

needs_volume_replace() {
    # Returns 0 (true) if the volume is missing the ZIM, or if its size /
    # magic bytes don't match the host-resident verified ZIM.
    local vol_size vol_magic
    vol_size=$(volume_zim_size)
    if [[ -z "$vol_size" ]]; then
        echo "    - volume does not contain $ZIM_NAME"
        return 0
    fi
    if [[ "$vol_size" != "$ZIM_EXPECTED_BYTES" ]]; then
        echo "    - volume ZIM size mismatch: expected=$ZIM_EXPECTED_BYTES actual=$vol_size"
        return 0
    fi
    vol_magic=$(volume_zim_magic_hex)
    if [[ "$vol_magic" != "5a494d04" ]]; then
        echo "    - volume ZIM magic mismatch (got 0x$vol_magic); replacing"
        return 0
    fi
    return 1
}

replace_zim_in_volume() {
    # Copy $ZIM_PATH into the volume atomically. The intermediate .new
    # guarantees a running wikipedia container never sees a half-written
    # file. See docstring at top of script for full recipe.
    echo "    - replacing $ZIM_NAME in $WIKI_VOLUME"
    if ! docker run --rm \
        -v "$DOWNLOADS_DIR":/src:ro \
        -v "$WIKI_VOLUME":/dst \
        alpine \
        sh -c "cp /src/$ZIM_NAME /dst/$ZIM_NAME.new && mv /dst/$ZIM_NAME.new /dst/$ZIM_NAME"; then
        echo "    - ERROR: cp+mv into volume failed"
        return 1
    fi
    echo "    - replaced (atomic rename)"
    return 0
}

# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------

start_wikipedia() {
    # $1 = 1 if ZIM was replaced (force restart), 0 otherwise
    local replaced="$1"
    local compose_args=("-f" "$COMPOSE_FILE")
    if [[ -f "$OVERRIDE_FILE" ]]; then
        compose_args+=("-f" "$OVERRIDE_FILE")
    fi

    if [[ "$replaced" -eq 1 ]]; then
        echo "    - ZIM replaced; restarting wikipedia"
        sudo docker compose "${compose_args[@]}" stop wikipedia 2>/dev/null || true
        if ! sudo docker compose "${compose_args[@]}" up -d wikipedia; then
            echo "    - ERROR: docker compose up -d wikipedia failed"
            return 1
        fi
    else
        # Only bring up if not running.
        local running
        running=$(docker inspect --format='{{.State.Running}}' webarena-verified-wikipedia 2>/dev/null || true)
        if [[ "$running" == "true" ]]; then
            echo "    - wikipedia already running; not restarting"
        else
            if ! sudo docker compose "${compose_args[@]}" up -d wikipedia; then
                echo "    - ERROR: docker compose up -d wikipedia failed"
                return 1
            fi
        fi
    fi
    return 0
}

# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

main() {
    echo "==> Sanity check host"
    sanity_check_host || return 1

    echo ""
    echo "==> Download wiki ZIM (resume-safe, parallel mirrors)"
    download_zim || {
        echo "    - download failed; continuing to verify what we have"
    }

    echo ""
    echo "==> Verify wiki ZIM (size + magic bytes)"
    if ! verify_zim; then
        echo "    - wiki ZIM verification failed; NOT touching volume. Re-run to resume."
        return 1
    fi

    echo ""
    echo "==> Ensure wiki volume exists"
    ensure_volume_exists

    echo ""
    echo "==> Compare volume-resident ZIM vs host-resident ZIM"
    local replaced=0
    if needs_volume_replace; then
        echo "    - replacement required"
        if ! replace_zim_in_volume; then
            echo "    - volume replacement failed"
            return 1
        fi
        replaced=1
    else
        echo "    - volume is already up to date"
    fi

    echo ""
    echo "==> Start wikipedia container"
    start_wikipedia "$replaced" || return 1

    echo ""
    echo "==> Done. Wikipedia container status:"
    docker ps --filter name=webarena-verified-wikipedia \
        --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    return 0
}

main "$@"
exit $?
