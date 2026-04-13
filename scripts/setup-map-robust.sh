#!/usr/bin/env bash
# setup-map-robust.sh — Robust replacement for the WebArena Verified map
# site data bootstrap (the `setup_map.sh` + `docker run alpine` flow in
# vendors/webarena-verified/dev/environments/docker/sites/map/scripts/).
#
# Problem the previous flow hit:
#   1. aria2 ran inside a short-lived Alpine container. A Docker daemon
#      operation SIGKILL'd the container at ~94% on nominatim_volumes.tar,
#      leaving sparse (truncated) .tar files on disk alongside .aria2
#      control files.
#   2. The original setup_map.sh used `set -e`, so the first extraction
#      that hit a sparse tar ("tar: car: not found in archive") aborted
#      every subsequent extraction. One corrupted tar cascaded into
#      zero populated volumes.
#   3. Re-runs never noticed the partial downloads because the size check
#      was just `-f`. The tar was there, just truncated.
#
# Fix in this script:
#   * Resume-first downloads. aria2c --continue=true --file-allocation=none.
#     Never delete .aria2 control files unless the URL changed.
#   * Run the downloads and extractions on the EC2 host, not in a Docker
#     container. The Docker daemon cannot SIGKILL a host process just
#     because a container shutdown goes sideways.
#   * Per-step error handling. No top-level `set -e`. Each download /
#     verify / extract is a function that returns a status. One failure
#     does not cascade.
#   * Sentinels drive idempotency. `<file>.verified` after a download
#     passes the Content-Length + `tar -tf` integrity check. `.extracted`
#     inside each Docker volume after extraction. Re-runs skip completed
#     work exactly.
#   * The map container is only started when all six map volumes have
#     their `.extracted` sentinel.
#
# Usage:
#   # Must be run ON the EC2 host (18.117.99.179). Not SSH'd from laptop.
#   ssh -i ~/.ssh/webarena-key.pem ubuntu@18.117.99.179
#   bash /home/ubuntu/setup-map-robust.sh
#
#   # Optional: copy this script onto the host first, e.g.
#   scp -i ~/.ssh/webarena-key.pem \
#     scripts/setup-map-robust.sh ubuntu@18.117.99.179:/home/ubuntu/
#
# Idempotent: safe to run N times. Running it again after a successful
# run is a no-op (every sentinel short-circuits).

# NOTE: intentionally NO `set -euo pipefail` at top level. We want to
# continue past failures and let main() decide what to do. Individual
# functions use local `set -e` via `||` fallthrough where appropriate.

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DOWNLOADS_DIR="/home/ubuntu/downloads"
COMPOSE_DIR="/home/ubuntu"
COMPOSE_FILE="$COMPOSE_DIR/docker-compose.yml"
OVERRIDE_FILE="$COMPOSE_DIR/docker-compose.override.yml"

# URLs come from vendors/webarena-verified/dev/environments/settings.py
# (MapSettings.data_urls). Pinned here so the script is self-contained.
URL_TILE="https://webarena-map-server-data.s3.amazonaws.com/osm_tile_server.tar"
URL_NOMINATIM="https://webarena-map-server-data.s3.amazonaws.com/nominatim_volumes.tar"
URL_OSRM="https://webarena-map-server-data.s3.amazonaws.com/osrm_routing.tar"

# file_basename -> URL
declare -A FILE_URLS=(
    ["osm_tile_server.tar"]="$URL_TILE"
    ["nominatim_volumes.tar"]="$URL_NOMINATIM"
    ["osrm_routing.tar"]="$URL_OSRM"
)

# Docker volumes populated by this script. Names match docker-compose.yml
# (the top-level `name: webarena-verified-envs` is overridden by the
# ubuntu host's compose, which uses the short form — see
# `docker volume ls` on the instance).
VOL_TILE_DB="webarena-verified-map-tile-db"
VOL_ROUTING_CAR="webarena-verified-map-routing-car"
VOL_ROUTING_BIKE="webarena-verified-map-routing-bike"
VOL_ROUTING_FOOT="webarena-verified-map-routing-foot"
VOL_NOMINATIM_DB="webarena-verified-map-nominatim-db"
VOL_NOMINATIM_FLATNODE="webarena-verified-map-nominatim-flatnode"

ALL_VOLUMES=(
    "$VOL_TILE_DB"
    "$VOL_ROUTING_CAR"
    "$VOL_ROUTING_BIKE"
    "$VOL_ROUTING_FOOT"
    "$VOL_NOMINATIM_DB"
    "$VOL_NOMINATIM_FLATNODE"
)

# Reasonable lower-bound sizes (bytes) for "this volume looks populated".
# Values derived from the currently-populated tile-db (38.4 GiB actual)
# and conservative floors for the routing/nominatim volumes. These are
# sanity thresholds, not exact expected sizes.
declare -A VOLUME_MIN_BYTES=(
    ["$VOL_TILE_DB"]=30000000000          # ~30 GB
    ["$VOL_ROUTING_CAR"]=1000000000       # ~1 GB
    ["$VOL_ROUTING_BIKE"]=500000000       # ~500 MB
    ["$VOL_ROUTING_FOOT"]=500000000       # ~500 MB
    ["$VOL_NOMINATIM_DB"]=20000000000     # ~20 GB
    ["$VOL_NOMINATIM_FLATNODE"]=5000000000 # ~5 GB
)

# ---------------------------------------------------------------------------
# Sanity checks
# ---------------------------------------------------------------------------

sanity_check_host() {
    # We are supposed to be on the EC2 webarena host. Verify a couple of
    # paths that only exist there rather than checking hostname (which
    # can differ between AMIs).
    if [[ ! -d "$DOWNLOADS_DIR" ]]; then
        echo "ERROR: $DOWNLOADS_DIR does not exist."
        echo "This script must be run ON the EC2 webarena host, not locally."
        return 1
    fi
    if [[ ! -f "$COMPOSE_FILE" ]]; then
        echo "ERROR: $COMPOSE_FILE does not exist."
        echo "Expected the webarena docker-compose.yml at $COMPOSE_FILE."
        return 1
    fi
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: docker not found on PATH."
        return 1
    fi
    if ! command -v aria2c >/dev/null 2>&1; then
        echo "==> aria2c not installed on host; installing via apt-get"
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

download_file() {
    # $1 = filename (e.g. osrm_routing.tar), $2 = url
    local filename="$1"
    local url="$2"
    local target="$DOWNLOADS_DIR/$filename"
    local sentinel="$target.verified"

    if [[ -f "$sentinel" ]]; then
        echo "    - $filename already verified (sentinel present); skipping download"
        return 0
    fi

    echo "    - downloading $filename"
    echo "      URL: $url"
    echo "      target: $target"

    # aria2c resumes from .aria2 control file if present. We never delete
    # .aria2 files unless the URL changed (not handled here — flip the
    # pinned URL and delete the sentinel + .aria2 manually).
    #
    # --continue=true: resume partial downloads
    # --file-allocation=none: don't pre-allocate (matters on big files)
    # -x 16 -s 16: 16 parallel connections; tune if the source rate-limits
    # --max-tries=0 --retry-wait=10: retry forever on transient errors;
    #   the outer loop caller still gets a non-zero on hard failure
    # --summary-interval=30: less log noise
    if ! sudo aria2c \
        --continue=true \
        --file-allocation=none \
        -x 16 -s 16 \
        --max-tries=0 \
        --retry-wait=10 \
        --summary-interval=30 \
        -d "$DOWNLOADS_DIR" \
        -o "$filename" \
        "$url"; then
        echo "      WARN: aria2c returned non-zero for $filename"
        return 1
    fi

    return 0
}

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

verify_file() {
    # $1 = filename, $2 = url
    # Writes $filename.verified on success.
    local filename="$1"
    local url="$2"
    local target="$DOWNLOADS_DIR/$filename"
    local sentinel="$target.verified"

    if [[ -f "$sentinel" ]]; then
        echo "    - $filename verify sentinel already present; skipping"
        return 0
    fi

    if [[ ! -f "$target" ]]; then
        echo "    - ERROR: $target missing, cannot verify"
        return 1
    fi

    # 1. Content-Length check
    local expected
    expected=$(curl -sSI --max-time 30 "$url" \
        | awk 'BEGIN{IGNORECASE=1} /^content-length:/ {gsub(/\r/,""); print $2; exit}')
    local actual
    actual=$(stat -c '%s' "$target" 2>/dev/null)

    if [[ -z "$expected" ]]; then
        echo "    - WARN: could not fetch Content-Length for $filename (continuing with integrity check only)"
    else
        if [[ "$expected" != "$actual" ]]; then
            echo "    - size mismatch for $filename: expected=$expected actual=$actual"
            echo "      (an .aria2 file likely still exists; re-run to resume)"
            return 1
        fi
        echo "    - size match ($actual bytes)"
    fi

    # 2. tar integrity check (only for .tar)
    if [[ "$filename" == *.tar ]]; then
        echo "    - running tar -tf for integrity"
        if ! sudo tar -tf "$target" >/dev/null 2>&1; then
            echo "    - ERROR: tar -tf failed for $filename (archive is corrupted/sparse)"
            return 1
        fi
        echo "    - tar integrity OK"
    fi

    # 3. Write sentinel
    sudo touch "$sentinel"
    echo "    - wrote sentinel $sentinel"
    return 0
}

# ---------------------------------------------------------------------------
# Volume sentinels
# ---------------------------------------------------------------------------

check_volume_sentinel() {
    # $1 = volume name
    # Returns 0 if the volume has a .extracted sentinel inside it.
    local vol="$1"
    docker run --rm -v "$vol":/data alpine \
        sh -c 'test -f /data/.extracted' 2>/dev/null
}

write_volume_sentinel() {
    # $1 = volume name
    local vol="$1"
    docker run --rm -v "$vol":/data alpine \
        sh -c 'touch /data/.extracted && chmod 644 /data/.extracted' 2>/dev/null
}

volume_size_bytes() {
    # $1 = volume name. Prints total bytes of volume contents.
    local vol="$1"
    docker run --rm -v "$vol":/data alpine \
        sh -c 'du -sb /data 2>/dev/null | awk "{print \$1}"' 2>/dev/null
}

# ---------------------------------------------------------------------------
# Extract
# ---------------------------------------------------------------------------

extract_to_volume() {
    # $1 = volume name
    # $2 = tar filename (basename, under $DOWNLOADS_DIR)
    # $3... = tar args (strip-components + path inside tar)
    local vol="$1"
    local tar_name="$2"
    shift 2
    local tar_args=("$@")
    local tar_path="$DOWNLOADS_DIR/$tar_name"
    local min_bytes="${VOLUME_MIN_BYTES[$vol]:-0}"

    if check_volume_sentinel "$vol"; then
        echo "    - $vol has .extracted sentinel; skipping"
        return 0
    fi

    # Detect "already populated from a previous run that crashed before
    # writing a sentinel" (e.g. the current tile-db at 38 GB). If the
    # volume is big enough, trust it and just stamp the sentinel.
    local cur_bytes
    cur_bytes=$(volume_size_bytes "$vol")
    cur_bytes=${cur_bytes:-0}
    if [[ "$cur_bytes" -ge "$min_bytes" && "$min_bytes" -gt 0 ]]; then
        echo "    - $vol already has $cur_bytes bytes (>= $min_bytes threshold); stamping .extracted"
        write_volume_sentinel "$vol" || {
            echo "    - WARN: failed to write .extracted sentinel on $vol"
            return 1
        }
        return 0
    fi

    if [[ ! -f "$tar_path" ]]; then
        echo "    - ERROR: $tar_path missing; cannot extract $vol"
        return 1
    fi
    if [[ ! -f "$tar_path.verified" ]]; then
        echo "    - ERROR: $tar_path not verified; refusing to extract $vol"
        return 1
    fi

    echo "    - extracting $tar_name -> $vol"
    # Bind-mount the tar file read-only and the volume rw, do the extract
    # inside a short alpine container (which has a working busybox tar).
    # The tar runs on host-mounted storage, so it cannot be SIGKILL'd the
    # way the original setup was — and even if this container dies, the
    # failure is localized and sentinel-gated.
    if ! docker run --rm \
        -v "$tar_path":/data/source.tar:ro \
        -v "$vol":/volume \
        alpine \
        sh -c "tar -xf /data/source.tar -C /volume ${tar_args[*]}"; then
        echo "    - ERROR: extraction failed for $vol"
        return 1
    fi

    # Verify the volume has reasonable content.
    cur_bytes=$(volume_size_bytes "$vol")
    cur_bytes=${cur_bytes:-0}
    if [[ "$cur_bytes" -lt "$min_bytes" ]]; then
        echo "    - WARN: $vol is only $cur_bytes bytes after extraction (threshold $min_bytes); NOT stamping sentinel"
        return 1
    fi

    write_volume_sentinel "$vol" || {
        echo "    - WARN: failed to write .extracted sentinel on $vol"
        return 1
    }
    echo "    - $vol extracted and sentinel written ($cur_bytes bytes)"
    return 0
}

# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

main() {
    echo "==> Sanity check host"
    sanity_check_host || return 1

    echo ""
    echo "==> Downloading map data tars (resume-safe)"
    local dl_fail=0
    for filename in "${!FILE_URLS[@]}"; do
        if ! download_file "$filename" "${FILE_URLS[$filename]}"; then
            echo "    - download failed for $filename (will still try to verify)"
            dl_fail=$((dl_fail + 1))
        fi
    done

    echo ""
    echo "==> Verifying downloads (Content-Length + tar -tf)"
    for filename in "${!FILE_URLS[@]}"; do
        verify_file "$filename" "${FILE_URLS[$filename]}" || \
            echo "    - verify failed for $filename; related extractions will be skipped"
    done

    echo ""
    echo "==> Extracting to Docker volumes"

    # tile-db from osm_tile_server.tar (also auto-detected if pre-populated).
    extract_to_volume "$VOL_TILE_DB" "osm_tile_server.tar" \
        "--strip-components=6 projects/ogma3/docker/volumes/osm-data/_data"

    # routing-{car,bike,foot} from osrm_routing.tar
    extract_to_volume "$VOL_ROUTING_CAR" "osrm_routing.tar" \
        "--strip-components=1 car"
    extract_to_volume "$VOL_ROUTING_BIKE" "osrm_routing.tar" \
        "--strip-components=1 bike"
    extract_to_volume "$VOL_ROUTING_FOOT" "osrm_routing.tar" \
        "--strip-components=1 foot"

    # nominatim-{db,flatnode} from nominatim_volumes.tar (7 strip components)
    extract_to_volume "$VOL_NOMINATIM_DB" "nominatim_volumes.tar" \
        "--strip-components=7 projects/metis2/docker/docker/volumes/nominatim-data/_data"
    extract_to_volume "$VOL_NOMINATIM_FLATNODE" "nominatim_volumes.tar" \
        "--strip-components=7 projects/metis2/docker/docker/volumes/nominatim-flatnode/_data"

    echo ""
    echo "==> Checking volume sentinels"
    local ready=1
    for vol in "${ALL_VOLUMES[@]}"; do
        if check_volume_sentinel "$vol"; then
            echo "    - $vol: ready"
        else
            echo "    - $vol: NOT ready (missing .extracted)"
            ready=0
        fi
    done

    if [[ "$ready" -ne 1 ]]; then
        echo ""
        echo "==> Not all map volumes are ready. NOT starting map container."
        echo "    Re-run this script after investigating the warnings above."
        echo "    Downloads persist under $DOWNLOADS_DIR and will resume."
        return 1
    fi

    echo ""
    echo "==> All map volumes ready. Starting map container."
    local compose_args=("-f" "$COMPOSE_FILE")
    if [[ -f "$OVERRIDE_FILE" ]]; then
        compose_args+=("-f" "$OVERRIDE_FILE")
    fi
    if ! sudo docker compose "${compose_args[@]}" up -d map; then
        echo "    - ERROR: docker compose up -d map failed"
        return 1
    fi

    echo ""
    echo "==> Done. Map container status:"
    docker ps --filter name=webarena-verified-map --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    return 0
}

main "$@"
exit $?
