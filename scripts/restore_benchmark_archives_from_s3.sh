#!/usr/bin/env bash
# restore_benchmark_archives_from_s3.sh - Restore WebArena setup artifacts
# from S3 cold storage to /home/ubuntu/downloads on EC2.
#
# These artifacts are the one-time setup tarballs and the Wikipedia ZIM file
# that were downloaded when we originally stood up the WebArena instances.
# They are not needed for running benchmarks against existing containers
# (the data lives in Docker volumes), but they are needed if you want to:
#   - Spin up a SECOND set of WebArena instances on another EC2 box
#   - Rebuild a container from scratch after its volumes are lost
#   - Scale horizontally (multiple instances of the same site, each needing
#     its own copy of the data seeded in)
#
# Bucket: s3://benchmark-archives/webarena/ (us-east-2, Standard-IA class)
# Size: ~265GB total
#   nominatim_volumes.tar       ~117GB  (OSM geocoding DB)
#   osm_tile_server.tar          ~39GB  (tile rendering DB)
#   osrm_routing.tar             ~20GB  (routing profiles car/foot/bike)
#   wikipedia_en_all_maxi_2022-05.zim  ~89GB  (Wikipedia Kiwix archive)
#
# Restore time from us-east-2 S3 to a us-east-2 EC2 is ~10-15 minutes
# (intra-region transfer is free and fast, typically 300-800 MB/s).
#
# Prerequisites on the EC2 box:
#   - awscli installed: `sudo apt install unzip && curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/aws.zip && cd /tmp && unzip -q aws.zip && sudo ./aws/install`
#   - AWS credentials in ~/.aws/credentials with read access to the bucket.
#     The scoped IAM user is `worldsim-ec2-benchmark-backup`.
#
# Usage:
#   ./scripts/restore_benchmark_archives_from_s3.sh [--wiki-only] [--skip-wiki]
#
# Or remotely:
#   ssh -i ~/.ssh/webarena-key.pem ubuntu@<host> 'bash -s' < scripts/restore_benchmark_archives_from_s3.sh

set -euo pipefail

BUCKET_PREFIX="s3://benchmark-archives/webarena"
DEST_DIR="/home/ubuntu/downloads"
CHECKSUM_FILE="/home/ubuntu/benchmark-archives-checksums.txt"
EMPTY_MAP_VOLUMES=(
    "webarena-verified-map-tiles"
    "webarena-verified-map-style"
    "webarena-verified-map-website-db"
)

# Parse flags
WIKI_ONLY=0
SKIP_WIKI=0
for arg in "$@"; do
    case "$arg" in
        --wiki-only) WIKI_ONLY=1 ;;
        --skip-wiki) SKIP_WIKI=1 ;;
    esac
done

mkdir -p "$DEST_DIR"

FILES=()
if [[ $WIKI_ONLY -eq 1 ]]; then
    FILES=("wikipedia_en_all_maxi_2022-05.zim")
elif [[ $SKIP_WIKI -eq 1 ]]; then
    FILES=(
        "nominatim_volumes.tar"
        "osm_tile_server.tar"
        "osrm_routing.tar"
    )
else
    FILES=(
        "nominatim_volumes.tar"
        "osm_tile_server.tar"
        "osrm_routing.tar"
        "wikipedia_en_all_maxi_2022-05.zim"
    )
fi

echo "=== Restoring ${#FILES[@]} file(s) from $BUCKET_PREFIX ==="
for f in "${FILES[@]}"; do
    if [[ -f "$DEST_DIR/$f" ]]; then
        echo "$f already present locally, skipping (delete to force re-download)"
        continue
    fi
    echo "Downloading $f..."
    aws s3 cp "$BUCKET_PREFIX/$f" "$DEST_DIR/$f" --only-show-errors
    echo "  $(ls -lh "$DEST_DIR/$f" | awk '{print $5, $9}')"
done

echo ""
echo "=== Verifying checksums (if $CHECKSUM_FILE present) ==="
if [[ -f "$CHECKSUM_FILE" ]]; then
    cd "$DEST_DIR"
    # Only verify files we actually downloaded this run.
    for f in "${FILES[@]}"; do
        if expected=$(grep "  $f\$" "$CHECKSUM_FILE" | awk '{print $1}'); then
            actual=$(sha256sum "$f" | awk '{print $1}')
            if [[ "$expected" == "$actual" ]]; then
                echo "  OK  $f"
            else
                echo "  FAIL  $f (expected $expected, got $actual)"
                exit 1
            fi
        else
            echo "  SKIP  $f (no reference checksum)"
        fi
    done
else
    echo "  No checksum file, skipping verification"
fi

if [[ $WIKI_ONLY -ne 1 ]]; then
    echo ""
    echo "=== Creating empty map volumes omitted from the archived tarballs ==="
    if command -v docker >/dev/null 2>&1; then
        for volume_name in "${EMPTY_MAP_VOLUMES[@]}"; do
            if docker volume inspect "$volume_name" >/dev/null 2>&1; then
                echo "  EXISTS  $volume_name"
            else
                docker volume create "$volume_name" >/dev/null
                echo "  CREATED $volume_name"
            fi
        done
    else
        echo "  SKIP  docker not found; create these later if needed:"
        for volume_name in "${EMPTY_MAP_VOLUMES[@]}"; do
            echo "    - $volume_name"
        done
    fi
fi

echo ""
echo "=== Done. ==="
echo "Files are in $DEST_DIR/. To seed a fresh container with one of these"
echo "tars, follow the docker volume hydration steps in scripts/bootstrap_ec2.sh."
