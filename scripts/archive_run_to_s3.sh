#!/usr/bin/env bash
# archive_run_to_s3.sh — archive a Phase 4 rigor run directory to S3.
#
# Pattern follows the AWS-recommended two-pass discipline:
#   1. sync to S3 with STANDARD_IA + SHA256 checksum
#   2. verify with a dryrun (must report zero `upload:` lines)
#   3. delete local copy only with explicit --delete-local
#
# Source of truth for archive locations: s3://benchmark-archives/worldsim-runs/<run_id>/
#
# Idempotent: aws s3 sync skips files already at the destination with matching
# size + mtime. Safe to re-run after an interruption — it resumes from where it
# left off without re-uploading completed files.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEFAULT_BUCKET="benchmark-archives"
DEFAULT_PREFIX="worldsim-runs"
DEFAULT_REGION="us-east-2"
DEFAULT_STORAGE_CLASS="STANDARD_IA"

RUN_ID=""
LOGS_DIR="$REPO_ROOT/logs"
BUCKET="$DEFAULT_BUCKET"
PREFIX="$DEFAULT_PREFIX"
REGION="$DEFAULT_REGION"
STORAGE_CLASS="$DEFAULT_STORAGE_CLASS"
DELETE_LOCAL=0
SKIP_VERIFY=0
DRYRUN=0

usage() {
    cat <<'USAGE'
archive_run_to_s3.sh <run_id> [options]

Archive a Phase 4 rigor run from logs/<run_id>/ to
s3://benchmark-archives/worldsim-runs/<run_id>/ as paper evidence.

Required positional:
  <run_id>                directory name under logs/ to archive

Options:
  --logs-dir <path>       default: <repo_root>/logs
  --bucket <name>         default: benchmark-archives
  --prefix <prefix>       default: worldsim-runs
  --region <region>       default: us-east-2
  --storage-class <cls>   default: STANDARD_IA
                          one of STANDARD, STANDARD_IA, GLACIER_IR, GLACIER, DEEP_ARCHIVE
  --delete-local          remove logs/<run_id>/ after verified upload (default: keep)
  --skip-verify           skip the post-sync dryrun verification (NOT recommended)
  --dryrun                preview what would be uploaded without writing
  -h, --help              this help

Examples:
  scripts/archive_run_to_s3.sh agentlab_native_browser_50task_w48_gpt52_priority_rerun_20260508
  scripts/archive_run_to_s3.sh <run_id> --delete-local
  scripts/archive_run_to_s3.sh <run_id> --dryrun     # see what would happen
USAGE
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
log() { printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }

while (($#)); do
    case "$1" in
        --logs-dir) LOGS_DIR="$2"; shift 2 ;;
        --bucket) BUCKET="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --storage-class) STORAGE_CLASS="$2"; shift 2 ;;
        --delete-local) DELETE_LOCAL=1; shift ;;
        --skip-verify) SKIP_VERIFY=1; shift ;;
        --dryrun) DRYRUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        -*) die "unknown option: $1" ;;
        *)
            if [[ -z "$RUN_ID" ]]; then
                RUN_ID="$1"
            else
                die "unexpected positional arg: $1 (run_id already set to '$RUN_ID')"
            fi
            shift
            ;;
    esac
done

[[ -n "$RUN_ID" ]] || { usage >&2; die "<run_id> is required"; }
command -v aws >/dev/null 2>&1 || die "aws CLI not found on PATH"

# Resolve and validate the run directory.
RUN_DIR="$LOGS_DIR/$RUN_ID"
[[ -d "$RUN_DIR" ]] || die "run directory not found: $RUN_DIR"

S3_TARGET="s3://$BUCKET/$PREFIX/$RUN_ID/"

# Pre-flight: confirm bucket is reachable. Avoids burning operator time on a
# 30 GB sync that would fail at byte 0.
if ! aws s3 ls "s3://$BUCKET/" --region "$REGION" >/dev/null 2>&1; then
    die "cannot list s3://$BUCKET/ in $REGION (check IAM and bucket existence)"
fi

# Write a manifest so the archive is self-describing if rediscovered later.
# The manifest survives lifecycle transitions; tags do too but cost API calls
# to read at scale.
MANIFEST="$RUN_DIR/ARCHIVE_MANIFEST.json"
GIT_SHA="$(cd "$REPO_ROOT" && git rev-parse HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(cd "$REPO_ROOT" && git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
ARCHIVED_AT="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
HOSTNAME_VAL="$(hostname -f 2>/dev/null || hostname)"
FILE_COUNT="$(find "$RUN_DIR" -type f | wc -l | tr -d ' ')"
BYTE_COUNT="$(du -sb "$RUN_DIR" | awk '{print $1}')"

cat > "$MANIFEST" <<EOF_MANIFEST
{
  "run_id": "$RUN_ID",
  "archived_at": "$ARCHIVED_AT",
  "archived_from_hostname": "$HOSTNAME_VAL",
  "git_sha": "$GIT_SHA",
  "git_branch": "$GIT_BRANCH",
  "source_path": "$RUN_DIR",
  "destination": "$S3_TARGET",
  "storage_class": "$STORAGE_CLASS",
  "file_count_local": $FILE_COUNT,
  "byte_count_local_deduped": $BYTE_COUNT,
  "note": "file_count_local counts hardlinks once; S3 expands hardlinks into separate objects so S3 object count and total bytes will exceed local."
}
EOF_MANIFEST

log "archiving $RUN_ID"
log "  source: $RUN_DIR ($(du -sh "$RUN_DIR" | cut -f1), $FILE_COUNT files)"
log "  target: $S3_TARGET ($STORAGE_CLASS)"
log "  git:    $GIT_BRANCH @ $GIT_SHA"

# Sync. --only-show-errors keeps operator logs scannable; AWS CLI v2 attaches
# server-side checksums automatically.
SYNC_ARGS=(
    aws s3 sync "$RUN_DIR/" "$S3_TARGET"
    --storage-class "$STORAGE_CLASS"
    --region "$REGION"
    --only-show-errors
)
if [[ "$DRYRUN" -eq 1 ]]; then
    SYNC_ARGS+=(--dryrun)
    log "DRYRUN — no objects will be uploaded"
fi

log "starting sync"
"${SYNC_ARGS[@]}"
log "sync returned exit 0"

if [[ "$DRYRUN" -eq 1 ]]; then
    log "dryrun complete; no upload happened"
    exit 0
fi

# Verify: a second sync in --dryrun mode must report zero upload lines.
if [[ "$SKIP_VERIFY" -eq 1 ]]; then
    log "WARNING: skipping post-sync verification per --skip-verify"
else
    log "verifying with dryrun"
    MISSING="$(aws s3 sync "$RUN_DIR/" "$S3_TARGET" \
        --storage-class "$STORAGE_CLASS" \
        --region "$REGION" \
        --dryrun 2>&1 | grep -c "upload:" || true)"
    if [[ "$MISSING" != "0" ]]; then
        die "verification failed: $MISSING files would still need uploading. Re-run without --delete-local."
    fi
    log "verification ok: 0 files missing on S3"
fi

# Tag the manifest object specifically for queryability via tagging.
# Tags on every object would multiply API calls; the manifest is the index.
MANIFEST_KEY="$PREFIX/$RUN_ID/ARCHIVE_MANIFEST.json"
aws s3api put-object-tagging \
    --region "$REGION" \
    --bucket "$BUCKET" \
    --key "$MANIFEST_KEY" \
    --tagging "TagSet=[{Key=project,Value=warp-taskgen},{Key=run_id,Value=$RUN_ID},{Key=archived_at,Value=$ARCHIVED_AT},{Key=git_sha,Value=$GIT_SHA},{Key=purpose,Value=paper-evidence},{Key=storage_class,Value=$STORAGE_CLASS}]" \
    >/dev/null
log "tagged manifest object"

if [[ "$DELETE_LOCAL" -eq 1 ]]; then
    if [[ "$SKIP_VERIFY" -eq 1 ]]; then
        die "--delete-local requires verification; remove --skip-verify or re-run without --delete-local"
    fi
    log "deleting local copy: $RUN_DIR"
    rm -rf "$RUN_DIR"
    log "local copy removed"
else
    log "local copy kept at $RUN_DIR (pass --delete-local to remove)"
fi

log "done"
