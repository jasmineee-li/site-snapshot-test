#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/sync_warp_taskgen_snapshot.sh [--source PATH] [--dry-run]
                                        [--allow-dirty-source]
                                        [--allow-dirty-target]
                                        TARGET_REPO

Sync the current WARP Taskgen authoring checkout into a main-repo checkout at:

  TARGET_REPO/packages/warp-taskgen

Typical flow:

  # From the authoring checkout, after committing your work:
  ./scripts/sync_warp_taskgen_snapshot.sh ../browser-sim-main

  # Then in ../browser-sim-main:
  git switch -c codex/sync-taskgen-YYYYMMDD
  git add packages/warp-taskgen
  git commit -m "chore(taskgen): sync snapshot from feat/worldsim-v5"
  git push -u origin HEAD
  gh pr create --base main

Options:
  --source PATH          Source checkout to copy from. Defaults to the repo
                         that contains this script.
  --dry-run             Show rsync actions without writing to TARGET_REPO.
  --allow-dirty-source  Do not fail when SOURCE has tracked changes. The sync
                         still snapshots HEAD, not uncommitted edits.
  --allow-dirty-target  Do not fail when TARGET_REPO has tracked changes.
  -h, --help            Show this help.
EOF
}

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source_root="$(cd "$script_dir/.." && pwd)"
target_root=""
dry_run=0
allow_dirty_source=0
allow_dirty_target=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            [[ $# -ge 2 ]] || die "--source requires a path"
            source_root="$(cd "$2" && pwd)"
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --allow-dirty-source)
            allow_dirty_source=1
            shift
            ;;
        --allow-dirty-target)
            allow_dirty_target=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --*)
            die "unknown option: $1"
            ;;
        *)
            [[ -z "$target_root" ]] || die "only one TARGET_REPO may be supplied"
            target_root="$(cd "$1" && pwd)"
            shift
            ;;
    esac
done

[[ -n "$target_root" ]] || {
    usage >&2
    exit 2
}

git -C "$source_root" rev-parse --show-toplevel >/dev/null 2>&1 || die "SOURCE is not a git checkout: $source_root"
git -C "$target_root" rev-parse --show-toplevel >/dev/null 2>&1 || die "TARGET_REPO is not a git checkout: $target_root"

source_root="$(git -C "$source_root" rev-parse --show-toplevel)"
target_root="$(git -C "$target_root" rev-parse --show-toplevel)"
dest_root="$target_root/packages/warp-taskgen"

[[ "$source_root" != "$target_root" ]] || die "SOURCE and TARGET_REPO must be different checkouts"

if [[ "$allow_dirty_source" -eq 0 ]] && [[ -n "$(git -C "$source_root" status --porcelain --untracked-files=no)" ]]; then
    die "SOURCE has tracked changes; commit or pass --allow-dirty-source"
fi

if [[ "$allow_dirty_target" -eq 0 ]] && [[ -n "$(git -C "$target_root" status --porcelain --untracked-files=no)" ]]; then
    die "TARGET_REPO has tracked changes; commit/stash or pass --allow-dirty-target"
fi

source_branch="$(git -C "$source_root" branch --show-current || true)"
source_branch="${source_branch:-detached}"
source_sha="$(git -C "$source_root" rev-parse HEAD)"
source_short_sha="$(git -C "$source_root" rev-parse --short=12 HEAD)"
source_remote="$(git -C "$source_root" config --get remote.origin.url || true)"
synced_at="$(TZ=UTC date '+%Y-%m-%dT%H:%M:%SZ')"

if [[ "$allow_dirty_source" -eq 1 ]] && [[ -n "$(git -C "$source_root" status --porcelain --untracked-files=no)" ]]; then
    printf 'warning: SOURCE has tracked changes; this sync snapshots HEAD and ignores uncommitted edits\n' >&2
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/warp-taskgen-snapshot.XXXXXX")"
trap 'rm -rf "$tmp_dir"' EXIT

git -C "$source_root" archive --format=tar HEAD | tar -xf - -C "$tmp_dir"
mkdir -p "$dest_root"

rsync_args=(
    -a
    --delete
    --exclude '/.venv/'
    --exclude '.venv/'
    --exclude '/venv/'
    --exclude 'venv/'
    --exclude '/env/'
    --exclude 'env/'
    --exclude '/.uv-cache/'
    --exclude '.uv-cache/'
    --exclude '/.pytest_cache/'
    --exclude '.pytest_cache/'
    --exclude '/.ruff_cache/'
    --exclude '.ruff_cache/'
    --exclude '/.mypy_cache/'
    --exclude '.mypy_cache/'
    --exclude '/.cache/'
    --exclude '.cache/'
    --exclude '__pycache__/'
    --exclude '*.pyc'
    --exclude '*.pyo'
    --exclude '/build/'
    --exclude 'build/'
    --exclude '/dist/'
    --exclude 'dist/'
    --exclude '/*.egg-info/'
    --exclude '*.egg-info/'
    --exclude '/logs/'
    --exclude '/pipeline_outputs/'
    --exclude '/vendors/'
    --exclude '/.codex-worktrees/'
    --exclude '/.cursor/'
    --exclude '/.gradio/'
    --exclude '/.claude/settings.local.json'
    --exclude '/.claude/local.md'
    --exclude '/.claude/worktrees/'
    --exclude '/.env'
    --exclude '/.env.local'
    --exclude '/.env.*.local'
    --exclude '/.benchmark_host_id'
    --exclude '/.benchmark_proxy_metadata'
    --exclude '/.benchmark_proxy_ports.conf'
    --exclude '/.proxy_token'
    --exclude '/aws-credentials'
    --exclude '/aws-credentials.*'
    --exclude '/instances.json'
    --exclude '/instances.*.local.json'
    --exclude '/instances.scale.json'
    --exclude '/instances.scale.json.fragment'
    --exclude '/instances.smoke.json'
    --exclude '/instances.smoke.json.fragment'
    --exclude '/compose.scale.yml'
    --exclude '/compose.smoke.yml'
    --exclude '/scripts/docker-compose.scale.yml'
    --exclude '/scripts/docker-compose.smoke.yml'
    --exclude '/scripts/proxy_ports.conf'
    --exclude '/packages/warp-taskgen/'
)

if [[ "$dry_run" -eq 1 ]]; then
    rsync_args+=(--dry-run --itemize-changes)
fi

rsync "${rsync_args[@]}" "$tmp_dir/" "$dest_root/"

if [[ "$dry_run" -eq 0 ]]; then
    cat > "$dest_root/SNAPSHOT.md" <<EOF
# WARP Taskgen Snapshot

This package subtree is a committed snapshot of the WARP Taskgen authoring
checkout. It is not a git submodule and does not update automatically.

- Source remote: ${source_remote:-unknown}
- Source branch: ${source_branch}
- Source commit: ${source_sha}
- Synced at: ${synced_at}
- Destination: \`packages/warp-taskgen\`

Refresh with:

\`\`\`bash
scripts/sync_warp_taskgen_snapshot.sh /path/to/main-checkout
\`\`\`
EOF
fi

printf '\nSynced WARP Taskgen snapshot\n'
printf '  source: %s (%s@%s)\n' "$source_root" "$source_branch" "$source_short_sha"
printf '  target: %s\n' "$dest_root"

if [[ "$dry_run" -eq 0 ]]; then
    printf '\nPost-sync checks:\n'
    git -C "$target_root" diff --check -- packages/warp-taskgen
    git -C "$target_root" status --short -- packages/warp-taskgen
fi

printf '\nSuggested commit message:\n'
printf '  chore(taskgen): sync snapshot from %s@%s\n' "$source_branch" "$source_short_sha"
