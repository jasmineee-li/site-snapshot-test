#!/usr/bin/env bash
# regen_archive_index.sh — regenerate docs/runs/archive-index.md by querying
# S3 (canonical archive location) and the local HuggingFace dataset metadata
# (which run_dir each HF dataset row points at).
#
# The output file is gitignored; this script is the only way to refresh it.
#
# Usage:
#   scripts/regen_archive_index.sh
#   scripts/regen_archive_index.sh --bucket benchmark-archives --prefix worldsim-runs

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUCKET="benchmark-archives"
PREFIX="worldsim-runs"
REGION="us-east-2"
OUTPUT="$REPO_ROOT/docs/runs/archive-index.md"
HF_DIR="$REPO_ROOT/data/hf"

usage() {
    cat <<'USAGE'
regen_archive_index.sh — refresh docs/runs/archive-index.md from S3 + HF.

Options:
  --bucket <name>    default: benchmark-archives
  --prefix <prefix>  default: worldsim-runs
  --region <region>  default: us-east-2
  --output <path>    default: docs/runs/archive-index.md
  --hf-dir <path>    default: data/hf
  -h, --help         this help
USAGE
}

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }

while (($#)); do
    case "$1" in
        --bucket) BUCKET="$2"; shift 2 ;;
        --prefix) PREFIX="$2"; shift 2 ;;
        --region) REGION="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --hf-dir) HF_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) die "unknown arg: $1" ;;
    esac
done

command -v aws >/dev/null 2>&1 || die "aws CLI not found"
command -v python3 >/dev/null 2>&1 || die "python3 not found"

python3 - "$BUCKET" "$PREFIX" "$REGION" "$OUTPUT" "$HF_DIR" <<'PYEOF'
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime, timezone

bucket, prefix, region, output_path, hf_dir = sys.argv[1:6]
hf_dir = Path(hf_dir)
output_path = Path(output_path)

# 1. List all archived runs in S3 — one prefix per run_id.
result = subprocess.run(
    [
        "aws", "s3api", "list-objects-v2",
        "--bucket", bucket,
        "--prefix", f"{prefix}/",
        "--region", region,
        "--delimiter", "/",
        "--query", "CommonPrefixes[].Prefix",
        "--output", "json",
    ],
    capture_output=True, text=True, check=True,
)
common_prefixes = json.loads(result.stdout or "null") or []
run_ids = sorted(p.removeprefix(f"{prefix}/").rstrip("/") for p in common_prefixes if p.startswith(f"{prefix}/"))

# 2. For each run_id, try to fetch its ARCHIVE_MANIFEST.json (written by
#    archive_run_to_s3.sh) for richer metadata. Missing manifests are OK —
#    older archives predate the manifest format.
def fetch_manifest(run_id: str) -> dict | None:
    key = f"{prefix}/{run_id}/ARCHIVE_MANIFEST.json"
    proc = subprocess.run(
        ["aws", "s3", "cp", f"s3://{bucket}/{key}", "-", "--region", region],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None

manifests = {rid: fetch_manifest(rid) for rid in run_ids}

# 3. HF cross-reference: for each local HF dataset, read its metadata.json and
#    map (model_key, run_dir) entries.
hf_refs: dict[str, list[tuple[str, str]]] = {}
if hf_dir.exists():
    for ds in sorted(hf_dir.iterdir()):
        meta = ds / "metadata.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text())
        ds_id = m.get("dataset_id", ds.name)
        for run in m.get("runs", []):
            run_dir = run.get("run_dir", "")
            rid = Path(run_dir).name
            hf_refs.setdefault(rid, []).append((ds_id, run.get("model_key", "?")))

# 4. Render the index.
now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
lines = [
    "# Run Archive Index",
    "",
    "Live mapping of rigor run artifacts to where they actually live. **Gitignored**",
    f"because it tracks operational state that drifts between sessions.",
    "",
    f"Regenerate with `scripts/regen_archive_index.sh`. Last regenerated: {now}.",
    "",
    f"Source bucket: `s3://{bucket}/{prefix}/` ({region}).",
    "",
    "## Format",
    "",
    "Each entry shows the canonical S3 location, archive timestamp, originating",
    "git sha, and any HuggingFace dataset rows that reference this run_id.",
    "",
    "## Archived runs",
    "",
]

if not run_ids:
    lines.append("_(no archives in S3 yet)_")
else:
    for rid in run_ids:
        man = manifests.get(rid)
        hf = hf_refs.get(rid, [])
        lines.append(f"### `{rid}`")
        lines.append("")
        lines.append(f"- S3: `s3://{bucket}/{prefix}/{rid}/`")
        if man:
            lines.append(f"- Archived: {man.get('archived_at', '?')}")
            lines.append(f"- Git: `{man.get('git_branch', '?')}` @ `{man.get('git_sha', '?')[:12]}`")
            lines.append(f"- Storage class: {man.get('storage_class', '?')}")
            lines.append(f"- Local file count at archive time: {man.get('file_count_local', '?')}")
        else:
            lines.append("- (no ARCHIVE_MANIFEST.json — pre-manifest archive)")
        if hf:
            lines.append(f"- HuggingFace: {', '.join(f'{d} (model={m})' for d, m in hf)}")
        lines.append("")

# 5. Local HF datasets that have NO matching S3 archive — surface as gaps.
gaps = []
for rid, refs in hf_refs.items():
    if rid not in run_ids:
        gaps.append((rid, refs))
if gaps:
    lines.append("## Gaps")
    lines.append("")
    lines.append("HuggingFace datasets reference these run_ids, but no S3 archive exists.")
    lines.append("Sync via `scripts/archive_run_to_s3.sh <run_id>`.")
    lines.append("")
    for rid, refs in sorted(gaps):
        lines.append(f"- `{rid}` — HF: {', '.join(f'{d} (model={m})' for d, m in refs)}")
    lines.append("")

# 6. Static section: live AWS resources for r8a (operator handy reference).
lines.extend([
    "## AWS resources (live)",
    "",
    "| Resource | Purpose |",
    "|---|---|",
    "| EC2 `i-0bf197c9d4e41d500` | r8a benchmark host (r8a.24xlarge) |",
    "| EBS `vol-0feaec77fb454f4a1` | 1000 GiB gp3 root, DeleteOnTermination=false |",
    "| EIP `eipalloc-0dbaa9aa5f553f295` (3.16.230.234) | canonical r8a address |",
    "| Security group `sg-08792057943b27a65` | benchmark-proxy, managed SSH ingress via CFN |",
    "| CloudFormation stack `worldsim-r8a-control-plane` | EIP + operator SSH |",
    "| IAM role `worldsim-ec2-benchmark-backup` | instance role; inline `R8AControlPlaneAudit` + `S3BucketAccess` |",
    "",
])

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text("\n".join(lines) + "\n")
print(f"wrote {output_path}")
print(f"  {len(run_ids)} runs in S3")
print(f"  {sum(1 for m in manifests.values() if m)} runs have ARCHIVE_MANIFEST.json")
print(f"  {len(gaps)} HF runs missing from S3 (gaps)")
PYEOF
