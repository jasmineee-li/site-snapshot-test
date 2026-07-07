#!/usr/bin/env bash
# Idempotent installer for the safety benchmarks used by the eval-awareness pipeline.
#
# Clones (or updates) WASP, AgentDojo, InjecAgent, and EIA into ./external_benchmarks/,
# installs Python deps via uv where applicable, and unzips EIA's eval_results.zip for
# the import-only track. Does NOT set up Docker containers — see
# eval_awareness_experiments/SAFETY_BENCHMARKS_HANDOFF.md for the Docker prereqs.
#
# Usage:
#   bash eval_awareness_experiments/setup_benchmarks.sh            # all four
#   bash eval_awareness_experiments/setup_benchmarks.sh wasp eia   # subset
#
# Prereqs (checked at startup; script exits non-zero if missing):
#   - git, uv (this repo uses uv — see pyproject.toml + uv.lock)
#   - The repo venv exists (`uv sync` at least once). uv pip install targets this venv.
#   - OPENROUTER_API_KEY set in the environment or in ./.env (enforced by runners, not here)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT_DIR="${REPO_ROOT}/external_benchmarks"
mkdir -p "${EXT_DIR}"

log()  { printf '\033[1;34m[setup]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[setup]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[setup]\033[0m %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null 2>&1 || die "git not found on PATH"
command -v uv  >/dev/null 2>&1 || die "uv not found on PATH (install: https://github.com/astral-sh/uv)"

# Ensure uv can find a venv. `uv pip install` requires VIRTUAL_ENV or a project .venv.
if [ -z "${VIRTUAL_ENV:-}" ] && [ ! -d "${REPO_ROOT}/.venv" ]; then
  die "No active venv and no ${REPO_ROOT}/.venv. Run 'uv sync' in ${REPO_ROOT} first."
fi

log "uv: $(uv --help 2>&1 | head -1)"
log "Using venv: ${VIRTUAL_ENV:-${REPO_ROOT}/.venv}"

clone_or_update() {
  local url="$1" dest="$2"
  if [ -d "${dest}/.git" ]; then
    log "Updating $(basename "$dest") (pull)"
    git -C "$dest" pull --ff-only || warn "Pull failed for $dest — leaving as-is"
  else
    log "Cloning $url → $dest"
    git clone --depth=1 "$url" "$dest"
  fi
}

setup_wasp() {
  local dest="${EXT_DIR}/wasp"
  clone_or_update "https://github.com/facebookresearch/wasp" "$dest"
  if [ -f "${dest}/pyproject.toml" ] || [ -f "${dest}/setup.py" ]; then
    log "Installing WASP as editable package (uv pip)"
    (cd "$REPO_ROOT" && uv pip install -e "$dest")
  else
    warn "WASP has no installable package manifest; add ${dest} to PYTHONPATH manually if needed"
  fi
  log "WASP config dir: ${dest}/webarena_prompt_injections/configs (should contain *.json)"
}

setup_agentdojo() {
  local dest="${EXT_DIR}/agentdojo"
  # AgentDojo is on PyPI — prefer the pinned version for reproducibility.
  log "Installing agentdojo from PyPI (uv pip)"
  (cd "$REPO_ROOT" && uv pip install --upgrade 'agentdojo[transformers]') || {
    warn "PyPI install failed — falling back to source clone"
    clone_or_update "https://github.com/ethz-spylab/agentdojo" "$dest"
    (cd "$REPO_ROOT" && uv pip install -e "$dest")
  }
}

setup_injecagent() {
  local dest="${EXT_DIR}/injecagent"
  clone_or_update "https://github.com/uiuc-kang-lab/InjecAgent" "$dest"
  # InjecAgent ships a dataset + evaluator scripts; no package install.
  local data_dir="${dest}/data"
  [ -d "$data_dir" ] || die "InjecAgent data dir not found at $data_dir — unexpected repo layout"
  log "InjecAgent data dir: $data_dir"
  log "  test case files: $(ls "$data_dir" 2>/dev/null | grep -c '^test_cases_') found"
}

setup_eia() {
  local dest="${EXT_DIR}/eia"
  clone_or_update "https://github.com/OSU-NLP-Group/EIA_against_webagent" "$dest"
  local zip_path="${dest}/eval_results.zip"
  local extract_dir="${dest}/eval_results"
  if [ -f "$zip_path" ]; then
    if [ ! -d "$extract_dir" ]; then
      log "Unzipping ${zip_path}"
      (cd "$dest" && unzip -q eval_results.zip -d eval_results) || die "unzip failed"
    else
      log "eval_results/ already extracted — skipping unzip"
    fi
  else
    warn "EIA eval_results.zip not found at ${zip_path}."
    warn "  The zip is NOT checked into the repo — fetch it from the authors' Globus endpoint:"
    warn "    https://app.globus.org/file-manager?origin_id=6e0b9952-da25-4b74-a4f9-7450f0bb96b9&origin_path=%2F"
    warn "  (the README calls this the 'data.zip' folder; eval_results.zip lives alongside it.)"
    warn "  After downloading, place eval_results.zip at ${zip_path} and re-run this setup step,"
    warn "  or unzip manually to ${extract_dir}. import_eia_trajectories.py will fail until then."
  fi
  # Intentionally do NOT install SeeAct or any runtime — EIA is import-only.
}

# Dispatch
if [ $# -eq 0 ]; then
  targets=(wasp agentdojo injecagent eia)
else
  targets=("$@")
fi

for target in "${targets[@]}"; do
  case "$target" in
    wasp)       setup_wasp       ;;
    agentdojo)  setup_agentdojo  ;;
    injecagent) setup_injecagent ;;
    eia)        setup_eia        ;;
    *) die "Unknown benchmark '$target' (valid: wasp agentdojo injecagent eia)" ;;
  esac
done

log "Done. external_benchmarks/ contents:"
ls -la "${EXT_DIR}"
