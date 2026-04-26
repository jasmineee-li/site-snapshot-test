#!/usr/bin/env bash
# One-command setup for everything the causal-pilot needs.
#
# Idempotent: re-running is safe. Skips clones that already exist, skips
# installs of packages already in the venv, skips datasets that are already
# present.
#
# Uses uv (not pip) — the project's pyproject.toml + uv.lock are the source
# of truth. Installs editable clones into ./external_benchmarks/.
#
# After this script finishes successfully, ./eval_awareness_experiments/launch_pilot.sh
# can fire all 16 streams.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXT_DIR="${REPO_ROOT}/external_benchmarks"
mkdir -p "${EXT_DIR}"

bold()  { printf '\033[1;34m%s\033[0m\n' "$*"; }
ok()    { printf '\033[1;32m  ✓\033[0m %s\n' "$*"; }
warn()  { printf '\033[1;33m  !\033[0m %s\n' "$*" >&2; }
die()   { printf '\033[1;31m  ✗\033[0m %s\n' "$*" >&2; exit 1; }

command -v git >/dev/null 2>&1 || die "git not found on PATH"
command -v uv  >/dev/null 2>&1 || die "uv not found on PATH (https://github.com/astral-sh/uv)"
[ -d "${REPO_ROOT}/.venv" ] || die "No ${REPO_ROOT}/.venv. Run 'uv sync' in repo root first."

cd "$REPO_ROOT"

clone_or_skip() {
    local url="$1" dest="$2"
    if [ -d "$dest/.git" ]; then
        ok "$(basename "$dest"): already cloned"
    else
        bold "Cloning $(basename "$dest") from $url"
        git clone --depth=1 "$url" "$dest"
        ok "$(basename "$dest"): cloned"
    fi
}

uv_install_if_missing() {
    local pkg="$1"
    shift
    if .venv/bin/python -c "import ${pkg}" 2>/dev/null; then
        ok "${pkg}: already installed"
    else
        bold "Installing ${pkg} via uv pip"
        uv pip install "$@"
        ok "${pkg}: installed"
    fi
}

bold "==[1/5]== BrowserGym (browser benchmark scaffold)"
clone_or_skip "https://github.com/ServiceNow/BrowserGym" "${EXT_DIR}/BrowserGym"
uv_install_if_missing browsergym \
    -e "${EXT_DIR}/BrowserGym/browsergym/core" \
    -e "${EXT_DIR}/BrowserGym/browsergym/webarena" \
    -e "${EXT_DIR}/BrowserGym/browsergym/experiments" \
    -e "${EXT_DIR}/BrowserGym"

bold "==[2/5]== DoomArena (popup-attack benchmark, also imported by WASP)"
clone_or_skip "https://github.com/ServiceNow/DoomArena" "${EXT_DIR}/DoomArena"
uv_install_if_missing doomarena \
    -e "${EXT_DIR}/DoomArena/doomarena/core" \
    -e "${EXT_DIR}/DoomArena/doomarena/browsergym"

bold "==[3/5]== AgentDojo + InjecAgent + WASP (via setup_benchmarks.sh)"
bash "${REPO_ROOT}/eval_awareness_experiments/setup_benchmarks.sh" agentdojo injecagent wasp

bold "==[4/5]== Playwright Chromium (for browser benchmarks)"
if .venv/bin/python -c "from playwright.sync_api import sync_playwright; sync_playwright().__enter__().chromium.launch().close()" 2>/dev/null; then
    ok "playwright chromium: already installed"
else
    bold "Installing playwright chromium"
    uv run playwright install chromium
    ok "playwright chromium: installed"
fi

bold "==[5/5]== Sanity check — import each package"
for pkg in agentlab browsergym agentdojo doomarena; do
    if .venv/bin/python -c "import ${pkg}" 2>/dev/null; then
        ok "${pkg}: importable"
    else
        die "${pkg}: import failed — see errors above"
    fi
done

bold "==[done]== Everything is ready."
echo
echo "Next:  ./eval_awareness_experiments/launch_pilot.sh"
echo "Watch: .venv/bin/python -m eval_awareness_experiments.dashboard \\"
echo "         --results-dir eval_awareness_experiments/results/causal_pilot --port 8000"
