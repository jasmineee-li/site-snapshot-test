#!/usr/bin/env bash
# stop-verify.sh — Claude Code Stop hook for context-efficient back-pressure.
#
# Runs ruff check over Python files changed since the last commit (per
# `git diff`). Silent on success. On failure, emits ruff output to stderr
# and exits 2, which the Claude Code harness interprets as "re-engage the
# agent" — the agent then sees the errors and is forced to fix them before
# the turn actually ends.
#
# Why scope to changed files only: the repo carries a backlog of
# pre-existing lint warnings in files we haven't touched this session.
# Flagging those on every Stop would pollute context with issues Claude did
# not introduce — the exact anti-pattern the HumanLayer back-pressure post
# warns about (4,000 lines of irrelevant output in the context window
# wrecks the agent's reasoning). We only surface what *this* session
# actually changed.
#
# Why ruff-check-no-fix: the sibling `ruff-autofix.sh` hook runs on every
# Write/Edit with `--fix`, but the repo's pyproject keeps F401
# (unused-import) as a non-autofixable warning (see the comment in
# pyproject.toml: "don't let ruff check --fix silently delete imports").
# This Stop hook is where those warnings — plus any other autofix misses —
# surface.
#
# Why not pytest: the unit suite takes ~40s and has a pre-existing failure
# outside the Phase 4 scope as of 2026-04-22. Adding it would slow the
# per-turn loop and fire on every Stop. Revisit once the baseline is green
# and a per-file test runner (e.g., `pytest --last-failed --lf-diff` or an
# ast-import walk) can keep it fast.

set -o pipefail
cd "$CLAUDE_PROJECT_DIR" || exit 0

# Scope to tracked .py files with staged or unstaged changes vs HEAD. We
# deliberately skip untracked files: new scripts/plans/etc. sitting in the
# working tree often predate the current session, and flagging their
# warnings on every Stop would pollute context with issues Claude did not
# introduce. Once a new file is staged (git add) it becomes tracked and
# this hook starts watching it.
CHANGED=$(git diff --name-only --diff-filter=ACMR HEAD 2>/dev/null | grep -E '\.py$' || true)

if [ -z "$CHANGED" ]; then
    exit 0
fi

# Only lint files that still exist (deletions can slip through on edge cases).
EXISTING=$(echo "$CHANGED" | while IFS= read -r f; do [ -f "$f" ] && echo "$f"; done)
if [ -z "$EXISTING" ]; then
    exit 0
fi

OUTPUT=$(echo "$EXISTING" | xargs uv run ruff check 2>&1)
RC=$?

if [ $RC -ne 0 ]; then
    echo "Stop-verify: ruff check failed on changed files" >&2
    echo "$OUTPUT" >&2
    exit 2
fi

exit 0
