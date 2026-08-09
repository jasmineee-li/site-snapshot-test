# Codex `.worktreeinclude` research — 2026-08-09

## Decision

Use a tracked repository-root `.worktreeinclude` with exact file paths, not
directory globs. For WARP Taskgen's persistent-main and disposable-development
workflow, the lean default is:

```gitignore
packages/warp-taskgen/.benchmark_host_id
packages/warp-taskgen/instances.json
packages/warp-taskgen/instances.smoke.local.json
```

These three small ignored files preserve the selected host identifier and make
configuration-aware development easier. The host identifier is operational
metadata, not an authentication credential. The current instance configs use
`token_file`/`token_env` indirection and contain no literal `token` scalar. Do
not include `.proxy_token`, datasets, vendors, virtual environments, logs,
`.claude/`, or `.codex-worktrees/`.

## Documented scope

- `.worktreeinclude` belongs at the repository root and contains ignored paths
  or `.gitignore`-style patterns. Codex copies only matching ignored files;
  tracked files and unmatched untracked files are outside this mechanism.
- The copy happens when the ChatGPT desktop app creates a local Codex-managed
  worktree. It does not apply to remote worktrees or worktrees made with
  `git worktree add` on the command line.
- Ignored `AGENTS.override.md` files are copied automatically and should not be
  listed.
- Source symlinks are skipped, and an existing destination is not overwritten.

Source: [OpenAI Worktrees documentation](https://learn.chatgpt.com/docs/environments/git-worktrees#copy-ignored-local-files-into-managed-worktrees).

## Matching and copy semantics

The installed ChatGPT desktop app, version `26.803.41515` (build `6321`), was
inspected read-only at `/Applications/ChatGPT.app/Contents/Resources/app.asar`
(SHA-256 `5f6e773aafd542d3cf09e10b5dca6cabd301d0a155f4b8ce870e3915fc3da25e`).
Its implementation confirms and sharpens the public contract:

1. Codex enumerates untracked ignored paths with `git ls-files --others
   --ignored --exclude-standard`, separately enumerates paths matched by
   `.worktreeinclude`, and copies the intersection. A file must therefore be
   both ignored by normal Git rules and matched by `.worktreeinclude`.
2. Patterns are passed to Git with `--exclude-from=.worktreeinclude`. Git
   defines that format as gitignore-style, one pattern per line, relative to
   the top of the repository. Exact paths containing `/` are therefore the
   least surprising choice.
3. Codex copies only paths whose no-follow filesystem type is a regular file.
   It neither copies nor follows a source symlink. A directory pattern can
   still select every ignored regular file below that directory, which can
   duplicate substantial data.
4. File copies are exclusive. If the target already exists, Codex skips it
   instead of replacing it. Codex also refuses to create a copied file through
   a symlinked destination-parent path.
5. The copier runs only for a local host during managed-worktree creation. It
   is not a continuous sync: later changes in persistent main do not update an
   existing worktree.

Git's primary documentation defines the underlying pattern rules and confirms
that `--exclude-from` patterns are root-relative:
[gitignore pattern format](https://git-scm.com/docs/gitignore#_pattern_format)
and [`git ls-files` exclude patterns](https://git-scm.com/docs/git-ls-files#_exclude_patterns).

## WARP Taskgen implications

- `packages/warp-taskgen/vendors/webarena-verified` and the three
  `packages/warp-taskgen/data/hf/*` entries are source symlinks. Codex will skip
  them, not follow or duplicate their 7+ GiB of targets. They should remain a
  persistent-main facility; a future opt-in setup script could recreate links
  for an exceptional worktree without copying data.
- `packages/warp-taskgen/.proxy_token` is a real proxy secret under the
  repository's [secret policy](../../packages/warp-taskgen/agent_docs/secrets.md).
  Although OpenAI documents secrets as a supported use case, automatically
  duplicating this token into every disposable worktree conflicts with this
  repository's least-copy workflow.
- `packages/warp-taskgen/.venv`, generated logs, and large artifacts are not
  needed by the canonical `bash scripts/accept_taskgen.sh` development gate and
  would make managed-worktree cleanup expensive. OpenAI notes that each
  worktree has its own files, dependencies, and caches and can consume
  significant disk space.
- The host identifier and two recommended instance configs total under 5 KiB,
  are ignored, and contain no live token. The configs use external token
  indirection. Copying them adds negligible disk cost while preserving the
  live credential boundary.

## Verification contract

Before merging the repository configuration:

1. Prove every listed source is ignored with `git check-ignore -v`.
2. Reproduce Codex's candidate selection with `git ls-files --others --ignored`
   and confirm that only the intended three regular files match.
3. Create a fresh **ChatGPT desktop app managed worktree** from a checkout that
   already contains the tracked `.worktreeinclude`; a manual Git worktree is
   not a valid end-to-end test.
4. Verify the three copied files by SHA-256 without printing contents.
5. Verify `.proxy_token`, dataset/vendor symlinks, logs, and dependency
   directories are absent.
6. Modify neither side: the test must also confirm the copies are independent
   snapshots rather than links or ongoing synchronization.

This keeps ordinary development fast and reproducible while persistent main
remains the only local/live Taskgen environment.
