# Runtime Boundaries

Use this before changing auth, Modal sandbox setup, host API calls, AgentLab
references, benchmark surface identity, or file routing.

## Auth and API paths

Sandbox Claude Code resolves credentials in
`warp_taskgen/modal_sandbox.py::_build_claude_secrets`; host-side Phase 4 calls use
`warp_taskgen/phase_4/anthropic_client.py`. Supported modes are
`CLAUDE_CODE_OAUTH_TOKEN`, `ANTHROPIC_API_KEY`, and
`ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL`. Sandbox precedence is OAuth,
proxy, then direct key; host Messages precedence is proxy, OAuth, then direct
key. Let the helpers choose when multiple modes are present.

The shared client owns transport retries, model normalization, API-exception
classification, and bounded concurrency. Semantic retries are for parse or
validation failures. Diagnostics record compact request/response metadata,
thinking mode, and provider extra-body mode without prompts, cookies, secrets,
or raw trajectories.
The default eval-awareness rewrite additionally retains exact model-facing SDK
arguments in a separate feature-owned Run Artifact; its compact diagnostics keep
references and summaries. See the spec's Phase 4 Per-Task Output contract.

Phase 4 judge, variant, TP, VEA, placement-fix, cue diagnosis, and rewrite calls
use the host Messages API. Sandbox work is reserved for the phases that need
isolated filesystem exploration or generation.

## Sandbox and references

Modal isolation comes from explicit `image.add_local_file` and
`image.add_local_dir` inclusion. Ignore-file patterns are not a filesystem
boundary. Read `secrets.md` before changing tracked credentials, instance
configs, or proxy-token handling.

`AgentLab/src/agentlab/benchmarks/redteam/{execution.py,claude_code.py}` is
read-only reference material for sandbox mechanics. Retype equivalent behavior
inside `warp_taskgen/`; runtime imports from `AgentLab/` are forbidden.

## Surface identity

Canonical carrier IDs are `issue.description`, `note.body`, `submission.body`,
and `comment.body`. Benchmark profiles may expose adapter-local IDs such as
`profile_issue_description` or `thread_comment_body`. Resolve profile aliases
through `BoundSite.resolve_profile_surface` with route/editor context; ambiguous
or unknown mappings fail closed instead of choosing arbitrarily.

Completion means the selected runtime boundary is explicit, auth stays in the
central helper, sandbox files are intentionally included, and no benchmark-
specific runtime dependency crosses the AgentLab boundary.
