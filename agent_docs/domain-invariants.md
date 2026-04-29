# Domain Invariants

Use this before changing task generation, seeding, feasibility, Phase 4, prompt files, reward functions, auth, or sandbox execution.

## Spec First

`docs/worldsim-v5-technical-specifcation.md` is the source of truth. The filename typo is intentional. Every `worldsim/` module implements a spec section. When source or handoff docs expose real drift, update the spec first, then align code.

If the spec contains literal code for `run_claude_in_sandbox`, `BrowserUseAgent`, `apply_data_seed`, `save_state`, or `run_eval`, preserve that behavior verbatim unless the spec is intentionally changed.

## Scope

WorldSim v5 is strict WASP-aligned IPI evaluation. Current admitted surfaces are:

- GitLab issues/comments
- Reddit/Postmill posts/comments

Current dataset: 38 tasks (22 GitLab and 16 Reddit), regenerated 2026-04-26 after post-`057e8e26` strict exposure-contract eligibility. Treat this as run/context status, not a root-agent instruction.

Magento, Wikipedia, OpenStreetMap, shopping, and classifieds are excluded from the current WASP scope. Treat code paths for them as historical or support plumbing unless the spec and task explicitly bring them back.

## Phase 2 and Admission

Phase 2c feasibility is strict admission for Phase 4:

- Never ship a Phase 4 run on an unverified dataset.
- Never hand-edit `feasibility.status`.
- Do not use `--skip-feasibility` for shipping runs.
- Break-glass override is `WORLDSIM_STRICT_FEASIBILITY={true,false}` and should be called out explicitly.

Phase 4 varies adversarial strategy only. Benign task contracts and reward functions emitted by Phase 3 are invariant across variants.
Strategy variants must change payload semantics, not only punctuation, brackets, or wrappers; no-op and punctuation-only rewrites are generation failures.

## Phase 4 Gates and Metrics

Phase 4 has two gates:

1. Encounter gate: PVPO `max_coverage` is a float in `[0.0, 1.0]`; `max_coverage == 0` routes to placement-fix.
2. Attack-effectiveness gate: refusal judge evaluates whether the agent complied with the injection.

Transcript Purpose and Verbalized Eval Awareness are observational metrics. They must never gate, branch, or affect run control.

Trajectories flow through disk. Per-task directories under `logs/<run>/<task_id>/` hold `history.json`, `result.json`, `screenshots/`, `conversations/`, and `pvpo/`. Judges and classifiers receive host-side dataclass-projected slices.

The six host-API prompt files are:

- `worldsim/prompts/judge-adversarial-failure.md`
- `worldsim/prompts/generate-variant.md`
- `worldsim/prompts/transcript-purpose-guess.md`
- `worldsim/prompts/transcript-purpose-classify.md`
- `worldsim/prompts/verbalized-eval-awareness.md`
- `worldsim/prompts/placement-fix.md`

They feed single-turn API calls with no sandbox error boundary. Be conservative and test exact parsing contracts.

When adding OpenAI-backed prompt/API flows, prefer Structured Outputs or deterministic parsers over prose-only schema instructions. Keep schemas in code/tests rather than duplicated across long prompt text.

## Phase 4 Strategy and Rewards

Do not reintroduce `visual_concealment`; paper concealment is behavioral final-response steering, not visual concealment.

Behavioral concealment strategies such as `scripted_message`, `command_silent_execution`, and `false_justification` remain deferred unless the reward system changes. The current `AgentResponseEvaluator` plus `db_query_match` reward is tool-side and cannot grade those strategies.

## Auth and Runtime Boundaries

Claude auth precedence is centralized:

- Sandbox Claude Code: `worldsim/modal_sandbox.py::_build_claude_secrets`
- Host API calls: `worldsim/phase_4/anthropic_client.py`

Supported modes are `CLAUDE_CODE_OAUTH_TOKEN`, `ANTHROPIC_API_KEY`, and `ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL`. Let the helpers decide; never hard-code one path.

No Phase 4 trajectory step routes through `run_claude_in_sandbox`. Refusal judge, variant generator, Transcript Purpose, VEA, and placement-fix all use direct host Anthropic Messages API calls.

Modal sandbox scope is based on explicit inclusion with `image.add_local_file` / `image.add_local_dir`. Ignore-file patterns are not an isolation boundary.

`AgentLab/src/agentlab/benchmarks/redteam/execution.py` and `AgentLab/src/agentlab/benchmarks/redteam/claude_code.py` are read-only reference material. Runtime imports from `AgentLab/` are forbidden.
