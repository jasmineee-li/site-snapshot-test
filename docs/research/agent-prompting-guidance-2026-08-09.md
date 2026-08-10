# Agent-instruction research — 2026-08-09

Scope: official Anthropic documentation reviewed on 2026-08-09, translated into
rules for the `agent-guidance-rebuild` worktree. The repository's
`writing-for-agents` skill supplies the progressive-disclosure and completion-
criterion lens; the links below establish Claude's loading and prompting
behavior.

## Decision

Keep one canonical instruction source and make the load boundary explicit:

| Need | Canonical location | Loading behavior |
| --- | --- | --- |
| Project-wide facts, conventions, and always-on workflows | Root `CLAUDE.md` (or a canonical `AGENTS.md` imported by it) | Loaded at session start; keep it concise |
| Rules for a path or file family | `.claude/rules/*.md` with path scope | Loaded only for matching work |
| An occasional workflow or long reference | `.claude/skills/<name>/SKILL.md` plus supporting files | Description is discoverable; full body loads on invocation |
| Machine-local learnings | Claude auto memory | Local to the machine/repository, not team policy |
| A guarantee that must happen every time | Hook/permission/configuration | Deterministic enforcement, not prompt advice |

This hierarchy follows Anthropic's distinction between always-loaded `CLAUDE.md`,
path-scoped rules, on-demand skills, auto memory, and deterministic hooks. See
[Claude Code memory](https://code.claude.com/docs/en/memory) and [Extend Claude
with skills](https://code.claude.com/docs/en/skills).

## Prompt and instruction principles

- Define a checkable result before writing instructions. Anthropic recommends
  specific, measurable success criteria and evaluations rather than a vague
  "good" outcome ([success criteria and
  evals](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests)).
- Use direct, concrete language. State the desired behavior, constraints, and
  output format; use numbered steps when order or completeness matters. Include
  context or rationale when it helps Claude generalize, and prefer positive
  instructions to a list of prohibitions ([prompting best
  practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)).
- Make every procedure end with a visible completion criterion: the exact test,
  lint/typecheck command, artifact, or review result that proves it is done.
  For larger work, the spec should name files/interfaces, state what is out of
  scope, and end with an end-to-end verification step ([Claude Code best
  practices](https://code.claude.com/docs/en/best-practices)).
- Treat guidance as advisory. `CLAUDE.md` shapes behavior but is not a hard
  enforcement layer; move non-negotiable checks or blocks to hooks, permissions,
  or other configuration ([memory: troubleshooting and
  enforcement](https://code.claude.com/docs/en/memory)).

## `CLAUDE.md` and `AGENTS.md`

- `CLAUDE.md` is the file Claude Code reads. It walks up from the working
  directory, concatenates ancestor files, and loads nested files when Claude
  reads files in those directories. Keep each always-loaded file specific,
  structured, and under Anthropic's roughly 200-line target; larger files cost
  context and reduce adherence ([write effective
  instructions](https://code.claude.com/docs/en/memory)).
- If `AGENTS.md` is the repository's shared source, avoid a second copied body.
  Add a `CLAUDE.md` containing `@AGENTS.md` (and any genuinely Claude-specific
  additions), or make `CLAUDE.md` a symlink to `AGENTS.md` when no additions are
  needed. Anthropic documents both choices; on Windows, prefer the import because
  symlink creation may require elevated privileges ([AGENTS.md guidance](https://code.claude.com/docs/en/memory)).
- Imports are for organization, not context reduction: imported files are
  expanded into the startup context. Use a path-scoped rule or a skill when the
  material should be loaded conditionally. After changing the linkage, verify
  the loaded set with Claude Code's `/context` command ([imports and loading
  order](https://code.claude.com/docs/en/memory)).
- Auto memory is Claude's machine-local notes, separate from project policy. It
  is per repository and shared across that repository's worktrees; only the
  first 200 lines or 25 KB of `MEMORY.md` load at session start. Put team rules,
  architecture, and required checks in versioned instruction files, not auto
  memory ([auto memory](https://code.claude.com/docs/en/memory)).

## Model-invoked skills and progressive disclosure

- Use a model-invoked skill (the default, with a `description`) only when Claude
  should discover it autonomously or another skill needs it. The description is
  always part of the skill listing, so put the leading use case and concrete
  trigger phrases first; the combined `description`/`when_to_use` text is capped
  at 1,536 characters ([skill frontmatter](https://code.claude.com/docs/en/skills)).
- Set `disable-model-invocation: true` for side-effectful or timing-sensitive
  workflows such as commit, deploy, or external messaging. The user can still
  invoke these manually, but Claude will not trigger them. Use
  `user-invocable: false` for background reference that should be available to
  Claude but is not a useful slash command ([control who invokes a
  skill](https://code.claude.com/docs/en/skills)).
- Keep `SKILL.md` focused on the actionable procedure and concise standing
  guidance. The rendered body stays in context for the session; Anthropic
  recommends keeping it under 500 lines. Put detailed API references, examples,
  and scripts in supporting files and link to them from `SKILL.md`, so they are
  read only when needed ([supporting files and skill
  lifecycle](https://code.claude.com/docs/en/skills)).
- Apply progressive disclosure by branch: universal facts belong in root
  `CLAUDE.md`; path-specific facts belong in scoped rules; rare procedures and
  long references belong in skills/supporting files. A pointer should say what
  the target contains and when to load it, not duplicate the target's body.

## Completion and review contract

For each workflow encoded in `CLAUDE.md`, a rule, or a skill:

1. State the ordered actions and the scope (files, directories, or systems).
2. State the evidence required at the end: exact commands and expected outcome,
   generated artifact, or an explicit reviewer result.
3. Require verification against the task's criteria, including edge cases where
   relevant. Anthropic's Claude Code guidance says to provide verification and
   not ship work that cannot be verified.
4. For unattended or high-impact work, run a fresh-context review of the diff
   against the requirements; ask the reviewer to report correctness/scope gaps,
   not style preferences ([adversarial review](https://code.claude.com/docs/en/best-practices)).

## Prevent duplicate and stale instructions

- Give each behavior one owner. Do not maintain parallel copies of `AGENTS.md`
  and `CLAUDE.md`; import or symlink the canonical file instead.
- Keep commands and paths aligned with the repository's actual scripts and
  layout. A short pointer to a live source is safer than a hand-copied cache;
  update guidance when the source changes.
- Periodically review root and nested `CLAUDE.md` files and `.claude/rules/` for
  contradictions and obsolete paths. Claude concatenates them and may choose
  arbitrarily when rules conflict. Anthropic explicitly warns that an
  over-specified `CLAUDE.md` causes important rules to get lost; prune no-op,
  stale, or duplicated lines and move conditional material behind the appropriate
  rule/skill boundary ([avoid common failure patterns](https://code.claude.com/docs/en/best-practices)).
- If a behavior is already guaranteed by a hook or permission, keep only the
  human-facing rationale in guidance; do not restate a second competing
  procedure. After pruning, test both a representative trigger and a nearby
  non-trigger for every model-invoked skill.

## OpenAI/Codex cross-check

- OpenAI's GPT-5.6 guidance favors lean prompts: remove one group of repeated
  instructions, examples, or tools at a time, rerun the same representative
  evals, state each instruction once, and expose only tools relevant to the
  task. This supports the 60–80-line root router and the branch-specific
  disclosure above; measure any further pruning rather than assuming it helps
  ([GPT-5.6 model guidance](https://developers.openai.com/api/docs/guides/latest-model)).
- Keep autonomy and approval boundaries in one compact policy: answer/review/
  diagnose requests inspect and report; change/build/fix requests may make
  in-scope local edits and run non-destructive validation; external writes,
  destructive actions, purchases, or material scope expansion require
  confirmation. GPT-5.6 still needs domain context, hard constraints, approval
  boundaries, and success criteria even when it infers intent ([GPT-5.6 model
  guidance](https://developers.openai.com/api/docs/guides/latest-model)).
- Codex reads `AGENTS.md` before work, walking from the project root to the
  current directory, choosing `AGENTS.override.md` before `AGENTS.md`, and
  concatenating at most one instruction file per directory in root-to-leaf
  order. The combined project instruction budget is 32 KiB by default. Root
  and evaluation `AGENTS.md` symlinks therefore make Codex consume the same
  canonical `CLAUDE.md` text without a second copy; verify active files in a
  fresh run ([Codex `AGENTS.md` discovery](https://learn.chatgpt.com/docs/agent-configuration/agents-md)).

## Sources

- [Prompt engineering overview](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/overview)
- [Prompting best practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices)
- [Define success criteria and build evaluations](https://platform.claude.com/docs/en/test-and-evaluate/develop-tests)
- [How Claude remembers your project](https://code.claude.com/docs/en/memory)
- [Extend Claude with skills](https://code.claude.com/docs/en/skills)
- [Best practices for Claude Code](https://code.claude.com/docs/en/best-practices)
- [OpenAI GPT-5.6 model guidance](https://developers.openai.com/api/docs/guides/latest-model)
- [Codex `AGENTS.md` discovery](https://learn.chatgpt.com/docs/agent-configuration/agents-md)
