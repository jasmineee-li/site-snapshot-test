# Code Review Prompt — Phase 4 Judge + Variant-Gen API Cutover

> **ARCHIVED REVIEW PROMPT.** This prompt reviews the old judge and
> variant-generator cutover. It mentions ecoval-fix and placement-fix sandbox
> paths that are no longer current. Use the technical spec for live review
> prompts.

Copy-paste the block below into a **fresh** Claude Code session (or any other reviewer agent) at the repo root. The prompt is self-contained, references only the handoff doc + the actual code, and is written to surface disagreements rather than endorse the author's framing.

---

```
You are performing an independent, adversarial code review on a pull request that
cuts over Phase 4 of WorldSim v5 from Modal-sandbox Claude Code to direct
Anthropic Messages API calls for the judge and variant generator.

Your job is to find what the author got wrong, not to validate what they got right.
Treat the handoff document as a claim set to verify, not as truth. The author is
also running this review — so bias toward scrutinizing their claims.

## Required reading (in order)

Read these files IN FULL before forming opinions. Do not skim.

1. docs/handoffs/codex-handoff-phase-4-judge-variant-api-cutover.md
   — The author's handoff. Treat every numbered claim as an assertion to verify.

2. CLAUDE.md
   — Project invariants. Look for any the PR violates.

3. docs/worldsim-v5-technical-specifcation.md (section "Adaptive Strategy
   Variation" and "Sandbox model" bullet at ~line 187)
   — The spec is the source of truth per CLAUDE.md line 11. If the PR diverges,
   the spec is right — unless the spec itself was updated in this PR, in which
   case verify the updates are internally consistent.

4. worldsim/prompts/judge-adversarial-failure.md
5. worldsim/prompts/generate-variant.md
   — The new prompts. Read them as if you were Claude Sonnet 4.6 receiving them.
   Flag any phrasing that could trip safety reflexes. Flag any gap between the
   prompt's output schema and the tool schemas in judge_api.py / variant_api.py.

## Parallel investigation (use multiple Agent tool calls in one message)

Dispatch these investigations in parallel using the Explore subagent. Each gets
its own focus so your context stays clean.

### Agent A — claim verification against code

Prompt:
"I'm reviewing a PR at /Users/ashtonchew/projects/browser-sim. The author's
handoff is at docs/handoffs/codex-handoff-phase-4-judge-variant-api-cutover.md.
For each numbered claim in sections 2 (Architecture), 3 (Files), 4 (Key design
decisions), and 5 (Backwards compatibility), verify the claim against the
actual code. Report ONLY mismatches — skip claims that verify. Specifically
check:
- Section 2.1 'Before' description against git history (`git log -p` on
  worldsim/phases/phase_4_adversarial.py)
- Section 2.3 claim that _revise_adversarial_task is untouched — grep for
  callers and confirm it's still wired to ecoval-fix and placement-fix
- Section 3.2 'Modified' table — for each file, diff against main and verify
  the described change is what actually happened
- Section 4.2 claim that the judge never recommends strategies — grep
  worldsim/phase_4/judge_api.py and worldsim/prompts/judge-adversarial-failure.md
  for any residual 'recommend' language
- Section 4.3 tool-use forcing — verify the tool_choice field is actually
  {'type': 'tool', 'name': ...} and that response parsing reads
  response.content[0].input, not response.content[0].text
- Section 4.7 authorization preamble — check that the preamble is present in
  BOTH prompts verbatim, not just paraphrased
Report any mismatch with file:line citations. ≤600 words."

### Agent B — risk surface the author didn't mention

Prompt:
"I'm reviewing a PR at /Users/ashtonchew/projects/browser-sim that moves
Phase 4 judge + variant generator from Modal sandbox to direct Anthropic
Messages API. The author's risk list is in section 10 of
docs/handoffs/codex-handoff-phase-4-judge-variant-api-cutover.md. Find risks
the author did NOT list. Specifically investigate:
- worldsim/phase_4/anthropic_client.py _resolve_auth() precedence vs
  _call_anthropic_fallback at worldsim/phases/phase_2_text_fill.py:626 — do
  they actually match, or do we have two slightly-different precedences?
- The Semaphore(250) at worldsim/phase_4/concurrency.py — is this a
  module-level singleton that's safe under Python's import-order rules? What
  happens under resume with partial state?
- Cost tracker integration — the author synthesizes a _summary-shaped dict.
  Verify the dict shape matches what cost_tracker.record() at
  worldsim/cost_tracker.py expects. Grep for all callers of cost_tracker.record
  and compare shapes.
- Prompt caching — cache_control is position-sensitive. Verify that
  json.dumps(sort_keys=True) is used CONSISTENTLY so the cache key is stable.
- The trajectory slicer — look at edge cases the author may have missed: what
  if history.json is empty? What if every step has model_output = None? What
  if it's a symlink to a tarball?
- Resume semantics — _phase_4_variant_fingerprint at
  worldsim/phases/phase_4_adversarial.py includes sandbox_model. Does anything
  about the API path change the fingerprint input structure? Silent
  invalidation would re-run expensive Browser-Use evaluations.
- Failpoint compatibility — failpoints.py labels at
  worldsim/phases/phase_4_adversarial.py:2342 etc. The crash-resume window
  widened when we moved to API calls. Did the author actually move the
  checkpoint write to the right place? Verify.
- Security: does the new code log the full adversarial payload text anywhere
  that could leak through observability?
- OpenRouter-specific: the author says 'AsyncAnthropic base_url=<proxy root>'
  works for OpenRouter. Verify empirically by reading the smoke test results
  in the plan file (~/.claude-ashton-2/plans/view-the-phase-4-delegated-pizza.md)
  or test this hypothesis by reading the SDK source.
Report each finding with severity (high/medium/low), evidence (file:line),
and a concrete mitigation. ≤1200 words."

### Agent C — test quality and coverage

Prompt:
"Review the new tests under /Users/ashtonchew/projects/browser-sim/tests/:
- tests/test_phase_4_strategy_catalog.py
- tests/test_phase_4_trajectory_slice.py
- tests/test_phase_4_judge_api.py
- tests/test_phase_4_variant_api.py
- tests/integration/test_phase_4_judge_api_smoke.py
- tests/conftest.py (added patched_anthropic_client fixture)
- tests/test_phase_4_adversarial.py (existing file, one test rewritten —
  test_generate_variant_merges_api_tool_use_output around line 180)

Assess:
1. Are there tests that only exercise the mock and never the real contract?
   A test that asserts 'the mock was called' without asserting on production
   behavior is worse than no test.
2. Are there untested failure modes? Cross-reference failure_class values in
   worldsim/phase_4/judge_api.py and worldsim/phase_4/variant_api.py against
   test coverage.
3. Is the patched_anthropic_client fixture correct for new callers that might
   be added later? Does the fixture fail loudly if a new module imports
   get_client without being cross-patched?
4. The crash-resume scenario: the author said they added a test for 'process
   killed between judge response and checkpoint write'. Find this test. If it
   doesn't exist, flag it.
5. Run `uv run pytest tests/ -q --ignore=tests/integration` and confirm it
   passes. Report the count.
6. Run `uv run pytest tests/test_phase_4_*.py -v` and check if any test
   actually covers the OAuth token auth path (Bearer header, not x-api-key).
Report gaps with severity and a concrete test to add. ≤600 words."

## After the parallel investigations

Do NOT trust agents blindly. Spot-check their findings:

1. Pick 2-3 high-severity findings and manually read the cited file:line to
   confirm.
2. If an agent says a claim verifies — don't re-verify everything, but pick
   one claim you found interesting and check it yourself.

## Your own direct checks

These you do yourself, not via agents:

1. Read worldsim/phase_4/judge_api.py and worldsim/phase_4/variant_api.py end
   to end. Count the number of distinct failure_class strings produced.
   Cross-reference against the author's section 4.9 list. Any missed?
2. Diff CLAUDE.md: verify the integration-test trigger list actually includes
   both prompt files (author claim in section 3.2).
3. Read the last 50 lines of logs/phase_4_demo_20260417_190422.log and verify
   the '3570 verbatim refusal' claim in section 1.1 of the handoff is exact.
4. Read worldsim/phase_4/strategy_catalog.py ALLOWED_STRATEGIES. Count.
   Verify it is 22. Verify visual_concealment is absent. Verify the three
   concealment-steering strategies (scripted_message, command_silent_execution,
   false_justification) are absent.
5. Look for any `# TODO`, `# FIXME`, `# XXX`, or `# HACK` in the new files.
   These are self-flagged technical debt — list them.

## Required output

Produce a single review document with:

A. **Verdict** — one of {approve, approve with changes, request changes,
   reject}. Justify in ≤3 sentences.

B. **Must-fix before merge** — bullet list with file:line and a concrete
   one-sentence fix. These are blockers.

C. **Should-fix soon** — bullet list; not blocking but clear regressions or
   gaps.

D. **Design concerns** — points worth discussion but not clearly wrong. Lean
   into the ten "review questions for the reviewer" in section 10 of the
   handoff — do not rubber-stamp those; push back where you disagree.

E. **Things the author missed** — risks, edge cases, or couplings that are
   not in the handoff. This is where adversarial review earns its keep.

F. **Claim verification summary** — for each verifiable claim in the handoff,
   state {verified / mismatched / partial}. Cite evidence for any mismatch.

G. **Test quality note** — from Agent C; one paragraph.

H. **Decision on the ten reviewer questions** — for each of the ten questions
   in handoff section 10, give a one-sentence concrete answer. Do not evade.

## Constraints on the review

- Cite file:line for every claim you make. No unattributed assertions.
- Do not accept 'the author says X' as evidence for X. Confirm X by reading
  code.
- If you find yourself agreeing with the author on everything, you are
  probably not reviewing hard enough. Re-check.
- Pay particular attention to: (a) anything that silently changes behavior,
  (b) anything that appears correct but only because a test is mocking it,
  (c) anything the author flagged as 'deferred' — is deferring it actually
  safe?
- You may NOT spend more than 45 minutes on this review. Scope aggressively.
- When in doubt about a judgment call, lean toward 'request changes' and
  make the author argue for their choice.

Begin.
```

---

## How to use this prompt

1. Open a new Claude Code session (recommended: `claude --model opus` for the
   deepest review; `sonnet` is fine too).
2. `cd /Users/ashtonchew/projects/browser-sim`.
3. Paste the prompt above verbatim (everything between the ``` markers).
4. Review the output. Responses in the D/E/H sections will most usefully
   surface disagreement.

The prompt is designed so the reviewer (a) cannot skip the handoff,
(b) cannot rubber-stamp without citing file:line, (c) is explicitly told to
look for what the author missed, and (d) has specific parallel-agent
investigations with scoped focus so context doesn't bloat.
