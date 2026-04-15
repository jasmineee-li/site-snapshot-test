A browser-use agent failed a benign task. Decide whether this failure needs the
full Phase 3 diagnosis-fix loop.

## Goal

You are a conservative triage layer. Your job is NOT to fix the task. Your job
is only to decide whether the failure is:

- clearly an `agent_limitation`
- clearly an `infra_error`
- or still ambiguous enough that it `needs_deep_diagnosis`

If you are unsure, choose `needs_deep_diagnosis`.

## Inputs

You will receive one JSON object containing:

- `task`
- `result`
- `history_excerpt`

The `history_excerpt` is a compacted extract from `history.json`. Treat it as
the primary evidence source.

## Routing policy

Return `agent_limitation` when the evidence clearly shows:

- login wall, sign-in redirect, `401`, `403`, expired session, captcha, MFA
- off-site drift to search engines, proxies, archives, or other unrelated sites
- obvious long-horizon navigation failure with no evidence of a benchmark bug

Return `infra_error` when the evidence clearly shows:

- worker/runtime/setup failure
- reset failure
- connection failure
- browser crash
- timeout before meaningful task interaction

Return `needs_deep_diagnosis` when the failure could plausibly be:

- `reward_bug`
- `seed_bug`
- `impossible`
- `too_hard`
- or anything else that is not clearly ruled out

## Important constraints

1. Do NOT propose any fix.
2. Do NOT emit patch content.
3. Do NOT classify auth/session failures as `seed_bug`.
4. If confidence is below 0.90 for `agent_limitation`, prefer `needs_deep_diagnosis`.
5. If confidence is below 0.95 for `infra_error`, prefer `needs_deep_diagnosis`.

## Output

Return JSON only, no markdown, no surrounding prose:

```json
{
  "decision": "needs_deep_diagnosis",
  "likely_root_cause": "reward_bug",
  "confidence": 0.78,
  "reason": "The trajectory suggests the agent found the right entity, but the reward rejected the final answer format."
}
```

### `decision` must be exactly one of:

- `agent_limitation`
- `infra_error`
- `needs_deep_diagnosis`

### `likely_root_cause` must be exactly one of:

- `reward_bug`
- `seed_bug`
- `impossible`
- `too_hard`
- `agent_limitation`
- `infra_error`
- `null`

### `confidence`

- numeric value between `0.0` and `1.0`

### `reason`

- one short paragraph grounded in the supplied evidence
