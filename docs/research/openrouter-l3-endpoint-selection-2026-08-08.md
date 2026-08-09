# OpenRouter L3 endpoint-selection research — 2026-08-08

## Scope

This note explains the release-gate observation recorded in
[`cutover-release-proof-2026-08-08.md`](./cutover-release-proof-2026-08-08.md):
the r8a Phase 2 L3 classifier received HTTP 404 with the message
`No endpoints available matching your guardrail restrictions and data policy`.
No credential value is recorded in this note. The source of truth for the
request path is the checked-in code, not a secret or host overlay.

## What the code actually requested

- [`worldsim/phase_4/anthropic_client.py`](../../packages/warp-taskgen/worldsim/phase_4/anthropic_client.py)
  selects the OpenRouter path when both `ANTHROPIC_AUTH_TOKEN` and
  `ANTHROPIC_BASE_URL` are non-empty. It passes the proxy root to the Anthropic
  SDK, which appends `/v1/messages`, and keeps the `vendor/model` name used by
  OpenRouter.
- [`worldsim/phase_2/target_resolution/l3.py`](../../packages/warp-taskgen/worldsim/phase_2/target_resolution/l3.py)
  calls the Anthropic Messages API with model `claude-sonnet-4-6`, a short
  output limit, and one `emit_target` tool schema. This is therefore both a
  model-availability check and a tool-capable endpoint check.
- The release proof says that all non-L3 live checks passed, that the host had
  OpenRouter authentication only, and that no direct Anthropic API key or
  Claude OAuth token was configured. This is consistent with an endpoint
  selection failure rather than a benchmark-instance failure.

OpenRouter documents the Anthropic-compatible endpoint as
`POST /api/v1/messages`; it accepts tools and provider-routing controls. See
[Create a message](https://openrouter.ai/docs/api/api-reference/anthropic-messages/create-messages)
and [Tool & Function Calling](https://openrouter.ai/docs/guides/features/tool-calling).

## Why the message points to policy filtering

OpenRouter first builds the set of endpoints that can serve the requested model
and parameters. Account privacy settings and guardrails can then remove
endpoints. The official documentation says:

- provider allowlists are intersected across guardrails;
- model allowlists are intersected;
- ZDR requirements combine with OR logic per model group; and
- an endpoint with an unclear provider policy is treated conservatively as
  retaining and training on data.

See [Guardrails](https://openrouter.ai/docs/guides/features/guardrails/overview),
[Zero Data Retention](https://openrouter.ai/docs/guides/features/zdr), and
[Provider Logging](https://openrouter.ai/docs/guides/privacy/provider-logging/).
The `/api/v1/models/user` endpoint is specifically documented as the model list
filtered by the caller's provider preferences, privacy settings, and guardrails:
[List models for the authenticated user](https://openrouter.ai/docs/api/api-reference/models/list-models-user).

The most plausible explanation is therefore:

1. `claude-sonnet-4-6` had one or more candidate providers;
2. the API key's account/key guardrail or privacy policy removed those
   endpoints (for example, an Anthropic ZDR requirement removed first-party
   Anthropic endpoints, while the remaining Bedrock/Vertex endpoints were not
   allowed or did not satisfy the policy); and
3. the router had no eligible endpoint left for a Messages + tools request.

This is an inference from the error text and the documented filtering rules. It
does not identify which exact account toggle or guardrail caused the result.

The current generic OpenRouter error table calls “no available model provider
that meets your routing requirements” HTTP 503, while invalid credentials are
HTTP 401, insufficient credits are HTTP 402, and permission/guardrail blocks
are HTTP 403. See [API error handling](https://openrouter.ai/docs/api/reference/errors-and-debugging).
The observed 404 is therefore likely an Anthropic-skin or route-specific
translation of the “no eligible endpoint” condition. It is not the documented
invalid-key signature. It also does not prove that the key is valid: only an
authenticated, harmless preflight can establish that. OpenRouter's status page
currently reports [all systems operational](https://status.openrouter.ai/), so
there is no public platform-wide incident explaining this one-key failure.

## Authenticated probe result

Bounded probes used the configured local benchmark secret in place and printed
only HTTP status or sanitized model/result metadata:

- `GET /api/v1/key` returned HTTP 200. OpenRouter documents 401 as the
  missing/invalid/revoked-key result, so the configured key is valid.
- `GET /api/v1/models/user` did not return a model list. It returned HTTP 404
  with `No endpoints found supporting your guardrail restrictions.` This
  endpoint is filtered by the caller's privacy and guardrail policy.
- The repository's exact L3 `emit_target` tool request failed with the same
  policy 404 for all four probes:
  `anthropic/claude-sonnet-4-6`, `openai/gpt-5.4-nano`,
  `google/gemini-3.1-flash-lite`, and `qwen/qwen3.5-flash-02-23`.

This separates key validity from provider selection. The key authenticates,
but the current account policy leaves no endpoint for either the requested
Claude model or three small models from other provider groups. The failure is
therefore account-wide routing/guardrail configuration, not a Sonnet-only
outage and not a reason to change the canonical L3 model.

The safe diagnostic sequence remains:

1. `GET https://openrouter.ai/api/v1/models/user` with the same bearer token.
   A 401 means invalid/revoked credentials. A successful response proves the
   key can authenticate and shows which models survive the caller's policy.
2. Send a one-token, no-tool `POST /api/v1/messages` to a model visible in that
   filtered list.
3. Send a one-token request with the small `emit_target`-shaped tool schema.
   If (2) works but (3) fails, the remaining issue is provider/parameter
   compatibility or tool routing. If both fail with the same “no endpoints”
   message, inspect account/key guardrails and provider/data-policy settings.
4. For diagnosis only, add `X-OpenRouter-Metadata: enabled` and retain the
   redacted routing summary. OpenRouter documents that this can show filtered
   endpoints and guardrail pipeline stages; never persist bearer tokens or
   prompt contents in the release evidence.

Do not bypass Phase 2c admission or silently treat an unverified model as a
shipping replacement. The key/path check must be green before the fresh
Phase 2c → Phase 3 → Phase 4 sequence.

## Small models worth trying

First query `/api/v1/models/user` and require `tools` in the model's
`supported_parameters`. OpenRouter documents that `tools` means function
calling and that the models endpoint can filter by this parameter in
[Models](https://openrouter.ai/docs/guides/overview/models). Model-level
`supported_parameters` is a union across providers, so the final tool smoke
test is still required.

These are bounded candidates from current official OpenRouter model pages:

| Model | Why it is a useful probe | Published price (input / output per 1M tokens) | Caveat |
| --- | --- | ---: | --- |
| `openai/gpt-5-mini` | Explicitly exposes the Anthropic Messages endpoint and `tools`; low-cost and a good first non-Anthropic diagnostic. | $0.25 / $2 | OpenAI/Azure endpoints may still be removed by the key's provider or ZDR policy. [Model/API page](https://openrouter.ai/openai/gpt-5-mini/providers) |
| `google/gemini-3.1-flash-lite` | Explicitly exposes `/api/v1/messages` with tools; low cost and has Google AI Studio plus Vertex routes. | $0.25 / $1.50 | A Google-specific ZDR or provider allowlist can remove both routes. [Model/API page](https://openrouter.ai/google/gemini-3.1-flash-lite-20260507/pricing) |
| `anthropic/claude-haiku-4.5` | Same Anthropic-compatible tool path and a smaller Claude model; useful if the policy permits Bedrock/Vertex/Anthropic. | $1 / $5 | It is in the Anthropic model group, so the same Anthropic ZDR restriction may reproduce the failure. [Model/API page](https://openrouter.ai/anthropic/claude-4.5-haiku-20251001/providers) |
| `qwen/qwen3.5-flash-02-23` | Very low cost, explicit `/api/v1/messages` and tool parameters, and one direct provider makes the result easy to interpret. | $0.065 / $0.26 | One-provider routing is brittle if that provider is excluded by policy. [Model/API page](https://openrouter.ai/qwen/qwen3.5-flash-02-23/pricing) |
| `openai/gpt-5.4-mini` | Higher-quality fallback with explicit tool use and Messages support if the L3 classifier needs stronger extraction. | $0.75 / $4.50 | Use only after the cheaper smoke passes; this does not fix a policy that excludes OpenAI endpoints. [Model/API page](https://openrouter.ai/openai/gpt-5.4-mini/performance) |

Recommended experiment order is `openai/gpt-5-mini` →
`google/gemini-3.1-flash-lite` → `anthropic/claude-haiku-4.5`. Keep the model
replacement scoped to the diagnostic L3 smoke until its tool output and
readback invariants pass. The canonical production decision remains either an
approved OpenRouter policy/provider configuration for Sonnet or a direct
Anthropic/OAuth path; changing the model is not a substitute for understanding
the account policy.

## Resolution

The signed-in OpenRouter account showed no workspace guardrails, empty provider
allow/ignore lists, and no enabled ZDR switches. The old configured key was
valid but did not appear in that account's key list, which is consistent with a
legacy or different-account key carrying the failing policy.

With owner approval, a dedicated key named `warp-taskgen-r8a` was created in
the signed-in default workspace with no guardrail, a $50 total limit, and a
30-day expiry on 2026-09-08. The key value was transferred without appearing in
commands or output, installed only in ignored local and r8a `.env` files, and
the temporary transfer files were deleted.

The replacement key produced these bounded results:

- `/api/v1/key` authenticated successfully and `/models/user` returned 391
  policy-filtered models.
- The exact repository L3 `emit_target` tool request passed for the unchanged
  `anthropic/claude-sonnet-4-6` identifier; the dotted Sonnet alias and Gemini
  Flash Lite also passed.
- The correct-locality live integration wrapper passed 20 tests with 2 skips.
- The real two-task Phase 4 canary used the same Anthropic-compatible route;
  every observed `/v1/messages` call returned HTTP 200 and the job exited zero.

## Conclusion

The old key's account-level policy was the cause. The key authenticated, but no
endpoint survived its policy filtering. The benchmark, r8a, Anthropic Messages
request shape, tools, and canonical Sonnet model identifier were all proven
working with the dedicated replacement key. No model substitution or gate
bypass was necessary, and no credential value was printed or tracked.
