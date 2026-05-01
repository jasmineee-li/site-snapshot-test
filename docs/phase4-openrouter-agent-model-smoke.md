# Phase 4 OpenRouter Agent Model Smoke Matrix

Date: 2026-05-01

Purpose: record which OpenRouter model slugs have passed local smoke checks for the Phase 4 Browser Use web-agent path. This is operational compatibility evidence only. It is not a Phase 4 benchmark result, not a PVPO encounter result, and not evidence of IPI resistance or compliance.

## Smoke Scope

- Provider auth: `OPENROUTER_API_KEY` loaded from local `.env`, never printed.
- Direct API smoke: `https://openrouter.ai/api/v1/chat/completions`.
- Browser agent smoke: `worldsim.browser_use_agent.BrowserUseAgent` with Browser Use `0.12.6`, `use_vision=False`, local `data:` page, no WebArena site, no PVPO, no IPI payload.
- Browser task: read a simple page containing `Code: ...` and finish with the code.
- Runtime controls: local smoke used explicit Browser Use deadlines, usually `llm_timeout=120` and `step_timeout=180`.

## Model Slugs

| Requested model | OpenRouter slug used | Direct structured smoke | Browser Use agent smoke | Notes |
| --- | --- | --- | --- | --- |
| Opus 4.7 | `anthropic/claude-opus-4.7` | Passed | Passed | Claude-family OpenRouter models must use the Anthropic Messages-compatible endpoint, not generic `ChatOpenRouter`, to avoid schema keyword rejection. |
| Sonnet 4.6 | `anthropic/claude-sonnet-4.6` | Passed | Passed | Same Claude-family routing rule as Opus 4.7. |
| GLM 5 | `z-ai/glm-5` | Passed | Passed | One local Browser Use smoke had one structured parse retry, then completed. Treat as smoke-compatible, not reliability-proven. |
| GPT 5.2 | `openai/gpt-5.2` | Passed | Passed | Direct chat required `max_tokens >= 16` through the upstream provider. |
| Gemini 2.5 Pro | `google/gemini-2.5-pro` | Passed | Passed | Smoke-compatible through generic `ChatOpenRouter`. |
| Kimi K2.5 | `moonshotai/kimi-k2.5` | Passed | Passed | Use this slug for Kimi K2.5. It is reasoning-enabled. |

Previous exploratory smoke also passed `moonshotai/kimi-k2-thinking`, but the intended Kimi K2.5 entry is `moonshotai/kimi-k2.5`.

### Unsupported Historical Smoke

MiniMax M2.7 (`minimax/minimax-m2.7`) passed the toy local direct and Browser
Use smoke checks, but the full 32-task Phase 4 WebArena sweep showed broad
Browser Use/action-schema and final-answer contract incompatibility: 28/32
tasks broke and 31/32 final answers were plain prose. Do not include this slug
as a supported Phase 4 Browser Use harness condition unless a future adapter
change proves reliable structured actions and final-answer compliance on a
live WebArena task set.

## Kimi K2.5 Reasoning Smoke

OpenRouter model metadata for `moonshotai/kimi-k2.5` reported:

- `context_length`: `262144`
- `supported_parameters` included `reasoning`, `response_format`, `structured_outputs`, `tools`, `tool_choice`, and `parallel_tool_calls`.

Raw streaming request with:

```json
{
  "model": "moonshotai/kimi-k2.5",
  "messages": [
    {
      "role": "user",
      "content": "How many r's are in the word strawberry? Reply with the number only."
    }
  ],
  "reasoning": {
    "enabled": true
  },
  "stream": true,
  "max_tokens": 256
}
```

returned:

- final content: `3`
- `usage.completion_tokens_details.reasoning_tokens`: `90`
- reasoning delta chunks observed: `108`
- reasoning details chunks observed: `108`

Edge case: the same request with `max_tokens=64` returned reasoning telemetry but no final content because the output budget was consumed by reasoning tokens. For reasoning-enabled OpenRouter models, smoke and benchmark runs need enough output budget for both reasoning and final action JSON.

## Implementation Notes

- Slash-qualified model IDs such as `z-ai/glm-5` and `google/gemini-2.5-pro`
  route to OpenRouter.
- Claude-family OpenRouter slugs route through `ChatAnthropic` with `base_url=https://openrouter.ai/api` and `auth_token=$OPENROUTER_API_KEY`.
- Generic OpenRouter `ChatOpenRouter` uses OpenAI-style `response_format=json_schema`. That path rejected Browser Use's strict action schema for Claude-family models because the schema includes integer constraints such as `minimum`.
- Opus 4.7 sampling handling must omit explicit `temperature` for both `claude-opus-4-7` and `anthropic/claude-opus-4.7` style slugs.
- Browser Use's current OpenRouter adapter returns parsed structured content and usage, but does not preserve OpenRouter `reasoning_details` across multi-turn tool-use calls. This does not block basic Browser Use smoke, but it means reasoning-model long-horizon continuity is not fully proven by this smoke matrix.

## Sources

- OpenRouter Kimi K2.5 API page: https://openrouter.ai/moonshotai/kimi-k2.5/api
- OpenRouter reasoning tokens guide: https://openrouter.ai/docs/guides/best-practices/reasoning-tokens
- OpenRouter model metadata endpoint: https://openrouter.ai/docs/guides/overview/models
