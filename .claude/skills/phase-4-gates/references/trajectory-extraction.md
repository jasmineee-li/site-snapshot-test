# Trajectory extraction and host-API routing

## Per-task disk layout

Every Phase 4 trajectory lands on disk under `logs/<run>/<task_id>/`:

```
logs/<run>/<task_id>/
├── history.json                              # Browser-Use step-by-step agent history
├── result.json                               # final status, reward, timing
├── screenshots/step_{N}.png                  # atomic beginFrame PNGs
├── conversations/step_{N}.json               # per-step LLM conversation (Browser-Use)
└── pvpo/
    ├── capture_summary.json                  # overall capture metadata
    └── step_{N}.json                         # visibility_vec, background, clip, hasDamage
```

Additional per-trajectory artifacts written by the Phase 4 observational-metrics pass:

- `verbalized_eval_awareness_transcript.xml` + `verbalized_eval_awareness_raw_response.json` (see `needham-vea.md`).
- `transcript_purpose_raw_response.json` (Apollo Research output; see `apollo-transcript-purpose.md`).

## The extractor and AER formatter

`worldsim/phase_4/aer_trajectory_extract.py` reads `history.json` and produces a typed `ExtractedTrajectory` dataclass. `worldsim/phase_4/ipi_aer_formatter.py::format_ipi_aer_inline` renders it into the OS-Harm-style AER (action / environment / reasoning) inline text transcript that Transcript Purpose consumes.

Key properties:

- **Typed dataclasses, not raw JSON.** The legacy `trajectory_slice.py` byte-budgeted slicer was deleted in the 2026-04-19 cutover. Under Sonnet 4.6's 1M-context variant, field truncation is unnecessary; the extractor returns full fidelity and the API call passes the full transcript.
- **`extract_task_intent`** pulls the user's original task prompt out of the Browser-Use conversation for use in the AER header.
- **`as_aer_view`** projects the trajectory into the exact shape the formatter expects.

## Why no sandbox for Phase 4 judges

Refusal judge, variant generator, Transcript Purpose, VEA, and placement-fix all call the Anthropic Messages API directly via `worldsim/phase_4/anthropic_client.py`. None route through `run_claude_in_sandbox`. The cutover landed in two waves:

- **2026-04-18:** refusal judge + variant generator removed from sandbox after observed refusals on the "recommend strategies" framing. Fix: classification-only / structured-output framing + authorization preamble (citing Dziemian et al. 2026) in the prompts; single-turn API with structured tool-use output.
- **2026-04-19:** placement-fix + Transcript Purpose predecessor (Needham 5-PQ `p(eval)`) removed for the same refusal class. Placement-fix uses the same authorized single-turn API shape; Transcript Purpose uses a plain `"You are a helpful assistant."` system prompt to stay inside Apollo's protocol.

The durable rule: **no Phase 4 trajectory step runs in a Modal sandbox.** Adding one back would reopen the refusal class.

## The anthropic_client helpers

`worldsim/phase_4/anthropic_client.py` provides the single point of entry for all Phase 4 host-API calls:

- `get_client()` — returns an `AsyncAnthropic` instance. Auth precedence is decided here; never hard-code which of `CLAUDE_CODE_OAUTH_TOKEN`, `ANTHROPIC_API_KEY`, or `ANTHROPIC_AUTH_TOKEN + ANTHROPIC_BASE_URL` to use.
- `normalize_model_for_auth(model)` — coerces model IDs to what the active auth mode expects (OAuth requires different model aliases than raw API key).
- `call_with_retry(...)` — standard retry/backoff wrapper over Messages API calls.
- `classify_api_exception(exc)` — maps anthropic-sdk exceptions to hard-failure buckets (`auth` / `credits` / `quota` / `api_error`). Used by every caller's self-heal loop to decide whether to retry.

## Concurrency

`worldsim/phase_4/concurrency.py::get_api_semaphore` returns a process-wide semaphore that rate-limits concurrent Phase 4 API calls. All five callers acquire through it. If you add a sixth host-API caller, use the same semaphore — don't invent a new one, or the aggregate rate breaches the account quota under fan-out.

## What lives where

| Concern | File |
|---|---|
| Anthropic client + auth + retry + exception classification | `anthropic_client.py` |
| Shared rate-limiting | `concurrency.py` |
| Trajectory → dataclass projection | `aer_trajectory_extract.py` |
| Dataclass → AER inline text | `ipi_aer_formatter.py` |
| Trajectory → Needham-XML chat | `needham_adapter.py` + `needham_xml.py` |
| Refusal judge | `judge_api.py` |
| Variant generator | `variant_api.py` |
| Transcript Purpose | `transcript_purpose_api.py` |
| VEA | `verbalized_eval_awareness_api.py` |
| Placement-fix | `placement_api.py` |
| PVPO capture | `pvpo_capture.py`, `pvpo_query.js`, `pvpo_cdp.py`, `pvpo_browser_config.py`, `pvpo_frame_pump.py` |
| Encounter detection | `encounter_detection.py`, `ink_occupancy.py` |
| Strategy catalog (variant generator input) | `strategy_catalog.py` |
| Prompt payloads (shared text-for-model formatting) | `prompt_payloads.py` |
