# Upstream cdp-use issue - cancellation-aware request futures

**Status:** draft note. WorldSim works around this locally by shielding
PVPO CDP deadlines and draining late protocol responses.

**Observed symptom:** `cdp_use.client WARNING Received duplicate response for
request <id> - ignoring`.

## Root Cause

`cdp_use.CDPClient.send_raw()` stores a Future in `pending_requests`, sends a
CDP message, and awaits that Future. If a caller wraps the await in plain
`asyncio.wait_for(...)`, timeout cancellation marks the Future done/cancelled.
When Chrome later sends the normal response for that request id, cdp_use finds
the id still in `pending_requests` but the Future is already done, then logs the
response as a duplicate.

That warning is usually not a protocol duplicate. It means the local caller
abandoned or cancelled its wait before Chrome replied.

## Desired Upstream Behavior

`send_raw()` should be cancellation-aware:

- On caller cancellation, remove the request id from `pending_requests` or move
  it into a short-lived cancelled-request tombstone map.
- When a late response arrives for a tombstoned id, consume it at debug level as
  a late response for a cancelled request, not as a duplicate response warning.
- Keep request-id allocation and websocket send atomic enough for concurrent
  callers on one event loop.

## WorldSim Workaround

WorldSim treats CDP timeouts as local deadlines, not protocol cancellation. For
PVPO-owned CDP calls it now uses `asyncio.wait_for(asyncio.shield(task), ...)`,
then attaches a done callback to consume any late result or exception. Stateful
`HeadlessExperimental.beginFrame` calls remain serialized by
`BeginFrameCoordinator`; read/probe calls may be abandoned locally but still
drained when Chrome replies.

This preserves research semantics: a timeout is observable runtime telemetry,
not a reason to mutate the task, weaken PVPO, or hand-edit an outcome.
