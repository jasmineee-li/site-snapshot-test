# Upstream browser-use PR — `on_navigation_tick` hook

**Status:** draft, ready-to-submit. Not applied upstream. WorldSim ships the equivalent
behavior locally in `worldsim/browser_use_agent.py` by patching
`BrowserSession._navigate_and_wait`; `worldsim/phase_4/pvpo_frame_pump.py` now only
provides the shared capture event/coordinator context in production.

**Target repo:** https://github.com/browser-use/browser-use (the package we pin via `pyproject.toml`).

**Pinned version we care about:** `0.12.6`. If a later version has added `firstPaint` / lifecycle workarounds, revalidate against those first.

## Motivation

Chrome launched with `--enable-begin-frame-control` (the `kEnableBeginFrameControl` switch defined at `headless/public/switches.h`: `// Whether or not begin frames should be issued over DevToolsProtocol (experimental).`) only produces compositor frames on explicit `HeadlessExperimental.beginFrame` calls. This is the required mode for deterministic headless rendering (paint-verified evaluation, pixel-diffing test harnesses, etc.).

browser-use 0.12.6 never issues `HeadlessExperimental.beginFrame` — a repo-wide grep of the installed package returns zero matches for `beginFrame`, `BeginFrame`, or `HeadlessExperimental`. As a result, navigation on such a Chrome instance stalls: `Page.navigate()` blocks awaiting commit, `Page.lifecycleEvent(load)` never fires (no paint), and the built-in 20s `Page.navigate` timeout in `browser_use/browser/session.py:981-993` trips.

An evaluation harness can work around this by spawning a sidecar coroutine that drives `beginFrame` on the focused target, but that duplicates browser-use's per-target CDP bookkeeping and races its tab-switch logic. A small upstream hook lets callers plug a per-tick callback into browser-use's existing navigation path.

## Proposed minimal diff

```diff
--- a/browser_use/browser/session.py
+++ b/browser_use/browser/session.py
@@
 class BrowserSession:
-    def __init__(
+    def __init__(
         self,
         ...,
+        navigation_tick_cb: Callable[[CDPSession], Awaitable[None]] | None = None,
+        navigation_tick_interval_s: float = 0.05,
         ...,
     ) -> None:
         ...
+        self._navigation_tick_cb = navigation_tick_cb
+        self._navigation_tick_interval_s = navigation_tick_interval_s

     async def on_NavigateToUrlEvent(self, event: NavigateToUrlEvent) -> None:
         ...
         try:
             ...
-            await self._navigate_and_wait(
+            tick_task = self._start_navigation_tick(target_id)
+            try:
+                await self._navigate_and_wait(
                 event.url,
                 target_id,
                 timeout=event.timeout_ms / 1000 if event.timeout_ms is not None else None,
                 wait_until=event.wait_until,
                 nav_timeout=event.event_timeout,
-            )
+                )
+            finally:
+                if tick_task is not None:
+                    tick_task.cancel()
+                    with contextlib.suppress(asyncio.CancelledError):
+                        await tick_task

+    def _start_navigation_tick(self, target_id: str) -> asyncio.Task | None:
+        if self._navigation_tick_cb is None:
+            return None
+
+        async def _tick_loop() -> None:
+            while True:
+                try:
+                    cdp_session = await self.get_or_create_cdp_session(
+                        target_id, focus=False
+                    )
+                    await self._navigation_tick_cb(cdp_session)
+                except asyncio.CancelledError:
+                    raise
+                except Exception:
+                    self.logger.debug("navigation_tick_cb raised", exc_info=True)
+                await asyncio.sleep(self._navigation_tick_interval_s)
+
+        return asyncio.create_task(_tick_loop(), name="browser-use-nav-tick")
```

That is the complete behavioral change. No existing caller is affected (default `None` means the ticker is never spawned). The tick task is scoped to a single navigation and cancelled when it returns; no lifecycle leak.

## Suggested test plan (to include in the upstream PR)

1. Default: `BrowserSession()` with no `navigation_tick_cb` — verify no task named `browser-use-nav-tick` is ever created during a standard navigation (regression guard).
2. With a stub callback that increments a counter: navigate to a page that's known to take >150ms to paint; assert the counter is >= 2 by the time the navigation returns.
3. With a callback that raises: assert navigation still completes (the loop swallows exceptions at debug level).
4. Cancel-on-return: assert the tick task is no longer running after `on_NavigateToUrlEvent` exits, regardless of success or failure.
5. Integration against a Chrome launched with `--enable-begin-frame-control`: with the callback wired to `HeadlessExperimental.beginFrame`, navigation completes within the default timeout; without the callback, it times out. This is the motivation-repro.

## PR description template

```
Allow callers to run a per-tick callback during navigation

Adds an optional `navigation_tick_cb: Callable[[CDPSession], Awaitable[None]] | None`
kwarg on `BrowserSession`. When set, the callback is invoked every
`navigation_tick_interval_s` (default 50ms) against the target's CDP
session for the duration of a single navigation and is cancelled as
soon as the navigation returns.

Motivation: Chrome launched with `--enable-begin-frame-control` only
produces frames on explicit `HeadlessExperimental.beginFrame` calls.
browser-use never issues that command, so navigation blocks waiting
for a paint that never happens. The new hook lets evaluation harnesses
and test tools drive the compositor without patching browser-use
internals or forking the package.

Non-breaking: default is `None` → no task is spawned, existing
behavior is unchanged. Covered by new tests for the default path,
callback invocation, error swallowing, task cleanup, and integration
against `--enable-begin-frame-control`.
```

## Deprecation path for our sidecar

Once this lands upstream and our `pyproject.toml` bumps to a `browser-use` version containing it:

1. Replace WorldSim's local `_navigate_and_wait` patch with the upstream
   `navigation_tick_cb` kwarg and pass the same beginFrame callback through
   `BrowserSession(...)`.
2. Keep the `WORLDSIM_PVPO_FRAME_PUMP_MS=0` kill switch: map that to the
   upstream `navigation_tick_interval_s=0` (or pass `navigation_tick_cb=None`)
   for parity.
3. Once the upstream path is confirmed green on a rigor run, delete `_pump_loop`
   / `_pump_once` and leave `frame_pump` as the capture-context wrapper only.
4. After two green rigor runs on the upstream path, delete the monkey-patch shim
   entirely.
