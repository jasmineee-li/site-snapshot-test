# Codex Handoff — PVPO beginFrame / Page.navigate deadlock

> **HISTORICAL AFTER PVPO HARD CUTOVER (2026-05-08):** This handoff describes
> the removed beginFrame architecture. Active PVPO now uses page-surface-stable
> capture on the runner-owned browser; `worldsim/phase_4/pvpo_frame_pump.py`,
> the dedicated browser container, and related tests/scripts were removed in
> commit `35ef05f3`. Do not follow this document for current setup.

**Status:** RESOLVED. Fix landed on `feat/worldsim-v5` via the per-session `HeadlessExperimental.beginFrame` pump at `worldsim/phase_4/pvpo_frame_pump.py`, wired into `AgentRunner` in `worldsim/browser_use_agent.py` and gated against atomic PVPO capture through the new `capturing` `asyncio.Event` kwarg on `atomic_capture_with_visibility`. See the "Resolution" section at the bottom for the one-paragraph summary.

**Symptom when this was open:** 85% of trajectories failed at step 1 with
`browser-use ❌ Navigation failed:` before the agent ever reasoned.

**Constraint the original handoff set (still good advice):** every claim in this doc is a
hypothesis unless backed by a specific doc, file path, commit, or test. The updates in §2 and
§4 below are now cited — if they disagree with what future-you finds at the source, the
source still wins.

---

## 1. What reproduces

**On r5 (ubuntu@3.12.221.9, SSH key `~/.ssh/webarena-key.pem`):**

```bash
cd /home/ubuntu/browser-sim
# Preconditions (already true as of 2026-04-20 23:15 local):
# - 22 pvpo-chrome-<port> containers running on host network, ports 9222-9243
# - instances.trimmed.json has pvpo_cdp_url per instance
# - WebArena stack up with Magento base_url pointing to 127.0.0.1

uv run python -m worldsim.main phase 4 \
  --instances instances.trimmed.json \
  --benchmark /home/ubuntu/.build/webarena-verified \
  --sites shopping --max-tasks-per-site 2 \
  --skip-host-bound-storage-state-auth \
  --agent-model gpt-5.4-mini --agent-provider openai --agent-service-tier priority \
  --sandbox-model claude-sonnet-4-6
```

**Observed outcome (run `20260420_224711` on r5, 22 result.json samples):**

- 17/22 `outcome_fine = task_broke_agent_exception`, `steps = 1`
- All 17 carry `errors: ["Navigation failed: "]` in `result.json`
- `final_status: injection_not_encountered` (misleading — classifier sees no encounter and maps
  `task_broke` → `injection_not_encountered`; the raw failure is in `outcome_fine`)

## 2. The mechanism (verified)

Chrome starts with the PVPO flags (`worldsim/docker/entrypoint.sh:17-19`):

```
--enable-begin-frame-control
--run-all-compositor-stages-before-draw
--disable-checker-imaging
```

Cited semantics (resolved by pulling authoritative sources during the fix):

1. **`--enable-begin-frame-control`** is defined at Chromium `headless/public/switches.h`:
   ```
   // Whether or not begin frames should be issued over DevToolsProtocol (experimental).
   inline constexpr char kEnableBeginFrameControl[] = "enable-begin-frame-control";
   ```
   *Effect:* Chrome's compositor no longer produces frames on its own scheduler. Every frame
   must be issued via `HeadlessExperimental.beginFrame`. This is what causes the deadlock.
2. **`--run-all-compositor-stages-before-draw`** is defined at Chromium
   `components/viz/common/switches.cc`: *"Effectively disables pipelining of compositor frame
   production stages by waiting for each stage to finish before completing a frame."* This
   concerns determinism, not scheduling — dropping it alone does NOT unblock navigation (the
   original handoff Fix 1 was wrong on this).
3. **`--disable-checker-imaging`** is about screenshot quality, not scheduling. Orthogonal to
   the deadlock.

How the deadlock actually manifested in browser-use 0.12.6:

1. browser-use never calls `HeadlessExperimental.beginFrame` (grep on
   `.venv/lib/python3.12/site-packages/browser_use/` returns zero matches for
   `beginFrame`/`BeginFrame`/`HeadlessExperimental`).
2. `on_NavigateToUrlEvent` in `browser_use/browser/session.py:841-949` waits for
   `Page.lifecycleEvent` (not `Page.loadEventFired` as the earlier hypothesis said) — it
   polls a per-session deque at `session.py:1007-1053`. Acceptable event names are
   `{networkIdle}` plus `{load}`/`{DOMContentLoaded}` when the `wait_until` param is set
   appropriately.
3. Before that lifecycle wait, browser-use first does `await cdp_session.cdp_client.send.Page.navigate(...)`
   wrapped in `asyncio.wait_for(..., timeout=20.0)` (`session.py:981-993`). Under a suspended
   compositor this CDP call blocks awaiting commit, trips the 20s timeout, and raises
   `RuntimeError('Page.navigate() timed out after 20s ...')` which then bubbles up and is
   logged as `Navigation failed: ...` by the outer handler at `session.py:938`.
4. PVPO only issues `beginFrame` from the per-step callback in
   `worldsim/phase_4/pvpo_capture.py::atomic_capture_with_visibility`, registered via
   `register_new_step_callback` in `worldsim/browser_use_agent.py:1063` (now `:1073` after
   the pump wiring). Step 1's navigation cannot return, so the callback never fires — the
   scheduler stays suspended indefinitely.

The 30s figure in the original log line (`timed out after 30.0s`) was the bubus dispatcher
timeout, configurable via `TIMEOUT_NavigateToUrlEvent`
(`browser_use/browser/events.py:16-43,122`), not the 20s CDP-level timeout that was actually
tripping.

## 3. What we did NOT try

Listing honestly so you don't waste time re-deriving:

- We did NOT inspect `browser-use==0.12.6` source to confirm `NavigateToUrlEvent` blocks on
  `load`. It might block on `DOMContentLoaded`, `networkIdle`, or something else entirely.
- We did NOT confirm via Chrome tracing that `load` requires a committed frame under the
  combined `--enable-begin-frame-control --run-all-compositor-stages-before-draw` flags. It's
  plausible that `load` fires regardless.
- We did NOT test a container with `--enable-begin-frame-control` but WITHOUT
  `--run-all-compositor-stages-before-draw` (they're separate flags with different effects).
- We did NOT test whether PVPO's capture fails if we remove both compositor flags (the doc
  claim is atomicity, but "atomicity" here needs a precise definition — see §5).

Do not repeat speculation that we already recorded. Go to the source for each.

## 4. Citations resolved during the fix

### 4.1 browser-use 0.12.6 navigation event handler (verified)

- `.venv/lib/python3.12/site-packages/browser_use/browser/session.py:841-949` —
  `on_NavigateToUrlEvent`, which calls `_navigate_and_wait` (lines 951-1052).
- `_navigate_and_wait` first invokes `Page.navigate()` wrapped in `asyncio.wait_for(...,
  timeout=20.0)`; on CDP-level errortext it raises `Navigation failed: <text>`; on our
  wedge case it raises `Page.navigate() timed out after 20s ...`. Both propagate up to the
  outer handler's `except Exception as e: self.logger.error(f'Navigation failed: ...')` at
  `session.py:938`, which is what the orchestrator captures as
  `errors: ["Navigation failed: "]` (with the trailing exception text truncated by the log
  format).
- After `Page.navigate()` returns a `loaderId`, the handler polls `_lifecycle_events` (a
  deque populated by `session_manager.py:899` via `Page.lifecycleEvent`) every 50ms
  (`session.py:1021`) for a matching event name. The readiness timeout is 3s same-domain,
  8s cross-domain (`session.py:975`). **If the readiness timeout fires the handler logs a
  warning and *returns without raising* (`session.py:1046-1052`) — so the observed error is
  NOT from that path.**
- The outer bubus dispatcher timeout defaults to 30s and IS configurable via env var
  `TIMEOUT_NavigateToUrlEvent` (`browser_use/browser/events.py:16-43,122`), but extending
  it alone cannot fix the deadlock — the 20s `Page.navigate()` timeout dominates.
- browser-use never calls `HeadlessExperimental.beginFrame`; repo-wide grep across the
  installed package returns zero matches. Anything that needs to drive paint must do so
  externally.

### 4.2 Chrome DevTools Protocol semantics

- Canonical reference: https://chromedevtools.github.io/devtools-protocol/
- Specific pages:
  - Page domain — `loadEventFired`, `frameStoppedLoading`, `lifecycleEvent`:
    https://chromedevtools.github.io/devtools-protocol/tot/Page/
  - HeadlessExperimental domain — `beginFrame`:
    https://chromedevtools.github.io/devtools-protocol/tot/HeadlessExperimental/#method-beginFrame
- **Claims to verify:**
  1. Does `Page.loadEventFired` require paint, or does it fire once all blocking resources
     load regardless of first paint? The CDP spec is the authoritative answer.
  2. Does `Page.lifecycleEvent` (with name `"load"` or `"networkIdle"`) offer a
     paint-independent ready signal we could use?
  3. Under `--enable-begin-frame-control`, is the default behavior "no frames until
     beginFrame" or "frames on internal schedule until explicit beginFrame arrives"? The
     flag name suggests the former but I haven't confirmed.

### 4.3 Chromium source for the flags

The Dockerfile comment cites
`headless/test/headless_compositor_browsertest.cc` ("BeginFrameControl is not supported on
MacOS yet"). Chromium source: https://source.chromium.org/. Search terms:
- `enable-begin-frame-control` (the switch registration)
- `run-all-compositor-stages-before-draw` (separate switch — do not conflate)
- `HeadlessExperimental::BeginFrame` (the command implementation)

**Claims to verify:**
- Does `--enable-begin-frame-control` strictly disable the normal frame scheduler, or does
  it only enable the `beginFrame` command while keeping the scheduler running?
- Does `--run-all-compositor-stages-before-draw` interact with the scheduler in a way that
  freezes `load` event dispatch?

### 4.4 Our PVPO code

- `worldsim/phase_4/pvpo_capture.py` — read `atomic_capture_with_visibility`. Does it call
  `HeadlessExperimental.beginFrame` eagerly, or only once per step when invoked?
- `worldsim/browser_use_agent.py` — this is where the PVPO callback is registered with
  browser-use. See around line 1190 (approximate — read the file) for the
  `get_or_create_cdp_session` + `atomic_capture_with_visibility` call.
- `worldsim/phase_4/pvpo_browser_config.py` — if any browser-launch params live here,
  relevant.
- Prior handoff: `docs/handoffs/codex-handoff-paint-verified-oracle.md` — the original PVPO
  design doc. Read §3.1 specifically for the flag rationale.

### 4.5 Our Dockerfile + entrypoint

- `worldsim/docker/chrome-headless-shell.Dockerfile`
- `worldsim/docker/entrypoint.sh`

The relevant flag block in `entrypoint.sh:14-20`. If you change the flags, rebuild the image
(the setup script stamps by sha256 of the Dockerfile, so a file change forces rebuild:
`scripts/setup_phase4_on_host.sh:112-130`).

## 5. What "PVPO rigor" actually guarantees (needs definition)

The CLAUDE.md §"What NOT to do" says:

> Rigor runs require dedicated chrome-headless-shell Docker containers because
> HeadlessExperimental.beginFrame is not supported on native macOS and the deterministic
> paint-compare depends on --enable-begin-frame-control +
> --run-all-compositor-stages-before-draw.

Before picking a fix, define precisely what invariant each flag enforces and whether the
Phase 4 measurement depends on it. The PVPO paper / design doc should have the answer, but
read the code too — what does `atomic_capture_with_visibility` actually USE?

- Does it use `beginFrame`'s `hasDamage` return to decide whether a paint happened between
  this capture and the last? If yes, does "between captures" include navigation or only the
  step-to-step delta?
- Does it use the PNG returned by `beginFrame(screenshot=true)` as the pixel ground-truth
  for ink-occupancy? If yes, does the compositor flag actually affect those pixels (by
  removing async tile paint) or is it paranoia?
- `--disable-checker-imaging` prevents checkerboard tiles — orthogonal to the deadlock. Do
  not touch it.

If the research question ("does injection content appear under agent view") can be answered
without the `--run-all-compositor-stages-before-draw` flag, dropping that one flag alone
might unblock navigation without reducing PVPO quality. Test this.

## 6. Candidate fixes, ranked by effort × risk

Each has unknowns that you must confirm before committing.

### Fix 1: keep `--enable-begin-frame-control`, drop `--run-all-compositor-stages-before-draw`

- **Unknown:** whether the scheduler-pause is caused by `--enable-begin-frame-control` alone
  or only by the combination. Verify via Chromium source (§4.3) and a live test.
- **Risk:** PVPO captures may contain mid-tile-paint artifacts. Measure against a known
  reference (`scripts/pvpo_live_render_check.py` or equivalent) before declaring safe.
- **Effort:** trivial. One-line Dockerfile edit, rebuild image, relaunch.

### Fix 2: fire `beginFrame` eagerly on `BrowserConnected`

- Patch `worldsim/browser_use_agent.py` (or the appropriate hook) to issue
  `HeadlessExperimental.beginFrame` with `noDisplayUpdates=false` every N ms from the moment
  a session connects, independent of step boundaries. That ensures paint progresses during
  navigation so `load` can fire.
- **Unknown:** whether concurrent eager beginFrame interferes with PVPO's atomic step capture
  (read `atomic_capture_with_visibility` — does it expect to be the only beginFrame caller?).
- **Risk:** complicates PVPO invariants. Need a careful read of `pvpo_capture.py`.
- **Effort:** medium. Probably ~20 lines of code + testing.

### Fix 3: extend browser-use's navigation timeout

- **Unknown:** whether the 30s timeout is configurable in 0.12.6. If it's a hardcoded
  `asyncio.wait_for(..., timeout=30.0)`, monkey-patching it upstream is fragile.
- **Risk:** if the underlying deadlock persists, this just defers the failure. Not a root-cause
  fix.
- **Effort:** small if configurable, not-worth-it if not.

### Fix 4: pin browser-use to a version where `NavigateToUrlEvent` tolerates missing `load`

- **Unknown:** whether any 0.12.x or 0.13.x release has different nav semantics. Check the
  browser-use changelog on GitHub.
- **Risk:** version churn may reintroduce bugs we already patched (the 4 PVPO bugs fixed in
  commits `29873618`, `daa61ec3`, etc.).
- **Effort:** medium. Requires re-running the integration test suite.

### Fix 5: switch to `Page.navigate` with explicit `lifecycleEvent` wait for `"DOMContentLoaded"`

- Patch browser-use's nav handler to return once DOM is parsed, not waiting for `load`. DOM
  parse doesn't require paint.
- **Unknown:** whether this introduces agent-observability issues (e.g., agent inspects DOM
  before images render, making visual tasks unreliable).
- **Risk:** deviation from stock browser-use; upstream drift.
- **Effort:** medium. Requires patching the installed package or forking.

## 7. Non-negotiable constraints

- Respect `CLAUDE.md`. Do not regress:
  - Phase 4 admission gate (`STRICT_FEASIBILITY_ADMISSION`, `feasibility.status=verified`)
  - Direct Messages API for judge/variant/TP/VEA/placement-fix (no sandbox regressions)
  - Per-worker PVPO isolation (`daa61ec3`)
  - Payload-text sync in `_merge_variant_task` (fixed in `daa61ec3`)
- Do not touch WebArena compose, nginx proxy config, or the SG without explicit ask.
- Do not skip `uv lock --check` by editing `uv.lock` by hand. If deps change, regenerate.
- Do not write debug code to main paths; gate all diagnostic logging behind a flag that
  defaults off.
- If you change `worldsim/docker/*`, stamping via `scripts/setup_phase4_on_host.sh:112-130`
  will detect and force a rebuild — verify the rebuild happens end-to-end.

## 8. How to prove your fix works

Do all three before declaring done:

1. **Synthetic test (cheapest):** adapt `scripts/pvpo_live_render_check.py` — it already
   proves end-to-end capture against live Magento with a known payload. Post-fix it should
   still report `max_coverage: 1.0` on a known-visible snippet. If PVPO quality regressed,
   this catches it.
2. **Single-trajectory test:** one-task Phase 4 invocation against a shopping task, priority
   tier. Confirm `result.json.errors` is empty and `steps > 1`. No `Navigation failed:`.
3. **Multi-worker test:** 4-task × 2-site calibration (shopping + gitlab, `--max-tasks-per-site
   2`). Same criteria plus the 4th task must also complete past step 1. This catches
   regressions in per-worker isolation.

All three must pass before you report the fix as landed.

## 9. Tools + commands you'll want

- **SSH to r5:** `ssh -i ~/.ssh/webarena-key.pem ubuntu@3.12.221.9`
- **Recent full-run logs:**
  `/home/ubuntu/browser-sim/logs/phase_4/full_20260420_224710.log` (the failed run)
- **Recent completed trajectories:**
  `/home/ubuntu/browser-sim/logs/phase_4/20260420_224711/` (22 result.json samples)
- **Inspect a specific task's crash:**
  `cat /home/ubuntu/browser-sim/logs/phase_4/20260420_224711/adv-008/result.json | jq '.errors, .outcome_fine, .steps, .final_status'`
- **Chrome container flags:**
  `docker inspect pvpo-chrome-9222 --format '{{json .Config.Cmd}}'`
- **Rebuild image after Dockerfile change:**
  `cd /home/ubuntu/browser-sim && docker build -t worldsim/chrome-headless-shell:latest -f worldsim/docker/chrome-headless-shell.Dockerfile .`
  Then `docker rm -f $(docker ps -aq --filter name=pvpo-chrome) && bash scripts/setup_phase4_on_host.sh --instances instances.trimmed.json --skip-gitlab-mint` (step 3 relaunches containers).
- **uv venv entry:** `export PATH="$HOME/.local/bin:$PATH"` on r5.

## 10. Cost + time spent so far so you can judge priority

- Calibration + partial 174 run: ~$10 in OpenAI priority burn across the failed nav events
- Infrastructure debugging this session: ~8 hours of pipeline work (per-worker PVPO isolation,
  Magento base_url fix, auth validation_endpoint fix, etc.) — all productive and committed,
  unrelated to this bug
- **Rigor Phase 4 run is blocked on this fix.** The research question needs agent trajectories
  to actually advance past step 1.

## 11. Out-of-scope (verified already shipped, 2026-04-20)

Rechecked when the fix landed; none of these needed work in this branch:

- `scripts/generate_compose_scale.py` **already emits** `api_auth.validation_endpoint`.
  See `_default_validation_endpoint(site_name)` at lines 291-295 (returns
  `/rest/V1/modules` for `shopping`/`shopping_admin` and `/api/v4/user` for `gitlab`), and
  the application at lines 332-343.
- `scripts/setup_phase4_on_host.sh` **already runs** `docker run … --network host` in
  committed code — see lines 198-199 of the file. The inspect at lines 190-196 also
  recreates any container that's not `host`-networked.
- Preflight signature fixes **already landed** in commit `04cfe215`
  (`fix(phase4): preflight test signatures + skip removed playwright dep`, 2026-04-20).

## 12. Final reminder

Every mechanism claim in this document is a hypothesis unless you confirm it against a
specific file, docs page, or test. **Do not propagate unverified claims into code.** When
you write your diagnosis + fix, cite:

- The exact browser-use file + line where navigation waits for its readiness signal
- The exact CDP docs URL that defines what that readiness signal actually waits on
- The exact Chromium source/switch definition for any flag you touch
- The PVPO design doc passage that justifies (or doesn't) retaining each flag

If any of those citations are missing, your fix is not finished.

## 13. Resolution (2026-04-20)

**Fix:** `feat(phase4): per-session beginFrame pump to unblock PVPO navigation deadlock`.

- New module `worldsim/phase_4/pvpo_frame_pump.py` exposes an
  `async with frame_pump(session) as capturing:` context manager. It spawns a background
  task that issues `HeadlessExperimental.beginFrame` with default parameters every
  `WORLDSIM_PVPO_FRAME_PUMP_MS` milliseconds (default 50, set to `0` to disable) against
  `session.agent_focus_target_id`. The yielded `capturing` :class:`asyncio.Event` is held
  `set()` by `atomic_capture_with_visibility` for the duration of its virtual-time-paused
  capture so the pump doesn't race the atomic screenshot.
- `worldsim/phase_4/pvpo_capture.py::atomic_capture_with_visibility` grows a
  `capturing: asyncio.Event | None = None` kwarg, sets it before `setVirtualTimePolicy(pause)`,
  clears it in `finally` (wrapped around the existing advance) so that any exception still
  resumes the pump.
- `worldsim/browser_use_agent.py::AgentRunner._run_task_impl` wraps the `Agent(...).run(...)`
  block in `async with frame_pump(self._session) as capturing:` and threads `capturing`
  into `_make_pvpo_step_callback`. The `async with` teardown runs inside the existing
  `try/finally`, so the pump always stops before `self._session.kill()` is called.

**Citations the fix is built on:**

- `headless/public/switches.h` — `kEnableBeginFrameControl` comment: *"Whether or not begin
  frames should be issued over DevToolsProtocol (experimental)."*
- `components/viz/common/switches.cc` — `kRunAllCompositorStagesBeforeDraw` comment:
  *"Effectively disables pipelining of compositor frame production stages by waiting for
  each stage to finish before completing a frame."*
- CDP docs, HeadlessExperimental.beginFrame — params `frameTimeTicks` (opt), `interval`
  (opt, default ~16.666ms), `noDisplayUpdates` (opt, default false), `screenshot` (opt).
  Returns `{hasDamage, screenshotData}`. `enable/disable` methods are deprecated; no
  explicit domain enable is required.
- browser-use 0.12.6 `browser_use/browser/session.py:841-1052` — the navigation handler
  analyzed above.

**Why the sidecar pump (instead of patching browser-use):** forking or monkey-patching
`.venv/` would be wiped by the next `uv sync` and would bypass our source-of-truth hygiene.
See `docs/upstream-patches/browser-use-on-navigation-tick.md` for the ready-to-submit
upstream PR that would eventually let us delete this sidecar.

**Tests:** `tests/test_phase_4_pvpo_frame_pump.py` (9 tests), plus two additions to
`tests/test_phase_4_pvpo_capture.py` covering the `capturing` event success and error
paths. Existing tests updated: `test_pvpo_callback_writes_artifacts_on_browser_use_success_path`
now asserts the new `capturing=None` kwarg flows through.

**Non-regressions preserved:** `test_atomic_capture_sequences_cdp_calls_and_extracts_bg`,
`test_atomic_capture_resumes_virtual_time_on_error`, `test_atomic_capture_has_damage_false_does_not_retry`,
`test_atomic_capture_accepts_browser_use_cdp_session_surface` all still pass.
