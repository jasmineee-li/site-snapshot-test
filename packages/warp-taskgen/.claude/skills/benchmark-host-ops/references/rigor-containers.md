# chrome-headless-shell containers for PVPO rigor

## The constraint

`HeadlessExperimental.beginFrame` — the CDP method that gives us deterministic, atomic per-step screenshots — is not supported on native macOS Chrome builds. On Linux it requires two launch flags: `--enable-begin-frame-control` and `--run-all-compositor-stages-before-draw`. The flags are available in Chrome's headless-shell build but not in regular headful Chrome.

The image `worldsim/docker/chrome-headless-shell.Dockerfile` builds a container that:

- Uses public Chrome-for-Testing (not Google Chrome, which has more restrictive licensing for automation).
- Launches `chrome-headless-shell` with the two required flags plus `--remote-debugging-address=0.0.0.0` and a port.
- Includes a socat forwarder to make the CDP endpoint reachable on the loopback port the orchestrator expects.

## Without the container

PVPO capture falls back to zero coverage per step: `pvpo_query.js` still runs, but `beginFrame` never fires so the PNG side of the ink-occupancy compare is empty. Every trajectory routes to placement-fix via `final_status="injection_not_encountered"`, and the placement-fix loop tries forever to find a placement that registers non-zero coverage (which never happens, because the problem is the browser, not the placement).

This behavior is **correct** in the sense that the gate is doing its job — it refuses to count an uncaptured trajectory as encountered. But the result is a run with 100% `injection_not_encountered` outcomes, which is useless for rigor analysis. If you see that pattern on a macOS developer machine, the container is missing or not running.

## Per-instance CDP endpoints

Each Phase 4 execution instance carries its own `pvpo_cdp_url` in `instances.scale.json`, typically `127.0.0.1:9222`, `127.0.0.1:9223`, … Browser-Use connects to the instance-bound endpoint, not a single shared browser. Preflight enforces uniqueness — two instances pointing at the same endpoint would race on the CDP target and corrupt each other's screenshots.

If you scale the number of parallel Phase 4 workers up, you need to add more chrome-headless-shell containers each on a unique port and add corresponding `pvpo_cdp_url` entries. The Docker compose scaling is handled by `scripts/generate_compose_scale.py` + `docker-compose.scale.yml`.

## Known CDP contract gotcha

`Animation.setPaused` semantics changed in Chrome stable during the series. The pre-fix code held the browser's animation state paused across `beginFrame` calls, which on newer Chrome blocks compositor-commit and deadlocks the pump. The fix landed in commit `19111ea8`: the pump now pauses animations only while frames are being generated, not across the whole navigation tick. If you see `beginFrame` timing out and the trajectory stuck at step 0, verify the compose config pulls a Chrome-for-Testing version at or after the version documented in the Dockerfile.

## Testing the container

- `scripts/pvpo_live_render_check.py` — live smoke test that connects to a running container, loads a page, and verifies ink-occupancy reports a non-zero coverage for known-painted text.
- `scripts/pvpo_live_validation.py` — broader validation suite.
- `tests/test_pvpo_docker_parity.py` — asserts the Dockerfile launch flags match what the orchestrator expects.

If rendering works in a headful Chrome on your dev machine but returns zero coverage through the container, the first thing to check is that the container actually has the two required flags in its launch command — re-run `docker inspect` on the container and read the `Cmd` field.
