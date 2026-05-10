# Placement-Fix Cutover Integration Summary, 2026-04-19

> **Historical integration summary.** This note records why the placement-fix
> API cutover was accepted at the time. It predates the current strict
> GitLab/Reddit WASP scope and should be used only as provenance for the
> placement-fix path, not as an active runbook or benchmark result.

## Scope

This run validated the Phase 4 placement-fix cutover that moved placement-fix
LLM calls onto the direct host Anthropic Messages API path. The relevant code
surface was the Phase 4 placement-fix loop, its prompt/schema handling, and the
shared Anthropic API infrastructure used by Phase 4 judges.

The live gate used a private benchmark host and the standard integration
wrapper:

```bash
bash scripts/run_integration_tests.sh --host-config configs/benchmark_hosts/r5.yaml
```

The exact historical host address, temporary security-group changes, and local
operator IPs are intentionally omitted from this public-facing summary. They
were operational access details, not scientific evidence.

## Result

The integration run completed with:

```text
16 passed, 6 failed, 6 skipped
```

The passed coverage included:

- the live Phase 4 judge API smoke path,
- GitLab and Reddit seed resolver checks,
- editor seed plumbing for the in-scope browser-agent flow,
- mocked placement-fix unit coverage for schema, failure bucketing, raw
  response persistence, and shared cost/summary handling.

The 6 failures were all in the live Phase 2 feasibility suite. They were
classified as pre-existing host/data health issues rather than placement-fix
regressions:

- two out-of-scope shopping/Magento seed calls returned request failures,
- two out-of-scope shopping/Magento oversize checks returned request failures
  instead of length-classified failures,
- the mixed-site concurrency check inherited those shopping failures,
- a GitLab cleanup assertion found residual projects from prior test runs.

These failures were outside the placement-fix code path and outside the current
GitLab/Reddit WASP-only scope.

## Interpretation

The evidence supported shipping the placement-fix API cutover relative to the
Phase 4 integration surface because the live test exercised the same direct
Anthropic API primitives used by placement-fix:

- `AsyncAnthropic` client construction,
- retry/error classification,
- semaphore/cost accounting,
- forced-tool structured response parsing,
- Phase 4 summary synthesis.

The run did **not** include a dedicated live task engineered to trigger the
placement-fix loop end to end. That remained a reasonable future improvement,
but was not treated as a blocker because unit tests covered the placement-fix
loop itself and the live judge smoke covered the shared API path.

## Current Guidance

Do not use this note to justify running out-of-scope shopping, Magento, or
shopping-admin carriers. Current active scope is GitLab and Reddit/Postmill UGC
surfaces only. For present-day Phase 4 setup and live-run procedure, use:

- `agent_docs/remote-runs.md`
- `docs/handoffs/rigor-run-setup.md`
- `docs/warp-taskgen-technical-spec.md`
