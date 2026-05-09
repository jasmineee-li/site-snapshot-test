# Phase 4 ASR Naming Migration

Phase 4 now reports two ASR quantities explicitly:

- `headline_asr` / `asr_raw`: fixed-cohort ASR over all scorable rows. This is
  the WASP-comparable headline metric.
- `gate1_asr` / `asr_valid`: encounter-conditional ASR over rows where PVPO
  proved painted exposure.

The legacy text labels `Final Gate-1 ASR` and `Phase 4 ASR` referred to the
encounter-conditional metric. Command output keeps those labels as compatibility
aliases for one migration window, but new scripts and docs should prefer
`Headline ASR` and `Gate-1 ASR`.

When updating historical reports, preserve the old numerator and denominator as
`Gate-1 ASR`, then add the fixed-cohort denominator as `Headline ASR`. Do not
rewrite historical results JSON by hand; regenerate summaries from
`phase_4/results.json` or update prose/table labels only when the underlying
counts are already present in the report.
