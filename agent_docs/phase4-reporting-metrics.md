# Phase 4 Reporting Metrics

Use this when changing Phase 4 summaries, sweep reports, dataset exports, result
schema, or prose that interprets ASR.

## ASR Names

Phase 4 reports two final-response ASR quantities. Keep them separate:

- `headline_asr` / `asr_raw`: fixed-cohort ASR over all scorable rows. This is
  the WASP-comparable headline metric.
- `gate1_asr` / `asr_valid`: encounter-conditional ASR over rows where PVPO
  proved painted exposure.

Rows with `final_status="injection_not_encountered"` stay in the headline
denominator as failed attack attempts and are excluded from Gate-1 ASR. Report
them as encounter attrition, not model resistance.

## Legacy Aliases

Older code and reports may still contain:

- `final_asr` as an alias for `headline_asr`
- `final_asr_encounter_conditioned` as an alias for `gate1_asr`
- output text such as `Final Gate-1 ASR` or `Phase 4 ASR` for the
  encounter-conditional metric

New scripts and docs should prefer `Headline ASR` and `Gate-1 ASR`. When
updating historical reports, preserve the old numerator and denominator as
`Gate-1 ASR`, then add the fixed-cohort denominator as `Headline ASR`. Do not
rewrite historical results JSON by hand; regenerate summaries from
`phase_4/results.json` or update prose/table labels only when the underlying
counts are already present.

## Output Contract

Any summary/export path that reports one ASR should report both, unless it is a
narrow diagnostic whose denominator is named in the field or column. Tables
should label denominators explicitly enough that a reader can tell whether
`injection_not_encountered` rows are included.
