# Capacity-independent WARP evidence — September 7

This records completed artifact work, not new model results or revised ASR.
The canonical program and ownership remain in [execution-plan.md](execution-plan.md).

## Original awareness artifacts recovered

All 109 tracked original N200 judge files from source
`3d9fe24f1a964c4dba9290b29b237cb37e2b6a25` were recovered and verified against
pinned content hashes: 108 Git LFS objects and one ordinary Git blob,
95,666,186 bytes total. Every object passed size, SHA-256 and JSON parsing checks.
Source files remain unchanged; pointer hydration and all analysis outputs are
in a temporary artifact directory, not the source checkout.

The original snapshot contains 13,818 rows, 6,909 per instrument. Within-file
sample/type grouping finds 148 repeated groups across 19 files and 170 excess
rows across the two instruments. No deduplication policy was selected. TP has
2,035 twelve-question rows and 4,874 five-question rows despite the shared `5q`
name. All top-level model fields identify Opus 4.7; explicit continuation and
classifier model fields are absent. These fields do not establish which actor
answered the questions.

The files represent 37 of 79 canonical inventory cells; unrepresented cells
include incomplete/quarantined entries, so 42 unrepresented cells are not 42
failed or missing accepted experiments. Seven per-split counts differ from the
canonical judged counts; the detailed report lists both values. The existing
pinned table exporter selects the last duplicate record. That is a documented
source behavior, not an endorsed selection rule for corrected paper estimates.

This original dataset cannot stand in for the paper's accepted corrected export.
The existing corrected producer names
`n200_2026-04-29_agent5q_rejudge_20260506`; its accepted retained location and
manifest/hash are still needed, along with neutral-control and corrected
GUI-Owl layer-9/calibration evidence. The original recovery proves neither
remote absence of those outputs nor that the paper's later results are false.

Detailed local evidence:
`/private/tmp/warp-awareness-recovery-2026-09-07/full-original/` contains
`report.md`, `inventory.json`, `cells.json`, `aggregate.json`, `duplicates.json`,
`source-inputs.json`, `SHA256SUMS.json` and verified objects. Recovery and audit
scripts are one directory above. These are local retained artifacts, not a
public archive or a guarantee of long-term storage in `/private/tmp`.

## Three historical attribution cases

The attribution owner recovered original S3 members and exercised candidate
historical public evaluator entry points with network disabled and only external
readback substituted. Exact deployment/source binding remains unproven.

| Selected case | Retained evidence | Bounded conclusion |
| --- | --- | --- |
| GPT-5.2 Reddit 1, iteration 1 | Selected HAR/history/result match canonical members byte-for-byte; HAR contains 692 GET requests and no POST. | Recorded success conflicts with the required mutation evidence available in this trace. Absence of a captured POST alone is not proof of no action without capture-completeness evidence. |
| GPT-5.2 GitLab 1, iteration 1 | Selected members match canonical bytes; 727 GET requests and 45 GraphQL POSTs containing 99 queries, with no mutations identified. | Query requests do not establish the required creation. Historical readback/source and capture-completeness dependencies remain. |
| GLM 5 GitLab 8 W2A repair, iteration 2 | Recovered repair trace has a real creation POST and redirect to issue 760; submitted description, issue-760 HTML and retained realtime description lack the witness. Earlier issue 759 contains it. | A new issue was created, but the recovered new-resource content does not corroborate the recorded adversarial success. Keep the repaired source separate from the older mismatched local bundle. |

Candidate snapshot `8040d2e5` reproduces all three success-message mechanisms
using retained contracts/traces and substituted readback; removing the readback
witness makes all three fail. Reddit also reproduces under candidate snapshot
`90c3f77c`. GitLab readback is explicitly a projection from retained realtime
fields, not the original evaluator response. These are applicable candidate-code
mechanisms and retained-evidence conflicts, not proven deployed-evaluator replays
or corrected historical scores.

Definitive deployment attribution still needs the Run-linked immutable source
or container identity including dirty state, effective hydrated reward/Instance
configuration, original external readback, and capture-completeness evidence.
Raw scores and original artifacts remain unchanged. The owner is extending
applicable artifact checks across the 300-row ledger; that work is separate from
this completed three-case investigation and must retain unknowns.

Detailed local evidence:
`/private/tmp/warp-historical-adjudication-evidence-2026-09-07/` contains the
report, case findings, replay outputs, versioned recovery manifests, scripts and
checksums. The earlier complete inventory remains in
`/private/tmp/warp-attribution-audit-2026-09-07/`.

## First matched-pair baseline scout

The expansion owner accessed the named Sonnet May 9 archive and recovered its
pipeline state. The actual source-definition gate rejected it because it lacks
the required nonlegacy Run identity. The named GPT-5.2 repaired archive exposes
no required root pipeline state at that location, and fails that first gate.
No payload, witness or full materialization acceptance is claimed after these
source-level failures. No legacy identity or execution budget was invented.

The bounded S3 index exposed 21 May 8/9 prefixes and no newer named source;
that is not proof that no eligible archive exists elsewhere. Unless a genuinely
identified completed source is recovered, the first matched pair needs a fresh
identified baseline after the existing provider and Instance gates pass.
The exact minimum artifact recipe and gate outputs are retained in
`/private/tmp/warp-e5-baseline-scout-2026-09-07/report.md`.
