# WASP E2E Replay ASR Snapshot

Completed arms only: `bare`, `xml_safety`, and `xml_scenario`. Counts are deduped by `(arm, model, task_id)` across the lane/resume replay output directories.

## Per-Model ASR

| Model | `bare` | `xml_safety` | `xml_scenario` |
|---|---:|---:|---:|
| Opus 4.7 | 44/168 (26.2%) | 44/168 (26.2%) | 44/168 (26.2%) |
| Sonnet 4.6 | 44/168 (26.2%) | 44/168 (26.2%) | 44/168 (26.2%) |
| Gemini 2.5 Pro | 45/168 (26.8%) | 44/168 (26.2%) | 44/168 (26.2%) |
| Kimi K2.5 | 44/168 (26.2%) | 44/168 (26.2%) | 44/168 (26.2%) |
| GPT-5.2 | 51/168 (30.4%) | 44/168 (26.2%) | 44/168 (26.2%) |
| GLM-5 | 44/168 (26.2%) | 44/168 (26.2%) | 44/168 (26.2%) |

## Total ASR

| Metric | `bare` | `xml_safety` | `xml_scenario` |
|---|---:|---:|---:|
| n | 1008 | 1008 | 1008 |
| Success | 272 | 264 | 264 |
| ASR | 27.0% | 26.2% | 26.2% |
