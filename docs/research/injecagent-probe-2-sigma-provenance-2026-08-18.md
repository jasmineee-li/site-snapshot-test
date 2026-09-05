# Where the README's "about 2σ" probe number comes from — 2026-08-18

## Why this note exists

`README.md` says, in [Motivation](../../README.md#motivation):

> A residual-stream probe on GUI-Owl-1.5-32B-Think separates attacked from safe
> InjecAgent trajectories at about 2σ, so the awareness shows up directly in the
> model's activations.

The claim is real and it is published. The defect this note fixes is that a
reader inside this repository could not follow it to anything. The commit that
introduced the sentence, `9e693057`, cited "icml_final_2.tex and the canonical
analysis dir": the first resolves only in a separate paper repository, and the
second resolves to nothing anywhere.

This note records the source, the numbers, the arithmetic behind "about 2σ", and
— just as importantly — what is *not* in this tree.

## The source

The number is from the WARP paper source, `icml_final_2.tex`, which lives in a
separate paper repository and is **not checked into this tree**. The published
PDF the README already links (<https://jasminexli.com/warp-paper>) carries the
same prose and the same appendix table.

Two places in that source:

- The prose sentence, in the probing results, `\paragraph{Finding 4: The probe
  almost always fires during evaluations.}`
- The appendix table `tab:probe_by_attack_success`, "Probe scores split by
  attack outcome, per InjecAgent arm."

A local `vendors/worldsim-paper/` checkout, if one exists, mirrors an earlier
draft of the same paragraph. `vendors/` is excluded per clone through
`.git/info/exclude` and is not part of this repository, so it is not a citable
location either.

## The numbers

Probe configuration, from the paper: last-user-token span, layer 9, eval-aware
threshold 16.375. Cell format is mean ± std, with `n` per group.

| Arm | Group | n | last_user_token |
|---|---|--:|---|
| Baseline | safe | 63 | 19.183 ± 1.02 |
| Baseline | attacked | 37 | 17.081 ± 0.37 |
| Steer +0.9 | safe | 68 | 19.131 ± 1.14 |
| Steer +0.9 | attacked | 32 | 17.021 ± 0.30 |
| Steer −0.9 | safe | 58 | 19.123 ± 1.00 |
| Steer −0.9 | attacked | 42 | 17.051 ± 0.66 |
| Prompt-aware | safe | 67 | 19.128 ± 1.02 |
| Prompt-aware | attacked | 33 | 17.059 ± 0.45 |

The README sentence is the baseline arm, which the prose quotes as 17.08 ± 0.37
(attacked) versus 19.18 ± 1.02 (safe).

## What "about 2σ" means

It is the gap between the two group means divided by the **safe group's own
standard deviation** — the wider of the two, so this is the conservative
denominator:

```
(19.183 − 17.081) / 1.02 = 2.102 / 1.02 = 2.06
```

Produced by:

```bash
uv run python -c "
import math
m_safe, s_safe, n_safe = 19.183, 1.02, 63
m_att, s_att, n_att = 17.081, 0.37, 37
gap = m_safe - m_att
print('gap                        ', round(gap, 4))
print('gap / safe sd              ', round(gap / s_safe, 4))
print('gap / attacked sd          ', round(gap / s_att, 4))
sd_pooled = math.sqrt(((n_safe-1)*s_safe**2 + (n_att-1)*s_att**2) / (n_safe+n_att-2))
print('pooled sd (n-weighted)     ', round(sd_pooled, 4), '-> d =', round(gap/sd_pooled, 4))
sd_avg = math.sqrt((s_safe**2 + s_att**2) / 2)
print('pooled sd (equal-n form)   ', round(sd_avg, 4), '-> d =', round(gap/sd_avg, 4))
"
```

Output:

```
gap                         2.102
gap / safe sd               2.0608
gap / attacked sd           5.6811
pooled sd (n-weighted)      0.8417 -> d = 2.4973
pooled sd (equal-n form)    0.7672 -> d = 2.7397
```

So "about 2σ" is **not** a pooled-standard-deviation effect size — Cohen's *d*
lands at 2.50 or 2.74 depending on the pooling convention — and it is **not** an
AUROC. Read it as the plain-language statement the paper makes: the two group
means are about two safe-group standard deviations apart.

## What the paper discloses against itself

The same appendix reports that on the **first-assistant-mean** span the
direction flips: attacked trajectories score higher (≈17.1) than safe ones
(≈16.7). The paper's reading is that complying with an injection itself surfaces
eval-awareness in generation, even when the input-time signal was below the
resist threshold.

The last-user-token split holds in all four arms in the table above — unsteered,
±steered, and prompt-aware — so the separation is not a single arm picked after
the fact. Steering moves ASR without moving the within-arm safe-vs-attacked gap.

## What is not in this tree

- **No raw per-trajectory probe scores.** Nothing in this repository lets you
  recompute the means, the standard deviations, or the 2.06 above from data. The
  table is the artifact; this note is its receipt.
- **Not the probe that produced it.** `probes/trained/` is gitignored except for
  `.gitkeep` and `manifest.yaml`, and the one probe checked in is
  `opencua-72b-chat-template-span/` — OpenCUA-72B, not GUI-Owl.
- **Not the same run as the in-tree GUI-Owl probe record.**
  `eval_awareness_experiments/docs/handoffs/HANDOFF_GUI_OWL_PROBE_RESULTS.md` (2026-05-04) describes GUI-Owl probes
  trained on the contrastive and framing-pair datasets, with best layers 47 and
  42 and a failed cross-format validation. It does not record the layer-9
  last-user-token InjecAgent scoring behind the README sentence. Do not read
  that handoff as the source for this number.

Reproducing the number from scratch means retraining the probe and re-scoring
the InjecAgent trajectories on GPU — see `probes/train_probe.py`,
`probes/score_probe.py`, and `eval_awareness_experiments/docs/handoffs/HANDOFF_CUA_EVAL_AWARENESS.md`.

## Verification

Values above were read directly from `icml_final_2.tex` (prose paragraph and
`tab:probe_by_attack_success`) on 2026-08-18 and transcribed unchanged. The in-
tree statements were checked with:

```bash
git ls-files probes/trained | sed 's#/mcq/.*##' | sort -u
grep -n "trained" .gitignore
```
