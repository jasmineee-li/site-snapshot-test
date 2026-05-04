# Handoff: GUI-Owl-1.5-32B-Think probe results

Status as of 2026-05-04. Picks up from `HANDOFF_CUA_EVAL_AWARENESS.md`
(the run guide for the broader causal experiment).

## TL;DR

We trained both DoM probes (MCQ + framing-pair) on
`mPLUG/GUI-Owl-1.5-32B-Think` and ran a cross-format validation. Each
probe hits **AUROC = 1.0** on its own held-out split, but they fail to
classify each other's data (**AUROC ≈ 0.50, both directions**) and their
best-layer vectors are **nearly orthogonal** (cos ≈ −0.04 / −0.01).

**Verdict: both probes overfit their own training distribution.** They
do not point at a shared "eval-awareness" representation. As-is, they
are not reliable steering vectors for the planned causal experiment.

## What was done

1. **RunPod setup on a fresh pod** (network volume at `/workspace`).
   `bash scripts/runpod_setup.sh` runs cleanly after the two fixes
   listed below.
2. **Pre-downloaded** `mPLUG/GUI-Owl-1.5-32B-Think` (~63 GB, 26 files)
   into `/workspace/hf_cache`.
3. **Trained MCQ probe** on 204-pair `probes/data/contrastive_dataset.json`
   → `probes/trained/gui-owl-32b/` (gitignored — reproduce by retraining).
   - Best layer **47**, AUROC **1.0000** on 41 held-out pairs.
   - Top-6 layers (by AUROC): `[47, 50, 62, 61, 53, 59]`.
4. **Trained framing-pair DoM probe** on 16-pair `probes/data/framing_pairs.json`
   → `probes/trained/gui-owl-32b-framing/`.
   - Best layer **42**, AUROC **1.0000** on 4+4 held-out prompts.
   - Top-6 layers: `[42, 43, 47, 48, 5, 32]`.
   - Note: held-out is only 4 vs 4. AUROC=1.0 here is nearly trivial.
5. **Cross-format validation** (`probes/validate_probes_cross.py`):

   | Test | AUROC | Mean separation |
   |---|---|---|
   | MCQ probe → framing prompts | **0.5000** | mean(real)=−2.83, mean(fake)=−2.86, Δ=0.03 |
   | Framing probe → MCQ prompts | **0.5020** | mean(eval)=−8.69, mean(deploy)=−8.68, Δ=0.005 |

6. **Cosine similarity between probe vectors:**
   - At MCQ best layer (47): **−0.0454**
   - At framing best layer (42): **−0.0100**
   - README's hypothesis was cos > 0.5. Strongly rejected.

## Why this matters for the causal experiment

The matrix script (`scripts/run_cua_eval_awareness_matrix.sh`) cells 3
and 4 work by ablating `−α·v` from the residual stream, where `v` is
the probe's best-layer vector. The whole interpretation —
"eval-awareness was causally responsible for the behavior change" —
depends on `v` actually pointing at eval-awareness.

If `v` is an artifact direction specific to the probe's training prompts
(which is what AUROC=0.5 on cross-format suggests), then steering with
`v` pushes the model in *some* direction that affects behavior, but not
necessarily the direction we want. The downstream ASR deltas would be
real but not interpretable as "removing eval-awareness."

## Recommended next steps (ranked by informativeness)

### 1. Run the same cross-test on OpenCUA-32B *(highest priority, ~45 min)*

If both probes generalize on OpenCUA but neither does on GUI-Owl, this
is a model-specific finding (probably about Qwen3-VL vs Qwen2.5-VL
representations) and the experiment can proceed on OpenCUA. If neither
generalizes on OpenCUA either, the methodology has a deeper issue and
the framing probe pipeline needs revisiting before any steering work.

```bash
cd /workspace/browser-sim
source .venv/bin/activate
set -a && source .env && set +a
export HF_HOME=/workspace/hf_cache

# Pre-download (one-time; ~64 GB):
python -c "from huggingface_hub import snapshot_download; snapshot_download('xlangai/OpenCUA-32B')"

# Train both probes:
bash scripts/train_probe_opencua_32b.sh                  # MCQ probe
bash scripts/train_probe_framing_opencua_32b.sh          # framing probe

# Cross-validate:
python -m probes.validate_probes_cross \
    --model xlangai/OpenCUA-32B \
    --mcq-probe probes/trained/opencua-32b/ \
    --framing-probe probes/trained/opencua-32b-framing/
```

### 2. Within-format generalization test on MCQ *(~15 min, doesn't need a new model)*

Even on the format the probe was trained for, does it generalize to
held-out questions? Split the 204 MCQ pairs into 10 random folds, train
on 9, score the 10th, and look at fold-mean AUROC. If even *within*
format the probe is at ~0.5 on held-out questions (as opposed to the
fixed-split AUROC=1.0 we got), the probe is overfitting at the dataset
level, not just at the format level.

This script doesn't exist yet — quick to write on top of
`probes/train_probe.py`. ~10 min of work + 1 hour of compute (10× model
loads), or ~15 min if you load the model once and reuse it across folds.

### 3. Compare to the `eval-awareness` reference repo

`probes/` was ported from
[`jasmineee-li/eval-awareness`](https://github.com/jasmineee-li/eval-awareness)
(`eval-awareness-testbed/src/eval_awareness_probes/`). The original was
text-only; this version added Qwen2.5-VL / Qwen3-VL handling. If the
reference repo reported successful cross-format transfer on text-only
models, then something about the multimodal port is the culprit:

- Token positions for VL models (chat template renders differently)
- BOS / image-token handling in `to_tokens()`
- `output_hidden_states` semantics differing between Qwen2_5_VLForConditionalGeneration and a plain causal LM

Reading those tests / results from the original repo would help bound
the problem before investing in dataset expansion.

### 4. (Lower priority) Expand `framing_pairs.json` to ≥50 pairs

The 16-pair dataset gives a 4+4 held-out split, where AUROC=1.0 is
basically meaningless. Even if the cross-format issue is solved, you
want a more robust framing dataset. The original Hua et al. paper's set
should be extensible — same template, different surface forms.

## Reproduction recipe (if probe weights need to be regenerated)

Probe weights are gitignored (`probes/trained/*`). To reproduce
exactly what was done:

```bash
# 0. Fresh pod with /workspace volume mounted, an A100/H100 80GB,
#    NVIDIA driver supporting CUDA 12.6+ (see "Gotchas" below).
cd /workspace
git clone https://github.com/jasmineee-li/browser-sim.git
cd browser-sim
git checkout claude/general-session-rsdA2

# 1. Setup. Idempotent.
bash scripts/runpod_setup.sh

# 2. Fix torch — see Gotchas. Default install pulls cu130 wheels which
#    are incompatible with the current pod's driver.
source .venv/bin/activate
uv pip uninstall torch torchvision torchaudio
uv pip install --index-url https://download.pytorch.org/whl/cu126 \
    torch torchvision torchaudio

# 3. Add HF_TOKEN to .env (.env is gitignored — do this manually).
echo "HF_TOKEN=hf_..." >> .env

# 4. Pre-download model (~63 GB).
set -a && source .env && set +a
export HF_HOME=/workspace/hf_cache
python -c "from huggingface_hub import snapshot_download; snapshot_download('mPLUG/GUI-Owl-1.5-32B-Think')"

# 5. Train both probes (~10 min each, sequential — single GPU).
python -m probes.train_probe \
    --model mPLUG/GUI-Owl-1.5-32B-Think \
    --data probes/data/contrastive_dataset.json \
    --dataset-format mcq \
    --output probes/trained/gui-owl-32b/ \
    --layers all --top-n-layers 6 --eval-split 0.2 --dtype bfloat16

python -m probes.train_probe \
    --model mPLUG/GUI-Owl-1.5-32B-Think \
    --data probes/data/framing_pairs.json \
    --dataset-format framing \
    --output probes/trained/gui-owl-32b-framing/ \
    --layers all --token-position -2 --top-n-layers 6 --eval-split 0.25 --dtype bfloat16

# 6. Cross-validate (~5 min).
python -m probes.validate_probes_cross \
    --model mPLUG/GUI-Owl-1.5-32B-Think \
    --mcq-probe probes/trained/gui-owl-32b/ \
    --framing-probe probes/trained/gui-owl-32b-framing/
```

## Gotchas (all hit during this run, all fixed)

1. **`pyproject.toml` had no setuptools packages declaration.**
   `uv pip install -e .` fails with "Multiple top-level packages
   discovered in a flat-layout." Fixed in commit `b6042827`.

2. **`probes/model_loader.py` only handled Qwen2.5-VL.**
   GUI-Owl-1.5 ships as `Qwen3VLConfig`; the Qwen2.5-VL loader raised
   `AttributeError: 'Qwen3VLVisionConfig' object has no attribute
   'fullatt_block_indexes'`. Fixed in commit `8a977969` — now dispatches
   on `cfg_hf.model_type`.

3. **CUDA driver vs torch wheel mismatch.**
   The `[cua]` extras pull `vllm 0.20.1`, which pulls `torch 2.11.0+cu130`.
   This pod's driver was 565.57.01 (max CUDA 12.7), so
   `torch.cuda.is_available()` returned `False` and the model silently
   loaded on CPU. Workaround: reinstall torch with cu126 wheels (step 2
   in the recipe above). This breaks vLLM serving but probe extraction
   doesn't need vLLM. If you also need vLLM, provision a pod with a
   ≥570.x driver instead.

4. **`runpod_setup.sh` does `git checkout claude/general-session-rsdA2`.**
   Running it auto-switches the working tree off whatever branch you
   were on. Be aware before running on a non-clean tree.

5. **`n_layers=64, d_model=None` log line is cosmetic.**
   Qwen3-VL nests `hidden_size` under `text_config`, so the
   `_get_config_value` walk doesn't find it. The DoM probe doesn't use
   `d_model`; activation extraction works fine.

## Files added / changed in this work

| File | Status | Description |
|---|---|---|
| `pyproject.toml` | modified (`b6042827`) | Added `[tool.setuptools]` packages list |
| `probes/model_loader.py` | modified (`8a977969`) | Added Qwen3-VL dispatch |
| `probes/validate_probes_cross.py` | new (this commit) | Cross-format validation CLI |
| `HANDOFF_GUI_OWL_PROBE_RESULTS.md` | new (this commit) | This file |

## Open questions for the collaborator

- **Is this a known issue with the framing-pair methodology?** The
  paper (Hua et al. 2025) reports cross-prompt-template stability of
  the eval-awareness direction in their text-only models. Whether they
  validated against MCQ-style probes is worth checking.
- **Does `vllm` need to stay in `[cua]` extras?** If probe extraction
  is the primary use of `[cua]`, dropping vllm avoids the cu130 issue
  entirely. vLLM could move to its own `[serve]` extra.
- **Should `framing_pairs.json` be expanded before more model runs?**
  16 pairs is too few for a stable held-out signal regardless of which
  model you're probing.
