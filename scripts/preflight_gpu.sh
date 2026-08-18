#!/usr/bin/env bash
# Confirm this host can run the models on its GPU. Exit non-zero if it cannot.
#
# Why this is a blocking check and not a report:
#
# A driver that is too old for the installed CUDA build does not raise. Instead
# `torch.cuda.is_available()` returns False, and the loaders fall back to CPU:
#
#   probes/model_loader.py    device = "cuda" if torch.cuda.is_available() else "cpu"
#   models/common/hf_openai_server.py   same pattern
#
# The run then completes on the wrong device and looks normal. This happened
# once already, on driver 565.57.01 — see docs/HANDOFF_GUI_OWL_PROBE_RESULTS.md,
# gotcha 3. A 32B model on CPU wastes the whole booking.
#
# The floor is 580.95.05, not the 580.65.06 that CUDA 13.0 GA requires.
# `torch` 2.11.0 pins `cuda-toolkit==13.0.2`, and CUDA 13.0 Update 2 raises the
# floor. Both numbers come from NVIDIA's table:
# https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
#
# `torch>=2.2` is a default dependency and the lock holds 2.11.0, so this
# applies to every GPU host, not only hosts that install the `[cua]` extra.
#
# Usage:
#   bash scripts/preflight_gpu.sh              # before training or serving
#   bash scripts/preflight_gpu.sh --server     # again after a server starts
#
# Set PREFLIGHT_ALLOW_CPU=1 to skip. Use that only on a machine you intend to
# run on CPU.

set -euo pipefail

REQUIRED_DRIVER="580.95.05"
MIN_SERVER_MIB=60000

fail() { echo "[preflight] FAIL: $*" >&2; exit 1; }
pass() { echo "[preflight] ok: $*"; }

if [[ "${PREFLIGHT_ALLOW_CPU:-0}" == "1" ]]; then
    echo "[preflight] PREFLIGHT_ALLOW_CPU=1 — skipping. Do not use this for a real run."
    exit 0
fi

# 1. The driver must be new enough for the CUDA build torch carries.
command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found. This host has no usable NVIDIA driver."

DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d '[:space:]')
[[ -n "$DRIVER" ]] || fail "nvidia-smi reported no driver version."

# sort -V puts the lower version first. If the lowest is the required one, the
# installed driver is at or above the floor.
LOWEST=$(printf '%s\n%s\n' "$REQUIRED_DRIVER" "$DRIVER" | sort -V | head -1)
if [[ "$LOWEST" != "$REQUIRED_DRIVER" && "$DRIVER" != "$REQUIRED_DRIVER" ]]; then
    fail "driver $DRIVER is below the required $REQUIRED_DRIVER.
       The model will load on CPU without raising. Update the driver.
       Do not reinstall torch from a cu126 index: that breaks vLLM serving."
fi
pass "driver $DRIVER (floor $REQUIRED_DRIVER)"

# 2. torch must see the GPU. This is an assert, so a false result stops the run.
python - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit(
        "torch.cuda.is_available() is False. The model would load on CPU.\n"
        f"       torch {torch.__version__}, built against CUDA {torch.version.cuda}"
    )
cuda = torch.version.cuda or ""
print(f"[preflight] ok: torch {torch.__version__}, CUDA {cuda}, {torch.cuda.device_count()} device(s)")

# 3. A live bf16 matmul. A driver can enumerate a GPU it cannot compute on.
a = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
if not torch.isfinite(a @ a).all():
    sys.exit("a bf16 matmul on the GPU produced a non-finite result.")
print("[preflight] ok: bf16 matmul on device")
PY

# 4. After a server starts, confirm the weights are resident on the GPU.
if [[ "${1:-}" == "--server" ]]; then
    USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    if (( USED < MIN_SERVER_MIB )); then
        fail "only ${USED} MiB of GPU memory is in use.
       A 32B model in bf16 needs about 63 GB resident. The weights are on CPU,
       or the server is not up yet."
    fi
    pass "${USED} MiB resident on GPU"
fi

echo "[preflight] PASS"
