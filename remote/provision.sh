#!/usr/bin/env bash
# Set up a rented CUDA box to run this project's folds. Idempotent.
#
# Run ON the rented instance, after sync.sh has pushed the repo:
#   bash remote/provision.sh
#
# Everything below that looks defensive is a fix for something that actually
# went wrong on a RunPod RTX 4090, in the order it bit.
set -euo pipefail

REPO="${REPO:-$HOME/BiomolecularDesign}"
cd "$REPO"

echo "== host =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "no nvidia-smi -- is this actually a GPU box?"; exit 1; }
python3 --version
df -h . | tail -1

echo "== python env =="
if python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    # Keep the image's torch: it is built against the host driver, and a
    # pip-resolved one usually is not. These images also mark system Python
    # externally managed (PEP 668), so pip refuses to install into it at all.
    # A venv with --system-site-packages inherits the working torch and still
    # lets everything else install normally.
    #
    # The venv goes on local disk, never the network volume. RunPod's /workspace
    # is MooseFS; pip writes tens of thousands of small files and doing that over
    # the network takes minutes instead of seconds. Weights and MSAs still belong
    # on /workspace -- few files, large, worth keeping across a restart.
    echo "system torch is CUDA-enabled, inheriting it into a venv"
    VENV="${VENV:-/root/.venv-gpu}"
    python3 -m venv --system-site-packages "$VENV"
    PY="$VENV/bin/python"
    $PY -m pip install -q --upgrade pip
else
    echo "system torch unusable, building a venv with torch from PyPI"
    VENV="${VENV:-/root/.venv-gpu}"
    python3 -m venv "$VENV"
    PY="$VENV/bin/python"
    $PY -m pip install -q --upgrade pip
    $PY -m pip install -q torch
fi

TORCH_BEFORE=$($PY -c "import torch; print(torch.__version__)")
echo "  torch before installs: $TORCH_BEFORE"

$PY -m pip install -q boltz gemmi biopython scipy scikit-learn pandas pyyaml numpy

# Boltz's fused triangle-attention kernels import cuequivariance_ops_torch, a
# CUDA-only wheel that is not a dependency of this project because the MPS path
# never touches those kernels. On a CUDA host the failure appears at CALL time,
# not import time, so every import check passes and every fold still dies.
#
# Installing it unpinned is worse than not installing it: it dragged torch 2.13
# (CUDA 13) over the image's 2.8.0+cu128 and broke torch outright against a
# 12.8 driver. --no-deps is what makes this safe.
echo "== optional CUDA kernels =="
if [ "${WITH_KERNELS:-0}" = "1" ]; then
    $PY -m pip install -q --no-deps \
        cuequivariance-torch cuequivariance-ops-cu12 cuequivariance-ops-torch-cu12 || true
    TORCH_AFTER=$($PY -c "import torch; print(torch.__version__)" 2>/dev/null || echo BROKEN)
    if [ "$TORCH_AFTER" != "$TORCH_BEFORE" ]; then
        echo "  kernels moved torch $TORCH_BEFORE -> $TORCH_AFTER; reverting"
        $PY -m pip uninstall -y -q torch triton 2>/dev/null || true
    fi
    NO_KERNELS=0
else
    echo "  skipped (set WITH_KERNELS=1 to try them)"
    NO_KERNELS=1
fi

echo "== caches on the big volume =="
VOL="${VOL:-$REPO/.cache}"
mkdir -p "$VOL"/{torch,hf,boltz}
{
  echo "PY=$PY"
  echo "export TORCH_HOME=$VOL/torch"
  echo "export HF_HOME=$VOL/hf"
  echo "export BOLTZ_CACHE=$VOL/boltz"
  [ "$NO_KERNELS" = "1" ] && echo "export BOLTZ_NO_KERNELS=1"
} > .remote_env

echo "== check =="
set -a; . ./.remote_env; set +a
$PY -c "import torch; print(f'  torch {torch.__version__} | cuda {torch.cuda.is_available()} | {torch.cuda.get_device_name(0)}')"

# A smoke fold, because every static check passed on a box that could not fold
# anything. Thirty seconds here beats discovering it from an empty result file.
echo "== smoke fold =="
SMOKE=$(mktemp -d)
mkdir -p "$SMOKE/in"
cat > "$SMOKE/in/smoke.yaml" <<YAML
version: 1
sequences:
  - protein: {id: A, sequence: SQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVVVNQQESSDSGTSVSEN, msa: empty}
  - protein: {id: B, sequence: SQETFSDLWKLLPEN, msa: empty}
YAML
ARGS=(--model boltz1 --accelerator gpu --output_format pdb --override
      --recycling_steps 1 --sampling_steps 10 --diffusion_samples 1
      --num_workers 0 --preprocessing-threads 1)
[ "$NO_KERNELS" = "1" ] && ARGS+=(--no_kernels)
if $PY -m boltz.main predict "$SMOKE/in" --out_dir "$SMOKE/out" "${ARGS[@]}" 2>&1 | tail -2 \
   | grep -q "failed examples: 0"; then
    echo "  smoke fold: OK"
    rm -rf "$SMOKE"
else
    echo "  smoke fold FAILED -- do not start the queue; rerun boltz by hand to see why"
    rm -rf "$SMOKE"
    exit 1
fi

echo "provisioned. next: bash remote/bench.sh"
