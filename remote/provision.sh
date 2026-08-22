#!/usr/bin/env bash
# Set up a rented GPU box to run this project's folds. Idempotent.
#
# Run ON the rented instance, after sync.sh has pushed the repo.
#   bash remote/provision.sh
set -euo pipefail

REPO="${REPO:-$HOME/BiomolecularDesign}"
cd "$REPO"

echo "== host =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || {
  echo "no nvidia-smi -- is this actually a GPU box?"; exit 1; }
python3 --version
df -h . | tail -1

echo "== python env =="
# Rented images usually ship torch already, matched to their CUDA. Reusing it
# beats reinstalling: a pip-resolved torch often mismatches the driver.
if python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "system torch is CUDA-enabled, reusing it"
    PY=python3
else
    echo "system torch unusable, building a venv with torch from PyPI"
    python3 -m venv .venv-gpu
    PY=.venv-gpu/bin/python
    $PY -m pip install -q --upgrade pip
    $PY -m pip install -q torch
fi

$PY -m pip install -q boltz gemmi biopython scipy scikit-learn pandas pyyaml numpy
echo "PY=$PY" > .remote_env

echo "== caches on the big volume =="
# Model weights and MSAs are several GB; keep them off the small root disk.
VOL="${VOL:-$REPO/.cache}"
mkdir -p "$VOL"/{torch,hf,boltz}
{
  echo "export TORCH_HOME=$VOL/torch"
  echo "export HF_HOME=$VOL/hf"
  echo "export BOLTZ_CACHE=$VOL/boltz"
} >> .remote_env

echo "== check =="
set -a; . ./.remote_env; set +a
$PY -c "import torch, boltz; print(' torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0))"
echo "provisioned. next: bash remote/bench.sh"
