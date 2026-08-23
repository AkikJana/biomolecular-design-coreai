#!/usr/bin/env bash
# The three two-knob arms Section 7.16.6 lists as not run.
#
# 7.16.4 concludes the settings are synergistic -- the single-knob shares sum
# above 100% -- but says so from three one-knob arms and the full arm. Whether
# any particular pair interacts was inferred, not measured. These three arms
# complete the factorial: with reduced and full already folded, all eight cells
# of the 2x2x2 exist.
set -euo pipefail
REPO="${REPO:-$HOME/BiomolecularDesign}"
cd "$REPO"; set -a; . ./.remote_env; set +a
B="${BATCH:-12}"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "1/3 sampling + recycling  (200 / 3 / 32)"
BOLTZ_NO_KERNELS=1 BOLTZ_ACCELERATOR=gpu $PY src/settings_confound.py \
    --batch-size "$B" --sampling-steps 200 --recycling-steps 3 --msa-depth 32 \
    --run-tag samp_recyc --per-fold-budget 1800 --min-free-gib 4

log "2/3 sampling + alignment  (200 / 1 / full)"
BOLTZ_NO_KERNELS=1 BOLTZ_ACCELERATOR=gpu $PY src/settings_confound.py \
    --batch-size "$B" --sampling-steps 200 --recycling-steps 1 --msa-depth 0 \
    --run-tag samp_align --per-fold-budget 1800 --min-free-gib 4

log "3/3 recycling + alignment (10 / 3 / full)"
BOLTZ_NO_KERNELS=1 BOLTZ_ACCELERATOR=gpu $PY src/settings_confound.py \
    --batch-size "$B" --sampling-steps 10 --recycling-steps 3 --msa-depth 0 \
    --run-tag recyc_align --per-fold-budget 1800 --min-free-gib 4

log "done -- three pairwise arms, 396 folds"
