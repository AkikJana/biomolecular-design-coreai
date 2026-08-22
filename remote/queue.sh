#!/usr/bin/env bash
# The full run: expanded panel at full settings, then the Chai-1 comparison arm.
# Resumable -- the runners skip folds already scored, so re-running after an
# interruption picks up where it stopped. Safe on spot instances.
set -euo pipefail
REPO="${REPO:-$HOME/BiomolecularDesign}"
cd "$REPO"; set -a; . ./.remote_env; set +a

BATCH="${BATCH:-12}"
log() { echo "[$(date +%H:%M:%S)] $*"; }

log "1/2 expanded panel, full settings"
$PY src/settings_confound.py \
    --batch-size "$BATCH" --sampling-steps 200 --recycling-steps 3 --msa-depth 0 \
    --run-tag expanded --per-fold-budget 1800 --min-free-gib 4

log "2/2 held-out panel, full settings, third draw"
$PY src/heldout_panel.py \
    --batch-size "$BATCH" --sampling-steps 200 --recycling-steps 3 --msa-depth 0 \
    --base boltz1 --run-tag full3

log "done. pull results with sync.sh down"
