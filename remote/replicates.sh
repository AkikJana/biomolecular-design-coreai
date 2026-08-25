#!/usr/bin/env bash
# Widen the noise floor from 4 receptors to all 22.
#
# Section 7.5 measured fold-to-fold noise on 4 receptors, 96 folds: pooled SD
# 0.0628 for ipTM, 1.917 for interface pLDDT. Two load-bearing numbers divide by
# that SD -- the effect-to-noise ratios in the preprint's Table 11, including the
# "8.6x" that reaches the abstract -- and an SD from 4 receptors is the least
# supported quantity under any headline in the paper.
#
# The 4 were chosen to span the outcome range (cognate ranked #1/#2/#3/#4), which
# is a reasonable spanning sample but is not a random one, and a spanning sample
# is exactly the wrong shape for estimating a spread: picking the extremes first
# inflates it. Whether that biased the SD up or down has never been checked.
#
# Same panel, same settings, same script -- only --receptors changes. 22 x 6 x 4
# = 528 folds.
set -euo pipefail
REPO="${REPO:-$HOME/BiomolecularDesign}"
cd "$REPO"; set -a; . ./.remote_env; set +a
B="${BATCH:-12}"
log() { echo "[$(date +%H:%M:%S)] $*"; }

RECS=$($PY -c "
import json
d=json.load(open('artifacts/pdb_binders_b2_n22/pdb_binder_scores.json'))
print(' '.join(sorted({x['receptor_id'] for x in d})))")
log "receptors: $RECS"

# --work-dir is separate from the 4-receptor run so the original stays intact and
# the two SDs can be compared rather than one overwriting the other.
BOLTZ_NO_KERNELS=1 BOLTZ_ACCELERATOR=gpu $PY src/seed_variance_study.py \
    --receptors $RECS \
    --replicates 4 \
    --batch-size "$B" \
    --work-dir artifacts/seed_variance_n22 \
    --summary artifacts/seed_variance_summary_n22.json \
    2>&1 | tee artifacts/seed_variance_n22.log

log "done -- 528 folds across 22 receptors"
