#!/usr/bin/env bash
# Time a handful of folds at full settings and project the whole run.
#
# Do this FIRST, in the first ten minutes of the rental. Every estimate of what
# the full run costs is a guess until this prints a real number.
set -euo pipefail
REPO="${REPO:-$HOME/BiomolecularDesign}"
cd "$REPO"; set -a; . ./.remote_env; set +a

N="${N:-6}"
echo "timing $N folds at full settings (200 steps, 3 recycles, full MSA)…"
START=$(date +%s)
BOLTZ_NO_KERNELS=1 $PY src/settings_confound.py \
    --batch-size "$N" --sampling-steps 200 --recycling-steps 3 --msa-depth 0 \
    --run-tag bench --labels cognate --per-fold-budget 1800 --min-free-gib 4
END=$(date +%s)

SEC=$(( END - START ))
PER=$(( SEC / N ))
echo
echo "  $N folds in ${SEC}s  ->  ${PER}s per fold"
echo "  354 folds  ->  $(( PER * 354 / 3600 ))h $(( (PER * 354 % 3600) / 60 ))m"
echo
echo "  at \$0.40/hr that is about \$$(( PER * 354 * 40 / 360000 ))"
echo "  (MSA fetch is included above and is cached, so later folds run faster)"
