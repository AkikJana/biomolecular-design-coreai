#!/bin/bash
# Which of the three raised settings carries Section 7.13's effect?
#
# 7.13 raised sampling steps, recycling passes and alignment depth together and
# measured a 3-7x larger standardised effect. It could not say which knob did it.
# Section 7.11's geometry result -- 14% physically plausible backbone bonds at
# ten steps against 99.7% at two hundred -- points at sampling steps, and this
# tests that directly.
#
# Three arms, each moving ONE knob from the reduced baseline (10 steps, 1
# recycling, MSA 32). The two endpoints already exist:
#   reduced 10/1/32   -> boltz1_scramble_result.json  (Section 7.8)
#   full    200/3/full-> settings_confound.json       (Section 7.13)
#
# --labels cognate,scrambled runs the scramble control only: 66 folds instead of
# 132, and it is the test the suppression figure is computed from.
#
# Sampling first, because it is the hypothesis. If the machine dies after one
# arm, that is the arm worth having.
set -u
cd "$HOME/Developer/BiomolecularDesign" || exit 1
PY="$PWD/.venv/bin/python"

run_arm () {           # tag  sampling  recycling  msadepth  label
  local tag=$1 s=$2 r=$3 m=$4 name=$5
  local n
  n=$("$PY" -c "
import json,pathlib
f=pathlib.Path('artifacts/settings_confound/scores${tag}.json')
print(len(json.loads(f.read_text())) if f.exists() else 0)" 2>/dev/null)
  if [ "${n:-0}" -ge 66 ]; then
    echo "=== ${name} already complete (${n} folds), skipping ==="; return
  fi
  local free; free=$(df -g . | tail -1 | awk '{print $4}')
  if [ "$free" -lt 7 ]; then
    echo "=== ABORT: ${free} GiB free, refusing to start ${name} ==="; exit 1
  fi
  echo "=== ${name}: ${s} steps / ${r} recycling / MSA ${m} — start $(date +%H:%M:%S), ${free} GiB free ==="
  "$PY" -u src/settings_confound.py --run-tag "${tag}" \
      --sampling-steps "${s}" --recycling-steps "${r}" --msa-depth "${m}" \
      --labels cognate,scrambled --batch-size 6 \
      >> "artifacts/decomp${tag}.log" 2>&1
  n=$("$PY" -c "
import json,pathlib
f=pathlib.Path('artifacts/settings_confound/scores${tag}.json')
print(len(json.loads(f.read_text())) if f.exists() else 0)" 2>/dev/null)
  echo "=== ${name} finished $(date +%H:%M:%S) with ${n}/66 folds ==="
}

run_arm _samp 200 1 32 "SAMPLING only"
run_arm _msa   10 1  0 "ALIGNMENT only"
run_arm _recyc 10 3 32 "RECYCLING only"

echo
echo "=== all arms done; analysing ==="
"$PY" src/settings_decomposition.py || true
echo "=== DECOMPOSITION COMPLETE $(date +%H:%M:%S) ==="
