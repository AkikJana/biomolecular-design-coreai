#!/bin/bash
# Resume the two unfinished decomposition arms under memory pressure.
#
# The first attempt stalled with swap at 15.9 of 16.4 GB and the folding process
# at 0.01 GB resident. Two changes rather than a plain retry:
#
#   Order. The sampling arm runs first. It subsamples the alignment to 32 rows,
#   so it is the light one, and it is also the arm that carries the effect --
#   worth having if only one completes. The alignment arm holds alignments of up
#   to 14,464 rows in memory and is what actually stalled.
#
#   Batch size. The heavy arm drops to 3 folds a batch from 6, halving peak
#   residency. It costs one extra model construction per six folds, which is
#   cheap next to thrashing.
#
# Both arms resume by fold name, so nothing already folded is repeated.
set -u
cd "$HOME/Developer/BiomolecularDesign" || exit 1
PY="$PWD/.venv/bin/python"

folds () { "$PY" -c "
import json,pathlib
f=pathlib.Path('artifacts/settings_confound/scores$1.json')
print(len(json.loads(f.read_text())) if f.exists() else 0)" 2>/dev/null; }

reclaimable () { "$PY" -c "
import subprocess,re
o=subprocess.run(['vm_stat'],capture_output=True,text=True).stdout
g=lambda k:int(re.search(rf'{k}:\s+(\d+)',o).group(1))
print(int((g('Pages free')+g('Pages inactive'))*4096/1e9))" 2>/dev/null; }

run () {   # tag sampling recycling msa batch label
  local tag=$1 s=$2 r=$3 m=$4 bs=$5 name=$6 n
  n=$(folds "$tag")
  if [ "${n:-0}" -ge 132 ]; then echo "=== ${name} already 132, skipping ==="; return; fi
  echo "=== ${name}: resuming at ${n}/132, batch ${bs}, $(df -g . | tail -1 | awk '{print $4}') GiB disk, $(reclaimable) GB reclaimable RAM — $(date +%H:%M:%S) ==="
  "$PY" -u src/settings_confound.py --run-tag "${tag}" \
      --sampling-steps "${s}" --recycling-steps "${r}" --msa-depth "${m}" \
      --labels cognate,scrambled,decoy --batch-size "${bs}" \
      >> "artifacts/decomp${tag}.log" 2>&1
  echo "=== ${name} finished $(date +%H:%M:%S) at $(folds "$tag")/132 ==="
}

run _samp 200 1 32 6 "SAMPLING (light: MSA 32)"
run _msa   10 1  0 3 "ALIGNMENT (heavy: full MSA, small batches)"

echo
echo "=== analysing ==="
"$PY" src/settings_decomposition.py || true
echo "=== RESUME QUEUE COMPLETE $(date +%H:%M:%S) ==="
