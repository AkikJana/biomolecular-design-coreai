#!/bin/bash
# Add the decoy folds to each decomposition arm, so the RANK test decomposes too.
#
# Section 7.16 ran --labels cognate,scrambled: 66 folds an arm, enough for the
# scramble control that 7.13's suppression figure is computed from, but not for
# the receptor-specificity test, which needs each cognate ranked against its own
# decoys. 7.16.5 records that as a limitation. Section 7.13.2b calls the ranking
# result "the practical conclusion of the dissertation", so decomposing it is
# worth 66 more folds an arm.
#
# Each run resumes its existing store: the 66 cognate+scramble folds are already
# present by name, so only the decoys are folded.
set -u
cd "$HOME/Developer/BiomolecularDesign" || exit 1
PY="$PWD/.venv/bin/python"

run () {   # tag sampling recycling msa label
  local tag=$1 s=$2 r=$3 m=$4 name=$5 n
  n=$("$PY" -c "
import json,pathlib
f=pathlib.Path('artifacts/settings_confound/scores${tag}.json')
print(len(json.loads(f.read_text())) if f.exists() else 0)" 2>/dev/null)
  if [ "${n:-0}" -ge 132 ]; then echo "=== ${name} already 132, skipping ==="; return; fi
  local free; free=$(df -g . | tail -1 | awk '{print $4}')
  if [ "$free" -lt 7 ]; then echo "=== ABORT: ${free} GiB free ==="; exit 1; fi
  echo "=== ${name} decoys: start $(date +%H:%M:%S), at ${n}/132, ${free} GiB free ==="
  "$PY" -u src/settings_confound.py --run-tag "${tag}" \
      --sampling-steps "${s}" --recycling-steps "${r}" --msa-depth "${m}" \
      --labels cognate,scrambled,decoy --batch-size 6 \
      >> "artifacts/decomp${tag}.log" 2>&1
  n=$("$PY" -c "
import json,pathlib
f=pathlib.Path('artifacts/settings_confound/scores${tag}.json')
print(len(json.loads(f.read_text())) if f.exists() else 0)" 2>/dev/null)
  echo "=== ${name} done $(date +%H:%M:%S) at ${n}/132 ==="
}

run _recyc 10 3 32 "RECYCLING"      # cheapest first: if the machine dies, most arms are done
run _msa   10 1  0 "ALIGNMENT"
run _samp 200 1 32 "SAMPLING"

echo
echo "=== analysing ==="
"$PY" src/settings_decomposition.py || true
echo "=== DECOY QUEUE COMPLETE $(date +%H:%M:%S) ==="
