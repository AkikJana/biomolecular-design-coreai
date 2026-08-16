#!/bin/bash
# Two more independent draws of the held-out panel at full settings.
#
# Section 7.10's five reduced-settings draws range over 38% (ipTM) to 89%
# (receptor side) of their mean, so one draw cannot pin a retention figure --
# it can only say whether the penalty is still there. Three draws is the
# minimum that supports a number.
#
# Folds are unseeded, so re-running the identical panel gives an independent
# draw. --seed 0 is kept so the same 132 pairs are folded each time; it seeds
# panel construction, not diffusion.
set -u
cd "$HOME/Developer/BiomolecularDesign" || exit 1
PY="$PWD/.venv/bin/python"

echo "=== waiting for the in-flight draw to finish ==="
while pgrep -f "heldout_panel.py" > /dev/null; do sleep 60; done
echo "in-flight draw done at $(date +%H:%M:%S)"

for tag in _full2 _full3; do
  n=$("$PY" -c "
import json,pathlib
f=pathlib.Path('artifacts/heldout_panel/heldout_scores${tag}.json')
print(len(json.loads(f.read_text())) if f.exists() else 0)" 2>/dev/null)
  if [ "${n:-0}" -ge 132 ]; then
    echo "=== ${tag} already complete (${n} folds), skipping ==="
    continue
  fi
  free=$(df -g . | tail -1 | awk '{print $4}')
  if [ "$free" -lt 5 ]; then
    echo "=== ABORT: only ${free} GiB free, refusing to start ${tag} ==="
    exit 1
  fi
  echo "=== ${tag} starting $(date +%H:%M:%S), ${free} GiB free ==="
  "$PY" -u src/heldout_panel.py --base boltz1 --sampling-steps 200 \
      --recycling-steps 3 --msa-depth 0 --run-tag "${tag}" --batch-size 6 \
      >> "artifacts/heldout_full${tag}.log" 2>&1
  n=$("$PY" -c "
import json,pathlib
f=pathlib.Path('artifacts/heldout_panel/heldout_scores${tag}.json')
print(len(json.loads(f.read_text())) if f.exists() else 0)" 2>/dev/null)
  echo "=== ${tag} finished $(date +%H:%M:%S) with ${n}/132 folds ==="
done

echo
echo "=== all draws done; analysing ==="
"$PY" src/heldout_at_full.py || true
echo
echo "=== across the full-settings draws ==="
"$PY" src/heldout_replicates.py --regime "boltz1@200/3/full" \
    --out artifacts/heldout_replicates_full.json || true
echo "=== QUEUE COMPLETE $(date +%H:%M:%S) ==="
