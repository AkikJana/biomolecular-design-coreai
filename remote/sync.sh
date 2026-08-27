#!/usr/bin/env bash
# Move code up and results back. Run on YOUR machine, not the rented one.
#
# Uses -rlptz rather than -a on purpose: -a preserves ownership, and network
# volumes (RunPod's /workspace is MooseFS) refuse chown, which fails the whole
# transfer over something cosmetic.
#
#   remote/sync.sh up          push code (no artifacts, no venv, no PDFs)
#   remote/sync.sh down        pull scores back -- JSON only, no structures
#   remote/sync.sh down --all  pull structures too (hundreds of MB)
#
# Set the endpoint your provider gave you:
#   export GPU_HOST=root@1.2.3.4  GPU_PORT=22222
set -euo pipefail

: "${GPU_HOST:?set GPU_HOST, e.g. export GPU_HOST=root@1.2.3.4}"
PORT="${GPU_PORT:-22}"
REMOTE_DIR="${REMOTE_DIR:-BiomolecularDesign}"
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

case "${1:-}" in
  up)
    # boltz/ used to be excluded here to save 6 MB. provision.sh then had
    # nothing to install and fell back to `pip install boltz`, which is a
    # different codebase -- 46 of 106 shared files differ. That saved transfer
    # cost the attribution of a published number (preprint 2.4). It ships now.
    echo "pushing code to $GPU_HOST:$REMOTE_DIR (including the boltz fork, ~6 MB)"
    rsync -rlptz --stats -e "$SSH" \
      --exclude '.git' --exclude '.venv*' --exclude 'artifacts' \
      --exclude '*.pdf' --exclude '*.docx' --exclude '*.pptx' \
      --exclude '.pb_work' --exclude 'hf_dataset' \
      --exclude 'compiled_surrogate*' --exclude '.worktrees' --exclude '.agents' \
      --exclude 'boltz_results_*' --exclude '.venv-gpu' --exclude '.cache' \
      "$LOCAL/" "$GPU_HOST:$REMOTE_DIR/"

    # artifacts/ is excluded wholesale above because it is ~200 MB of
    # predictions, but the runners read four things out of it: the panel
    # definition, the reduced-settings scores they compare against, the
    # expansion list, and the MSA cache. The cache is 16 MB and saves
    # refetching 132 alignments from the ColabFold server.
    echo "pushing panel inputs (~17 MB)"
    rsync -rlptz --stats -R -e "$SSH" \
      artifacts/pdb_binders_b2_n22/pdb_binder_scores.json \
      artifacts/pdb_binders_b2_n22/msa_cache \
      artifacts/boltz1_scramble_result.json \
      artifacts/panel_expansion.json \
      "$GPU_HOST:$REMOTE_DIR/" || {
        echo "  panel inputs FAILED to transfer -- queue.sh will not run without them"
        exit 1
      }
    ;;
  down)
    mkdir -p "$LOCAL/artifacts"
    if [ "${2:-}" = "--all" ]; then
      echo "pulling everything, structures included -- this is large"
      rsync -rlptz --stats -e "$SSH" \
        "$GPU_HOST:$REMOTE_DIR/artifacts/" "$LOCAL/artifacts/"
    else
      # Scores only. Structures are what filled the local disk last time and
      # cost this project its converged models; leave them on the remote unless
      # something specifically needs them.
      echo "pulling scores only (JSON/CSV/logs)"
      rsync -rlptz --stats -e "$SSH" \
        --include '*/' --include '*.json' --include '*.csv' --include '*.log' \
        --exclude '*' \
        "$GPU_HOST:$REMOTE_DIR/artifacts/" "$LOCAL/artifacts/"
    fi
    ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
