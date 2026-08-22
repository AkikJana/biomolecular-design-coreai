#!/usr/bin/env bash
# Move code up and results back. Run on YOUR machine, not the rented one.
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
SSH="ssh -p $PORT -o StrictHostKeyChecking=accept-new"
LOCAL="$(cd "$(dirname "$0")/.." && pwd)"

case "${1:-}" in
  up)
    echo "pushing code to $GPU_HOST:$REMOTE_DIR"
    rsync -az --info=stats1 -e "$SSH" \
      --exclude '.git' --exclude '.venv*' --exclude 'artifacts' \
      --exclude '*.pdf' --exclude '*.docx' --exclude '*.pptx' \
      --exclude 'boltz/' --exclude '.pb_work' --exclude 'hf_dataset' \
      "$LOCAL/" "$GPU_HOST:$REMOTE_DIR/"
    ;;
  down)
    mkdir -p "$LOCAL/artifacts"
    if [ "${2:-}" = "--all" ]; then
      echo "pulling everything, structures included -- this is large"
      rsync -az --info=progress2 -e "$SSH" \
        "$GPU_HOST:$REMOTE_DIR/artifacts/" "$LOCAL/artifacts/"
    else
      # Scores only. Structures are what filled the local disk last time and
      # cost this project its converged models; leave them on the remote unless
      # something specifically needs them.
      echo "pulling scores only (JSON/CSV/logs)"
      rsync -az --info=stats1 -e "$SSH" \
        --include '*/' --include '*.json' --include '*.csv' --include '*.log' \
        --exclude '*' \
        "$GPU_HOST:$REMOTE_DIR/artifacts/" "$LOCAL/artifacts/"
    fi
    ;;
  *) sed -n '2,10p' "$0"; exit 1 ;;
esac
