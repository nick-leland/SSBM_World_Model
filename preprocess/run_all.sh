#!/usr/bin/env bash
# Full preprocessing pipeline: download → extract states → render frames → build HDF5
#
# Required env vars for the render step:
#   DOLPHIN_BIN  path to Slippi Playback Dolphin binary
#   MELEE_ISO    path to SSBM NTSC 1.02 ISO
#
# Optional:
#   N_REPLAYS    number of replays to download (default: 1000)
#   WORKERS      number of parallel workers for state extraction (default: CPU count)
#
# Usage:
#   DOLPHIN_BIN=/path/to/dolphin MELEE_ISO=/path/to/melee.iso bash preprocess/run_all.sh

set -euo pipefail

N_REPLAYS="${N_REPLAYS:-1000}"
WORKERS="${WORKERS:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "=== Step 1: Download ${N_REPLAYS} replays ==="
python preprocess/download.py --n-replays "$N_REPLAYS"

echo ""
echo "=== Step 2: Extract game state to parquet ==="
if [ -n "$WORKERS" ]; then
    python preprocess/extract_states.py --workers "$WORKERS"
else
    python preprocess/extract_states.py
fi

echo ""
echo "=== Step 3: Render frames via Dolphin ==="
if [ -z "${DOLPHIN_BIN:-}" ] || [ -z "${MELEE_ISO:-}" ]; then
    echo "WARNING: DOLPHIN_BIN or MELEE_ISO not set — skipping frame rendering."
    echo "         Run render_frames.py manually once Dolphin and ISO are available:"
    echo "         python preprocess/render_frames.py --dolphin-bin \$DOLPHIN_BIN --iso \$MELEE_ISO"
else
    python preprocess/render_frames.py \
        --dolphin-bin "$DOLPHIN_BIN" \
        --iso "$MELEE_ISO"
fi

echo ""
echo "=== Step 4: Build HDF5 datasets ==="
python preprocess/build_hdf5.py

echo ""
echo "=== Done! HDF5 files are in data/hdf5/ ==="
