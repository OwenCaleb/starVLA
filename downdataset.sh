#!/usr/bin/env bash
set -euo pipefail

# =========================
# CONFIG (edit only these)
# =========================

# ---- Choose ONE dataset each time ----
# Example A) Bridge (LeRobot OXE)
REPO_ID="IPEC-COMMUNITY/bridge_orig_lerobot"
DATASET_NAME="bridge_orig_lerobot"

# # Example B) Fractal (LeRobot OXE)
# REPO_ID="IPEC-COMMUNITY/fractal20220817_data_lerobot"
# DATASET_NAME="fractal20220817_data_lerobot"

# Example C) Any other dataset
# REPO_ID="YOUR_ORG/YOUR_DATASET_REPO"
# DATASET_NAME="your_local_dataset_dirname"

# Real storage location (large disk)
REAL_ROOT_DIR="/mnt/data/liwenbo_datas/OXE_LEROBOT_DATASET"
REAL_DIR="$REAL_ROOT_DIR/$DATASET_NAME"

# Where starVLA expects the dataset root directory (symlink target)
LINK_ROOT_DIR="$HOME/projects/VLA/starVLA/playground/Datasets/OXE_LEROBOT_DATASET"

# (Recommended) Put HF cache on large disk so resume works well and avoids filling home disk.
export HF_HOME="/mnt/data/liwenbo_datas/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"

# Avoid symlinks inside downloaded folder (more robust across filesystems)
LOCAL_DIR_USE_SYMLINKS="False"
# =========================


echo "[1/4] Ensure real storage directory exists:"
mkdir -p "$REAL_DIR"
echo "  REPO_ID=$REPO_ID"
echo "  DATASET_NAME=$DATASET_NAME"
echo "  REAL_DIR=$REAL_DIR"
echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"

echo "[2/4] Download dataset snapshot into REAL_DIR (retry on 429):"

MAX_RETRIES=12
SLEEP_SECS=30

download_once() {
  if command -v hf >/dev/null 2>&1; then
    hf download --repo-type dataset "$REPO_ID" --local-dir "$REAL_DIR"
  elif command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download --repo-type dataset "$REPO_ID" --local-dir "$REAL_DIR"
  else
    echo "ERROR: Neither 'hf' nor 'huggingface-cli' is available in PATH." >&2
    return 127
  fi
}

rc=0
for i in $(seq 1 "$MAX_RETRIES"); do
  LOG="/tmp/hf_dl_${DATASET_NAME}.attempt_${i}.log"
  echo "  Attempt $i/$MAX_RETRIES ... (log: $LOG)"

  set +e
  # stream output to terminal AND log file
  download_once 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ $rc -eq 0 ]]; then
    echo "  Download finished."
    break
  fi

  if grep -q "429 Client Error" "$LOG"; then
    echo "  Hit 429 rate limit. Tip: 'huggingface-cli login' or set HF_TOKEN. Backing off..." >&2
  else
    echo "  Download failed (rc=$rc). Backing off..." >&2
  fi

  echo "  Sleep ${SLEEP_SECS}s then retry..." >&2
  sleep "$SLEEP_SECS"
  if [[ "$SLEEP_SECS" -lt 600 ]]; then
    SLEEP_SECS=$((SLEEP_SECS * 2))
  fi
done

if [[ $rc -ne 0 ]]; then
  echo "ERROR: Download failed after $MAX_RETRIES attempts." >&2
  exit "$rc"
fi


echo "[3/4] Create/update symlink at LINK_ROOT_DIR -> REAL_ROOT_DIR:"
mkdir -p "$(dirname "$LINK_ROOT_DIR")"
ln -sfn "$REAL_ROOT_DIR" "$LINK_ROOT_DIR"

echo "[4/4] Verify:"
ls -l "$LINK_ROOT_DIR"
echo "  Dataset dir:"
ls -lah "$LINK_ROOT_DIR/$DATASET_NAME" | head
echo "Done."
