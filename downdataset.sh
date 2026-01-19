#!/usr/bin/env bash
set -euo pipefail

# Demo : nohup bash downdataset.sh > downdataset.log 2>&1 &

# =========================
# CONFIG (edit only these)
# =========================

# ---- Choose ONE dataset each time ----
# Example A) Bridge (LeRobot OXE)
# REPO_ID="IPEC-COMMUNITY/bridge_orig_lerobot"
# DATASET_NAME="bridge_orig_lerobot"

# Example B) Fractal (LeRobot OXE)
REPO_ID="IPEC-COMMUNITY/fractal20220817_data_lerobot"
DATASET_NAME="fractal20220817_data_lerobot"

# Example C) Any other dataset
# REPO_ID="YOUR_ORG/YOUR_DATASET_REPO"
# DATASET_NAME="your_local_dataset_dirname"

# Real storage location (large disk)
REAL_ROOT_DIR="/mnt/data/liwenbo_datas/OXE_LEROBOT_DATASET"
REAL_DIR="$REAL_ROOT_DIR/$DATASET_NAME"

# Where starVLA expects the dataset root directory (symlink path)
LINK_ROOT_DIR="$HOME/projects/VLA/starVLA/playground/Datasets/OXE_LEROBOT_DATASET"

# (Recommended) Put HF cache on large disk so resume works well and avoids filling home disk.
export HF_HOME="/mnt/data/liwenbo_datas/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"

# Retry/backoff for HF download
MAX_RETRIES=12
SLEEP_SECS=30

# Optional: disable symlinks inside local-dir if CLI supports it.
# We will auto-detect support; if not supported, we simply omit it.
LOCAL_DIR_USE_SYMLINKS="False"

# =========================
# END CONFIG
# =========================


echo "[0/4] Preflight:"
if [[ -z "${REPO_ID}" || -z "${DATASET_NAME}" ]]; then
  echo "ERROR: REPO_ID and DATASET_NAME must be set." >&2
  exit 2
fi

if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "ERROR: Neither 'hf' nor 'huggingface-cli' is available in PATH." >&2
  echo "Tip: pip install -U huggingface_hub" >&2
  exit 127
fi

echo "[1/4] Ensure real storage directory exists:"
mkdir -p "$REAL_DIR"
echo "  REPO_ID=$REPO_ID"
echo "  DATASET_NAME=$DATASET_NAME"
echo "  REAL_DIR=$REAL_DIR"
echo "  LINK_ROOT_DIR=$LINK_ROOT_DIR"
echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"


# Detect whether current CLI supports --local-dir-use-symlinks
supports_local_dir_use_symlinks() {
  local help_out=""
  if command -v hf >/dev/null 2>&1; then
    help_out="$(hf download --help 2>/dev/null || true)"
  else
    help_out="$(huggingface-cli download --help 2>/dev/null || true)"
  fi
  echo "$help_out" | grep -q -- "--local-dir-use-symlinks"
}

download_once() {
  if command -v hf >/dev/null 2>&1; then
    if supports_local_dir_use_symlinks; then
      hf download --repo-type dataset "$REPO_ID" \
        --local-dir "$REAL_DIR" \
        --local-dir-use-symlinks "$LOCAL_DIR_USE_SYMLINKS"
    else
      hf download --repo-type dataset "$REPO_ID" \
        --local-dir "$REAL_DIR"
    fi
  else
    if supports_local_dir_use_symlinks; then
      huggingface-cli download --repo-type dataset "$REPO_ID" \
        --local-dir "$REAL_DIR" \
        --local-dir-use-symlinks "$LOCAL_DIR_USE_SYMLINKS"
    else
      huggingface-cli download --repo-type dataset "$REPO_ID" \
        --local-dir "$REAL_DIR"
    fi
  fi
}

echo "[2/4] Download dataset snapshot into REAL_DIR (retry on failures / 429):"
rc=0
sleep_secs="$SLEEP_SECS"

for i in $(seq 1 "$MAX_RETRIES"); do
  LOG="/tmp/hf_dl_${DATASET_NAME}.attempt_${i}.log"
  echo "  Attempt $i/$MAX_RETRIES ... (log: $LOG)"

  set +e
  download_once 2>&1 | tee "$LOG"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ $rc -eq 0 ]]; then
    echo "  Download finished."
    break
  fi

  if grep -q "429 Client Error" "$LOG"; then
    echo "  Hit 429 rate limit." >&2
    echo "  Tip: run 'huggingface-cli login' (recommended) or export HF_TOKEN in your shell (do NOT commit tokens to git)." >&2
  else
    echo "  Download failed (rc=$rc)." >&2
  fi

  echo "  Sleep ${sleep_secs}s then retry..." >&2
  sleep "$sleep_secs"
  if [[ "$sleep_secs" -lt 600 ]]; then
    sleep_secs=$((sleep_secs * 2))
  fi
done

if [[ $rc -ne 0 ]]; then
  echo "ERROR: Download failed after $MAX_RETRIES attempts." >&2
  exit "$rc"
fi


echo "[3/4] Create/update symlink at LINK_ROOT_DIR -> REAL_ROOT_DIR:"
mkdir -p "$(dirname "$LINK_ROOT_DIR")"

if [[ -e "$LINK_ROOT_DIR" && ! -L "$LINK_ROOT_DIR" ]]; then
  echo "ERROR: $LINK_ROOT_DIR exists and is not a symlink." >&2
  echo "Please move/remove it manually, then re-run this script." >&2
  exit 3
fi

ln -sfn "$REAL_ROOT_DIR" "$LINK_ROOT_DIR"


echo "[4/4] Verify:"
echo "  Symlink:"
ls -l "$LINK_ROOT_DIR" || true

echo "  Dataset dir exists?"
if [[ -d "$LINK_ROOT_DIR/$DATASET_NAME" ]]; then
  echo "  OK: $LINK_ROOT_DIR/$DATASET_NAME"
else
  echo "  MISSING: $LINK_ROOT_DIR/$DATASET_NAME" >&2
  echo "  Listing LINK_ROOT_DIR:" >&2
  ls -lah "$LINK_ROOT_DIR" || true
  exit 4
fi

echo "  Sample files (maxdepth=2):"
find "$LINK_ROOT_DIR/$DATASET_NAME" -maxdepth 2 -type f | head -n 20

echo "Done."
