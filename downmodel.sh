#!/usr/bin/env bash
set -euo pipefail

# Demo : nohup bash downmodel.sh > downmodel.log 2>&1 &

# =========================
# CONFIG
# =========================
REPO_ID="StarVLA/Qwen3-VL-OFT-Robotwin2"

# Real storage location (large disk)
REAL_DIR="/mnt/data/liwenbo_datas/models/starVLA/Qwen3-VL-OFT-Robotwin2"

# Where starVLA expects the checkpoint directory (symlink target)
LINK_DIR="$HOME/projects/VLA/starVLA/playground/Pretrained_models/Qwen3-VL-OFT-Robotwin2"

# (Recommended) Put HF cache on large disk so resume works well and avoids filling home disk. 放缓存以恢复
# Comment these two lines if you don't want to change cache location.
export HF_HOME="/mnt/data/liwenbo_datas/hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
# =========================


echo "[1/4] Ensure real storage directory exists:"
mkdir -p "$REAL_DIR"
echo "  REAL_DIR=$REAL_DIR"
echo "  HF_HOME=$HF_HOME"
echo "  HF_HUB_CACHE=$HF_HUB_CACHE"

echo "[2/4] Download model snapshot into REAL_DIR (supports resume):"
if command -v huggingface-cli >/dev/null 2>&1; then
  # huggingface-cli is most consistent across versions for `--local-dir`
  huggingface-cli download "$REPO_ID" \
    --local-dir "$REAL_DIR" \
    --resume-download || \
  huggingface-cli download "$REPO_ID" \
    --local-dir "$REAL_DIR"
elif command -v hf >/dev/null 2>&1; then
  # `hf download` flag set varies by version; keep it minimal and robust.
  hf download "$REPO_ID" \
    --local-dir "$REAL_DIR"
else
  echo "ERROR: Neither 'huggingface-cli' nor 'hf' is available in PATH." >&2
  exit 1
fi

echo "[3/4] Create/update symlink at LINK_DIR -> REAL_DIR:"
mkdir -p "$(dirname "$LINK_DIR")"
ln -sfn "$REAL_DIR" "$LINK_DIR"

echo "[4/4] Verify symlink:"
ls -l "$LINK_DIR"
echo "Done."
