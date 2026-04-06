#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   export DEST=/mnt/nas_ssd/workspace/wenboli/projects/starVLA/playground/Datasets && bash examples/LIBERO/data_preparation.sh
# or
#   bash examples/LIBERO/data_preparation.sh /path/to/dir

DEST="${DEST:-${1:-}}"
if [[ -z "${DEST}" ]]; then
  echo "ERROR: DEST is not set."
  echo "  export DEST=/path/to/dir && bash examples/LIBERO/data_preparation.sh"
  echo "  or: bash examples/LIBERO/data_preparation.sh /path/to/dir"
  exit 1
fi

CUR="$(pwd)"
mkdir -p "$DEST"

if ! command -v hf >/dev/null 2>&1; then
  echo "ERROR: 'hf' command not found. Please install huggingface_hub CLI first."
  echo "  python -m pip install -U 'huggingface_hub[cli]'"
  exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
  echo "ERROR: 'unzip' command not found. Install unzip first."
  exit 1
fi

python -m pip install -U "huggingface-hub==0.35.3" -i https://pypi.org/simple

# Mitigate HF rate-limit issues:
# - Disable Xet path to avoid xet-read-token bursts
# - Lower concurrent workers
# - Add retry with exponential backoff
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
HF_MAX_WORKERS="${HF_MAX_WORKERS:-2}"
HF_MAX_RETRIES="${HF_MAX_RETRIES:-12}"
HF_SLEEP_SECS="${HF_SLEEP_SECS:-20}"

if ! [[ "$HF_MAX_WORKERS" =~ ^[0-9]+$ ]] || [[ "$HF_MAX_WORKERS" -lt 1 ]]; then
  echo "ERROR: HF_MAX_WORKERS must be a positive integer. Current: $HF_MAX_WORKERS"
  exit 1
fi
if ! [[ "$HF_MAX_RETRIES" =~ ^[0-9]+$ ]] || [[ "$HF_MAX_RETRIES" -lt 1 ]]; then
  echo "ERROR: HF_MAX_RETRIES must be a positive integer. Current: $HF_MAX_RETRIES"
  exit 1
fi
if ! [[ "$HF_SLEEP_SECS" =~ ^[0-9]+$ ]] || [[ "$HF_SLEEP_SECS" -lt 1 ]]; then
  echo "ERROR: HF_SLEEP_SECS must be a positive integer. Current: $HF_SLEEP_SECS"
  exit 1
fi

hf_download_retry() {
  local repo_type="$1"
  local repo_id="$2"
  local local_dir="$3"

  local rc=0
  local sleep_secs="$HF_SLEEP_SECS"
  mkdir -p "$local_dir"

  for i in $(seq 1 "$HF_MAX_RETRIES"); do
    local log="/tmp/libero_hf_${repo_id##*/}_attempt_${i}.log"
    echo "[HF] Attempt $i/$HF_MAX_RETRIES => $repo_id (log: $log)"

    set +e
    hf download "$repo_id" --repo-type "$repo_type" --local-dir "$local_dir" --max-workers "$HF_MAX_WORKERS" 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    set -e

    if [[ $rc -eq 0 ]]; then
      echo "[HF] Done: $repo_id"
      return 0
    fi

    if grep -qi "429\|Too Many Requests\|rate limit" "$log"; then
      echo "[HF] Rate limited for $repo_id, backing off ${sleep_secs}s..."
    else
      echo "[HF] Failed (rc=$rc) for $repo_id, retry in ${sleep_secs}s..."
    fi

    sleep "$sleep_secs"
    if [[ "$sleep_secs" -lt 600 ]]; then
      sleep_secs=$((sleep_secs * 2))
    fi
  done

  return 1
}

for repo in \
  IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot
do
  hf_download_retry dataset "$repo" "$DEST/libero/${repo##*/}"
done

hf_download_retry dataset "StarVLA/LLaVA-OneVision-COCO" "$DEST/LLaVA-OneVision-COCO"
if [[ -f "$DEST/LLaVA-OneVision-COCO/sharegpt4v_coco.zip" ]]; then
  unzip -o -- "$DEST/LLaVA-OneVision-COCO/sharegpt4v_coco.zip" -d "$DEST/LLaVA-OneVision-COCO/"
fi

mkdir -p "$CUR/playground/Datasets"
ln -sfn "$DEST/libero" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA"
ln -sfn "$DEST/LLaVA-OneVision-COCO" "$CUR/playground/Datasets/LLaVA-OneVision-COCO"

## move modality
mkdir -p "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot/meta"
mkdir -p "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot/meta"
mkdir -p "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot/meta"
mkdir -p "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot/meta"
cp "$CUR/examples/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot/meta"
cp "$CUR/examples/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot/meta"
cp "$CUR/examples/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot/meta"
cp "$CUR/examples/LIBERO/train_files/modality.json" "$CUR/playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot/meta"
