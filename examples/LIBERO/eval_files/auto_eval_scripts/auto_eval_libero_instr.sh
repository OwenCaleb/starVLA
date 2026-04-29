#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
cd "$REPO_ROOT"

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/mnt/nas_ssd/workspace/wenboli/projects/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}
export LIBERO_python=/opt/conda/envs/starVLA/bin/python
export starVLA_python=/opt/conda/envs/starVLA/bin/python

export PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="${PYTHONPATH:+${PYTHONPATH}:}${LIBERO_HOME}"
export PYTHONPATH="$(pwd):${PYTHONPATH}"
###########################################################################################

your_ckpt=/mnt/nas_ssd/workspace/wenboli/projects/starVLA/playground/Pretrained_models/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
run_index_base=346
SCRIPT_PATH="./examples/LIBERO/eval_files/auto_eval_scripts/eval_libero_instr_parall.sh"
task_pids=()

cleanup() {
    for pid in "${task_pids[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    for pid in "${task_pids[@]:-}"; do
        wait "$pid" 2>/dev/null || true
    done
}

trap cleanup EXIT INT TERM

task_suites=(libero_10 libero_goal libero_object libero_spatial)

for offset in "${!task_suites[@]}"; do
    task_suite_name="${task_suites[$offset]}"
    run_index=$((run_index_base + offset))
    echo "Launching ${task_suite_name} on shared GPU 0 (run_index=${run_index})"
    bash "$SCRIPT_PATH" "$your_ckpt" "$task_suite_name" "$run_index" &
    task_pids+=("$!")
done

for pid in "${task_pids[@]}"; do
    wait "$pid"
done

trap - EXIT INT TERM