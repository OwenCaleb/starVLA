#!/bin/bash

cd /home/liwenbo/projects/VLA/starVLA
# conda activate starVLA

###########################################################################################
# === Please modify the following paths according to your environment ===
export LIBERO_HOME=/home/liwenbo/projects/environment/LIBERO
export LIBERO_CONFIG_PATH=${LIBERO_HOME}/

# export LIBERO_Python=/mnt/petrelfs/share/yejinhui/Envs/miniconda3/envs/lerobot/bin/python
export LIBERO_Python=/home/liwenbo/anaconda3/envs/starVLA/bin/python

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo


host="127.0.0.1"
base_port=5694
unnorm_key="franka"
your_ckpt=/home/liwenbo/projects/VLA/starVLA/playground/Pretrained_models/StarVLA_Qwen2.5_VL/checkpoints/steps_30000_pytorch_model.pt
export DEBUG=true
# export DEBUG=false

folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
# === End of environment variable configuration ===
###########################################################################################

LOG_DIR="logs/$(date +"%Y%m%d_%H%M%S")"
mkdir -p ${LOG_DIR}


task_suite_name=libero_spatial
num_trials_per_task=50
video_out_path="results/${task_suite_name}/${folder_name}"


${LIBERO_Python} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"
