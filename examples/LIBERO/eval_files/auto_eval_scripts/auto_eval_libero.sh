#!/bin/bash

cd /home/liwenbo/projects/VLA/starVLA
SCRIPT_PATH="./examples/LIBERO/eval_files/auto_eval_scripts/eval_libero_parall.sh"
your_ckpt=/home/liwenbo/projects/VLA/starVLA/playground/Pretrained_models/StarVLA_Qwen2.5_VL/checkpoints/steps_30000_pytorch_model.pt
run_index_base=346

#####################################################
task_suite_name=libero_10 # align with your model
run_index=$((run_index_base + 0))
bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index &
#####################################################

sleep 15
#####################################################
task_suite_name=libero_goal # align with your model
run_index=$((run_index_base + 1))
bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index &
#####################################################
sleep 15
#####################################################
task_suite_name=libero_object # align with your model
run_index=$((run_index_base + 2))
bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index &
#####################################################
sleep 15
####################################################
task_suite_name=libero_spatial # align with your model
run_index=$((run_index_base + 3))
bash $SCRIPT_PATH $your_ckpt $task_suite_name $run_index &
#####################################################

