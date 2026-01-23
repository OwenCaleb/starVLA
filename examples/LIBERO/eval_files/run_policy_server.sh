#!/bin/bash
cd /home/liwenbo/projects/VLA/starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo
export star_vla_python=/home/liwenbo/anaconda3/envs/starVLA/bin/python
your_ckpt=/home/liwenbo/projects/VLA/starVLA/playground/Pretrained_models/StarVLA_Qwen2.5_VL/checkpoints/steps_30000_pytorch_model.pt
gpu_id=6
port=5694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
