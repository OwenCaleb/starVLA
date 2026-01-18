
cd /home/liwenbo/projects/VLA/starVLA
export PYTHONPATH=$(pwd):${PYTHONPATH}

port=6678
gpu_id=4
# export DEBUG=true
export star_vla_python=/home/liwenbo/anaconda3/envs/starVLA/bin/python

your_ckpt=/home/liwenbo/projects/VLA/starVLA/playground/Pretrained_models/Qwen3VL-GR00T-Bridge-RT-1/checkpoints/steps_20000_pytorch_model.pt

#### build output directory #####
ckpt_dir=$(dirname "${your_ckpt}")
ckpt_base=$(basename "${your_ckpt}")
ckpt_name="${ckpt_base%.*}"
output_server_dir="${ckpt_dir}/output_server"
mkdir -p "${output_server_dir}"
log_file="${output_server_dir}/${ckpt_name}_policy_server_${port}.log"
# .../checkpoints/output_server/steps_20000_pytorch_model_policy_server_6678.log


#### run server #####
CUDA_VISIBLE_DEVICES=${gpu_id} ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16 \
    2>&1 | tee "${log_file}"