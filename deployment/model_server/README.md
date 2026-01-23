
# start policy server


```bash

your_ckpt=./results/Checkpoints/1003_qwenfast/checkpoints/steps_50000_pytorch_model.pt

python deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port 10093 \
    --use_bf16
```


# connect to policy server for debug

```bash
python deployment/model_server/debug_server_policy.py

# plus server_policy.py into your vla controler by ref to debug_server_policy.py
```

客户端发的就是这种：

msg = {
  "type": "infer",                 # 协议字段：告诉 server 这是推理请求
  "request_id": "xxx",             # 协议字段：链路追踪
  "batch_images": [[image_np]],    # 模型字段：给 policy 的输入
  "instructions": ["..."],         # 模型字段：给 policy 的输入
}

server 收到后干了一句：

policy.predict_action(**msg)

这会导致：type 也会被传给模型。模型如果不接受 type 这个参数就会炸。