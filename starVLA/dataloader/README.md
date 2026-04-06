# starVLA dataloader 数据流说明

本文档描述一次训练 step 中，数据从配置到模型前向的执行顺序。
重点回答两个问题：
- VLA 数据在一次 step 中经过了哪些文件
- VLM 数据在一次 step 中经过了哪些文件

说明：
- 这里按"时间顺序"描述一次 step 的主链路。
- DataLoader 可能有多 worker 并行取样，本文按逻辑先后描述，不展开并发细节。

---

## 0. 共同入口（一次 step 的起点）

1. 训练脚本启动并构建两路 dataloader
   - 文件：`starVLA/training/train_starvla_cotrain.py`
   - 调用：`prepare_data(...)` -> `build_dataloader(...)`

2. dataloader 分发器选择具体管线
   - 文件：`starVLA/dataloader/__init__.py`
   - 逻辑：
     - `dataset_py=lerobot_datasets` 走 VLA
     - `dataset_py=vlm_datasets` 走 VLM

3. 训练循环取下一批数据
   - 文件：`starVLA/training/train_starvla_cotrain.py`
   - 调用：`_get_next_batch()` 同时拿 `batch_vla` 与 `batch_vlm`

---

## 1. VLA 一次数据流（按时间顺序）

### Step A: 配置展开为数据集混合规范

1. 读取 `vla_data.data_mix`（例如 `libero_all`）
   - 文件：`starVLA/dataloader/lerobot_datasets.py`
   - 调用：`get_vla_dataset(data_cfg)`

2. 根据 mix 名称查表得到数据集列表
   - 文件：`starVLA/dataloader/gr00t_lerobot/mixtures.py`
   - 产物：`[(data_name, weight, robot_type), ...]`

### Step B: 单数据集构造（每个 data_name 一次）

3. 按 `robot_type` 选择模态配置与变换
   - 文件：`starVLA/dataloader/gr00t_lerobot/data_config.py`
   - 产物：
     - `modality_config()`：字段键名 + delta_indices
     - `transform()`：归一化/格式变换策略

4. 初始化单数据集读取器
   - 文件：`starVLA/dataloader/lerobot_datasets.py`
   - 调用：`make_LeRobotSingleDataset(...)`

5. 读取并校验元数据结构
   - 文件：`starVLA/dataloader/gr00t_lerobot/datasets.py`
   - 辅助文件：
     - `schema.py`：元数据 schema 校验
     - `embodiment_tags.py`：robot_type 到 tag 映射

### Step C: 组装混合数据集并执行采样

6. 将多个单数据集合并为混采数据集
   - 文件：`starVLA/dataloader/gr00t_lerobot/datasets.py`
   - 类：`LeRobotMixtureDataset`

7. DataLoader 在一次 `__getitem__` 中完成混采
   - 文件：`starVLA/dataloader/gr00t_lerobot/datasets.py`
   - 关键动作：
     - 按权重采样某个数据集
     - 在该数据集采样轨迹步
     - 读取原始 step 数据

### Step D: 模态读取、变换、打包

8. 读取视觉帧与低维信号
   - 文件：`starVLA/dataloader/gr00t_lerobot/datasets.py`
   - 视频解码工具：`starVLA/dataloader/gr00t_lerobot/video.py`

9. 执行 transform 链
   - 文件：
     - `starVLA/dataloader/gr00t_lerobot/transform/base.py`
     - `starVLA/dataloader/gr00t_lerobot/transform/state_action.py`
     - `starVLA/dataloader/gr00t_lerobot/transform/video.py`
     - `starVLA/dataloader/gr00t_lerobot/transform/concat.py`

10. 打包为训练样本 dict
    - 文件：`starVLA/dataloader/gr00t_lerobot/datasets.py`
    - 样本典型字段：`image`, `lang`, `action`, 可选 `state`

11. collate 输出
    - 文件：`starVLA/dataloader/lerobot_datasets.py`
    - `collate_fn` 直接返回 list（不改样本结构）

12. 进入模型 VLA 分支
    - 下游：`starVLA/model/framework/*` 中 `forward(examples=batch_vla)`

---

## 2. VLM 一次数据流（按时间顺序）

### Step A: 注册数据源并创建数据模块

1. 读取 `vlm_data.dataset_use`
   - 文件：`starVLA/dataloader/vlm_datasets.py`
   - 类：`LazySupervisedDataset`

2. dataset 名称映射到标注路径/数据路径
   - 文件：`starVLA/dataloader/qwenvl_llavajson/qwen_data_config.py`
   - 产物：`[{annotation_path, data_path, sampling_rate}, ...]`

3. 构建 tokenizer/image_processor 与 data module
   - 文件：`starVLA/dataloader/vlm_datasets.py`
   - 调用：`make_vlm_dataloader(...)`

### Step B: 单样本读取与多模态编码

4. DataLoader 调用 `LazySupervisedDataset.__getitem__`
   - 文件：`starVLA/dataloader/vlm_datasets.py`

5. 根据样本类型读取 image 或 video
   - 文件：`starVLA/dataloader/vlm_datasets.py`
   - 关键函数：`process_image_unified`, `process_video`

6. 构建 Qwen 对话输入与监督 labels
   - 文件：`starVLA/dataloader/vlm_datasets.py`
   - 函数：`preprocess_qwen_2_visual(...)`

7. 计算多模态 RoPE 位置索引
   - 文件：`starVLA/dataloader/qwenvl_llavajson/rope2d.py`
   - 函数：`get_rope_index_25` / `get_rope_index_2`

### Step C: collate 与 batch 产出

8. 执行 collator
   - 文件：`starVLA/dataloader/vlm_datasets.py`
   - 类：
     - `DataCollatorForSupervisedDataset`（常规 pad）
     - `FlattenedDataCollatorForSupervisedDataset`（packed）

9. 输出 VLM batch
   - 典型字段：`input_ids`, `labels`, `attention_mask`, `position_ids`, `pixel_values`/`pixel_values_videos`

10. 进入模型 VLM 分支
    - 下游：`model.qwen_vl_interface(**batch_vlm)`

---

## 3. 一次 step 的总时间线（VLA + VLM）

1. `train_starvla_cotrain.py:prepare_data` 构建两路 DataLoader
2. `dataloader/__init__.py:build_dataloader` 分发 VLA/VLM
3. 训练循环 `_get_next_batch` 同步获取 `batch_vla` 与 `batch_vlm`
4. VLA 路径：mixture 采样 -> step 读取 -> transform -> 样本 list
5. VLM 路径：样本懒加载 -> 视觉/文本编码 -> rope -> tensor collate
6. 训练步 `_train_step`：
   - VLA: `model.forward(batch_vla)`
   - VLM: `model.qwen_vl_interface(**batch_vlm)`

---

## 4. 快速定位问题建议

- VLA 数据找不到/字段错误：优先检查
  - `gr00t_lerobot/mixtures.py`
  - `gr00t_lerobot/data_config.py`
  - 子数据集 `meta/modality.json`

- VLM 数据集名报错：优先检查
  - `qwenvl_llavajson/qwen_data_config.py`

- batch 形状或类型异常：优先检查
  - `vlm_datasets.py` 中 collator
  - `gr00t_lerobot/datasets.py` 中 `_pack_sample`
