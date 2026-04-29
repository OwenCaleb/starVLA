# exp 目录说明

这个目录用于做当前 `MyVLA Stage I / II / III` 的机制分析，不负责训练本身，也不负责最终画图。当前主要产出三类内容：

1. 三个 stage 的中间特征
2. final-layer attention 聚合后的 Q/K Sankey CSV
3. UMAP 与 stage-level 指标表

---

## 一、当前实验共识

- `Stage III` 按当前 **full-mask** 语义做分析
- `act` 表示 action placeholder / action slot
- token group 复用模型真实布局：
  - `vis`
  - `lang`
  - `sub`
  - `dyn`
  - `spa`
  - `act`
- 当前分析输入可以直接是：
  - 图片目录
  - 多视角子目录
  - 单个视频文件
  - 多视频目录
- 当前视频分析默认 temporal 偏移：
  - `--temporal-delta-indices 0 19`
- 当前 Sankey 分析需要 attention tensor，所以分析脚本会把 Qwen attention backend 临时切到 `eager`
  - 这只影响分析实例
  - 不影响训练默认的 `sdpa`

---

## 二、当前脚本

### 1. `extract_stage_features.py`

作用：

- 加载 Stage I / II / III checkpoint
- 对输入 episode 的每一帧做分析前向
- 导出：
  - final hidden
  - final-layer attention
  - token group
  - frame/sample 标识

当前输出：

- `intermediate/stage1_features.pt`
- `intermediate/stage2_features.pt`
- `intermediate/stage3_features.pt`

### 2. `build_qk_sankey.py`

作用：

- 从 `stage*_features.pt` 中读取 final attention
- 聚合成 group-level 的 Q/K 流向表

当前 Sankey 导出格式：

```csv
source,target,value
K_vis,Q_vis,0.31
K_lang,Q_vis,0.08
K_sub,Q_vis,0.04
```

也就是：

- 左边固定是 `K_*`
- 右边固定是 `Q_*`
- 数值列固定是 `value`

同时会导出：

- `qk_flow_stage*_long.csv`
- `qk_heatmap_stage*.csv`

### 3. `build_token_umap.py`

作用：

- 从 `stage*_features.pt` 中读取 hidden
- 构建共享 PCA + UMAP reducer
- 导出 all-token 与 slot-only 两套 UMAP

**当前 UMAP 不是 token-level 点图。**

现在的逻辑是：

- 对每个 frame / sample
- 按 token group 把同组 token hidden 先取平均
- 每个 group 均值作为一个点

因此：

- all-token 模式下，每帧最多 6 个点：
  - `lang / vis / sub / dyn / spa / act`
- slot-only 模式下，每帧最多 4 个点：
  - `sub / dyn / spa / act`

当前导出格式：

```tsv
sample	x	y	group
0	1.23	-0.44	vis
1	0.88	-1.01	lang
```

说明：

- 文件扩展名是 `.tsv`
- 分隔符是制表符
- 列顺序固定为：
  - `sample`
  - `x`
  - `y`
  - `group`
- `sample` 是每个点唯一编号，不再保留原始 `frame_xxx` 文本

当前默认预处理与距离口径：

- hidden 先做 `L2 normalize`
- PCA 后，如果 `UMAP metric='cosine'`，再做一次 `L2 normalize`
- UMAP 默认 `metric='cosine'`
- silhouette 与 IIR 也默认使用同一归一化/距离口径

显式控制参数：

- `build_token_umap.py`
  - `--l2-normalize`
  - `--umap-metric {cosine, euclidean}`
- `build_stage_metric_tables.py`
  - `--l2-normalize`
  - `--distance-metric {cosine, euclidean}`

说明：

- 第二次 `L2 normalize` 主要服务于 `cosine` UMAP
- 如果不用 `cosine`，一般不做 PCA 后第二次归一化

### 4. `build_stage_metric_tables.py`

作用：

- 从 `stage*_features.pt` 直接构建主图指标表

当前主推荐输出：

- `silhouette_stage_table.tsv`
- `log_iir_stage_table.tsv`

其中：

- `silhouette_stage_table.tsv`
  - 列：`stage, s.all, s.vis, s.lang, s.sub, s.dyn, s.spa, s.act`
- `log_iir_stage_table.tsv`
  - 列：`stage, lgIIR.all, lgIIR.vis, lgIIR.lang, lgIIR.sub, lgIIR.dyn, lgIIR.spa, lgIIR.act`

---

## 三、输入格式

### 1. 图片平铺目录

例如：

```text
/mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/episode/
  frame_000001.png
  frame_000002.png
  frame_000003.png
```

语义：

- 每张图是一个 sample
- 动态 teacher 的 temporal 输入会按 `temporal-delta-indices` 自动构造

### 2. 多视角子目录

例如：

```text
episode/
  step_0001/
    cam0.png
    cam1.png
  step_0002/
    cam0.png
    cam1.png
```

语义：

- 每个子目录是一个 sample
- 子目录中的多张图作为 multi-view 输入

### 3. 单个视频文件

例如：

```text
/mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/video/a.mp4
```

语义：

- 视频会被逐帧解码
- 每一帧都是一个 sample
- 当前这段视频实际会被解析成：
  - `166` 个 sample
  - `frame_000001` 到 `frame_000166`

### 4. 多视频目录

例如：

```text
/mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/video/
  Open the top drawer and put the bowl inside.mp4
  Pick up the butter and place it in the basket.mp4
  Turn on the stove.mp4
```

语义：

- 目录下每个视频都会被逐帧解码
- 每一帧都是一个 sample
- 每个 sample 的：
  - `instruction`
  - `subtask`
  默认都取自视频文件名 stem
- sample id 会自动带视频名前缀，避免多视频时 frame id 冲突
  - 例如：
    - `Open the top drawer and put the bowl inside__frame_000001`

当前这组 `exp/video/` 会被解析成：

- `1136` 个 sample

---

## 四、当前固定路径

### Checkpoint

- Stage I
  - `/mnt/nas_ssd/workspace/wenboli/projects/starVLA/results/Checkpoints/myvla_stage1/checkpoints/steps_15000_pytorch_model.pt`
- Stage II
  - `/mnt/nas_ssd/workspace/wenboli/projects/starVLA/results/Checkpoints/myvla_stage2/checkpoints/steps_65000_pytorch_model.pt`
- Stage III
  - `/mnt/nas_ssd/workspace/wenboli/projects/starVLA/results/Checkpoints/myvla_stage3/checkpoints/steps_50000_pytorch_model.pt`

### 视频输入

- `/mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/video`

### 指令

- 多视频目录模式下可以不传 `--instruction`
- 默认直接使用每个视频文件名 stem 作为：
  - `instruction`
  - `subtask`

### 推荐输出目录

- `exp/output/video_analysis`

---

## 五、推荐执行顺序

### Step 1：从多视频目录逐帧抽取三阶段中间特征

```bash
python exp/extract_stage_features.py \
  --stage1-config starVLA/config/training/myvla_stage1.yaml \
  --stage1-ckpt /mnt/nas_ssd/workspace/wenboli/projects/starVLA/results/Checkpoints/myvla_stage1/checkpoints/steps_15000_pytorch_model.pt \
  --stage2-config starVLA/config/training/myvla_stage2.yaml \
  --stage2-ckpt /mnt/nas_ssd/workspace/wenboli/projects/starVLA/results/Checkpoints/myvla_stage2/checkpoints/steps_65000_pytorch_model.pt \
  --stage3-config starVLA/config/training/myvla_stage3.yaml \
  --stage3-ckpt /mnt/nas_ssd/workspace/wenboli/projects/starVLA/results/Checkpoints/myvla_stage3/checkpoints/steps_50000_pytorch_model.pt \
  --episode-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/video \
  --output-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis \
  --device cuda \
  --analysis-progress 1.0 \
  --temporal-delta-indices 0 19


只跑stage3:
python exp/extract_stage_features.py \
  --stage1-config starVLA/config/training/myvla_stage1.yaml \
  --stage1-ckpt /mnt/nas_ssd/workspace/wenboli/projects/starVLA/results/Checkpoints/myvla_stage1/checkpoints/steps_15000_pytorch_model.pt \
  --stage2-config starVLA/config/training/myvla_stage2.yaml \
  --stage2-ckpt /mnt/nas_ssd/workspace/wenboli/projects/starVLA/results/Checkpoints/myvla_stage2/checkpoints/steps_65000_pytorch_model.pt \
  --stage3-config starVLA/config/training/myvla_stage3.yaml \
  --stage3-ckpt /mnt/nas_ssd/workspace/wenboli/projects/starVLA/results/Checkpoints/myvla_stage3/checkpoints/steps_50000_pytorch_model.pt \
  --episode-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/video \
  --output-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis \
  --device cuda \
  --analysis-progress 1.0 \
  --temporal-delta-indices 0 19 \
  --stages stage3
```

当前脚本会自动：

- 逐帧解码目录下所有视频
- 从每个视频文件名 stem 自动生成：
  - `instruction`
  - `subtask`
- 构造 `video` 时间对
- 临时把 Qwen attention backend 切到 `eager`

期望输出：

- `/mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis/intermediate/stage1_features.pt`
- `/mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis/intermediate/stage2_features.pt`
- `/mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis/intermediate/stage3_features.pt`

### Step 2：构建 Sankey / QK CSV

```bash
python exp/build_qk_sankey.py \
  --features-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis/intermediate \
  --output-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis
```

期望输出：

- `sankey_qk_stage1.csv`
- `sankey_qk_stage2.csv`
- `sankey_qk_stage3.csv`
- `qk_flow_stage1_long.csv`
- `qk_flow_stage2_long.csv`
- `qk_flow_stage3_long.csv`
- `qk_heatmap_stage1.csv`
- `qk_heatmap_stage2.csv`
- `qk_heatmap_stage3.csv`

### Step 3：拟合共享 UMAP artifact

```bash
python exp/build_token_umap.py \
  --features-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis/intermediate \
  --output-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis \
  --fit-only \
  --skip-metrics \
  --umap-neighbors 30 \
  --max-tokens-per-stage 0 \
  --l2-normalize \
  --umap-metric euclidean \
  --umap-min-dist 0.2
```

说明：

- 这里的“token”现在实际上是“每帧每组均值点”
- `--max-tokens-per-stage 0` 表示不再对子采样上限做裁剪
- `--umap-min-dist 0.1` 会比 `0.3` 更紧凑

期望输出：

- `/mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis/umap_artifacts.pkl`

### Step 4：导出三阶段 UMAP TSV

```bash
python exp/build_token_umap.py \
  --features-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis/intermediate \
  --output-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis \
  --transform-only \
  --transform-stages stage1 stage2 stage3 \
  --l2-normalize \
  --umap-metric euclidean \
  --skip-metrics
```

期望输出：

- `umap_stage1.tsv`
- `umap_stage2.tsv`
- `umap_stage3.tsv`
- `umap_stage1_slot_only.tsv`
- `umap_stage2_slot_only.tsv`
- `umap_stage3_slot_only.tsv`

### Step 5：导出主图指标表

```bash
python exp/build_stage_metric_tables.py \
  --features-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis/intermediate \
  --output-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis \
  --l2-normalize \
  --distance-metric cosine
```

期望输出：

- `silhouette_stage_table.tsv`
- `log_iir_stage_table.tsv`

---

## 六、几个容易误解的点

### 1. 训练为什么能用 `sdpa`

因为训练默认不请求 `output_attentions=True`。  
你现在的 Sankey 分析需要 final attention，所以分析实例必须切到 `eager`。

### 2. 当前 UMAP 为什么点数比以前少

因为当前 UMAP 已经不是“每个 token 一个点”，而是：

- 每帧
- 每个 token group
- 先均值
- 再作为一个点

这是当前的设计，不是丢点 bug。

### 3. metrics 要不要重跑

- 如果你重新 fit 了 UMAP：
  - 不需要为了 UMAP 形状去重跑 hidden-space metrics
- 但如果你重新抽了 `stage*_features.pt`
  - 那对应的 stage metrics 自然应该一起更新
