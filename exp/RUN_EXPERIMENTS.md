# 运行命令

当前这份文件只保留可直接执行的命令，全部路径已经填成当前实际使用的绝对路径。

## 固定输入

- 视频目录：
  - `/mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/video`
- 指令来源：
  - 每个视频文件名 stem
- subtask 来源：
  - 与 instruction 相同
- 输出目录：
  - `/mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis`

## 固定 checkpoint

- Stage I
  - `/mnt/nas_ssd/workspace/wenboli/projects/starVLA/results/Checkpoints/myvla_stage1/checkpoints/steps_15000_pytorch_model.pt`
- Stage II
  - `/mnt/nas_ssd/workspace/wenboli/projects/starVLA/results/Checkpoints/myvla_stage2/checkpoints/steps_65000_pytorch_model.pt`
- Stage III
  - `/mnt/nas_ssd/workspace/wenboli/projects/starVLA/results/Checkpoints/myvla_stage3/checkpoints/steps_50000_pytorch_model.pt`

---

## 1. 从多视频目录抽取三阶段中间特征

```bash
python exp/extract_stage_features.py \
  --stage1-config /mnt/nas_ssd/workspace/wenboli/projects/starVLA/starVLA/config/training/myvla_stage1.yaml \
  --stage1-ckpt /mnt/nas_ssd/workspace/wenboli/projects/starVLA/results/Checkpoints/myvla_stage1/checkpoints/steps_15000_pytorch_model.pt \
  --stage2-config /mnt/nas_ssd/workspace/wenboli/projects/starVLA/starVLA/config/training/myvla_stage2.yaml \
  --stage2-ckpt /mnt/nas_ssd/workspace/wenboli/projects/starVLA/results/Checkpoints/myvla_stage2/checkpoints/steps_65000_pytorch_model.pt \
  --stage3-config /mnt/nas_ssd/workspace/wenboli/projects/starVLA/starVLA/config/training/myvla_stage3.yaml \
  --stage3-ckpt /mnt/nas_ssd/workspace/wenboli/projects/starVLA/results/Checkpoints/myvla_stage3/checkpoints/steps_50000_pytorch_model.pt \
  --episode-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/video \
  --output-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis \
  --device cuda \
  --analysis-progress 1.0 \
  --temporal-delta-indices 0 19
```

---

## 2. 构建 QK Sankey CSV

```bash
python exp/build_qk_sankey.py \
  --features-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis/intermediate \
  --output-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis
```

---

## 3. 拟合共享 UMAP artifact

```bash
python exp/build_token_umap.py \
  --features-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis/intermediate \
  --output-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis \
  --fit-only \
  --skip-metrics \
  --max-tokens-per-stage 0 \
  --l2-normalize \
  --umap-metric cosine \
  --umap-min-dist 0.1
```

如果你想更紧凑，可以继续减小：

```bash
--umap-min-dist 0.05
```

---

## 4. 导出三阶段 UMAP

```bash
python exp/build_token_umap.py \
  --features-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis/intermediate \
  --output-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis \
  --transform-only \
  --transform-stages stage1 stage2 stage3 \
  --l2-normalize \
  --umap-metric cosine \
  --skip-metrics
```

---

## 5. 导出主图指标表

```bash
python exp/build_stage_metric_tables.py \
  --features-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis/intermediate \
  --output-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/exp/output/video_analysis \
  --l2-normalize \
  --distance-metric cosine
```

---

## 6. 备注

- 当前 `extract_stage_features.py` 会自动：
  - 从 `exp/video/` 下所有 `.mp4/.avi/.mov/.mkv/.webm` 逐帧解码
  - 从每个视频文件名 stem 自动生成 `instruction/subtask`
  - 构造 temporal `video`
  - 为 attention 导出临时切到 `eager`
- 当前 `build_token_umap.py` 不是 token-level：
  - 每帧每个 group 的 token hidden 会先平均
  - 再作为一个点进入 UMAP
- 当前默认还会：
  - 对 hidden 做 `L2 normalize`
  - 如果 `--umap-metric cosine`，则在 PCA 后再做一次 `L2 normalize`
  - 使用 `cosine` 距离
- 当前 UMAP all-token 模式下每帧最多 6 个点：
  - `lang / vis / sub / dyn / spa / act`
- 当前 UMAP slot-only 模式下每帧最多 4 个点：
  - `sub / dyn / spa / act`
