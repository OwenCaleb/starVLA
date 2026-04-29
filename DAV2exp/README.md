# DAV2exp

这个目录用于从一段 RGB episode 中导出三类 probe target：

- `spa_hq`：高质量深度图
- `spa_lq`：低质量深度图
- `dyn`：基于高质量深度图的相邻帧残差变化图

当前主脚本：

- [export_depth_targets.py](/mnt/nas_ssd/workspace/wenboli/projects/starVLA/DAV2exp/export_depth_targets.py)

## 目录约定

### 输入

默认输入目录：

- `DAV2exp/episode/`

要求：

- 帧名可按时间顺序排序，例如 `frame_000001.jpg`
- 脚本按文件名字典序逐帧处理

### 输出

默认输出目录：

- `DAV2exp/output/`

输出结构：

- `spa_hq/depth_npy/`
  - 高质量深度 target，原始 `float32 .npy`
- `spa_hq/depth_png/`
  - 高质量深度可视化图，`.png`

- `spa_lq/depth_npy/`
  - 低质量深度 target，原始 `float32 .npy`
- `spa_lq/depth_png/`
  - 低质量深度可视化图，`.png`

- `dyn/residual_npy/`
  - 动态残差 target，原始 `float32 .npy`
- `dyn/mask_npy/`
  - 动态区域二值 mask，`uint8 .npy`
- `dyn/residual_png/`
  - 黑底热图可视化，`.png`

## 哪些文件用于训练

如果你后面要做 probe 或监督训练，应该优先使用：

- `spa_hq/depth_npy/*.npy`
- `spa_lq/depth_npy/*.npy`
- `dyn/residual_npy/*.npy`
- `dyn/mask_npy/*.npy`

下面这些文件只是为了人工看效果：

- `*_png/*.png`

不要把 `png` 直接拿去做训练目标，因为它们已经做过归一化和 colormap 映射。

## 三类 target 的定义

### 1. spa_hq

高质量深度分支默认使用：

- `Depth-Anything-V2-Large`
- `encoder = vitl`
- `input_size = 518`

这条分支尽量贴近你当前 spatial teacher 的质量上限。

### 2. spa_lq

低质量深度分支默认使用：

- `Depth-Anything-V2-Small`
- `encoder = vits`
- `input_size = 224`

然后再附加一个非常轻的高斯模糊：

- `kernel = 3`
- `sigma = 0.6`

这条分支的目标不是“错误深度”，而是“质量显著更差、但仍保持基本结构”的深度图。

### 3. dyn

`dyn` 不再使用简单的 `depth(t+1) - depth(t)`，因为单目深度本身会有尺度抖动，直接相减通常很花。

当前方案如下。

#### 第一步：相邻帧深度仿射对齐

设两帧深度分别为：

- `D_t`
- `D_{t+1}`

先拟合：

```text
D'_{t+1} = a * D_{t+1} + b
```

其中 `(a, b)` 用最小二乘估计，使得对齐后的下一帧尽量接近当前帧。

#### 第二步：计算绝对残差

```text
R_t = |D_t - D'_{t+1}|
```

这样得到的是“哪里发生了明显变化”，而不是带正负号的混乱差分图。

#### 第三步：轻度平滑

默认做一次小高斯平滑：

- `kernel = 3`
- `sigma = 1.0`

目的是压掉深度抖动带来的细碎噪点。

#### 第四步：阈值筛选显著变化区域

默认使用：

- `90th percentile`

也就是只保留 top 10% 的变化区域，这样图通常更干净，也更适合可视化。

如果你想更保守，也支持：

- `median + 2.5 * MAD`

#### 第五步：形态学和连通域清理

默认做：

- 开运算一次
- 闭运算一次
- 去掉面积小于 `50 px` 的连通块

最后得到一张干净的变化区域 mask。

#### 第六步：最终可视化

最终 `dyn` 可视化图采用：

- 黑底
- 仅对 mask 内保留下来的区域做归一化
- `inferno` colormap

所以图的语义很直接：

- 黑色：基本不变
- 亮色：明显变化
- 越亮：残差越大

## 可视化策略

### spa 的可视化

深度图可视化采用：

- `2% - 98%` 的 robust min-max normalization
- `COLORMAP_INFERNO`

这比直接全局 min-max 更稳，不容易被少量极端值压扁。

### dyn 的可视化

动态图可视化采用：

- 黑底
- 只在有效变化区域内归一化
- `COLORMAP_INFERNO`

目的不是“重现真实 motion”，而是更清楚地显示“哪里发生了明显深度变化”。

## 默认超参数

当前脚本默认参数：

- `spa_hq`
  - `checkpoint = Depth-Anything-V2-Large`
  - `encoder = vitl`
  - `input_size = 518`

- `spa_lq`
  - `checkpoint = Depth-Anything-V2-Small`
  - `encoder = vits`
  - `input_size = 224`
  - `blur(kernel=3, sigma=0.6)`

- `dyn`
  - affine alignment
  - absolute residual
  - `Gaussian blur(kernel=3, sigma=1.0)`
  - threshold = `90th percentile`
  - morphology = `open -> close`
  - `min_area = 50`
  - black background + `inferno`

## 推荐命令

```bash
python DAV2exp/export_depth_targets.py \
  --episode-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/DAV2exp/episode \
  --output-dir /mnt/nas_ssd/workspace/wenboli/projects/starVLA/DAV2exp/output \
  --hq-checkpoint-path /mnt/nas_ssd/workspace/wenboli/projects/starVLA/playground/Pretrained_models/Depth-Anything-V2-Large \
  --lq-checkpoint-path /mnt/nas_ssd/workspace/wenboli/projects/starVLA/playground/Pretrained_models/Depth-Anything-V2-Small \
  --hq-encoder vitl \
  --lq-encoder vits \
  --hq-input-size 518 \
  --lq-input-size 224 \
  --device cuda
```

## 常见调参

### 想让低质量深度图更糊一点

可以调：

```bash
--lq-blur-kernel 5 --lq-blur-sigma 0.8
```

### 想让 dyn 图更保守

可以改成 MAD 阈值：

```bash
--dyn-threshold-mode mad --dyn-mad-scale 2.5
```

### 想让 dyn 图更激进

可以把百分位再抬高，比如：

```bash
--dyn-threshold-percentile 92
```

## 额外说明

- `dyn` 默认基于 `spa_hq` 计算，而不是 `spa_lq`。这是刻意的，因为高质量深度更稳，更适合作为动态 target。
- 最后一帧没有下一帧，因此不会生成对应的 `dyn` target。
- 如果你后面要把这批结果接到 probe 里，建议把 `residual_npy` 和 `mask_npy` 一起保留，这样可以做：
  - 纯 residual 回归
  - residual + mask 的显著区域监督
