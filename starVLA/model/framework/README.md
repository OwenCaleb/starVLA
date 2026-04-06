# starVLA Model Frameworks

本文档深入介绍 `starVLA` 框架下各个 VLA（Vision-Language-Action）模型架构、核心特点和应用场景。所有框架继承自 `baseframework`，支持统一的模型加载、推理和训练接口。

---

## 目录

1. [基础架构](#基础架构)
2. [框架矩阵对比](#框架矩阵对比)
3. [详细架构分析](#详细架构分析)
4. [技术栈和创新](#技术栈和创新)
5. [选型指南](#选型指南)

---

## 基础架构

### baseframework（BaseClass）

所有框架模型继承自 `baseframework`，该基类提供：

- **模型加载**: 从 checkpoint 恢复配置、权重和归一化统计
- **预训练接口**: `from_pretrained()` 标准化权重加载
- **HF 集成**: 继承自 `PreTrainedModel`（HuggingFace Transformers）
- **模块发现**: 自动识别可训练参数

**核心接口**：

```python
# build_framework 工厂函数
model = build_framework(cfg)  # 根据 cfg.framework.name 调度不同框架

# 支持的方法
model(batch)  # forward 训练
model.predict_action(images, instructions, states)  # inference
```

---

## 框架矩阵对比

| 框架 | 完整名 | Action Head | 条件机制 | 特色 | 状态 |
|------|--------|-------------|---------|------|------|
| **QwenPI** | Prediction with Instruction | LayerwiseFM（Flow-Matching） | 多层交叉注意力 | 层级化 DiT + 去噪 | 生产 |
| **QwenGR00T** | GR00T N1.5 适配 | GR00T FlowMatching | 单层全局条件 | 官方算法 + DINO 多视 | 生产 |
| **QwenAdapter** | 适配器架构 | MLP 回归头 | 动作查询 tokens | 轻量级、稳定、无扩散 | 生产 |
| **QwenFast** | FAST 离散动作 | FAST Tokenizer | 离散 token BPE | 离散动作·高效压缩 | 试验 |
| **QwenOFT** | 在线流程微调 | MLP + OFT | 动作 token 特殊化 | 实时微调、紧凑 token | 试验 |
| **QwenDual** | 双分支预测 | 混合 + MLP | VLM + 专用 Encoder | 多任务解耦 | 试验 |
| **NeuroVLA** | 神经脉冲网络 | SNN LIF 神经元 | GRU 状态调制 | 生物启发·时间滚动 | 研究 |
| **LangForce** | 语言驱动控制 | 混合 head | 多粒度语言分解 | 指令细粒度·符号化 | 试验 |
| **M1 (InternVLA)** | InternVLA-M1 | 定制 head | 跨模态对齐 | 通用视觉·多任务 | 基线 |
| **ABot_M0** | ABot 第 0 版 | 简单 MLP | 基础条件 | 轻量参考 | 基线 |

---

## 详细架构分析

### 1. QwenPI（Prediction with Instruction） ⭐⭐⭐⭐⭐

**设计哲学**: 多层级流匹配 + 层序在树化去噪

**架构**：

```
Input (Images, Instruction)
    ↓
Qwen2.5-VL (36 个隐层)
    ↓
提取最后 N 层隐藏态
    ↓
LayerwiseFlowmatchingActionHead (DiT)
  - Block 0 接最后一层
  - Block 1 接倒数第二层
  - ...
  - Block (N-1) 接倒数第 N 层
    ↓
Action Noise → Denoise → 预测动作
```

**关键特性**：

- **层级化条件**: 每个 DiT Block 从不同语义层次的 VLM 特征条件化
- **多时间步采样**: `repeated_diffusion_steps=4`，一个 batch 训练 4 倍的噪声去噪轨迹
- **流匹配**（vs 扩散）: 学习 ODE 速度场而非噪声预测，收敛更快
- **图位置对齐**: 支持 `past_action_window_size` 和 `future_action_window_size` 组合

**优势**：
- ✅ 充分利用多层 VLM 表示
- ✅ 生成质量高（流匹配训练稳定）
- ✅ 支持变长动作预测

**劣势**：
- ❌ 模型较大、推理需多步去噪
- ❌ 计算复杂度高

**应用场景**: 需要高质量、多样化动作的复杂任务

---

### 2. QwenGR00T（GR00T N1.5 实现）⭐⭐⭐⭐

**设计哲学**: 单层全局流匹配 + DINO 多视图稀疏特征

**架构**：

```
Input (Multi-view Images, Instruction)
    ↓
Qwen2.5-VL (提取最后一层)
    ↓
[可选] DINO 编码器 (dense spatial tokens)
    ↓
FlowmatchingActionHead（DiT）
  - 全局条件化（所有 Block 共用同一 encoder_hidden_states）
    ↓
Flow ODE Solver → Actions
```

**关键特性**：

- **单层条件**: VLM 最后隐层直接作为所有 DiT 块的全局条件
- **DINO 增强**: 可选的密集多视角空间特征编码
- **官方算法**: 接近 GR00T 原始论文实现
- **轻量级**: 比 QwenPI 更简洁

**优势**：
- ✅ 推理速度快于 QwenPI
- ✅ 多视角天然支持
- ✅ 少参数量（单层条件化）

**劣势**：
- ❌ 比 QwenPI 少一个维度特征利用
- ❌ DINO 增加依赖

**应用场景**: 权衡质量和速度，多视角数据集

---

### 3. QwenAdapter（轻量级线性适配器）⭐⭐⭐⭐

**设计哲学**: 动作查询 token + 基于注意力的状态聚合

**架构**：

```
Input (Images, Instruction with Action Placeholders)
    ↓
Qwen2.5-VL (所有层)
    ↓
Hook: 将 🔍 符号向量
替换为可学习的 action_query tokens
    ↓
提取多层 vision features 和 action_query features
    ↓
VLA_Adapter_L1RegressionActionHead (MLP ResNet)
  - 24 个 ResNet 块
  - 注意力层对接 vision/action/proprioception
  - L1 直接回归
    ↓
Continuous Actions (无去噪)
```

**关键特性**：

- **即插即用的 action token**: 通过 embedding hook 植入动作查询位置
- **多源融合**: Vision patches + Action queries + Proprioception
- **即时预测**: 无扩散步，单 forward = 最终动作
- **稳定训练**: L1 损失 + 线性预测

**优势**：
- ✅ 最快推理（无迭代去噪）
- ✅ 最轻量级模型
- ✅ 多视角视觉 token 场景下 bug 修复（multi-segment vision blocks）
- ✅ 支持本体感觉融合

**劣势**：
- ❌ 生成多样性受限（直接回归）
- ❌ 动作平滑性依赖训练数据

**应用场景**: 实时机器人控制、边缘设备部署、构型相似任务

---

### 4. QwenFast（离散 FAST Action Tokenizer）⭐⭐⭐

**设计哲学**: 动作空间离散化 + BPE 压缩

**架构**：

```
Input (Images, Instruction)
    ↓
Qwen2.5-VL
    ↓
FAST_ActionHeader
  │
  ├─ DCT 变换 (时频分解)
  ├─ 量化 (离散化)
  ├─ BPE tokenizer (可变长编码)
  │
  └─ 预测 action_token_id
    ↓
Inverse Transform → Continuous Actions
```

**关键特性**：

- **动作 TOKEN 化**: 通过 DCT + 量化将连续动作转为离散 token
- **变长编码**: 简单动作少 token，复杂动作多 token
- **压缩效率**: 相比浮点序列，参数量大幅降低
- **特殊 token 扩展**: 支持往 Qwen tokenizer 增加 FAST 专用 token

**优势**：
- ✅ 参数高效（变长 token 表示）
- ✅ 天然对应离散行为库
- ✅ 便于多模态学习融合

**劣势**：
- ❌ 量化过程可能丧失精度
- ❌ 逆变换计算开销

**应用场景**: 离散动作空间、轻量部署、预算受限场景

---

### 5. QwenOFT（在线流程微调）⭐⭐⭐

**设计哲学**: 参数高效微调 + 动作 token 特殊化

**架构**：

```
Input (Images, Instruction)
    ↓
Qwen2.5-VL (可选冻结)
    ↓
OFT (Orthogonal Finetuning)
    ↓
MLP Action Head
  │
  ├─ 提取 action_token (<action>) 隐藏态
  ├─ MLP 投影
  │
  └─ 动作预测
    ↓
Actions
```

**关键特性**：

- **OFT 约束**: 参数更新限制在正交变换，参数高效 ~ 1-2% 原始量
- **动作 token 标记化**: 特殊 `<action>` token 锚定动作预测位置
- **实时适配**: 支持快速域适应不需要全参数微调

**优势**：
- ✅ 参数高效（OFT）
- ✅ 快速适配新机器人/任务
- ✅ 推理简洁

**劣势**：
- ❌ 动作 token 多 id 映射脆弱性（需强制特殊化）
- ❌ 生成多样性受限

**应用场景**: 快速任务迁移、资源受限训练、实时微调

---

### 6. NeuroVLA（生物启发·脉冲神经）⭐⭐⭐

**设计哲学**: 脉冲神经网络 + GRU 时间建模 + 滚动状态递推

**架构**：

```
Input (Images, Instruction, Robot States)
    ↓
Qwen2.5-VL (视觉编码)
    ↓
LayerwiseQFormer
  (跨层注意力聚合)
    ↓
Iterative Refinement Loop (2× 迭代):
  ├─ GRU_GatedFiLModulator
  │  └─ 用 GRU 编码状态历史 → FiLM 参数
  │
  ├─ L1RegressionActionHead (SNN + LIF)
  │  └─ 脉冲神经元 + 膜电位 → 动作
  │
  └─ _roll_states_with_predictions
     └─ 时间滚动窗口 + 新动作追加末尾
    ↓
Final Actions (自然时间语义)
```

**关键特性**：

- **脉冲神经元**: Leaky Integrate-and-Fire (LIF) + Surrogate Gradient
- **时间建模**: GRU 捕捉状态历史的长期依赖
- **滚动状态递推**: 避免"伪历史"，每轮迭代状态自然前进
- **多轮迭代**: 2-10 轮迭代精化动作预测

**优势**：
- ✅ 时间语义自然（GRU + 状态滚动）
- ✅ 生物启发（脉冲计算）
- ✅ 多次迭代精化

**劣势**：
- ❌ 模型复杂，收敛慢
- ❌ SNN 参数调优困难
- ❌ 最终动作长度 = 单轮 × 迭代轮数（需确认）

**应用场景**: 需要时间一致性的复杂任务、强化学习集成

---

### 7. QwenDual（双分支解耦预测）⭐⭐⭐

**设计哲学**: 视觉与语言分离处理 + 二部图融合

**架构**：

```
Input (Images, Instruction)
    ↓
Qwen2.5-VL
    ↓
Branch 1 (Vision)          Branch 2 (Language)
    ↓                           ↓
Vision Encoder             Language Encoder
    ↓                           ↓
Feature Fusion (Cross-Attention)
    ↓
Action Head (MLP)
    ↓
Actions
```

**关键特性**：

- **任务解耦**: 显式分离视觉和语言流
- **双编码器**: 各自的投影和处理空间
- **晚融合**: 在高层特征空间融合

**优势**：
- ✅ 模块化、易于调试
- ✅ 支持多任务微调

**劣势**：
- ❌ 较少使用（相比单一 VLM 路径）

**应用场景**: 需要视觉-语言解耦的多任务学习

---

### 8. LangForce（语言驱动控制）⭐⭐⭐

**设计哲学**: 细粒度语言分解 + 符号化动作编码

**架构**：

```
Input (Images, Structured Instruction, Goals)
    ↓
Qwen2.5-VL (或 Qwen3-VL)
    ↓
语言解析器
  (提取子目标、约束、符号化表征)
    ↓
混合 Action Head
  ├─ 连续执行器动作
  ├─ 离散工具调用
  │
  └─ 符号约束投影
    ↓
Actions (受语言约束)
```

**关键特性**：

- **细粒度语言**: 不仅指令，还包括目标分解
- **多粒度表示**: 从符号到连续
- **约束投影**: 动作需满足语言指定的约束

**优势**：
- ✅ 指令跟随精度高
- ✅ 可解释性强

**劣势**：
- ❌ 语言解析复杂
- ❌ 推理开销大

**应用场景**: 复杂交互任务、高精度指令跟随

---

### 9. M1 / InternVLA-M1（参考基线）⭐⭐

**特点**：
- 通用视觉-语言模型
- 多任务适配
- 较为简洁的框架

**应用场景**: 基线对比、泛化性能测试

---

### 10. ABot_M0（轻量参考）⭐⭐

**特点**：
- 高度精简
- 基础 MLP head
- 无特殊优化

**应用场景**: 学习用参考实现

---

## 技术栈和创新

### 核心模块

1. **VLM Backend 抽象**
   - Qwen2.5-VL / Qwen3-VL / Qwen3.5-VL
   - Florence2 / CosmosReason2
   - 统一 `get_vlm_model()` 接口

2. **Action Head 多样化**
   - Flow-Matching DiT (QwenPI, QwenGR00T)
   - MLP Regression (QwenAdapter, QwenOFT)
   - SNN LIF (NeuroVLA)
   - FAST Tokenizer (QwenFast)

3. **数据路径两条**
   - VLA (LeRobot 多数据集混合)
   - VLM (LLaVA-JSON format)

4. **状态条件化机制**
   - 无状态 (QwenPI, QwenGR00T)
   - Proprioception Projector (QwenAdapter)
   - GRU State Encoder (NeuroVLA)
   - FiLM Modulation (多模型)

### 设计亮点

- **多视角支持**: 自动处理可变数目的图像 + 特殊 token 边界修复
- **归一化独立**: 每个数据集保持独立的 min/delta 参数（解决 Wall-X 多数据集冲突）
- **冻结机制**: 通过 trainer 层级的 `freeze_modules` 灵活控制
- **时间语义修正**: NeuroVLA 中的滚动状态递推避免伪历史

---

## 选型指南

### 决策树

```
需求: 实时机器人控制?
  ├─ 是 → QwenAdapter (最快、稳定)
  └─ 否
       需要高质量多样动作?
         ├─ 是 → QwenPI (层级化 FlowMatching)
         └─ 否
              需要离散动作?
                ├─ 是 → QwenFast (FAST tokenizer)
                └─ 否
                     需要参数高效微调?
                       ├─ 是 → QwenOFT (OFT + Action Token)
                       └─ 否
                            需要时间建模?
                              ├─ 是 → NeuroVLA (GRU + SNN)
                              └─ 否 → QwenGR00T (平衡)
```

### 参考配置

- **快速原型**: `starvla_train_adapter.yaml` + QwenAdapter
- **通用任务**: `starvla_cotrain_libero.yaml` + QwenGR00T
- **高质量生成**: `starvla_cotrain_oxe.yaml` + QwenPI
- **实验研究**: NeuroVLA / LangForce / QwenDual

---

## 模块内部组织

### 文件结构

```
framework/
├── base_framework.py           # 基类（单体）
├── share_tools.py              # 公共工具
├── __init__.py                 # 工厂函数
│
├── QwenPI.py                   # 层册化 FlowMatching
├── QwenGR00T.py                # 单层 FlowMatching
├── QwenAdapter.py              # 线性适配 + 多视处理修复
├── QwenFast.py                 # FAST Tokenizer
├── QwenOFT.py                  # OFT 微调
├── QwenDual.py                 # 双分支
├── NeuroVLA.py                 # 脉冲 + GRU 滚动更新
├── LangForce.py                # 语言驱动
├── M1.py                        # InternVLA-M1
├── ABot_M0.py                  # ABot 参考
│
└── README.md                   # 本文档
```

---

## 常见问题

### Q: 多视角图片如何处理？
**A**: 通过 `build_qwenvl_inputs(images=[img1, img2, ...])` 直接传入列表。Qwen VLM 内部会拼接，`QwenAdapter` 新增了视觉边界 token 修复确保真实 image_pad 不被误算。

### Q: 怎样添加新的 action head？
**A**: 
1. 在 `model/modules/action_model/` 下创建新文件
2. 实现 `predict_action(hidden_states) → actions` 方法
3. 在 framework 模型的 `__init__` 里调用 `get_action_model()`
4. 注册到 `@FRAMEWORK_REGISTRY`

### Q: Freezing 工作原理？
**A**: 在 trainer 的 `prepare_training()` 中，读取 `cfg.trainer.freeze_modules` 字符串列表，通过 `freeze_backbones()` 递归冻结模块路径。optimizer 会通过 `build_param_lr_groups()` 自动排除冻结参数。

### Q: 如何选择 diffusion steps？
**A**: 
- 生成质量优先: `num_inference_timesteps=10~20`
- 速度优先: `num_inference_timesteps=4`
- 训练时 `repeated_diffusion_steps=4` 充分利用梯度

---

## 参考文献与相关工作

- **GR00T**: [GrooT: Scaling 3D Object Detection in Autonomous Driving]
- **Flow-Matching**: [Flow Matching for Generative Modeling]
- **OFT**: [Orthogonal Finetuning of Large Language Models]
- **FAST**: [Flexible Action Space Tokenization]
- **LeRobot**: [LeRobot Dataset Collection & Training]

---

## 贡献和扩展

欢迎添加新框架！请确保：

1. ✅ 继承 `baseframework`
2. ✅ 实现 `forward()` 和 `predict_action()`
3. ✅ 注册到 `@FRAMEWORK_REGISTRY`
4. ✅ 在本 README 更新描述
5. ✅ 提供配置示例 YAML

---

**最后更新**: 2025-04-02  
**维护者**: starVLA 社区  
**许可**: MIT License
