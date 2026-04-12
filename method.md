# 0. 最终建议的研究主线

## 核心 claim

**在尽量不破坏预训练 Qwen3-VLM backbone 的前提下，将 dynamic / spatial / subtask 三类信息通过 token-aware compression + trainable projection 映射为固定 slot token，并以占位符替换的方式直接注入主干，从而提升 VLA 的控制表征质量与控制性能。**

## 当前版本故意不做的事

* **不把 QFormer + Flow Matching 当核心创新点**
* **不使用双分支蒸馏 / teacher-student 双前向**
* **不使用 learned top-k router**
* **不把整个 backbone 全解冻**

## 当前代码实现快照（2026-04-07 更新）

**核心框架与测试组织：**
* 三路 teacher wrapper 已接入：dynamic / spatial / subtask （全量冻结）
* 三路 slot 已接入：压缩投影 + 占位符替换注入
* 全层 typed residual FFN 已接入（按 token type residual expert，层间 expert 不共享）
* typed self-attention mask 已实现生成与可视化 smoke，当前已真正接入 backbone 全层 self-attention 前向（`BackHalfTypedSelfAttention` 包装器，现用于全层）
* slot readout 已改为纯 MLP residual fuser（无 SA / 无 cross-attn）

**损失与阶段训练：**
* 三路 slot 对齐损失已接入：`MSE + cosine`，并并入 `action_loss`
* 监督动作损失已接入（有 GT action 时 L1）
* Stage3 action header 已接入：使用 `MyVLA_AdapterHeader`，读取所有层 hidden
* fast loss 与 continuous loss 已显式区分：`fast_action_loss` / `continuous_fm_loss`
* detach 语义已接入：两条分支均可独立控制是否反传到上游
* continuous_fm_loss 已迁移到真实 FM 目标（velocity supervision + configurable loss_type）
* Stage II masked target prediction 闭环已接入：slot keep mask 与 slot align loss 显式绑定

**代码组织（NEW：2026-04-07 重构）：**
* **QwenMyVLA.py**：Framework 核心运行时模块，1367 行（从2050路减少33%）
  - 包含：TypedResidualFFN、BackHalfTypedSelfAttention、TokenAwareResampler、SlotMLPFuser、QwenMyVLA 主类
  - 方法：_build_dynamic_slots、_build_spatial_slots、_build_subtask_slots、forward、predict_action、loss 计算等
  - 简化的 __main__：5行内链接到 qwen_myvla_smoke.main()
* **qwen_myvla_smoke.py**：独立 smoke 测试模块，330 行（NEW）
  - 包含：_build_mock_model() 工厂函数、6个 smoke test 函数、main() argparse 入口
  - 所有前推合约、loss 语义、typed attention、masked target 等测试都在这里
  - 可独立运行或作为 pytest fixtures 导入

**配置对齐：**
* **myvla.yaml**：已完整对齐 starvla_cotrain_libero.yaml 结构
  - Framework 块：qwenvl、lam_stage2、depth_encoder、qwen3_embedding、slot_*、slot_fuse、slot_align_loss、continuous_fm_loss、fast_action_loss、action_model、stage3_action_header、placeholder 定义、attention_mask
  - Datasets 块：vla_data (lerobot_datasets, bridge_rt_1, delta_ee, CoT fields)
  - Trainer 块：10个 LR 组 (base, qwen_vl_interface, 三路 teacher encoder, 三路 compressor, action_predictor, stage3_action_head)，optimizer (AdamW)，scheduler (cosine_with_min_lr)，gradient checkpointing + mixed precision
  - 所有必要字段已补全；pragmatic 注释（非冗长）已添加

**验证状态：**
* ✓ Python 语法检查：QwenMyVLA.py + qwen_myvla_smoke.py 通过 py_compile
* ✓ OmegaConf 解析：myvla.yaml 正确加载，所有字段可访问
* ✓ 所有 smoke tests：5/5 通过 `[OK]` 验证 (train/predict 合约、loss 分支、typed attention、all-layer apply、masked target prediction)
* ✓ 模块导入：核心类、smoke 函数都可从 Python 导入
* ✓ 文件大小：Framework 1142 行（非注释），smoke 267 行，总计 1409 行有效代码

**接口对齐：**
* `predict_action` 对外接口已对齐为 `normalized_actions`
* 文件内 train/predict 合同测试 + loss 分支语义 smoke 已接入并跑通
* **实时数据前向 smoke 已验证通过**：LIBERO dataloader + 三路 teacher 编码器在真实数据上前向成功
* **MyVLA 专用训练入口已提供**：`starVLA/training/train_starvla_myvla.py`（默认配置 `starVLA/config/training/myvla.yaml`）

## 下面要做什么（优先级）

1. 完整端到端训练验证（P2 可选）
- 集成与 Qwen3-VL 或兼容模型架构的完整训练实跑
- 验证多 epoch 收敛趋势

2. Coarse action 分支与可见性规则（P2）
- 加入 coarse action token/head，约束只读 D/S/U/R/指令信息
- Smoke: 前向输出与 loss 稳定

---

修订后的Method版本（最终版本）：

---

# 3 Method

## 3.1 Overview

我们关注的核心问题是：机器人控制所依赖的**动态信息、空间信息与任务语义信息**本质上是异质的，但现有 VLA 往往将其压缩在同一条统一的 token stream 中，依赖 backbone 自行完成区分、保留与调用。这会导致强语义 backbone 虽能理解任务，却未必形成真正**面向控制**的中间表征。

为此，我们提出一个三阶段的结构化训练框架。整体思路是：首先在稳定 teacher 监督下为不同类型的信息建立各自的 specialized latent pathway；随后逐步移除显式 scaffold，并引入离散 coarse action 约束，强制模型建立
[
\text{Image/Instruction}\rightarrow\text{Structured Slots}\rightarrow\text{Action}
]
的信息桥；最后再以 coarse discrete actions 为条件，通过 DiT-based flow matching 生成连续动作，实现 coarse-to-fine 的控制链路。

整个方法包含三部分核心设计：

1. **多源异质先验的 slot 化输入**：引入 dynamic / spatial / subtask 三类 slot；
2. **全层 type-specific FFN**：在所有 transformer layers 中保留 shared base FFN，并为不同 token 类型引入 specialized FFN；
3. **分阶段训练**：从 slot alignment，到 scaffold annealing + coarse action grounding，再到 continuous action refinement。

---

## 3.2 Tokenization and Frozen Teachers

### Base tokens

给定当前观测 (o_t) 与语言指令 (x)，我们首先构造两类基础 token：

* **Image tokens**：由视觉编码器从图像观测中提取；
* **Instruction tokens**：由 Qwen tokenizer / text stack 从语言指令中提取。

这两类 token 共同构成基础多模态上下文，记为
[
\mathbf{z}^{\text{base}}=[\mathbf{z}^{\text{img}}, \mathbf{z}^{\text{ins}}].
]

### Dynamic slots

我们使用冻结的 dynamic encoder 提取与未来变化相关的动态先验。具体地，给定 ((o_t, o_{t+\Delta}))，dynamic teacher 输出 latent action code indices。随后我们将其映射为 dynamic slot scaffold，并经由线性投影映射到 Qwen hidden space，得到
[
\mathbf{z}^{\text{dyn}} \in \mathbb{R}^{K_d \times H}.
]

### Spatial slots

我们使用冻结的 spatial encoder 从时间窗中的后续观测提取空间几何先验。实现中优先使用 temporal clip 的后帧（next image / later image），再由冻结的深度/空间编码器（如基于深度特征的 encoder）输出中间特征，经投影得到
[
\mathbf{z}^{\text{spa}} \in \mathbb{R}^{K_s \times H}.
]

### Subtask slots

我们使用冻结的 **Qwen3-Embedding** 作为 subtask semantic teacher，对 subtask text 进行编码。其输出语义表示经过投影后形成 subtask slot scaffold：
[
\mathbf{z}^{\text{sub}} \in \mathbb{R}^{K_u \times H}.
]

### Teacher stability

上述三个外部 encoder（dynamic / spatial / subtask）在所有训练阶段均保持冻结。这样做的目的有二：

1. 保证三类 teacher target 在 staged training 中始终稳定；
2. 将训练重点集中在 VLM 内部的 type-specific latent organization，而非 teacher 本身的漂移。

最终，输入到 backbone 的 structured token groups 为
[
[\mathbf{z}^{\text{img}}, \mathbf{z}^{\text{ins}}, \mathbf{z}^{\text{sub}}, \mathbf{z}^{\text{dyn}}, \mathbf{z}^{\text{spa}}, \mathbf{z}^{\text{act}}],
]
其中 (\mathbf{z}^{\text{act}}) 仅在 Stage II/III 中出现。

---

## 3.3 Full-Layer Typed FFN Backbone

在 backbone 结构上，我们不改变 Qwen 的 transformer 主干形式，但在**所有层**保留一个 shared `Base-FFN`，并为不同 token 类型引入专属的 type-specific FFN：

* `Dynamic-FFN`
* `Spatial-FFN`
* `Subtask-FFN`
* `Action-FFN`

设第 (l) 层 self-attention 输出为 (\mathbf{h}_l)，则对某个 token (t) 的 FFN 更新可写为：

[
\mathbf{h}_{l+1}^{(t)}=
\mathbf{h}_l^{(t)}

* \mathrm{FFN}_{\text{base}}^{(l)}(\mathbf{h}_l^{(t)})
* \mathbb{I}[t\in \mathcal{T}*k]\cdot \mathrm{FFN}*{k}^{(l)}(\mathbf{h}_l^{(t)}),
  ]

其中 (k \in {\text{dyn}, \text{spa}, \text{sub}, \text{act}})，(\mathcal{T}_k) 表示对应 token 类型集合。也就是说：

* image / instruction tokens 仅走 `Base-FFN`；
* dynamic slots 走 `Base-FFN + Dynamic-FFN`；
* spatial slots 走 `Base-FFN + Spatial-FFN`；
* subtask slots 走 `Base-FFN + Subtask-FFN`；
* action tokens 走 `Base-FFN + Action-FFN`。

这一设计的目的不是把所有信息彻底分裂，而是让 backbone 在保留共享多模态主干的同时，为不同性质的控制先验提供**持续存在的专门处理通路**。

---

## 3.4 Stage-Specific Attention Patterns

为了让不同阶段承担不同功能，我们在三个阶段采用不同的 group-wise attention pattern。为了描述方便，我们按以下顺序组织 token groups：

[
[\text{Image},\ \text{Instruction},\ \text{Subtask},\ \text{Dynamic},\ \text{Spatial},\ \text{Action}]
]

这里的 attention pattern 是**group-level self-attention visibility rule**，不是输入 mask。输入 mask（intra-mask / inter-mask）将在 Stage II 单独说明。

---

### Stage I attention

Stage I 中不引入 action tokens。三类 slots 的目标是先与各自 teacher 对齐，并建立稳定的 specialized pathway，因此不允许它们在一开始就进行过强的跨类型信息交换。当前阶段采用受限可见模式：

* Image 只看 Image；
* Instruction 看 Image + Instruction；
* Subtask 看 Image + Instruction + Subtask；
* Dynamic 看 Image + Instruction + Dynamic；
* Spatial 看 Image + Spatial。

也就是说，Stage I 的重点是**局部稳定对齐**，而不是立即进行充分融合。此时 action 路径完全关闭。

---

### Stage II attention

Stage II 的关键目标是：在 scaffold 逐步退场的同时，引入 FAST action 约束，并强制 action 通过 structured slots 获取信息，而不能直接读取 base context。当前阶段采用：

* Image 看 Image；
* Instruction 看 Image + Instruction；
* Subtask 看 Image + Instruction + Subtask + Dynamic + Spatial；
* Dynamic 看 Image + Instruction + Subtask + Dynamic + Spatial；
* Spatial 看 Image + Instruction + Subtask + Dynamic + Spatial；
* **Action 只看 Subtask + Dynamic + Spatial + past Action**，**不能直接看 Image / Instruction**。

这是整个方法的重要设计：
在 Stage II 中，coarse action 必须依赖三类 slot 才能生成，从而建立
[
\text{Image, Instruction}\rightarrow\text{Slots}\rightarrow\text{Action}
]
的中间桥梁，并使 action loss 反向约束 slot 表征。

---

### Stage III attention

Stage III 中，我们保留 FAST action 建模，但目标已经从“强制通过 slots 建桥”转向“支持完整的 coarse-to-fine action modeling”。因此恢复更标准的因果可见性模式：

* Image 看 Image；
* Instruction 看 Image + Instruction；
* Subtask 看 Image + Instruction + Subtask；
* Dynamic 看 Image + Instruction + Subtask + Dynamic；
* Spatial 看 Image + Instruction + Subtask + Dynamic + Spatial；
* Action 看全部先前 group 以及 past Action。

也就是说，Stage III 采用**正常因果自回归**的 coarse action 建模方式，为 continuous refinement 提供更完整的条件上下文。

---

## 3.5 Stage I: Slot Alignment

### Goal

Stage I 的目标不是直接学习控制，而是先在稳定 teacher 监督下，让三类 slot 与 backbone 内部的 type-specific pathway 完成对齐，为后续 teacher-free latent formation 提供稳定起点。

### Trainable modules

Stage I 中仅训练：

* `Dynamic-FFN`
* `Spatial-FFN`
* `Subtask-FFN`
* 三类 slot 输入投影
* 三类 slot adapter

### Frozen modules

Stage I 中冻结：

* `Self-Attention`
* `Base-FFN`
* `Action-FFN`
* `LM Head`
* FAST action token parameters
* continuous action head / DiT action head
* 所有 teacher encoders

### Losses

Stage I 只使用三类 alignment losses：

[
\mathcal{L}_{\text{I}}
======================

\lambda_{\text{dyn}}\mathcal{L}*{\text{dyn}}
+
\lambda*{\text{spa}}\mathcal{L}*{\text{spa}}
+
\lambda*{\text{sub}}\mathcal{L}_{\text{sub}}.
]

其中：

* (\mathcal{L}_{\text{dyn}})：dynamic slots 与 frozen dynamic teacher target 的对齐损失；
* (\mathcal{L}_{\text{spa}})：spatial slots 与 frozen spatial teacher target 的对齐损失；该 teacher 默认来自 temporal clip 的后帧，以约束后续时刻的空间结构；
* (\mathcal{L}_{\text{sub}})：subtask slot summary 与 frozen Qwen3-Embedding semantic target 的对齐损失。

这一阶段的本质是**先稳住**。我们不让 backbone 的共享主干在训练初期大范围更新，而是先让 specialized FFN 与 slot projection 学会各自负责的结构化信息。

---

## 3.6 Stage II: Scaffold Annealing

### Goal

Stage II 同时承担两件事：

1. 逐步移除显式 scaffold，使三类 slots 从“teacher-guided”过渡到“teacher-reduced”；
2. 引入 FAST discrete action，自回归地生成 coarse action，并用 action loss 反向约束三类 slots。

这也是本方法中最关键的阶段。

---

### Input masking curriculum

Stage II 在输入端引入两类 mask：

#### Intra-mask

在同一类 slot 内部随机失活部分 token。
例如，对于 Dynamic slots，随机遮蔽其中若干 token，使模型不能完全依赖完整动态 scaffold。

#### Inter-mask

一次性失活一种 slot。
例如，在某个训练样本中完全移除 Spatial slots，或完全移除 Subtask slots。

二者都作用于 **VLM 输入端**，即输入给 transformer 的 slot token 序列，而不是作用于 self-attention mask。
随着训练推进，intra-mask 与 inter-mask 的比例逐步增加，从而让显式 scaffold 逐渐退场。

---

### Trainable modules

Stage II 中训练：

* `Self-Attention`
* `Base-FFN`
* `Dynamic-FFN`
* `Spatial-FFN`
* `Subtask-FFN`
* `Action-FFN`
* 三类 slot 输入投影
* 三类 slot adapter
* `LM Head`
* FAST action token 相关参数

### Frozen modules

冻结：

* 所有 teacher encoders
* continuous action head / DiT action head

---

### FAST action supervision

我们使用扩充词表后的 Qwen LM Head 自回归生成 FAST action tokens。记 discrete coarse action 序列为 (\mathbf{a}^{\text{fast}})，则其损失为标准 next-token cross-entropy：

[
\mathcal{L}_{\text{fast}}
=========================

-\sum_t \log p(a_t^{\text{fast}} \mid a_{<t}^{\text{fast}}, \mathbf{z}^{\text{sub}}, \mathbf{z}^{\text{dyn}}, \mathbf{z}^{\text{spa}}).
]

注意，在 Stage II 中 coarse action **不能直接看 Image / Instruction**，因此它必须依赖三类 slots 才能形成有效动作预测。这使得 FAST action 不再只是一个输出端监督，而成为连接 structured latent 与 action generation 的桥梁。

---

### Total loss

Stage II 的总损失为：

[
\mathcal{L}_{\text{II}}
=======================

\lambda_{\text{dyn}}^{(II)}\mathcal{L}*{\text{dyn}}
+
\lambda*{\text{spa}}^{(II)}\mathcal{L}*{\text{spa}}
+
\lambda*{\text{sub}}^{(II)}\mathcal{L}*{\text{sub}}
+
\lambda*{\text{fast}}^{(II)}\mathcal{L}_{\text{fast}}.
]

其中，随着训练推进：

* (\lambda_{\text{dyn}}^{(II)},\lambda_{\text{spa}}^{(II)},\lambda_{\text{sub}}^{(II)}) 逐渐下降；
* (\lambda_{\text{fast}}^{(II)}) 逐渐上升。

这一权重迁移对应“从 scaffold 监督过渡到 action grounding”。

---

## 3.7 Stage III: Action Refinement

### Goal

Stage III 的目标是：在保留 FAST action 训练的同时，引入 continuous action generation。我们将 Stage II 产生的 coarse discrete action 作为条件输入，通过 DiT-based flow matching 生成最终 continuous action chunk。

---

### Continuous action head

设连续动作序列为 (\mathbf{u})，其噪声版本为 (\tilde{\mathbf{u}})。DiT action head 接收：

* coarse discrete action tokens
* noisy actions (\tilde{\mathbf{u}})
* optional robot state

并在 flow matching 目标下预测连续动作更新方向，最终输出 continuous action chunk。

我们将这一头记为：

[
\hat{\mathbf{u}} = \mathrm{DiT}*{\theta}(\tilde{\mathbf{u}}, \mathbf{a}^{\text{fast}}, \mathbf{s}*{\text{robot}}).
]

**Implementation note (current code path):**
当前仓库默认 Stage III 采用 coarse-conditioned `VLA_AdapterHeader` 路线作为可复现基线：

* coarse 条件优先来自 action token hidden（回退为 light coarse action 投影）；
* continuous 目标仍采用 flow-matching 风格监督。

完整 DiT-based refinement 作为后续增强路径保留，不影响当前三阶段训练闭环的可运行性。

其主损失为：

[
\mathcal{L}*{\text{cont}} = \mathcal{L}*{\text{FM}}.
]

---

### Trainable modules

Stage III 中的训练重点转向 continuous action header。具体来说：

#### 重点训练

* continuous action header / DiT action head

#### 低学习率协同训练

VLM 部分仍然参与训练，但采用显著更低的学习率：

* `Self-Attention`
* `Base-FFN`
* `Dynamic-FFN`
* `Spatial-FFN`
* `Subtask-FFN`
* `Action-FFN`
* slot 输入投影
* slot adapter
* `LM Head`
* FAST action 相关参数

### Frozen modules

* 所有 teacher encoders 继续冻结

---

### Losses

Stage III 的总损失为：

[
\mathcal{L}_{\text{III}}
========================

\lambda_{\text{cont}}\mathcal{L}*{\text{cont}}
+
\lambda*{\text{fast}}^{(III)}\mathcal{L}*{\text{fast}}
+
\epsilon*{\text{dyn}}\mathcal{L}*{\text{dyn}}
+
\epsilon*{\text{spa}}\mathcal{L}*{\text{spa}}
+
\epsilon*{\text{sub}}\mathcal{L}_{\text{sub}},
]

其中：

* (\mathcal{L}_{\text{cont}}) 为主损失；
* (\mathcal{L}_{\text{fast}}) 继续保留，用于维持 coarse action 建模；
* (\mathcal{L}*{\text{dyn}},\mathcal{L}*{\text{spa}},\mathcal{L}_{\text{sub}}) 仅保留很小权重，用于防止原有 structured slots 退化。

这意味着 Stage III 不是完全抛弃前两个阶段，而是在 coarse discrete action 的基础上继续细化控制。

---

## 3.8 Training Summary

将三阶段合并来看，训练策略可以概括为：

### Stage I

先只训练：

* specialized FFN
* slot 输入投影
* slot adapter

在 frozen backbone 与 frozen teachers 下，建立稳定的 structured latent pathway。

### Stage II

解冻 backbone 的 shared computation（Self-Attention + Base-FFN），同时引入 Action-FFN 和 FAST action。
此时通过 scaffold annealing 与 action constraint，构建
[
\text{base context} \rightarrow \text{slots} \rightarrow \text{coarse action}
]
的桥梁。

### Stage III

保持 FAST action 训练，并进一步引入 continuous action flow matching。
DiT-based action header 成为训练重点，而 VLM 侧仅进行低学习率协同更新，从而在不破坏前期 slot 结构的前提下完成 coarse-to-fine 控制。

---

# 4 Implementation-Oriented Notes

为了方便实现，我们建议将 token group、FFN routing 与 mask builder 明确模块化。

## 4.1 Token groups

在代码中显式维护 token type id：

* `IMAGE`
* `INSTRUCTION`
* `SUBTASK`
* `DYNAMIC`
* `SPATIAL`
* `ACTION`

## 4.2 Full-layer FFN routing

每一层 transformer block 中：

* 先执行 shared self-attention；
* 再执行 `Base-FFN`；
* 最后根据 token type 叠加对应 specialized FFN residual。

## 4.3 Stage-specific attention mask builder

建议单独实现一个 stage-aware group mask 构造函数：

* `build_stage1_mask()`
* `build_stage2_mask()`
* `build_stage3_mask()`

其中：

* Stage I：action 全部 excluded；
* Stage II：action 不可见 image / instruction；
* Stage III：action 恢复标准因果可见性。

## 4.4 Input masking scheduler

建议将 `intra-mask` 与 `inter-mask` 独立封装为输入端 curriculum scheduler，用于 Stage II。

---