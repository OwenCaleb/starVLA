# starVLA MyVLA Integration Plan

## 已完成
- Dynamic teacher wrapper 已接入: UniVLA lam-stage-2 encoder
- Spatial teacher wrapper 已接入: Depth-Anything-V2 encoder tokens + decoder maps
- Subtask text encoder 已接入: Qwen3-Embedding (纯encoder与slot adapter解耦)
- 三路slot压缩与投影已接入: dynamic/spatial/subtask -> fixed slot tokens
- 占位符注入已接入: 通过embedding hook替换slot占位位置
- slot顺序已固定: [Qwen原生多模态序列 | subtask | dynamic | spatial]
- slot mask 已接入: inner mask + outside mask
- sequence layout 记录已接入: text/image/subtask/dynamic/spatial 的 start/end/length/positions
- typed attention mask 生成器已接入 + 可视化smoke
- 全层 typed residual experts 已接入: 复制Qwen gated MLP为 subtask/dynamic/spatial expert（层间不共享）
- slot readout 已改为纯MLP residual fuser（无SA/cross-attn）
- Stage I最小损失闭环已接入: 三路slot对齐损失 (MSE+cosine)
- 监督动作损失已接入: 基于 GT action 的 L1
- Stage3 action header 已接入: 新增 `MyVLA_AdapterHeader`，读取所有层hidden
- fast/continuous 两类损失已拆分: `fast_action_loss` 与 `continuous_fm_loss`
- 分支 detach 语义已接入: fast 与 continuous 可独立控制反传
- continuous_fm_loss 已迁移为真实 FM 目标（最小实现，velocity supervision）
- Stage II masked target prediction 闭环已接入（slot keep mask 与 align loss 绑定）
- predict_action 接口已对齐: 返回 normalized_actions
- 全层 typed attention mask 真正接入 backbone self-attention（新 BackHalfTypedSelfAttention 包装器，现用于全层）
- continuous_fm_loss 已完全迁移至真实 FM 目标（velocity supervision, configurable）
- 真实数据最小 smoke 已通过: LIBERO dataloader + 三路 teacher 编码器前向验证
- MyVLA 专用训练入口已提供: `starVLA/training/train_starvla_myvla.py`（默认读取 `myvla.yaml`）
- 三阶段训练入口的配置一致性检查已接入：会根据 `trainer.stage_name` 提醒关键开关是否匹配
- attention_mask 已支持按 stage_name 选择 stage_visibility，并在 stage YAML 中显式声明
- slot_mask curriculum 已接入：trainer 传训练进度，模型按进度插值 inner/outside mask 参数
- loss 权重调度已接入：slot_align/fast/continuous 支持按训练进度插值权重
- fast 分支已接入 action-aware attention：`<robot_action_*>` token 会进入 typed attention/mask 路径
- stage1/2/3 的 stage_visibility 已补充 action 行列（stage2 为 action-only 约束，stage3 放宽）
- 主 forward 已支持 action token group：可通过 `framework.action_tokens` 注入 action span，并进入 typed layout/token_type/attention
- Stage3 action query 位置已支持优先复用 action span（不足/超出时自动 pad/truncate 到 `action_query_num`）
- stage 配置校验已增强：新增 `framework.action_tokens` 开关/参数校验，含 stage3 `num_tokens == action_query_num` 一致性提醒
- 训练入口已支持 validate-only CLI：`--validate-stage-config` 与 `--validate-action-tokenizer`（校验后直接退出，不触发训练）
- P0 冻结策略已落地：`freeze_modules` 支持模块路径 + `param_regex:` 参数名正则，三阶段 YAML 已显式配置冻结策略
- 冻结生效检查已接入：freeze 过程打印每条规则命中参数数量，并新增 freeze policy smoke
- Stage II attention 可见性已按 method 语义对齐：slot 组互可见且可读 image+instruction，action 仍保持 action-only 读槽约束
- subtask adapter 主路径已接入：framework 优先消费 `subtask_slots`，并移除 adapter 侧 `no_grad` 包裹以恢复可训练性
- Stage III 连续分支已接入 coarse 条件：优先使用 action token hidden，回退到 light coarse action 投影
- Stage III 实现决策已收敛并文档对齐：当前默认采用 coarse-conditioned `VLA_AdapterHeader` 连续细化基线，DiT 路线作为后续增强
- 测试已接入并通过:
	- 注入smoke、mask可视化smoke、配置驱动smoke
	- typed residual FFN smoke
	- 文件内 train/predict contract smoke
	- loss-branch semantics smoke (fast vs continuous)
	- masked target prediction slot-align smoke
	- **all-layer typed attention apply smoke** (P0 #1)
	- **main forward action-group smoke**（验证 action span 进入主干 layout/token_type/attention）
	- **freeze policy smoke**（验证 stage1 关键冻结规则可执行且不误冻非 action expert）
	- **Stage2 attention visibility semantics smoke**（验证 stage2 连边符合 method）
	- **Subtask adapter primary-path smoke**（验证主路径优先读取 adapter slots）
	- **Stage3 coarse-condition smoke**（验证 stage3 coarse 条件优先/回退路径）
	- **real data forward smoke** (P1 #2 offline validation)
- **Stage II 无 continuous 损失**（2026-04-08 修复）：continuous_fm_loss.weight 设为 0.0, loss_weight_schedule 中 continuous_fm_start/end 均设为 0.0，对齐 method.md 定义 Stage II 仅含 slot_align + fast_action 损失
- **Stage II freeze_modules 参数正则化**（2026-04-08 修复）：将 `stage3_action_head` 更新为 `param_regex:^stage3_action_head.*` 以支持模块名带后缀或变化的场景
- **清理冗余 MyVLA_AdapterHeader**（2026-04-08 删除）：删除 `starVLA/model/modules/action_model/MyVLA_AdapterHeader.py`，统一使用 `VLA_AdapterHeader` 作为 Stage III continuous head 基础，避免混淆

## 未完成
- slot curriculum 缺少真实训练中的稳定性与收敛验证（当前仅 smoke/配置校验）
- 更完整的训练/评测实跑（真实数据与真实权重，不只是合同 smoke）
- action tokenizer 实机校验受阻：`Qwen3-VL-4B-Instruct` 已启动后台下载但 processor/tokenizer 文件尚未就绪，暂无法完成 tokenizer 与真实训练冒烟

## 后续执行计划
1. P2: 完整训练实跑验证
- 先跑 Stage I 的最小真实训练步数，确认梯度与 loss 曲线正常
- 再跑 Stage II 的 action grounding
- 最后再接 Stage III 连续 refinement

## 验收标准
- 可复现的配置驱动训练入口
- 每一步有最小 smoke test
- method.md 与代码接口一致
- 训练接口与推理接口对齐 starVLA 习惯：`forward -> action_loss`、`predict_action -> normalized_actions`
- 三阶段训练逻辑通过 YAML 配置可明确切换，且入口会做一致性校验
- Stage II 的 action-only 约束和 slot curriculum 可以被配置显式控制
- 三阶段冻结策略与 method 一致，且可通过可训练参数检查复现
- Stage II slot 可见性规则与 method 目标一致（不再仅 slot 互隔离）

## 最小训练冒烟命令（MyVLA）
- `python starVLA/training/train_starvla_myvla.py --config_yaml starVLA/config/training/myvla.yaml trainer.max_train_steps=1 datasets.vla_data.per_device_batch_size=1`
- `python starVLA/training/train_starvla_myvla.py --config_yaml starVLA/config/training/myvla_stage2.yaml trainer.max_train_steps=1 datasets.vla_data.per_device_batch_size=1`
- `python starVLA/training/train_starvla_myvla.py --config_yaml starVLA/config/training/myvla_stage3.yaml trainer.max_train_steps=1 datasets.vla_data.per_device_batch_size=1`
