# 从步骤10000恢复训练配置文档

## 📋 修改总结

### 1. **配置文件修改** [`starVLA/config/training/myvla_stage2.yaml`]

#### 修改 A：Window Skip 配置（第117-122行）
```yaml
# 改动前:
window_active_flags: [true, true, true, true]   # 所有窗口启用

# 改动后:
window_active_flags: [false, true, true, true]  # 跳过第1个窗口[0-10000]，从窗口2[10000-20000]开始
```
**含义**: 跳过前10000步，从步骤10000继续训练

#### 修改 B：Checkpoint 恢复配置（第300-304行）
```yaml
# 改动前:
pretrained_checkpoint: .../myvla_stage1/checkpoints/steps_10000_pytorch_model.pt
is_resume: false
resume_epoch: null
resume_step: null

# 改动后:
pretrained_checkpoint: /mnt/nas_ssd/workspace/wenboli/projects/starVLA/results/Checkpoints/myvla_stage2/checkpoints/steps_10000_pytorch_model_flag.pt
is_resume: false
resume_epoch: null
resume_step: 10000
```
**含义**: 
- `pretrained_checkpoint` 指向步骤10000的checkpoint
- `is_resume: false` 禁用自动checkpoint查找（我们用explicit resume_step）
- `resume_step: 10000` 明确告诉trainer从步骤10000继续

---

### 2. **代码修改** [`starVLA/training/train_starvla_myvla.py` 第485-501行]

添加了 `resume_step` 支持逻辑：

```python
if pretrained_checkpoint:
    reload_modules = getattr(self.config.trainer, "reload_modules", None)
    self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)
    
    # Support override via resume_step if explicitly specified
    resume_step = getattr(self.config.trainer, "resume_step", None)
    if resume_step is not None:
        self.completed_steps = int(resume_step)
        logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, resuming from step {self.completed_steps}")
    else:
        self.completed_steps = 0
        logger.info(f"Loaded pretrained checkpoint: {pretrained_checkpoint}, steps: {self.completed_steps}")
    
    self.resume_from_checkpoint = pretrained_checkpoint
```

**功能**: 当配置中指定 `resume_step` 时，trainer会从该步骤继续计数，而不是从0开始。

---

## ✅ 验证结果

运行 [`test_resume_window_config.py`] 验证：

### 配置正确性 ✓
- `enable_window_mapping: True`
- `window_active_flags: [False, True, True, True]` 
- `is_resume: False`
- `resume_step: 10000`
- Checkpoint路径指向 `steps_10000_pytorch_model_flag.pt`

### Window Skip 逻辑 ✓
```
步骤范围              是否跳过  进度  说明
─────────────────────────────────────────────────
[0-10000)             ✓ 跳过   0%   禁用的窗口1
[10000-20000)         ✗ 激活   0-33%  激活的窗口2 ← 从这里开始
[20000-30000)         ✗ 激活   33-67% 激活的窗口3
[30000-40000)         ✗ 激活   67-100% 激活的窗口4
```

### Progress 计算 ✓
- 步骤10000: progress=0.0 (活动窗口2的开始)
- 步骤20000: progress=0.333 (3个激活窗口*10000步=30000总步, 10000/30000)
- 步骤30000: progress=0.667
- 步骤40000: progress=1.0 (完成)

---

## 🚀 启动训练

### 命令
```bash
cd /mnt/nas_ssd/workspace/wenboli/projects/starVLA

python starVLA/training/train_starvla_myvla.py \
  --config_yaml starVLA/config/training/myvla_stage2.yaml \
  --run_id myvla_stage2_resume_10k
```

### 预期输出
```
[RANK 0] Loaded pretrained checkpoint: ...steps_10000_pytorch_model_flag.pt, 
         resuming from step 10000
[RANK 0] Adjusting LR scheduler for resume from step 10000
[RANK 0] Step 10010 | training_progress=0.0003 | ...
[RANK 0] Step 10020 | training_progress=0.0007 | ...
...
[RANK 0] Step 15000 | checkpoint saved → steps_15000_pytorch_model_flag.pt
...
[RANK 0] Step 20000 | checkpoint saved → steps_20000_pytorch_model_flag.pt
```

---

## 📊 行为预期

### Slot Mask Curriculum
在步骤10000恢复后，继续适用slot masking curriculum：
- **进度范围**: 0% (步骤10000) → 100% (步骤40000)
- **Dynamic/Spatial/Subtask Mask**:
  - 开始: 5% inner mask (步骤10000)
  - 结束: 100% inner mask, 完全掩蔽 (步骤40000)

### Loss Weight Schedule
- **Slot Align Loss**: 1.0 → 0.4 (跨progress 0→1)
- **Fast Action Loss**: 0.6 → 1.0

### 学习率
- 继续使用cosine scheduler，从步骤10000的状态继续

---

## 📁 文件变更清单

```
✓ starVLA/config/training/myvla_stage2.yaml
  - progress_control.window_active_flags: [false, true, true, true]
  - trainer.pretrained_checkpoint: .../steps_10000_pytorch_model_flag.pt
  - trainer.resume_step: 10000

✓ starVLA/training/train_starvla_myvla.py
  - _setup_checkpoint_and_model() 方法: 添加resume_step支持

✓ test_resume_window_config.py (新文件)
  - 验证配置和progress计算逻辑
```

---

## ⚠️ 注意事项

1. **Checkpoint 命名**: 新保存的模型会变成 `steps_15000_pytorch_model_flag.pt` 等
2. **LR Scheduler**: 自动调整到步骤10000的状态（通过循环运行500次step）
3. **Wandb 日志**: 步数会从10000开始记录（非从0)
4. **最大步数**: 配置保持 `max_train_steps: 100000`，所以会从10k训练到100k

---

## 🧪 如何验证恢复工作正常

1. 检查日志输出:
   ```bash
   grep "resuming from step 10000" results/Checkpoints/myvla_stage2/*/training.log
   ```

2. 检查checkpoint保存:
   ```bash
   ls -lh results/Checkpoints/myvla_stage2/checkpoints/steps_*
   # 应该看到 steps_15000, steps_20000 等
   ```

3. 检查wandb metrics:
   - Steps 应该从 10000, 10010, 10020... (非0, 10, 20...)
   - Progress 应该 [false, true, true, true] 生效

---

**最后更新**: 2026-04-17 11:59 UTC | **状态**: ✅ 配置完成，可启动训练
