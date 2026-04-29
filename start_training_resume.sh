#!/bin/bash
# 从步骤10000恢复stage2训练的启动脚本

set -e

echo "=========================================="
echo "📊 starVLA Stage2 Resume Training Script"
echo "=========================================="
echo ""

# 设置环境
PROJECT_ROOT="/mnt/nas_ssd/workspace/wenboli/projects/starVLA"
cd "$PROJECT_ROOT"

echo "✓ Project root: $PROJECT_ROOT"
echo "✓ Current dir: $(pwd)"
echo ""

# 验证checkpoint文件存在
CHECKPOINT_FILE="results/Checkpoints/myvla_stage2/checkpoints/steps_10000_pytorch_model_flag.pt"
if [ ! -f "$CHECKPOINT_FILE" ]; then
    echo "❌ ERROR: Checkpoint not found at $CHECKPOINT_FILE"
    exit 1
fi
echo "✓ Checkpoint verified: $CHECKPOINT_FILE ($(du -h $CHECKPOINT_FILE | cut -f1))"
echo ""

# 检查配置
echo "📋 Configuration Settings:"
echo "  Window Active Flags: [false, true, true, true]"
echo "  Resume Step: 10000"
echo "  Max Train Steps: 100000"
echo "  Expected Output: steps_15000, steps_20000, ... steps_100000"
echo ""

# 启动训练
echo "🚀 Starting training..."
echo ""

python starVLA/training/train_starvla_myvla.py \
  --config_yaml starVLA/config/training/myvla_stage2.yaml \
  2>&1 | tee training_resume_10k.log

echo ""
echo "=========================================="
echo "✅ Training started successfully"
echo "📝 Log file: training_resume_10k.log"
echo "=========================================="
