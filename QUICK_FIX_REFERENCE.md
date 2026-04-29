# ⚡ QUICK FIX REFERENCE CARD

## Problem
```
Before restart: Step 10230, w_slot_dynamic=0.847 ✓
After restart:  Step 10010, w_slot_dynamic=1.000 ✗ (reset!)
```

## Root Cause
config改变zwischen两次运行，导致进度计算方式改变：
- 第一次：`window_active_flags=[T,T,T,T]` (default)  → denominator=40000
- 重启：`window_active_flags=[F,T,T,T]` (explicit) → denominator=30000

## Solution Applied ✅

### File: `starVLA/config/training/myvla_stage2.yaml`

**Change 1** (Line 121):
```diff
- window_active_flags: [false, true, true, true]
+ window_active_flags: [true, true, true, true]
```

**Change 2** (Line 305):
```diff
- resume_step: 10000
+ resume_step: 10230
```

## Verification
```bash
# 检查修改
grep -n "window_active_flags:\|resume_step:" \
  starVLA/config/training/myvla_stage2.yaml

# 运行验证脚本
python verify_fix.py
```

## What to Expect on Next Run
```
[INFO] Resuming from step 10230
[INFO] window_active_flags: [true, true, true, true]
Step 10230: w_slot_dynamic ≈ 0.847 ✓ (continuous)
Step 10231: w_slot_dynamic ≈ 0.847 ✓ (smooth slope)
```

## For Phase 2 (After Step 20000)
```bash
# Create Phase 2 config with window-based curriculum
cp starVLA/config/training/myvla_stage2.yaml \
   starVLA/config/training/myvla_stage2_phase2.yaml

# Edit Phase 2: restore window_active_flags and adjust resume_step
# window_active_flags: [false, true, true, true]
# resume_step: 20000
```

---

## Key Files Changed
✏️ `starVLA/config/training/myvla_stage2.yaml` (2 changes, 2 lines)

## Documentation
📄 `WEIGHT_DISCONTINUITY_FIX_REPORT.md` - Complete analysis
📄 `fix_weight_discontinuity.md` - Detailed options
📄 `verify_fix.py` - Verification script
📄 `diagnose_progress_mismatch.py` - Diagnostic script

---

## Status
✅ FIXED | Ready for next training run

Last updated: 2025
