# Fix: Weight Schedule Discontinuity on Resume

## Problem Summary
- **First run** (old config): No `progress_control` → defaults to `window_active_flags=[T,T,T,T]`
  - Step 10230: progress = 10230/40000 = 0.2557, w_slot_dynamic = 0.847 ✓
  
- **Restart** (new config): Has `window_active_flags=[F,T,T,T]`  
  - Step 10010: progress = (10010-10000)/30000 = 0.0003, w_slot_dynamic = 1.000 ✗
  - **Weights reset to schedule start, losing continuity**

## Root Cause
Configuration changed **BETWEEN** runs, but model checkpoint wasn't retrained with new config.

### What Happened
```yaml
# BEFORE (first run)
# ... no progress_control section ...
resume_step: null

# AFTER (restart)
progress_control:
  window_active_flags: [false, true, true, true]
resume_step: 10000
```

## Solution Options

### Option A: Revert to old config system temporarily ⭐ RECOMMENDED
**Idea**: Restore `window_active_flags=[True,True,True,True]` during restart phase to maintain progress continuity.

**Steps**:
1. Edit `myvla_stage2.yaml`:
   ```yaml
   progress_control:
     enable_window_mapping: true
     step_windows: [[0,10000], [10000,20000], [20000,30000], [30000,40000]]
     window_active_flags: [true, true, true, true]  # ← CHANGE: [F,T,T,T] → [T,T,T,T]
   
   trainer:
     resume_step: 10230  # ← CHANGE: 10000 → 10230 (actual reached step)
   ```

2. Resume training: it will continue from step 10230 with smooth progress
   - Step 10230: progress = 10230/40000 = 0.2557 ✓
   - Weights continue from w_slot_dynamic = 0.847

3. After reaching desired step (e.g., 20000), can transition back to windowed config

**Pros**: 
- ✅ Maintains exact continuity  
- ✅ No code changes needed
- ✅ Only config adjustments

**Cons**: 
- Requires manual config edits
- Must remember to switch back later

---

### Option B: Implement progress compensation logic (Complex)
Modify `QwenMyVLA._resolve_training_progress_info()` to detect config changes and auto-adjust progress scale.

**Idea**:
```python
# When window_active_flags changes from [T,T,T,T] to [F,T,T,T]:
# Old progress base: 40000
# New progress base: 30000
# Compensation: scale up completed_steps by ratio (40000/30000)

if checkpoint_was_created_with_all_windows and config_now_has_masked_windows:
    # Adjust completed_steps to maintain progress continuity
    effective_steps = 10230 * (30000/40000) = 7672
    but_actual_steps_trained = 10230
    # So apply progress correction...
```

**Cons**: 
- Complex logic, hard to debug
- Requires tracking previous config state
- Risk of introducing new bugs

---

### Option C: Start fresh (Simplest but wastes progress)
Delete checkpoint and retrain from scratch with correct config.

**Cons**: 
- ❌ Loses all progress from first 10230 steps
- Time-consuming

---

## RECOMMENDED FIX: Option A

Edit `/mnt/nas_ssd/workspace/wenboli/projects/starVLA/starVLA/config/training/myvla_stage2.yaml`:

```diff
  progress_control:
    enable_window_mapping: true
    step_windows:
      - [0, 10000]
      - [10000, 20000]
      - [20000, 30000]
      - [30000, 40000]
-   window_active_flags: [false, true, true, true]
+   window_active_flags: [true, true, true, true]

  trainer:
    stage_name: stage2
    epochs: 100
    max_train_steps: 100000
    num_warmup_steps: 1000
    save_interval: 5000
    eval_interval: 100
    learning_rate:
      base_learning_rate: 1.0e-04
      lr_scheduler: cosine
      weighted_lr: true
      weight_decay_wd_only: false
-   resume_step: 10000
+   resume_step: 10230
    is_resume: false
    pretrained_checkpoint: .../steps_10000_pytorch_model_flag.pt
```

---

## Verification After Fix

After restarting training with the fix:

```bash
# Launch training
python starVLA/training/train_starvla_myvla.py --config starVLA/config/training/myvla_stage2.yaml

# Expected logs:
# [INFO] Loaded pretrained checkpoint: ..., resuming from step 10230
# [INFO] Loaded progress_window_active_flags: [True, True, True, True]
# Step 10230: w_slot_dynamic = 0.847, training_progress = 0.2557  ✓ CONTINUOUS
# Step 10231: w_slot_dynamic ≈ 0.848, training_progress ≈ 0.2560  ✓ SMOOTH SLOPE
```

---

## Next Steps (Long-term fix)

Once training completes a stable phase (e.g., step 20000):

1. Save a checkpoint at step 20000
2. Create a "stage2_phase2.yaml" with the windowed config:
   ```yaml
   window_active_flags: [false, true, true, true]
   resume_step: 20000
   ```
3. Continue training from step 20000 with the windowed curriculum

This allows the windowed curriculum to take effect in later phases without breaking progress continuity.

---

## Why This Works

**Temporary all-windows config**:
- Progress: `step / 40000` (all steps counted equally)
- Step 10230 → progress = 0.2557
- Matches checkpoint's historical progress

**Eventually returns from step 10000 to window masking**:
- After reaching step 20000, transition to `window_active_flags=[F,T,T,T]`
- New checkpoint at step 20000 becomes baseline for windowed curriculum
- Progress then calculated as: (20000-20000)/(30000) = 0 (restarts curriculum, which is intended)

This approach:
1. ✅ Maintains continuity through step 20000
2. ✅ Allows windowed curriculum from step 20000 onwards  
3. ✅ No code changes needed
4. ✅ Fully reversible
