#!/usr/bin/env python3
"""
Diagnose the progress calculation mismatch between runs.
"""

print("="*80)
print("PROGRESS CALCULATION DIAGNOSIS")
print("="*80)
print()

# First run data
step_1st = 10230
progress_1st = 0.2557
weight_1st = 0.847

# Second run (after restart)
step_2nd = 10010
progress_2nd = 0.0003
weight_2nd = 1.000

print("📊 First Run (before restart):")
print(f"  Step: {step_1st}")
print(f"  Progress: {progress_1st}")
print(f"  Weight: {weight_1st}")
print()

print("📊 Second Run (after restart):")
print(f"  Step: {step_2nd}")
print(f"  Progress: {progress_2nd}")
print(f"  Weight: {weight_2nd}")
print()

print("🔍 Reverse-engineer the formula:")
print()

# Try to find what denominator was used in first run
print("First run: what denominator gives progress=0.2557?")
candidates = [
    (step_1st, 100000, "max_train_steps"),
    (step_1st, 40000, "[0, 40000]"),
    (step_1st, 30000, "active window total"),
    (step_1st - 10000, 30000, "step offset from 10000, active total"),
]

for numerator, denominator, desc in candidates:
    result = numerator / denominator
    match = "✓" if abs(result - progress_1st) < 0.0001 else " "
    print(f"  {match} {numerator}/{denominator} = {result:.4f} ({desc})")

print()
print("Second run: what denominator gives progress=0.0003?")
candidates_2 = [
    (step_2nd - 10000, 30000, "(step-10000)/30000 with window"),
    (step_2nd, 100000, "step/max_train_steps (global)"),
]

for numerator, denominator, desc in candidates_2:
    result = numerator / denominator
    match = "✓" if abs(result - progress_2nd) < 0.0001 else " "
    print(f"  {match} {numerator}/{denominator} = {result:.4f} ({desc})")

print()
print("="*80)
print("DIAGNOSIS:")
print("="*80)
print()

print("🔴 First run used GLOBAL progress: step / 40000")
print("   ↳ Denominator 40000 is range [0, 40000] (not windowed)")
print("   ↳ This suggests window_active_flags was NOT applied")
print()

print("🟢 Second run uses WINDOWED progress: (step-10000) / 30000")
print("   ↳ Denominator 30000 = 10000 + 10000 + 10000 (active windows)")
print("   ↳ This suggests window_active_flags WAS applied correctly")
print()

print("="*80)
print("ROOT CAUSE:")
print("="*80)
print()

print("The configuration changed between runs:")
print()
print("❌ First run:")
print("   • window_active_flags was NOT being applied")
print("   • Progress used global range [0, 40000]")
print("   • Weights scaled from 0% → 100% over [0, 40000]")
print()

print("✅ Second run (after restart):")
print("   • window_active_flags [false, true, true, true] IS applied")
print("   • Progress uses active window range [10000, 40000] = 30000 steps")
print("   • Weights scale from 0% → 100% over active windows only")
print()

print("="*80)
print("WHY THIS HAPPENED:")
print("="*80)
print()

print("Possible causes:")
print("1. First run config didn't include window_active_flags (YAML syntax error?)")
print("2. Config was updated AFTER first run started")
print("3. resume_step logic affects how window_active_flags are loaded")
print()

print("SOLUTION:")
print("Verify the current myvla_stage2.yaml has:")
print("  progress_control:")
print("    enable_window_mapping: true")
print("    window_active_flags: [false, true, true, true]")
print()

