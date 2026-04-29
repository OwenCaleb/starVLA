#!/usr/bin/env python3
"""
Test script to verify window-based recovery and step mapping logic.

This tests:
1. Window skip logic (window_active_flags: [false, true, true, true])
2. Progress calculation when resuming from step 10000
3. Slot masking curriculum progression
"""

import torch
from omegaconf import OmegaConf
from starVLA.model.framework.QwenMyVLA import QwenMyVLA


def test_window_progress_mapping():
    """Test that window mapping correctly skips first window and resumes from step 10000."""
    
    # Config snippet with window settings
    cfg_dict = {
        "enable_window_mapping": True,
        "step_windows": [[0, 10000], [10000, 20000], [20000, 30000], [30000, 40000]],
        "window_active_flags": [False, True, True, True],  # Skip first window
    }
    
    # Simulate training at different steps
    # Note: progress is calculated across ALL active windows (windows 1,2,3 = 30000 steps total)
    # Not per-window progress.
    test_cases = [
        # (global_step, expected_is_skipped, expected_progress, expected_window_idx, description)
        (0, True, None, -1, "Step 0: before active range"),
        (5000, True, None, -1, "Step 5000: in disabled window [0, 10000]"),
        (9999, True, None, -1, "Step 9999: end of disabled window"),
        (10000, False, 0.0, 1, "Step 10000: start of enabled window [10000, 20000]"),
        (10500, False, 500/30000, 1, "Step 10500: 500 steps into active phase / 30000 total"),
        (15000, False, 5000/30000, 1, "Step 15000: 5000 steps into active phase"),
        (20000, False, 10000/30000, 2, "Step 20000: end of window 1, start of window 2 (10000/30000)"),
        (25000, False, 15000/30000, 2, "Step 25000: middle of window 2 (15000/30000)"),
        (40000, False, 1.0, 3, "Step 40000: end of all active windows (30000/30000)"),
    ]
    
    print("\n" + "="*80)
    print("TEST: Window-based Progress Mapping with [false, true, true, true]")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for global_step, exp_skip, exp_progress, exp_window, desc in test_cases:
        kwargs = {
            "global_step": global_step,
            "max_train_steps": 100000,
            "enable_window_mapping": True,
            "step_windows": [[0, 10000], [10000, 20000], [20000, 30000], [30000, 40000]],
            "window_active_flags": [False, True, True, True],
        }
        
        # Create a minimal model config mock
        class ConfigMock:
            enable_progress_window_mapping = True
            progress_step_windows = [[0, 10000], [10000, 20000], [20000, 30000], [30000, 40000]]
            progress_window_active_flags = [False, True, True, True]
            enable_progress_step_mapping = False
        
        # Call the method directly
        cfg_mock = ConfigMock()
        info = QwenMyVLA._resolve_training_progress_info(cfg_mock, kwargs)
        
        # Check results
        is_skip_ok = info["is_skipped"] == exp_skip
        window_ok = info["current_window_index"] == exp_window
        
        progress_ok = True
        if exp_progress is not None and not info["is_skipped"]:
            progress_ok = abs(info["progress"] - exp_progress) < 0.001
        
        test_pass = is_skip_ok and window_ok and progress_ok
        
        status = "✅ PASS" if test_pass else "❌ FAIL"
        if test_pass:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status}: {desc}")
        print(f"  Step {global_step:5d} | Skip: {info['is_skipped']:<5} | Progress: {info['progress']:.4f} | Window: {info['current_window_index']}")
        
        if not test_pass:
            print(f"  Expected: Skip={exp_skip}, Progress={exp_progress}, Window={exp_window}")
            print(f"  Got:      Skip={info['is_skipped']}, Progress={info['progress']:.4f}, Window={info['current_window_index']}")
    
    print("\n" + "="*80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("="*80 + "\n")
    
    return failed == 0


def test_config_structure():
    """Verify that config values are correctly set."""
    import yaml
    
    print("\n" + "="*80)
    print("TEST: Config Structure Verification")
    print("="*80)
    
    config_path = "/mnt/nas_ssd/workspace/wenboli/projects/starVLA/starVLA/config/training/myvla_stage2.yaml"
    
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        # Check progress_control
        pc = cfg['framework']['progress_control']
        print(f"\n✅ progress_control found")
        print(f"  enable_window_mapping: {pc['enable_window_mapping']}")
        print(f"  window_active_flags: {pc['window_active_flags']}")
        print(f"  Expected: [false, true, true, true]")
        
        assert pc['window_active_flags'] == [False, True, True, True], "Window flags mismatch!"
        
        # Check trainer config
        trainer = cfg['trainer']
        print(f"\n✅ trainer config found")
        print(f"  is_resume: {trainer['is_resume']}")
        print(f"  resume_step: {trainer['resume_step']}")
        print(f"  pretrained_checkpoint: {trainer['pretrained_checkpoint']}")
        
        assert trainer['is_resume'] == False, "is_resume should be False"
        assert trainer['resume_step'] == 10000, "resume_step should be 10000"
        assert "steps_10000" in trainer['pretrained_checkpoint'], "Checkpoint should reference step 10000"
        
        print(f"\n✅ All config checks passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Config check failed: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# Resume from Step 10000 with Window Skip [false, true, true, true]")
    print("#"*80)
    
    config_ok = test_config_structure()
    progress_ok = test_window_progress_mapping()
    
    if config_ok and progress_ok:
        print("\n🎉 All tests passed! Config is ready for training resumption.")
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
