#!/usr/bin/env python3
"""
Verify the parameter counts for frozen modules in stage2.
"""

import torch
import torch.nn as nn
from omegaconf import OmegaConf

# Load the config
config_path = "starVLA/config/training/myvla_stage2.yaml"
cfg = OmegaConf.load(config_path)

print("="*80)
print("Stage2 Frozen Module Verification")
print("="*80)
print()

# Key config values
hidden_size = cfg.framework.qwenvl.vl_hidden_dim  # 2560
action_dim = cfg.framework.action_model.action_dim  # 7
num_actions_chunk = cfg.framework.action_model.num_actions_chunk  # 8
freeze_modules = cfg.trainer.freeze_modules

print(f"Config Parameters:")
print(f"  hidden_size: {hidden_size}")
print(f"  action_dim: {action_dim}")
print(f"  num_actions_chunk: {num_actions_chunk}")
print()

print(f"Freeze modules spec: {freeze_modules}")
print()

# Simulate the modules
print("Simulated Module Parameter Counts:")
print("-" * 80)

# action_predictor
print("\n1️⃣  action_predictor:")
action_predictor = nn.Sequential(
    nn.LayerNorm(hidden_size * 3),
    nn.Linear(hidden_size * 3, hidden_size),
    nn.GELU(),
    nn.Linear(hidden_size, num_actions_chunk * action_dim),
)

action_predictor_params = sum(p.numel() for p in action_predictor.parameters())
print(f"  Total parameters: {action_predictor_params:,}")

for name, module in action_predictor.named_modules():
    if name and hasattr(module, 'parameters'):
        count = sum(p.numel() for p in module.parameters())
        if count > 0:
            print(f"    {name}: {count:,}")

print()
print("="*80)
print("Analysis:")
print("="*80)

if action_predictor_params > 1000000:
    print()
    print("⚠️  The parameter counts seem very large compared to the reported '6 params'.")
    print()
    print("Possible explanations:")
    print("1. The output '6 params' might refer to something else (e.g., a wrapper module)")
    print("2. There might be a custom implementation or override")
    print("3. The modules might be in a different structure than expected")
    print()
    print("🔍 Recommendation: Run actual model initialization and check frozen params")
else:
    print("✅ Parameter counts match expected values")

print()
print("="*80)
print()
