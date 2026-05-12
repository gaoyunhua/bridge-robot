#!/usr/bin/env python3
"""Debug the level penalty for encouraging lower bids"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch.nn.functional as F

def debug_level_penalty(name, legal_actions, action_values):
    print("=" * 70)
    print(name)
    print("=" * 70)
    print(f"Legal actions: {legal_actions}")
    print(f"Action values: {action_values}")
    
    # Let's manually reproduce the logic
    values_tensor = torch.tensor(action_values)
    
    # Step 1: Amplify negative values
    negative_penalty_weight = 2.0
    amplified_values = torch.where(
        values_tensor < 0,
        values_tensor * negative_penalty_weight,
        values_tensor
    )
    print(f"\n1. After negative amplification: {amplified_values}")
    
    # Step 2: Apply level penalty
    positive_mask = (values_tensor >= 0)
    if positive_mask.any():
        level_penalty = torch.zeros(len(action_values))
        for i, act_idx in enumerate(legal_actions):
            if positive_mask[i]:
                level_penalty[i] = act_idx * 0.5
        amplified_values = amplified_values - level_penalty
        print(f"2. Level penalties applied: {level_penalty}")
        print(f"3. After level penalty: {amplified_values}")
    
    # Step 3: Compute exp values
    temperature = 1.0
    exp_values = torch.exp(amplified_values / temperature)
    print(f"\n4. Exp values (before normalization): {exp_values}")
    
    # Step 4: Normalize to get target distribution
    exp_sum = exp_values.sum()
    if exp_sum > 0:
        target_dist = exp_values / exp_sum
        print(f"5. Target distribution (should prefer lower positive bids):")
        for i, (act_idx, prob) in enumerate(zip(legal_actions, target_dist)):
            print(f"   Action {act_idx:2d}: {prob:.6f} (value: {action_values[i]:.1f})")
        
        # Find which action is most likely
        max_prob_idx = target_dist.argmax()
        print(f"\n   MOST PREFERRED: Action {legal_actions[max_prob_idx]} (prob: {target_dist[max_prob_idx]:.6f})")

# Test 1: All positive values
legal_actions = [0, 13, 31]  # Indices for 1C, 1S, 3NT
action_values_all_positive = [10.0, 12.0, 15.0]  # All positive - higher bids have higher values
debug_level_penalty("Test 1: All positive values", legal_actions, action_values_all_positive)
print()

# Test 2: Mixed positive and negative values
action_values_mixed = [10.0, -5.0, 8.0]
debug_level_penalty("Test 2: Mixed positive and negative values", legal_actions, action_values_mixed)
print()

# Test 3: Some positive, some negative, some lower bids better
action_values_mixed2 = [12.0, 10.0, -3.0, 15.0]
legal_actions2 = [0, 13, 20, 31]  # More bids
debug_level_penalty("Test 3: More mixed values", legal_actions2, action_values_mixed2)

print("\n" + "=" * 70)
print("SUMMARY: All positive bids should prefer the LOWEST ACTION INDEX!")
print("=" * 70)
