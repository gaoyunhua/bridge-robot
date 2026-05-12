#!/usr/bin/env python3
"""Example: How the code discourages high bids"""

import torch
import sys
sys.path.insert(0, 'src')

def demonstrate_level_penalty():
    print("=" * 70)
    print("EXAMPLE: How Level Penalty Discourages High Bids")
    print("=" * 70)
    
    # Simulate a bidding scenario
    print("\nScenario: Player has a strong hand and can bid 1C, 1S, or 3NT")
    print("All bids have POSITIVE values (all are winning bids)")
    print("But we want to encourage the LOWEST winning bid")
    
    # Define legal actions (bids) and their values
    # Action indices: 0=1C, 13=1S, 31=3NT
    legal_actions = [0, 13, 31]
    action_values = [10.0, 12.0, 15.0]  # 3NT has highest value but is highest bid
    
    print(f"\nOriginal action values:")
    for act, val in zip(legal_actions, action_values):
        print(f"  Action {act:2d}: Value = {val:.1f}")
    
    # Apply negative value amplification (for negative values only)
    print("\nStep 1: Negative value amplification (no change for positive values)")
    negative_penalty_weight = 2.0
    values_tensor = torch.tensor(action_values)
    amplified_values = torch.where(
        values_tensor < 0,
        values_tensor * negative_penalty_weight,
        values_tensor
    )
    print(f"  Amplified values: {amplified_values}")
    
    # Apply level penalty (this is where we discourage high bids!)
    print("\nStep 2: Apply LEVEL PENALTY to DISCOURAGE HIGH BIDS")
    print("  Penalty = action_index * 0.5")
    level_penalty = torch.zeros(len(action_values))
    for i, act_idx in enumerate(legal_actions):
        level_penalty[i] = act_idx * 0.5
        print(f"  Action {act_idx:2d}: Penalty = {level_penalty[i]:.2f}")
    
    # Subtract penalty from values
    final_values = amplified_values - level_penalty
    print(f"\n  Final values after penalty: {final_values}")
    
    # Compute target distribution
    print("\nStep 3: Compute target probabilities")
    temperature = 1.0
    exp_values = torch.exp(final_values / temperature)
    exp_sum = exp_values.sum()
    target_probs = exp_values / exp_sum
    
    print("\nResult: Target Probability Distribution")
    print("-" * 40)
    action_names = {0: "1C", 13: "1S", 31: "3NT"}
    for i, act in enumerate(legal_actions):
        print(f"  {action_names[act]:4s} (action {act}): {target_probs[i]:.4f} ({(target_probs[i]*100):.1f}%)")
    
    print("\n" + "=" * 70)
    print("CONCLUSION: Lower bids get higher probability!")
    print("1C: Highest probability even though it has the lowest raw value")
    print("3NT: Lowest probability even though it has the highest raw value")
    print("=" * 70)

if __name__ == "__main__":
    demonstrate_level_penalty()
