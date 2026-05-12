#!/usr/bin/env python3
"""
Check the stepwise training data
"""

import os
import sys
import json
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from config import EnvConfig


def check_data(data_file):
    print("="*80)
    print("Checking Training Data")
    print("="*80)

    if not os.path.exists(data_file):
        print(f"Error: File not found: {data_file}")
        return False

    boards = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                board = json.loads(line.strip())
                boards.append(board)
            except Exception as e:
                print(f"Warning: Skipping invalid line: {e}")

    print(f"\nTotal boards: {len(boards)}")

    if len(boards) == 0:
        print("Error: No valid data found")
        return False

    # Check each board
    total_steps = 0
    phase_counts = {}
    bid_actions = []
    play_actions = []
    obs_shapes = set()
    valid = True

    for board in boards:
        # Check board structure
        if not all(k in board for k in ['board_idx', 'vul', 'dealer', 'steps']):
            print(f"  ERROR: Board {board.get('board_idx', '?')} missing required keys")
            valid = False
            continue

        total_steps += len(board['steps'])

        # Check each step
        for step in board['steps']:
            if not all(k in step for k in ['phase', 'obs', 'optimal_act']):
                print(f"  ERROR: Step in board {board['board_idx']} missing keys")
                valid = False
                continue

            phase = step['phase']
            phase_counts[phase] = phase_counts.get(phase, 0) + 1

            obs_shapes.add(len(step['obs']))

            if len(step['obs']) != EnvConfig.obs_dim:
                print(f"  ERROR: Step in board {board['board_idx']} invalid obs shape: {len(step['obs'])}")
                valid = False

            if phase == 'bidding':
                bid_actions.append(step['optimal_act'])
            else:
                play_actions.append(step['optimal_act'])

    print(f"\nTotal steps: {total_steps}")
    print(f"\nPhase distribution:")
    for phase, count in phase_counts.items():
        print(f"  {phase}: {count} ({count/total_steps*100:.1f}%)")

    print(f"\nObservation shapes found: {sorted(obs_shapes)}")
    print(f"Expected obs dim from config: {EnvConfig.obs_dim}")

    if bid_actions:
        print(f"\nBidding actions:")
        print(f"  min: {min(bid_actions)}")
        print(f"  max: {max(bid_actions)}")
        print(f"  unique: {len(set(bid_actions))}")
        if min(bid_actions) < 0 or max(bid_actions) >= 38:
            print("  ERROR: Bidding actions out of range")
            valid = False

    if play_actions:
        print(f"\nPlay actions:")
        print(f"  min: {min(play_actions)}")
        print(f"  max: {max(play_actions)}")
        print(f"  unique: {len(set(play_actions))}")
        if min(play_actions) < 0 or max(play_actions) >= 52:
            print("  ERROR: Play actions out of range")
            valid = False

    # Check some random boards
    print("\n" + "="*80)
    print("Checking Random Boards")
    print("="*80)
    np.random.seed(42)
    sample_idxs = np.random.choice(len(boards), min(3, len(boards)), replace=False)
    for i in sample_idxs:
        board = boards[i]
        print(f"\nBoard {board['board_idx']}:")
        print(f"  dealer: {board['dealer']}")
        print(f"  vul: {board['vul']}")
        print(f"  steps: {len(board['steps'])}")
        if len(board['steps']) > 0:
            print(f"  First step:")
            first_step = board['steps'][0]
            print(f"    phase: {first_step['phase']}")
            print(f"    optimal_act: {first_step['optimal_act']}")

    print("\n" + "="*80)
    if valid:
        print("✅ Data is valid!")
    else:
        print("❌ Data has errors!")
    print("="*80)
    return valid


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Check Stepwise Training Data")
    parser.add_argument("--data-file", type=str, default="data/stepwise_training_data.jsonl")
    args = parser.parse_args()
    check_data(args.data_file)


if __name__ == "__main__":
    main()
