#!/usr/bin/env python3
"""
Show a complete board from the generated stepwise_training_data.jsonl
"""
import sys
import os
import json

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from config import EnvConfig
from env_core import action_to_bid


def main():
    print("=" * 80)
    print("Show Complete Board from Data")
    print("=" * 80)
    print()
    
    # 加载数据
    data_file = "data/stepwise_training_data.jsonl"
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        sys.exit(1)
    
    boards = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                boards.append(json.loads(line))
    
    print(f"Loaded {len(boards)} boards from {data_file}")
    if len(boards) == 0:
        print("No boards found!")
        sys.exit(1)
    
    # 选择第一个 board
    board = boards[0]
    print()
    print("=" * 80)
    print(f"BOARD {board.get('board_idx', 1)}")
    print("=" * 80)
    print()
    print(f"  - Dealer: {board.get('dealer', 'Unknown')}")
    print(f"  - Vulnerability: {board.get('vul', 'Unknown')}")
    print(f"  - Total Steps: {len(board.get('steps', []))}")
    print()
    
    # --- 分阶段展示 ---
    steps = board.get('steps', [])
    
    # Bidding phase
    bidding_steps = [s for s in steps if s['phase'] == 'bidding']
    if len(bidding_steps) > 0:
        print("-" * 80)
        print("PHASE 1: BIDDING")
        print("-" * 80)
        print()
        for i, step in enumerate(bidding_steps):
            act = step['optimal_act']
            act_str, _, _ = action_to_bid(act)
            print(f"  {i + 1}. {act_str}")
        print()
    
    # Play phase
    play_steps = [s for s in steps if s['phase'] == 'play']
    if len(play_steps) > 0:
        print("-" * 80)
        print("PHASE 2: PLAY")
        print("-" * 80)
        print()
        suits = ['C', 'D', 'H', 'S']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        trick_no = 0
        current_trick = []
        for i, step in enumerate(play_steps):
            act = step['optimal_act']
            s_idx = act // 13
            r_idx = act % 13
            card_str = f"{ranks[r_idx]}{suits[s_idx]}"
            current_trick.append(card_str)
            print(f"  {i + 1}. {card_str}")
            
            if len(current_trick) == 4:
                trick_no += 1
                print(f"    ✓ Trick {trick_no} Complete: {' '.join(current_trick)}")
                current_trick = []
        print()
    
    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
