#!/usr/bin/env python3
"""调试 Step 58 的剩余牌计算 - 使用 generate_training_data 生成的同一副牌"""
import sys
import os
import json
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from config import EnvConfig, OFS_HANDS, OFS_AUCTION, OFS_PLAY, OFS_LEAD
from endplay.types import Card, Denom, Rank


def decode_original_hand(obs):
    """从 OBS 解码原始手牌"""
    obs = np.array(obs)
    players = ['North', 'East', 'South', 'West']
    suits = ['S', 'H', 'D', 'C']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

    player_cards = {i: [] for i in range(4)}
    for pl_idx in range(4):
        hand_offset = OFS_HANDS + pl_idx * 52
        for s in range(4):
            for r in range(13):
                idx = hand_offset + s * 13 + r
                if len(obs) > idx and obs[idx] > 0.5:
                    player_cards[pl_idx].append(f"{ranks[r]}{suits[s]}")

    return player_cards


def decode_play_history(obs):
    """从 OBS 解码 play history"""
    obs = np.array(obs)
    play_history = []
    suits = ['C', 'D', 'H', 'S']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

    for i in range(52):
        idx = OFS_PLAY + i
        if len(obs) > idx and obs[idx] > 0.0001:
            act_idx = int(obs[idx])
            s_idx = act_idx // 13
            r_idx = act_idx % 13
            play_history.append(f"{ranks[r_idx]}{suits[s_idx]}")

    return play_history


def decode_optimal_act(act_idx):
    """解码出牌动作"""
    suits = ['C', 'D', 'H', 'S']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    s_idx = act_idx // 13
    r_idx = act_idx % 13
    return f"{ranks[r_idx]}{suits[s_idx]}"


def main():
    # 加载 generate_training_data.py 生成的同一副牌
    data_file = "data/stepwise_training_data.jsonl"
    with open(data_file, 'r') as f:
        steps = [json.loads(line) for line in f]

    # 获取 Step 58 的数据
    step58 = steps[58]

    print("=" * 60)
    print("Step 58 分析 (使用 generate_training_data 生成的同一副牌)")
    print("=" * 60)

    player_idx = step58['player_idx']
    print(f"player_idx: {player_idx} ({['North', 'East', 'South', 'West'][player_idx]})")

    optimal_act = step58['optimal_act']
    print(f"optimal_act: {optimal_act} ({decode_optimal_act(optimal_act)})")

    # 解码原始手牌
    original_cards = decode_original_hand(step58['obs'])
    print("\n原始手牌:")
    for pl_idx in range(4):
        print(f"  {['North', 'East', 'South', 'West'][pl_idx]}: {original_cards[pl_idx]}")

    # 解码 play_history
    play_history = decode_play_history(step58['obs'])
    print(f"\nplay_history 长度: {len(play_history)}")
    print(f"play_history: {play_history}")

    # 计算谁出了哪张牌
    print("\n谁出了哪张牌 (play_history[0] 是 Step 8 第一张牌，North 出):")
    for i, card in enumerate(play_history):
        step_num = 8 + i
        player_at_step = (8 + i) % 4
        player_name = ['North', 'East', 'South', 'West'][player_at_step]
        print(f"  play_history[{i}] = {card} -> Step {step_num} {player_name} 出")

    # 检查 2S 在哪
    if '2S' in play_history:
        idx = play_history.index('2S')
        step_num = 8 + idx
        player_at_step = (8 + idx) % 4
        player_name = ['North', 'East', 'South', 'West'][player_at_step]
        print(f"\n2S 在 play_history[{idx}]，Step {step_num} {player_name} 出")
    else:
        print("\n2S 不在 play_history 中（还未打出）")

    # 检查 West 原始手牌
    west_original = set(original_cards[3])
    print(f"\nWest 原始手牌: {west_original}")

    # 减去已打出的牌
    west_played = set()
    for i, card in enumerate(play_history):
        player_at_step = (8 + i) % 4
        if player_at_step == 3:  # West
            west_played.add(card)
    print(f"West 打出的牌: {west_played}")

    west_remaining = west_original - west_played
    print(f"West 剩余牌: {west_remaining}")

    # 检查 optimal_act 是否在剩余牌中
    optimal_act_str = decode_optimal_act(optimal_act)
    if optimal_act_str in west_remaining:
        print(f"\n✓ {optimal_act_str} 在 West 剩余牌中")
    else:
        print(f"\n✗ {optimal_act_str} 不在 West 剩余牌中！")


if __name__ == "__main__":
    main()
