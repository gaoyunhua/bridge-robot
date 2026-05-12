#!/usr/bin/env python3
"""调试 Step 58 的剩余牌计算"""
import sys
import os
import json
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from config import EnvConfig, OFS_HANDS, OFS_AUCTION, OFS_PLAY, OFS_LEAD
from env_core import BridgeEnv, PLAYER_ORDER
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
    # 加载数据
    data_file = "data/stepwise_training_data.jsonl"
    with open(data_file, 'r') as f:
        steps = [json.loads(line) for line in f]

    # 获取 Step 58 的数据
    step58 = steps[58]
    step57 = steps[57]

    print("=" * 60)
    print("Step 58 分析")
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

    # 检查 play_history 中是否有 2S
    if '2S' in play_history:
        idx = play_history.index('2S')
        step_num = 8 + idx
        player_at_step = (8 + idx) % 4
        player_name = ['North', 'East', 'South', 'West'][player_at_step]
        print(f"\n2S 在 play_history[{idx}]，Step {step_num} {player_name} 出")
    else:
        print("\n2S 不在 play_history 中（还未打出）")

    # 用 env 验证
    print("\n" + "=" * 60)
    print("用 env 验证")
    print("=" * 60)

    env = BridgeEnv()
    env.reset(board_num=1)

    # 叫牌
    while not env.done and env.phase == 'bidding':
        legal_mask = env.legal_mask()
        legal_actions = [i for i in range(38) if legal_mask[i]]
        env.step(legal_actions[0])

    # 出牌到 Step 58
    for i in range(52):
        legal_mask = env.legal_mask()
        legal_actions = [i for i in range(52) if legal_mask[i]]
        obs, reward, done, info = env.step(legal_actions[0])

        if i == 50:  # Step 57 (0-indexed)
            print(f"\nStep 57 (出牌 {i+1}) 后的状态:")
            print(f"  当前玩家: {env._play_state._current_player}")
            print(f"  _played_cards: {len(env._play_state._played_cards)} 张")
            print(f"  _remaining_cards (North): {env._get_remaining_cards(env._play_state._current_player)}")

        if i == 51:  # Step 58 (0-indexed)
            print(f"\nStep 58 (出牌 {i+1}) 后的状态:")
            print(f"  当前玩家: {env._play_state._current_player}")
            print(f"  _played_cards: {len(env._play_state._played_cards)} 张")
            remaining = env._get_remaining_cards(env._play_state._current_player)
            print(f"  _remaining_cards (West): {remaining}")

            # 检查 West 的剩余牌
            west_idx = 3
            west_remaining = env._get_remaining_cards(PLAYER_ORDER[west_idx])
            print(f"  West 剩余牌: {west_remaining}")

            # 检查 Legal Actions
            legal_mask = env.legal_mask()
            legal_actions = [i for i in range(52) if legal_mask[i]]
            print(f"  Legal Actions: {[decode_optimal_act(a) for a in legal_actions]}")


if __name__ == "__main__":
    main()
