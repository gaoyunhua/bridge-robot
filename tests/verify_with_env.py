#!/usr/bin/env python3
"""直接用 env 验证 Step 58 的剩余手牌"""
import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from env_core import BridgeEnv, PLAYER_ORDER
from endplay.types import Player


def decode_card(act_idx):
    suits = ['C', 'D', 'H', 'S']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    s_idx = act_idx // 13
    r_idx = act_idx % 13
    return f"{ranks[r_idx]}{suits[s_idx]}"


def main():
    env = BridgeEnv()
    env.reset(board_num=1)

    print("=" * 60)
    print("验证牌局")
    print("=" * 60)

    # 叫牌
    while not env.done and env.phase == 'bidding':
        legal_mask = env.legal_mask()
        legal_actions = [i for i in range(38) if legal_mask[i]]
        env.step(legal_actions[0])

    print(f"叫牌结束，declarer={env.bidding.declarer}")

    # 出牌到 Step 58
    step_num = 0
    while not env.done and env.phase == 'play':
        legal_mask = env.legal_mask()
        legal_actions = [i for i in range(52) if legal_mask[i]]
        current_player = env._play_state._current_player

        # 记录状态
        remaining = {}
        for pl in PLAYER_ORDER:
            rem = env._get_remaining_cards(pl)
            remaining[pl] = rem

        print(f"\nStep {step_num + 8}: {current_player}")
        print(f"  Legal Actions: {[decode_card(a) for a in legal_actions]}")

        if step_num == 57:  # Step 58 (0-indexed = 57)
            print(f"  ** Step 58 验证 **")
            print(f"  当前玩家: {current_player}")
            print(f"  West 剩余: {remaining[Player.west]}")
            print(f"  North 剩余: {remaining[Player.north]}")
            print(f"  2S 在 West 剩余中: {'2S' in [str(c) for c in remaining[Player.west]]}")
            print(f"  2S 在 North 剩余中: {'2S' in [str(c) for c in remaining[Player.north]]}")

        # 执行
        env.step(legal_actions[0])
        step_num += 1

        if step_num >= 58:
            break

    print(f"\n最终状态: step_num={step_num}, done={env.done}")


if __name__ == "__main__":
    main()
