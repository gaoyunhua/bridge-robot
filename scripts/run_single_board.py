#!/usr/bin/env python3
"""
Run a single complete bridge board and show every step in detail
"""
import sys
import os
import torch
from typing import List, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from config import EnvConfig
from env_core import BridgeEnv, PLAYER_ORDER, action_to_bid
from stepwise_utils import DDSTeacher, process_bidding_step, process_play_step
from model_transformer import BridgeTransformerV2


def format_hand(hand):
    """格式化手牌显示，按花色排序"""
    suits = ['S', 'H', 'D', 'C']
    ranks = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
             'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
    sorted_cards = []
    for s in suits:
        suit_cards = []
        for c in hand:
            if len(c) == 2 and c[1] == s:
                suit_cards.append(c[0])
            elif len(c) == 3 and c[2] == s:  # like "10S"
                suit_cards.append(c[:2])
        # 按大小排序
        suit_cards.sort(key=lambda x: ranks[x[0]], reverse=True)
        if suit_cards:
            sorted_cards.append(f"{s}:{''.join(suit_cards)}")
    return " ".join(sorted_cards)


def main():
    print("=" * 80)
    print("Single Complete Bridge Board Playthrough")
    print("=" * 80)
    print()
    
    # 初始化环境、模型、教师
    env = BridgeEnv()
    model = BridgeTransformerV2()
    teacher = DDSTeacher()
    
    # 重置环境
    print("Step 0: Reset Environment")
    print("-" * 80)
    env.reset(board_num=1)
    
    # 获取 deal
    print()
    print("  - Deal:")
    players_str = ["North", "East", "South", "West"]
    for idx, player in enumerate(PLAYER_ORDER):
        hand = env._get_hand(player)
        # 转换 endplay.Hand -> list of strings
        hand_cards = []
        for c in hand:
            # c is Card, let's get string representation
            r = str(c.rank)[0] if str(c.rank)[0] != "1" else "T"
            s = c.suit.abbr
            hand_cards.append(f"{r}{s}")
        # 格式化手牌
        formatted = format_hand(hand_cards)
        print(f"      {players_str[idx]}: {formatted}")
    
    print()
    vul_str = ["None", "NS", "EW", "Both"][env.vul.value]
    dealer_str = players_str[PLAYER_ORDER.index(env.dealer)]
    print(f"  - Vulnerability: {vul_str}")
    print(f"  - Dealer: {dealer_str}")
    print()
    
    # 开始游戏流程
    step = 0
    obs_orig = env._encode_obs()
    deal = env.deal
    vul = env.vul
    dealer = env.dealer
    
    # --- Bidding Phase ---
    print("=" * 80)
    print("PHASE 1: BIDDING")
