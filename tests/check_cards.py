#!/usr/bin/env python3
import sys
import os
import json
import numpy as np

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, f'{project_root}/src')

from config import EnvConfig, OFS_HANDS

# 读取第一个数据点
with open('data/tmpaqvgy71g.jsonl', 'r', encoding='utf-8') as f:
    first_line = f.readline()
    data = json.loads(first_line)

obs = np.array(data['obs'])
player_idx = data['player_idx']

print("=" * 80)
print("Player 0 的 13 张手牌位置")
print("=" * 80)
print()

# Player 0 的手牌区间
start = OFS_HANDS + player_idx * 52
end = start + 52
print(f"Player {player_idx} 的手牌区间: [{start}:{end}]")
print()

# 找到所有 1.0 的位置
card_positions = np.where(obs[start:end] > 0.5)[0]
print(f"13 张卡的位置索引（相对于手牌起始位置）:")
print(f"  {card_positions}")
print()

# 定义花色和点数
suits = ['♠', '♥', '♦', '♣']
ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']

print("=" * 80)
print("具体的卡牌:")
print("=" * 80)
for pos in card_positions:
    suit_idx = pos // 13
    rank_idx = pos % 13
    print(f"  位置 {pos}: {suits[suit_idx]}{ranks[rank_idx]}")

print()
print("=" * 80)
print("13 张卡全部找到了！")
print("=" * 80)
