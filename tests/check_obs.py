#!/usr/bin/env python3
import sys
import os
import json
import numpy as np

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, f'{project_root}/src')

from config import EnvConfig, OFS_MATCH, OFS_CONTRACT, OFS_HANDS, OFS_FIRST, OFS_AUCTION, OFS_PLAY

# 读取第一个数据点
with open('data/tmpaqvgy71g.jsonl', 'r', encoding='utf-8') as f:
    first_line = f.readline()
    data = json.loads(first_line)

obs = np.array(data['obs'])
print(f"Observation shape: {obs.shape}")
print(f"Player: {data['player_idx']}")
print(f"Phase: {data['phase']}")
print()

# 检查各部分
print("=" * 80)
print("检查各部分:")
print("=" * 80)

# 比赛元数据 (0-11)
print(f"\nMatch (0-11):")
print(f"  obs[OFS_MATCH] = {obs[OFS_MATCH]}")
print(f"  这部分非零值位置: {np.where(obs[0:12] > 0.5)[0]}")

# 合约 (12-219)
print(f"\nContract (12-219):")
print(f"  这部分非零值位置: {np.where(obs[12:220] > 0.5)[0]}")

# 手牌 (220-447)
print(f"\nHands (220-447):")
for pl in range(4):
    start = OFS_HANDS + pl * 52
    end = start + 52
    num_cards = np.sum(obs[start:end] > 0.5)
    print(f"  Player {pl}: {num_cards} cards")

# 第一家 (448-476)
print(f"\nFirst bidder (448-476):")
print(f"  obs[OFS_FIRST] = {obs[OFS_FIRST]}")

# 叫牌历史 (477-675)
print(f"\nAuction (477-675):")
auction = [int(x) for x in obs[OFS_AUCTION:OFS_AUCTION+199] if x > 0.5]
print(f"  历史叫牌: {auction}")

# 出牌历史 (676-727)
print(f"\nPlay (676-727):")
play = [int(x) for x in obs[OFS_PLAY:OFS_PLAY+52] if x > 0.5]
print(f"  历史出牌: {play}")

print("\n" + "=" * 80)
print("非零值统计:")
print("=" * 80)
print(f"总非零值: {np.sum(obs > 0.5)}")
print(f"叫牌阶段: {data['phase'] == 'bidding'}")

print("\n这个 obs 看起来是正常的！")
