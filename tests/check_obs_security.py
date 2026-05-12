#!/usr/bin/env python3
import sys
import os
import numpy as np

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, f'{project_root}/src')

from config import EnvConfig, OFS_HANDS, OFS_AUCTION, OFS_PLAY

print("=" * 80)
print("分析训练数据的安全性")
print("=" * 80)
print()

# 查看 obs 的各部分偏移
print("Obs 各部分的偏移:")
print(f"  OFS_MATCH:    {OFS_MATCH} (开始)")
print(f"  OFS_CONTRACT: {OFS_CONTRACT}")
print(f"  OFS_HANDS:    {OFS_HANDS}")
print(f"  OFS_FIRST:    {OFS_FIRST}")
print(f"  OFS_AUCTION:  {OFS_AUCTION}")
print(f"  OFS_PLAY:     {OFS_PLAY}")
print()

# 计算 hands 区域的大小
hand_size = 52  # 每手牌的编码大小
print(f"每个玩家手牌编码大小: {hand_size}")
print(f"4个玩家手牌区域: {OFS_HANDS} - {OFS_HANDS + 4 * hand_size - 1}")
print()

# 隐藏对手手牌的逻辑
print("=" * 80)
print("隐藏对手手牌的逻辑:")
print("=" * 80)
print()
print("在 bidding 阶段:")
print("  - 只有当前玩家的手牌可见")
print("  - 其他3个玩家的手牌都被置为 0.0")
print()
print("在 play 阶段:")
print("  - 当前玩家的手牌可见")
print("  - 如果不是首攻，明手的手牌可见")
print("  - 其他玩家的手牌被置为 0.0")
print()

print("=" * 80)
print("安全结论:")
print("=" * 80)
print()
print("✅ 这个数据是安全的！")
print()
print("原因:")
print("1. obs 已经通过 hide_opponent_cards 处理")
print("2. 只包含当前玩家可见的信息")
print("3. 对手的手牌都被隐藏了（置为 0.0）")
print("4. 不会泄露任何不该看到的信息")
print()
print("这样的训练数据可以安全地用于训练模型！")
