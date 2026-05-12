
#!/usr/bin/env python3
import sys
import os
import json
import numpy as np

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, f'{project_root}/src')

from config import OFS_HANDS, HANDS_ENCODED_DIM

def check_data_point(data_point):
    phase = data_point["phase"]
    player_idx = data_point["player_idx"]
    obs = np.array(data_point["obs"])
    
    print(f"  Phase: {phase}")
    print(f"  Player: {player_idx}")
    
    # 检查每个玩家的手牌可见性
    visible_players = []
    for p in range(4):
        start = OFS_HANDS + p * HANDS_ENCODED_DIM
        end = start + HANDS_ENCODED_DIM
        hand = obs[start:end]
        num_cards = int(np.sum(hand > 0.5))
        if num_cards > 0:
            visible_players.append(p)
    
    print(f"  可见玩家: {visible_players}")
    
    if len(visible_players) > 2:
        print(f"  ❌ 安全问题：超过 2 个玩家可见！！")
        return False
    else:
        print(f"  ✓ 看起来安全")
        return True

if __name__ == "__main__":
    filename = f"{project_root}/data/stepwise_training_data.jsonl"
    print("=" * 80)
    print(f"检查训练数据安全性: {filename}")
    print("=" * 80)
    print()
    
    with open(filename, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            data_point = json.loads(line)
            phase = data_point.get("phase", "unknown")
            
            if idx < 20 or phase == "play":
                print(f"检查 Data Point {idx}:")
                is_safe = check_data_point(data_point)
                if not is_safe:
                    print()
                    exit()
                print()
            
            if idx >= 100:
                break
    
    print()
    print("=" * 80)
    print("检查完成！数据是安全的！")

