#!/usr/bin/env python3
import sys
import os
import numpy as np
from collections import Counter

# 导入 config 模块
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from config import EnvConfig, OFS_MATCH, OFS_CONTRACT, OFS_HANDS, OFS_FIRST, OFS_AUCTION, OFS_LEAD, OFS_PLAY

# 设置 numpy 打印选项
np.set_printoptions(threshold=np.inf, linewidth=np.inf)

# 定义辅助函数
SUITS = ['S', 'H', 'D', 'C']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
PLAYERS = ['North', 'East', 'South', 'West']
VUL_NAMES = ['None', 'NS', 'EW', 'Both']
BID_DENOMS = ['C', 'D', 'H', 'S', 'NT']
BID_LEVELS = list(range(1, 8))
BID_ACTIONS = []
for level in BID_LEVELS:
    for denom in BID_DENOMS:
        BID_ACTIONS.append(f"{level}{denom}")
BID_ACTIONS.extend(['Pass', 'Double', 'Redouble'])


def decode_obs(obs, phase):
    """详细解析观测值"""
    obs = np.array(obs)
    result = {}
    
    # --- MATCH META ---
    result['vul'] = int(obs[OFS_MATCH]) if len(obs) > OFS_MATCH else 0
    result['vul_name'] = VUL_NAMES[result['vul']]
    
    # --- CONTRACT ---
    contract = {}
    if len(obs) > OFS_CONTRACT:
        # Level (OFS_CONTRACT 0-6)
        level = None
        for i in range(7):
            if obs[OFS_CONTRACT + i] > 0.5:
                level = i + 1
        contract['level'] = level
        
        # Denom (OFS_CONTRACT 7-11: S H D C NT)
        denom = None
        denom_idx = None
        for i in range(5):
            if obs[OFS_CONTRACT + 7 + i] > 0.5:
                denom = ['S', 'H', 'D', 'C', 'NT'][i]
                denom_idx = i
        contract['denom'] = denom
        contract['denom_idx'] = denom_idx
        
        # Declarer (OFS_CONTRACT 12-15: 0-3: N E S W)
        declarer = None
        for i in range(4):
            if obs[OFS_CONTRACT + 12 + i] > 0.5:
                declarer = PLAYERS[i]
        contract['declarer'] = declarer
        
        # Doubled (OFS_CONTRACT 16)
        contract['doubled'] = int(obs[OFS_CONTRACT + 16]) if len(obs) > OFS_CONTRACT + 16 else 0
    
    result['contract'] = contract
    
    # --- HANDS ---
    hands = []
    for pl_i in range(4):
        hand_offset = OFS_HANDS + pl_i * 52
        player_cards = []
        for s in range(4):
            for r in range(13):
                idx = hand_offset + s * 13 + r
                if len(obs) > idx and obs[idx] > 0.5:
                    player_cards.append(f"{RANKS[r]}{SUITS[s]}")
        hands.append({
            'player': PLAYERS[pl_i],
            'cards': player_cards,
            'count': len(player_cards)
        })
    result['hands'] = hands
    
    # --- CURRENT PLAYER ---
    current_player = None
    if len(obs) > OFS_FIRST:
        current_player = PLAYERS[int(obs[OFS_FIRST])] if int(obs[OFS_FIRST]) < 4 else None
    result['current_player'] = current_player
    
    # --- AUCTION HISTORY ---
    auction = []
    for i in range(199):
        idx = OFS_AUCTION + i
        if len(obs) > idx and obs[idx] > 0.0001:
            act = int(obs[idx])
            if 0 <= act < len(BID_ACTIONS):
                auction.append(BID_ACTIONS[act])
            else:
                auction.append(f"?{act}")
    result['auction'] = auction[:len(auction)]
    
    # --- LEAD (current trick) ---
    lead_cards = []
    for s in range(4):
        for r in range(13):
            idx = OFS_LEAD + s * 13 + r
            if len(obs) > idx and obs[idx] > 0.5:
                lead_cards.append(f"{RANKS[r]}{SUITS[s]}")
    result['lead'] = lead_cards
    
    # --- PLAY HISTORY ---
    play = []
    for i in range(53):
        idx = OFS_PLAY + i
        if len(obs) > idx and obs[idx] > 0.0001:
            act = int(obs[idx])
            if 0 <= act < 52:
                s_idx = act // 13
                r_idx = act % 13
                card = f"{RANKS[r_idx]}{SUITS[s_idx]}"
                play.append(card)
            else:
                play.append(f"?{act}")
    result['play_history'] = play
    
    return result


def main():
    print("=" * 80)
    print("Detailed OBS Inspection")
    print("=" * 80)
    print()
    
    # 加载数据
    npz_file = "data/stepwise_training_data.npz"
    if not os.path.exists(npz_file):
        print(f"Error: {npz_file} not found! Please generate data first!")
        sys.exit(1)
    
    data = np.load(npz_file)
    obs_list = data['obs']
    phase_list = data['phase']
    optimal_act_list = data['optimal_act']
    
    print(f"NPZ file loaded!")
    print(f"  - Obs shape: {obs_list.shape}")
    print(f"  - Phase shape: {phase_list.shape}")
    print(f"  - Optimal act shape: {optimal_act_list.shape}")
    print()
    
    # 阶段统计
    phase_counts = Counter(phase_list)
    print("Phase distribution:")
    print(f"  - Bidding (0): {phase_counts[0]}")
    print(f"  - Play (1): {phase_counts[1]}")
    print()
    
    # 最优动作统计
    print("Optimal action stats:")
    print(f"  - Min: {np.min(optimal_act_list)}")
    print(f"  - Max: {np.max(optimal_act_list)}")
    print(f"  - Unique: {len(np.unique(optimal_act_list))}")
    print()
    
    # 详细展示前 5 个样本
    for i in range(min(5, len(obs_list))):
        print("=" * 80)
        phase_name = "Bidding" if phase_list[i] == 0 else "Play"
        print(f"Sample {i} - Phase: {phase_name}")
        optimal = optimal_act_list[i]
        
        if phase_list[i] == 0:
            optimal_str = BID_ACTIONS[optimal] if 0 <= optimal < len(BID_ACTIONS) else f"?{optimal}"
        else:
            if 0 <= optimal < 52:
                s_idx = optimal // 13
                r_idx = optimal % 13
                optimal_str = f"{RANKS[r_idx]}{SUITS[s_idx]}"
            else:
                optimal_str = f"?{optimal}"
        
        print(f"Optimal Action: {optimal_str}")
        
        # 详细解码观测值
        decoded = decode_obs(obs_list[i], phase_list[i])
        
        # 打印各部分
        print()
        print(f"  - Vulnerability: {decoded['vul_name']}")
        print()
        
        if decoded['contract']['level'] is not None:
            print("  - Contract:")
            print(f"    - Level: {decoded['contract']['level']}")
            print(f"    - Denom: {decoded['contract']['denom']}")
            print(f"    - Declarer: {decoded['contract']['declarer']}")
            print(f"    - Doubled: {decoded['contract']['doubled']}")
            print()
        
        print("  - Hands (Visible only):")
        for hand in decoded['hands']:
            if hand['count'] > 0:
                print(f"    {hand['player']}: {' '.join(hand['cards'])}")
        print()
        
        if decoded['current_player']:
            print(f"  - Current Player: {decoded['current_player']}")
            print()
        
        if len(decoded['auction']) > 0:
            print(f"  - Auction History (len: {len(decoded['auction'])}):")
            print(f"    {' → '.join(decoded['auction'])}")
            print()
        
        if len(decoded['lead']) > 0:
            print(f"  - Current Trick Cards: {decoded['lead']}")
            print()
        
        if len(decoded['play_history']) > 0:
            print(f"  - Play History (len: {len(decoded['play_history'])}):")
            print(f"    {' → '.join(decoded['play_history'])}")
            print()
    
    print("=" * 80)
    print("Done!")


if __name__ == '__main__':
    main()
