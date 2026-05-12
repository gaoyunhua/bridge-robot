#!/usr/bin/env python3
"""验证 env_core.py 的 Play History 和 Played Cards 是正确的！"""
import sys
import os
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from env_core import BridgeEnv
from endplay.types import Player

env = BridgeEnv()
env.reset(board_num=1)

print('=== 开始叫牌 ===')
while not env.done and env.phase == 'bidding':
    legal_mask = env.legal_mask()
    legal_actions = [i for i in range(38) if legal_mask[i]]
    act_idx = legal_actions[0]
    player = env._current_player()
    print(f'{player} 叫 {act_idx}')
    env.step(act_idx)

print(f'=== 叫牌结束，phase={env.phase} ===')

print('\n=== 开始出牌 ===')
i = 1
while not env.done and env.phase == 'play':
    legal_mask = env.legal_mask()
    legal_actions = [i for i in range(52) if legal_mask[i]]
    act_idx = legal_actions[0]
    player = env._play_state._current_player
    
    print(f'Step {i}')
    print(f'  出牌前: play_history={len(env._play_history)}, played_cards={len(env._play_state._played_cards)}, current_trick={len(env._play_state._current_trick_cards)}')
    print(f'  玩家: {player}, 动作: {act_idx}')
    
    obs_before = env._encode_obs()
    obs_after, reward, done, info = env.step(act_idx)
    
    print(f'  出牌后: play_history={len(env._play_history)}, played_cards={len(env._play_state._played_cards)}, current_trick={len(env._play_state._current_trick_cards)}, done={done}')
    
    i += 1

print(f'\n=== 最终 ===')
print(f'play_history: {len(env._play_history)} 条')
print(f'_play_state._played_cards: {len(env._play_state._played_cards)} 张')
print(f'_trick_no: {env._play_state._trick_no} 墩')
print(f'_tricks_won_ns: {env._play_state._tricks_won_ns}')
print(f'_tricks_won_ew: {env._play_state._tricks_won_ew}')

print(f'\n✅ 总计: {len(env._play_history)} 步出牌，正好是 52 张！' if len(env._play_history) == 52 else f'❌ 总计: {len(env._play_history)} 步出牌，应该是 52 张！')
