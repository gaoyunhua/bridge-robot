#!/usr/bin/env python3
"""
加载训练好的模型，生成一幅完整桥牌的打牌过程
入口文件：调用 src/stepwise_utils
"""

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import numpy as np
import torch
from endplay.types import Player
import config as _cfg
from config import EnvConfig, set_seed
from model_transformer import BridgeTransformerV2
from env_core import BridgeEnv, PLAYER_ORDER

from stepwise_utils import (
    hide_opponent_cards,
    format_hand,
    deal_log,
    NUM_BID, NUM_PLAY,
    BID_NAMES_CORRECT
)


def show_all_hands(deal):
    """显示四家牌"""
    from endplay.types import Player
    n = format_hand(deal, Player.north)
    e = format_hand(deal, Player.east)
    s = format_hand(deal, Player.south)
    w = format_hand(deal, Player.west)
    print(f'  N: {n}')
    print(f'  E: {e}')
    print(f'  S: {s}')
    print(f'  W: {w}')


def card_idx_to_str(idx):
    """出牌动作索引(0-51) → 牌面字符串"""
    suit = idx // 13
    rank = idx % 13
    return f'{_cfg.SUIT_SYMBOLS[suit]}{_cfg.RANK_NAMES[rank]}'


def bid_idx_to_str(idx):
    """叫牌动作索引(0-37) → 叫牌字符串"""
    return BID_NAMES_CORRECT[idx]


def action_to_bid_history(history):
    """将历史记录转为可读叫牌过程"""
    lines = []
    for turn, (player, action) in enumerate(history):
        if action == 35:
            lines.append(f'  {PLAYER_NAMES[player.value]}: Pass')
        elif action == 36:
            lines.append(f'  {PLAYER_NAMES[player.value]}: Double')
        elif action == 37:
            lines.append(f'  {PLAYER_NAMES[player.value]}: Redouble')
        else:
            lines.append(f'  {PLAYER_NAMES[player.value]}: {bid_idx_to_str(action)}')
    return '\n'.join(lines)


PLAYER_NAMES = ['N', 'E', 'S', 'W']


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    model_path = os.path.join(os.path.dirname(__file__), '..', 'checkpoints', 'stepwise_dds_model.pt')
    if not os.path.exists(model_path):
        print(f'未找到模型: {model_path}')
        print(f'请先运行: python scripts/stepwise_dds_train.py --num-boards 500 --epochs 2')
        return

    model = BridgeTransformerV2().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f'模型已加载: {model_path}')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'参数量: {total_params:,}\n')

    env = BridgeEnv()
    env.reset()
    deal = env.deal

    print('=' * 80)
    print('  初始牌局')
    print('=' * 80)
    show_all_hands(deal)
    print()

    while not env.done and env.phase == 'bidding':
        player = env._current_player()
        player_idx = PLAYER_ORDER.index(player)
        obs = env._encode_obs()
        obs_hidden = hide_opponent_cards(obs, player_idx, 'bidding')
        legal = env.legal_mask()

        obs_t = torch.from_numpy(obs_hidden).float().unsqueeze(0).to(device)
        with torch.no_grad():
            bid_logits, _ = model(obs_t)
            masked = bid_logits.clone()
            legal_bool = torch.from_numpy(legal[:NUM_BID]).bool().to(device)
            masked[0, ~legal_bool] = float('-inf')
            pred = masked.argmax(dim=-1).item()

        env.step(pred)
        action_str = bid_idx_to_str(pred)
        player_name = PLAYER_NAMES[player.value]

        info = f'  >> {player_name} 叫: {action_str}'
        if pred < 35:
            level = pred // 5 + 1
            denom = ['♣', '♦', '♥', '♠', 'NT'][pred % 5]
            info += f'  ({level}{denom})'

        print(info)

        if env.done:
            break

    if env.bidding.contract_level == 0:
        print('  ⚠  全部 Pass，无合约')
        return
    else:
        contract_denom = env.bidding.contract_denom
        contract_level = env.bidding.contract_level
        declarer = env.bidding.declarer
        print(f'\n  📋  最终合约: {contract_level}{contract_denom.abbr}')
        print(f'  👤  庄家: {PLAYER_NAMES[declarer.value]}')
        print(f'  🃏  明手: {PLAYER_NAMES[(declarer.value + 2) % 4]}')
        print(f'  ⚡  首攻: {PLAYER_NAMES[(declarer.value + 1) % 4]}')
        print()

    print('=' * 80)
    print('  出牌阶段')
    print('=' * 80)

    while not env.done and env.phase == 'play':
        assert env.play_state is not None
        player = env._current_player()
        player_idx = PLAYER_ORDER.index(player)
        declarer = env.bidding.declarer
        declarer_idx = PLAYER_ORDER.index(declarer)
        dummy_idx = (declarer_idx + 2) % 4
        is_first_trick = (env.play_state._trick_no == 0)
        trick_cards_played = len(env.play_state._current_trick_cards)

        obs = env._encode_obs()
        obs_hidden = hide_opponent_cards(
            obs, player_idx, 'play',
            declarer_idx, dummy_idx,
            is_first_trick, trick_cards_played
        )

        legal = env.legal_mask()
        obs_t = torch.from_numpy(obs_hidden).float().unsqueeze(0).to(device)
        with torch.no_grad():
            _, play_logits = model(obs_t)
            masked = play_logits.clone()
            legal_bool = torch.from_numpy(legal[:NUM_PLAY]).bool().to(device)
            masked[0, ~legal_bool] = float('-inf')
            pred = masked.argmax(dim=-1).item()

        card_str = card_idx_to_str(pred)
        player_name = PLAYER_NAMES[player.value]

        print(f'  墩 {env.play_state._trick_no + 1} | {player_name} 出: {card_str}')
        env.step(pred)

    ns_tricks = env.play_state._tricks_won_ns if env.play_state else 0
    ew_tricks = env.play_state._tricks_won_ew if env.play_state else 0

    declarer = env.bidding.declarer
    declarer_idx = PLAYER_ORDER.index(declarer)
    dec_side_name = 'NS' if declarer_idx % 2 == 0 else 'EW'
    dec_tricks = ns_tricks if declarer_idx % 2 == 0 else ew_tricks
    needed = env.bidding.contract_level + 6

    print(f'\n  最终结果')
    print(f'  NS={ns_tricks}  EW={ew_tricks}')
    if dec_tricks >= needed:
        print(f'  ✅ {dec_side_name} 定约完成! ({dec_tricks} vs {needed})')
    else:
        print(f'  ❌ {dec_side_name} 定约失败! ({dec_tricks} vs {needed})')
    print()


if __name__ == '__main__':
    main()
