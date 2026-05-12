#!/usr/bin/env python3
"""
从 stepwise_training_data.jsonl 详细展示完整的单个 board (扁平格式)
使用 Endplay 进行牌的转换
"""
import sys
import os
import json
import argparse
import numpy as np
from collections import defaultdict

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from config import EnvConfig, OFS_HANDS, OFS_AUCTION, OFS_PLAY
from env_core import action_to_bid
from endplay.types import Card, Denom, Rank, Player

PLAYER_ORDER = [Player.north, Player.east, Player.south, Player.west]

_DENOM_TO_SUIT = {
    Denom.spades: 'S',
    Denom.hearts: 'H',
    Denom.diamonds: 'D',
    Denom.clubs: 'C',
    Denom.nt: 'NT'
}

_DENOM_TO_SYMBOL = {
    Denom.spades: '♠',
    Denom.hearts: '♥',
    Denom.diamonds: '♦',
    Denom.clubs: '♣',
    Denom.nt: 'NT'
}


def action_to_symbolic_bid(action_idx: int) -> str:
    """将动作索引转换为符号形式的叫牌字符串，如 '1♥'"""
    if action_idx == 35:  # PASS_ACTION
        return "Pass"
    if action_idx == 36:  # DOUBLE_ACTION
        return "Double"
    if action_idx == 37:  # REDOUBLE_ACTION
        return "Redouble"
    level = action_idx // 5 + 1
    denom = [Denom.clubs, Denom.diamonds, Denom.hearts, Denom.spades, Denom.nt][action_idx % 5]
    return f"{level}{_DENOM_TO_SYMBOL[denom]}"


def card_to_str(card: Card) -> str:
    """将 Endplay Card 转换为符号字符串，如 '♠K'"""
    suit_sym = _DENOM_TO_SYMBOL.get(card.suit, '?')
    rank_char = card.rank.abbr
    return f"{suit_sym}{rank_char}"


def act_idx_to_str(act_idx: int) -> str:
    """将动作索引转换为牌的字符串"""
    suits = [Denom.spades, Denom.hearts, Denom.diamonds, Denom.clubs]
    ranks = list(Rank)
    s_idx = act_idx // 13
    r_idx = act_idx % 13
    if s_idx < len(suits) and r_idx < len(ranks):
        card = Card(suit=suits[s_idx], rank=ranks[r_idx])
        return card_to_str(card)
    return f"?{act_idx}"


def decode_history_from_obs(obs, phase):
    """从 OBS 解码 bidding history 和 play history！"""
    obs = np.array(obs)
    bidding_history = []
    play_history = []

    # 解码 bidding history
    for i in range(199):
        idx = OFS_AUCTION + i
        if len(obs) > idx and obs[idx] > 0.0001:
            act_idx = int(obs[idx])
            act_str = action_to_symbolic_bid(act_idx)
            bidding_history.append(act_str)

    # 解码 play history
    for i in range(52):
        idx = OFS_PLAY + i
        if len(obs) > idx and obs[idx] > 0.0001:
            act_idx = int(obs[idx])
            act_str = act_idx_to_str(act_idx)
            play_history.append(act_str)

    return {
        'bidding_history': bidding_history,
        'play_history': play_history
    }


def decode_visible_hand(obs, player_idx):
    """从 OBS 解码当前玩家的可见手牌（使用 Endplay）"""
    obs = np.array(obs)

    hand_offset = OFS_HANDS + player_idx * 52
    suits = [Denom.spades, Denom.hearts, Denom.diamonds, Denom.clubs]
    ranks = list(Rank)

    player_cards = []
    for s_idx, suit in enumerate(suits):
        for r_idx, rank in enumerate(ranks):
            idx = hand_offset + s_idx * 13 + r_idx
            if len(obs) > idx and obs[idx] > 0.5:
                card = Card(suit=suit, rank=rank)
                player_cards.append(card_to_str(card))

    return player_cards


def parse_declarer_dummy(bidding_history):
    """从叫牌历史解析 declarer 和 dummy"""
    dealer_idx = 0  # North 开叫
    last_bid_player = PLAYER_ORDER[0]

    for i, bid in enumerate(bidding_history):
        if bid not in ['Pass', 'Double', 'Redouble'] and not any(sym in bid for sym in ['♠', '♥', '♦', '♣', 'NT']):
            # 字母形式的叫牌
            player_idx = (dealer_idx + i) % 4
            last_bid_player = PLAYER_ORDER[player_idx]
        elif bid not in ['Pass', 'Double', 'Redouble']:
            # 符号形式的叫牌
            player_idx = (dealer_idx + i) % 4
            last_bid_player = PLAYER_ORDER[player_idx]

    if last_bid_player == PLAYER_ORDER[0]:
        return PLAYER_ORDER[0].name.capitalize(), PLAYER_ORDER[2].name.capitalize()
    elif last_bid_player == PLAYER_ORDER[1]:
        return PLAYER_ORDER[1].name.capitalize(), PLAYER_ORDER[3].name.capitalize()
    elif last_bid_player == PLAYER_ORDER[2]:
        return PLAYER_ORDER[2].name.capitalize(), PLAYER_ORDER[0].name.capitalize()
    else:
        return PLAYER_ORDER[3].name.capitalize(), PLAYER_ORDER[1].name.capitalize()


def show_detailed_step(step_idx, step_data):
    """详细展示单个 step"""
    phase = step_data['phase']
    optimal_act = step_data['optimal_act']
    player_idx = step_data['player_idx']

    if phase == 'bidding':
        act_str = action_to_symbolic_bid(optimal_act)
    else:
        act_str = act_idx_to_str(optimal_act)

    player_str = PLAYER_ORDER[player_idx].name.capitalize()

    # 解码历史记录（用于解析，保持字母形式）
    # 先解码字母形式的历史记录用于解析
    obs = np.array(step_data['obs'])
    bidding_history_for_parse = []
    for i in range(199):
        idx = OFS_AUCTION + i
        if len(obs) > idx and obs[idx] > 0.0001:
            act_idx = int(obs[idx])
            act_str_alpha, _, _ = action_to_bid(act_idx)
            bidding_history_for_parse.append(act_str_alpha)

    # 解码用于显示的历史记录
    history = decode_history_from_obs(step_data['obs'], phase)

    # 解析 declarer 和 dummy
    declarer, dummy = parse_declarer_dummy(bidding_history_for_parse)

    # 解码可见手牌
    visible_cards = decode_visible_hand(step_data['obs'], player_idx)

    # 解码 Legal Actions
    legal_acts_natural = []
    if phase == 'bidding':
        for act_idx in step_data['legal_actions']:
            act_label = action_to_symbolic_bid(act_idx)
            legal_acts_natural.append(act_label)
    else:
        for act_idx in step_data['legal_actions']:
            legal_acts_natural.append(act_idx_to_str(act_idx))

    # 解码 Action Losses
    action_losses_with_label = []
    if phase == 'bidding':
        for act_idx in range(38):
            act_label = action_to_symbolic_bid(act_idx)
            loss = step_data['action_losses'][act_idx]
            val = step_data['action_values'][act_idx] if 'action_values' in step_data and act_idx < len(step_data['action_values']) else None
            action_losses_with_label.append((act_label, loss, val))
        action_losses_with_label.sort(key=lambda x: x[1])
    else:
        for act_idx in range(52):
            act_label = act_idx_to_str(act_idx)
            loss = step_data['action_losses'][act_idx]
            val = step_data['action_values'][act_idx] if 'action_values' in step_data and act_idx < len(step_data['action_values']) else None
            action_losses_with_label.append((act_label, loss, val))
        action_losses_with_label.sort(key=lambda x: x[1])

    # 获取 model_act
    model_act = step_data.get('model_act')
    if model_act is not None:
        if phase == 'bidding':
            model_act_str = action_to_symbolic_bid(model_act)
        else:
            model_act_str = act_idx_to_str(model_act)
        # 检查 model_act 是否合法
        is_legal = model_act in step_data['legal_actions']
    else:
        model_act_str = None
        is_legal = None

    # 展示
    print(f"    Step {step_idx}:")
    print(f"    - Phase: {phase}")
    print(f"    - Player: {player_str}")
    print(f"    - Declarer: {declarer}, Dummy: {dummy}")
    print(f"    - Optimal Action: {act_str}")
    if model_act_str is not None:
        legal_marker = "✓" if is_legal else "✗"
        print(f"    - Model Action: {model_act_str} ({legal_marker} legal={is_legal})")
    total_acts = 38 if phase == 'bidding' else 52
    print(f"    - Legal Actions: {len(step_data['legal_actions'])} types / {total_acts} total")
    print(f"        {', '.join(legal_acts_natural)}")
    print(f"    - Action Losses computed: {len(step_data['action_losses'])} losses (All)")
    for i, (act_label, loss, val) in enumerate(action_losses_with_label):
        marker = "✓" if act_label == act_str else " "
        if val is not None:
            print(f"        [{i+1}] {marker} {act_label} loss={loss:.4f} val={val:+.2f}")
        else:
            print(f"        [{i+1}] {marker} {act_label} loss={loss:.4f}")

    # 显示 Play History 和 Bidding History
    if len(history['bidding_history']) > 0:
        print(f"    - Bidding History: {' → '.join(history['bidding_history'])}")
    if phase == 'play' and len(history['play_history']) > 0:
        print(f"    - Play History: {' → '.join(history['play_history'])}")

    # 显示可见手牌
    player_names = [p.name.capitalize() for p in PLAYER_ORDER]
    if phase == 'bidding':
        if len(visible_cards) > 0:
            print(f"    - Visible Cards ({player_str}): {len(visible_cards)} cards: {' '.join(visible_cards)}")
    else:
        # 出牌阶段：显示所有玩家的手牌
        print(f"    - Visible Cards:")
        for pl_idx in range(4):
            pl_name = PLAYER_ORDER[pl_idx].name.capitalize()
            pl_cards = decode_visible_hand(step_data['obs'], pl_idx)
            if len(pl_cards) > 0:
                label = f"{pl_name} (明手)" if pl_name == dummy else (f"{pl_name} (庄家)" if pl_name == declarer else pl_name)
                print(f"        {label}: {len(pl_cards)} cards: {' '.join(pl_cards)}")


def main():
    parser = argparse.ArgumentParser(description="详细展示完整的 Board (扁平格式)")
    parser.add_argument('--data-file', '-d', type=str, default='data/stepwise_training_data.jsonl',
                        help='数据文件路径 (default: data/stepwise_training_data.jsonl)')
    parser.add_argument('--num-boards', '-n', type=int, default=None,
                       help='显示前 N 个 boards')
    parser.add_argument('--last-boards', '-l', type=int, default=None,
                       help='显示后 N 个 boards')
    args = parser.parse_args()

    print("=" * 80)
    print("详细展示完整的 Board (扁平格式)")
    print("=" * 80)
    print()

    data_file = args.data_file
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        sys.exit(1)

    steps = []
    with open(data_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                steps.append(json.loads(line))

    print(f"共 {len(steps)} 个 steps")
    print()

    # 按 board_idx 分组
    boards = defaultdict(list)
    for step in steps:
        boards[step.get('board_idx', 0)].append(step)

    print(f"共 {len(boards)} 个 boards")
    print()

    # 限制显示数量
    board_indices = sorted(boards.keys())
    
    if args.last_boards is not None and args.last_boards < len(board_indices):
        board_indices = board_indices[-args.last_boards:]
        print(f"显示后 {args.last_boards} 个 boards")
        print()
    
    if args.num_boards is not None and args.num_boards < len(board_indices):
        board_indices = board_indices[:args.num_boards]
        print(f"显示前 {args.num_boards} 个 boards")
        print()

    for board_idx in board_indices:
        board_steps = boards[board_idx]
        print("=" * 80)
        print(f"BOARD {board_idx}")
        print("=" * 80)
        print()

        for step_idx, step in enumerate(board_steps):
            show_detailed_step(step_idx, step)
            print()

        print("-" * 80)
        print(f"Board {board_idx} 统计:")
        print(f"  - 总步数: {len(board_steps)}")
        print(f"  - 叫牌步数: {sum(1 for s in board_steps if s['phase'] == 'bidding')}")
        print(f"  - 出牌步数: {sum(1 for s in board_steps if s['phase'] == 'play')}")
        print()

    print("=" * 80)
    print("Done!")
    print("=" * 80)


if __name__ == "__main__":
    main()
