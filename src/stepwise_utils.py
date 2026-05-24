#!/usr/bin/env python3
"""
Stepwise Training - 核心工具函数
包含：数据生成、隐藏对手牌、显示工具、步骤处理、损失计算
"""

from __future__ import annotations

import sys, os
from dataclasses import dataclass
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import numpy as np
import torch
import torch.nn.functional as F

from config import EnvConfig
from dds_teacher import (
    BidTeacher, PlayTeacher, DDTableCache,
    compute_full_policy_loss,
    card_idx_to_name, bid_idx_to_name,
    NUM_BID as _NUM_BID,
    NUM_PLAY as _NUM_PLAY
)
from env_core import BridgeEnv
from endplay.dds import par as dds_par
from endplay.types import Denom, Player
from rewards import contract_score

# =========================================================================
# Constants
# =========================================================================
NUM_BID = _NUM_BID  # 38
NUM_PLAY = _NUM_PLAY  # 52

# Bid action string names
BID_NAMES_CORRECT = []
for l in range(1, 8):
    for d in ['C', 'D', 'H', 'S', 'NT']:
        BID_NAMES_CORRECT.append(f'{l}{d}')
BID_NAMES_CORRECT += ['Pass', 'Double', 'Redouble']


# =========================================================================
# Training Data
# =========================================================================
@dataclass
class BoardSample:
    """单个牌局的一个步骤"""
    obs: np.ndarray
    label: int
    legal_mask: np.ndarray
    phase: str
    bid_level: int = 0
    bid_denom: int = -1
    is_ns_bidder: bool = False


# =========================================================================
# DDS Teacher (Re-export for convenience)
# =========================================================================
class DDSTeacher:
    """DDS 教师 - 为每步决策生成最优标签"""
    def __init__(self):
        self.bid_teacher = BidTeacher()
        self.play_teacher = PlayTeacher()

    def optimal_bid(self, dd_table, deal, vul, dealer, history, player):
        """DDS 最优叫牌动作"""
        return self.bid_teacher.optimal_bid(
            dd_table, deal, vul, dealer, history, player
        )

    def optimal_card(self, dd_table, contract_denom, declarer,
                     legal_actions, current_player, tricks_taken_declarer=0):
        """DDS 最优出牌"""
        return self.play_teacher.optimal_card(
            dd_table, contract_denom, declarer, legal_actions,
            current_player, tricks_taken_declarer
        )


# =========================================================================
# Observation Processing
# =========================================================================
def hide_opponent_cards(obs: np.ndarray, player_idx: int, phase: str,
                        declarer_idx: int = -1, dummy_idx: int = -1,
                        is_first_trick: bool = False,
                        trick_cards_played: int = 0) -> np.ndarray:
    """
    隐藏对手牌，只保留当前玩家可见的牌
    """
    import config as _cfg
    OFS_HANDS = _cfg.OFS_HANDS
    HANDS_ENCODED = _cfg.HANDS_ENCODED_DIM

    new_obs = obs.copy()

    if phase == 'bidding':
        for i in range(4):
            if i != player_idx:
                start = OFS_HANDS + i * HANDS_ENCODED
                new_obs[start:start + HANDS_ENCODED] = 0.0

    elif phase == 'play':
        is_lead = (is_first_trick and trick_cards_played == 0)

        for i in range(4):
            visible = False
            if i == player_idx:
                visible = True
            elif i == dummy_idx and not is_lead:
                visible = True
            elif i == dummy_idx and player_idx == declarer_idx and not is_lead:
                visible = True
            elif i == declarer_idx and player_idx == dummy_idx and not is_lead:
                visible = True

            if not visible:
                start = OFS_HANDS + i * HANDS_ENCODED
                new_obs[start:start + HANDS_ENCODED] = 0.0

    return new_obs


# =========================================================================
# Display Utilities
# =========================================================================
def format_hand(deal, player):
    """格式化一手牌"""
    hand = deal.__getitem__(player)
    if hand is None:
        return "?"
    suits = {'S': [], 'H': [], 'D': [], 'C': []}
    for card in hand:
        s = str(card.suit.abbr).upper()
        suits.setdefault(s, []).append(str(card.rank.abbr))
    parts = []
    for s in ['S', 'H', 'D', 'C']:
        if suits[s]:
            parts.append(f"{s}:{' '.join(suits[s])}")
    return ' | '.join(parts)


def deal_log(deal):
    """打印牌局"""
    from endplay.types import Player
    pls = [Player.north, Player.east, Player.south, Player.west]
    nms = ['N', 'E', 'S', 'W']
    parts = [f"{nm}:{format_hand(deal, pl)}" for pl, nm in zip(pls, nms)]
    print(f'  {" | ".join(parts)}')


def print_visible_hands(obs_hidden, player_idx):
    """打印可见的手牌"""
    import config as _cfg
    OFS_HANDS = _cfg.OFS_HANDS
    HANDS_ENCODED = _cfg.HANDS_ENCODED_DIM

    visible_hands = []
    players = [Player.north, Player.east, Player.south, Player.west]
    for i, p in enumerate(players):
        start = OFS_HANDS + i * HANDS_ENCODED
        hand_cards = obs_hidden[start:start + HANDS_ENCODED]
        if hand_cards.sum() == 0:
            visible_hands.append(f"{p.name}:(hidden)")
        else:
            cards_per_suit = _cfg.CARDS_PER_SUIT
            suits = {'S': [], 'H': [], 'D': [], 'C': []}
            for card_idx in range(HANDS_ENCODED):
                if hand_cards[card_idx] > 0:
                    suit = card_idx // cards_per_suit
                    rank = card_idx % cards_per_suit
                    suit_names = ['S', 'H', 'D', 'C']
                    rank_names = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K', 'A']
                    suits[suit_names[suit]].append(rank_names[rank])
            parts = []
            for s in ['S', 'H', 'D', 'C']:
                if suits[s]:
                    parts.append(f"{s}:{' '.join(suits[s])}")
            hand_str = ' | '.join(parts) if parts else '(empty)'
            if i == player_idx:
                visible_hands.append(f"{p.name}:{hand_str} *")
            else:
                visible_hands.append(f"{p.name}:{hand_str}")
    print(f'    可见: {" | ".join(visible_hands)}\n\n')


def print_bid_step(current_player, raw_pred_act, is_illegal_prediction,
                   advance_act, optimal_act, legal_names, loss_policy, loss_par,
                   bid_values, bid_tricks, legal_actions_int, bid_action_losses):
    """打印叫牌阶段的一步"""
    illegal_tag = "非法" if is_illegal_prediction else "合法"
    advance_name = BID_NAMES_CORRECT[advance_act]
    print(f'  叫牌阶段 - 玩家: {current_player} | 模型: {BID_NAMES_CORRECT[raw_pred_act]} {illegal_tag} →推进: {advance_name} | '
          f'DDS最优: {BID_NAMES_CORRECT[optimal_act]} | 合法: {legal_names} | 总损失: Policy={loss_policy.item():.3f} Par={loss_par.item():.3f}')

    val_dict = {a: bid_values[i] for i, a in enumerate(legal_actions_int)}
    tricks_dict = {a: bid_tricks[i] for i, a in enumerate(legal_actions_int)}
    bid_action_losses_display = [(BID_NAMES_CORRECT[a], bid_action_losses[a], val_dict.get(a), tricks_dict.get(a))
                                  for a in range(NUM_BID) if a < len(bid_action_losses)]
    display_parts = []
    for n, v, val, tricks in bid_action_losses_display:
        if val is not None:
            display_parts.append(f"{n} loss={v:.4f} val={val:+.2f}(t{tricks})")
        else:
            display_parts.append(f"{n} loss={v:.4f}")
    print(f'    损失表: {", ".join(display_parts)}')


def print_play_step(current_player, raw_pred_act, is_illegal_prediction,
                   advance_act, optimal_act, loss_policy, loss_par,
                   legal_actions_int, play_action_losses):
    """打印出牌阶段的一步"""
    legal_set = set(legal_actions_int)
    action_losses = [(card_idx_to_name(i), play_action_losses[i], i in legal_set) for i in range(NUM_PLAY)]

    illegal_tag = "非法" if is_illegal_prediction else "合法"
    print(f'  出牌阶段 - 玩家: {current_player} | 模型: {card_idx_to_name(raw_pred_act)} {illegal_tag} →推进: {card_idx_to_name(advance_act)} | '
          f'DDS最优: {card_idx_to_name(optimal_act)} | 总损失: Policy={loss_policy.item():.3f} Par={loss_par.item():.3f}')

    loss_parts = []
    for name, loss, is_legal in action_losses:
        marker = "*" if not is_legal else ""
        loss_parts.append(f"{marker}{name}={loss:.3f}")
    print(f'    损失表: {", ".join(loss_parts)}')


# =========================================================================
# Step Processing - Core Logic
# =========================================================================
def pick_action_from_middle_low_third(legal_actions: list, bid_values: list, history: list = None) -> int:
    """
    从 legal_actions 中选择动作，优先叫低，有正分叫牌不 Pass
    DDS 的 bid_values 已经考虑了所有因素（包括对方已叫花色），直接信任 DDS 分数

    bid_values 来自 compute_bid_values():
    - 合约叫品(0-34): 正分=能做成，负分=会宕
    - Double(36): _compute_double_value() 已处理定约方判断
      - 防守方+对方宕→正分，防守方+对方做成→负分
      - 庄家方→低分
    - Redouble(37): 同上

    :param legal_actions: 合法动作列表
    :param bid_values: 对应动作的 bid_value 列表（DDS 计算）
    :param history: 叫牌历史 [(player, action), ...]，用于参考
    :return: 选中的动作
    """
    if len(legal_actions) == 0:
        return 35  # 默认 PASS

    # 1) 优先考虑合约叫品 (0-34)：能叫自己的定约就不加倍
    positive_bids = [(act, val) for act, val in zip(legal_actions, bid_values) if act < 35 and val > 0]

    if positive_bids:
        def get_bid_level(action):
            return action // 5 + 1
        min_level = min(get_bid_level(act) for act, _ in positive_bids)
        lowest_level_bids = [(act, val) for act, val in positive_bids if get_bid_level(act) == min_level]
        selected = max(lowest_level_bids, key=lambda x: x[1])
        return selected[0]

    # 2) 没有能叫的定约 → 检查 Double/Redouble
    # _compute_double_value 已正确处理定约方:
    #   防守方+对方宕→正分 👍 防守方+对方做成→负分 👎
    #   庄家方→低分（加倍自己没意义）
    for act, val in zip(legal_actions, bid_values):
        if act == 36 and val > 0:
            return 36  # Double — 对方这个定约要宕，加倍！
        if act == 37 and val > 0:
            return 37  # Redouble — 再加倍有利可图

    return 35  # 什么都没有 → Pass


def process_bidding_step(env, model, teacher, dd_table, deal, vul, dealer, obs_orig, device):
    """处理叫牌阶段的一步，返回数据字典"""
    current_player = env._current_player()
    player_idx = current_player.value

    obs_hidden = hide_opponent_cards(obs_orig, player_idx, 'bidding')

    legal_mask = env.legal_mask()
    legal_t = torch.from_numpy(legal_mask).float().unsqueeze(0).to(device)

    obs_t = torch.from_numpy(obs_hidden).float().unsqueeze(0).to(device)
    obs_t.requires_grad_(True)
    bid_logits, _ = model(obs_t)

    masked_logits = bid_logits.clone()
    legal_bool = legal_t[:, :NUM_BID].bool().squeeze(0)
    masked_logits[0, ~legal_bool] = float('-inf')

    legal_actions_int = [int(x) for x in np.where(legal_mask[:NUM_BID])[0].tolist()]
    
    bid_values, _ = teacher.bid_teacher.compute_bid_values(
        dd_table, deal, vul, dealer,
        env.bidding.history, current_player,
        legal_actions_int
    )
    
    raw_pred_act = int(bid_logits.argmax(dim=-1).item())
    legal_actions = np.where(legal_mask[:NUM_BID])[0].tolist()
    is_illegal_prediction = raw_pred_act not in legal_actions_int
    
    # 强制使用低阶叫牌策略生成训练数据，鼓励低阶叫牌
    model_act = pick_action_from_middle_low_third(legal_actions_int, bid_values, env.bidding.history)

    try:
        optimal_act = teacher.optimal_bid(
            dd_table, deal, vul, dealer,
            env.bidding.history, current_player
        )
        if optimal_act not in legal_actions:
            if 35 in legal_actions:
                optimal_act = 35
            else:
                optimal_act = legal_actions[0]
    except Exception:
        optimal_act = 35 if 35 in legal_actions else legal_actions[0]

    _, bid_tricks = teacher.bid_teacher.compute_bid_values(
        dd_table, deal, vul, dealer,
        env.bidding.history, current_player,
        legal_actions_int
    )
    loss_policy, bid_action_losses_tensor = compute_full_policy_loss(
        bid_logits, legal_t[:, :NUM_BID].bool(),
        bid_values, legal_actions_int,
        temperature=5.0, illegal_penalty=2.0,
        negative_penalty_weight=2.0,
        phase='bidding',
        return_per_action=True
    )
    bid_action_losses = bid_action_losses_tensor.cpu().detach().numpy().tolist()
    
    action_values = [0.0] * NUM_BID
    for i, act_idx in enumerate(legal_actions_int):
        if act_idx < NUM_BID:
            action_values[act_idx] = bid_values[i]
    
    for act_idx in range(NUM_BID):
        if act_idx not in legal_actions_int:
            if act_idx == 36:
                action_values[act_idx] = -20.0
            elif act_idx == 37:
                action_values[act_idx] = -30.0

    return {
        'obs_hidden': obs_hidden,
        'obs_t': obs_t,
        'bid_logits': bid_logits,
        'model_act': model_act,
        'raw_pred_act': raw_pred_act,
        'legal_actions': legal_actions,
        'legal_actions_int': legal_actions_int,
        'legal_t': legal_t,
        'is_illegal_prediction': is_illegal_prediction,
        'optimal_act': optimal_act,
        'bid_values': bid_values,
        'bid_tricks': bid_tricks,
        'loss_policy': loss_policy,
        'bid_action_losses': bid_action_losses,
        'action_values': action_values,
        'player_idx': player_idx,
    }


def process_play_step(env, model, teacher, dd_table, obs_orig, declarer, declarer_idx,
                     dummy, dummy_idx, lead_player, device):
    """处理出牌阶段的一步，返回数据字典"""
    current_player = env._current_player()
    player_idx = current_player.value

    is_first_trick = (env.play_state._trick_no == 0)
    trick_cards_played = len(env.play_state._current_trick_cards)

    obs_hidden = hide_opponent_cards(
        obs_orig, player_idx, 'play',
        declarer_idx, dummy_idx,
        is_first_trick, trick_cards_played
    )

    legal_mask = env.legal_mask()
    legal_actions = np.where(legal_mask[:NUM_PLAY])[0].tolist()
    if not legal_actions:
        return None

    legal_t = torch.from_numpy(legal_mask).float().unsqueeze(0).to(device)

    obs_t = torch.from_numpy(obs_hidden).float().unsqueeze(0).to(device)
    obs_t.requires_grad_(True)
    _, play_logits = model(obs_t)

    masked_logits = play_logits.clone()
    legal_bool = legal_t[:, :NUM_PLAY].bool().squeeze(0)
    masked_logits[0, ~legal_bool] = float('-inf')

    model_act = int(masked_logits.argmax(dim=-1).item())

    raw_pred_act = int(play_logits.argmax(dim=-1).item())
    legal_actions_int = [int(x) for x in legal_actions]
    is_illegal_prediction = raw_pred_act not in legal_actions_int

    try:
        optimal_act = teacher.optimal_card(
            dd_table, env.bidding.contract_denom,
            declarer, legal_actions, current_player,
            env.play_state._tricks_won_ns if declarer_idx % 2 == 0 else 0
        )
    except Exception:
        optimal_act = legal_actions[0]

    card_values = teacher.play_teacher.compute_card_values(
        dd_table, env.bidding.contract_denom,
        declarer, legal_actions_int, current_player,
        lead_suit=None, is_declarer_lead=(trick_cards_played == 0)
    )
    loss_policy, play_action_losses_tensor = compute_full_policy_loss(
        play_logits, legal_t[:, :NUM_PLAY].bool(),
        card_values, legal_actions_int,
        temperature=5.0, illegal_penalty=10.0,
        negative_penalty_weight=2.0,
        phase='play',
        return_per_action=True
    )
    play_action_losses = play_action_losses_tensor.cpu().detach().numpy().tolist()
    
    action_values = [0.0] * NUM_PLAY
    for i, act_idx in enumerate(legal_actions_int):
        if act_idx < NUM_PLAY:
            action_values[act_idx] = card_values[i]

    return {
        'obs_hidden': obs_hidden,
        'obs_t': obs_t,
        'play_logits': play_logits,
        'model_act': model_act,
        'raw_pred_act': raw_pred_act,
        'legal_actions': legal_actions,
        'legal_actions_int': legal_actions_int,
        'legal_t': legal_t,
        'is_illegal_prediction': is_illegal_prediction,
        'optimal_act': optimal_act,
        'card_values': card_values,
        'loss_policy': loss_policy,
        'play_action_losses': play_action_losses,
        'action_values': action_values,
        'player_idx': player_idx,
    }


# =========================================================================
# Loss Calculation - Par Gap
# =========================================================================
def compute_bid_par_loss(action_idx: int, dd_table: DDTableCache, env: BridgeEnv, deal, device):
    """计算叫牌动作的 DDS Par 差距损失"""
    if action_idx >= 36:
        return torch.tensor(0.0, device=device)

    current_player = env._current_player()
    is_ns = current_player.value % 2 == 0
    vul_e = env.vul

    try:
        dealer = Player.north
        par_result = dds_par(deal, vul_e, dealer)
        par_score = par_result.score
    except Exception:
        par_score = 0

    if action_idx == 35:
        best_score_for_player = -99999
        denom_map = [Denom.clubs, Denom.diamonds, Denom.hearts, Denom.spades, Denom.nt]
        for lv in range(1, 8):
            for di in range(5):
                denom = denom_map[di]
                expected_tricks = dd_table.tricks(denom, current_player)
                tricks_needed = lv + 6
                if expected_tricks >= tricks_needed:
                    score = contract_score(lv, denom, 0, vul_e, expected_tricks)
                else:
                    score = contract_score(lv, denom, 0, vul_e, expected_tricks)
                if not is_ns:
                    score = -score
                if score > best_score_for_player:
                    best_score_for_player = score

        if best_score_for_player == -99999:
            pass_gap = 0.0
        else:
            pass_gap = max(0.0, (best_score_for_player - par_score) / 2000.0)
        return torch.tensor(min(pass_gap, 1.0), device=device)

    level = action_idx // 5 + 1
    denom_idx = action_idx % 5
    denom_map = [Denom.clubs, Denom.diamonds, Denom.hearts, Denom.spades, Denom.nt]
    denom = denom_map[denom_idx]

    expected_tricks = dd_table.tricks(denom, current_player)
    bid_score = contract_score(level, denom, 0, vul_e, expected_tricks)
    if not is_ns:
        bid_score = -bid_score

    gap = abs(bid_score - par_score)
    return torch.tensor(min(gap / 2000.0, 1.0), device=device)


def compute_play_par_loss(dd_table: DDTableCache, env: BridgeEnv, deal, device):
    """计算出牌阶段的 DDS Par 差距损失"""
    if env.bidding.contract_level == 0 or env.play_state is None:
        return torch.tensor(0.0, device=device)

    contract_level = env.bidding.contract_level
    contract_denom = env.bidding.contract_denom
    declarer = env.bidding.declarer
    vul_e = env.vul

    if declarer.value % 2 == 0:
        tricks_won = env.play_state._tricks_won_ns
    else:
        tricks_won = env.play_state._tricks_won_ew

    dd_optimal = dd_table.tricks(contract_denom, declarer)
    remaining = _cfg.CARDS_PER_HAND - env.play_state._trick_no
    dec_remaining_optimal = max(0, dd_optimal - tricks_won)
    if dec_remaining_optimal > remaining:
        dec_remaining_optimal = remaining
    final_tricks = tricks_won + dec_remaining_optimal

    bid_score = contract_score(contract_level, contract_denom, 0, vul_e, final_tricks)

    try:
        dealer = Player.north
        par_result = dds_par(deal, vul_e, dealer)
        par_score = par_result.score
    except Exception:
        par_score = 0

    if declarer.value % 2 != 0:
        bid_score = -bid_score

    gap = abs(bid_score - par_score)
    return torch.tensor(min(gap / 2000.0, 1.0), device=device)
