"""
DDS Teacher — bridge supervised learning via double-dummy optimal actions.

For each decision point (bidding or play), evaluates all legal actions
via DDS and picks the optimal one as the training label.

Loss structure:
  * Cross-entropy on (predicted_logits, optimal_label)
  * PLUS illegal-action penalty: penalise any non-zero probability
    on illegal actions (set to -inf in logits, but we also add an
    auxiliary loss term that penalises logits[illegal] being high)
"""

from __future__ import annotations
import random as _random
import numpy as np
from typing import List, Optional, Tuple

from endplay import Deal, generate_deal
from endplay.types import Player, Denom, Vul, Rank
from endplay.dds import calc_dd_table, par as dds_par

from config import EnvConfig
from rewards import contract_score, calculate_contract_score

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_BID_ACTIONS = list(range(35))  # 0..34 = 1C..7NT
_PASS = 35
_DOUBLE = 36
_REDOUBLE = 37
NUM_BID = _REDOUBLE + 1  # 38 = 37 + 1
NUM_PLAY = EnvConfig.num_play_actions  # 52
NUM_SUITS = 4
CARDS_PER_SUIT = NUM_PLAY // NUM_SUITS         # 13

PLAYER_ORDER = [Player.north, Player.east, Player.south, Player.west]

BID_LEVELS = list(range(1, 8))
BID_DENOMS = [Denom.clubs, Denom.diamonds, Denom.hearts, Denom.spades, Denom.nt]
SUIT_TO_DENOM = [Denom.spades, Denom.hearts, Denom.diamonds, Denom.clubs]

# Map (level, denom) → action index (0..34)
LEVEL_DENOM_TO_ACTION: dict[tuple[int, Denom], int] = {}
for level in BID_LEVELS:
    for denom in BID_DENOMS:
        idx = (level - 1) * 5 + BID_DENOMS.index(denom)
        LEVEL_DENOM_TO_ACTION[(level, denom)] = idx


def action_to_bid_str(a: int) -> str:
    """Convert action index to readable bid string using Endplay."""
    from endplay.types import Bid
    if a == _PASS:
        return str(Bid("Pass"))
    if a == _DOUBLE:
        return str(Bid("X"))
    if a == _REDOUBLE:
        return str(Bid("XX"))
    level = a // 5 + 1
    denom_abbr = BID_DENOMS[a % 5].abbr
    return str(Bid(f"{level}{denom_abbr}"))


def random_deal() -> Deal:
    """Generate a random deal using Endplay's built-in function."""
    return generate_deal()


# ---------------------------------------------------------------------------
# DDS helpers — single DD table per deal to minimise DDS calls
# ---------------------------------------------------------------------------

class DDTableCache:
    """Caches a single calc_dd_table result per deal.

    Usage:
        cache = DDTableCache(deal)
        tricks = cache.tricks(Denom.spades, Player.north)
    """

    def __init__(self, deal: Deal):
        self._table = calc_dd_table(deal)

    def tricks(self, denom: Denom, player: Player) -> int:
        return self._table[(denom, player)]


def _parscore_ns(deal: Deal, vul: Vul, dealer: Player) -> int:
    """NS parscore. Positive = NS gains; negative = EW gains."""
    res = dds_par(deal, vul, dealer)
    return res.score  # int property


# ---------------------------------------------------------------------------
# Illegal-action penalty loss
# ---------------------------------------------------------------------------
def illegal_action_loss(logits: torch.Tensor,
                        legal_mask: torch.Tensor,
                        penalty_weight: float = 0.1) -> torch.Tensor:
    """Penalty loss for assigning probability mass to illegal actions.

    Args:
        logits: (B, num_actions) raw logits.
        legal_mask: (B, num_actions) bool — True for legal actions.
        penalty_weight: Scaling factor.

    Returns:
        Scalar loss tensor.
    """
    probs = F.softmax(logits, dim=-1)
    illegal_probs = probs * (~legal_mask).float()
    penalty = illegal_probs.sum(dim=-1).mean()
    return penalty_weight * penalty


def compute_full_policy_loss(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    action_values: list[float],
    legal_actions: list[int],
    temperature: float = 1.0,
    illegal_penalty: float = 1.0,
    negative_penalty_weight: float = 2.0,
    phase: str = 'bidding',
    return_per_action: bool = False
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute loss over ALL actions based on DDS-evaluated values.

    For each action:
    - If illegal: target = 0 (will be masked)
    - If legal: target probability proportional to exp(value / temperature)

    Then compute KL divergence between model output and target.

    Args:
        logits: (B, num_actions) raw logits.
        legal_mask: (B, num_actions) bool — True for legal actions.
        action_values: List of DDS-evaluated values for each legal action.
                      action_values[i] corresponds to legal_actions[i].
        temperature: Controls how peaked the target distribution is.
                    Higher = more uniform, Lower = more peaked on best action.
        illegal_penalty: Multiplier for illegal action penalty.
        negative_penalty_weight: Multiplier for amplifying negative values.
                    Higher values discourage negative bids more strongly.
        phase: 'bidding' or 'play' — level penalty only applies to bidding.
        return_per_action: If True, also return per-action losses for display.

    Returns:
        If return_per_action=False: Scalar loss tensor (KL divergence + illegal penalty).
        If return_per_action=True: Tuple of (total_loss, per_action_losses).

    Note:
        When there are positive action values (winning bids) in bidding phase,
        a small level penalty is applied to encourage lower bids among the positive ones.
        Higher action indices get a small negative bias, preferring the lowest winning bid.
        No level penalty is applied in play phase.
    """
    num_actions = logits.size(-1)
    probs = F.softmax(logits, dim=-1)

    target = torch.zeros(num_actions, device=logits.device)
    if len(action_values) > 0 and len(legal_actions) > 0:
        # Safety check: make sure lengths match to avoid crashes
        min_len = min(len(action_values), len(legal_actions))
        safe_action_values = action_values[:min_len]
        safe_legal_actions = legal_actions[:min_len]
        
        values_tensor = torch.tensor(safe_action_values, device=logits.device)
        amplified_values = torch.where(
            values_tensor < 0,
            values_tensor * negative_penalty_weight,
            values_tensor
        )
        
        # Encourage lower bids for positive action values (only in bidding phase)
        # Add a small negative bias to higher action indices among positive values
        if phase == 'bidding':
            try:
                positive_mask = (values_tensor >= 0)
                if positive_mask.any():
                    # Penalty for higher action indices (lower bids are preferred)
                    # Apply based on actual action index (lower index = lower bid)
                    level_penalty = torch.zeros(len(safe_action_values), device=logits.device)
                    for i, act_idx in enumerate(safe_legal_actions):
                        if i < len(positive_mask) and positive_mask[i]:
                            # Penalty increases with action index (higher bids get more penalty)
                            # Strong penalty to strongly prefer lower bids among positive ones
                            level_penalty[i] = act_idx * 2.0
                    amplified_values = amplified_values - level_penalty
            except Exception:
                # If anything goes wrong, skip the level penalty to avoid crashing
                pass
        
        exp_values = torch.exp(amplified_values / temperature)
        exp_values_sum = exp_values.sum()
        if exp_values_sum > 0:
            exp_dist = exp_values / exp_values_sum
            for i, act_idx in enumerate(safe_legal_actions):
                if act_idx < num_actions:
                    target[act_idx] = exp_dist[i]

    target = target.unsqueeze(0).expand_as(probs)

    legal_mask_float = legal_mask.float()
    target = target * legal_mask_float

    target_sum = target.sum(dim=-1, keepdim=True)
    target = target / (target_sum + 1e-8)

    safe_target = target.clamp(min=1e-8)
    safe_probs = probs.clamp(min=1e-8)
    # D(P||Q) = P * log(P/Q) 当 P 大 Q 小时，损失大
    kl_loss_per_action = safe_target * (safe_target.log() - safe_probs.log())
    illegal_mask = (~legal_mask.bool()).float()
    illegal_penalty_per_action = illegal_mask * probs * illegal_penalty
    per_action_losses = kl_loss_per_action + illegal_penalty_per_action

    kl_loss = kl_loss_per_action.sum(dim=-1).mean()
    illegal_penalty_loss = illegal_penalty_per_action.sum(dim=-1).mean()

    total_loss = kl_loss + illegal_penalty_loss

    if return_per_action:
        return total_loss, per_action_losses.squeeze(0)
    return total_loss


def compute_per_action_loss(
    logits: torch.Tensor,
    legal_mask: torch.Tensor,
    action_values: list[float],
    legal_actions: list[int] = [],
    temperature: float = 1.0
) -> tuple[list[float], list[float]]:
    """Compute per-action loss and target values for display.

    Args:
        logits: (B, num_actions) raw logits.
        legal_mask: (B, num_actions) bool — True for legal actions.
        action_values: List of DDS-evaluated values for each legal action.
                      action_values[i] corresponds to legal_actions[i].
        legal_actions: List of actual action indices that are legal.
        temperature: Temperature for softmax.

    Returns:
        Tuple of (target_probs, kl_divs) where each is a list of floats,
        one per action (0-51 for play, 0-37 for bid).
    """
    logits_2d = logits.squeeze(0) if logits.dim() > 2 else logits
    num_actions = logits_2d.size(-1)
    probs = F.softmax(logits_2d, dim=-1).cpu().detach().numpy()
    if probs.ndim > 1:
        probs = probs.squeeze(0)

    target = np.zeros(num_actions)
    if len(action_values) > 0 and len(legal_actions) > 0:
        exp_values = np.exp(np.array(action_values) / temperature)
        exp_values_sum = exp_values.sum()
        if exp_values_sum > 0:
            exp_dist = exp_values / exp_values_sum
            for i, act_idx in enumerate(legal_actions):
                if act_idx < num_actions:
                    target[act_idx] = exp_dist[i]

    target_sum = target.sum()
    if target_sum > 0:
        target = target / target_sum

    safe_target = np.clip(target, 1e-8, 1.0)
    safe_probs = np.clip(probs, 1e-8, 1.0)

    illegal_mask = (target < 1e-7).astype(float)
    kl_divs = safe_probs * np.log(safe_probs / safe_target)
    kl_divs = kl_divs + illegal_mask * safe_probs * 10

    return target.tolist(), kl_divs.tolist()


# Card index to string name (0-51 -> C2, D2, ..., AD)
def card_idx_to_name(idx: int) -> str:
    suits = ['C', 'D', 'H', 'S']
    ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    suit = suits[idx // CARDS_PER_SUIT]
    rank = ranks[idx % CARDS_PER_SUIT]
    return f"{rank}{suit}"


# Bid action index to string name (0-34 -> 1C..7NT, 35=Pass, 36=Double, 37=Redouble)
def bid_idx_to_name(idx: int) -> str:
    if idx < 35:
        level = idx // 5 + 1
        denom = ['C', 'D', 'H', 'S', 'NT'][idx % 5]
        return f"{level}{denom}"
    elif idx == 35:
        return "Pass"
    elif idx == 36:
        return "Double"
    else:
        return "Redouble"


# ===================================================================
# Bidding teacher
# ===================================================================
class BidTeacher:
    """Generates optimal bidding labels via DDS parscore analysis.

    For each state in the auction, evaluates all makeable contracts
    that are legally reachable and picks the one that maximises
    the bidding side's score.

    Performance: uses a single DDTable per deal (precomputed externally)
    to avoid calling calc_dd_table repeatedly.
    """

    def __init__(self):
        pass

    def optimal_bid(self, dd_table: DDTableCache,
                    deal: Deal, vul: Vul, dealer: Player,
                    history: list[tuple[Player, int]],
                    player: Player) -> int:
        """Find the optimal bid action for a given auction state.

        Args:
            dd_table: Precomputed DDTable for this deal.
            deal: The full deal.
            vul: Vulnerability.
            dealer: Dealer (Player).
            history: Auction so far, list of (Player, action_idx).
            player: Whose turn it is.

        Returns:
            Action index (0-37).
        """
        # ---- Find current contract ----
        first_bid_player = None
        last_bid_player = None
        last_level = 0
        last_denom_idx = -1
        for p, a in history:
            if a < 35:
                last_level = a // 5 + 1
                last_denom_idx = a % 5
                last_bid_player = p
            if a < 35 and first_bid_player is None:
                first_bid_player = p

        if last_bid_player is not None:
            is_ns = last_bid_player.value % 2 == 0
        else:
            is_ns = dealer.value % 2 == 0

        bidding_side = Player.north if is_ns else Player.east

        # Build a list of all (level, denom_idx, denom, action, tricks)
        # for makeable contracts at every level
        all_makeable: list[tuple[int, int, Denom, int, int]] = []
        for denom_idx, denom in enumerate(BID_DENOMS):
            tricks = dd_table.tricks(denom, bidding_side)
            for level in range(1, 8):
                if tricks >= level + 6:
                    action = LEVEL_DENOM_TO_ACTION[(level, denom)]
                    all_makeable.append((level, denom_idx, denom, action, tricks))

        if not all_makeable:
            return _PASS  # Nothing makes — just pass

        # ---- Opening (no bids yet) ----
        # For consistency with compute_bid_values, find the highest scoring makeable contract
        if last_level == 0:
            scored_contracts = []
            for denom_idx, denom in enumerate(BID_DENOMS):
                tricks, score = calculate_contract_score(deal, 1, denom, bidding_side, vul, 0)
                for level in range(1, 8):
                    if tricks >= level + 6:
                        action = LEVEL_DENOM_TO_ACTION.get((level, denom))
                        if action is not None:
                            _, score = calculate_contract_score(deal, level, denom, bidding_side, vul, 0)
                            if not is_ns:
                                score = -score
                            scored_contracts.append((level, denom_idx, action, tricks, score))
            if scored_contracts:
                # Sort by score descending, then level descending (higher level better when scores equal)
                scored_contracts.sort(key=lambda x: (-x[4], -x[0]))  # (-score, -level)
                return scored_contracts[0][2]  # Return highest scoring action
            return _PASS

        # ---- Non-opening (responding / overcalling) ----
        # Find all makeable contracts we can legally bid
        candidates = []
        for level, denom_idx, denom, action, tricks in all_makeable:
            if level < last_level:
                continue
            if level == last_level and denom_idx <= last_denom_idx:
                continue
            candidates.append((level, denom_idx, action, tricks))

        if not candidates:
            return _PASS

        # Sort: most profitable contract first (considering both tricks and score)
        # For scoring, we need to compute actual scores
        scored_candidates = []
        for level, denom_idx, action, tricks in candidates:
            denom = BID_DENOMS[denom_idx]
            _, score = calculate_contract_score(deal, level, denom, bidding_side, vul, 0)
            if not is_ns:
                score = -score  # EW perspective
            scored_candidates.append((level, denom_idx, action, tricks, score))

        # Sort by score descending (most profitable first), then level descending
        scored_candidates.sort(key=lambda x: (-x[4], -x[0]))

        # Return the most profitable makeable contract
        if scored_candidates:
            return scored_candidates[0][2]  # action

        return _PASS

    def compute_bid_values(
        self,
        dd_table: DDTableCache,
        deal: Deal,
        vul: Vul,
        dealer: Player,
        history: list[tuple[Player, int]],
        player: Player,
        legal_actions: list[int]
    ) -> list[float]:
        """Compute value for each legal bid action.

        Evaluates each candidate action as if it were already executed,
        so DDS PAR reflects the "pushed" state after that action.

        Args:
            dd_table: Precomputed DDTable.
            deal: The full deal.
            vul: Vulnerability.
            dealer: Dealer (Player).
            history: Auction so far (before this turn).
            player: Whose turn it is.
            legal_actions: List of legal bid action indices.

        Returns:
            List of float values, one per action in legal_actions.
            Higher = better for the player.
        """
        if not legal_actions:
            return [], []

        first_bid_player = None
        last_bid_player = None
        last_level = 0
        last_denom_idx = -1
        for p, a in history:
            if a < 35:
                last_level = a // 5 + 1
                last_denom_idx = a % 5
                last_bid_player = p
            if a < 35 and first_bid_player is None:
                first_bid_player = p

        if last_bid_player is not None:
            is_ns = last_bid_player.value % 2 == 0
        else:
            is_ns = dealer.value % 2 == 0

        bidding_side = Player.north if is_ns else Player.east

        all_makeable = []
        all_tricks_by_denom = {}
        for denom_idx, denom in enumerate(BID_DENOMS):
            tricks, score = calculate_contract_score(deal, 1, denom, bidding_side, vul, 0)
            all_tricks_by_denom[denom] = tricks
            for level in range(1, 8):
                if tricks >= level + 6:
                    action = LEVEL_DENOM_TO_ACTION[(level, denom)]
                    _, score = calculate_contract_score(deal, level, denom, bidding_side, vul, 0)
                    if not is_ns:
                        score = -score
                    all_makeable.append((level, denom_idx, denom, action, tricks, score))

        if not all_makeable:
            # 没有更优的叫牌，说明当前合约已经是最优的
            # Pass 的价值 = 当前合约的价值
            all_possible_scores = []
            for act in legal_actions:
                if act == 35:
                    continue
                elif act == 36 or act == 37:
                    continue
                else:
                    level = act // 5 + 1
                    denom_idx = act % 5
                    denom = BID_DENOMS[denom_idx]
                    _, score = calculate_contract_score(deal, level, denom, bidding_side, vul, 0)
                    if not is_ns:
                        score = -score
                    all_possible_scores.append(score)

            max_score = max(abs(s) for s in all_possible_scores) if all_possible_scores else 1.0
            if max_score == 0:
                max_score = 1.0

            # 计算当前合约的分数作为 Pass 的价值
            is_player_ns = player.value % 2 == 0
            declarer = last_bid_player if last_bid_player else dealer
            declarer_is_ns = declarer.value % 2 == 0
            is_declarer_side = (is_player_ns and declarer_is_ns) or (not is_player_ns and not declarer_is_ns)
            
            action_values = []
            action_tricks = []
            for act in legal_actions:
                if act == 35:
                    value = self._compute_pass_value(last_level, last_denom_idx, is_declarer_side, 
                                                    deal, declarer, vul, max_score)
                    action_values.append(value)
                    action_tricks.append(0)
                elif act == 36:
                    if last_level > 0:
                        current_denom = BID_DENOMS[last_denom_idx]
                        value = self._compute_double_value(last_level, last_denom_idx, is_declarer_side, 
                                                          deal, current_denom, declarer, vul, max_score, doubled=1)
                    else:
                        # 没有当前定约（开叫前），Double 不应该存在，给予极大惩罚
                        value = -max_score * 2
                    action_values.append(value)
                    action_tricks.append(0)
                elif act == 37:
                    if last_level > 0:
                        current_denom = BID_DENOMS[last_denom_idx]
                        value = self._compute_redouble_value(last_level, last_denom_idx, is_declarer_side, 
                                                           deal, current_denom, declarer, vul, max_score, doubled=2)
                    else:
                        # 没有当前定约（开叫前），Redouble 不应该存在，给予极大惩罚
                        value = -max_score * 3
                    action_values.append(value)
                    action_tricks.append(0)
                else:
                    level = act // 5 + 1
                    denom_idx = act % 5
                    denom = BID_DENOMS[denom_idx]
                    tricks, score = calculate_contract_score(deal, level, denom, bidding_side, vul, 0)
                    needed = level + 6

                    if not is_ns:
                        score = -score

                    # 对超叫（会宕）施加额外惩罚
                    if tricks < needed:
                        down_count = needed - tricks
                        score = score - down_count * 50

                    value = score / max_score * 10
                    # Clip value to prevent overflow in softmax
                    value = max(min(value, 20.0), -20.0)
                    action_values.append(value)
                    action_tricks.append(tricks)

            return action_values, action_tricks

        # Compute all legal bidding scores to find par_score for normalization
        # 超叫定义：得分低于 DDS/par 给出的能完成的叫牌中最低得分的叫牌
        # 二三法则：估算安全水平 = level + 2 (二法则) 或 level + 3 (三法则)
        all_legal_scores = []
        for idx, act in enumerate(legal_actions):
            if act == 35 or act == 36 or act == 37:
                continue
            else:
                level = act // 5 + 1
                denom_idx = act % 5
                denom = BID_DENOMS[denom_idx]
                tricks, score = calculate_contract_score(deal, level, denom, bidding_side, vul, 0)
                needed = level + 6
                bidding_side = 0 if is_ns else 1
                if bidding_side == 1:
                    score = -score
                all_legal_scores.append((idx, score, tricks, needed))

        # 找出能完成的叫牌的最低分作为基准
        normal_scores = [s for (_, s, t, n) in all_legal_scores if t >= n]
        min_normal_score = min(normal_scores) if normal_scores else 0.0
        max_score = max(abs(s) for s in normal_scores) if normal_scores else 1.0
        if max_score == 0:
            max_score = 1.0
        if max_score < 100:
            max_score = 100.0

        # 二三法则：找出所有叫牌中最高的"安全水平" (level + 2)
        # 如果某个叫牌能拿到的墩数 < level + 2，则为超叫
        max_safe_level = 0
        for (_, score, tricks, needed) in all_legal_scores:
            if score > 0:  # 只考虑能完成的叫牌
                level = needed - 6
                safe_tricks = level + 2  # 二法则
                if tricks >= safe_tricks:
                    max_safe_level = max(max_safe_level, level)

        # contract_score 返回：正值 = 庄家得分，负值 = 防守方得分
        # 庄家视角的得分 = NS 得分（因为 Player.north 是庄家）
        is_player_ns = player.value % 2 == 0
        declarer = last_bid_player if last_bid_player else dealer
        declarer_is_ns = declarer.value % 2 == 0
        declarer_score = 0  # 先初始化一个默认值
        
        if last_level > 0:
            denom = BID_DENOMS[last_denom_idx]
            actual_tricks, declarer_score = calculate_contract_score(deal, last_level, denom, declarer, vul, 0)
            # declarer_score > 0 表示庄家得分（正分），< 0 表示防守方得分
            # 如果玩家和庄家同侧，player_par = declarer_score
            # 如果玩家和庄家不同侧，player_par = -declarer_score
            if (is_player_ns and declarer_is_ns) or (not is_player_ns and not declarer_is_ns):
                player_par = declarer_score
            else:
                player_par = -declarer_score
        else:
            ns_par = _parscore_ns(deal, vul, dealer)
            # ns_par 是 NS 方应该得的分数
            if is_player_ns:
                player_par = ns_par
            else:
                player_par = -ns_par

        action_values = []
        action_tricks = []
        
        # 计算玩家是否在庄家方
        is_declarer_side = (is_player_ns and declarer_is_ns) or (not is_player_ns and not declarer_is_ns)
        
        # 准备加倍/再加倍需要的 denom 参数
        current_denom = BID_DENOMS[last_denom_idx] if last_level > 0 else None
        
        for act in legal_actions:
            # ============================================
            # 元动作：Pass, Double, Redouble
            # ============================================
            if act == _PASS:
                value = self._compute_pass_value(last_level, last_denom_idx, is_declarer_side, 
                                                deal, declarer, vul, max_score)
                action_values.append(value)
                action_tricks.append(0)
                continue
                
            if act == _DOUBLE:
                value = self._compute_double_value(last_level, last_denom_idx, is_declarer_side, 
                                                  deal, current_denom, declarer, vul, max_score, doubled=1)
                action_values.append(value)
                action_tricks.append(0)
                continue
                
            if act == _REDOUBLE:
                value = self._compute_redouble_value(last_level, last_denom_idx, is_declarer_side, 
                                                   deal, current_denom, declarer, vul, max_score, doubled=2)
                action_values.append(value)
                action_tricks.append(0)
                continue
            
            # ============================================
            # 普通叫牌动作
            # ============================================
            level = act // 5 + 1
            denom_idx = act % 5

            if level < last_level:
                value = -max_score / 3
                value = max(min(value, 20.0), -20.0)
                action_values.append(value)
                action_tricks.append(0)
                continue
            if level == last_level and denom_idx <= last_denom_idx:
                value = -max_score / 3
                value = max(min(value, 20.0), -20.0)
                action_values.append(value)
                action_tricks.append(0)
                continue

            denom = BID_DENOMS[denom_idx]
            actual_tricks, score = calculate_contract_score(deal, level, denom, bidding_side, vul, 0)
            needed = level + 6

            # 防守方竞叫时，bidding_side 应该从防守方视角计算
            # bidding_side: 0=NS 庄家, 1=EW 庄家
            bidding_side_player = Player.north if is_ns else Player.east
            if bidding_side == 1:
                score = -score  # 从防守方视角，EW 得分为正

            # 归一化
            value = score / max_score * 10

            # 二三法则超叫惩罚：如果能拿到的墩数 < level + 2 (二法则)，则为超叫
            safe_tricks = level + 2
            if actual_tricks < safe_tricks:
                # 宕的墩数越多，惩罚越大
                down_count = safe_tricks - actual_tricks
                penalty = down_count * 3.0
                value = value - penalty

            value = max(min(value, 20.0), -20.0)
            action_values.append(value)
            action_tricks.append(actual_tricks)

        return action_values, action_tricks
    
    def _compute_pass_value(self, last_level, last_denom_idx, is_declarer_side, 
                           deal, declarer, vul, max_score):
        """Compute value for Pass action.
        
        Pass = 接受当前定约
        - 如果是防守方，看对方定约是否能完成
        - 如果是庄家方，看我们的定约是否能完成
        """
        if last_level > 0:
            denom = BID_DENOMS[last_denom_idx]
            actual_tricks, declarer_score = calculate_contract_score(deal, last_level, denom, declarer, vul, 0)
            needed = last_level + 6
            
            if not is_declarer_side:
                # 防守方视角：对方不能完成则 Pass 好，能完成则 Pass 不好
                if actual_tricks < needed:
                    value = abs(declarer_score) / max_score * 10
                else:
                    value = -abs(declarer_score) / max_score * 10
            else:
                # 庄家方视角：Pass = 接受当前定约
                value = declarer_score / max_score * 10
        else:
            # 没有叫牌历史（开叫前），Pass 应该受到惩罚
            value = -10.0
        return max(min(value, 20.0), -20.0)
    
    def _compute_double_value(self, last_level, last_denom_idx, is_declarer_side, 
                             deal, denom, declarer, vul, max_score, doubled=1):
        """Compute value for Double action using calculate_contract_score."""
        if last_level > 0 and denom is not None:
            denom = BID_DENOMS[last_denom_idx]
            actual_tricks, double_score = calculate_contract_score(deal, last_level, denom, declarer, vul, doubled=doubled)
            needed = last_level + 6
            
            if not is_declarer_side:
                # 防守方视角：分数是庄家视角的相反数
                player_score = -double_score
                if actual_tricks < needed:
                    # 对方宕了，防守方得分，加倍好
                    value = abs(player_score) / max_score * 10 * 1.5
                else:
                    # 对方做成了，防守方失分，加倍不好
                    value = -abs(player_score) / max_score * 10 * 2
            else:
                # 庄家方加倍没有意义
                value = -max_score / 5
        else:
            # 没有当前定约（开叫前），Double 不应该存在，给予极大惩罚
            value = -max_score * 2
        return max(min(value, 20.0), -20.0)
    
    def _compute_redouble_value(self, last_level, last_denom_idx, is_declarer_side, 
                             deal, denom, declarer, vul, max_score, doubled=2):
        """Compute value for Redouble action using calculate_contract_score."""
        if last_level > 0 and denom is not None:
            denom = BID_DENOMS[last_denom_idx]
            actual_tricks, redouble_score = calculate_contract_score(deal, last_level, denom, declarer, vul, doubled=doubled)
            needed = last_level + 6
            
            if not is_declarer_side:
                # 防守方视角：分数是庄家视角的相反数
                player_score = -redouble_score
                if actual_tricks < needed:
                    # 对方宕了，防守方得分，再加倍好
                    value = abs(player_score) / max_score * 10 * 2
                else:
                    # 对方做成了，防守方失分，再加倍不好
                    value = -abs(player_score) / max_score * 10 * 3
            else:
                # 庄家方再加倍没有意义
                value = -max_score / 4
        else:
            # 没有当前定约（开叫前），Redouble 不应该存在，给予极大惩罚
            value = -max_score * 3
        return max(min(value, 20.0), -20.0)


# ===================================================================
# Play teacher
# ===================================================================
class PlayTeacher:
    """Generates optimal play labels via DDS.

    For each legal card, computes remaining tricks for declarer's side
    and picks the card that maximises (declarer) or minimises (defender).
    """

    def __init__(self):
        pass

    def optimal_card(self, dd_table: DDTableCache,
                     contract_denom: Denom,
                     declarer: Player,
                     legal_actions: list[int],
                     current_player: Player,
                     tricks_taken_declarer: int = 0) -> int:
        """Pick the optimal card for the current player.

        Args:
            dd_table: Precomputed DDTable for this deal.
            contract_denom: Trump suit.
            declarer: Declarer.
            legal_actions: List of legal card indices (0-51).
            current_player: Who's playing.
            tricks_taken_declarer: How many tricks declarer has already won.

        Returns:
            Optimal card index (0-51).
        """
        if len(legal_actions) == 1:
            return legal_actions[0]

        is_declarer_side = declarer.value % 2 == current_player.value % 2
        total_tricks = dd_table.tricks(contract_denom, declarer)
        remaining_tricks = CARDS_PER_SUIT - tricks_taken_declarer
        
        # DDS 结果：对于庄家方，总共能赢 total_tricks 墩
        # 剩余需要赢的墩数
        needed_tricks = max(0, (contract_denom.value * 0) + 6 + 1 - tricks_taken_declarer)  # 庄家需要 7 墩
        
        def card_tricks_value(action: int) -> float:
            suit = action // CARDS_PER_SUIT
            rank = action % CARDS_PER_SUIT  # 0=2, 12=A

            suit_denom = SUIT_TO_DENOM[suit]
            suit_tricks = dd_table.tricks(suit_denom, declarer)
            max_rank = CARDS_PER_SUIT - 1  # 12

            if is_declarer_side:
                rank_value = rank / max_rank
                dds_value = suit_tricks / CARDS_PER_SUIT
                return rank_value * 0.3 + dds_value * 0.7
            else:
                rank_value = rank / max_rank
                return rank_value * 0.4 + (CARDS_PER_SUIT - suit_tricks) / CARDS_PER_SUIT * 0.6
        
        best_action = max(legal_actions, key=card_tricks_value)
        return best_action

    def compute_card_values(
        self,
        dd_table: DDTableCache,
        contract_denom: Denom,
        declarer: Player,
        legal_actions: list[int],
        current_player: Player,
        lead_suit: int = None,
        is_declarer_lead: bool = False
    ) -> list[float]:
        """Compute value for each legal card action.

        Uses DDS tricks to compute more accurate values than just rank-based.
        Higher value = better for the player.

        Args:
            dd_table: Precomputed DDTable for this deal.
            contract_denom: Trump suit.
            declarer: Declarer.
            legal_actions: List of legal card indices (0-51).
            current_player: Who's playing.
            lead_suit: Suit of the card led this trick (0-3), None if leading.
            is_declarer_lead: Whether declarer is leading this trick.

        Returns:
            List of float values, one per action in legal_actions.
            Higher = better for the player.
        """
        if not legal_actions:
            return []

        is_declarer_side = declarer.value % 2 == current_player.value % 2

        values = []
        for action in legal_actions:
            suit = action // CARDS_PER_SUIT
            rank = action % CARDS_PER_SUIT  # 0=2, 12=A

            suit_denom = SUIT_TO_DENOM[suit]
            suit_tricks = dd_table.tricks(suit_denom, declarer)

            max_rank = CARDS_PER_SUIT - 1  # 12
            
            if is_declarer_side:
                # 庄家方：更大牌更有价值，该花色能赢更多墩也更有价值
                rank_value = rank / max_rank
                dds_value = suit_tricks / CARDS_PER_SUIT
                value = rank_value * 0.3 + dds_value * 0.7
            else:
                # 防守方：
                # - 在庄家弱的花色（suit_tricks 低），出大牌可能赢得墩
                # - 在庄家强的花色（suit_tricks 高），出小牌更安全
                # 使用相对差值而不是绝对差值
                rank_value = rank / max_rank
                # 庄家弱的合约，防守方大牌价值高
                # 庄家强的合约，防守方小牌价值高
                dds_value = suit_tricks / CARDS_PER_SUIT
                # 反转：庄家赢墩少的花色，给防守方大牌更高价值
                # 但 rank_value 仍然奖励更大牌
                # 综合：rank_value + (1 - dds_value) * 0.5
                # 即 rank_value + (CARDS_PER_SUIT - suit_tricks) / CARDS_PER_SUIT * 0.5
                value = rank_value * 0.4 + (CARDS_PER_SUIT - suit_tricks) / CARDS_PER_SUIT * 0.6

            if lead_suit is not None:
                if suit == lead_suit:
                    value += 0.2
                elif suit == contract_denom.value:
                    value += 0.1

            values.append(value)

        max_val = max(values) if values else 1.0
        if max_val > 0:
            values = [v / max_val * 10 for v in values]

        return values
