#!/usr/bin/env python3
"""⚠️ LEGACY: Alternative full bridge environment — not used.

This file contains an experimental ``BridgeEnv`` and ``Deal`` class that
duplicate functionality from ``env_core.py`` (which is the active
environment used by ``predictor.py``). Kept for reference only.

- ``BridgeEnv`` here is NOT the same as ``env_core.BridgeEnv``.
- Obs encoding here uses a trivial ``np.zeros(obs_dim)`` placeholder.
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field

from endplay import generate_deal as endplay_generate_deal
from endplay.types import Player, Denom

from obs import (
    Suit, Rank, Direction, Vulnerable,
    encode_contract, encode_hands, encode_auction,
    encode_lead, encode_play, board_to_obs
)


@dataclass
class Card:
    """单张牌"""
    suit: Suit
    rank: int
    led: bool = False  # 是否被领出

    def __repr__(self):
        return f"Card({self.suit.name}, {self.rank})"


@dataclass
class Deal:
    """完整的牌局状态"""
    hands: Dict[Direction, List[Card]] = field(default_factory=dict)
    vul: Vulnerable = Vulnerable.NONE
    declarer: Direction = Direction.NORTH
    contract: Tuple[Optional[Suit], int, int] = (None, 0, 0)
    auction: List[Dict] = field(default_factory=list)
    lead: Optional[Card] = None
    play: List[Card] = field(default_factory=list)
    round_num: int = 0
    _turn: int = 0
    _bid_phase: bool = True

    def __post_init__(self):
        # 初始化空手牌
        for d in Direction:
            if d not in self.hands:
                self.hands[d] = []

    def _encode(self) -> np.ndarray:
        """编码牌局状态"""
        # 简化版本：只编码基本状态
        obs = np.zeros(757)  # LEGACY: hardcoded obs_dim
        obs[0] = self.round_num
        return obs

    def _random_deal(self):
        """生成随机发牌（使用 Endplay）"""
        # 使用 Endplay 生成随机牌局
        deal = endplay_generate_deal()

        # 定义玩家映射
        player_map = {
            Player.north: Direction.NORTH,
            Player.east: Direction.EAST,
            Player.south: Direction.SOUTH,
            Player.west: Direction.WEST
        }

        # 定义花色映射
        suit_map = {
            Denom.spades: Suit.SPADES,
            Denom.hearts: Suit.HEARTS,
            Denom.diamonds: Suit.DIAMONDS,
            Denom.clubs: Suit.CLUBS
        }

        # 将 Endplay 牌局转换为自定义格式
        for endplay_player, direction in player_map.items():
            hand = []
            for suit in [Denom.spades, Denom.hearts, Denom.diamonds, Denom.clubs]:
                for card in deal.hand(endplay_player).cards_of(suit):
                    # Endplay rank: 0=2, 1=3, ..., 12=A
                    rank = card.rank.value + 2
                    hand.append(Card(suit_map[suit], rank))
            self.hands[direction] = hand

    def next_player(self, current: Direction) -> Direction:
        """下一个玩家"""
        indices = list(Direction)
        idx = indices.index(current)
        return indices[(idx + 1) % 4]

    def is_valid_bid(self, bid: Dict) -> bool:
        """检查叫牌是否合法"""
        # 简化版本：允许任何叫牌
        return True

    def play_card(self, player: Direction, card: Card) -> bool:
        """跟牌"""
        if card not in self.hands[player]:
            return False
        card.led = True
        self.hands[player].remove(card)
        self.play.append(card)
        return True

    def end_round(self):
        """结束回合"""
        self.round_num += 1
        self._turn = 0
        self._bid_phase = True

    def reset(self):
        """重置牌局"""
        self._random_deal()
        self.auction.clear()
        self.play.clear()
        self.round_num = 0
        self._turn = 0

    def render(self):
        """渲染牌局状态"""
        print(f"\n=== 牌局 {self.round_num} ===")
        print(f"庄家: {self.declarer.name}, 合约: {self.contract}")
        print(f"叫牌: {self.auction}")
        print(f"领出: {self.lead}")
        print(f"打牌: {self.play}")
        for d in Direction:
            print(f"  {d.name}: {len(self.hands[d])} 张牌")


class BridgeEnv:
    """完整的桥牌训练环境"""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.deal = Deal()

    def reset(self) -> Dict:
        """重置环境"""
        self.deal.reset()
        return self.deal._encode()

    def step(self, action: Dict) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步动作"""
        # 简化版本：只处理基本动作
        return self.deal._encode(), 0.0, False, {}

    def render(self):
        """渲染环境状态"""
        self.deal.render()


if __name__ == '__main__':
    # 测试环境
    env = BridgeEnv(seed=42)
    state = env.reset()
    print(f"初始状态形状: {state.shape}")
    env.render()