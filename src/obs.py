"""Bridge AI 观测编码模块 - Part 1 (757 维观测结构)"""

import numpy as np
from enum import IntEnum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

# Import ALL canonical dimension constants from config
from config import (
    OBS_DIM, OFS_MATCH, OFS_CONTRACT, OFS_HANDS, OFS_FIRST,
    OFS_AUCTION, OFS_LEAD, OFS_PLAY,
    CONTRACT_DIM, HANDS_ENCODED_DIM, HANDS_DIM,
    HAND_CARD_DIM, FIRST_BIDDER_DIM, AUCTION_DIM, AUCTION_PADDED,
    AUCTION_SLOTS, AUCTION_ENTRY_DIM, LEAD_DIM, PLAY_DIM,
    MATCH_META_DIM, NUM_BID_ACTIONS, NUM_PLAY_ACTIONS,
    CARDS_PER_HAND,
)


# ============================================================================
# 枚举定义
# ============================================================================

class Suit(IntEnum):
    """花色枚举"""
    SPADES = 0  # ♠ 黑桃
    HEARTS = 1  # ♥ 红桃
    DIAMONDS = 2  # ♦ 方块
    CLUBS = 3  # ♣ 梅花


class Rank(IntEnum):
    """牌值枚举"""
    TWO = 0
    THREE = 1
    FOUR = 2
    FIVE = 3
    SIX = 4
    SEVEN = 5
    EIGHT = 6
    NINE = 7
    TEN = 8
    JACK = 9
    QUEEN = 10
    KING = 11
    ACE = 12


class Direction(IntEnum):
    """方向枚举"""
    NORTH = 0  # 北家
    EAST = 1   # 东家
    SOUTH = 2  # 南家
    WEST = 3   # 西家


class Vulnerable(IntEnum):
    """易失枚举（局况）"""
    NONE = 0   # 无局
    NORTH = 1  # 北家有局
    EAST = 2   # 东家有局
    SOUTH = 3  # 南家有局
    WEST = 4   # 西家有局


class VulnerableInfo(IntEnum):
    """易失信息枚举"""
    NONE = 0   # 无
    LEFT = 1   # 左家（东家）
    RIGHT = 2  # 右家（西家）


class DirectionInfo(IntEnum):
    """方向信息枚举"""
    NONE = 0   # 无
    LEFT = 1   # 左家（东家）
    RIGHT = 2  # 右家（西家）


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class BoardInfo:
    """牌局基础信息"""
    board_number: int = 0
    vul: Vulnerable = Vulnerable.NONE
    vul_info: int = 0
    declarer: Direction = Direction.NORTH
    contract: tuple = (None, None, None)
    hands: Dict[Direction, np.ndarray] = None

    def __post_init__(self):
        if self.hands is None:
            self.hands = {Direction.NORTH: np.zeros(53),
                         Direction.EAST: np.zeros(53),
                         Direction.SOUTH: np.zeros(53),
                         Direction.WEST: np.zeros(53)}


@dataclass
class ContractInfo:
    """庄约定信息"""
    vul: Vulnerable = Vulnerable.NONE
    vul_info: int = 0
    declarer: Direction = Direction.NORTH
    contract: tuple = (None, None, None)


@dataclass
class HandInfo:
    """手牌信息"""
    suit: Suit = Suit.SPADES
    rank: Rank = Rank.ACE
    led: bool = False
    led_pos: int = 0


@dataclass
class BidInfo:
    """叫牌信息"""
    bidder: int = 0
    bid: tuple = (None, None)


@dataclass
class LeadInfo:
    """领出信息"""
    suit: Suit = Suit.SPADES
    rank: Rank = Rank.ACE
    led_by: int = 0
    led_pos: int = 0


@dataclass
class PlayInfo:
    """跟牌信息"""
    card: tuple = (None, None)


# ============================================================================
# 观测编码函数
# ============================================================================

def encode_contract(contract: tuple) -> np.ndarray:
    """编码庄约定信息 (CONTRACT_DIM 维)"""
    # CONTRACT_DIM = 208 — defined in config.py
    vul = Vulnerable.NONE
    vul_info = 0
    
    # 根据花色确定局况
    if contract[0] == Suit.DIAMONDS:
        vul = Vulnerable.EAST  # 方块：东家有局
    elif contract[0] == Suit.CLUBS:
        vul = Vulnerable.WEST  # 梅花：西家有局
    elif contract[0] == Suit.HEARTS:
        vul = Vulnerable.NORTH  # 红桃：北家有局
    elif contract[0] == Suit.SPADES:
        vul = Vulnerable.SOUTH  # 黑桃：南家有局
    
    # 局况编码 (1 维)
    vul_arr = np.array([vul.value])
    # 局况信息 (1 维)
    vul_info_arr = np.array([vul_info])
    
    # 庄家位置 one-hot (4 维)
    declarer_arr = np.zeros(4)
    declarer_arr[Direction.NORTH.value] = 1
    
    # 合约等级 + 花色 one-hot (2+5=7 维: level, nt)
    contract_level = contract[0] if contract[0] else 0  # 1-7
    contract_nt = contract[1] if contract[1] else 0     # 0=有将, 1=无将
    # 等级 one-hot (7 维)
    level_arr = np.zeros(7)
    if 1 <= contract_level <= 7:
        level_arr[contract_level - 1] = 1
    # 无将标记 (1 维)
    nt_arr = np.array([contract_nt])
    # 花色 one-hot (4 维)
    suit_arr = np.zeros(4)
    if contract[0] is not None and contract[0] != 0:
        suit_arr[contract[0] - 1] = 1
    
    # 合约编码: level(7) + nt(1) + suit(4) = 12 维
    contract_detail_arr = np.concatenate([level_arr, nt_arr, suit_arr])
    
    # 定约方标记 (4 维) - 默认北家
    declarer_side_arr = np.zeros(4)
    declarer_side_arr[Direction.NORTH.value] = 1
    
    # 加倍信息 (3 维): 无加倍, 加倍, 再加倍
    double_arr = np.array([1, 0, 0])
    
    # 合约完成情况 (3 维): 待定, 完成, 宕
    result_arr = np.array([1, 0, 0])
    
    result = np.concatenate([
        vul_arr, vul_info_arr, declarer_arr, contract_detail_arr,
        declarer_side_arr, double_arr, result_arr
    ])  # 1+1+4+12+4+3+3 = 28 维
    
    # padding 到 208 维
    padded = np.zeros(CONTRACT_DIM)
    padded[:len(result)] = result
    return padded


def encode_hands(hands: Dict[Direction, np.ndarray]) -> np.ndarray:
    """编码手牌信息 (228 维)"""
    # 对手牌各 53 维: 13张牌 × 4 花色/点数 编码
    # 取前 52 维 (去掉末尾 padding)
    hands_arr = np.zeros((4, HANDS_ENCODED_DIM))
    for i, hand in enumerate(hands.values()):
        if hand is not None and len(hand) > 0:
            hand_flat = hand.flatten() if hasattr(hand, 'flatten') else np.array(hand)
            hands_arr[i] = hand_flat[:HANDS_ENCODED_DIM]
    flat = hands_arr.flatten()  # 4 * HANDS_ENCODED_DIM
    padded = np.zeros(HANDS_DIM)
    padded[:len(flat)] = flat
    return padded


def encode_auction(auction: List[BidInfo]) -> np.ndarray:
    """编码叫牌历史 (AUCTION_PADDED 维)"""
    auction_arr = np.zeros(AUCTION_DIM)  # AUCTION_SLOTS × AUCTION_ENTRY_DIM
    for i, bid in enumerate(auction):
        if bid is not None:
            bid_arr = np.zeros(AUCTION_ENTRY_DIM)
            bid_arr[0] = bid.bidder
            bid_arr[1] = 0 if bid.bid[0] is None else bid.bid[0]
            bid_arr[2] = 0 if bid.bid[1] is None else bid.bid[1]
            if i < AUCTION_SLOTS:
                auction_arr[i*AUCTION_ENTRY_DIM:(i+1)*AUCTION_ENTRY_DIM] = bid_arr
    padded = np.zeros(AUCTION_PADDED)
    padded[:AUCTION_DIM] = auction_arr
    return padded


def encode_lead(lead: LeadInfo) -> np.ndarray:
    """编码领出信息 (LEAD_DIM 维)"""
    # LEAD_DIM = 28
    lead_arr = np.zeros(LEAD_DIM)
    lead_arr[0] = lead.led_by
    lead_arr[1] = lead.led_pos
    lead_arr[2] = lead.suit.value
    lead_arr[3] = lead.rank.value
    return lead_arr


def encode_play(play: PlayInfo) -> np.ndarray:
    """编码跟牌信息 (PLAY_DIM 维)"""
    # PLAY_DIM = 53
    play_arr = np.zeros(PLAY_DIM)
    play_arr[0] = play.card[0] if play.card[0] is not None else 0
    play_arr[1] = play.card[1] if play.card[1] is not None else 0
    return play_arr.flatten()


# ============================================================================
# 757 维观测生成
# ============================================================================

def board_to_obs(board: BoardInfo, auction: List[BidInfo], lead: LeadInfo, play: PlayInfo) -> np.ndarray:
    """
    将牌局数据转换为 OBS_DIM 维观测向量

    维度分布 (定义见 config.py):
    0-11:      match metadata (MATCH_META_DIM)
    12-219:    contract (CONTRACT_DIM)
    220-447:   hands (HANDS_DIM)
    448-476:   first bidder (FIRST_BIDDER_DIM)
    477-675:   auction (AUCTION_PADDED)
    676-703:   lead (LEAD_DIM)
    704-756:   play (PLAY_DIM)
    """
    obs = np.zeros(OBS_DIM)

    # 0-11: 比赛元数据 (MATCH_META_DIM 维)
    board_number = 0
    table_count = 1
    round_number = 0
    total_rounds = 16
    obs[OFS_MATCH + 0] = board_number
    obs[OFS_MATCH + 1] = table_count
    obs[OFS_MATCH + 2] = round_number
    obs[OFS_MATCH + 3] = total_rounds

    # 12-219: 庄约定 (CONTRACT_DIM 维)
    contract = encode_contract(board.contract)
    obs[OFS_CONTRACT:OFS_CONTRACT + CONTRACT_DIM] = contract

    # 220-447: 手牌 (HANDS_DIM 维)
    hands = encode_hands(board.hands)
    obs[OFS_HANDS:OFS_HANDS + HANDS_DIM] = hands

    # 448-476: 第一家叫牌 (FIRST_BIDDER_DIM 维)
    first_bidder = auction[0] if auction else BidInfo()
    first_bidder_section = np.zeros(FIRST_BIDDER_DIM)
    first_bidder_section[0] = first_bidder.bidder
    first_bidder_section[1:3] = np.array([0, 0])
    obs[OFS_FIRST:OFS_FIRST + FIRST_BIDDER_DIM] = first_bidder_section

    # 477-675: 叫牌历史 (AUCTION_PADDED 维)
    auction_section = encode_auction(auction)
    obs[OFS_AUCTION:OFS_AUCTION + AUCTION_PADDED] = auction_section

    # 676-703: 第一墩领出 (LEAD_DIM 维)
    lead_section = encode_lead(lead)
    obs[OFS_LEAD:OFS_LEAD + LEAD_DIM] = lead_section

    # 704-756: 后续打牌 (PLAY_DIM 维)
    play_section = encode_play(play)
    obs[OFS_PLAY:OFS_PLAY + PLAY_DIM] = play_section

    return obs


def obs_to_vis(obs: np.ndarray) -> Dict[str, Any]:
    """将观测解码为可视化信息"""
    return {
        'board': {
            'board_number': int(obs[OFS_MATCH + 0]),
            'table_count': int(obs[OFS_MATCH + 1]),
            'round_number': int(obs[OFS_MATCH + 2]),
            'total_rounds': int(obs[OFS_MATCH + 3])
        },
        'contract': {
            'vul': int(obs[OFS_CONTRACT + 0]),
            'vul_info': int(obs[OFS_CONTRACT + 1]),
            'declarer': Direction(obs[OFS_CONTRACT + 2]),
            'contract': (int(obs[OFS_CONTRACT + 3]), int(obs[OFS_CONTRACT + 4]))
        },
        'hands': obs[OFS_HANDS:OFS_HANDS + HANDS_DIM].tolist(),
        'auction': obs[OFS_AUCTION:OFS_AUCTION + AUCTION_PADDED].tolist(),
        'lead': {
            'led_by': int(obs[OFS_LEAD + 0]),
            'led_pos': int(obs[OFS_LEAD + 1]),
            'suit': int(obs[OFS_LEAD + 2]),
            'rank': int(obs[OFS_LEAD + 3])
        },
        'play': {
            'card': (int(obs[OFS_PLAY + 0]), int(obs[OFS_PLAY + 1]))
        }
    }
