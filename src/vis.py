"""
桥牌 AI 视觉信息模块
视觉维度：与 obs 相同的 OBS_DIM 维
"""

import numpy as np
from enum import IntEnum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


# 导入 obs 模块的枚举和数据类
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from obs import (
    Suit, Rank, Direction, Vulnerable, VulnerableInfo, DirectionInfo,
    BoardInfo, ContractInfo, HandInfo, BidInfo, LeadInfo, PlayInfo
)
from config import (
    OBS_DIM, MATCH_META_DIM, CONTRACT_DIM, HANDS_DIM,
    HANDS_ENCODED_DIM, HAND_CARD_DIM, FIRST_BIDDER_DIM,
    AUCTION_DIM, AUCTION_PADDED, AUCTION_SLOTS, AUCTION_ENTRY_DIM,
    LEAD_DIM, PLAY_DIM, NUM_BID_ACTIONS, NUM_PLAY_ACTIONS,
    OFS_MATCH, OFS_CONTRACT, OFS_HANDS, OFS_FIRST,
    OFS_AUCTION, OFS_LEAD, OFS_PLAY, CARDS_PER_HAND,
)


@dataclass
class VisualBoardInfo:
    """视觉牌局元数据"""
    board_number: int = 0
    table_count: int = 1
    round_number: int = 0
    total_rounds: int = 1


@dataclass
class VisualContractInfo:
    """视觉庄约定信息"""
    vul: Vulnerable = Vulnerable.NONE
    vul_info: VulnerableInfo = VulnerableInfo.NONE
    declarer: Direction = Direction.NORTH
    contract: tuple = (None, None, None)


@dataclass
class VisualHandInfo:
    """视觉手牌信息"""
    suit: Suit = Suit.SPADES
    rank: Rank = Rank.ACE
    led: bool = False
    led_pos: int = 0


@dataclass
class VisualBidInfo:
    """视觉叫牌信息"""
    bidder: int = 0
    bid: tuple = (None, None)


@dataclass
class VisualLeadInfo:
    """视觉领出信息"""
    suit: Suit = Suit.SPADES
    rank: Rank = Rank.ACE
    led_by: int = 0
    led_pos: int = 0


@dataclass
class VisualPlayInfo:
    """视觉跟牌信息"""
    card: tuple = (None, None)


def encode_visual_contract(contract: tuple) -> np.ndarray:
    """编码视觉庄约定信息 (8 维)"""
    vul = Vulnerable.NONE
    vul_info = VulnerableInfo.NONE
    
    # 根据花色确定局况
    if contract[0] == Suit.DIAMONDS:
        vul = Vulnerable.EAST  # 方块：东家有局
    elif contract[0] == Suit.CLUBS:
        vul = Vulnerable.WEST  # 梅花：西家有局
    elif contract[0] == Suit.HEARTS:
        vul = Vulnerable.NORTH  # 红桃：北家有局
    elif contract[0] == Suit.SPADES:
        vul = Vulnerable.SOUTH  # 黑桃：南家有局
    
    # 根据局况确定局况信息
    if vul in [Vulnerable.NORTH, Vulnerable.EAST]:  # 北家或东家有局
        vul_info = VulnerableInfo.LEFT
    elif vul in [Vulnerable.SOUTH, Vulnerable.WEST]:  # 南家或西家有局
        vul_info = VulnerableInfo.RIGHT
    else:
        vul_info = VulnerableInfo.NONE
    
    vul_arr = np.array([vul.value])
    vul_info_arr = np.array([vul_info.value])
    
    return np.concatenate([vul_arr, vul_info_arr])


def encode_direction(direction: Direction) -> np.ndarray:
    """编码方向 (1 维)"""
    return np.array([direction.value])


def encode_contract_full(contract: tuple) -> np.ndarray:
    """编码完整视觉庄约定信息 (208 维)"""
    vul_arr = np.array([Vulnerable.NONE.value])
    vul_info_arr = np.array([VulnerableInfo.NONE.value])
    declarer_arr = np.array([Direction.NORTH.value])
    contract_arr = np.array([Suit.SPADES.value, Rank.ACE.value])
    
    hands = np.zeros((4, 53))
    
    return np.concatenate([
        vul_arr,
        vul_info_arr,
        np.array([Direction.NORTH.value]),
        contract_arr,
        hands
    ])


def encode_hand(hand: VisualHandInfo) -> np.ndarray:
    """编码视觉手牌 (53 维)"""
    suit = hand.suit
    rank = hand.rank
    led = hand.led
    led_pos = hand.led_pos
    
    led_arr = np.array([led.value])
    led_pos_arr = np.array([led_pos])
    suit_arr = np.array([suit.value])
    rank_arr = np.array([rank.value])
    
    rank_encode = np.zeros(48)
    for i in range(48):
        if i == rank.value:
            rank_encode[i] = 1
    
    return np.concatenate([
        led_arr,
        led_pos_arr,
        suit_arr,
        rank_arr,
        rank_encode
    ])


def encode_vis_auction(auction: List[VisualBidInfo]) -> np.ndarray:
    """编码视觉叫牌历史 (272 维) — 仅用于 vis.py"""
    auction_arr = []
    for bid in auction:
        bid_arr = np.array([bid.bidder.value])
        bid_arr = np.concatenate([
            bid_arr,
            np.array([Suit.SPADES.value]),
            np.array([Rank.ACE.value])
        ])
        auction_arr.append(bid_arr)
    return np.array(auction_arr)


def encode_vis_lead(lead: VisualLeadInfo) -> np.ndarray:
    """编码视觉领出信息 (128 维) — 仅用于 vis.py"""
    lead_arr = np.zeros(128)
    lead_arr[:1] = np.array([lead.led_by.value])
    lead_arr[1:2] = np.array([lead.led_pos.value])
    lead_arr[2:3] = np.array([lead.suit.value])
    lead_arr[3:4] = np.array([lead.rank.value])
    lead_arr[4:5] = np.array([lead.suit.value])
    lead_arr[5:6] = np.array([lead.rank.value])
    return lead_arr


def encode_vis_play(play: VisualPlayInfo) -> np.ndarray:
    """编码视觉跟牌信息 (128 维) — 仅用于 vis.py"""
    play_arr = np.zeros(128)
    play_arr[:1] = np.array([play.bidder.value])
    play_arr[1:2] = np.array([play.card[0].value])
    play_arr[2:3] = np.array([play.card[1].value])
    return play_arr


def encode_vis(board: VisualBoardInfo, contract: VisualContractInfo, 
               hands: List[VisualHandInfo], auction: List[VisualBidInfo],
               lead: VisualLeadInfo, play: VisualPlayInfo) -> np.ndarray:
    """
    编码完整视觉观测 (OBS_DIM 维)
    
    结构与 obs 相同 (定义见 config.py):
    0-11:     match metadata (MATCH_META_DIM)
    12-219:   vul, vul_info, declarer, contract (CONTRACT_DIM)
    220-447:  hands (HANDS_DIM) - 4 players × HANDS_ENCODED_DIM
    448-476:  first_bidder (FIRST_BIDDER_DIM)
    477-675:  auction (AUCTION_PADDED)
    676-703:  lead (LEAD_DIM)
    704-756:  play (PLAY_DIM)
    """
    # 0-11: 比赛元数据 (MATCH_META_DIM 维)
    obs = np.zeros(OBS_DIM)
    obs[OFS_MATCH + 0] = board.board_number
    obs[OFS_MATCH + 1] = board.table_count
    obs[OFS_MATCH + 2] = board.round_number
    obs[OFS_MATCH + 3] = board.total_rounds
    
    # 12-219: 庄约定 (CONTRACT_DIM 维)
    vul_arr = np.array([contract.vul.value])
    vul_info_arr = np.array([contract.vul_info.value])
    declarer_arr = np.array([contract.declarer.value])
    contract_arr = np.array([contract.contract[0].value, contract.contract[1].value])
    
    hands_arr = np.zeros((4, HAND_CARD_DIM))
    for i, hand in enumerate(hands):
        hands_arr[i] = encode_hand(hand)
    
    contract_section = np.concatenate([
        vul_arr,
        vul_info_arr,
        declarer_arr,
        contract_arr,
        hands_arr.flatten()
    ])
    obs[OFS_CONTRACT:OFS_CONTRACT + CONTRACT_DIM] = contract_section
    
    # 220-447: 手牌 (HANDS_DIM 维)
    hands_section = np.zeros((4, HANDS_ENCODED_DIM))
    for i, hand in enumerate(hands):
        hands_section[i] = encode_hand(hand)
    obs[OFS_HANDS:OFS_HANDS + 4 * HANDS_ENCODED_DIM] = hands_section
    
    # 448-476: 第一家叫牌 (FIRST_BIDDER_DIM 维)
    first_bidder = auction[0]
    first_bidder_section = np.zeros(FIRST_BIDDER_DIM)
    first_bidder_section[:1] = np.array([first_bidder.bidder.value])
    first_bidder_section[1:2] = np.array([first_bidder.bid[0].value])
    first_bidder_section[2:3] = np.array([first_bidder.bid[1].value])
    obs[OFS_FIRST:OFS_FIRST + FIRST_BIDDER_DIM] = first_bidder_section
    
    # 477-675: 叫牌历史 (AUCTION_PADDED 维)
    auction_section = np.zeros(AUCTION_DIM)
    for i, bid in enumerate(auction[1:]):
        bid_arr = np.zeros(AUCTION_ENTRY_DIM)
        bid_arr[0] = bid.bidder.value
        bid_arr[1] = bid.bid[0].value
        bid_arr[2] = bid.bid[1].value
        if i < AUCTION_SLOTS:
            auction_section[i*AUCTION_ENTRY_DIM:(i+1)*AUCTION_ENTRY_DIM] = bid_arr
    obs[OFS_AUCTION:OFS_AUCTION + AUCTION_PADDED] = auction_section
    
    # 676-703: 第一墩打牌 (LEAD_DIM 维)
    lead_section = np.zeros(LEAD_DIM)
    lead_section[:1] = np.array([lead.led_by.value])
    lead_section[1:2] = np.array([lead.led_pos.value])
    lead_section[2:3] = np.array([lead.suit.value])
    lead_section[3:4] = np.array([lead.rank.value])
    obs[OFS_LEAD:OFS_LEAD + LEAD_DIM] = lead_section
    
    # 704-756: 后续打牌 (PLAY_DIM 维)
    auction_leads_section = np.zeros(PLAY_DIM)
    auction_leads_section[:1] = np.array([play.bidder.value])
    auction_leads_section[1:2] = np.array([play.card[0].value])
    auction_leads_section[2:3] = np.array([play.card[1].value])
    obs[OFS_PLAY:OFS_PLAY + PLAY_DIM] = auction_leads_section
    
    return obs.flatten()


def decode_vis(obs: np.ndarray) -> Dict[str, Any]:
    """解码视觉观测数据 — 委托至 obs.obs_to_vis"""
    from obs import obs_to_vis
    return obs_to_vis(obs)
