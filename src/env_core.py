"""Bridge environment using Endplay Board — full bidding + play with DDS scoring.

State machine: reset → bidding (N/E/S/W sequential) → play → done with reward.

This version uses Endplay Board to manage all auction and play state.
"""

from __future__ import annotations
import random as _random
import numpy as np
from typing import List, Optional, Tuple

from endplay import Deal, Board, generate_deal
from endplay.types import Player, Vul, Denom, Rank, Bid as EndplayBid

from config import EnvConfig, OFS_MATCH, OFS_CONTRACT, OFS_HANDS, OFS_FIRST, OFS_AUCTION, OFS_PLAY
from rewards import RewardsModule


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_BID = EnvConfig.num_bid_actions   # 38
NUM_PLAY = EnvConfig.num_play_actions  # 52
NUM_SUITS = 4
CARDS_PER_SUIT = NUM_PLAY // NUM_SUITS         # 13

# Bid encoding constants
PASS_ACTION = 35
DOUBLE_ACTION = 36
REDOUBLE_ACTION = 37

# Rank to idx (matches config)
_RANK_TO_IDX = {Rank.R2: 0, Rank.R3: 1, Rank.R4: 2, Rank.R5: 3,
                Rank.R6: 4, Rank.R7: 5, Rank.R8: 6, Rank.R9: 7,
                Rank.RT: 8, Rank.RJ: 9, Rank.RQ: 10, Rank.RK: 11,
                Rank.RA: 12}

_SUIT_TO_IDX = {Denom.spades: 0, Denom.hearts: 1, Denom.diamonds: 2, Denom.clubs: 3}


# ---------------------------------------------------------------------------
# Bid encoding/decoding
# ---------------------------------------------------------------------------
BID_LEVELS = list(range(1, 8))   # 1..7
BID_DENOMS = [Denom.clubs, Denom.diamonds, Denom.hearts, Denom.spades, Denom.nt]

LEVEL_DENOM_TO_ACTION: dict = {}
for level in BID_LEVELS:
    for denom in BID_DENOMS:
        idx = (level - 1) * 5 + BID_DENOMS.index(denom)
        LEVEL_DENOM_TO_ACTION[(level, denom)] = idx


def action_to_bid(action_idx: int) -> Tuple[str, Optional[int], Optional[Denom]]:
    """Convert action index (0-37) to (string, level, denom)."""
    if action_idx == PASS_ACTION:
        return ("Pass", None, None)
    if action_idx == DOUBLE_ACTION:
        return ("Double", None, None)
    if action_idx == REDOUBLE_ACTION:
        return ("Redouble", None, None)
    level = action_idx // 5 + 1
    denom = BID_DENOMS[action_idx % 5]
    return (f"{level}{denom.abbr}", level, denom)


def bid_to_action(level: int, denom: Denom) -> int:
    """Convert (level, denom) to action index 0-34."""
    return (level - 1) * 5 + BID_DENOMS.index(denom)


def action_idx_to_endplay_bid(action_idx: int) -> EndplayBid:
    """Convert action index (0-37) to Endplay Bid object."""
    if action_idx == PASS_ACTION:
        return EndplayBid("Pass")
    if action_idx == DOUBLE_ACTION:
        return EndplayBid("X")
    if action_idx == REDOUBLE_ACTION:
        return EndplayBid("XX")
    level = action_idx // 5 + 1
    denom_idx = action_idx % 5
    denom_names = ['C', 'D', 'H', 'S', 'NT']
    return EndplayBid(f"{level}{denom_names[denom_idx]}")


# ---------------------------------------------------------------------------
# BiddingBox — validate and track auction state using Endplay Board
# ---------------------------------------------------------------------------
class BiddingBox:
    """Track the state of an auction and validate bids using Endplay Board."""

    def __init__(self, dealer: Player = Player.north):
        self.dealer = dealer
        self.history: List[Tuple[Player, int]] = []  # (player, action_idx)
        self._endplay_bids: List[EndplayBid] = []
        self._contract = None
        self._is_doubled = False
        self._is_redoubled = False

    def reset(self, dealer: Player = Player.north):
        self.dealer = dealer
        self.history.clear()
        self._endplay_bids.clear()
        self._contract = None
        self._is_doubled = False
        self._is_redoubled = False

    def is_legal(self, action_idx: int, player: Player) -> bool:
        """Check if a bid action is legal using Endplay logic."""
        if action_idx == PASS_ACTION:
            return True
        
        if action_idx == DOUBLE_ACTION:
            if len(self._endplay_bids) == 0:
                return False
            for bid in reversed(self._endplay_bids):
                bid_str = str(bid)
                if bid_str != 'P':
                    if bid_str not in ['X', 'XX']:
                        last_bidder_idx = len(self._endplay_bids) - 1 - list(reversed(self._endplay_bids)).index(bid)
                        last_bidder = self.dealer.next(last_bidder_idx)
                        if (player.value - last_bidder.value) % 2 == 1:
                            return not self._is_doubled
                    return False
            return False
        
        if action_idx == REDOUBLE_ACTION:
            if not self._is_doubled or self._is_redoubled:
                return False
            for i, bid in enumerate(self._endplay_bids):
                if str(bid) == 'X':
                    doubler = self.dealer.next(i)
                    if (player.value - doubler.value) % 2 == 1:
                        return True
            return False
        
        # Contract bid: check if higher than last bid
        level = action_idx // 5 + 1
        denom_idx = action_idx % 5
        
        for bid in reversed(self._endplay_bids):
            bid_str = str(bid)
            if bid_str not in ['P', 'X', 'XX']:
                cb = bid.as_contract()
                if cb:
                    # Compare heights
                    # Bridge denom rank: C=0, D=1, H=2, S=3, NT=4
                    # Endplay denom: S=0, H=1, D=2, C=3, NT=4
                    endplay_to_bridge = {0: 3, 1: 2, 2: 1, 3: 0, 4: 4}
                    current_height = level * 10 + denom_idx
                    last_height = cb.level * 10 + endplay_to_bridge.get(cb.denom.value, 4)
                    return current_height > last_height
        
        return True

    def record(self, action_idx: int, player: Player):
        """Record a bid action using Endplay."""
        if not self.is_legal(action_idx, player):
            raise ValueError(f"Illegal bid {action_idx} by {player}")
        
        self.history.append((player, action_idx))
        
        endplay_bid = action_idx_to_endplay_bid(action_idx)
        self._endplay_bids.append(endplay_bid)
        
        if action_idx == DOUBLE_ACTION:
            self._is_doubled = True
        elif action_idx == REDOUBLE_ACTION:
            self._is_redoubled = True
        elif action_idx < 35:
            self._is_doubled = False
            self._is_redoubled = False
        
        self._update_contract()

    def _update_contract(self):
        """Update contract from Endplay."""
        from endplay.types import Contract
        try:
            self._contract = Contract.from_auction(self.dealer, self._endplay_bids)
        except Exception:
            self._contract = None

    @property
    def is_auction_over(self) -> bool:
        """Check if auction is over using Endplay logic."""
        if len(self._endplay_bids) < 4:
            return False
        
        if len(self._endplay_bids) == 4 and all(str(b) == 'P' for b in self._endplay_bids):
            return True
        
        if len(self._endplay_bids) >= 4:
            last_three = [str(b) for b in self._endplay_bids[-3:]]
            if all(s == 'P' for s in last_three):
                for bid in self._endplay_bids[:-3]:
                    if str(bid) not in ['P', 'X', 'XX']:
                        return True
        
        return False

    @property
    def contract_level(self) -> int:
        """Get contract level from Endplay Contract."""
        if self._contract is None or self._contract.is_passout():
            return 0
        return self._contract.level

    @property
    def contract_denom(self) -> Denom:
        """Get contract denom from Endplay Contract."""
        if self._contract is None or self._contract.is_passout():
            return Denom.clubs
        return self._contract.denom

    @property
    def declarer(self) -> Optional[Player]:
        """Get declarer from Endplay Contract."""
        if self._contract is None or self._contract.is_passout():
            return None
        return self._contract.declarer

    @property
    def is_doubled(self) -> bool:
        return self._is_doubled and not self._is_redoubled

    @property
    def is_redoubled(self) -> bool:
        return self._is_redoubled

    def doubled_status(self) -> int:
        """0=undoubled, 1=doubled, 2=redoubled."""
        if self._is_redoubled:
            return 2
        if self._is_doubled:
            return 1
        return 0


# ---------------------------------------------------------------------------
# Helper: random deal
# ---------------------------------------------------------------------------
def random_deal() -> Deal:
    """Return a random Deal using Endplay's built-in function."""
    return generate_deal()


# ---------------------------------------------------------------------------
# BridgeEnv — full bridge environment using Endplay Board
# ---------------------------------------------------------------------------
class BridgeEnv:
    """Full bridge environment with bidding + play + DDS scoring reward.

    Uses Endplay Board to manage auction and play state.
    """

    def __init__(self, cfg=None, seed: int = 42):
        self.cfg = cfg if cfg is not None else EnvConfig()
        self.rewards = RewardsModule()
        self.board: Optional[Board] = None
        self.deal: Optional[Deal] = None
        self.bidding = BiddingBox()
        self._turn_idx = 0
        _random.seed(seed)
        self.reset()

    @property
    def dealer(self) -> Player:
        """Get dealer from Board."""
        if self.board is not None:
            return self.board.dealer
        return Player.north

    @property
    def vul(self) -> Vul:
        """Get vulnerability from Board."""
        if self.board is not None:
            return self.board.vul
        return Vul.none

    @property
    def play_state(self):
        """Alias for _play_state for backward compatibility."""
        return getattr(self, '_play_state', None)

    def _current_player(self) -> Player:
        """Get current player based on turn index."""
        if self.phase == 'bidding':
            return self.dealer.next(self._turn_idx % 4)
        elif hasattr(self, '_play_state') and self._play_state is not None:
            return self._play_state._current_player
        return Player.north

    def _get_hand(self, player: Player):
        """Get hand for a player."""
        if self.deal is None:
            return []
        _MAP = {Player.north: self.deal.north,
                Player.east: self.deal.east,
                Player.south: self.deal.south,
                Player.west: self.deal.west}
        return _MAP[player]

    def _encode_obs(self) -> np.ndarray:
        """Encode current state into 780-dim observation (757-dim transmitted)."""
        obs = np.zeros(self.cfg.obs_dim, dtype=np.float32)

        OFS_LEAD = 728  # 内部使用，不传输
        PLAY_DIM = NUM_PLAY

        # ---- 0: Vulnerability ----
        obs[OFS_MATCH] = float(getattr(self.vul, 'value', 0))

        # ---- 12-219: Contract ----
        if self.bidding.is_auction_over and self.bidding.contract_level > 0:
            lvl = self.bidding.contract_level
            denom = self.bidding.contract_denom
            decl = self.bidding.declarer
            dbl = self.bidding.doubled_status()
            
            obs[OFS_CONTRACT + lvl - 1] = 1.0
            denom_idx = _SUIT_TO_IDX.get(denom, 4)
            if denom_idx < 4:
                obs[OFS_CONTRACT + 7 + denom_idx] = 1.0
            else:
                obs[OFS_CONTRACT + 7 + 4] = 1.0
            obs[OFS_CONTRACT + 12 + decl.value] = 1.0
            obs[OFS_CONTRACT + 16] = float(dbl)

        # ---- 220-447: Hands ----
        # 叫牌阶段：编码所有玩家的原始手牌（用于训练数据）
        # 出牌阶段：编码所有玩家的原始手牌（用于计算剩余手牌）+ 当前玩家和 dummy 的剩余手牌
        for pl in [Player.north, Player.east, Player.south, Player.west]:
            hand = self._get_hand(pl)
            hand_offset = OFS_HANDS + pl.value * NUM_PLAY

            # 编码原始手牌（所有玩家）
            for card in hand:
                if card is None:
                    continue
                s = _SUIT_TO_IDX.get(card.suit, 0)
                r = _RANK_TO_IDX.get(card.rank, 0)
                obs[hand_offset + s * CARDS_PER_SUIT + r] = 1.0

            # 额外编码当前玩家和 dummy 的剩余手牌（用于训练模型）
            if (hasattr(self, '_play_state') and self._play_state is not None and
                self.phase == 'play'):
                current = self._current_player()
                is_lead = (self._play_state._trick_no == 0 and
                          len(self._play_state._current_trick_cards) == 0)
                dummy_open = not is_lead or self._play_state._trick_no > 0
                if pl == current or (pl == self._play_state.dummy and dummy_open):
                    remaining = self._get_remaining_cards(pl)
                    for card in remaining:
                        if card is None:
                            continue
                        s = _SUIT_TO_IDX.get(card.suit, 0)
                        r = _RANK_TO_IDX.get(card.rank, 0)
                        obs[hand_offset + s * CARDS_PER_SUIT + r] = 1.0

        # ---- 448: Current player ----
        obs[OFS_FIRST] = float(self._current_player().value)

        # ---- 477-675: Auction history ----
        for i, (pl, act) in enumerate(self.bidding.history):
            if i >= 199:
                break
            obs[OFS_AUCTION + i] = float(act)

        # ---- 676-727: Play history ----
        if hasattr(self, '_play_history'):
            for i, (pl, act) in enumerate(self._play_history):
                if i >= PLAY_DIM:
                    break
                obs[OFS_PLAY + i] = float(act)

        return obs

    def reset(self, vul: Optional[Vul] = None, dealer: Optional[Player] = None,
              board_num: Optional[int] = None) -> np.ndarray:
        """Reset environment with a new deal."""
        if board_num is not None:
            self.board = Board(board_num=board_num)
            self.deal = random_deal()
            self._turn_idx = self.board.dealer.value
        else:
            self.board = None
            self.deal = random_deal()
            if dealer is not None:
                self._turn_idx = dealer.value
            else:
                self._turn_idx = Player.north.value

        self.done = False
        self.phase = 'bidding'
        actual_dealer = self.dealer if self.board else (dealer or Player.north)
        self.bidding.reset(actual_dealer)
        
        if hasattr(self, '_play_state'):
            delattr(self, '_play_state')
        if hasattr(self, '_play_history'):
            self._play_history = []
        else:
            self._play_history = []
        
        self._final_reward = 0.0
        return self._encode_obs()

    def step(self, action_idx: int):
        """Execute action. Returns (obs, reward, done, info)."""
        if self.done:
            raise RuntimeError('Episode already done')

        info = {'phase': self.phase}

        if self.phase == 'bidding':
            player = self._current_player()
            self.bidding.record(action_idx, player)
            self._turn_idx += 1

            if self.bidding.is_auction_over:
                if self.bidding.contract_level == 0:
                    self.done = True
                    self._final_reward = 0.0
                    return self._encode_obs(), 0.0, True, info

                self.phase = 'play'
                declarer = self.bidding.declarer
                bid_reward = self.rewards.bidding_reward(
                    self.deal, self.bidding.contract_denom,
                    self.bidding.contract_level, declarer, self.vul
                )
                self._init_play_state(declarer)
                self._turn_idx = 0

            return self._encode_obs(), 0.0, False, info

        # ---- Play phase ----
        return self._play_step(action_idx, info)

    def _init_play_state(self, declarer: Player):
        """Initialize play state."""
        from types import SimpleNamespace
        dummy = declarer.next().next()
        lead_player = declarer.next()

        self._play_state = SimpleNamespace(
            declarer=declarer,
            dummy=dummy,
            lead_player=lead_player,
            _current_player=lead_player,
            _tricks_won_ns=0,
            _tricks_won_ew=0,
            _current_trick_cards=[],
            _trick_no=0,
            _played_cards=[],
            done=False
        )

    def _get_remaining_cards(self, player: Player) -> List:
        """Get remaining cards for a player."""
        hand = self._get_hand(player)
        played = {card for p, card in self._play_state._played_cards}
        return [c for c in hand if c not in played]

    def _play_legal_mask(self) -> np.ndarray:
        """Get legal card mask for current player."""
        mask = np.zeros(NUM_PLAY, dtype=bool)
        current = self._play_state._current_player
        remaining = self._get_remaining_cards(current)

        if not remaining:
            return mask

        if len(self._play_state._current_trick_cards) > 0:
            lead_suit = self._play_state._current_trick_cards[0][1].suit
            same_suit_cards = [c for c in remaining if c.suit == lead_suit]
            if same_suit_cards:
                for c in same_suit_cards:
                    idx = _SUIT_TO_IDX[c.suit] * CARDS_PER_SUIT + _RANK_TO_IDX[c.rank]
                    mask[idx] = True
                return mask

        for c in remaining:
            idx = _SUIT_TO_IDX[c.suit] * CARDS_PER_SUIT + _RANK_TO_IDX[c.rank]
            mask[idx] = True
        return mask

    def _play_step(self, action_idx: int, info: dict):
        """Execute a play step."""
        current = self._play_state._current_player
        remaining = self._get_remaining_cards(current)

        card_suit_idx = action_idx // CARDS_PER_SUIT
        card_rank_idx = action_idx % CARDS_PER_SUIT

        suit_map = {0: Denom.spades, 1: Denom.hearts, 2: Denom.diamonds, 3: Denom.clubs}
        card_suit = suit_map[card_suit_idx]

        played_card = None
        for c in remaining:
            if _SUIT_TO_IDX[c.suit] == card_suit_idx and _RANK_TO_IDX[c.rank] == card_rank_idx:
                played_card = c
                break

        if played_card is None:
            raise ValueError(f"Card not in hand: suit={card_suit_idx} rank={card_rank_idx}")

        self._play_history.append((current, action_idx))
        self._play_state._played_cards.append((current, played_card))

        if len(self._play_state._current_trick_cards) == 0:
            self._play_state._lead_suit = played_card.suit

        self._play_state._current_trick_cards.append((current, played_card))

        is_trick_complete = len(self._play_state._current_trick_cards) == 4

        if is_trick_complete:
            self._award_trick()

            if self._play_state._trick_no >= CARDS_PER_SUIT:
                self._play_state.done = True

        if self._play_state.done:
            self.done = True
            reward = self.rewards.episode_reward(
                self.deal,
                self.bidding.contract_denom,
                self.bidding.contract_level,
                self.bidding.declarer,
                self.vul,
                self._play_state._tricks_won_ns if self.bidding.declarer.value % 2 == 0 else self._play_state._tricks_won_ew,
            )
            self._final_reward = reward
            info['tricks_won'] = self._play_state._tricks_won_ns if self.bidding.declarer.value % 2 == 0 else self._play_state._tricks_won_ew

            ev = self.rewards.full_evaluation(
                self.deal,
                self.bidding.contract_denom,
                self.bidding.contract_level,
                self.bidding.declarer,
                self.vul.value if hasattr(self.vul, 'value') else 0,
            )
            info['dds'] = ev
            info['contract'] = {
                'level': self.bidding.contract_level,
                'denom': str(self.bidding.contract_denom.abbr),
                'declarer': str(self.bidding.declarer),
                'doubled': self.bidding.doubled_status(),
            }
            return self._encode_obs(), reward, True, info

        # Next player
        self._play_state._current_player = current.next()

        return self._encode_obs(), 0.0, False, info

    def _award_trick(self):
        """Award the completed trick."""
        lead_suit = getattr(self._play_state, '_lead_suit', self._play_state._current_trick_cards[0][1].suit)
        winner = self._play_state._current_trick_cards[0][0]
        highest_rank = None

        for pl, card in self._play_state._current_trick_cards:
            if card.suit == lead_suit:
                if highest_rank is None or _RANK_TO_IDX[card.rank] > _RANK_TO_IDX[highest_rank.rank]:
                    highest_rank = card
                    winner = pl

        if winner.value % 2 == 0:
            self._play_state._tricks_won_ns += 1
        else:
            self._play_state._tricks_won_ew += 1

        self._play_state._current_trick_cards = []
        self._play_state._current_player = winner
        self._play_state._trick_no += 1

    def legal_mask(self) -> np.ndarray:
        """Return bool mask of legal actions."""
        if self.phase == 'bidding':
            mask = np.zeros(NUM_BID, dtype=bool)
            player = self._current_player()
            for act in range(NUM_BID):
                mask[act] = self.bidding.is_legal(act, player)
            return mask
        elif self.phase == 'play' and hasattr(self, '_play_state'):
            return self._play_legal_mask()
        return np.zeros(NUM_PLAY, dtype=bool)

    def render(self):
        """Print current state."""
        print(f"Phase: {self.phase}")
        if self.phase == 'bidding':
            print(f"Turn: {self._current_player()}")
            print(f"History ({len(self.bidding.history)} bids):")
            for pl, act in self.bidding.history:
                bid_str, _, _ = action_to_bid(act)
                print(f"  {pl}: {bid_str}")
        else:
            print(f"Contract: {self.bidding.contract_level}{self.bidding.contract_denom.abbr} by {self.bidding.declarer}")
            print(f"Tricks: NS={self._play_state._tricks_won_ns}, EW={self._play_state._tricks_won_ew}")
            print(f"Current player: {self._play_state._current_player}")
