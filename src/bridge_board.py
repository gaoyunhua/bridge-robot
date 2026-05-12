#!/usr/bin/env python3
"""
Bridge Board - Extended endplay Board with all bridge logic embedded.
"""
import os
import random
import sys
import io

import numpy as np
import endplay as ep
import endplay.config as ep_config
from endplay.types import (
    Board, Deal, Player, Card, Bid, Contract, Vul, Denom, Penalty
)
from endplay import dds

ep_config.use_unicode = False


class SuppressDDSOutput:
    """Context manager to suppress DDS warning messages from C library"""
    def __enter__(self):
        import os
        self._old_stdout = os.dup(1)
        self._old_stderr = os.dup(2)
        
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        
        return self
    
    def __exit__(self, *args):
        import os
        os.dup2(self._old_stdout, 1)
        os.dup2(self._old_stderr, 2)
        os.close(self._old_stdout)
        os.close(self._old_stderr)


class ExtendedBoard(Board):
    _next_board_num = 1
    
    def __init__(self, *args, **kwargs):
        self.human_player_idx = kwargs.pop('human_player_idx', None)
        self.bid_sources = []
        self._dds_cache = None  # Cache for DDS analysis results
        self._trick_history = []  # Track all played cards
        super().__init__(*args, **kwargs)
    
    @classmethod
    def new_game(cls, mode='4ai'):
        board_num = cls._next_board_num
        cls._next_board_num += 1
        deal = ep.generate_deal()
        hp_idx = random.randint(0, 3) if mode == '3ai1human' else None
        return cls(deal=deal, board_num=board_num, human_player_idx=hp_idx)
    
    def is_bidding(self) -> bool:
        if not self.auction:
            return True
        
        if len(self.auction) >= 3:
            last3 = self.auction[-3:]
            if all(str(b.as_penalty()) == 'P' for b in last3):
                return False
        
        return True
    
    def is_game_over(self) -> bool:
        if not self.auction:
            return False
        
        if self.is_bidding():
            return False
        
        if self.is_playing():
            return False
        
        return True
    
    def is_playing(self) -> bool:
        contract = self.get_contract()
        if not contract:
            return False
        
        total_remaining = sum(len(list(getattr(self.deal, p))) for p in ['north', 'east', 'south', 'west'])
        if total_remaining == 0:
            return False
        
        if len(self.auction) >= 3:
            last3 = self.auction[-3:]
            if all(str(b.as_penalty()) == 'P' for b in last3):
                return True
        
        return False
    
    def get_contract(self):
        if len(self.auction) == 0:
            return None
        return Contract.from_auction(self.dealer, self.auction)
    
    def get_declarer(self) -> Player:
        contract = self.get_contract()
        if contract is None:
            return None
        return contract.declarer
    
    def get_dummy(self) -> Player:
        declarer = self.get_declarer()
        if declarer is None:
            return None
        return (declarer + 2) % 4
    
    def get_legal_plays(self, player_idx: int) -> list:
        """Get list of legal cards to play for a player"""
        if not self.is_playing():
            return []
        
        player = Player(player_idx)
        hand = getattr(self.deal, player.name.lower())
        
        if not self.deal.curtrick:
            return list(hand)
        
        led_suit = self.deal.curtrick[0].suit
        
        cards_in_suit = [c for c in hand if c.suit == led_suit]
        
        if cards_in_suit:
            return cards_in_suit
        
        return list(hand)
    
    def play_card(self, player_idx: int, card_str: str) -> bool:
        """Play a card, return True if successful"""
        if not self.is_playing():
            return False
        
        try:
            player = Player(player_idx)
            card = ep.Card(card_str)
            
            legal_plays = self.get_legal_plays(player_idx)
            if card not in legal_plays:
                return False
            
            self.deal.play(card)
            self._trick_history.append(str(card))
            
            return True
        except Exception:
            return False
    
    def get_current_trick(self):
        """Get the current trick being played"""
        if not hasattr(self.deal, 'curtrick'):
            return []
        return list(self.deal.curtrick)
    
    def get_tricks_won(self):
        """Get number of tricks won by each side"""
        ns_tricks = 0
        ew_tricks = 0
        
        if hasattr(self.deal, 'history'):
            for trick in self.deal.history:
                winner = trick.winner()
                if winner in [Player.north, Player.south]:
                    ns_tricks += 1
                else:
                    ew_tricks += 1
        
        return ns_tricks, ew_tricks
    
    def get_current_bidder(self) -> Player:
        if self.dealer is None:
            return Player.north
        return self.dealer.next(len(self.auction))
    
    def get_current_bidder_idx(self) -> int:
        return int(self.get_current_bidder())
    
    def get_current_player(self) -> Player:
        if self.is_bidding():
            return self.get_current_bidder()
        elif self.is_playing():
            return self.deal.curplayer
        else:
            return Player.north
    
    def get_current_player_idx(self) -> int:
        return int(self.get_current_player())
    
    def get_vul_display(self):
        if self.vul is None:
            return {'ns':False, 'ew':False}
        ns_vul = Player.north.is_vul(self.vul)
        ew_vul = Player.east.is_vul(self.vul)
        return {'ns': ns_vul, 'ew': ew_vul}
    
    def is_legal_bid(self, bid_str):
        try:
            test_bid = Bid(bid_str)
            temp_board = Board(deal=self.deal, dealer=self.dealer, vul=self.vul)
            temp_board.auction = list(self.auction)
            temp_board.auction.append(test_bid)
            return True
        except Exception:
            return False
    
    @staticmethod
    def determine_bid_color(bid_str):
        if bid_str == 'Pass':
            return 'green'
        elif bid_str in ['Double', 'Redouble']:
            return 'red'
        else:
            return 'normal'
    
    def get_game_state(self):
        hands = {}
        players = [Player.north, Player.east, Player.south, Player.west]
        player_strs = ['north', 'east', 'south', 'west']
        
        is_playing = self.is_playing()
        dummy_idx = int(self.get_dummy()) if is_playing else None
        is_watching = (self.human_player_idx is None)
        # 检查是否有牌已经打出（首攻后）
        has_played_cards = len(self._trick_history) > 0
        
        for idx, (p, p_str) in enumerate(zip(players, player_strs)):
            try:
                hand_object = getattr(self.deal, p_str)
                card_list = list(hand_object)
                is_human = (self.human_player_idx is not None and idx == self.human_player_idx)
                is_dummy = (dummy_idx is not None and idx == dummy_idx)
                
                if is_watching:
                    cards = [str(c) for c in card_list]
                elif is_human:
                    cards = [str(c) for c in card_list]
                elif is_dummy and has_played_cards:
                    cards = [str(c) for c in card_list]
                else:
                    cards = ['?' for _ in card_list]
                
                hands[p_str] = cards
            except Exception:
                hands[p_str] = []
        
        bidding = []
        for idx, bid_obj in enumerate(self.auction):
            penalty = bid_obj.as_penalty()
            if str(penalty) == 'P':
                bid_str = 'Pass'
            elif str(penalty) == 'X':
                bid_str = 'Double'
            elif str(penalty) == 'XX':
                bid_str = 'Redouble'
            else:
                bid_str = str(bid_obj)
            player_idx = (int(self.dealer) + idx) %4 if self.dealer else idx %4
            
            is_ai_player = self.human_player_idx is None or player_idx != self.human_player_idx
            
            color = ExtendedBoard.determine_bid_color(bid_str)
            source = 'human'
            
            if is_ai_player:
                source = self.bid_sources[idx] if idx < len(self.bid_sources) else 'dds'
                
                if color == 'normal':
                    color = source
                elif color == 'green':
                    color = f'{source}_pass'
                elif color == 'red':
                    color = f'{source}_double'
            
            bidding.append({
                'bid': bid_str,
                'player': player_idx,
                'color': color,
                'source': source
            })
        
        cur_player = self.get_current_player_idx()
        
        _, dds_action, _ = self.make_smart_bid() if self.is_bidding() else (None, None, None)
        
        game_over = self.is_game_over()
        
        ns_tricks, ew_tricks = self.get_tricks_won()
        current_trick = [str(c) for c in self.get_current_trick()]
        
        state = {
            'phase': 'bidding' if self.is_bidding() else 'play' if self.is_playing() else 'game_over',
            'game_over': game_over,
            'bidding': bidding,
            'current_player': cur_player,
            'human_player': self.human_player_idx,
            'dummy_player': dummy_idx,
            'hands': hands,
            'vul': self.get_vul_display(),
            'tricks_ns': ns_tricks,
            'tricks_ew': ew_tricks,
            'current_trick': current_trick,
            'trick_history': self._trick_history,
            'dds_recommended_action': dds_action
        }
        
        contract = self.get_contract()
        if contract:
            state['contract'] = str(contract)
        
        return state
    
    def bid_to_action_idx(self, bid: Bid) -> int:
        penalty = bid.as_penalty()
        if str(penalty) == 'P':
            return 35
        elif str(penalty) == 'X':
            return 36
        elif str(penalty) == 'XX':
            return 37
        level = bid.level - 1
        denom_idx = bid.denom.value
        if bid.denom == Denom.nt:
            denom_idx = 4
        return level * 5 + denom_idx
    
    def get_legal_action_indices(self) -> list[int]:
        actions = []
        
        if len(self.auction) == 0:
            for i in range(35):
                actions.append(i)
        else:
            last_contract_bid = None
            for b in reversed(self.auction):
                if hasattr(b, 'level') and hasattr(b, 'denom'):
                    last_contract_bid = b
                    break
            
            if last_contract_bid:
                next_level = last_contract_bid.level + 1
                if next_level <= 7:
                    base = (next_level - 1) * 5
                    for i in range(5):
                        actions.append(base + i)
        
        actions.append(35)
        return actions
    
    def get_action_name(self, action_idx: int) -> str:
        if action_idx == 35:
            return 'Pass'
        elif action_idx == 36:
            return 'Double'
        elif action_idx == 37:
            return 'Redouble'
        else:
            level = action_idx // 5 + 1
            denoms = ['C', 'D', 'H', 'S', 'NT']
            denom = denoms[action_idx % 5]
            return f'{level}{denom}'
    
    def get_dds_analysis(self) -> tuple:
        """Get or compute DDS par analysis, returns (par_list,) cached if successful"""
        if self._dds_cache is not None:
            return self._dds_cache
        
        try:
            with SuppressDDSOutput():
                par_list = ep.par(self.deal, self.vul, self.dealer)
            self._dds_cache = (par_list,)
            return self._dds_cache
        except Exception:
            self._dds_cache = (None,)
            return self._dds_cache
    
    
    
    def dds_best_bid(self) -> tuple:
        """Get best bid from DDS analysis, returns (bid_str, action_idx) or (None, None) on failure"""
        if not self.is_bidding():
            return None, None
        
        if self.dealer is None or self.deal is None:
            return None, None
        
        legal_actions = self.get_legal_action_indices()
        if not legal_actions:
            return None, None
        
        par_list, = self.get_dds_analysis()
        if not par_list:
            return None, None
        
        first_contract = list(par_list)[0]
        level = first_contract.level
        denom_abbr = first_contract.denom.abbr
        bid_str = f'{level}{denom_abbr}'
        
        action_idx = self.bid_str_to_action_idx(bid_str)
        if action_idx in legal_actions:
            return bid_str, action_idx
        
        return None, None
    
    def bid_str_to_action_idx(self, bid_str):
        """Convert bid string like '1H' to action index"""
        if bid_str == 'Pass':
            return 35
        if bid_str == 'Double':
            return 36
        if bid_str == 'Redouble':
            return 37
        
        level = int(bid_str[0])
        denom = bid_str[1:]
        denom_map = {'C': 0, 'D': 1, 'H': 2, 'S': 3, 'NT': 4}
        
        if denom in denom_map:
            return (level - 1) * 5 + denom_map[denom]
        
        return 35
    
    def model_best_bid(self) -> tuple:
        """Get best bid from trained model, returns (bid_str, action_idx) or (None, None) if unavailable"""
        if not self.is_bidding():
            return None, None
        
        if self.dealer is None or self.deal is None:
            return None, None
        
        try:
            import torch
            from model_transformer import BridgeTransformerV2
            
            model = getattr(self, '_cached_model', None)
            if model is None:
                project_root = os.path.dirname(os.path.dirname(__file__))
                checkpoint_dir = os.path.join(project_root, 'checkpoints')
                model_path = os.path.join(checkpoint_dir, 'policy_model_v2.pt')
                if not os.path.exists(model_path):
                    return None, None
                model = BridgeTransformerV2()
                model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
                model.eval()
                self._cached_model = model
            
            obs = self._build_model_obs()
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                bid_logits, _ = model(obs_tensor)
            
            legal_actions = self.get_legal_action_indices()
            if not legal_actions:
                return None, None
            
            logits = bid_logits.squeeze(0)
            for i in range(len(logits)):
                if i not in legal_actions:
                    logits[i] = float('-inf')
            
            best_action = logits.argmax().item()
            if best_action not in legal_actions:
                return None, None
            
            return self.get_action_name(best_action), best_action
        except Exception:
            return None, None
    
    def _build_model_obs(self) -> np.ndarray:
        """Build observation vector for the model from current board state"""
        from config import EnvConfig
        
        cfg = EnvConfig()
        obs = np.zeros(cfg.obs_dim, dtype=np.float32)
        
        _RANK_TO_IDX = {ep.Rank.R2: 0, ep.Rank.R3: 1, ep.Rank.R4: 2, ep.Rank.R5: 3,
                        ep.Rank.R6: 4, ep.Rank.R7: 5, ep.Rank.R8: 6, ep.Rank.R9: 7,
                        ep.Rank.RT: 8, ep.Rank.RJ: 9, ep.Rank.RQ: 10, ep.Rank.RK: 11,
                        ep.Rank.RA: 12}
        _SUIT_TO_IDX = {ep.Denom.spades: 0, ep.Denom.hearts: 1, ep.Denom.diamonds: 2, ep.Denom.clubs: 3}
        
        OFS_MATCH = 0
        OFS_CONTRACT = 12
        OFS_HANDS = 220
        OFS_FIRST = 448
        OFS_AUCTION = 477
        OFS_PLAY = 676
        NUM_PLAY = 52
        
        obs[OFS_MATCH] = float(getattr(self.vul, 'value', 0)) if self.vul else 0.0
        
        contract = self.get_contract()
        if contract and contract.level > 0:
            lvl = contract.level
            denom = contract.denom
            decl = contract.declarer
            
            obs[OFS_CONTRACT + lvl - 1] = 1.0
            denom_idx = _SUIT_TO_IDX.get(denom, 4)
            if denom_idx < 4:
                obs[OFS_CONTRACT + 7 + denom_idx] = 1.0
            else:
                obs[OFS_CONTRACT + 7 + 4] = 1.0
            obs[OFS_CONTRACT + 12 + int(decl)] = 1.0
            obs[OFS_CONTRACT + 16] = float(contract.doubled)
        
        players = [ep.Player.north, ep.Player.east, ep.Player.south, ep.Player.west]
        for pl_i, pl in enumerate(players):
            hand = getattr(self.deal, pl.name.lower())
            hand_offset = OFS_HANDS + pl_i * 52
            for card in hand:
                if card is None:
                    continue
                s = _SUIT_TO_IDX.get(card.suit, 0)
                r = _RANK_TO_IDX.get(card.rank, 0)
                if 0 <= s * 13 + r < 52:
                    obs[hand_offset + s * 13 + r] = 1.0
        
        current_player = self.get_current_bidder()
        obs[OFS_FIRST] = float(current_player.value)
        
        for i, bid_obj in enumerate(self.auction):
            if i >= 199:
                break
            action_idx = self.bid_to_action_idx(bid_obj)
            obs[OFS_AUCTION + i] = float(action_idx)
        
        return obs
    
    def make_smart_bid(self) -> tuple:
        """Make a smart bid: model first, then DDS, then Pass as last resort"""
        if not self.is_bidding():
            return None, None, None
        
        model_bid, model_action = self.model_best_bid()
        if model_bid is not None and model_action != 35:
            if self.is_legal_bid(model_bid):
                return model_bid, model_action, 'model'
        
        dds_bid, dds_action = self.dds_best_bid()
        if dds_bid is not None:
            return dds_bid, dds_action, 'dds'
        
        return 'Pass', 35, 'fallback'