"""
Rewards Module — DDS/Par scoring for bridge AI training.

Computes real bridge scores from endplay DDS analysis, used as RL
reward signals:
  * Bidding reward:  parscore for the declared contract
  * Play reward:     trick difference between declarer and defenders
  * Final episode reward: IMP/MP conversion of the result
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

from endplay import Deal, generate_deal
from endplay.dds import calc_dd_table, par as dds_par
from endplay.types import Player, Vul, Denom, Contract


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vul_to_endplay(vul: int) -> Vul:
    """Map internal vul index (0=None, 1=NS, 2=EW, 3=Both) to endplay Vul."""
    _MAP = {0: Vul.none, 1: Vul.ns, 2: Vul.ew, 3: Vul.both}
    return _MAP.get(vul, Vul.none)


# Denom ordering in endplay: Spades=0, Hearts=1, Diamonds=2, Clubs=3, NT=4
_DENOM_ORDER = [Denom.spades, Denom.hearts, Denom.diamonds, Denom.clubs, Denom.nt]
# Player ordering: North=0, South=1, East=2, West=3
_PLAYER_ORDER = [Player.north, Player.south, Player.east, Player.west]


def hcp(deal: Deal, player: Player) -> int:
    """Calculate HCP for a given player's hand (A=4, K=3, Q=2, J=1)."""
    # endplay Rank enum uses powers of 2: 2=2, 3=4, 4=8, ..., 10=512, J=1024, Q=2048, K=4096, A=8192
    _HCP = {8192: 4, 4096: 3, 2048: 2, 1024: 1}
    hand = getattr(deal, player.name)
    total = 0
    for card in hand:
        total += _HCP.get(card.rank.value, 0)
    return total


def compute_dd_table_array(deal: Deal) -> np.ndarray:
    """
    Compute double-dummy table for a deal.
    Returns 5×4 array [strain][player]:
      strains: Spades(0), Hearts(1), Diamonds(2), Clubs(3), NT(4)
      players: North(0), South(1), East(2), West(3)
    """
    table = calc_dd_table(deal)
    out = np.zeros((5, 4), dtype=np.int32)
    for di, denom in enumerate(_DENOM_ORDER):
        for pi, player in enumerate(_PLAYER_ORDER):
            out[di, pi] = table[(denom, player)]
    return out


def trick_count(table: np.ndarray, contract_denom: Denom, declarer: Player) -> int:
    """
    Extract the number of tricks from a DD table for a given contract.
    table shape: 5×4  (strain × player)
    denom_index: spades=0, hearts=1, diamonds=2, clubs=3, NT=4
    player_index: north=0, south=1, east=2, west=3
    """
    denom_to_idx = {Denom.spades: 0, Denom.hearts: 1, Denom.diamonds: 2,
                    Denom.clubs: 3, Denom.nt: 4}
    player_to_idx = {Player.north: 0, Player.south: 1, Player.east: 2, Player.west: 3}
    return table[denom_to_idx[contract_denom], player_to_idx[declarer]]


# ---------------------------------------------------------------------------
# IMP table (standard bridge IMP scale)
# ---------------------------------------------------------------------------
_IMP_TABLE = [
    (0, 0), (20, 1), (50, 2), (90, 3), (130, 4), (170, 5), (220, 6),
    (270, 7), (320, 8), (370, 9), (430, 10), (500, 11), (600, 12),
    (750, 13), (900, 14), (1100, 15), (1300, 16), (1500, 17),
    (1750, 18), (2000, 19), (2250, 20), (2500, 21), (3000, 22),
    (3500, 23), (4000, 24),
]


def imp(diff: int) -> int:
    """Convert point difference to IMPs."""
    diff = abs(diff)
    for threshold, imps in reversed(_IMP_TABLE):
        if diff >= threshold:
            return imps if diff > 0 or diff == 0 else 0
    return 0


# ---------------------------------------------------------------------------
# Contract scoring (ACBL standard)
# ---------------------------------------------------------------------------
def contract_score(level: int, denom: Denom, doubled: int,
                   vul: Vul, tricks_won: int) -> int:
    """
    Calculate the score for a contract using endplay's built-in scoring.

    Returns score from declarer's perspective (positive = declarer scores).

    Args:
        level: contract level (1-7)
        denom: strain
        doubled: 0=undoubled, 1=doubled, 2=redoubled
        vul: vulnerability
        tricks_won: tricks made by declarer
    """
    needed = level + 6
    overtricks = tricks_won - needed  # result = overtricks (positive=made, negative=down)
    
    c = Contract()
    c.level = level
    c.denom = denom
    c.declarer = Player.north  # declarer doesn't affect score calculation
    c.doubled = doubled  # 0=undoubled, 1=doubled, 2=redoubled
    c.result = overtricks
    
    return c.score(vul)


def calculate_contract_score(deal: Deal, level: int, denom: Denom, 
                             declarer: Player, vul: Vul, doubled: int = 0,
                             dd_table_cache = None) -> Tuple[int, int]:
    """
    Calculate contract score using DDS to determine the number of tricks won.

    Args:
        deal: complete deal (all cards)
        level: contract level (1-7)
        denom: contract strain
        declarer: declarer player
        vul: vulnerability
        doubled: 0=undoubled, 1=doubled, 2=redoubled (default: 0)
        dd_table_cache: Optional DDTableCache. If provided, avoids recomputing the DDS table.

    Returns:
        (tricks_won, score) - number of tricks declarer can win double-dummy,
                              and the resulting score (positive = declarer scores)
    """
    if dd_table_cache is not None:
        tricks_won = dd_table_cache.tricks(denom, declarer)
    else:
        table = compute_dd_table_array(deal)
        tricks_won = trick_count(table, denom, declarer)
    score = contract_score(level, denom, doubled, vul, tricks_won)
    return tricks_won, score


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

class RewardsModule:
    """
    Compute bridge scores suitable for RL reward signals.

    Usage in a game loop::

        rm = RewardsModule()
        deal = generate_deal()
        table = rm.compute_dd_table(deal)

        # After bidding is done, compute contract score
        bid_score = rm.bidding_reward(deal, contract_suit, contract_level,
                                      declarer, vul)

        # After full play, compute final IMP score
        final_reward = rm.episode_reward(deal, contract, tricks_won,
                                         declarer, vul)
    """

    def compute_dd_table(self, deal: Deal) -> np.ndarray:
        """Compute double-dummy table for a deal. Returns 5×4 array [strain][player]."""
        return compute_dd_table_array(deal)

    def parscore(self, deal: Deal, vul: Vul, dealer: Player) -> Tuple[str, int]:
        """
        Calculate par contract and par score for NS.

        Returns:
            (par_contract_string, ns_par_score)
        """
        # dds_par accepts DDTable (endplay object) directly
        table = calc_dd_table(deal)
        res = dds_par(table, vul, dealer)
        # res.score is an int property — the NS par score
        contracts = list(res)
        if len(contracts) >= 1:
            return str(contracts[0]), res.score
        return "PASS", 0

    def bidding_reward(self,
                       deal: Deal,
                       contract_denom: Denom,
                       contract_level: int,
                       declarer: Player,
                       vul: Vul) -> float:
        """
        Reward based on whether the declared contract is makeable.

        Returns:
            +1.0 if contract is makeable (DD tricks >= contract_level + 6)
            -1.0 if contract goes down
             0.0 if passout / no contract
        """
        if contract_level == 0:
            return 0.0  # passout

        table = compute_dd_table_array(deal)
        tricks = trick_count(table, contract_denom, declarer)
        needed = contract_level + 6

        return 1.0 if tricks >= needed else -1.0

    def play_reward(self,
                    deal: Deal,
                    contract_denom: Denom,
                    contract_level: int,
                    declarer: Player,
                    vul: Vul,
                    tricks_won: int) -> float:
        """
        Reward after all 13 tricks have been played, using IMPs.

        Returns normalized IMP score in [-1, 1] range.
        """
        # Ideal: DD optimal play makes needed tricks
        table = compute_dd_table_array(deal)
        dd_tricks = trick_count(table, contract_denom, declarer)
        dd_diff = dd_tricks - (contract_level + 6)

        # Actual result vs par
        actual_diff = tricks_won - (contract_level + 6)

        # Score in points, then IMP
        ideal_score = contract_score(contract_level, contract_denom, 0, vul,
                                     contract_level + 6 + dd_diff)
        actual_score = contract_score(contract_level, contract_denom, 0, vul,
                                      tricks_won)
        pt_diff = actual_score - ideal_score

        # Convert to IMPs (capped at ±24)
        imps = imp(pt_diff)
        return np.clip(imps / 24.0, -1.0, 1.0)

    def episode_reward(self,
                       deal: Deal,
                       contract_denom: Denom,
                       contract_level: int,
                       declarer: Player,
                       vul: Vul,
                       tricks_won: int) -> float:
        """
        Composite episode reward: bidding (makeable?) + play (tricks vs DD par).

        Returns a single scalar in range [-2, 2].
        """
        bid_r = self.bidding_reward(deal, contract_denom, contract_level,
                                     declarer, vul)

        # For episode reward, play portion compares actual tricks vs DD optimum
        table = compute_dd_table_array(deal)
        dd_tricks = trick_count(table, contract_denom, declarer)
        needed = contract_level + 6

        # Optimal tricks for this contract
        # Actual tricks vs optimal — how well did defense/declarer play?
        actual_vs_dd = tricks_won - dd_tricks
        play_r = np.clip(actual_vs_dd / 7.0, -1.0, 1.0)

        return bid_r + play_r

    def full_evaluation(self,
                        deal: Deal,
                        contract_denom: Denom,
                        contract_level: int,
                        declarer: Player,
                        vul: int = 0) -> Dict:
        """
        Run all scoring and return a detailed dict.

        Useful for analysis and debugging.
        """
        vul_e = _vul_to_endplay(vul)
        table = compute_dd_table_array(deal)
        par_str, par_score = self.parscore(deal, vul_e, declarer)
        bid_r = self.bidding_reward(deal, contract_denom, contract_level,
                                     declarer, vul_e)
        dd_tricks = trick_count(table, contract_denom, declarer)

        needed = contract_level + 6
        dd_diff = dd_tricks - needed

        return {
            'dd_table': table,
            'dd_tricks': dd_tricks,
            'tricks_needed': needed,
            'dd_diff': dd_diff,
            'par_contract': par_str,
            'par_score': par_score,
            'bid_reward': bid_r,
            'contract_str': f"{contract_level}{contract_denom.abbr}",
        }
