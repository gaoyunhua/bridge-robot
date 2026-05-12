"""Configuration definitions for the Bridge AI project.

This module defines two simple dataclasses:
- ``EnvConfig`` – contains environment‑level constants such as
  observation dimension, number of bidding & playing actions, and the
  enumeration shortcuts for ``Vul``, ``Denom``, ``Player`` and ``Rank``
  from the ``endplay`` library.
- ``TrainConfig`` – hyper‑parameters for the training loop (learning rate,
  batch size, number of epochs, etc.).

The values are deliberately kept minimal – they can be tweaked later
without touching the rest of the code base.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

# Import Endplay enums -- they are re-exported for convenience
from endplay.types import Vul, Denom, Player, Rank


# ---------------------------------------------------------------------------
# Canonical dimension constants — SINGLE SOURCE OF TRUTH for every
# sub‑section of the observation vector.
#
# Every module that encodes or decodes observations MUST import these
# constants from here, never hard‑code them inline.
# ---------------------------------------------------------------------------

# ── Observation sub‑dimensions ──────────────────────────────────────────
# Total: 780 dimensions
# 0‑11: match metadata
MATCH_META_DIM = 12             # board_number, table_count, round_number, total_rounds

# 12‑219: contract (vul, vul_info, declarer, level, nt, suit, double, result)
CONTRACT_DIM = 208              # padded to 208

# 220‑447: hands (4 players × 52 used + padding)
HAND_CARD_DIM = 53              # each hand encoded as 53 floats
HANDS_ENCODED_DIM = 52          # used per hand before padding
HANDS_DIM = 228                 # 4 × 52 = 208 → padded to 228

# 448‑476: first bidder
FIRST_BIDDER_DIM = 29

# 477‑675: auction history
AUCTION_SLOTS = 66              # max bid entries
AUCTION_ENTRY_DIM = 3           # (bidder, level, denom)
AUCTION_DIM = 198               # 66 × 3
AUCTION_PADDED = 199            # padded to 199

# 676‑727: play history (52 cards)
PLAY_DIM = 52

# ── Core constants ────────────────────────────────────────────────────
NUM_PLAYERS = 4                 # number of players
NUM_SUITS = 4                   # number of suits (S, H, D, C)

# ── Action dimensions ──────────────────────────────────────────────────
NUM_BID_ACTIONS = 38            # 1C‑7NT + Pass + Double + Redouble
CARDS_PER_SUIT = 13             # cards per suit (2-A)
NUM_PLAY_ACTIONS = NUM_SUITS * CARDS_PER_SUIT  # 4 suits × 13 ranks
NUM_CARDS = NUM_PLAY_ACTIONS    # total cards in a deck
CARDS_PER_HAND = NUM_CARDS // NUM_PLAYERS  # cards per player

# ── Derived offsets (for readability, computed here once) ───────────────
OBS_DIM = 757                  # total observation length

OFS_MATCH    = 0
OFS_CONTRACT = OFS_MATCH    + MATCH_META_DIM          # 12
OFS_HANDS    = OFS_CONTRACT + CONTRACT_DIM             # 220
OFS_FIRST    = OFS_HANDS    + HANDS_DIM                # 448
OFS_AUCTION  = OFS_FIRST    + FIRST_BIDDER_DIM         # 477
OFS_PLAY     = OFS_AUCTION  + AUCTION_PADDED            # 676


# ── EnvConfig / TrainConfig (unchanged, but now reference the constants above) ──


@dataclass(frozen=True)
class EnvConfig:
    """Static environment configuration.

    * ``obs_dim`` – length of the observation vector (780).
    * ``num_bid_actions`` – number of possible bidding actions (38).
    * ``num_play_actions`` – number of possible playing actions (52).
    """
    obs_dim: int = OBS_DIM
    num_bid_actions: int = NUM_BID_ACTIONS
    num_play_actions: int = NUM_PLAY_ACTIONS

    # expose the Endplay enums so other modules can import from here
    Vul = Vul
    Denom = Denom
    Player = Player
    Rank = Rank

    @staticmethod
    def action_names() -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
        """Return human readable names for bidding and playing actions.

        The first tuple contains 38 bidding symbols (1C … 7NT, Pass,
        Double, Redouble).  The second tuple contains 52 card names
        (C2 … SA).
        """
        # Bidding names – simplified ordering used in the original code
        bids = [
            f"{d}{l}" for d in ["C", "D", "H", "S", "NT"] for l in range(1, 8)
        ]
        # 35 bids (1C-7NT) + 3 meta-actions = 38 total
        bids += ["Pass", "Double", "Redouble"]
        # Playing cards – 4 suits × 13 ranks
        suits = ["C", "D", "H", "S"]
        ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
        cards = [f"{s}{r}" for s in suits for r in ranks]
        return tuple(bids), tuple(cards)


@dataclass
class TrainConfig:
    """Training hyper‑parameters.

    The defaults are suitable for a quick sanity‑check run.  Adjust as
    needed for full‑scale training.
    """

    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4
    device: str = "cpu"  # will be overridden to "cuda" if available
    seed: int = 42
    log_interval: int = 10

    # PPO / RL specific placeholders – not used in the simplified run
    ppo_clip: float = 0.2
    ppo_epochs: int = 3
    gamma: float = 0.99


# Utility function to set random seed (used by training scripts)
def set_seed(seed: int) -> None:
    import random, numpy as np, torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
