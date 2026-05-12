"""BridgeTransformerV2 – a tiny 4‑layer transformer.

Supports:
  - Supervised training:  forward(obs) → (bid_logits, play_logits)
  - RL training:         forward(obs, return_hidden=True) → (bid_logits, play_logits, hidden)

Uses PyTorch built‑in TransformerEncoder.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from config import EnvConfig


class BridgeTransformerV2(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 4, num_layers: int = 4,
                 input_dim: int = None, hidden_dim: int = None,
                 num_bid_actions: int = None, num_play_actions: int = None):
        super().__init__()
        self.cfg = EnvConfig()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                         batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # Input projection
        obs_dim = input_dim if input_dim is not None else self.cfg.obs_dim
        self.input_proj = nn.Linear(obs_dim, d_model)

        # Output heads
        n_bid = num_bid_actions if num_bid_actions is not None else self.cfg.num_bid_actions
        n_play = num_play_actions if num_play_actions is not None else self.cfg.num_play_actions
        self.bid_head = nn.Linear(d_model, n_bid)
        self.play_head = nn.Linear(d_model, n_play)

    def forward(self, obs: torch.Tensor, return_hidden: bool = False):
        """
        Args:
            obs: (B, obs_dim)
            return_hidden: if True, also return the shared hidden representation
        Returns:
            Without return_hidden: (bid_logits, play_logits)
            With return_hidden:    (bid_logits, play_logits, hidden)
        """
        # obs shape: (B, obs_dim)
        x = self.input_proj(obs)               # (B, d_model)

        # Add a dummy sequence dimension → (B, 1, d_model)
        x = x.unsqueeze(1)
        x = self.encoder(x)                     # (B, 1, d_model)
        x = x.squeeze(1)                        # (B, d_model)

        if return_hidden:
            hidden = x
            bid_logits = self.bid_head(hidden)
            play_logits = self.play_head(hidden)
            return bid_logits, play_logits, hidden

        bid_logits = self.bid_head(x)
        play_logits = self.play_head(x)
        return bid_logits, play_logits

    def predict(self, obs: torch.Tensor, phase: str = "bidding"):
        self.eval()
        with torch.no_grad():
            bid, play = self.forward(obs)
            if phase == "bidding":
                return bid.argmax(dim=-1)
            else:
                return play.argmax(dim=-1)
