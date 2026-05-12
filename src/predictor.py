"""BridgeAIPredictor – sequencing predictions."""

from __future__ import annotations
import torch
from typing import List, NamedTuple
from config import EnvConfig
from model_transformer import BridgeTransformerV2
from env_core import BridgeEnv

class PredictionStep(NamedTuple):
    phase: str
    player_idx: int
    action_idx: int
    action_name: str

class BridgeAIPredictor:
    def __init__(self, model: BridgeTransformerV2, config: EnvConfig = None):
        self.model = model
        self.cfg = config or EnvConfig()
        bid_names, play_names = self.cfg.action_names()
        self.bid_names = bid_names
        self.play_names = play_names

    def predict(self, env: BridgeEnv, max_steps: int = 80) -> List[PredictionStep]:
        env.reset()
        steps = []
        for i in range(max_steps):
            obs = torch.tensor(env._encode_obs(), dtype=torch.float32).unsqueeze(0)
            if env.phase == 'bidding':
                action_idx = self.model.predict(obs, phase='bidding').item()
                name = self.bid_names[action_idx]
            else:
                action_idx = self.model.predict(obs, phase='playing').item()
                name = self.play_names[action_idx]
            step = PredictionStep(env.phase, env._current_player(), action_idx, name)
            steps.append(step)
            obs2, rew, done, info = env.step(action_idx)
            if done:
                break
        return steps

    def predict_single(self, obs: torch.Tensor, phase: str):
        return self.model.predict(obs, phase=phase)
