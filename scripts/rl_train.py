#!/usr/bin/env python3
"""
PPO (Proximal Policy Optimization) training for bridge AI.

Architecture:
  - Policy: BridgeTransformerV2 (obs → action logits)
  - Value:  Separate critic head (obs → scalar value estimate)
  - Reward: RewardsModule (DDS/Par score at episode end)

Training loop:
  1. Run N episodes with current policy, collect (obs, action, logprob, value, reward)
  2. Compute advantages (GAE)
  3. Update policy: PPO clip objective
  4. Update value: MSE loss
  5. Repeat
"""

import os
import sys
import time
import json
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Project imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
# sys.path handled dynamically below

from model_transformer import BridgeTransformerV2
from env_core import BridgeEnv, action_to_bid, NUM_BID, NUM_PLAY


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class RLConfig:
    """PPO hyperparameters."""
    # Environment
    env_seed: int = 42
    num_episodes: int = 10000
    max_steps_per_episode: int = 150

    # PPO
    batch_size: int = 64
    mini_batch_size: int = 16
    ppo_epochs: int = 4
    clip_epsilon: float = 0.15
    value_coef: float = 0.5
    entropy_coef: float = 0.05
    gamma: float = 0.95
    gae_lambda: float = 0.92
    learning_rate: float = 2e-4
    max_grad_norm: float = 0.5

    # Model
    hidden_dim: int = 256
    obs_dim: int = 757

    # Save/load
    save_dir: str = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    save_interval: int = 500
    log_interval: int = 10


# ---------------------------------------------------------------------------
# Actor-Critic
# ---------------------------------------------------------------------------
class ActorCritic(nn.Module):
    """Combined policy (actor) and value (critic) network.

    Wraps BridgeTransformerV2 with a shared body and separate heads.
    """

    def __init__(self, cfg: RLConfig):
        super().__init__()
        self.cfg = cfg

        # BridgeTransformerV2 already handles 757-dim input and
        # produces 38 bid + 52 play logits.
        self.body = BridgeTransformerV2(
            input_dim=cfg.obs_dim,
            hidden_dim=cfg.hidden_dim,
            num_bid_actions=NUM_BID,
            num_play_actions=NUM_PLAY,
        )

        # Critic head (state value)
        self.critic = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            obs: (B, 757) tensor
        Returns:
            bid_logits: (B, 38)
            play_logits: (B, 52)
            value: (B, 1)
        """
        # BridgeTransformerV2 forward returns [bid_logits, play_logits]
        # and internally produces hidden states.
        bid_logits, play_logits, hidden = self.body(obs, return_hidden=True)
        value = self.critic(hidden)
        return bid_logits, play_logits, value

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate only."""
        with torch.no_grad():
            _, _, value = self.forward(obs)
        return value

    def act(self, obs: torch.Tensor, legal_mask: torch.Tensor,
            phase: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy given legal mask.

        Args:
            obs: (1, 757) or (B, 757)
            legal_mask: (1, num_actions) or (B, num_actions) — bool
            phase: 'bidding' or 'play'
        Returns:
            action: (B,) tensor of chosen action indices
            logprob: (B,) log probabilities
            entropy: (B,) entropy values
        """
        bid_logits, play_logits, value = self.forward(obs)

        if phase == 'bidding':
            logits = bid_logits
        else:
            logits = play_logits

        # Mask illegal actions (set to -inf)
        illegal_mask = ~legal_mask
        masked_logits = logits.clone()
        masked_logits[illegal_mask] = float('-inf')

        # Sample
        dist = Categorical(logits=masked_logits)
        action = dist.sample()
        logprob = dist.log_prob(action)
        entropy = dist.entropy()

        return action, logprob, entropy

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor,
                          legal_masks: torch.Tensor, mask_lengths: torch.Tensor,
                          phases: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probs and entropy for given actions.

        Args:
            obs: (B, 757)
            actions: (B,) chosen action indices
            legal_masks: (B, max_actions) bool — padded to max(NUM_BID, NUM_PLAY)
            mask_lengths: (B,) int — length of the effective mask (38 or 52)
            phases: list of 'bidding'/'play' per item
        Returns:
            log_probs: (B,)
            entropy: (B,)
            values: (B, 1)
        """
        bid_logits, play_logits, values = self.forward(obs)

        log_probs_list = []
        entropies_list = []

        for i in range(obs.size(0)):
            n_actions = mask_lengths[i].item()
            phases_i = phases[i]

            if phases_i == 'bidding':
                logits = bid_logits[i:i+1, :n_actions]
            else:
                logits = play_logits[i:i+1, :n_actions]

            mask_i = legal_masks[i:i+1, :n_actions]
            illegal = ~mask_i
            masked = logits.clone()
            masked[illegal] = float('-inf')

            dist = Categorical(logits=masked)
            log_probs_list.append(dist.log_prob(actions[i:i+1]))
            entropies_list.append(dist.entropy())

        return torch.cat(log_probs_list), torch.cat(entropies_list), values


# ---------------------------------------------------------------------------
# Experience buffer
# ---------------------------------------------------------------------------
@dataclass
class Transition:
    obs: np.ndarray
    action: int
    log_prob: float
    value: float
    reward: float
    done: bool
    legal_mask: np.ndarray
    phase: str


class RolloutBuffer:
    """Stores trajectory data for PPO updates."""

    def __init__(self):
        self.obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.log_probs: List[float] = []
        self.values: List[float] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.legal_masks: List[np.ndarray] = []
        self.phases: List[str] = []

    def push(self, t: Transition):
        self.obs.append(t.obs)
        self.actions.append(t.action)
        self.log_probs.append(t.log_prob)
        self.values.append(t.value)
        self.rewards.append(t.reward)
        self.dones.append(t.done)
        self.legal_masks.append(t.legal_mask)
        self.phases.append(t.phase)

    def clear(self):
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.legal_masks.clear()
        self.phases.clear()

    def size(self) -> int:
        return len(self.obs)

    def get(self, cfg: RLConfig, device: torch.device):
        """Convert buffer to tensors and compute advantages."""
        obs = torch.FloatTensor(np.array(self.obs)).to(device)
        actions = torch.LongTensor(self.actions).to(device)
        log_probs_old = torch.FloatTensor(self.log_probs).to(device)
        values = torch.FloatTensor(self.values).to(device)
        rewards = torch.FloatTensor(self.rewards).to(device)
        dones = torch.FloatTensor(self.dones).to(device)

        # Max # of actions for mask padding
        max_actions = max(NUM_BID, NUM_PLAY)
        masks_padded = np.zeros((len(self.legal_masks), max_actions), dtype=bool)
        for i, m in enumerate(self.legal_masks):
            masks_padded[i, :len(m)] = m
        legal_masks = torch.BoolTensor(masks_padded).to(device)

        # GAE advantage calculation
        advantages = torch.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0.0 if dones[t] else values[t].item()
            else:
                next_value = values[t + 1].item() if not dones[t] else 0.0
            delta = rewards[t].item() + cfg.gamma * next_value - values[t].item()
            gae = delta + cfg.gamma * cfg.gae_lambda * (1 - dones[t]) * gae
            advantages[t] = gae

        returns = advantages + values

        # Store mask lengths (38 for bidding, 52 for play)
        mask_lengths = torch.LongTensor([
            NUM_BID if p == 'bidding' else NUM_PLAY for p in self.phases
        ]).to(device)

        return obs, actions, log_probs_old, values, advantages, returns, legal_masks, mask_lengths, self.phases

    def get_phases(self) -> List[str]:
        return self.phases


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------
class PPOTrainer:
    """PPO training loop for bridge AI."""

    def __init__(self, cfg: RLConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.agent = ActorCritic(cfg).to(device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=cfg.learning_rate, eps=1e-5)
        self.buffer = RolloutBuffer()
        self.envs: List[BridgeEnv] = []  # one env per worker (single for now)
        self.episode = 0
        self.best_reward = float('-inf')
        self.total_steps = 0

        # Stats
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.win_rate: List[bool] = []

        self._ensure_save_dir()

    def _ensure_save_dir(self):
        os.makedirs(self.cfg.save_dir, exist_ok=True)

    def save(self, path: str = None):
        """Save model checkpoint."""
        if path is None:
            path = os.path.join(self.cfg.save_dir, 'rl_model.pt')
        torch.save({
            'episode': self.episode,
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_reward': self.best_reward,
        }, path)
        print(f"  [SAVE] Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode = checkpoint.get('episode', 0)
        self.best_reward = checkpoint.get('best_reward', float('-inf'))
        print(f"  [LOAD] Loaded model from {path} (episode {self.episode})")

    # -----------------------------------------------------------------
    def collect_episode(self, env: BridgeEnv) -> float:
        """Run one episode and store transitions in buffer.

        Returns: episode reward.
        """
        obs = env.reset()
        episode_reward = 0.0
        steps = 0

        while not env.done and steps < self.cfg.max_steps_per_episode:
            # Prepare inputs
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mask = env.legal_mask()
            mask_t = torch.BoolTensor(mask).unsqueeze(0).to(self.device)
            phase = env.phase

            # Get action from policy
            with torch.no_grad():
                action_t, logprob_t, _ = self.agent.act(obs_t, mask_t, phase)

            action = action_t.item()
            logprob = logprob_t.item()
            value = self.agent.get_value(obs_t).item()

            # Step environment
            next_obs, reward, done, info = env.step(action)

            # Store transition
            self.buffer.push(Transition(
                obs=obs if isinstance(obs, np.ndarray) else np.array(obs),
                action=action,
                log_prob=logprob,
                value=value,
                reward=reward,
                done=done,
                legal_mask=mask,
                phase=phase,
            ))

            episode_reward += reward
            obs = next_obs
            steps += 1
            self.total_steps += 1

        self.episode += 1
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(steps)
        self.win_rate.append(episode_reward > 0)

        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self.save(os.path.join(self.cfg.save_dir, 'rl_best.pt'))

        return episode_reward

    # -----------------------------------------------------------------
    def update(self) -> dict:
        """Perform PPO update on collected rollout buffer.

        Returns: dict with loss metrics.
        """
        obs, actions, old_log_probs, old_values, advantages, returns, legal_masks, mask_lengths, phases = \
            self.buffer.get(self.cfg, self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        num_updates = 0

        dataset_size = obs.size(0)
        indices = np.arange(dataset_size)

        for _epoch in range(self.cfg.ppo_epochs):
            np.random.shuffle(indices)

            for start in range(0, dataset_size, self.cfg.mini_batch_size):
                end = start + self.cfg.mini_batch_size
                batch_idx = indices[start:end]

                batch_obs = obs[batch_idx]
                batch_actions = actions[batch_idx]
                batch_old_logprobs = old_log_probs[batch_idx]
                batch_advantages = advantages[batch_idx]
                batch_returns = returns[batch_idx]
                batch_masks = legal_masks[batch_idx]
                batch_phases = [phases[i] for i in batch_idx]

                # Evaluate actions
                batch_mask_lengths = mask_lengths[batch_idx]
                log_probs, entropy, values = self.agent.evaluate_actions(
                    batch_obs, batch_actions, batch_masks, batch_mask_lengths, batch_phases
                )

                # PPO ratio
                ratio = torch.exp(log_probs - batch_old_logprobs)

                # Clipped surrogate objective
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.cfg.clip_epsilon,
                                    1.0 + self.cfg.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                values_pred = values.squeeze(-1)
                value_loss_unclipped = F.mse_loss(values_pred, batch_returns)
                value_clipped = old_values[batch_idx] + torch.clamp(
                    values_pred - old_values[batch_idx],
                    -self.cfg.clip_epsilon,
                    self.cfg.clip_epsilon,
                )
                value_loss_clipped = F.mse_loss(value_clipped, batch_returns)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Combined loss
                loss = (policy_loss
                        + self.cfg.value_coef * value_loss
                        + self.cfg.entropy_coef * entropy_loss)

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy_loss.item()
                num_updates += 1

        return {
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
        }

    # -----------------------------------------------------------------
    def train(self, num_episodes: int = None):
        """Main training loop."""
        if num_episodes is None:
            num_episodes = self.cfg.num_episodes

        env = BridgeEnv(seed=self.cfg.env_seed)
        print(f"Starting PPO training for {num_episodes} episodes...")
        print(f"  Device: {self.device}")
        print(f"  Obs dim: {self.cfg.obs_dim}")
        print(f"  Model params: {sum(p.numel() for p in self.agent.parameters()):,}")
        print()

        start_time = time.time()

        for ep in range(1, num_episodes + 1):
            self.buffer.clear()
            reward = self.collect_episode(env)
            stats = self.update()

            # Logging
            if ep % self.cfg.log_interval == 0:
                elapsed = time.time() - start_time
                avg_reward = np.mean(self.episode_rewards[-self.cfg.log_interval:])
                avg_len = np.mean(self.episode_lengths[-self.cfg.log_interval:])
                win_pct = np.mean(self.win_rate[-self.cfg.log_interval:]) * 100

                print(
                    f"Ep {ep:6d}/{num_episodes} | "
                    f"Reward={avg_reward:+.3f} | "
                    f"Win={win_pct:5.1f}% | "
                    f"Len={avg_len:.0f} | "
                    f"PolL={stats['policy_loss']:.4f} | "
                    f"ValL={stats['value_loss']:.4f} | "
                    f"Ent={stats['entropy']:.4f} | "
                    f"Steps={self.total_steps:,} | "
                    f"Time={elapsed:.0f}s"
                )

            # Save checkpoint
            if ep % self.cfg.save_interval == 0:
                self.save(os.path.join(self.cfg.save_dir, f'rl_iter_{ep}.pt'))

        # Final save
        self.save(os.path.join(self.cfg.save_dir, 'rl_final.pt'))

        elapsed = time.time() - start_time
        print(f"\nTraining complete! Total time: {elapsed:.0f}s")
        print(f"Best reward: {self.best_reward:+.3f}")
        print(f"Total steps: {self.total_steps:,}")

        return self.episode_rewards


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description='PPO bridge AI training')
    parser.add_argument('--episodes', type=int, default=10000, help='Number of episodes')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--batch', type=int, default=64, help='Rollout batch size')
    parser.add_argument('--hidden', type=int, default=256, help='Hidden dim')
    parser.add_argument('--save-interval', type=int, default=500, help='Save interval in episodes')
    parser.add_argument('--load', type=str, default=None, help='Load checkpoint path')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42, help='Environment seed')

    args = parser.parse_args()

    cfg = RLConfig(
        num_episodes=args.episodes,
        learning_rate=args.lr,
        batch_size=args.batch,
        hidden_dim=args.hidden,
        save_interval=args.save_interval,
        env_seed=args.seed,
    )

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

    trainer = PPOTrainer(cfg, device)

    if args.load:
        trainer.load(args.load)

    trainer.train(cfg.num_episodes)


if __name__ == '__main__':
    main()
