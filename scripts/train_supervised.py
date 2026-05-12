#!/usr/bin/env python3
"""
Stepwise DDS Training — complete bridge game training pipeline (using stepwise logic).

For each deal:
  1. Bidding step: compute DDS optimal bid + par loss, backward pass per step
  2. Play step: compute DDS optimal card + par loss, backward pass per step
  3. Opponents' hands are hidden per bridge rules

Uses:
- src/stepwise_utils.py for core training logic
- src/stepwise_trainer.py for full board trainer
- src/dds_teacher.py for DDS computation
- src/env_core.py for BridgeEnv

Usage:
    python train_supervised.py
    python train_supervised.py --episodes 1000 --lr 5e-6 --resume checkpoints/dds_ep200.pt
"""

from __future__ import annotations

import argparse
import os
import random as _random
import sys
import time
from typing import List, Optional, Tuple
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root and src to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from endplay.types import Player, Denom, Vul, Rank

from model_transformer import BridgeTransformerV2
from env_core import BridgeEnv
from config import EnvConfig

# Import stepwise core modules
from stepwise_utils import (
    DDSTeacher,
    format_hand,
    deal_log,
    NUM_BID,
    NUM_PLAY
)
from stepwise_trainer import FullBoardTrainer, run_val_game


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Stepwise DDS Bridge Training')
    parser.add_argument('--episodes', type=int, default=200,
                        help='Number of training episodes (boards)')
    parser.add_argument('--epochs', type=int, default=2,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for AdamW optimizer')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Save interval (episodes)')
    parser.add_argument('--checkpoint-dir', type=str,
                        default=os.path.join(os.path.dirname(__file__), '..', 'checkpoints'),
                        help='Checkpoint save directory')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--val-interval', type=int, default=20,
                        help='Validation interval (episodes)')
    args = parser.parse_args()

    # Seed everything for reproducibility
    set_seed(args.seed)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()

    # Setup directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Initialize model, teacher, optimizer
    model = BridgeTransformerV2().to(device)
    teacher = DDSTeacher()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Create trainer
    trainer = FullBoardTrainer(model, teacher, optimizer, device)
    env = BridgeEnv()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model.__class__.__name__}")
    print(f"Params: {total_params:,}")
    print(f"NUM_BID: {NUM_BID}, NUM_PLAY: {NUM_PLAY}")
    print()

    # Resume training if specified
    if args.resume:
        if os.path.exists(args.resume):
            ckpt = torch.load(args.resume, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            if 'optimizer_state_dict' in ckpt:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            print(f"Resumed from {args.resume}")
        else:
            print(f"Warning: checkpoint {args.resume} not found, starting fresh")

    print(f"Training for {args.epochs} epochs x {args.episodes} episodes = {args.epochs * args.episodes} total boards...")
    print(f"LR: {args.lr}")
    print()

    # Training metrics
    best_match_rate = 0.0
    checkpoint_path = os.path.join(args.checkpoint_dir, 'stepwise_dds_model.pt')

    # Start training
    for epoch in range(args.epochs):
        print("=" * 80)
        print(f"EPOCH {epoch + 1} / {args.epochs}")
        print("=" * 80)
        print()

        for board_idx in range(args.episodes):
            print(f"Board {board_idx + 1} / {args.episodes}")
            print("-" * 80)

            # Train one full board using stepwise logic
            result = trainer.train_board(env, board_idx=args.seed + epoch * args.episodes + board_idx)

            if result:
                board_loss = result.get('board_loss', 0.0)
                total_steps = result.get('n_steps', 0)
                print(f"  Loss: {board_loss:.4f}")

                # Logging & periodic validation
                if (board_idx + 1) % args.val_interval == 0:
                    print()
                    print("-" * 80)
                    print("Validation game")
                    print("-" * 80)
                    env.reset(board_num=10000 + args.seed + epoch + board_idx)
                    deal = env.deal
                    deal_log(deal)
                    run_val_game(env, model, device, teacher)
                    print("-" * 80)

                # Periodic save
                if (board_idx + 1) % args.save_interval == 0:
                    print()
                    print(f"Saving to {checkpoint_path}")
                    torch.save(model.state_dict(), checkpoint_path)
                    print()

            print()

        # Epoch save
        epoch_path = os.path.join(args.checkpoint_dir, f'stepwise_dds_epoch{epoch+1}.pt')
        print()
        print(f"Saving epoch {epoch+1} to {epoch_path}")
        torch.save(model.state_dict(), epoch_path)
        print()

    # Final save
    final_path = os.path.join(args.checkpoint_dir, 'stepwise_dds_final.pt')
    print()
    print("=" * 80)
    print(f"TRAINING COMPLETE")
    print("=" * 80)
    print()
    print(f"Total steps: {trainer.total_steps}")
    print(f"Bid accuracy: {trainer.bid_correct / max(trainer.bid_total, 1):.1%}")
    print(f"Play accuracy: {trainer.play_correct / max(trainer.play_total, 1):.1%}")
    print(f"Saving to {final_path}")
    torch.save(model.state_dict(), final_path)
    print()

    # Final validation
    print("=" * 80)
    print("FINAL VALIDATION GAME")
    print("=" * 80)
    env.reset(board_num=99999)
    deal = env.deal
    deal_log(deal)
    run_val_game(env, model, device, teacher)
    print("=" * 80)
    print()


def set_seed(seed: int):
    """Set random seed for all modules."""
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    main()
