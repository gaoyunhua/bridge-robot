#!/usr/bin/env python3
"""
Policy Training - 使用保存的 action_losses 数据训练模型
利用 DDS 评估的相对价值来指导训练
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from model_transformer import BridgeTransformerV2
from config import set_seed
from dds_teacher import compute_full_policy_loss


class PolicyDataset(Dataset):
    def __init__(self, data_file, phase=None):
        self.data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    dp = json.loads(line.strip())
                    if phase is None or dp["phase"] == phase:
                        self.data.append(dp)
                except Exception:
                    pass
        print(f"Loaded {len(self.data)} samples (phase={phase})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dp = self.data[idx]
        obs = torch.tensor(dp["obs"], dtype=torch.float32)
        legal_actions = dp.get("legal_actions", [])
        action_losses = dp.get("action_losses", [])
        action_values = dp.get("action_values", [])
        phase = dp["phase"]
        optimal_act = dp.get("optimal_act", 0)
        return obs, legal_actions, action_losses, action_values, phase, optimal_act


def collate_fn(batch):
    obs = torch.stack([item[0] for item in batch])
    legal_actions = [item[1] for item in batch]
    action_losses = [item[2] for item in batch]
    action_values = [item[3] for item in batch]
    phases = [item[4] for item in batch]
    optimal_acts = [item[5] for item in batch]
    return obs, legal_actions, action_losses, action_values, phases, optimal_acts


def train_policy(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BridgeTransformerV2().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    dataset = PolicyDataset(args.data_file, phase=args.phase)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            obs, legal_actions_list, action_losses_list, action_values_list, phases, optimal_acts = batch
            obs = obs.to(device)
            
            bid_logits, play_logits = model(obs)
            
            batch_loss = 0.0
            for i in range(len(obs)):
                phase = phases[i]
                legal_actions = legal_actions_list[i]
                action_losses = action_losses_list[i]
                action_values = action_values_list[i]
                
                if phase == "bidding":
                    logits = bid_logits[i]
                    num_actions = 38
                else:
                    logits = play_logits[i]
                    num_actions = 52
                
                legal_mask = torch.zeros(num_actions, device=device)
                for act in legal_actions:
                    if act < num_actions:
                        legal_mask[act] = 1.0
                
                loss = compute_full_policy_loss(
                    logits.unsqueeze(0),
                    legal_mask.unsqueeze(0),
                    action_values,
                    legal_actions,
                    temperature=1.0,
                    illegal_penalty=1.0,
                    negative_penalty_weight=2.0,
                    phase=phase
                )
                batch_loss += loss
            
            batch_loss = batch_loss / len(obs)
            
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += batch_loss.item() * len(obs)

        avg_loss = total_loss / len(dataset)
        
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch+1}/{args.epochs} | LR: {current_lr:.2e} | Loss: {avg_loss:.4f}")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), args.output_model)
            print(f"  ✓ Saved best model (Loss: {best_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 10:
            print(f"Early stopping at epoch {epoch+1}")
            break

    print("\n" + "="*60)
    print(f"Training complete! Best Loss: {best_loss:.4f}")
    print(f"Model saved to: {args.output_model}")


def main():
    parser = argparse.ArgumentParser(description="Policy Training")
    parser.add_argument("--data-file", type=str, default="data/stepwise_training_data.jsonl")
    parser.add_argument("--output-model", type=str, default="checkpoints/policy_model.pt")
    parser.add_argument("--phase", type=str, default=None, choices=["bidding", "play", None],
                       help="Phase to train (default: all)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("="*60)
    print("POLICY TRAINING (using action_values)")
    print(f"  Phase: {args.phase or 'all'}")
    print(f"  Data: {args.data_file}")
    print(f"  Epochs: {args.epochs}")
    print(f"  LR: {args.lr}")
    print("="*60)
    
    train_policy(args)


if __name__ == "__main__":
    main()