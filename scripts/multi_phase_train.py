#!/usr/bin/env python3
"""
Multi-Phase Training Script
Trains model with multiple epoch phases on all JSONL files under data directory
"""

import argparse
import json
import random
import os
import glob
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from tqdm import tqdm  # Progress bar

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from model_transformer import BridgeTransformerV2
from config import set_seed, EnvConfig
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
        print(f"Loaded {len(self.data)} samples from {os.path.basename(data_file)} (phase={phase})")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dp = self.data[idx]
        obs = torch.tensor(dp["obs"], dtype=torch.float32)
        legal_actions = dp.get("legal_actions", [])
        action_values = dp.get("action_values", [])
        phase = dp["phase"]
        optimal_act = dp.get("optimal_act", 0)
        return obs, legal_actions, action_values, phase, optimal_act


def collate_fn(batch):
    obs = torch.stack([item[0] for item in batch])
    legal_actions = [item[1] for item in batch]
    action_values = [item[2] for item in batch]
    phases = [item[3] for item in batch]
    optimal_acts = [item[4] for item in batch]
    return obs, legal_actions, action_values, phases, optimal_acts


def split_dataset(dataset, train_ratio=0.8, seed=42):
    """分割训练集和验证集"""
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = total_size - train_size
    
    indices = list(range(total_size))
    random.seed(seed)
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    return train_dataset, val_dataset


def train_one_epoch(model, dataloader, optimizer, device, epoch, phase_name):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    cfg = EnvConfig()
    
    # Add progress bar for training
    progress_bar = tqdm(dataloader, desc=f"  [{phase_name}] Epoch {epoch}", unit="batch")
    for batch in progress_bar:
        obs, legal_actions_list, action_values_list, phases, optimal_acts = batch
        obs = obs.to(device)
        
        bid_logits, play_logits = model(obs)
        
        batch_loss = 0.0
        for i in range(len(obs)):
            phase = phases[i]
            legal_actions = legal_actions_list[i]
            action_values = action_values_list[i]
            
            if phase == "bidding":
                logits = bid_logits[i]
                num_actions = cfg.num_bid_actions
            else:
                logits = play_logits[i]
                num_actions = cfg.num_play_actions
            
            legal_mask = torch.zeros(num_actions, device=device)
            legal_mask[legal_actions] = 1.0
            
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
        optimizer.step()
        
        total_loss += batch_loss.item()
        num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    print(f"  [{phase_name}] Epoch {epoch}: Train Loss = {avg_loss:.6f}")
    return avg_loss


def evaluate_model(model, dataloader, device):
    """评估模型"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    cfg = EnvConfig()
    
    with torch.no_grad():
        # Add progress bar for evaluation
        progress_bar = tqdm(dataloader, desc="  Evaluating", unit="batch")
        for batch in progress_bar:
            obs, legal_actions_list, action_values_list, phases, optimal_acts = batch
            obs = obs.to(device)
            
            bid_logits, play_logits = model(obs)
            
            batch_loss = 0.0
            for i in range(len(obs)):
                phase = phases[i]
                legal_actions = legal_actions_list[i]
                action_values = action_values_list[i]
                optimal_act = optimal_acts[i]
                
                if phase == "bidding":
                    logits = bid_logits[i]
                    num_actions = cfg.num_bid_actions
                else:
                    logits = play_logits[i]
                    num_actions = cfg.num_play_actions
                
                legal_mask = torch.zeros(num_actions, device=device)
                legal_mask[legal_actions] = 1.0
                
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
                
                probs = torch.softmax(logits, dim=0)
                masked_probs = probs * legal_mask
                predicted_action = torch.argmax(masked_probs).item()
                
                if predicted_action == optimal_act:
                    correct_predictions += 1
                total_predictions += 1
            
            total_loss += batch_loss.item()
    
    avg_loss = total_loss / total_predictions if total_predictions > 0 else 0.0
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="Multi-Phase Training Script")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing JSONL files")
    parser.add_argument("--phases", type=int, nargs="+", default=[20, 10, 5, 3, 2, 1],
                       help="Epochs for each phase (default: 20 10 5 3 2 1)")
    parser.add_argument("--model-path", type=str, default="models/policy_model_v2.pt",
                       help="Path to save/load model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Find all JSONL files
    jsonl_files = sorted(glob.glob(os.path.join(args.data_dir, "*.jsonl")))
    if not jsonl_files:
        print(f"No JSONL files found in {args.data_dir}")
        return
    
    print(f"Found {len(jsonl_files)} JSONL files:")
    for f in jsonl_files:
        print(f"  - {os.path.basename(f)}")
    
    # Load all datasets with progress bar
    print("\nLoading datasets...")
    all_datasets = []
    for f in tqdm(jsonl_files, desc="Loading files", unit="file"):
        dataset = PolicyDataset(f)
        all_datasets.append(dataset)
    
    full_dataset = ConcatDataset(all_datasets)
    print(f"\nTotal samples: {len(full_dataset)}")
    
    # Split into train/val
    train_dataset, val_dataset = split_dataset(full_dataset, train_ratio=0.8, seed=args.seed)
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=0)
    
    # Initialize model
    model = BridgeTransformerV2()
    if os.path.exists(args.model_path):
        print(f"\nLoading existing model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    
    # Create model directory if needed
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    # Multi-phase training
    print(f"\n{'='*60}")
    print(f"Starting multi-phase training with phases: {args.phases}")
    print(f"{'='*60}\n")
    
    total_epochs = 0
    best_val_loss = float('inf')
    
    for phase_idx, epochs_in_phase in enumerate(args.phases):
        phase_name = f"Phase {phase_idx + 1}/{len(args.phases)}"
        print(f"\n{'-'*60}")
        print(f"{phase_name}: {epochs_in_phase} epochs")
        print(f"{'-'*60}")
        
        for epoch in range(1, epochs_in_phase + 1):
            total_epochs += 1
            
            # Train
            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, phase_name)
            
            # Evaluate
            val_loss, val_acc = evaluate_model(model, val_loader, device)
            print(f"  [{phase_name}] Epoch {epoch}: Val Loss = {val_loss:.6f}, Val Acc = {val_acc:.4f}")
            
            scheduler.step(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), args.model_path)
                print(f"  [SAVED] Best model (Val Loss: {best_val_loss:.6f})")
        
        # Save checkpoint after phase
        phase_checkpoint = args.model_path.replace('.pt', f'_phase{phase_idx + 1}.pt')
        torch.save(model.state_dict(), phase_checkpoint)
        print(f"\n[CHECKPOINT] Saved phase {phase_idx + 1} model to {phase_checkpoint}")
    
    print(f"\n{'='*60}")
    print(f"Training complete! Total epochs: {total_epochs}")
    print(f"Best Val Loss: {best_val_loss:.6f}")
    print(f"Final model saved to: {args.model_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
