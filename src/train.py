#!/usr/bin/env python3
"""
Stepwise Training Script
使用 generate_training_data.py 生成的数据进行训练
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from config import EnvConfig, set_seed
from model_transformer import BridgeTransformerV2
from stepwise_utils import NUM_BID, NUM_PLAY


class StepwiseDataset(Dataset):
    def __init__(self, data_file):
        self.data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    dp = json.loads(line.strip())
                    self.data.append(dp)
                except Exception:
                    pass
        print(f"Loaded {len(self.data)} training samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        dp = self.data[idx]
        obs = torch.tensor(dp["obs"], dtype=torch.float32)
        optimal_act = dp["optimal_act"]
        phase = dp["phase"]
        phase_idx = 0 if phase == "bidding" else 1
        return obs, optimal_act, phase_idx


def train_model(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = BridgeTransformerV2().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()

    dataset = StepwiseDataset(args.data_file)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            obs, optimal_act, phase = batch
            obs = obs.to(device)
            optimal_act = optimal_act.to(device)
            phase = phase.to(device)
            bid_logits, play_logits = model(obs)
            loss = 0.0
            for i in range(len(obs)):
                if phase[i].item() == 0:
                    loss += criterion(bid_logits[i].unsqueeze(0), optimal_act[i].unsqueeze(0))
                else:
                    loss += criterion(play_logits[i].unsqueeze(0), optimal_act[i].unsqueeze(0))
            loss = loss / len(obs)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item() * len(obs)
        avg_train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                obs, optimal_act, phase = batch
                obs = obs.to(device)
                optimal_act = optimal_act.to(device)
                phase = phase.to(device)
                bid_logits, play_logits = model(obs)
                loss = 0.0
                for i in range(len(obs)):
                    if phase[i].item() == 0:
                        loss += criterion(bid_logits[i].unsqueeze(0), optimal_act[i].unsqueeze(0))
                        pred = bid_logits[i].argmax().item()
                    else:
                        loss += criterion(play_logits[i].unsqueeze(0), optimal_act[i].unsqueeze(0))
                        pred = play_logits[i].argmax().item()
                    if pred == optimal_act[i].item():
                        val_correct += 1
                loss = loss / len(obs)
                total_val_loss += loss.item() * len(obs)
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(os.path.dirname(args.output_model), exist_ok=True)
            torch.save(model.state_dict(), args.output_model)
            print("Saved best model")

    print("Training complete")


def main():
    parser = argparse.ArgumentParser(description="Stepwise Bridge Training")
    parser.add_argument("--data-file", type=str, default="data/stepwise_training_data.jsonl")
    parser.add_argument("--output-model", type=str, default="checkpoints/stepwise_bridge_model.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_model(args)


if __name__ == "__main__":
    main()
