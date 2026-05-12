#!/usr/bin/env python3
"""
Policy Training v2 - 使用保存的 action_values 数据训练模型
改进功能：
- 训练/验证集分割（80/20）
- 基于验证损失的早停机制
- 更丰富的评估指标
- 改进的学习率调度
"""

import argparse
import json
import random
import os
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
    
    # 随机打乱索引
    indices = list(range(total_size))
    random.seed(seed)
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 创建子数据集
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    return train_dataset, val_dataset


def evaluate_model(model, dataloader, device):
    """评估模型（计算验证损失和准确率）"""
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for batch in dataloader:
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
                    num_actions = 38
                else:
                    logits = play_logits[i]
                    num_actions = 52
                
                # 构建合法掩码
                legal_mask = torch.zeros(num_actions, device=device)
                for act in legal_actions:
                    if act < num_actions:
                        legal_mask[act] = 1.0
                
                # 计算损失
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
                
                # 计算准确率（预测最优动作）
                probs = torch.softmax(logits, dim=-1)
                _, predicted_act = torch.max(probs, dim=-1)
                if predicted_act == optimal_act:
                    correct_predictions += 1
                total_predictions += 1
            
            batch_loss = batch_loss / len(obs)
            total_loss += batch_loss.item() * len(obs)
    
    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0
    
    return avg_loss, accuracy


def train_policy(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 加载完整数据集
    full_dataset = PolicyDataset(args.data_file, phase=args.phase)
    
    # 分割训练/验证集
    train_dataset, val_dataset = split_dataset(full_dataset, train_ratio=0.8, seed=args.seed)
    print(f"Train set: {len(train_dataset)} samples, Val set: {len(val_dataset)} samples")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 初始化模型和优化器
    model = BridgeTransformerV2().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 从预训练模型恢复训练
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Loading pre-trained model from: {args.resume_from}")
        model.load_state_dict(torch.load(args.resume_from, map_location=device))
        print("✓ Model loaded successfully")
    elif args.resume_from:
        print(f"Warning: Model file not found: {args.resume_from}")
        print("Training from scratch instead")
    
    # 学习率调度器：基于验证损失
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    train_loss_history = []
    val_loss_history = []
    val_acc_history = []

    print("\n" + "="*80)
    print("Training Progress")
    print("="*80)
    print(f"{'Epoch':<6} {'Train Loss':<12} {'Val Loss':<12} {'Val Acc':<10} {'LR':<10} {'Notes'}")
    print("-"*80)

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        
        # 训练阶段
        for batch in train_loader:
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

            total_train_loss += batch_loss.item() * len(obs)

        avg_train_loss = total_train_loss / len(train_dataset)
        train_loss_history.append(avg_train_loss)
        
        # 验证阶段
        avg_val_loss, val_acc = evaluate_model(model, val_loader, device)
        val_loss_history.append(avg_val_loss)
        val_acc_history.append(val_acc)
        
        # 更新学习率调度器
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 打印进度
        notes = ""
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_train_loss = avg_train_loss
            best_epoch = epoch + 1
            torch.save(model.state_dict(), args.output_model)
            notes = "✓ Saved best model"
            patience_counter = 0
        else:
            patience_counter += 1
        
        print(f"{epoch+1:<6} {avg_train_loss:<12.4f} {avg_val_loss:<12.4f} {val_acc:<10.2%} {current_lr:<10.2e} {notes}")
        
        # 早停检查
        if patience_counter >= args.early_stop:
            print("\n" + "="*80)
            print(f"Early stopping at epoch {epoch+1} (patience exhausted)")
            break

    # 训练完成总结
    print("\n" + "="*80)
    print("Training Summary")
    print("="*80)
    print(f"Best epoch: {best_epoch}")
    print(f"Best train loss: {best_train_loss:.4f}")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Best val accuracy: {val_acc_history[best_epoch-1]:.2%}")
    print(f"Model saved to: {args.output_model}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Policy Training v2 (with validation)")
    parser.add_argument("--data-file", type=str, default="data/stepwise_training_data.jsonl")
    parser.add_argument("--output-model", type=str, default="checkpoints/policy_model_v2.pt")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Path to pre-trained model to resume training from")
    parser.add_argument("--phase", type=str, default=None, choices=["bidding", "play", None],
                       help="Phase to train (default: all)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stop", type=int, default=10,
                       help="Early stopping patience (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    print("="*80)
    print("POLICY TRAINING v2 (using action_values + validation)")
    print(f"  Phase: {args.phase or 'all'}")
    print(f"  Data: {args.data_file}")
    print(f"  Max epochs: {args.epochs}")
    print(f"  Early stop patience: {args.early_stop}")
    print(f"  LR: {args.lr}")
    print("="*80)
    
    train_policy(args)


if __name__ == "__main__":
    main()
