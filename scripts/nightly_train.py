#!/usr/bin/env python3
"""生成100局数据后训练到07:00"""
import sys, os, time, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
import numpy as np
import gc

from config import set_seed
from model_transformer import BridgeTransformerV2
from generate_training_data import StepwiseBridgeDataGenerator
from stepwise_utils import NUM_BID, NUM_PLAY

# ======== 配置 ========
TARGET_TIME = "07:00"
INIT_BOARDS = 100
BATCH_SIZE = 32
LR = 1e-4
EPOCHS_PER_CYCLE = 5
SAVE_EVERY = 300  # 保存间隔(秒)

device = torch.device('cuda')
print(f'Device: {device}')
print(f'Target: today {TARGET_TIME}')
print(f'Start: {datetime.now().strftime("%H:%M:%S")}')
print()

# 解析目标时间
now = datetime.now()
target = now.replace(hour=7, minute=0, second=0, microsecond=0)
if target <= now:
    target += timedelta(days=1)
max_runtime = (target - now).total_seconds()
print(f'Max runtime: {max_runtime/3600:.1f}h ({max_runtime:.0f}s)')
print()

# ======== 模型 & 优化器 ========
set_seed(42)
model = BridgeTransformerV2().to(device)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler() if torch.cuda.is_available() else None

# 尝试加载已有checkpoint
ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_base = os.path.join(ckpt_dir, 'nightly_model')
ckpt_files = sorted(glob.glob(f'{ckpt_base}_*.pt'))
if ckpt_files:
    latest_ckpt = ckpt_files[-1]
    model.load_state_dict(torch.load(latest_ckpt, map_location=device))
    print(f'Loaded checkpoint: {latest_ckpt}')
print()

# ======== 数据集 ========
class StepwiseDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        d = self.data[idx]
        obs = torch.tensor(d['obs'], dtype=torch.float32)
        act = d['optimal_act']
        phase = 0 if d['phase'] == 'bidding' else 1
        return obs, act, phase

# ======== 数据生成器 ========
gen = StepwiseBridgeDataGenerator(seed=42, device=device)

# ======== 训练循环 ========
all_data = []
total_steps = 0
cycle = 0
t_start = time.time()

while time.time() - t_start < max_runtime:
    cycle += 1
    t_cycle = time.time()
    
    # === 生成 ===
    print(f'\n[{datetime.now().strftime("%H:%M:%S")}] Cycle {cycle}: generating {INIT_BOARDS} boards...', flush=True)
    new_data = []
    for i in range(INIT_BOARDS):
        board_idx = cycle * INIT_BOARDS + i
        board_data = gen.generate_board_data(board_idx=board_idx)
        new_data.extend(board_data)
    all_data.extend(new_data)
    print(f'  Total data: {len(all_data)} steps (+{len(new_data)})', flush=True)
    
    # === 训练 ===
    dataset = StepwiseDataset(all_data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    bid_correct = 0
    bid_total = 0
    play_correct = 0
    play_total = 0
    
    for epoch in range(EPOCHS_PER_CYCLE):
        epoch_loss = 0.0
        epoch_correct = 0
        for obs, optimal_act, phase in loader:
            obs = obs.to(device)
            optimal_act = optimal_act.to(device)
            phase = phase.to(device)
            
            with torch.amp.autocast('cuda', enabled=scaler is not None):
                bid_logits, play_logits = model(obs)
                loss = 0.0
                correct = 0
                for i in range(len(obs)):
                    if phase[i].item() == 0:
                        l = criterion(bid_logits[i].unsqueeze(0), optimal_act[i].unsqueeze(0))
                        if bid_logits[i].argmax().item() == optimal_act[i].item():
                            bid_correct += 1
                        bid_total += 1
                    else:
                        l = criterion(play_logits[i].unsqueeze(0), optimal_act[i].unsqueeze(0))
                        if play_logits[i].argmax().item() == optimal_act[i].item():
                            play_correct += 1
                        play_total += 1
                    loss += l
                loss = loss / len(obs)
                total_loss += loss.item() * len(obs)
                total_samples += len(obs)
            
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad()
        
        # 每个epoch打完报告
        bid_acc = bid_correct / max(bid_total, 1) * 100
        play_acc = play_correct / max(play_total, 1) * 100
        avg_loss = total_loss / max(total_samples, 1)
        print(f'  Epoch {epoch+1}/{EPOCHS_PER_CYCLE}: loss={avg_loss:.4f} bid={bid_acc:.1f}% play={play_acc:.1f}%', flush=True)
    
    # 清理缓存
    del dataset, loader
    gc.collect()
    torch.cuda.empty_cache()
    
    # === 定期保存 ===
    elapsed = time.time() - t_start
    remaining = max_runtime - elapsed
    bid_acc = bid_correct / max(bid_total, 1) * 100
    play_acc = play_correct / max(play_total, 1) * 100
    avg_loss = total_loss / max(total_samples, 1)
    print(f'  [Cycle {cycle} done] {len(all_data)} steps, loss={avg_loss:.4f}, bid={bid_acc:.1f}%, play={play_acc:.1f}%, remaining={remaining/60:.0f}min', flush=True)
    
    ckpt_path = f'{ckpt_base}_{cycle:03d}.pt'
    torch.save(model.state_dict(), ckpt_path)
    print(f'  Saved: {ckpt_path}', flush=True)
    
    # 也保存全部数据
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', f'nightly_data_cycle{cycle:03d}.json')
    with open(data_path, 'w') as f:
        json.dump({'num_boards': cycle * INIT_BOARDS, 'total_steps': len(all_data), 'data': all_data}, f, indent=1)
    print(f'  Saved: {data_path}', flush=True)

# === 最终报告 ===
elapsed = time.time() - t_start
print(f'\n{"="*60}')
print(f'Training complete at {datetime.now().strftime("%H:%M:%S")}')
print(f'Total boards: {cycle * INIT_BOARDS}')
print(f'Total steps: {len(all_data)}')
print(f'Training time: {elapsed/60:.0f}min')
bid_acc = (bid_correct / max(bid_total, 1)) * 100
play_acc = (play_correct / max(play_total, 1)) * 100
total_acc = (bid_correct + play_correct) / max(bid_total + play_total, 1) * 100
print(f'Final: bid={bid_acc:.1f}% play={play_acc:.1f}% total={total_acc:.1f}%')
print(f'Model: {ckpt_base}_final.pt')
torch.save(model.state_dict(), f'{ckpt_base}_final.pt')
print(f'{"="*60}')
