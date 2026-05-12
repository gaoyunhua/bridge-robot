#!/usr/bin/env python3
"""桥牌 AI 训练 - 生成数据 + 100000 轮训练（CPU 版，日志到文件）"""
import os, sys, numpy as np, torch, random, time, math

project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))

from config import EnvConfig
from model_transformer import BridgeTransformerV2
from torch.utils.data import Dataset, DataLoader

CFG = EnvConfig()
EPOCHS = 100000
BATCH_SIZE = 64
LR = 0.001
DATA_DIR = os.path.join(project_root, 'data')
DATA_FILE = os.path.join(DATA_DIR, 'training_data.txt')
CKPT_DIR = os.path.join(project_root, 'checkpoints')
LOG_FILE = os.path.join(CKPT_DIR, 'train.log')
os.makedirs(CKPT_DIR, exist_ok=True)

def log(msg):
    with open(LOG_FILE, 'a') as f:
        f.write(f"{msg}\n")
    print(msg)

device = torch.device('cpu')

class SimpleDataset(Dataset):
    def __init__(self, data_file, obs_dim=757):
        t0 = time.time()
        log(f"加载数据: {data_file}...")
        raw = np.loadtxt(data_file, delimiter=',', skiprows=1)  # skip header
        # CSV格式: obs_757d (757 cols) | bid_label (1 col) | play_label (1 col) = 759 cols total
        self.data = raw
        self.obs_dim = obs_dim
        log(f"  加载完成: {len(self.data)} 行 x {raw.shape[1]} 列, 耗时 {time.time()-t0:.1f}s")
        log(f"  格式: {obs_dim}维观测 + bid_label + play_label = {obs_dim + 2}列")
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data[idx]
        obs = torch.FloatTensor(row[:self.obs_dim])
        bid = torch.tensor(int(row[self.obs_dim]), dtype=torch.long)
        play = torch.tensor(int(row[self.obs_dim + 1]), dtype=torch.long)
        return obs, bid, play

log("=" * 60)
log("桥牌 AI 训练 - 100000 轮")
log("=" * 60)

t0 = time.time()
dataset = SimpleDataset(DATA_FILE)
log(f"数据集: {len(dataset)} 样本")

train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
log(f"训练: {train_size} | 验证: {val_size} | Batch: {BATCH_SIZE}")

model = BridgeTransformerV2(d_model=256, nhead=4, num_layers=4)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR*3, total_steps=EPOCHS)
criterion = torch.nn.CrossEntropyLoss()

best_acc = 0.0
log_interval = 1000
total_train_time = 0
steps = 0

log(f"\n开始 {EPOCHS} 轮训练...\n")

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    for obs, bid_labels, play_labels in train_loader:
        bid_logits, _ = model(obs)
        loss = criterion(bid_logits, bid_labels)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0.0
    correct_bid = correct_play = total = 0
    with torch.no_grad():
        for obs, bid_labels, play_labels in val_loader:
            bid_logits, play_logits = model(obs)
            loss_bid = criterion(bid_logits, bid_labels)
            loss_play = criterion(play_logits, play_labels)
            val_loss += loss_bid.item() + loss_play.item()
            _, bid_pred = torch.max(bid_logits, 1)
            _, play_pred = torch.max(play_logits, 1)
            correct_bid += (bid_pred == bid_labels).sum().item()
            correct_play += (play_pred == play_labels).sum().item()
            total += bid_labels.size(0)

    bid_acc = correct_bid / total
    play_acc = correct_play / total
    combined_acc = (bid_acc + play_acc) / 2
    scheduler.step()
    steps += 1

    if epoch == 1 or epoch % log_interval == 0 or epoch == EPOCHS:
        elapsed = time.time() - t0
        epochs_per_sec = epoch / elapsed if elapsed > 0 else 0
        eta = (EPOCHS - epoch) / epochs_per_sec if epochs_per_sec > 0 else 0
        lr_now = optimizer.param_groups[0]['lr']
        log(f"[{epoch:6d}/{EPOCHS}]  "
            f"Loss: {train_loss/len(train_loader):.4f}/{val_loss/len(val_loader):.4f}  "
            f"Bid:{bid_acc*100:.2f}% Play:{play_acc*100:.2f}% Comb:{combined_acc*100:.2f}%  "
            f"LR:{lr_now:.6f}  "
            f"{elapsed/3600:.1f}h | ETA {eta/3600:.1f}h")

    if combined_acc > best_acc:
        best_acc = combined_acc
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'bid_acc': bid_acc, 'play_acc': play_acc, 'combined_acc': combined_acc
        }, os.path.join(CKPT_DIR, 'best_model.pt'))

    if epoch % 10000 == 0:
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, os.path.join(CKPT_DIR, f'checkpoint_iter_{epoch}.pt'))
        log(f"  -> 保存 checkpoint @ epoch {epoch}")

log("\n" + "=" * 60)
log(f"训练完成！最佳: {best_acc*100:.2f}% 总用时: {(time.time()-t0)/3600:.1f}h")
log("=" * 60)
torch.save(model.state_dict(), os.path.join(CKPT_DIR, 'final_model.pt'))
