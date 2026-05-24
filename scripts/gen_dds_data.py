#!/usr/bin/env python3
"""
桥牌训练数据生成 — 后台运行，带进度报告和自动保存
"""
import sys, os, json, time, datetime

project_root = '/mnt/d/gyh/Projects/TRAE/bridge-robot'
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
from config import set_seed
from generate_training_data import StepwiseBridgeDataGenerator

# ===== 配置 =====
NUM_BOARDS = 100
SAVE_INTERVAL = 10
OUTPUT_DIR = os.path.join(project_root, 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f'train_data_{datetime.datetime.now():%Y%m%d_%H%M}.json')
PROGRESS_FILE = os.path.join(OUTPUT_DIR, 'generation_progress.json')

def log(msg):
    t = datetime.datetime.now().strftime('%H:%M:%S')
    print(f'[{t}] {msg}', flush=True)

# ===== 开始 =====
log(f'========================================')
log(f'  桥牌训练数据生成')
log(f'  牌局数: {NUM_BOARDS}')
log(f'  保存间隔: {SAVE_INTERVAL} 局')
log(f'  输出: {OUTPUT_FILE}')
log(f'========================================')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log(f'  设备: {device}')
log(f'')

set_seed(42)
generator = StepwiseBridgeDataGenerator(seed=42, device=device)

t_start = time.time()
all_data = []

for i in range(NUM_BOARDS):
    t_board = time.time()
    
    board_data = generator.generate_board_data(board_idx=i+1, verbose=False)
    all_data.extend(board_data)
    
    elapsed = time.time() - t_board
    elapsed_total = time.time() - t_start
    
    # Progress summary
    steps_this_board = len(board_data)
    avg_speed = (i+1) / elapsed_total * 60  # boards per minute
    eta_min = (NUM_BOARDS - i - 1) / avg_speed if avg_speed > 0 else 0
    
    log(f'  [{i+1}/{NUM_BOARDS}] 完成 | {steps_this_board}步 | '
        f'{elapsed:.1f}s | avg {avg_speed:.1f}局/分 | ETA {eta_min:.0f}min')
    
    # Save at intervals
    if (i+1) % SAVE_INTERVAL == 0:
        with open(OUTPUT_FILE, 'w') as f:
            json.dump(all_data, f)
        
        # Save progress
        progress = {
            'boards_done': i+1,
            'total_boards': NUM_BOARDS,
            'total_steps': len(all_data),
            'elapsed_seconds': elapsed_total,
            'boards_per_minute': round(avg_speed, 1),
            'eta_minutes': round(eta_min, 1),
            'status': 'running'
        }
        with open(PROGRESS_FILE, 'w') as f:
            json.dump(progress, f, indent=2)
        
        log(f'  💾 保存: {len(all_data)}个训练步骤, ETA {eta_min:.0f}min')

# Final save
with open(OUTPUT_FILE, 'w') as f:
    json.dump(all_data, f)

total_time = time.time() - t_start
progress = {
    'boards_done': NUM_BOARDS,
    'total_boards': NUM_BOARDS,
    'total_steps': len(all_data),
    'elapsed_seconds': total_time,
    'boards_per_minute': round(NUM_BOARDS / total_time * 60, 1),
    'status': 'complete'
}
with open(PROGRESS_FILE, 'w') as f:
    json.dump(progress, f, indent=2)

log(f'')
log(f'========================================')
log(f'  ✅ 生成完成!')
log(f'  牌局: {NUM_BOARDS}')
log(f'  训练步骤: {len(all_data)}')
log(f'  耗时: {total_time:.0f}s ({total_time/60:.1f}min)')
log(f'  速度: {NUM_BOARDS/total_time*60:.1f} 局/分')
log(f'  输出: {OUTPUT_FILE}')
log(f'========================================')
