#!/usr/bin/env python3
"""
Stepwise DDS 训练 - 完整牌局逐步训练
入口文件：调用 src/stepwise_trainer
"""

import sys, os, argparse
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch
import torch.optim as optim

from config import EnvConfig, TrainConfig, set_seed
from model_transformer import BridgeTransformerV2
from env_core import BridgeEnv, PLAYER_ORDER

from stepwise_utils import (
    DDSTeacher,
    deal_log,
    NUM_BID, NUM_PLAY
)
from stepwise_trainer import FullBoardTrainer, run_val_game


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-boards', type=int, default=50, help='训练牌局数')
    parser.add_argument('--epochs', type=int, default=2, help='训练轮数')
    args = parser.parse_args()

    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    model = BridgeTransformerV2().to(device)
    teacher = DDSTeacher()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001)

    ckpt_dir = os.path.join(os.path.dirname(__file__), '..', 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'stepwise_dds_model.pt')

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f'Loaded model from {ckpt_path}\n')

    print(f'  Model: {model.__class__.__name__}')
    total_params = sum(p.numel() for p in model.parameters())
    print(f'  Params: {total_params:,}\n')

    print(f'  NUM_BID={NUM_BID} | NUM_PLAY={NUM_PLAY}\n')

    trainer = FullBoardTrainer(model, teacher, optimizer, device)
    env = BridgeEnv()

    for epoch in range(args.epochs):
        print()
        print('=' * 80)
        print(f'EPOCH {epoch + 1} / {args.epochs}')
        print('=' * 80)
        print()

        for i in range(args.num_boards):
            print(f'Board {i + 1} / {args.num_boards} (epoch {epoch + 1})')
            print('-' * 80)
            env.reset(board_num=i + 1)
            deal = env.deal
            deal_log(deal)

            result = trainer.train_board(env, board_idx=i + 1)

            print(f'  Board Loss: {result["board_loss"]:.4f}')
            print()

            if (i + 1) % 10 == 0:
                print()
                print('-' * 80)
                print('  Validation...')
                print('-' * 80)
                env.reset(board_num=1000 + i + 1)
                deal = env.deal
                deal_log(deal)
                run_val_game(env, model, device, teacher)
                print('-' * 80)
                print()

        print()
        print(f'Saving to {ckpt_path}')
        torch.save(model.state_dict(), ckpt_path)
        print()

        print()
        print('=' * 80)
        print('Final Stats:')
        print()
        print(f'  Total steps: {trainer.total_steps}')
        bid_acc = trainer.bid_correct / max(trainer.bid_total, 1) * 100
        play_acc = trainer.play_correct / max(trainer.play_total, 1) * 100
        total_acc = (trainer.bid_correct + trainer.play_correct) / max(trainer.bid_total + trainer.play_total, 1) * 100
        avg_loss = trainer.total_loss / max(trainer.total_steps, 1)
        print()
        print(f'  叫牌: {trainer.bid_correct}/{trainer.bid_total} ({bid_acc:.0f}%)')
        print(f'  出牌: {trainer.play_correct}/{trainer.play_total} ({play_acc:.0f}%)')
        print()
        print(f'  综合: {total_acc:.0f}%')
        print()
        print(f'  平均损失: {avg_loss:.4f}')
        print()
        print('=' * 80)
        print()

        print()
        print('=' * 80)
        print('Validation Game:')
        print('=' * 80)
        print()

        env.reset(board_num=99999)
        deal = env.deal
        deal_log(deal)
        run_val_game(env, model, device, teacher)

        print()
        print()


if __name__ == '__main__':
    main()
