#!/usr/bin/env python3
"""
Stepwise 训练数据生成 - 使用完整牌局逐步生成
生成完整的叫牌和出牌训练轨迹
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Tuple

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import torch

from config import EnvConfig, set_seed
from env_core import BridgeEnv
from model_transformer import BridgeTransformerV2
from dds_teacher import BidTeacher, PlayTeacher, DDTableCache
from stepwise_utils import (
    DDSTeacher, hide_opponent_cards,
    process_bidding_step, process_play_step,
    NUM_BID, NUM_PLAY
)


class StepwiseBridgeDataGenerator:
    """Stepwise 数据生成器 - 完整牌局逐步生成"""

    def __init__(self, seed: int = 42, device=None):
        self.seed = seed
        set_seed(seed)
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 初始化
        self.env = BridgeEnv()
        self.model = BridgeTransformerV2().to(self.device)
        self.teacher = DDSTeacher()

    def generate_board_data(self, board_idx: int = 1, verbose: bool = False) -> List[Dict]:
        """生成单个完整牌局的所有训练步骤"""
        board_data = []
        
        self.env.reset(board_num=board_idx)
        obs_orig = self.env._encode_obs()
        deal = self.env.deal
        vul = self.env.vul
        dealer = self.env.dealer
        dd_table = DDTableCache(deal)
        
        if verbose:
            print("\n┌─────────────────────────────────────────────────────┐")
            print(f"│ Board {board_idx} - Dealer: {dealer.name}, Vul: {vul} │")
            print(f"└─────────────────────────────────────────────────────┘")
            print("Bidding History: ", end="", flush=True)
        
        bidding_sequence = []
        
        # 叫牌阶段
        while not self.env.done and self.env.phase == 'bidding':
            if verbose:
                print(".", end="", flush=True)
            step_data = process_bidding_step(
                self.env, self.model, self.teacher, dd_table,
                deal, vul, dealer, obs_orig, self.device
            )
            
            # 存储训练数据点
            data_point = {
                'phase': 'bidding',
                'obs': step_data['obs_hidden'].tolist(),  # 隐牌用于训练
                'model_act': step_data['model_act'],
                'optimal_act': step_data['optimal_act'],
                'legal_actions': step_data['legal_actions_int'],
                'action_losses': step_data['bid_action_losses'],
                'action_values': step_data['action_values'],
                'player_idx': step_data['player_idx'],
                'board_idx': board_idx
            }
            board_data.append(data_point)
            
            # 获取叫牌字符串
            act = step_data['model_act']
            if act < 35:
                level = act // 5 + 1
                denom = ['C', 'D', 'H', 'S', 'NT'][act % 5]
                bid_str = f"{level}{denom}"
            elif act == 35:
                bid_str = "Pass"
            elif act == 36:
                bid_str = "Double"
            else:
                bid_str = "Redouble"
            
            bidding_sequence.append(bid_str)
            
            # 模型动作优先（如果合法）
            model_act_is_legal = step_data['model_act'] in step_data['legal_actions_int']
            advance_act = step_data['model_act'] if model_act_is_legal else step_data['optimal_act']
            with torch.no_grad():
                self.env.step(advance_act)
            obs_orig = self.env._encode_obs()
        
        if verbose:
            print(" → ".join(bidding_sequence))
            if self.env.bidding.contract_level > 0:
                contract = self.env.bidding._contract
                if contract and not contract.is_passout():
                    print(f"Final Contract: {contract}")
                else:
                    print("Final Contract: Passed out")
            else:
                print("Final Contract: Passed out")
            
            # 显示每个玩家的手牌（训练时对手手牌会被隐藏）
            print("\nPlayer Hands (training hides opponent cards):")
            from endplay.types import Player
            for player in [Player.north, Player.east, Player.south, Player.west]:
                hand = deal[player]
                print(f"  {player.name}: {hand}")
        
        # 出牌阶段
        if not self.env.done and self.env.phase == 'play':
            declarer = self.env.bidding.declarer
            declarer_idx = declarer.value
            dummy = declarer.next().next()
            dummy_idx = dummy.value
            lead_player = declarer.next()

            obs_orig = self.env._encode_obs()
            
            play_cards = []
            
            if verbose:
                print("\nPlay History:")
            
            while not self.env.done and self.env.phase == 'play':
                step_data = process_play_step(
                    self.env, self.model, self.teacher, dd_table, obs_orig,
                    declarer, declarer_idx, dummy, dummy_idx, lead_player, self.device
                )
                
                if step_data is None:
                    break
                
                # 存储训练数据点
                data_point = {
                    'phase': 'play',
                    'obs': step_data['obs_hidden'].tolist(),  # 隐牌用于训练
                    'model_act': step_data['model_act'],
                    'optimal_act': step_data['optimal_act'],
                    'legal_actions': step_data['legal_actions_int'],
                    'action_losses': step_data['play_action_losses'],
                    'action_values': step_data['action_values'],
                    'player_idx': step_data['player_idx'],
                    'board_idx': board_idx
                }
                board_data.append(data_point)
                
                # 收集出牌
                card_act = step_data['model_act']
                from dds_teacher import card_idx_to_name
                card_name = card_idx_to_name(card_act)
                current_player_name = ['north', 'east', 'south', 'west'][step_data['player_idx']]
                play_cards.append(f"{current_player_name}:{card_name}")
                
                # 每4张牌（一轮）显示一次
                if verbose and len(play_cards) % 4 == 0:
                    print(f"  Trick {len(play_cards)//4}: {' '.join(play_cards[-4:])}")
                
                # 模型动作优先（如果合法）
                model_act_is_legal = step_data['model_act'] in step_data['legal_actions_int']
                advance_act = step_data['model_act'] if model_act_is_legal else step_data['optimal_act']
                with torch.no_grad():
                    self.env.step(advance_act)
                obs_orig = self.env._encode_obs()
        
        return board_data

    def generate(self, num_boards: int = 100, output_file: str = None, save_interval: int = 5, verbose: bool = False) -> List[Dict]:
        """生成训练数据，每 save_interval 个牌局保存一次"""
        all_data = []
        
        for i in range(num_boards):
            print(f"生成牌局 {i+1} / {num_boards}")
            board_data = self.generate_board_data(board_idx=i+1, verbose=verbose)
            all_data.extend(board_data)
            
            # 每 save_interval 个牌局保存一次
            if (i+1) % save_interval == 0:
                print(f"已生成 {len(all_data)} 个训练步骤")
                if output_file:
                    self.save_json(all_data[-self._count_recent_steps(save_interval):], output_file, append=True)
                    print(f"已保存最近 {save_interval} 个牌局的数据到 {output_file}")
        
        return all_data
    
    def _count_recent_steps(self, num_boards: int) -> int:
        """估算最近 num_boards 个牌局对应的步骤数"""
        return num_boards * 30  # 平均每个牌局约30个步骤

    def save_json(self, data: List[Dict], output_file: str, append: bool = False):
        """保存数据为 JSONL"""
        mode = 'a' if append else 'w'
        with open(output_file, mode, encoding='utf-8') as f:
            for dp in data:
                f.write(json.dumps(dp, ensure_ascii=False) + '\n')
        print(f"数据已{'追加到' if append else '保存到'} {output_file}")

    def save_npz(self, data: List[Dict], output_file: str):
        """保存为 NPZ"""
        # 整理数据
        obs_list = []
        phase_list = []
        optimal_act_list = []
        
        for dp in data:
            obs_list.append(dp['obs'])
            phase_list.append(0 if dp['phase'] == 'bidding' else 1)
            optimal_act_list.append(dp['optimal_act'])
        
        np.savez(
            output_file,
            obs=np.array(obs_list, dtype=np.float32),
            phase=np.array(phase_list, dtype=np.int32),
            optimal_act=np.array(optimal_act_list, dtype=np.int32)
        )
        print(f"数据已保存到 {output_file}")


def main():
    print("=" * 80)
    print("Stepwise 桥牌训练数据生成")
    print("=" * 80)
    
    # 配置
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-boards', type=int, default=100, help='每次循环生成的牌局数')
    parser.add_argument('--output', type=str, default='data/stepwise_training_data.jsonl', help='输出文件')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true', help='覆盖现有文件而不是追加')
    parser.add_argument('--cycle', type=int, default=1, help='循环次数，每次循环生成 num_boards 个牌局并追加保存')
    parser.add_argument('--verbose', '-v', action='store_true', help='显示详细的叫牌过程')
    parser.add_argument('--save-interval', type=int, default=5, help='每多少个牌局保存一次数据')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 初始化生成器
    generator = StepwiseBridgeDataGenerator(seed=args.seed)
    
    # 总统计
    total_boards = 0
    total_steps = 0
    
    # 循环生成数据
    for cycle_idx in range(args.cycle):
        print(f"\n{'='*80}")
        print(f"循环 {cycle_idx + 1} / {args.cycle}")
        print(f"{'='*80}")
        
        # 设置当前循环的种子
        set_seed(args.seed + cycle_idx * 1000)
        
        # 生成数据
        print(f"开始生成 {args.num_boards} 个牌局...")
        data = generator.generate(
            num_boards=args.num_boards,
            output_file=args.output if cycle_idx > 0 else None,
            save_interval=args.save_interval,
            verbose=args.verbose
        )
        print(f"本循环生成 {len(data)} 个训练步骤")
        
        # 更新统计
        total_boards += args.num_boards
        total_steps += len(data)
        
        # 保存数据（第一次循环根据 overwrite 参数决定，后续循环都追加）
        if cycle_idx == 0:
            append_mode = not args.overwrite
        else:
            append_mode = True
        
        generator.save_json(data, args.output, append=append_mode)
        
        # 如果是追加模式，删除旧的 NPZ 文件（只在第一次追加时删除）
        if append_mode and cycle_idx == 0:
            npz_file = args.output.replace('.jsonl', '.npz')
            if os.path.exists(npz_file):
                os.remove(npz_file)
                print(f"已删除旧的 NPZ 文件: {npz_file}")
    
    # 最终保存一个完整的 NPZ 版本（读取所有数据）
    print("\n" + "="*80)
    print("合并并保存 NPZ 文件...")
    all_data = []
    with open(args.output, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                all_data.append(json.loads(line))
    
    npz_file = args.output.replace('.jsonl', '.npz')
    generator.save_npz(all_data, npz_file)
    
    print("\n数据生成完成！")
    print()
    print("统计:")
    print(f"  循环次数: {args.cycle}")
    print(f"  牌局数: {total_boards}")
    print(f"  总训练步骤: {total_steps}")


if __name__ == '__main__':
    main()
#