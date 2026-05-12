"""
模型评估器 - 计算模型输出匹配率
"""

import torch
import numpy as np
from typing import Tuple, Dict


class ModelEvaluator:
    """模型评估器"""
    
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        # Handle both config styles
        if hasattr(cfg, 'env'):
            self.num_bid_actions = cfg.env.num_bid_actions
            self.num_play_actions = cfg.env.num_play_actions
        else:
            self.num_bid_actions = cfg.num_bid_actions
            self.num_play_actions = cfg.num_play_actions
        
    def evaluate(self, model) -> Dict:
        """评估模型在训练数据上的表现"""
        from generate_training_data import BridgeDataGenerator
        
        # 生成测试数据
        gen = BridgeDataGenerator(num_samples=100)
        data_list = gen.generate(num_samples=100)
        
        # 模型预测
        model.eval()
        with torch.no_grad():
            total_bid_correct = 0
            total_play_correct = 0
            
            for obs_arr, bid_target, play_target in data_list:
                # 构建观测
                obs = torch.FloatTensor(obs_arr).to(self.device).unsqueeze(0)
                
                # 前向传播
                outputs = model(obs)
                if isinstance(outputs, tuple):
                    bid_logits, play_logits = outputs
                else:
                    # assume concatenated
                    bid_logits = outputs[:, :self.num_bid_actions]
                    play_logits = outputs[:, self.num_bid_actions:]
                
                # 预测
                bid_pred = torch.argmax(bid_logits, dim=1).item()
                play_pred = torch.argmax(play_logits, dim=1).item()
                
                if bid_pred == bid_target:
                    total_bid_correct += 1
                if play_pred == play_target:
                    total_play_correct += 1
            
            bid_accuracy = total_bid_correct / len(data_list) if data_list else 0.0
            play_accuracy = total_play_correct / len(data_list) if data_list else 0.0
            
            return {
                'bid_accuracy': bid_accuracy,
                'play_accuracy': play_accuracy
            }
