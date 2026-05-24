#!/usr/bin/env python3
"""
Show board data using endplay library with beautiful formatting

Usage:
    python show_board_from_data.py                    # Use default file
    python show_board_from_data.py -d data.jsonl     # Use custom file
    python show_board_from_data.py --data-file data.jsonl
    
Navigation:
    Page Up / Page Down: Navigate between boards
    Home / End: Jump to first/last board
    Q: Quit
"""
import sys
import os
import json
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

import endplay
from endplay.types import Player, Bid, Contract, Card

from config import EnvConfig


def action_to_bid_name(action):
    """将动作转换为叫牌名称"""
    if action < 35:
        level = action // 5 + 1
        denom = ['C', 'D', 'H', 'S', 'NT'][action % 5]
        return f"{level}{denom}"
    elif action == 35:
        return "Pass"
    elif action == 36:
        return "X"
    else:
        return "XX"


def card_action_to_name(action):
    """将出牌动作转换为牌张名称"""
    if action < 52:
        suits = ['C', 'D', 'H', 'S']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        suit_idx = action // 13
        rank_idx = action % 13
        return f"{ranks[rank_idx]}{suits[suit_idx]}"
    return f"Action{action}"


def load_data(data_file):
    """加载数据文件"""
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        return None
    
    data = []
    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    return data


def display_board_endplay(data, index, total):
    """使用 endplay 显示单个 board"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    board = data[index]
    board_idx = board.get('board_idx', index + 1)
    player_idx = board.get('player_idx', 0)
    phase = board.get('phase', 'unknown')
    
    # 使用 endplay 打印标题
    print("=" * 80)
    print(f"  Bridge Board {board_idx} [{index + 1}/{total}] - {phase.upper()} Phase")
    print("=" * 80)
    print()
    
    # 显示当前玩家信息
    player_names = ['North', 'East', 'South', 'West']
    print(f"  Current Player: {player_names[player_idx]}")
    print()
    
    # 获取叫牌历史
    legal_actions = board.get('legal_actions', [])
    model_act = board.get('model_act', 0)
    optimal_act = board.get('optimal_act', 0)
    action_values = board.get('action_values', [])
    
    if phase == 'bidding':
        # 显示叫牌相关信息
        print("-" * 80)
        print("  BIDDING INFORMATION")
        print("-" * 80)
        print()
        
        # 显示合法动作
        print("  Legal Actions:")
        action_display = []
        for i, act in enumerate(legal_actions):
            name = action_to_bid_name(act)
            val = action_values[i] if i < len(action_values) else 0
            action_display.append(f"{name:>6}({val:>7.2f})")
        
        # 每行显示8个动作
        for i in range(0, len(action_display), 8):
            print("    " + " ".join(action_display[i:i+8]))
        
        print()
        print("-" * 80)
        print()
        
        # 显示模型选择
        model_name = action_to_bid_name(model_act)
        optimal_name = action_to_bid_name(optimal_act)
        
        print(f"  Model Action:    {model_name}")
        print(f"  Optimal Action:  {optimal_name}")
        
        if model_act == optimal_act:
            print("  ✓ Model matches optimal!")
        else:
            diff = model_act - optimal_act
            if diff > 0:
                print(f"  ✗ Model bid higher than optimal by {diff}")
            else:
                print(f"  ✗ Model bid lower than optimal by {-diff}")
        
        print()
        
        # 显示动作值
        if action_values:
            print("-" * 80)
            print("  ACTION VALUES")
            print("-" * 80)
            print()
            
            positive_vals = [(action_to_bid_name(act), val) 
                          for act, val in zip(legal_actions, action_values) 
                          if val > 0]
            
            if positive_vals:
                print("  Positive Value Bids:")
                for name, val in positive_vals:
                    bar = "█" * min(int(val / 2), 40)
                    print(f"    {name:>6}: {bar} {val:>6.2f}")
            print()
    
    elif phase == 'play':
        # 显示出牌阶段信息
        print("-" * 80)
        print("  PLAY INFORMATION")
        print("-" * 80)
        print()
        
        # 显示合法出牌
        print(f"  Legal Actions ({len(legal_actions)} cards):")
        
        # 按花色分组显示
        suits = ['♣', '♦', '♥', '♠']
        suit_names = ['Clubs', 'Diamonds', 'Hearts', 'Spades']
        
        for suit_idx in range(4):
            cards_in_suit = []
            for act in legal_actions:
                if act // 13 == suit_idx:
                    rank_idx = act % 13
                    rank = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'][rank_idx]
                    cards_in_suit.append(f"{rank}{suits[suit_idx]}")
            
            if cards_in_suit:
                print(f"    {suit_names[suit_idx]:>10}: " + " ".join(f"{c:>3}" for c in cards_in_suit))
        
        print()
        print("-" * 80)
        print()
        
        # 显示模型选择
        model_name = card_action_to_name(model_act)
        optimal_name = card_action_to_name(optimal_act)
        
        print(f"  Model Card:    {model_name}")
        print(f"  Optimal Card:  {optimal_name}")
        
        if model_act == optimal_act:
            print("  ✓ Model matches optimal!")
        else:
            print("  ✗ Model differs from optimal")
        
        print()
    
    # 显示动作损失
    action_losses = board.get('action_losses', [])
    if action_losses:
        print("-" * 80)
        print("  LOSS ANALYSIS")
        print("-" * 80)
        print()
        
        # 找到损失最小的动作
        if phase == 'bidding':
            min_loss_idx = action_losses.index(min(action_losses)) if action_losses else 0
            if min_loss_idx < len(legal_actions):
                best_action = action_to_bid_name(legal_actions[min_loss_idx])
                print(f"  Best Bid (lowest loss): {best_action} (loss={action_losses[min_loss_idx]:.4f})")
        elif phase == 'play':
            min_loss_idx = action_losses.index(min(action_losses)) if action_losses else 0
            if min_loss_idx < len(legal_actions):
                best_card = card_action_to_name(legal_actions[min_loss_idx])
                print(f"  Best Card (lowest loss): {best_card} (loss={action_losses[min_loss_idx]:.4f})")
        
        print()
    
    # 页脚
    print("=" * 80)
    print("  Navigation: [Page Up] Previous | [Page Down] Next | [Home] First | [End] Last | [Q] Quit")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Show board data using endplay with keyboard navigation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python show_board_from_data.py
    python show_board_from_data.py -d data.jsonl
    python show_board_from_data.py --data-file training_data.npz.jsonl
        """
    )
    parser.add_argument('-d', '--data-file', type=str, 
                       default='data/stepwise_training_data.jsonl',
                       help='Path to the data file (default: data/stepwise_training_data.jsonl)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("  Bridge Board Data Viewer (using endplay)")
    print("=" * 80)
    print()
    print(f"  Loading data from: {args.data_file}")
    
    # 加载数据
    data = load_data(args.data_file)
    if data is None:
        sys.exit(1)
    
    print(f"  Loaded {len(data)} entries")
    if len(data) == 0:
        print("  No data found!")
        sys.exit(1)
    
    print()
    print("  Press Enter to start navigation...")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
    
    # 显示第一个 board
    current_index = 0
    display_board_endplay(data, current_index, len(data))
    
    # 导航循环
    try:
        while True:
            key = get_key()
            
            if key == 'q' or key == 'Q' or key == 'ESC':
                break
            elif key == 'PAGE_UP':
                current_index = (current_index - 1) % len(data)
                display_board_endplay(data, current_index, len(data))
            elif key == 'PAGE_DOWN':
                current_index = (current_index + 1) % len(data)
                display_board_endplay(data, current_index, len(data))
            elif key == 'HOME':
                current_index = 0
                display_board_endplay(data, current_index, len(data))
            elif key == 'END':
                current_index = len(data) - 1
                display_board_endplay(data, current_index, len(data))
    
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\nError: {e}")
    
    print("\n  Exiting...")


def get_key():
    """获取键盘按键"""
    try:
        import msvcrt
        if os.name == 'nt':
            key = msvcrt.getch()
            if key == b'\xe0':
                key = msvcrt.getch()
                if key == b'H':
                    return 'UP'
                elif key == b'P':
                    return 'PAGE_DOWN'
                elif key == b'I':
                    return 'PAGE_UP'
                elif key == b'G':
                    return 'HOME'
                elif key == b'O':
                    return 'END'
            return key.decode('utf-8', errors='ignore')
    except ImportError:
        pass
    
    try:
        import tty
        import termios
        import select
        
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                if key == '\x1b':
                    if select.select([sys.stdin], [], [], 0.1)[0]:
                        key += sys.stdin.read(2)
                        if key == '\x1b[5~':
                            return 'PAGE_UP'
                        elif key == '\x1b[6~':
                            return 'PAGE_DOWN'
                        elif key == '\x1b[H':
                            return 'HOME'
                        elif key == '\x1b[F':
                            return 'END'
                    return 'ESC'
                return key
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except ImportError:
        pass
    
    return ''


if __name__ == "__main__":
    main()
