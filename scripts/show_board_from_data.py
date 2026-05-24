#!/usr/bin/env python3
"""
Show a complete board from the generated stepwise_training_data.jsonl

Usage:
    python show_board_from_data.py                    # Use default file
    python show_board_from_data.py -d data.jsonl     # Use custom file
    python show_board_from_data.py --data-file data.jsonl
    
Navigation:
    Page Up / Page Down: Navigate between boards
    Up / Down: Navigate between steps
    Q: Quit
"""
import sys
import os
import json
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'src'))

from config import EnvConfig
from env_core import action_to_bid


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
                data.append(json.loads(line))
    
    return data


def display_board(data, index, total):
    """显示单个 board"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    board = data[index]
    board_idx = board.get('board_idx', index + 1)
    player_idx = board.get('player_idx', 0)
    
    print("=" * 80)
    print(f"BOARD {board_idx} [{index + 1}/{total}]")
    print("=" * 80)
    print()
    
    # 显示阶段信息
    phase = board.get('phase', 'unknown')
    print(f"  Phase: {phase.upper()}")
    print(f"  Player: {['North', 'East', 'South', 'West'][player_idx]}")
    
    # 显示叫牌动作
    legal_actions = board.get('legal_actions', [])
    model_act = board.get('model_act', 0)
    optimal_act = board.get('optimal_act', 0)
    action_values = board.get('action_values', [])
    
    print()
    print("-" * 80)
    print("ACTIONS")
    print("-" * 80)
    print()
    
    # 叫牌阶段
    if phase == 'bidding':
        print("Legal Actions:")
        action_names = []
        for i, act in enumerate(legal_actions):
            if act < 35:
                level = act // 5 + 1
                denom = ['C', 'D', 'H', 'S', 'NT'][act % 5]
                name = f"{level}{denom}"
            elif act == 35:
                name = "Pass"
            elif act == 36:
                name = "X (Double)"
            else:
                name = "XX (Redouble)"
            
            # 显示动作值
            if i < len(action_values):
                val = action_values[i]
                action_names.append(f"{name:8} (val={val:.2f})")
            else:
                action_names.append(name)
        
        # 每行显示4个动作
        for i in range(0, len(action_names), 4):
            print("  " + " | ".join(action_names[i:i+4]))
        
        print()
        print("-" * 80)
        print()
        
        # 显示模型选择和最优动作
        if model_act < 35:
            model_name = f"{model_act // 5 + 1}{['C', 'D', 'H', 'S', 'NT'][model_act % 5]}"
        elif model_act == 35:
            model_name = "Pass"
        elif model_act == 36:
            model_name = "X (Double)"
        else:
            model_name = "XX (Redouble)"
        
        if optimal_act < 35:
            optimal_name = f"{optimal_act // 5 + 1}{['C', 'D', 'H', 'S', 'NT'][optimal_act % 5]}"
        elif optimal_act == 35:
            optimal_name = "Pass"
        elif optimal_act == 36:
            optimal_name = "X (Double)"
        else:
            optimal_name = "XX (Redouble)"
        
        print(f"Model Action: {model_name}")
        print(f"Optimal Action: {optimal_name}")
        
        if model_act == optimal_act:
            print("✓ Model matches optimal!")
        else:
            print("✗ Model differs from optimal")
    
    # 出牌阶段
    elif phase == 'play':
        suits = ['C', 'D', 'H', 'S']
        ranks = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
        
        print(f"Legal Actions ({len(legal_actions)} cards):")
        legal_cards = []
        for act in legal_actions:
            s_idx = act // 13
            r_idx = act % 13
            card_str = f"{ranks[r_idx]}{suits[s_idx]}"
            legal_cards.append(card_str)
        
        # 每行显示13个牌
        for i in range(0, len(legal_cards), 13):
            print("  " + " ".join(legal_cards[i:i+13]))
        
        print()
        print("-" * 80)
        print()
        
        if model_act < 52:
            s_idx = model_act // 13
            r_idx = model_act % 13
            model_name = f"{ranks[r_idx]}{suits[s_idx]}"
        else:
            model_name = f"Action {model_act}"
        
        if optimal_act < 52:
            s_idx = optimal_act // 13
            r_idx = optimal_act % 13
            optimal_name = f"{ranks[r_idx]}{suits[s_idx]}"
        else:
            optimal_name = f"Action {optimal_act}"
        
        print(f"Model Action: {model_name}")
        print(f"Optimal Action: {optimal_name}")
        
        if model_act == optimal_act:
            print("✓ Model matches optimal!")
        else:
            print("✗ Model differs from optimal")
    
    print()
    print("=" * 80)
    print("Navigation: [Page Up] Previous | [Page Down] Next | [Q] Quit")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Show board data with keyboard navigation',
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
    print("Show Complete Board from Data")
    print("=" * 80)
    print()
    print(f"Loading data from: {args.data_file}")
    
    # 加载数据
    data = load_data(args.data_file)
    if data is None:
        sys.exit(1)
    
    print(f"Loaded {len(data)} entries from {args.data_file}")
    if len(data) == 0:
        print("No data found!")
        sys.exit(1)
    
    print()
    print("Press any key to start navigation...")
    try:
        input()  # Wait for user to press Enter
    except (EOFError, KeyboardInterrupt):
        pass
    
    # 显示第一个 board
    current_index = 0
    display_board(data, current_index, len(data))
    
    # 导航循环
    try:
        while True:
            key = get_key()
            
            if key == 'q' or key == 'Q' or key == 'ESC':
                break
            elif key == 'PAGE_UP':
                current_index = (current_index - 1) % len(data)
                display_board(data, current_index, len(data))
            elif key == 'PAGE_DOWN':
                current_index = (current_index + 1) % len(data)
                display_board(data, current_index, len(data))
            elif key == 'HOME':
                current_index = 0
                display_board(data, current_index, len(data))
            elif key == 'END':
                current_index = len(data) - 1
                display_board(data, current_index, len(data))
    
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\nError: {e}")
    
    print("\nExiting...")


def get_key():
    """获取键盘按键"""
    try:
        # Windows
        import msvcrt
        if os.name == 'nt':
            key = msvcrt.getch()
            if key == b'\xe0':  # Special key prefix
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
        # Unix/Linux/Mac
        import tty
        import termios
        import select
        
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            # 设置超时为 0.1 秒
            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.read(1)
                # 检查是否是特殊键
                if key == '\x1b':  # ESC
                    # 可能是 ANSI escape sequence
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
