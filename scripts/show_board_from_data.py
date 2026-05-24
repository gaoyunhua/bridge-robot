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

from endplay import Deal, Card
from endplay.types import Player, Bid, Contract, Denom, Rank
from config import EnvConfig, OFS_HANDS, NUM_PLAYERS

# 常量定义
BID_NAMES = ['1C', '1D', '1H', '1S', '1NT', 
            '2C', '2D', '2H', '2S', '2NT',
            '3C', '3D', '3H', '3S', '3NT',
            '4C', '4D', '4H', '4S', '4NT',
            '5C', '5D', '5H', '5S', '5NT',
            '6C', '6D', '6H', '6S', '6NT',
            '7C', '7D', '7H', '7S', '7NT']

SUITS = ['♣', '♦', '♥', '♠']
SUIT_NAMES = ['C', 'D', 'H', 'S']
RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
PLAYER_NAMES = ['North', 'East', 'South', 'West']

def action_to_bid_name(action):
    """将动作转换为叫牌名称"""
    if 0 <= action < 35:
        return BID_NAMES[action]
    elif action == 35:
        return "Pass"
    elif action == 36:
        return "X"
    elif action == 37:
        return "XX"
    return f"Unknown({action})"

def card_action_to_name(action):
    """将出牌动作转换为牌张名称"""
    if 0 <= action < 52:
        suit = action // 13
        rank = action % 13
        return f"{RANKS[rank]}{SUITS[suit]}"
    return f"Action{action}"

def load_data(data_file):
    """加载数据文件"""
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found!")
        return None
    
    data = []
    skipped_count = 0
    parse_error_count = 0
    
    with open(data_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            try:
                item = json.loads(line)
                if isinstance(item, dict) and 'phase' in item:
                    data.append(item)
                else:
                    skipped_count += 1
                    if skipped_count <= 5:  # 只打印前5个跳过警告
                        print(f"Skipping invalid line {line_num}: {type(item).__name__}")
            except json.JSONDecodeError as e:
                parse_error_count += 1
                if parse_error_count <= 5:  # 只打印前5个解析错误
                    print(f"JSON parse error line {line_num}: {e}")
    
    if skipped_count > 0 or parse_error_count > 0:
        print(f"\nWarning: Skipped {skipped_count} invalid lines, {parse_error_count} parse errors")
    
    return data

def display_board_with_endplay(data, index, total):
    """使用 endplay 显示单个 board"""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    # 安全检查
    if index < 0 or index >= len(data):
        print(f"Error: index {index} out of range (0-{len(data)-1})")
        return
    
    board = data[index]
    if not isinstance(board, dict):
        print(f"Error: invalid board at index {index}, type={type(board).__name__}")
        return
    
    board_idx = board.get('board_idx', index + 1)
    player_idx = board.get('player_idx', 0)
    phase = board.get('phase', 'unknown')
    
    # 使用 endplay 显示标题
    print("=" * 80)
    print(f"  Bridge Board {board_idx} [{index + 1}/{total}] - {phase.upper()} Phase")
    print("=" * 80)
    print()
    
    # 显示当前玩家信息
    print(f"  Current Player: {PLAYER_NAMES[player_idx]}")
    print()
    
    # 获取动作信息
    legal_actions = board.get('legal_actions', [])
    model_act = board.get('model_act', 0)
    optimal_act = board.get('optimal_act', 0)
    action_values = board.get('action_values', [])
    action_losses = board.get('action_losses', [])
    
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
        
        # 显示模型选择和最优动作
        model_name = action_to_bid_name(model_act)
        optimal_name = action_to_bid_name(optimal_act)
        
        print(f"  Model Action:    {model_name}")
        print(f"  Optimal Action: {optimal_name}")
        
        if model_act == optimal_act:
            print("  ✓ Model matches optimal!")
        else:
            diff = model_act - optimal_act
            if diff > 0:
                print(f"  ✗ Model bid higher than optimal by {diff}")
            else:
                print(f"  ✗ Model bid lower than optimal by {-diff}")
        
        print()
        
        # 显示动作值（如果可用）
        if action_values:
            print("-" * 80)
            print("  ACTION VALUES (DDS Evaluated)")
            print("-" * 80)
            print()
            
            positive_vals = [(action_to_bid_name(act), val) 
                         for act, val in zip(legal_actions, action_values) 
                         if val > 0]
            
            if positive_vals:
                print("  Positive Value Bids:")
                for name, val in positive_vals:
                    bar_len = min(int(val / 2), 40)
                    bar = "█" * bar_len
                    print(f"    {name:>6}: {bar} {val:>6.2f}")
            print()
    
    elif phase == 'play':
        # 显示出牌阶段信息
        print("-" * 80)
        print("  PLAY INFORMATION")
        print("-" * 80)
        print()
        
        # 显示合法出牌（使用endplay牌张格式）
        print(f"  Legal Actions ({len(legal_actions)} cards):")
        print()
        
        # 按花色分组显示
        for suit_idx, suit_name in enumerate(SUIT_NAMES):
            cards_in_suit = []
            for act in legal_actions:
                if act // 13 == suit_idx:
                    rank_idx = act % 13
                    cards_in_suit.append(f"{RANKS[rank_idx]:>2}")
            
            if cards_in_suit:
                print(f"    {SUITS[suit_idx]} ({suit_name:>7}): " + " ".join(cards_in_suit))
        
        print()
        print("-" * 80)
        print()
        
        # 显示模型选择
        model_name = card_action_to_name(model_act)
        optimal_name = card_action_to_name(optimal_act)
        
        print(f"  Model Card:    {model_name}")
        print(f"  Optimal Card: {optimal_name}")
        
        if model_act == optimal_act:
            print("  ✓ Model matches optimal!")
        else:
            print("  ✗ Model differs from optimal")
        
        print()
    
    # 显示损失分析（如果可用）
    if action_losses:
        print("-" * 80)
        print("  LOSS ANALYSIS")
        print("-" * 80)
        print()
        
        try:
            min_loss_idx = action_losses.index(min(action_losses)) if action_losses else 0
            if min_loss_idx < len(legal_actions):
                if phase == 'bidding':
                    best_action = action_to_bid_name(legal_actions[min_loss_idx])
                    print(f"  Best Bid (lowest loss): {best_action} (loss={action_losses[min_loss_idx]:.4f})")
                else:
                    best_card = card_action_to_name(legal_actions[min_loss_idx])
                    print(f"  Best Card (lowest loss): {best_card} (loss={action_losses[min_loss_idx]:.4f})")
        except:
            pass
        
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
    
    # 调试信息
    print(f"  Data type: {type(data)}")
    if data:
        print(f"  First item type: {type(data[0])}")
    
    print()
    print("  Press Enter to start navigation...")
    try:
        input()
    except (EOFError, KeyboardInterrupt):
        pass
    
    # 显示第一个 board
    current_index = 0
    try:
        display_board_with_endplay(data, current_index, len(data))
    except Exception as e:
        print(f"\nError displaying initial board: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 导航循环
    try:
        while True:
            key = get_key()
            
            if key == 'q' or key == 'Q' or key == 'ESC':
                break
            elif key == 'PAGE_UP':
                current_index = (current_index - 1) % len(data)
                display_board_with_endplay(data, current_index, len(data))
            elif key == 'PAGE_DOWN':
                current_index = (current_index + 1) % len(data)
                display_board_with_endplay(data, current_index, len(data))
            elif key == 'HOME':
                current_index = 0
                display_board_with_endplay(data, current_index, len(data))
            elif key == 'END':
                current_index = len(data) - 1
                display_board_with_endplay(data, current_index, len(data))
    
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\nError in navigation: {e}")
        import traceback
        traceback.print_exc()
    
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
