import argparse
import time
import subprocess
import sys
import json
import os
from generate_training_data import StepwiseBridgeDataGenerator


def load_existing_data(output_file):
    """加载现有数据"""
    existing_data = []
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    existing_data.append(json.loads(line.strip()))
                except:
                    pass
        print(f"加载了 {len(existing_data)} 个现有样本")
    return existing_data


def append_data(data, output_file):
    """追加数据到文件"""
    with open(output_file, 'a', encoding='utf-8') as f:
        for dp in data:
            f.write(json.dumps(dp, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description="Continuous Bridge Training v2")
    parser.add_argument("--num-boards", type=int, default=10,
                       help="Number of boards per iteration (default: 10)")
    parser.add_argument("--output-model", type=str, default="checkpoints/policy_model_v2.pt")
    parser.add_argument("--resume-from", type=str, default=None,
                       help="Path to pre-trained model to resume training from")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--early-stop", type=int, default=5,
                       help="Early stopping patience (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset", action="store_true",
                       help="重置数据文件，从头开始")
    args = parser.parse_args()

    iteration = 0
    total_boards = 0
    output_file = "data/stepwise_training_data.jsonl"
    
    if args.reset and os.path.exists(output_file):
        os.remove(output_file)
        print("已重置数据文件")

    print("=" * 80)
    print("持续训练模式 v2（累积数据 + 验证集）")
    print(f"每次迭代: 生成 {args.num_boards} 个牌局, 训练 {args.epochs} 个 epochs")
    print(f"早停耐心: {args.early_stop}")
    print(f"数据文件: {output_file}")
    print("按 Ctrl+C 停止训练")
    print("=" * 80)

    try:
        while True:
            iteration += 1
            print(f"\n{'='*80}")
            print(f"迭代 {iteration} 开始")
            print(f"{'='*80}")

            # 生成数据
            print(f"\n[1/2] 生成 {args.num_boards} 个牌局数据...")
            generator = StepwiseBridgeDataGenerator(seed=args.seed + iteration)
            data = generator.generate(num_boards=args.num_boards)
            total_boards += args.num_boards
            
            # 追加数据到文件
            append_data(data, output_file)
            
            # 统计总样本数
            total_samples = sum(1 for _ in open(output_file, 'r', encoding='utf-8'))
            
            print(f"数据生成完成！本次 {len(data)} 个训练步骤")
            print(f"累计牌局数: {total_boards}")
            print(f"累计样本总数: {total_samples}")

            # 训练 (使用基于 action_values 的策略训练 v2)
            print(f"\n[2/2] 开始策略训练 v2 (最多 {args.epochs} 个 epochs)...")

            # 构建训练命令
            cmd = [
                sys.executable, "src/train_policy_v2.py",
                "--data-file", output_file,
                "--output-model", args.output_model,
                "--batch-size", str(args.batch_size),
                "--lr", str(args.lr),
                "--epochs", str(args.epochs),
                "--early-stop", str(args.early_stop),
                "--seed", str(args.seed + iteration)
            ]
            
            # 第一轮使用 --resume-from，后续使用上一轮保存的模型
            if iteration == 1 and args.resume_from:
                cmd.extend(["--resume-from", args.resume_from])
            elif iteration > 1 and os.path.exists(args.output_model):
                cmd.extend(["--resume-from", args.output_model])

            result = subprocess.run(cmd)

            if result.returncode != 0:
                print(f"训练失败！返回码: {result.returncode}")
                break

            print(f"\n迭代 {iteration} 完成！")
            print(f"模型已保存到: {args.output_model}")

            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\n训练已停止")
        print(f"总共完成 {iteration} 次迭代")
        print(f"总共训练了 {total_boards} 个牌局")
        print(f"最终模型保存在: {args.output_model}")
        print(f"数据保存在: {output_file}")


if __name__ == "__main__":
    main()