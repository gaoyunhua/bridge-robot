#!/usr/bin/env python3
"""
Batch Training Script
1. Generate training data using generate_training_data.py
2. Train model using train.py
"""

import os
import sys
import argparse
import subprocess

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))


def run_command(cmd):
    print(f"\nRunning: {' '.join(cmd)}\n")
    try:
        result = subprocess.run(cmd, check=True)
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Batch Bridge Training")
    parser.add_argument("--num-boards", type=int, default=100, help="Number of boards to generate")
    parser.add_argument("--data-file", type=str, default="data/stepwise_training_data.jsonl")
    parser.add_argument("--output-model", type=str, default="checkpoints/stepwise_bridge_model.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("="*80)
    print("Batch Bridge Training")
    print("="*80)

    # Step 1: Generate training data
    print("\nStep 1: Generating training data...")
    generate_cmd = [
        "python",
        "src/generate_training_data.py",
        "--num-boards", str(args.num_boards),
        "--output", args.data_file,
        "--seed", str(args.seed)
    ]
    if not run_command(generate_cmd):
        print("Failed to generate training data")
        return 1

    # Step 2: Train model
    print("\nStep 2: Training model...")
    train_cmd = [
        "python",
        "src/train.py",
        "--data-file", args.data_file,
        "--output-model", args.output_model,
        "--batch-size", str(args.batch_size),
        "--lr", str(args.lr),
        "--epochs", str(args.epochs),
        "--seed", str(args.seed)
    ]
    if not run_command(train_cmd):
        print("Failed to train model")
        return 1

    print("\n" + "="*80)
    print("Batch training complete!")
    print(f"Model saved to: {args.output_model}")
    print("="*80)
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
