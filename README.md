# Bridge Robot AI

A deep learning-based Bridge card game AI using PyTorch and endplay library for bridge game logic.

## Project Overview

Bridge Robot AI is a reinforcement learning and supervised learning project that trains a transformer-based neural network to play the game of Bridge (contract bridge). The project implements:

- A 757-dimensional observation encoding for bridge game states
- A 4-layer Transformer model (BridgeTransformerV2) for policy and value estimation
- PPO (Proximal Policy Optimization) training for RL-based learning
- DDS (Double-Dummy Solver) based teacher for supervised learning labels
- Real bridge scoring using endplay library

## Project Structure

```
bridge-robot/
├── src/
│   ├── __init__.py                 # Package initialization
│   ├── config.py                   # Configuration (EnvConfig, TrainConfig, dimension constants)
│   ├── model_transformer.py        # BridgeTransformerV2 neural network model
│   ├── env_core.py                 # Core bridge environment with bidding and play logic
│   ├── env_full.py                 # Legacy alternative environment (not used)
│   ├── obs.py                      # 757-dimensional observation encoding
│   ├── rewards.py                  # DDS/Par scoring for RL rewards
│   ├── train.py                    # Legacy training script (not used)
│   ├── predictor.py                # BridgeAIPredictor for sequencing predictions
│   ├── evaluator.py                # ModelEvaluator for accuracy assessment
│   ├── dds_teacher.py             # DDS-based optimal action teacher for supervised learning
│   ├── generate_training_data.py   # Training data generator using endplay
│   └── vis.py                      # Visual information module
├── scripts/
│   ├── run_training.py             # Main training script (CPU optimized)
│   ├── rl_train.py                 # PPO reinforcement learning training
│   ├── stepwise_dds_training.py    # Stepwise DDS training
│   ├── train_supervised.py         # Supervised learning training
│   └── play_with_model.py          # Script to play with trained model
├── tests/
│   ├── test_visibility.py
│   ├── test_visibility2.py
│   ├── test_env_visibility.py
│   └── test_env_visibility2.py
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Requirements

```
Python>=3.8
torch>=2.0.0
numpy>=1.20.0
pandas>=1.3.0
endplay>=0.5.0
keyboard>=0.13.0
flask>=2.0.0
```

## Installation

```bash
pip install -r requirements.txt
```

## Configuration

The project uses a 757-dimensional observation vector with the following structure:

| Offset Range | Description | Dimension |
|-------------|-------------|-----------|
| 0-11 | Match metadata | 12 |
| 12-219 | Contract info (vul, declarer, level, denom, double, result) | 208 |
| 220-447 | Hands (4 players × 52 cards + padding) | 228 |
| 448-476 | First bidder | 29 |
| 477-675 | Auction history (66 bid entries × 3) | 198 |
| 676-703 | Lead card (first trick) | 28 |
| 704-756 | Subsequent play | 53 |

### Action Spaces

- **Bidding Actions**: 38 (1C-7NT, Pass, Double, Redouble)
- **Playing Actions**: 52 (4 suits × 13 ranks)

## Core Components

### Model (model_transformer.py)

`BridgeTransformerV2` is a 4-layer Transformer encoder network:

```python
BridgeTransformerV2(
    d_model=256,      # Model dimension
    nhead=4,          # Number of attention heads
    num_layers=4,    # Number of encoder layers
    input_dim=757,    # Observation dimension
    num_bid_actions=38,
    num_play_actions=52
)
```

Forward pass returns `(bid_logits, play_logits)` for supervised training, or `(bid_logits, play_logits, hidden)` for RL training with value estimation.

### Environment (env_core.py)

`BridgeEnv` implements the full bridge game state machine:

1. **Bidding Phase**: Sequential N→E→S→W bidding with legal bid validation
2. **Play Phase**: 52-card trick-taking play with suit-following rules
3. **Scoring**: Uses `RewardsModule` for DDS/Par-based scoring

Key classes:
- `BiddingBox`: Tracks auction state and validates bids
- `PlayState`: Manages card play and trick tracking

### Rewards (rewards.py)

`RewardsModule` computes real bridge scores using:

- `contract_score()`: ACBL standard scoring
- `compute_dd_table_array()`: Double-dummy analysis via endplay DDS
- `imp()`: IMP conversion for tournament scoring

### DDS Teacher (dds_teacher.py)

`BidTeacher` and `PlayTeacher` use double-dummy analysis to generate optimal action labels for supervised learning:

- Evaluates all legal actions via DDS
- Selects the action maximizing the bidding side's score
- Includes illegal-action penalty loss for policy training

## Training

### Supervised Training (Main)

```bash
python scripts/run_training.py
```

Trains on pre-generated data with:
- 90/10 train/validation split
- Cross-entropy loss for bidding and playing actions
- OneCycleLR scheduler with AdamW optimizer
- Checkpoints saved to `checkpoints/` directory

### PPO Reinforcement Learning

```bash
python scripts/rl_train.py
```

Trains using PPO with:
- GAE (Generalized Advantage Estimation)
- Actor-Critic architecture
- Legal action masking
- Separate policy and value heads

### DDS Supervised Learning

```bash
python scripts/stepwise_dds_training.py
```

Uses double-dummy optimal actions as training labels for improved bid/play decisions.

## Data Format

Training data CSV format (759 columns):
```
obs_757d (757 cols) | bid_label (1 col) | play_label (1 col)
```

Each row represents one training sample with:
- 757-dimensional observation vector
- Bidding action label (0-37)
- Playing action label (0-51)

## Usage Example

```python
from src.config import EnvConfig
from src.model_transformer import BridgeTransformerV2
from src.predictor import BridgeAIPredictor
from src.env_core import BridgeEnv

# Initialize
cfg = EnvConfig()
model = BridgeTransformerV2()
predictor = BridgeAIPredictor(model, cfg)
env = BridgeEnv()

# Predict a full board
steps = predictor.predict(env, max_steps=80)
for step in steps:
    print(f"{step.phase}: {step.player_idx} -> {step.action_name}")
```

## Testing

Run visibility and environment tests:

```bash
python -m pytest tests/
```

## Checkpoints

Model checkpoints are saved to `checkpoints/`:
- `best_model.pt`: Best model by combined accuracy
- `checkpoint_iter_*.pt`: Periodic checkpoints
- `final_model.pt`: Final trained model
- `train.log`: Training logs

## License

This project is for research and educational purposes.
