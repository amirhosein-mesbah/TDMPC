# TDMPC: Temporal Difference Learning for Model Predictive Control

A CleanRL-style implementation of TDMPC (Temporal Difference Learning for Model Predictive Control), a model-based reinforcement learning algorithm that combines learned world models with online planning for continuous control tasks.

> **Note**: This is an educational/research implementation intended for learning and experimentation. It is not the official TDMPC implementation.

## Overview

TDMPC learns a latent dynamics model of the environment and uses it for planning via the Cross-Entropy Method (CEM). The algorithm trains the following components end-to-end:

- **Encoder (h)**: Maps observations to latent representations
- **Dynamics Model (d)**: Predicts next latent states given current state and action
- **Reward Model (R)**: Predicts single-step rewards
- **Policy Network (π)**: Learns a parametric policy to guide planning
- **Twin Q-Networks (Q1, Q2)**: Estimate state-action values for policy improvement

## Key Features

- Model-based planning with CEM optimization
- Temporal Difference learning for value estimation
- Prioritized Experience Replay with SumTree implementation
- Action repeat wrapper for computational efficiency
- Linear scheduling for exploration noise and planning horizon
- CleanRL-style single-file implementation

## Installation

```bash
pip install gymnasium metaworld torch tensorboard tyro numpy
```

## Usage

### Basic Training

```bash
python tdmpc.py --env-name reach-v3 --total-timesteps 1000000
```

### With Weights & Biases Tracking

```bash
python tdmpc.py --track --wandb-project-name my-project --wandb-entity my-team
```

### With Video Capture

```bash
python tdmpc.py --capture-video
```

## Environment Support

**Note:** This implementation currently supports **Meta-World environments only**. The code is designed for state-based (non-pixel) observations and does not include CNN encoders for image-based tasks. 

Supported Meta-World tasks include: `reach-v3`, `push-v3`, `pick-place-v3`, `door-open-v3`, `drawer-open-v3`, `window-open-v3`, `button-press-v3`, and others from the Meta-World MT1 benchmark.

## TODO

Potential future improvements and extensions:

- [ ] Add support for vectorized environments (num_envs > 1)
- [ ] Add support for DeepMind Control Suite environments
- [ ] Implement CNN encoder for pixel-based observations
- [ ] Add experimental results and training curves
- [ ] Add model checkpoint saving/loading
- [ ] Add evaluation mode without exploration noise

## Code Structure

```
tdmpc.py
├── Args (dataclass)          # Hyperparameter configuration
├── ActionRepeatWrapper       # Environment wrapper for action repeat
├── make_env()               # Environment factory function
├── SumTree                  # Priority queue for prioritized replay
├── ReplayMemory             # Prioritized replay buffer
├── MLP, QNet                # Neural network modules
├── TruncatedNormal          # Action distribution for policy
├── TDMPC (nn.Module)        # Main agent class
│   ├── h()                  # Encoder
│   ├── next()               # Dynamics + Reward models
│   ├── pi()                 # Policy network
│   └── Q()                  # Twin Q-value functions
└── Training Loop            # Main training loop with planning and updates
```

## Acknowledgements

This implementation is based on:

- **Original TDMPC Repository**: [nicklashansen/tdmpc](https://github.com/nicklashansen/tdmpc) - The official implementation by Nicklas Hansen
- **CleanRL**: [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) - Clean implementations of RL algorithms by Costa Huang
- **Meta-World**: [Farama-Foundation/Metaworld](https://github.com/Farama-Foundation/Metaworld) - Benchmark for multi-task and meta reinforcement learning

The code structure follows CleanRL's philosophy of single-file implementations with minimal abstractions, making it easy to understand and modify.

## License

This implementation follows the same license as the original TDMPC repository (MIT License).
