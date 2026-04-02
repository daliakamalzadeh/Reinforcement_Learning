# Reinforcement Learning Assignment - Assignment 2

## Overview
This repository contains the source code for our CartPole reinforcement learning assignment. The project includes:

- A naive Deep Q-Learning implementation
- A DQN implementation with optional Target Networks and Experience Replay
- A hyperparameter ablation study
- A comparison of DQN variants

The code is structured so that each sub-task / experiment can be rerun with a single command on a Linux university machine.

## Files
- `Q_learning.py`  
  Runs the naive DQN / Q-learning baseline experiment on CartPole.

- `Q_learning_target_replay.py`  
  Runs the extended DQN implementation with optional:
  - Target Network (TN)
  - Experience Replay (ER)

- `ablation_study.py`  
  Runs the hyperparameter ablation study.

- `compare_dqn_variants.py`  
  Compares four DQN variants.

- `requirements.txt`  
  Python dependencies needed to run the experiments.

## Environment Setup

Use Python 3.10 or newer.

Create and activate a virtual environment:

```bash
python3 -m venv rl_env
source rl_env/bin/activate
```

### Install dependencies:

```bash
pip install -r requirements.txt
```

### Running the experiments:

Each experiment can be run with a single command.

1. Naive DQN / Q-learning baseline

```bash
python Q_learning.py
```

2. DQN with optional Target Network / Experience Replay

```bash
python Q_learning_target_replay.py
```

3. Hyperparameter ablation study

```bash

python ablation_study.py
```

4. Comparison of DQN variants

```bash

python compare_dqn_variants.py

```

### Output

Running the scripts generates plots as image files in the working directory.

Expected outputs include:

- q_learning_cartpole_vs_baseline.png
- q_learning_cartpole_vs_baseline_updated.png
- ablation_learning_rate.png
- ablation_network_architecture.png
- ablation_epsilon_decay.png
- dqn_variant_comparison.png


### Important Note

The scripts `Q_learning.py` and `Q_learning_target_replay.py` attempt to load a baseline CSV file:

- BaselineDataCartPole.csv

Make sure this file is present in the same directory when running the scripts. If it is missing, the training will still run, but the baseline comparison in the plots may not work correctly.
