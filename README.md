# Reinforcement Learning Assignment 3

This repository contains the code and results for Assignment 3 of the Reinforcement Learning course. The assignment compares three policy-gradient based reinforcement learning algorithms on `CartPole-v1`:

- REINFORCE
- Basic Actor-Critic (AC)
- Advantage Actor-Critic (A2C)

The main experiment runner is `experiment.py`. It allows every experiment to be rerun with a single command per algorithm or with one command for all algorithms.

## Project structure

```text
.
├── experiment.py              # Main experiment runner for REINFORCE, AC, and A2C
├── plots.py                   # Generates the combined learning-curve plot
├── reinforce.py               # Standalone REINFORCE script
├── AC.py                      # Standalone Actor-Critic script
├── A2C.py                     # Standalone A2C script
├── results/                   # Assignment 3 CSV results per algorithm and seed
├── results_a2/                # Assignment 2 DQN results used for comparison
├── learning_curves.png        # Combined comparison plot
├── reinforce_cartpole.png     # Standalone REINFORCE plot
├── actor_critic_cartpole.png  # Standalone AC plot
└── a2c_cartpole.png           # Standalone A2C plot
```

## Requirements

The code is intended to run on a Linux university machine, such as DSLab or a computer lab machine.

Required software:

- Python 3.9 or newer
- `pip`
- The Python packages listed below

Required Python packages:

```text
gymnasium
numpy
pandas
matplotlib
torch
```

## Setup

From the project folder, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required packages:

```bash
pip install gymnasium numpy pandas matplotlib torch
```

If a `requirements.txt` file is added later, the setup can instead be done with:

```bash
pip install -r requirements.txt
```

## Reproducing the experiments

All main experiments should be rerun using `experiment.py`. The default environment is `CartPole-v1`. The reported experiments use 5 seeds and 1,000,000 environment steps per algorithm.

### Run REINFORCE

```bash
python3 experiment.py --algo reinforce --seeds 5 --steps 1000000 --results_dir results
```

This writes the following files:

```text
results/reinforce_seed0.csv
results/reinforce_seed1.csv
results/reinforce_seed2.csv
results/reinforce_seed3.csv
results/reinforce_seed4.csv
```

### Run Basic Actor-Critic

```bash
python3 experiment.py --algo ac --seeds 5 --steps 1000000 --results_dir results
```

This writes the following files:

```text
results/ac_seed0.csv
results/ac_seed1.csv
results/ac_seed2.csv
results/ac_seed3.csv
results/ac_seed4.csv
```

### Run Advantage Actor-Critic

```bash
python3 experiment.py --algo a2c --seeds 5 --steps 1000000 --results_dir results
```

This writes the following files:

```text
results/a2c_seed0.csv
results/a2c_seed1.csv
results/a2c_seed2.csv
results/a2c_seed3.csv
results/a2c_seed4.csv
```

### Run all Assignment 3 experiments at once

```bash
python3 experiment.py --algo reinforce ac a2c --seeds 5 --steps 1000000 --results_dir results
```

This command reruns all three algorithms and saves all per-seed CSV files in the `results/` folder.

## Quick test run

To check that the code works before launching the full experiments, run a short version with fewer steps and seeds:

```bash
python3 experiment.py --algo reinforce ac a2c --seeds 1 --steps 10000 --results_dir test_results
```

This should finish much faster and create a `test_results/` folder with one CSV file per algorithm.

## Recreating the final plot

After the experiment CSV files have been generated, recreate the combined learning-curve plot with:

```bash
python3 plots.py --results_dir results --results_a2_dir results_a2 --out learning_curves.png --smooth 20
```

The script reads the Assignment 3 CSV files from `results/` and the Assignment 2 DQN comparison files from `results_a2/`. It saves the final plot as:

```text
learning_curves.png
```

## Output format

Each experiment produces one CSV file per algorithm and seed. Every CSV file contains two columns:

```text
env_step,episode_return
```

- `env_step`: cumulative number of environment interaction steps
- `episode_return`: total return obtained in that episode

These CSV files are used by `plots.py` to calculate smoothed learning curves and standard deviation bands across seeds.
