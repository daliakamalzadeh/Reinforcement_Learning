# Reinforcement_Learning

This project implements and compares several reinforcement learning algorithms on the **Stochastic Windy Gridworld** environment.  
It was developed as part of the Reinforcement Learning course.

The goal of the project is to study how different exploration strategies, backup methods, and backup depths affect learning performance.

---

# Authors

Agata Cieliczko, Ramone Bendt, Dalia Kamalzadeh  

---

# Environment

The environment is a stochastic version of the indy Gridworld described in Sutton & Barto.

Main characteristics:

- Grid size: 10 × 7
- Start state: (0,3)
- Goal state: (7,3)
- Reward structure:
  - −1 per step
  - +100 when reaching the goal
- Wind pushes the agent upward depending on the column with 90% probability**

The agent interacts with the environment using:

```python
s_next, r, done = env.step(action)
```

---

# Implemented Algorithms

## Dynamic Programming

Uses Q-value iteration with access to the full transition model.  
This provides the optimal baseline policy used for comparison.

File:
```
DynamicProgramming.py
```

---

## Q-learning (Off-policy)

Learns the optimal policy independently of the exploration strategy.

Update rule:

```
Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
```

File:
```
Q_learning.py
```

---

## SARSA (On-policy)

Updates the Q-values using the action actually selected by the behavior policy.

```
Q(s,a) ← Q(s,a) + α [ r + γ Q(s',a') − Q(s,a) ]
```

File:
```
SARSA.py
```

---

## n-step Q-learning

Extends temporal-difference learning using multi-step returns.

File:
```
Nstep.py
```

---

## Monte Carlo

Updates value estimates using complete episode returns.

File:
```
MonteCarlo.py
```

---

# Exploration Strategies

Implemented in:

```
Agent.py
```

Two exploration policies are compared:

### ε-greedy
- With probability ε, choose a random action
- Otherwise choose the greedy action

### Softmax (Boltzmann)
Actions are sampled according to a probability distribution based on the Q-values.

---

# Experiments

All experiments are run from:

```
Experiment.py
```

Each configuration:

- 20 repetitions
- 50,001 timesteps
- evaluation every 1000 steps
- results averaged across runs
- learning curves optionally smoothed

---

# Results

### Exploration strategies

Output file:

```
exploration.png
```

Comparison between:

- ε-greedy (ε = 0.03, 0.1, 0.3)
- softmax (τ = 0.01, 0.1, 1.0)

---

### On-policy vs Off-policy

Output file:

```
on_off_policy.png
```

Comparison between:

- Q-learning
- SARSA

Learning rates tested:

- α = 0.03
- α = 0.1
- α = 0.3

---

### Backup depth

Output file:

```
depth.png
```

Comparison between:

- 1-step Q-learning
- 3-step Q-learning
- 10-step Q-learning
- Monte Carlo

---

# Installation

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running Individual Algorithms
Run the following commands:

Dynamic Programming:

```
python DynamicProgramming.py
```

Q-learning:

```
python Q_learning.py
```

SARSA:

```
python SARASA.py
```
Monte Carlo:

```
python MonteCarlo.py
```
Nstep:

```
python Nstep.py
```
---

# Running the Experiments

Run the following command:

```
python Experiment.py
```

This will generate all learning curve plots.

---

# Project Structure

```
.
├── Agent.py
├── Environment.py
├── DynamicProgramming.py
├── Q_learning.py
├── SARSA.py
├── Nstep.py
├── MonteCarlo.py
├── Helper.py
├── Experiment.py
└── README.md
```
