import random
from dataclasses import dataclass

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


# config

@dataclass
class Config:
    env_name: str = "CartPole-v1"
    seed: int = 42

    total_env_steps: int = 50000
    gamma: float = 0.99
    learning_rate: float = 1e-3

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 20000

    hidden_sizes: tuple = (128, 128)

    log_every_episodes: int = 10
    plot_path: str = "q_learning_cartpole_vs_baseline.png"

    baseline_csv_path: str = "BaselineDataCartPole.csv"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def epsilon_by_step(step: int, cfg: Config) -> float:
    frac = min(step / cfg.epsilon_decay_steps, 1.0)
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def moving_average(x, window=20):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


# Q-Network

class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(128, 128)):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# agent

class QLearningAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: Config, device="cpu"):
        self.action_dim = action_dim
        self.gamma = cfg.gamma
        self.device = torch.device(device)

        self.q_net = QNetwork(obs_dim, action_dim, cfg.hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=cfg.learning_rate)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)

        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def update(self, state, action, reward, next_state, done):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state_t = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.tensor([[action]], dtype=torch.long, device=self.device)
        reward_t = torch.tensor([[reward]], dtype=torch.float32, device=self.device)
        done_t = torch.tensor([[float(done)]], dtype=torch.float32, device=self.device)

        # current Q(s, a)
        q_values = self.q_net(state_t)
        q_sa = q_values.gather(1, action_t)

        # naive Q-learning target:
        # y = r + gamma * max_a' Q(s', a') for non-terminal states
        with torch.no_grad():
            next_q_values = self.q_net(next_state_t)
            max_next_q = next_q_values.max(dim=1, keepdim=True).values
            target = reward_t + self.gamma * (1.0 - done_t) * max_next_q

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


# training

def train(cfg: Config):
    set_seed(cfg.seed)

    env = gym.make(cfg.env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = QLearningAgent(obs_dim, action_dim, cfg)

    state, _ = env.reset(seed=cfg.seed)

    env_steps = 0
    episode_idx = 0
    episode_return = 0.0

    returns = []
    return_steps = []
    losses = []

    while env_steps < cfg.total_env_steps:
        epsilon = epsilon_by_step(env_steps, cfg)
        action = agent.select_action(state, epsilon)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        loss = agent.update(state, action, reward, next_state, done)
        losses.append(loss)

        episode_return += reward
        env_steps += 1
        state = next_state

        if done:
            episode_idx += 1
            returns.append(episode_return)
            return_steps.append(env_steps)

            if episode_idx % cfg.log_every_episodes == 0:
                avg_last_10 = np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns)
                print(
                    f"Episode {episode_idx:4d} | "
                    f"Steps {env_steps:6d} | "
                    f"Return {episode_return:6.1f} | "
                    f"Avg(Last10) {avg_last_10:6.1f} | "
                    f"Epsilon {epsilon:.3f}"
                )

            state, _ = env.reset()
            episode_return = 0.0

    env.close()

    return np.array(return_steps), np.array(returns), np.array(losses)


# baseline loading

def load_baseline(csv_path: str, max_env_step=None):
    df = pd.read_csv(csv_path)

    # sort first, otherwise matplotlib connects points in the wrong order
    df = df.sort_values("env_step").reset_index(drop=True)

    if max_env_step is not None:
        df = df[df["env_step"] <= max_env_step].copy()

    env_steps = df["env_step"].to_numpy()
    returns = df["Episode_Return"].to_numpy()
    smooth = df["Episode_Return_smooth"].to_numpy()

    return env_steps, returns, smooth

# plotting

def plot_learning_curve(return_steps, returns, cfg: Config):
    plt.figure(figsize=(10, 6))

    max_q_step = int(return_steps[-1])

    # agent
    plt.plot(return_steps, returns, alpha=0.25, label="Naive Q-learning (raw)")
    if len(returns) >= 20:
        smoothed = moving_average(returns, window=20)
        smoothed_steps = return_steps[19:]
        plt.plot(smoothed_steps, smoothed, linewidth=2, label="Naive Q-learning (smoothed)")

    # baseline
    try:
        base_steps, base_returns, base_smooth = load_baseline(
            cfg.baseline_csv_path,
            max_env_step=max_q_step
        )

        plt.plot(base_steps, base_returns, alpha=0.15, label="Baseline (raw)")
        plt.plot(base_steps, base_smooth, linewidth=2, label="Baseline (smoothed)")

    except Exception as e:
        print(f"Could not load baseline CSV: {e}")

    plt.axhline(500, linestyle=":", linewidth=1.5, label="Optimal performance = 500")
    plt.xlim(0, max_q_step)
    plt.xlabel("Environment steps")
    plt.ylabel("Return")
    plt.title("CartPole: Naive Q-Learning vs Baseline")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.plot_path, dpi=200)
    plt.show()

# main

if __name__ == "__main__":
    cfg = Config()

    return_steps, returns, losses = train(cfg)
    plot_learning_curve(return_steps, returns, cfg)

    print("\nTraining finished.")
    print(f"Best episode return: {returns.max():.1f}")
    print(f"Final 20-episode average: {returns[-20:].mean() if len(returns) >= 20 else returns.mean():.2f}")