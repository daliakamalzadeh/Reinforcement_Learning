import random
from dataclasses import dataclass

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# config

@dataclass
class Config:
    env_name: str = "CartPole-v1"
    seed: int = 42

    total_env_steps: int = 1_000_000
    gamma: float = 0.99
    learning_rate: float = 1e-3

    hidden_sizes: tuple = (128, 128)

    normalize_returns: bool = True

    log_every_episodes: int = 10
    plot_path: str = "reinforce_cartpole.png"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def moving_average(x, window=20):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


# Policy Network

class PolicyNetwork(nn.Module):
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
        return self.net(x)  # logits


# agent

class REINFORCEAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: Config, device="cpu"):
        self.gamma = cfg.gamma
        self.normalize_returns = cfg.normalize_returns
        self.device = torch.device(device)

        self.policy_net = PolicyNetwork(obs_dim, action_dim, cfg.hidden_sizes).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)

    def select_action(self, state: np.ndarray):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.policy_net(state_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.squeeze()

    def compute_returns(self, rewards):
        returns = []
        G = 0.0

        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.append(G)

        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        if self.normalize_returns and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update(self, log_probs, rewards):
        returns = self.compute_returns(rewards)

        log_probs_t = torch.stack(log_probs)  # shape: [T]
        loss = -(log_probs_t * returns).sum()

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

    agent = REINFORCEAgent(obs_dim, action_dim, cfg)

    env_steps = 0
    episode_idx = 0

    returns = []
    return_steps = []
    losses = []

    while env_steps < cfg.total_env_steps:
        state, _ = env.reset(seed=cfg.seed + episode_idx)

        episode_return = 0.0
        episode_rewards = []
        episode_log_probs = []
        done = False

        while not done and env_steps < cfg.total_env_steps:
            action, log_prob = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_log_probs.append(log_prob)
            episode_rewards.append(reward)

            episode_return += reward
            env_steps += 1
            state = next_state

        loss = agent.update(episode_log_probs, episode_rewards)

        episode_idx += 1
        returns.append(episode_return)
        return_steps.append(env_steps)
        losses.append(loss)

        if episode_idx % cfg.log_every_episodes == 0:
            avg_last_10 = np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns)
            print(
                f"Episode {episode_idx:4d} | "
                f"Steps {env_steps:6d} | "
                f"Return {episode_return:6.1f} | "
                f"Avg(Last10) {avg_last_10:6.1f} | "
                f"Loss {loss:9.4f}"
            )

    env.close()

    return np.array(return_steps), np.array(returns), np.array(losses)


# plotting

def plot_learning_curve(return_steps, returns, cfg: Config):
    plt.figure(figsize=(10, 6))

    max_step = int(return_steps[-1])

    # agent
    plt.plot(return_steps, returns, alpha=0.25, label="REINFORCE (raw)")
    if len(returns) >= 20:
        smoothed = moving_average(returns, window=20)
        smoothed_steps = return_steps[19:]
        plt.plot(smoothed_steps, smoothed, linewidth=2, label="REINFORCE (smoothed)")


    plt.axhline(500, linestyle=":", linewidth=1.5, label="Optimal performance = 500")
    plt.xlim(0, max_step)
    plt.xlabel("Environment steps")
    plt.ylabel("Return")
    plt.title("CartPole: REINFORCE")
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