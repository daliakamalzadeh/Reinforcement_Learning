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
    """Configuration class setting up all hyperparameters and experiment settings."""

    env_name: str = "CartPole-v1"
    seed: int = 42

    total_env_steps: int = 1_000_000
    gamma: float = 0.99

    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3

    hidden_sizes: tuple = (128, 128)

    normalize_advantages: bool = True

    log_every_episodes: int = 10
    plot_path: str = "a2c_cartpole.png"


def set_seed(seed: int):
    """Setting random seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def moving_average(x, window=20):
    """Compute a moving average over a fixed window for smoothing learning curves."""

    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


# networks

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes=(128, 128)):
        """
        Initializing the policy network.

        Maps observations to action logits, which are later converted
        into probabilities using a categorical distribution.
        """

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


class ValueNetwork(nn.Module):
    def __init__(self, obs_dim: int, hidden_sizes=(128, 128)):
        """
        Initialize value network.

        Outputs a scalar value estimate for each state.
        """

        super().__init__()

        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # V(s)


# agent

class A2CAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: Config, device="cpu"):
        """
        Initialize actor, critic, optimizers, and loss function.
        """

        self.gamma = cfg.gamma
        self.normalize_advantages = cfg.normalize_advantages
        self.device = torch.device(device)

        self.actor = PolicyNetwork(obs_dim, action_dim, cfg.hidden_sizes).to(self.device)
        self.critic = ValueNetwork(obs_dim, cfg.hidden_sizes).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_learning_rate)

        self.value_loss_fn = nn.MSELoss()

    def select_action(self, state: np.ndarray):
        """Sample an action from the current policy."""

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.actor(state_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.squeeze()

    def compute_returns(self, rewards):
        """Compute discounted Monte Carlo returns."""

        returns = []
        G = 0.0

        for reward in reversed(rewards):
            G = reward + self.gamma * G
            returns.append(G)

        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32, device=self.device)

    def update(self, states, log_probs, rewards):
        """Update the actor and critic networks."""

        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)  
        log_probs_t = torch.stack(log_probs)                                                 
        returns_t = self.compute_returns(rewards)                                            

        # critic: V(s)
        values_t = self.critic(states_t)                                                     

        # advantage: A = G - V(s)
        advantages_t = returns_t - values_t

        # actor update: use detached advantages
        actor_advantages = advantages_t.detach()
        if self.normalize_advantages and len(actor_advantages) > 1:
            actor_advantages = (
                (actor_advantages - actor_advantages.mean()) /
                (actor_advantages.std() + 1e-8)
            )

        actor_loss = -(log_probs_t * actor_advantages).sum()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # critic update: fit V(s) to Monte Carlo returns
        critic_loss = self.value_loss_fn(values_t, returns_t)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()


# training

def train(cfg: Config):
    """
    Training the A2C agent.

    After each episode, discounted returns are computed and used to update
    the critic (value estimation) and the actor (policy improvement via advantages).
    """

    set_seed(cfg.seed)

    env = gym.make(cfg.env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(obs_dim, action_dim, cfg)

    env_steps = 0
    episode_idx = 0

    returns = []
    return_steps = []
    actor_losses = []
    critic_losses = []

    while env_steps < cfg.total_env_steps:
        state, _ = env.reset(seed=cfg.seed + episode_idx)

        episode_return = 0.0
        done = False

        episode_states = []
        episode_log_probs = []
        episode_rewards = []

        while not done and env_steps < cfg.total_env_steps:
            action, log_prob = agent.select_action(state)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_states.append(state)
            episode_log_probs.append(log_prob)
            episode_rewards.append(reward)

            episode_return += reward
            env_steps += 1
            state = next_state

        actor_loss, critic_loss = agent.update(
            episode_states,
            episode_log_probs,
            episode_rewards
        )

        episode_idx += 1
        returns.append(episode_return)
        return_steps.append(env_steps)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

        if episode_idx % cfg.log_every_episodes == 0:
            avg_last_10 = np.mean(returns[-10:]) if len(returns) >= 10 else np.mean(returns)
            print(
                f"Episode {episode_idx:4d} | "
                f"Steps {env_steps:6d} | "
                f"Return {episode_return:6.1f} | "
                f"Avg(Last10) {avg_last_10:6.1f} | "
                f"ActorLoss {actor_loss:9.4f} | "
                f"CriticLoss {critic_loss:9.4f}"
            )

    env.close()

    return (
        np.array(return_steps),
        np.array(returns),
        np.array(actor_losses),
        np.array(critic_losses),
    )


# plotting

def plot_learning_curve(return_steps, returns, cfg: Config):
    plt.figure(figsize=(10, 6))

    max_step = int(return_steps[-1])

    # agent
    plt.plot(return_steps, returns, alpha=0.25, label="A2C (raw)")
    if len(returns) >= 20:
        smoothed = moving_average(returns, window=20)
        smoothed_steps = return_steps[19:]
        plt.plot(smoothed_steps, smoothed, linewidth=2, label="A2C (smoothed)")


    plt.axhline(500, linestyle=":", linewidth=1.5, label="Optimal performance = 500")
    plt.xlim(0, max_step)
    plt.xlabel("Environment steps")
    plt.ylabel("Return")
    plt.title("CartPole: A2C")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.plot_path, dpi=200)
    plt.show()


# main

if __name__ == "__main__":
    cfg = Config()

    return_steps, returns, actor_losses, critic_losses = train(cfg)
    plot_learning_curve(return_steps, returns, cfg)

    print("\nTraining finished.")
    print(f"Best episode return: {returns.max():.1f}")
    print(f"Final 20-episode average: {returns[-20:].mean() if len(returns) >= 20 else returns.mean():.2f}")