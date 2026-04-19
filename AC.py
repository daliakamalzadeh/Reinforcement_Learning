import random
from dataclasses import dataclass

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


@dataclass
class Config:
    env_name: str = "CartPole-v1"
    seed: int = 42

    total_env_steps: int = 200_000
    gamma: float = 0.99

    actor_learning_rate: float = 1e-3
    critic_learning_rate: float = 1e-3

    hidden_sizes: tuple = (128, 128)

    log_every_episodes: int = 10
    plot_path: str = "actor_critic_cartpole.png"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def moving_average(x, window=20):
    if len(x) < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")


def compute_returns(rewards, gamma):
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns


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
        return self.net(x)


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


class ActorCriticAgent:
    def __init__(self, obs_dim: int, action_dim: int, cfg: Config, device="cpu"):
        self.gamma = cfg.gamma
        self.device = torch.device(device)

        self.actor = PolicyNetwork(obs_dim, action_dim, cfg.hidden_sizes).to(self.device)
        self.critic = QNetwork(obs_dim, action_dim, cfg.hidden_sizes).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_learning_rate)

        self.critic_loss_fn = nn.MSELoss()

    def sample_action(self, state):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits = self.actor(state_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return int(action.item()), log_prob.squeeze()

    def update_episode(self, states, actions, log_probs, returns):
        states_t = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device).unsqueeze(1)

        # ----- actor update -----
        # Use MC return minus critic baseline
        logits = self.actor(states_t)
        probs = torch.softmax(logits, dim=-1)

        with torch.no_grad():
            q_values_detached = self.critic(states_t)                  # [T, action_dim]
            baseline = (probs * q_values_detached).sum(dim=1)         # [T]
            advantage = returns_t.squeeze(1) - baseline               # [T]

        log_probs_t = torch.stack(log_probs)                          # [T]
        actor_loss = -(log_probs_t * advantage).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----- critic update -----
        # Fit Q(s,a) to MC return G_t
        q_values = self.critic(states_t)                              # [T, action_dim]
        q_sa = q_values.gather(1, actions_t)                          # [T, 1]

        critic_loss = self.critic_loss_fn(q_sa, returns_t)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()


def train(cfg: Config):
    set_seed(cfg.seed)

    env = gym.make(cfg.env_name)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = ActorCriticAgent(obs_dim, action_dim, cfg)

    env_steps = 0
    episode_idx = 0

    returns_history = []
    return_steps = []
    actor_losses = []
    critic_losses = []

    while env_steps < cfg.total_env_steps:
        state, _ = env.reset(seed=cfg.seed + episode_idx)
        done = False
        episode_return = 0.0

        states = []
        actions = []
        rewards = []
        log_probs = []

        while not done and env_steps < cfg.total_env_steps:
            action, log_prob = agent.sample_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)

            state = next_state
            episode_return += reward
            env_steps += 1

        mc_returns = compute_returns(rewards, cfg.gamma)

        actor_loss, critic_loss = agent.update_episode(
            states=states,
            actions=actions,
            log_probs=log_probs,
            returns=mc_returns,
        )

        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

        episode_idx += 1
        returns_history.append(episode_return)
        return_steps.append(env_steps)

        if episode_idx % cfg.log_every_episodes == 0:
            avg_last_10 = np.mean(returns_history[-10:]) if len(returns_history) >= 10 else np.mean(returns_history)
            avg_actor = np.mean(actor_losses[-100:]) if len(actor_losses) >= 100 else np.mean(actor_losses)
            avg_critic = np.mean(critic_losses[-100:]) if len(critic_losses) >= 100 else np.mean(critic_losses)

            print(
                f"Episode {episode_idx:4d} | "
                f"Steps {env_steps:6d} | "
                f"Return {episode_return:6.1f} | "
                f"Avg(Last10) {avg_last_10:6.1f} | "
                f"ActorLoss {avg_actor:10.6f} | "
                f"CriticLoss {avg_critic:10.6f}"
            )

    env.close()

    return (
        np.array(return_steps),
        np.array(returns_history),
        np.array(actor_losses),
        np.array(critic_losses),
    )


def plot_learning_curve(return_steps, returns, cfg: Config):
    plt.figure(figsize=(10, 6))

    max_step = int(return_steps[-1])

    plt.plot(return_steps, returns, alpha=0.25, label="AC (raw)")
    if len(returns) >= 20:
        smoothed = moving_average(returns, window=20)
        smoothed_steps = return_steps[19:]
        plt.plot(smoothed_steps, smoothed, linewidth=2, label="AC (smoothed)")

    plt.axhline(500, linestyle=":", linewidth=1.5, label="Optimal performance = 500")
    plt.xlim(0, max_step)
    plt.xlabel("Environment steps")
    plt.ylabel("Return")
    plt.title("CartPole: Actor-Critic")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(cfg.plot_path, dpi=200)
    plt.show()


if __name__ == "__main__":
    cfg = Config()

    return_steps, returns, actor_losses, critic_losses = train(cfg)
    plot_learning_curve(return_steps, returns, cfg)

    print("\nTraining finished.")
    print(f"Best episode return: {returns.max():.1f}")
    print(f"Final 20-episode average: {returns[-20:].mean() if len(returns) >= 20 else returns.mean():.2f}")