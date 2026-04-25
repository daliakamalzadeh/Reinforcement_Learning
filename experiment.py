import argparse
import os
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical


# Config

@dataclass
class Config:
    env_name: str = "CartPole-v1"
    total_env_steps: int = 1_000_000
    gamma: float = 0.99
    hidden_sizes: tuple = (128, 128)
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    normalize_returns: bool = True     # REINFORCE only
    normalize_advantages: bool = True  # A2C only
    n_seeds: int = 5
    base_seed: int = 0
    results_dir: str = "results"
    log_every_episodes: int = 50


# Helpers

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_mlp(in_dim: int, out_dim: int, hidden_sizes: tuple) -> nn.Sequential:
    layers = []
    d = in_dim
    for h in hidden_sizes:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


def compute_mc_returns(rewards: list, gamma: float) -> list:
    """Compute discounted Monte Carlo returns from a list of rewards."""
    returns, G = [], 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    returns.reverse()
    return returns


def save_results(algo: str, seed: int, steps: np.ndarray,
                 returns: np.ndarray, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    path = os.path.join(results_dir, f"{algo}_seed{seed}.csv")
    pd.DataFrame({"env_step": steps, "episode_return": returns}).to_csv(path, index=False)
    print(f"  Saved -> {path}")


# Networks
# All three algorithms share the same policy network architecture,
# ensuring a fair comparison across methods.

class PolicyNetwork(nn.Module):
    """Maps state -> action logits."""
    def __init__(self, obs_dim, action_dim, hidden_sizes):
        super().__init__()
        self.net = make_mlp(obs_dim, action_dim, hidden_sizes)

    def forward(self, x):
        return self.net(x)


class QNetwork(nn.Module):
    """Maps state -> Q-value per action (used by AC)."""
    def __init__(self, obs_dim, action_dim, hidden_sizes):
        super().__init__()
        self.net = make_mlp(obs_dim, action_dim, hidden_sizes)

    def forward(self, x):
        return self.net(x)


class ValueNetwork(nn.Module):
    """Maps state -> scalar V(s) (used by A2C)."""
    def __init__(self, obs_dim, hidden_sizes):
        super().__init__()
        self.net = make_mlp(obs_dim, 1, hidden_sizes)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# REINFORCE

def run_reinforce(cfg: Config, seed: int):
    set_seed(seed)
    env = gym.make(cfg.env_name)
    obs_dim, action_dim = env.observation_space.shape[0], env.action_space.n
    device = torch.device("cpu")

    policy = PolicyNetwork(obs_dim, action_dim, cfg.hidden_sizes).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=cfg.actor_lr)

    env_steps, episode_idx = 0, 0
    steps_log, returns_log = [], []

    while env_steps < cfg.total_env_steps:
        state, _ = env.reset(seed=seed + episode_idx)
        done = False
        ep_return, ep_rewards, ep_log_probs = 0.0, [], []

        while not done and env_steps < cfg.total_env_steps:
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            dist = Categorical(logits=policy(s_t))
            action = dist.sample()
            log_prob = dist.log_prob(action).squeeze()

            state, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated

            ep_log_probs.append(log_prob)
            ep_rewards.append(reward)
            ep_return += reward
            env_steps += 1

        # Compute discounted returns and optionally normalise to reduce variance
        returns = torch.tensor(compute_mc_returns(ep_rewards, cfg.gamma),
                               dtype=torch.float32, device=device)
        if cfg.normalize_returns and len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        loss = -(torch.stack(ep_log_probs) * returns).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        steps_log.append(env_steps)
        returns_log.append(ep_return)
        episode_idx += 1

        if episode_idx % cfg.log_every_episodes == 0:
            avg = np.mean(returns_log[-10:])
            print(f"  [REINFORCE seed={seed}] ep={episode_idx:4d}  "
                  f"steps={env_steps:7d}  ret={ep_return:6.1f}  avg10={avg:6.1f}")

    env.close()
    return np.array(steps_log), np.array(returns_log)


# Actor-Critic (basic AC)
# The critic learns Q(s, a) via MC returns; the actor is updated using
# Q(s, a_taken) directly as the return signal with no baseline subtraction.

def run_ac(cfg: Config, seed: int):
    set_seed(seed)
    env = gym.make(cfg.env_name)
    obs_dim, action_dim = env.observation_space.shape[0], env.action_space.n
    device = torch.device("cpu")

    actor  = PolicyNetwork(obs_dim, action_dim, cfg.hidden_sizes).to(device)
    critic = QNetwork(obs_dim, action_dim, cfg.hidden_sizes).to(device)
    actor_opt  = optim.Adam(actor.parameters(),  lr=cfg.actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    mse = nn.MSELoss()

    env_steps, episode_idx = 0, 0
    steps_log, returns_log = [], []

    while env_steps < cfg.total_env_steps:
        state, _ = env.reset(seed=seed + episode_idx)
        done = False
        ep_return = 0.0
        ep_states, ep_actions, ep_rewards, ep_log_probs = [], [], [], []

        while not done and env_steps < cfg.total_env_steps:
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            dist = Categorical(logits=actor(s_t))
            action = dist.sample()
            log_prob = dist.log_prob(action).squeeze()

            next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated

            ep_states.append(state)
            ep_actions.append(int(action.item()))
            ep_rewards.append(reward)
            ep_log_probs.append(log_prob)
            ep_return += reward
            env_steps += 1
            state = next_state

        states_t  = torch.tensor(np.array(ep_states), dtype=torch.float32, device=device)
        actions_t = torch.tensor(ep_actions, dtype=torch.long, device=device).unsqueeze(1)
        returns_t = torch.tensor(compute_mc_returns(ep_rewards, cfg.gamma),
                                 dtype=torch.float32, device=device).unsqueeze(1)

        # Critic first: fit Q(s, a_taken) to MC return
        q_sa = critic(states_t).gather(1, actions_t)
        critic_loss = mse(q_sa, returns_t)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        # Actor: use updated Q(s, a_taken) as the return signal
        with torch.no_grad():
            q_sa_detached = critic(states_t).gather(1, actions_t).squeeze(1)
        actor_loss = -(torch.stack(ep_log_probs) * q_sa_detached).mean()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        steps_log.append(env_steps)
        returns_log.append(ep_return)
        episode_idx += 1

        if episode_idx % cfg.log_every_episodes == 0:
            avg = np.mean(returns_log[-10:])
            print(f"  [AC      seed={seed}] ep={episode_idx:4d}  "
                  f"steps={env_steps:7d}  ret={ep_return:6.1f}  avg10={avg:6.1f}")

    env.close()
    return np.array(steps_log), np.array(returns_log)


# Advantage Actor-Critic (A2C)
# Extends AC by subtracting a V(s) baseline from the MC return to form
# the advantage A = G - V(s), reducing gradient variance.

def run_a2c(cfg: Config, seed: int):
    set_seed(seed)
    env = gym.make(cfg.env_name)
    obs_dim, action_dim = env.observation_space.shape[0], env.action_space.n
    device = torch.device("cpu")

    actor  = PolicyNetwork(obs_dim, action_dim, cfg.hidden_sizes).to(device)
    critic = ValueNetwork(obs_dim, cfg.hidden_sizes).to(device)
    actor_opt  = optim.Adam(actor.parameters(),  lr=cfg.actor_lr)
    critic_opt = optim.Adam(critic.parameters(), lr=cfg.critic_lr)
    mse = nn.MSELoss()

    env_steps, episode_idx = 0, 0
    steps_log, returns_log = [], []

    while env_steps < cfg.total_env_steps:
        state, _ = env.reset(seed=seed + episode_idx)
        done = False
        ep_return = 0.0
        ep_states, ep_rewards, ep_log_probs = [], [], []

        while not done and env_steps < cfg.total_env_steps:
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            dist = Categorical(logits=actor(s_t))
            action = dist.sample()
            log_prob = dist.log_prob(action).squeeze()

            next_state, reward, terminated, truncated, _ = env.step(int(action.item()))
            done = terminated or truncated

            ep_states.append(state)
            ep_rewards.append(reward)
            ep_log_probs.append(log_prob)
            ep_return += reward
            env_steps += 1
            state = next_state

        states_t  = torch.tensor(np.array(ep_states), dtype=torch.float32, device=device)
        returns_t = torch.tensor(compute_mc_returns(ep_rewards, cfg.gamma),
                                 dtype=torch.float32, device=device)

        # Advantage: A = G - V(s); detach before actor update so gradients
        # do not flow back through the critic during the actor step.
        values_t   = critic(states_t)
        advantages = (returns_t - values_t).detach()
        if cfg.normalize_advantages and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss = -(torch.stack(ep_log_probs) * advantages).sum()
        actor_opt.zero_grad()
        actor_loss.backward()
        actor_opt.step()

        # Critic: fit V(s) to MC returns
        critic_loss = mse(critic(states_t), returns_t)
        critic_opt.zero_grad()
        critic_loss.backward()
        critic_opt.step()

        steps_log.append(env_steps)
        returns_log.append(ep_return)
        episode_idx += 1

        if episode_idx % cfg.log_every_episodes == 0:
            avg = np.mean(returns_log[-10:])
            print(f"  [A2C     seed={seed}] ep={episode_idx:4d}  "
                  f"steps={env_steps:7d}  ret={ep_return:6.1f}  avg10={avg:6.1f}")

    env.close()
    return np.array(steps_log), np.array(returns_log)


# Runner

ALGO_FN = {
    "reinforce": run_reinforce,
    "ac":        run_ac,
    "a2c":       run_a2c,
}


def run_all(algos: list, cfg: Config):
    for algo in algos:
        print(f"\n{'='*60}")
        print(f"  {algo.upper()}  |  {cfg.n_seeds} seeds x {cfg.total_env_steps:,} steps")
        print(f"{'='*60}")
        for i in range(cfg.n_seeds):
            seed = cfg.base_seed + i
            print(f"\n  --- Seed {seed} ---")
            steps, returns = ALGO_FN[algo](cfg, seed)
            save_results(algo, seed, steps, returns, cfg.results_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 3 experiment runner")
    parser.add_argument("--algo", nargs="+", choices=["reinforce", "ac", "a2c"],
                        default=["reinforce", "ac", "a2c"])
    parser.add_argument("--seeds",       type=int, default=None)
    parser.add_argument("--steps",       type=int, default=None)
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()

    cfg = Config()
    cfg.results_dir = args.results_dir
    if args.seeds is not None:
        cfg.n_seeds = args.seeds
    if args.steps is not None:
        cfg.total_env_steps = args.steps

    run_all(args.algo, cfg)

    print("\nAll experiments finished.")
    print(f"Results saved to: {os.path.abspath(cfg.results_dir)}/")