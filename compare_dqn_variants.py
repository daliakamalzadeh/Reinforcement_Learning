import copy
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from Q_learning_target_replay import Config, train, moving_average


def run_multiple_seeds(cfg: Config, num_seeds=3):
    runs = []
    for i in range(num_seeds):
        cfg_i = copy.deepcopy(cfg)
        cfg_i.seed = cfg.seed + i
        return_steps, returns, losses = train(cfg_i)
        runs.append({
            "return_steps": return_steps,
            "returns": returns,
            "losses": losses,
        })
    return runs


def interpolate_runs(runs, total_env_steps, num_points=400):
    x_grid = np.linspace(1, total_env_steps, num_points)
    y_interp_runs = []

    for run in runs:
        x = run["return_steps"]
        y = run["returns"]
        if len(x) == 0:
            continue
        y_interp_runs.append(np.interp(x_grid, x, y))

    if not y_interp_runs:
        return x_grid, np.zeros_like(x_grid), np.zeros_like(x_grid)

    y_interp_runs = np.asarray(y_interp_runs)
    return x_grid, y_interp_runs.mean(axis=0), y_interp_runs.std(axis=0)


def plot_variant_comparison(curves, cfg: Config, save_path="dqn_variant_comparison.png", smoothing_window=20):
    plt.figure(figsize=(10, 6))

    for label, runs in curves.items():
        x, mean, std = interpolate_runs(runs, cfg.total_env_steps)
        if len(mean) >= smoothing_window:
            mean = moving_average(mean, smoothing_window)
            std = moving_average(std, smoothing_window)
            x = x[smoothing_window - 1:]
        plt.plot(x, mean, linewidth=2.2, label=label)
        plt.fill_between(x, mean - std, mean + std, alpha=0.2)

    plt.axhline(500, linestyle=":", linewidth=1.5, label="Optimal return = 500")
    plt.xlabel("Environment steps")
    plt.ylabel("Return")
    plt.title("CartPole: Naive vs TN vs ER vs TN + ER")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


def run_variant_study(base_cfg: Config, num_seeds=3, smoothing_window=20):
    variants = {
        "Naive": replace(base_cfg, use_target_network=False, use_replay_buffer=False),
        "Only TN": replace(base_cfg, use_target_network=True, use_replay_buffer=False),
        "Only ER": replace(base_cfg, use_target_network=False, use_replay_buffer=True),
        "TN + ER": replace(base_cfg, use_target_network=True, use_replay_buffer=True),
    }

    curves = {}
    for label, cfg_variant in variants.items():
        print(f"\n{'=' * 70}")
        print(f"Running variant: {label}")
        print(f"{'=' * 70}")
        curves[label] = run_multiple_seeds(cfg_variant, num_seeds=num_seeds)

    plot_variant_comparison(curves, base_cfg, smoothing_window=smoothing_window)


if __name__ == "__main__":
    cfg = Config(
        learning_rate=1e-3,
        hidden_sizes=(128, 128),
        epsilon_decay_steps=20_000,
        updates_per_step=2.0,
        total_env_steps=50_000,
        batch_size=64,
        replay_buffer_size=50_000,
        min_replay_size=1_000,
        target_update_freq=500,
        log_every_episodes=10_000_000,
    )
    run_variant_study(cfg, num_seeds=3, smoothing_window=20)
