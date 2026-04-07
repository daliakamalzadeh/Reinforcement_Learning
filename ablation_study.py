import copy
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np

from Q_learning import Config, train, moving_average


# multi-seed utilities

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

        y_interp = np.interp(x_grid, x, y)
        y_interp_runs.append(y_interp)

    y_interp_runs = np.array(y_interp_runs)

    if len(y_interp_runs) == 0:
        return x_grid, np.zeros_like(x_grid), np.zeros_like(x_grid)

    mean = y_interp_runs.mean(axis=0)
    std = y_interp_runs.std(axis=0)
    return x_grid, mean, std


# plotting

def plot_ablation_result(study_name, curves, cfg: Config, save_path: str, smoothing_window=20):
    plt.figure(figsize=(10, 6))

    for label, runs in curves.items():
        x, mean, std = interpolate_runs(runs, cfg.total_env_steps)

        if len(mean) >= smoothing_window:
            mean_sm = moving_average(mean, smoothing_window)
            std_sm = moving_average(std, smoothing_window)
            x_sm = x[smoothing_window - 1:]
        else:
            mean_sm = mean
            std_sm = std
            x_sm = x

        plt.plot(x_sm, mean_sm, linewidth=2.2, label=label)
        plt.fill_between(x_sm, mean_sm - std_sm, mean_sm + std_sm, alpha=0.2)

    plt.axhline(500, linestyle=":", linewidth=1.5, label="Optimal return = 500")
    plt.xlabel("Environment steps")
    plt.ylabel("Return")
    plt.title(f"Ablation Study: {study_name}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.show()


# ablation study

def run_ablation_study(base_cfg: Config, num_seeds=3, smoothing_window=20):
    studies = {
        # "learning_rate": {
        #     "lr = 1e-4": replace(base_cfg, learning_rate=1e-4),
        #     "lr = 1e-3": replace(base_cfg, learning_rate=1e-3),
        #     "lr = 5e-3": replace(base_cfg, learning_rate=5e-3),
        # },
        # "network_architecture": {
        #     "64, 64": replace(base_cfg, hidden_sizes=(64, 64)),
        #     "128, 128": replace(base_cfg, hidden_sizes=(128, 128)),
        #     "256, 256": replace(base_cfg, hidden_sizes=(256, 256)),
        # },
        # "epsilon_decay": {
        #     "10k steps": replace(base_cfg, epsilon_decay_steps=10_000),
        #     "20k steps": replace(base_cfg, epsilon_decay_steps=20_000),
        #     "50k steps": replace(base_cfg, epsilon_decay_steps=50_000),
        # },
        "update_to_data_ratio": {
            "0.5 updates/step": replace(base_cfg, updates_per_step=0.5),
            "1 update/step": replace(base_cfg, updates_per_step=1.0),
            "2 updates/step": replace(base_cfg, updates_per_step=2.0),
        },
    }

    for study_name, variants in studies.items():
        print(f"\n{'=' * 70}")
        print(f"Running ablation: {study_name}")
        print(f"{'=' * 70}")

        curves = {}

        for label, cfg_variant in variants.items():
            print(f"\nVariant: {label}")
            runs = run_multiple_seeds(cfg_variant, num_seeds=num_seeds)
            curves[label] = runs

        file_name = f"ablation_{study_name}.png".replace(" ", "_")
        plot_ablation_result(
            study_name,
            curves,
            base_cfg,
            file_name,
            smoothing_window=smoothing_window
        )


# main

if __name__ == "__main__":
    cfg = Config()
    cfg.total_env_steps = 1_000_000
    cfg.log_every_episodes = 10_000_000
    run_ablation_study(cfg, num_seeds=5, smoothing_window=20)