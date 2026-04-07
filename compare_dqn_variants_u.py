import copy
import csv
import os
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

from Q_learning_target_replay import Config, train, moving_average


def run_multiple_seeds(cfg: Config, num_seeds=5):
    runs = []
    for i in range(num_seeds):
        cfg_i = copy.deepcopy(cfg)
        cfg_i.seed = cfg.seed + i
        return_steps, returns, losses = train(cfg_i)
        runs.append({"return_steps": return_steps, "returns": returns, "losses": losses})
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


def plot_variant_comparison(curves, cfg: Config, save_path="results_variants/dqn_variant_comparison.png", smoothing_window=20):
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
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
    plt.close()
    print(f"Saved plot to {save_path}")


def summarize_runs(runs):
    final_returns = [run["returns"][-20:].mean() if len(run["returns"]) >= 20 else run["returns"].mean() for run in runs]
    return float(np.mean(final_returns)), float(np.std(final_returns))


def run_variant_study(base_cfg: Config, num_seeds=5, smoothing_window=20, output_dir="results_variants"):
    os.makedirs(output_dir, exist_ok=True)
    variants = {
        "Naive": replace(base_cfg, use_target_network=False, use_replay_buffer=False),
        "Only TN": replace(base_cfg, use_target_network=True, use_replay_buffer=False),
        "Only ER": replace(base_cfg, use_target_network=False, use_replay_buffer=True),
        "TN + ER": replace(base_cfg, use_target_network=True, use_replay_buffer=True),
    }

    curves = {}
    summary_rows = []
    for label, cfg_variant in variants.items():
        print(f"\n{'=' * 70}\nRunning variant: {label}\n{'=' * 70}")
        runs = run_multiple_seeds(cfg_variant, num_seeds=num_seeds)
        curves[label] = runs
        mean_final, std_final = summarize_runs(runs)
        print(f"Final-20 mean across seeds: {mean_final:.2f} ± {std_final:.2f}")
        summary_rows.append({"variant": label, "mean_final_20": mean_final, "std_final_20": std_final})

    plot_variant_comparison(curves, base_cfg, save_path=os.path.join(output_dir, "dqn_variant_comparison.png"), smoothing_window=smoothing_window)

    summary_path = os.path.join(output_dir, "variant_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["variant", "mean_final_20", "std_final_20"])
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Saved variant summary to {summary_path}")


def load_best_config_from_ablation(
    ablation_csv: str = "results_ablation/ablation_summary.csv",
) -> dict:
    """
    Read the ablation summary CSV produced by ablation_study.py and return
    the best (highest mean_final_20) hyperparameter value for each study.
    Falls back to sensible defaults if the file is missing.
    """
    defaults = {
        "learning_rate": 1e-3,
        "hidden_sizes": (128, 128),
        "epsilon_decay_steps": 50_000,
        "updates_per_step": 1.0,
    }

    if not os.path.exists(ablation_csv):
        print(
            f"[WARNING] Ablation CSV not found at '{ablation_csv}'. "
            "Using default hyperparameters. Run ablation_study.py first for best results."
        )
        return defaults

    df = pd.read_csv(ablation_csv)
    best = {}

    study_map = {
        "learning_rate": ("learning_rate", float),
        "network_size": ("hidden_sizes", None),   # special handling below
        "exploration_factor": ("epsilon_decay_steps", int),
        "update_to_data_ratio": ("updates_per_step", float),
    }

    for study, (param, cast) in study_map.items():
        rows = df[df["study"] == study]
        if rows.empty:
            best[param] = defaults[param]
            continue
        best_row = rows.loc[rows["mean_final_20"].idxmax()]
        variant_label = best_row["variant"]

        if study == "learning_rate":
            # label format: "lr = 1e-3"
            best[param] = float(variant_label.split("=")[1].strip())
        elif study == "network_size":
            # label format: "128, 128"
            sizes = tuple(int(x.strip()) for x in variant_label.split(","))
            best["hidden_sizes"] = sizes
        elif study == "exploration_factor":
            # label format: "epsilon_decay = 50k"
            token = variant_label.split("=")[1].strip().lower()
            multiplier = 1_000 if token.endswith("k") else 1
            best[param] = int(float(token.rstrip("k")) * multiplier)
        elif study == "update_to_data_ratio":
            # label format: "1 update/step" or "0.5 updates/step"
            best[param] = float(variant_label.split()[0])

    print("Loaded best hyperparameters from ablation:")
    for k, v in best.items():
        print(f"  {k}: {v}")
    return best


if __name__ == "__main__":

    best = load_best_config_from_ablation("results_ablation/ablation_summary.csv")

    cfg = Config(
        learning_rate=best["learning_rate"],
        hidden_sizes=best["hidden_sizes"],
        epsilon_decay_steps=best["epsilon_decay_steps"],
        updates_per_step=best["updates_per_step"],
        total_env_steps=1_000_000,
        batch_size=64,
        replay_buffer_size=100_000,
        min_replay_size=5_000,
        target_update_freq=1000,
        log_every_episodes=10_000_000,
    )
    run_variant_study(cfg, num_seeds=5, smoothing_window=20, output_dir="results_variants")
