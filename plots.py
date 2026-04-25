import argparse
import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# config

SMOOTH_WINDOW = 20   # moving-average window applied per seed before averaging
INTERP_POINTS = 500  # resolution of the shared step grid

ALGO_STYLE = {
    "reinforce": dict(color="#E07B39", label="REINFORCE"),
    "ac":        dict(color="#4A90D9", label="Actor-Critic (AC)"),
    "a2c":       dict(color="#4CAF50", label="A2C"),
}

DQN_A2_STYLE = dict(color="#9B59B6", label="DQN TN+ER (Assignment 2)", linestyle="--")



# Data loading and preprocessing


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    kernel = np.ones(window) / window
    padded = np.concatenate([np.full(window - 1, x[0]), x])
    return np.convolve(padded, kernel, mode="valid")


def load_seed_files(results_dir: str, algo: str) -> list:
    files = sorted(glob.glob(os.path.join(results_dir, f"{algo}_seed*.csv")))
    return [(pd.read_csv(f)["env_step"].to_numpy(),
             pd.read_csv(f)["episode_return"].to_numpy()) for f in files]


def interpolate_to_grid(runs: list, x_grid: np.ndarray, smooth_window: int) -> np.ndarray:
    return np.array([
        np.interp(x_grid, steps, moving_average(returns, smooth_window))
        for steps, returns in runs
    ])


# Plotting


def plot(results_dir: str, results_a2_dir: str, out_path: str,
         smooth_window: int = SMOOTH_WINDOW, n_points: int = INTERP_POINTS):

    all_runs = {algo: load_seed_files(results_dir, algo) for algo in ALGO_STYLE}
    max_step = min(
        max(steps[-1] for steps, _ in runs)
        for runs in all_runs.values() if runs
    )
    x_grid = np.linspace(0, max_step, n_points)

    fig, ax = plt.subplots(figsize=(11, 6))

    # Assignment 3 algorithms - REINFORCE, AC, A2C
    for algo, style in ALGO_STYLE.items():
        runs   = all_runs[algo]
        matrix = interpolate_to_grid(runs, x_grid, smooth_window)
        mean   = matrix.mean(axis=0)
        std    = matrix.std(axis=0)

        ax.plot(x_grid, mean, color=style["color"], linewidth=2.0,
                label=f"{style['label']} (n={len(runs)})")
        ax.fill_between(x_grid, mean - std, mean + std,
                        color=style["color"], alpha=0.18)

        print(f"  {algo:10s}: {len(runs)} seed(s)  "
              f"final mean={mean[-1]:.1f} +/- {std[-1]:.1f}")

    # Assignment 2 DQN TN+ER
    a2_runs = load_seed_files(results_a2_dir, "dqn_tn_er")
    matrix  = interpolate_to_grid(a2_runs, x_grid, smooth_window)
    mean    = matrix.mean(axis=0)
    std     = matrix.std(axis=0)

    ax.plot(x_grid, mean, color=DQN_A2_STYLE["color"],
            linestyle=DQN_A2_STYLE["linestyle"], linewidth=2.0,
            label=f"{DQN_A2_STYLE['label']} (n={len(a2_runs)})")
    ax.fill_between(x_grid, mean - std, mean + std,
                    color=DQN_A2_STYLE["color"], alpha=0.18)

    print(f"  {'dqn_tn_er':10s}: {len(a2_runs)} seed(s)  "
          f"final mean={mean[-1]:.1f} +/- {std[-1]:.1f}")

    # Aesthetics
    ax.axhline(500, color="red", linestyle=":", linewidth=1.5,
               label="Optimal return (500)")
    ax.set_xlim(0, max_step)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Environment steps", fontsize=12)
    ax.set_ylabel("Episode return", fontsize=12)
    ax.set_title("CartPole-v1: REINFORCE, AC, A2C vs DQN (Assignment 2)", fontsize=13)
    ax.legend(fontsize=10, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plt.savefig(out_path, dpi=200)
    print(f"\n  Plot saved -> {out_path}")
    plt.show()


# main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 3 - combined learning curve plot")
    parser.add_argument("--results_dir",    type=str, default="results",
                        help="Per-seed CSVs from run_experiments.py (Assignment 3)")
    parser.add_argument("--results_a2_dir", type=str, default="results_a2",
                        help="Per-seed CSVs from compare_dqn_variants_u.py (Assignment 2)")
    parser.add_argument("--out",            type=str, default="learning_curves.png",
                        help="Output path for the figure")
    parser.add_argument("--smooth",         type=int, default=SMOOTH_WINDOW,
                        help="Moving-average window in episodes")
    args = parser.parse_args()

    print(f"Assignment 3 results : {args.results_dir}")
    print(f"Assignment 2 results : {args.results_a2_dir}")
    print(f"Output               : {args.out}")
    print(f"Smoothing window     : {args.smooth} episodes\n")

    plot(
        results_dir    = args.results_dir,
        results_a2_dir = args.results_a2_dir,
        out_path       = args.out,
        smooth_window  = args.smooth,
    )