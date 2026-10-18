"""Compare Taxi learning rates while holding other saved settings fixed."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analyze_training import moving_average


HISTORY_KEYS = (
    "rewards", "steps", "successes", "epsilons",
    "illegal_pickups", "illegal_dropoffs", "illegal_actions",
)


def load_runs(paths):
    """Check that the experiment changes only alpha, including the same seed."""
    runs = []
    for path in paths:
        with np.load(path, allow_pickle=False) as archive:
            config = json.loads(archive["config"].item())
            history = {key: archive[key].copy() for key in HISTORY_KEYS}
        count = len(history["rewards"])
        if count < 1 or count != config["episodes"]:
            raise ValueError(f"{path}: history length does not match configured episodes.")
        if any(values.shape != (count,) or not np.isfinite(values).all()
               for values in history.values()):
            raise ValueError(f"{path}: invalid training history.")
        runs.append({"path": str(path), "config": config, "history": history})
    if len(runs) < 2:
        raise ValueError("Provide at least two different learning rates.")
    fixed = {key: value for key, value in runs[0]["config"].items() if key != "alpha"}
    for run in runs:
        settings = {key: value for key, value in run["config"].items() if key != "alpha"}
        if settings != fixed:
            raise ValueError("All saved settings except alpha must match, including seed and episodes.")
    if len({run["config"]["alpha"] for run in runs}) != len(runs):
        raise ValueError("Each run must have a different alpha.")
    return sorted(runs, key=lambda run: run["config"]["alpha"])


def summarize(run, start, stop):
    """Describe the same episode interval for every learning rate."""
    history = run["history"]
    section = slice(start, stop)
    return {
        "alpha": run["config"]["alpha"],
        "first_episode": start + 1,
        "last_episode": stop,
        "episodes": stop - start,
        "mean_reward": history["rewards"][section].mean(),
        "reward_std": history["rewards"][section].std(ddof=0),
        "success_rate": history["successes"][section].mean(),
        "mean_steps": history["steps"][section].mean(),
        "mean_illegal_pickups": history["illegal_pickups"][section].mean(),
        "mean_illegal_dropoffs": history["illegal_dropoffs"][section].mean(),
        "mean_illegal_actions": history["illegal_actions"][section].mean(),
    }


def write_table(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_alpha(paths, output_dir, window=100, zoom_episodes=2000, zoom_window=25,
                  final_window=100, stability_episodes=1000):
    """Export matched views of learning speed, reward variability, and final training results."""
    if min(window, zoom_episodes, zoom_window, final_window, stability_episodes) < 1:
        raise ValueError("Episode counts and averaging windows must be positive.")
    runs = load_runs(paths)
    count = len(runs[0]["history"]["rewards"])
    window = min(window, count)
    zoom_count = min(zoom_episodes, count)
    zoom_window = min(zoom_window, zoom_count)
    final_window = min(final_window, count)
    stability_count = min(stability_episodes, count)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    final_rows = [summarize(run, count - final_window, count) for run in runs]
    write_table(output_dir / "final_performance.csv", final_rows)
    write_table(output_dir / "episode_blocks.csv", [
        summarize(run, start, min(start + window, count))
        for run in runs for start in range(0, count, window)
    ])
    (output_dir / "comparison_config.json").write_text(json.dumps({
        "runs": [{"source": run["path"], "config": run["config"]} for run in runs],
        "window": window, "zoom_episodes": zoom_count, "zoom_window": zoom_window,
        "final_window": final_window, "stability_episodes": stability_count,
        "reward_std_ddof": 0, "metrics": "Training with exploration enabled",
    }, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = plt.get_cmap("tab10").colors
    styles = ("-", "--", "-.", ":")
    episodes = np.arange(1, count + 1)
    for name, shown, average_window, title in (
        ("reward_comparison.png", count, window, "Taxi: learning-rate comparison"),
        ("reward_comparison_early.png", zoom_count, zoom_window,
         f"Taxi: early learning (first {zoom_count:,} episodes)"),
    ):
        figure, axis = plt.subplots(figsize=(10, 5))
        for i, run in enumerate(runs):
            axis.plot(
                episodes[average_window - 1:shown],
                moving_average(run["history"]["rewards"][:shown], average_window),
                label=f"alpha = {run['config']['alpha']}",
                color=colors[i % len(colors)], linestyle=styles[i % len(styles)],
            )
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.set(xlabel="Training episode", ylabel="Mean total reward",
                 title=f"{title}\n{average_window}-episode moving average")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / name, dpi=180)
        plt.close(figure)

    # Show auxiliary early-learning trends using the same smoothing and x range.
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for i, run in enumerate(runs):
        for axis, key, scale in zip(axes, ("successes", "steps", "illegal_actions"), (100, 1, 1)):
            axis.plot(
                episodes[zoom_window - 1:zoom_count],
                scale * moving_average(run["history"][key][:zoom_count], zoom_window),
                label=f"alpha = {run['config']['alpha']}",
                color=colors[i % len(colors)], linestyle=styles[i % len(styles)],
            )
    axes[0].set(ylabel="Success rate (%)", ylim=(0, 100))
    axes[1].set(ylabel="Mean steps")
    axes[2].set(ylabel="Mean illegal attempts", xlabel="Training episode")
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].legend()
    figure.suptitle(f"Taxi: early training metrics ({zoom_window}-episode moving averages)")
    figure.tight_layout()
    figure.savefig(output_dir / "metrics_comparison_early.png", dpi=180)
    plt.close(figure)

    # Separate raw traces on matching axes keep fluctuations visible for each run.
    figure, axes = plt.subplots(len(runs) + 1, 1, figsize=(10, 3 * (len(runs) + 1)), sharex=True)
    tail_episodes = episodes[-stability_count:]
    tail_rewards = [run["history"]["rewards"][-stability_count:] for run in runs]
    lower = min(values.min() for values in tail_rewards) - 1
    upper = max(values.max() for values in tail_rewards) + 1
    std_window = min(window, stability_count)
    for i, (run, values) in enumerate(zip(runs, tail_rewards)):
        label = f"alpha = {run['config']['alpha']}"
        color = colors[i % len(colors)]
        axes[i].plot(tail_episodes, values, color=color, linewidth=0.8)
        axes[i].set(ylabel="Episode reward", title=label, ylim=(lower, upper))
        rolling_std = np.std(np.lib.stride_tricks.sliding_window_view(values, std_window), axis=1)
        axes[-1].plot(tail_episodes[std_window - 1:], rolling_std, label=label,
                      color=color, linestyle=styles[i % len(styles)])
    axes[-1].set(xlabel="Training episode", ylabel="Reward std. deviation",
                 title=f"{std_window}-episode rolling standard deviation", ylim=(0, None))
    axes[-1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(f"Taxi: late training variability (last {stability_count:,} episodes)")
    figure.tight_layout()
    figure.savefig(output_dir / "stability_comparison.png", dpi=180)
    plt.close(figure)
    return final_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=Path, help="Training archives with different alpha values")
    parser.add_argument("--output-dir", type=Path, default=Path("results/alpha_comparison"))
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--zoom-episodes", type=int, default=2000)
    parser.add_argument("--zoom-window", type=int, default=25)
    parser.add_argument("--final-window", type=int, default=100)
    parser.add_argument("--stability-episodes", type=int, default=1000)
    args = parser.parse_args()
    try:
        rows = compare_alpha(args.models, args.output_dir, args.window, args.zoom_episodes,
                             args.zoom_window, args.final_window, args.stability_episodes)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    for row in rows:
        print(f"alpha={row['alpha']}: mean reward={row['mean_reward']:.2f}, "
              f"reward std={row['reward_std']:.2f}, success={100 * row['success_rate']:.1f}%")
    print(f"Saved comparison plots and tables to {args.output_dir}")


if __name__ == "__main__":
    main()
