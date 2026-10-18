"""Export readable tables and plots from a saved Taxi training run."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def moving_average(values, window):
    """Average complete windows, without padding the start with zeros."""
    return np.convolve(values, np.ones(window) / window, mode="valid")


def export_training(result, config, output_dir, window=100, zoom_episodes=2000, zoom_window=25):
    """Save training history, summary statistics, Q-values, and PNG plots."""
    if window < 1:
        raise ValueError("The moving-average window must be positive.")
    if zoom_episodes < 1 or zoom_window < 1:
        raise ValueError("Zoom episode count and averaging window must be positive.")
    rewards = result["rewards"]
    steps = result["steps"]
    successes = result["successes"]
    epsilons = result["epsilons"]
    count = len(rewards)
    if count == 0 or any(len(values) != count for values in (steps, successes, epsilons)):
        raise ValueError("Training histories must have the same nonzero length.")
    illegal_keys = ("illegal_pickups", "illegal_dropoffs", "illegal_actions")
    has_illegal_counts = all(key in result for key in illegal_keys)
    if any(key in result for key in illegal_keys) and not has_illegal_counts:
        raise ValueError("Expected all three illegal-action count arrays.")
    if has_illegal_counts and any(len(result[key]) != count for key in illegal_keys):
        raise ValueError("Illegal-action histories must match the episode count.")
    window = min(window, count)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "episodes.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["episode", "reward", "steps", "success", "epsilon", *illegal_keys])
        for episode in range(count):
            writer.writerow([
                episode + 1, rewards[episode], steps[episode],
                int(successes[episode]), epsilons[episode],
                *(result[key][episode] if has_illegal_counts else "" for key in illegal_keys),
            ])

    # These statistics describe training, which includes exploratory actions.
    with (output_dir / "summary.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "period", "episodes", "mean_reward", "mean_steps", "success_rate",
            "first_epsilon", "last_epsilon",
            "mean_illegal_pickups", "mean_illegal_dropoffs", "mean_illegal_actions",
            "total_illegal_actions",
        ])
        for label, section in (
            ("all", slice(0, count)),
            (f"first_{window}", slice(0, window)),
            (f"last_{window}", slice(count - window, count)),
        ):
            writer.writerow([
                label, len(rewards[section]), rewards[section].mean(),
                steps[section].mean(), successes[section].mean(),
                epsilons[section][0], epsilons[section][-1],
                *(result[key][section].mean() if has_illegal_counts else "" for key in illegal_keys),
                result["illegal_actions"][section].sum() if has_illegal_counts else "",
            ])

    with (output_dir / "q_table.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["state", "south", "north", "east", "west", "pickup", "dropoff"])
        for state, row in enumerate(result["q_table"]):
            writer.writerow([state, *row])

    (output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    # A file-only backend also works when no desktop display is available.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    episodes = np.arange(1, count + 1)
    averaged_episodes = episodes[window - 1:]
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(episodes, rewards, alpha=0.25, linewidth=0.7, label="Episode reward")
    axis.plot(
        averaged_episodes, moving_average(rewards, window),
        label=f"{window}-episode moving average", color="tab:orange",
    )
    axis.set(xlabel="Training episode", ylabel="Total reward", title="Taxi: training reward")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "reward_curve.png", dpi=180)
    plt.close(figure)

    # Keep every early episode visible while smoothing over a shorter window.
    zoom_count = min(zoom_episodes, count)
    zoom_window = min(zoom_window, zoom_count)
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.plot(
        episodes[:zoom_count], rewards[:zoom_count],
        alpha=0.25, linewidth=0.7, label="Episode reward",
    )
    axis.plot(
        episodes[zoom_window - 1:zoom_count],
        moving_average(rewards[:zoom_count], zoom_window),
        label=f"{zoom_window}-episode moving average", color="tab:orange",
    )
    axis.set(
        xlabel="Training episode", ylabel="Total reward",
        title=f"Taxi: early learning (first {zoom_count:,} episodes)",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "reward_curve_early.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    axes[0].plot(averaged_episodes, 100 * moving_average(successes, window))
    axes[0].set(ylabel="Success rate (%)", ylim=(0, 100))
    axes[1].plot(averaged_episodes, moving_average(steps, window))
    axes[1].set(ylabel="Mean steps")
    axes[2].plot(episodes, epsilons)
    axes[2].set(ylabel="Epsilon", ylim=(0, 1))
    if has_illegal_counts:
        for key, label in zip(illegal_keys, ("Illegal pickups", "Illegal drop-offs", "Total")):
            axes[3].plot(averaged_episodes, moving_average(result[key], window), label=label)
        axes[3].set_ylim(bottom=0)
        axes[3].legend()
    else:
        axes[3].text(
            0.5, 0.5, "Illegal-action counts were not recorded for this run.",
            ha="center", va="center", transform=axes[3].transAxes,
        )
    axes[3].set(xlabel="Training episode", ylabel="Mean illegal attempts")
    for axis in axes:
        axis.grid(alpha=0.25)
    figure.suptitle(f"Taxi: training metrics ({window}-episode averages; epsilon shown directly)")
    figure.tight_layout()
    figure.savefig(output_dir / "training_metrics.png", dpi=180)
    plt.close(figure)
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Saved training .npz file")
    parser.add_argument("--output-dir", type=Path, help="Default: model path without .npz")
    parser.add_argument("--window", type=int, default=100, help="Moving-average window (default: 100)")
    parser.add_argument("--zoom-episodes", type=int, default=2000, help="Early-learning plot episode count (default: 2000)")
    parser.add_argument("--zoom-window", type=int, default=25, help="Early-learning moving-average window (default: 25)")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    if args.zoom_episodes < 1 or args.zoom_window < 1:
        parser.error("--zoom-episodes and --zoom-window must be positive")
    try:
        with np.load(args.model, allow_pickle=False) as data:
            config = json.loads(data["config"].item())
            output_dir = export_training(
                data, config, args.output_dir or args.model.with_suffix(""), args.window,
                args.zoom_episodes, args.zoom_window,
            )
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Saved training plots and tables to {output_dir}")


if __name__ == "__main__":
    main()
