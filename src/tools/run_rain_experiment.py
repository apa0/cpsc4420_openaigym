"""Part H: train in controlled rain and test the original dry-weather policy's robustness."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.tools.analyze_training import export_training, moving_average
from src.tools.evaluate_agent import evaluate_policy, write_csv
from src.taxi_qlearning import train


def summarize_evaluation(records, noise, role, model, training_noise):
    def values(key):
        return np.array([row[key] for row in records])

    return {
        "rain_noise": noise, "policy": role, "model": str(model),
        "training_rain_noise": training_noise, "evaluation_episodes": len(records),
        "mean_reward": values("reward").mean(), "reward_std": values("reward").std(ddof=0),
        "success_rate": values("success").mean(), "mean_steps": values("steps").mean(),
        "mean_illegal_pickups": values("illegal_pickups").mean(),
        "mean_illegal_dropoffs": values("illegal_dropoffs").mean(),
        "mean_illegal_actions": values("illegal_actions").mean(),
        "total_illegal_actions": int(values("illegal_actions").sum()),
        "truncations": int(values("truncated").sum()),
    }


def run_experiment(original, output_dir, noise_levels=(0.0, 0.10, 0.25),
                   evaluation_episodes=100, evaluation_seed=10000, window=100, zoom_episodes=2000):
    """Reuse the original dry run; train new Q-tables at each nonzero noise level."""
    levels = sorted(noise_levels)
    if len(levels) < 3 or len(set(levels)) != len(levels) or 0.0 not in levels:
        raise ValueError("Provide at least three distinct noise levels, including zero.")
    if any(not 0 <= noise <= 1 for noise in levels):
        raise ValueError("Noise probabilities must be in [0, 1].")
    if evaluation_episodes < 30 or evaluation_seed < 0 or window < 1 or zoom_episodes < 1:
        raise ValueError("Use at least 30 evaluation episodes, a nonnegative seed, and positive plot windows.")
    original = Path(original)
    with np.load(original, allow_pickle=False) as archive:
        original_config = json.loads(archive["config"].item())
        original_history = {key: archive[key].copy() for key in archive.files if key != "config"}
    if original_config.get("environment") != "Taxi-v4" or original_config.get("rain_noise", 0.0) != 0.0:
        raise ValueError("The original policy must have been trained in dry Taxi-v4.")
    keys = ("alpha", "gamma", "epsilon", "episodes", "epsilon_decay", "epsilon_min", "seed")
    training_settings = {key: original_config[key] for key in keys}
    if len(original_history["rewards"]) != training_settings["episodes"]:
        raise ValueError("Original training history does not match its configured episode count.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    histories = {}
    models = []
    training_rows = []
    evaluation_rows = []
    episode_rows = []

    for noise in levels:
        if noise == 0.0:
            model = original
            history = original_history
            config = {**original_config, "rain_noise": 0.0}
            print(f"Reusing dry-weather training run: {original}", flush=True)
        else:
            model = output_dir / f"rain_{noise}.npz"
            print(f"Training {training_settings['episodes']} episodes with movement noise {noise:.0%}...", flush=True)
            history = train(**training_settings, rain_noise=noise)
            config = {"environment": "Taxi-v4", **training_settings, "rain_noise": noise}
            np.savez_compressed(model, **history, config=json.dumps(config))
            export_training(history, config, model.with_suffix(""))
        histories[noise] = history
        models.append({
            "rain_noise": noise, "model": str(model), "training_config": config,
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        })

        average_window = min(window, len(history["rewards"]))
        for period, section in (
            (f"first_{average_window}", slice(0, average_window)),
            (f"last_{average_window}", slice(-average_window, None)),
        ):
            training_rows.append({
                "rain_noise": noise, "period": period, "episodes": average_window,
                "mean_reward": history["rewards"][section].mean(),
                "reward_std": history["rewards"][section].std(ddof=0),
                "success_rate": history["successes"][section].mean(),
                "mean_steps": history["steps"][section].mean(),
                "mean_illegal_actions": history["illegal_actions"][section].mean(),
            })

        _, matched_records = evaluate_policy(model, evaluation_episodes, evaluation_seed, rain_noise=noise)
        if noise == 0.0:
            original_records = matched_records
        else:
            _, original_records = evaluate_policy(original, evaluation_episodes, evaluation_seed, rain_noise=noise)
        if [r["initial_state"] for r in matched_records] != [r["initial_state"] for r in original_records]:
            raise ValueError("Matched and original policies did not receive matching initial states.")
        for role, source, train_noise, records in (
            ("trained_at_noise_level", model, noise, matched_records),
            ("original_dry_policy", original, 0.0, original_records),
        ):
            row = summarize_evaluation(records, noise, role, source, train_noise)
            evaluation_rows.append(row)
            episode_rows.extend({"rain_noise": noise, "policy": role, **record} for record in records)
            print(f"  {role}: reward={row['mean_reward']:.2f}, success={100 * row['success_rate']:.1f}%, "
                  f"steps={row['mean_steps']:.2f}", flush=True)

    write_csv(output_dir / "training_summary.csv", training_rows)
    write_csv(output_dir / "evaluation_summary.csv", evaluation_rows)
    write_csv(output_dir / "evaluation_episodes.csv", episode_rows)
    (output_dir / "experiment_config.json").write_text(json.dumps({
        "original_policy": str(original), "models": models, "noise_levels": levels,
        "noise_definition": "Movement uses intended direction with probability 1-p, each lateral direction with p/2",
        "environment": "Taxi-v4", "fickle_passenger": False,
        "training_settings": training_settings,
        "evaluation_episodes_per_policy_per_noise": evaluation_episodes,
        "evaluation_first_seed": evaluation_seed,
        "evaluation_seed_rule": "Reset independently with evaluation_first_seed + episode index for each policy and noise level",
        "evaluation_epsilon": 0.0, "evaluation_q_updates": False,
        "reward_std_ddof": 0, "training_plot_window": window, "zoom_episodes": zoom_episodes,
    }, indent=2) + "\n")
    plot_results(histories, evaluation_rows, output_dir, window, zoom_episodes)
    return evaluation_rows


def plot_results(histories, evaluation_rows, output_dir, window, zoom_episodes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    count = len(next(iter(histories.values()))["rewards"])
    for filename, shown in (
        ("training_reward_comparison.png", count),
        ("training_reward_early.png", min(zoom_episodes, count)),
    ):
        average_window = min(window, shown)
        episodes = np.arange(average_window, shown + 1)
        figure, axis = plt.subplots(figsize=(10, 5))
        for noise, history in histories.items():
            axis.plot(episodes, moving_average(history["rewards"][:shown], average_window),
                      label=f"Movement noise = {noise:.0%}")
        axis.axhline(0, color="gray", linewidth=0.8)
        axis.set(xlabel="Training episode", ylabel="Mean total reward",
                 title=f"Taxi: training under rain (first {shown:,} episodes)\n{average_window}-episode moving average")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for role, label, style in (
        ("trained_at_noise_level", "Trained at evaluated noise level", "o-"),
        ("original_dry_policy", "Original no-noise policy", "s--"),
    ):
        rows = [row for row in evaluation_rows if row["policy"] == role]
        noise_percent = [100 * row["rain_noise"] for row in rows]
        for axis, key in zip(axes.flat, ("mean_reward", "success_rate", "mean_steps", "mean_illegal_actions")):
            values = [row[key] * (100 if key == "success_rate" else 1) for row in rows]
            axis.plot(noise_percent, values, style, label=label)
    for axis, title in zip(axes.flat, ("Mean reward", "Success rate (%)", "Mean steps", "Mean illegal attempts")):
        axis.set(xlabel="Movement noise (%)", ylabel=title)
        axis.set_xticks([100 * noise for noise in histories])
        axis.grid(alpha=0.25)
    axes[0, 1].set_ylim(0, 105)
    axes[1, 1].set_ylim(bottom=0)
    axes[0, 0].legend(fontsize=8)
    episodes = evaluation_rows[0]["evaluation_episodes"]
    figure.suptitle(f"Taxi: rain performance and original-policy robustness\n{episodes} paired evaluation episodes per condition; epsilon = 0")
    figure.tight_layout()
    figure.savefig(output_dir / "evaluation_comparison.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("results/combined_candidate.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/rain_experiment"))
    parser.add_argument("--noise-levels", type=float, nargs="+", default=[0.0, 0.10, 0.25])
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--evaluation-seed", type=int, default=10000)
    parser.add_argument("--window", type=int, default=100)
    parser.add_argument("--zoom-episodes", type=int, default=2000)
    args = parser.parse_args()
    try:
        run_experiment(args.model, args.output_dir, args.noise_levels, args.evaluation_episodes,
                       args.evaluation_seed, args.window, args.zoom_episodes)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Saved rain training and robustness comparisons to {args.output_dir}")


if __name__ == "__main__":
    main()
