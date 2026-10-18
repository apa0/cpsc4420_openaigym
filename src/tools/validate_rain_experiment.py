"""Check rain results across additional training seeds and a fresh evaluation sample."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.tools.evaluate_agent import evaluate_policy, write_csv
from src.tools.run_rain_experiment import summarize_evaluation
from src.taxi_qlearning import train


def validate_rain(experiment_dir, output_dir, additional_seeds=(43, 44), episodes=500, seed=20000):
    if episodes < 30 or seed < 0 or not additional_seeds or any(s < 0 for s in additional_seeds):
        raise ValueError("Use at least 30 evaluation episodes and nonnegative seeds.")
    metadata = json.loads((Path(experiment_dir) / "experiment_config.json").read_text())
    settings = metadata["training_settings"]
    original_seed = settings["seed"]
    training_seeds = [original_seed, *additional_seeds]
    if len(set(training_seeds)) != len(training_seeds):
        raise ValueError("Additional training seeds must be distinct and differ from the original seed.")
    original = Path(metadata["original_policy"])
    original_hash = hashlib.sha256(original.read_bytes()).hexdigest()
    levels = metadata["noise_levels"]
    existing = {row["rain_noise"]: row for row in metadata["models"]}
    for row in existing.values():
        if hashlib.sha256(Path(row["model"]).read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("An original experiment archive has changed; rerun the original experiment first.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluations = []
    episode_rows = []
    models = []

    def collect(model, noise, role, training_seed, records):
        row = summarize_evaluation(records, noise, role, model, noise if role == "trained_at_noise_level" else 0.0)
        row["training_seed"] = training_seed
        evaluations.append(row)
        episode_rows.extend({"rain_noise": noise, "policy": role, "training_seed": training_seed, **r} for r in records)
        print(f"  {role}, seed={training_seed}: reward={row['mean_reward']:.2f}, "
              f"success={100 * row['success_rate']:.1f}%, steps={row['mean_steps']:.2f}", flush=True)

    for noise in levels:
        print(f"Checking movement noise {noise:.0%} on {episodes} fresh evaluation episodes...", flush=True)
        _, reference = evaluate_policy(original, episodes, seed, rain_noise=noise)
        collect(original, noise, "original_dry_policy", original_seed, reference)
        if noise == 0.0:
            collect(original, noise, "trained_at_noise_level", original_seed, reference)
            continue
        for training_seed in training_seeds:
            if training_seed == original_seed:
                model = Path(existing[noise]["model"])
                config = existing[noise]["training_config"]
            else:
                model = output_dir / f"seed_{training_seed}" / f"rain_{noise}.npz"
                model.parent.mkdir(parents=True, exist_ok=True)
                print(f"  Training fresh Q-table: seed={training_seed}, noise={noise:.0%}...", flush=True)
                parameters = {**settings, "seed": training_seed, "rain_noise": noise}
                history = train(**parameters)
                config = {"environment": "Taxi-v4", **parameters}
                np.savez_compressed(model, **history, config=json.dumps(config))
            models.append({
                "model": str(model), "training_seed": training_seed,
                "rain_noise": noise, "training_config": config,
                "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
            })
            _, records = evaluate_policy(model, episodes, seed, rain_noise=noise)
            if [r["initial_state"] for r in records] != [r["initial_state"] for r in reference]:
                raise ValueError("Policies did not receive matching initial states.")
            collect(model, noise, "trained_at_noise_level", training_seed, records)

    aggregate = []
    for noise in levels:
        for role in ("trained_at_noise_level", "original_dry_policy"):
            rows = [r for r in evaluations if r["rain_noise"] == noise and r["policy"] == role]
            rewards = np.array([r["mean_reward"] for r in rows])
            successes = np.array([r["success_rate"] for r in rows])
            aggregate.append({
                "rain_noise": noise, "policy": role, "training_runs": len(rows),
                "evaluation_episodes_per_run": episodes,
                "mean_reward_across_runs": rewards.mean(),
                "std_reward_across_runs": rewards.std(ddof=1) if len(rows) > 1 else 0.0,
                "min_mean_reward": rewards.min(), "max_mean_reward": rewards.max(),
                "mean_success_rate_across_runs": successes.mean(),
                "std_success_rate_across_runs": successes.std(ddof=1) if len(rows) > 1 else 0.0,
                "min_success_rate": successes.min(), "max_success_rate": successes.max(),
                "mean_steps_across_runs": np.mean([r["mean_steps"] for r in rows]),
                "mean_illegal_actions_across_runs": np.mean([r["mean_illegal_actions"] for r in rows]),
            })
    write_csv(output_dir / "per_run_evaluation.csv", evaluations)
    write_csv(output_dir / "aggregate_evaluation.csv", aggregate)
    write_csv(output_dir / "evaluation_episodes.csv", episode_rows)
    (output_dir / "validation_config.json").write_text(json.dumps({
        "source_experiment": str(experiment_dir), "original_policy": str(original),
        "original_policy_sha256": original_hash, "training_seeds": training_seeds,
        "original_dry_policy_is_fixed": True, "new_training_only_at_nonzero_noise": True,
        "models": models, "evaluation_episodes_per_model_per_noise": episodes,
        "evaluation_first_seed": seed, "evaluation_epsilon": 0.0, "evaluation_q_updates": False,
        "paired_evaluation": "Identical episode seed sequence for every model and noise level",
        "between_run_std_ddof": 1,
        "error_bars": "Standard deviation across noisy training seeds, not confidence intervals",
    }, indent=2) + "\n")
    plot_validation(aggregate, output_dir, episodes)
    return aggregate


def plot_validation(rows, output_dir, episodes):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for role, label, style in (
        ("trained_at_noise_level", "Trained with rain (3 seeds at nonzero noise)", "o-"),
        ("original_dry_policy", "Fixed original dry policy", "s--"),
    ):
        selected = [r for r in rows if r["policy"] == role]
        x = [100 * r["rain_noise"] for r in selected]
        axes[0].errorbar(x, [r["mean_reward_across_runs"] for r in selected],
                         yerr=[r["std_reward_across_runs"] for r in selected],
                         fmt=style, label=label, capsize=4)
        axes[1].errorbar(x, [100 * r["mean_success_rate_across_runs"] for r in selected],
                         yerr=[100 * r["std_success_rate_across_runs"] for r in selected],
                         fmt=style, label=label, capsize=4)
    axes[0].set(ylabel="Mean evaluation reward")
    axes[1].set(ylabel="Success rate (%)", ylim=(0, 105))
    for axis in axes:
        axis.set(xlabel="Movement noise (%)", xticks=x)
        axis.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle(f"Taxi: rain validation on {episodes} fresh paired episodes per model\nError bars: standard deviation across training seeds; epsilon = 0")
    figure.tight_layout()
    figure.savefig(output_dir / "validation_comparison.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=Path("results/rain_experiment"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/rain_validation"))
    parser.add_argument("--additional-training-seeds", nargs="+", type=int, default=[43, 44])
    parser.add_argument("--evaluation-episodes", type=int, default=500)
    parser.add_argument("--evaluation-seed", type=int, default=20000)
    args = parser.parse_args()
    try:
        validate_rain(args.experiment_dir, args.output_dir, args.additional_training_seeds,
                      args.evaluation_episodes, args.evaluation_seed)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(f"Saved repeatability check to {args.output_dir}")


if __name__ == "__main__":
    main()
