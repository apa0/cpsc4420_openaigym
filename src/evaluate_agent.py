"""Evaluate one selected Taxi policy and compare it with the baseline, with epsilon zero."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from taxi_qlearning import select_action


def evaluate_policy(model, episodes=100, seed=10000):
    """Reset with seed + episode for every policy, and keep its Q-table read-only."""
    if episodes < 1 or seed < 0:
        raise ValueError("Episodes must be positive and seed must be nonnegative.")
    model = Path(model)
    with np.load(model, allow_pickle=False) as data:
        q_table = data["q_table"].copy()
        training_config = json.loads(data["config"].item())
    if training_config.get("environment") != "Taxi-v4":
        raise ValueError(f"{model}: expected a Taxi-v4 policy.")
    if q_table.shape != (500, 6) or not np.isfinite(q_table).all():
        raise ValueError(f"{model}: expected a finite 500 by 6 Q-table.")
    q_table.setflags(write=False)
    records = []
    env = gym.make("Taxi-v4", is_rainy=False, fickle_passenger=False)
    try:
        for episode in range(episodes):
            episode_seed = seed + episode
            state, _ = env.reset(seed=episode_seed)
            initial_state = int(state)
            rng = np.random.default_rng(episode_seed)
            total_reward = 0
            steps = 0
            illegal_pickups = 0
            illegal_dropoffs = 0
            while True:
                action = select_action(q_table, state, epsilon=0.0, rng=rng)
                state, reward, terminated, truncated, _ = env.step(action)
                if reward == -10:
                    if action == 4:
                        illegal_pickups += 1
                    elif action == 5:
                        illegal_dropoffs += 1
                total_reward += reward
                steps += 1
                if terminated or truncated:
                    records.append({
                        "agent": model.stem, "episode": episode + 1,
                        "episode_seed": episode_seed, "initial_state": initial_state,
                        "reward": total_reward, "steps": steps,
                        "success": int(terminated), "truncated": int(truncated),
                        "illegal_pickups": illegal_pickups, "illegal_dropoffs": illegal_dropoffs,
                        "illegal_actions": illegal_pickups + illegal_dropoffs,
                    })
                    break
    finally:
        env.close()
    return training_config, records


def write_csv(path, rows):
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare_with_baseline(candidate, baseline, output_dir, episodes=100, seed=10000):
    """Compare only the selected policy and baseline using paired evaluation episodes."""
    candidate = Path(candidate)
    baseline = Path(baseline)
    if candidate.resolve() == baseline.resolve() or candidate.stem == baseline.stem:
        raise ValueError("The selected policy and baseline must have distinct filenames.")
    models = [baseline, candidate]
    all_records = []
    summaries = []
    provenance = []
    by_agent = {}
    for model in models:
        config, records = evaluate_policy(model, episodes, seed)
        by_agent[model.stem] = records
        all_records.extend(records)

        def values(key):
            return np.array([record[key] for record in records])

        summaries.append({
            "agent": model.stem, "alpha": config["alpha"], "gamma": config["gamma"],
            "epsilon_decay": config["epsilon_decay"], "evaluation_episodes": episodes,
            "mean_reward": values("reward").mean(), "reward_std": values("reward").std(ddof=0),
            "success_rate": values("success").mean(), "mean_steps": values("steps").mean(),
            "mean_illegal_pickups": values("illegal_pickups").mean(),
            "mean_illegal_dropoffs": values("illegal_dropoffs").mean(),
            "mean_illegal_actions": values("illegal_actions").mean(),
            "total_illegal_actions": int(values("illegal_actions").sum()),
            "truncations": int(values("truncated").sum()),
        })
        provenance.append({
            "agent": model.stem, "model": str(model),
            "sha256": hashlib.sha256(model.read_bytes()).hexdigest(), "training_config": config,
        })

    baseline_records = by_agent[baseline.stem]
    paired_rows = []
    for model in models:
        records = by_agent[model.stem]
        if [r["initial_state"] for r in records] != [r["initial_state"] for r in baseline_records]:
            raise ValueError("Policies were not evaluated on matching initial states.")
        reward_differences = np.array([r["reward"] - b["reward"] for r, b in zip(records, baseline_records)])
        step_differences = np.array([r["steps"] - b["steps"] for r, b in zip(records, baseline_records)])
        paired_rows.append({
            "agent": model.stem, "baseline": baseline.stem,
            "mean_reward_difference": reward_differences.mean(),
            "mean_steps_difference": step_differences.mean(),
            "higher_reward_episodes": int((reward_differences > 0).sum()),
            "equal_reward_episodes": int((reward_differences == 0).sum()),
            "lower_reward_episodes": int((reward_differences < 0).sum()),
        })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "summary.csv", summaries)
    write_csv(output_dir / "episodes.csv", all_records)
    write_csv(output_dir / "baseline_comparison.csv", paired_rows)
    (output_dir / "evaluation_config.json").write_text(json.dumps({
        "models": provenance, "environment": "Taxi-v4",
        "environment_settings": {"is_rainy": False, "fickle_passenger": False},
        "episodes_per_agent": episodes, "first_episode_seed": seed,
        "episode_seed_rule": "seed + zero-based episode index; reset independently every episode",
        "tie_breaking": "Random choice among highest-valued actions; RNG reset to episode seed",
        "epsilon": 0.0, "q_updates": False, "reward_std_ddof": 0,
        "baseline": baseline.stem,
        "selected_agent": candidate.stem,
    }, indent=2) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(10, 7))
    names = ["Baseline", "Selected candidate"]
    positions = np.arange(len(names))
    colors = ["tab:blue", "tab:orange"]
    for axis, key, title in zip(
        axes.flat, ("mean_reward", "success_rate", "mean_steps", "mean_illegal_actions"),
        ("Mean reward", "Success rate (%)", "Mean steps", "Mean illegal attempts"),
    ):
        heights = [row[key] * (100 if key == "success_rate" else 1) for row in summaries]
        bars = axis.bar(positions, heights, color=colors)
        axis.set_xticks(positions, names)
        axis.set(title=title)
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
        if min(heights) >= 0:
            axis.set_ylim(0, max(max(heights) * 1.2, 1))
        axis.bar_label(bars, labels=[f"{height:.2f}" for height in heights], padding=3, fontsize=8)
    figure.suptitle(f"Taxi: selected policy vs baseline ({episodes} paired episodes; epsilon = 0)")
    figure.tight_layout()
    figure.savefig(output_dir / "evaluation_comparison.png", dpi=180)
    plt.close(figure)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Saved Q-table for the selected candidate")
    parser.add_argument("--baseline", type=Path, default=Path("results/baseline.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10000)
    args = parser.parse_args()
    if args.episodes < 30:
        parser.error("Part G requires at least 30 evaluation episodes per agent")
    try:
        summaries = compare_with_baseline(
            args.model, args.baseline, args.output_dir, args.episodes, args.seed,
        )
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    for row in summaries:
        print(f"{row['agent']}: reward={row['mean_reward']:.2f}, "
              f"success={100 * row['success_rate']:.1f}%, steps={row['mean_steps']:.2f}, "
              f"illegal attempts={row['total_illegal_actions']}")
    print(f"Saved evaluation plots and tables to {args.output_dir}")


if __name__ == "__main__":
    main()
