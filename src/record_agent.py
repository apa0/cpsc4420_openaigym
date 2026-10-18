"""Record a saved Taxi Q-learning policy as MP4 videos, with exploration disabled."""

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np

from taxi_qlearning import select_action


def record_agent(model, output_dir, episodes=3, seed=43, fps=6):
    """Record complete greedy episodes without changing the learned Q-table."""
    if episodes < 1 or fps < 1 or seed < 0:
        raise ValueError("Episodes and fps must be positive; seed must be nonnegative.")
    with np.load(model, allow_pickle=False) as data:
        q_table = data["q_table"].copy()
        config = json.loads(data["config"].item())
    if config.get("environment") != "Taxi-v4":
        raise ValueError("This recording script expects a Taxi-v4 training run.")
    if q_table.shape != (500, 6) or not np.isfinite(q_table).all():
        raise ValueError("Expected a finite 500 by 6 Taxi Q-table.")

    output_dir = Path(output_dir)
    for episode in range(episodes):
        path = output_dir / f"taxi-episode-{episode}.mp4"
        if path.exists():
            raise FileExistsError(f"{path} already exists; choose another --output-dir.")
    # Match the deterministic environment used for training.
    env = gym.make(
        "Taxi-v4", is_rainy=False, fickle_passenger=False, render_mode="rgb_array",
    )
    try:
        env = RecordVideo(
            env, video_folder=str(output_dir), episode_trigger=lambda episode: True,
            name_prefix="taxi", fps=fps,
        )
        rng = np.random.default_rng(seed)
        records = []
        for episode in range(episodes):
            state, _ = env.reset(seed=seed if episode == 0 else None)
            total_reward = 0
            steps = 0
            illegal_pickups = 0
            illegal_dropoffs = 0
            while True:
                # Epsilon zero chooses a highest-valued action, with random ties.
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
                        "episode": episode + 1,
                        "video": f"taxi-episode-{episode}.mp4",
                        "reward": total_reward,
                        "steps": steps,
                        "success": int(terminated),
                        "truncated": int(truncated),
                        "illegal_pickups": illegal_pickups,
                        "illegal_dropoffs": illegal_dropoffs,
                        "illegal_actions": illegal_pickups + illegal_dropoffs,
                    })
                    break
    finally:
        # Flush the final episode's recording to disk.
        env.close()

    with (output_dir / "episodes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "recording_config.json").write_text(json.dumps({
        "model": str(Path(model).resolve()), "training_config": config,
        "episodes": episodes, "seed": seed, "fps": fps, "epsilon": 0.0,
    }, indent=2) + "\n")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Saved training .npz file")
    parser.add_argument("--output-dir", type=Path, help="Default: model path without .npz, then /videos")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes to record (default: 3)")
    parser.add_argument("--seed", type=int, default=43, help="Recording seed (default: 43)")
    parser.add_argument("--fps", type=int, default=6, help="Video playback frames per second (default: 6)")
    args = parser.parse_args()
    output_dir = args.output_dir or args.model.with_suffix("") / "videos"
    try:
        records = record_agent(args.model, output_dir, args.episodes, args.seed, args.fps)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    for record in records:
        print(f"Episode {record['episode']}: reward={record['reward']}, "
              f"steps={record['steps']}, success={bool(record['success'])}")
    print(f"Saved MP4 videos and episode metrics to {output_dir}")


if __name__ == "__main__":
    main()
