# cpsc4420_openaigym

CPSC 4420 Assignment 2: tabular Q-learning for Gymnasium Taxi, with training exports, learning-rate comparisons, and MP4 recording.

## Setup

Use Python 3.11. Run these commands from the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Activate `.venv` in each new terminal and select `.venv/bin/python` as your editor's interpreter. Dependencies are listed in `requirements.txt`.

All commands below run from the repository root. Output paths are relative to the directory where the command is run.

## File locations

| Location | Contents |
| --- | --- |
| `src/taxi_qlearning.py` | Q-learning training script. |
| `src/analyze_training.py` | Export plots and CSV tables from a saved training archive. |
| `src/compare_alpha.py` | Compare saved runs with different learning rates. |
| `src/record_agent.py` | Record a saved policy as MP4 videos. |
| `requirements.txt` | Python dependencies. |
| `results/baseline.npz` | Saved baseline Q-table, episode histories, and configuration. |
| `results/baseline/` | Baseline plots, tables, and configuration export. |
| `results/alpha_0.5.npz`, `results/alpha_0.9.npz` | Saved learning-rate experiment runs. |
| `results/alpha_0.5/`, `results/alpha_0.9/` | Exports for those runs. |
| `results/alpha_comparison/` | Comparison plots, tables, and source configurations. |
| `src/results/` | Earlier outputs created when commands were run from `src/`. |

## Train the baseline

```bash
python src/taxi_qlearning.py --alpha 0.1 --gamma 0.99 --epsilon 1.0 --epsilon-decay 0.995 --epsilon-min 0.01 --episodes 10000 --seed 42 --output results/baseline.npz
```

Training uses `Taxi-v4` with `is_rainy=False` and `fickle_passenger=False`. The parameters above are the defaults. Running `python src/taxi_qlearning.py` uses the default output path `results/training.npz`.

Use `--epsilon-decay 1.0` for a fixed epsilon. Use `--help` to list all training options. Choose distinct `--output` paths to preserve separate runs; an existing archive at the selected path is overwritten.

Training saves the `.npz` archive and automatically exports plots and tables to a folder with the same name without `.npz`. The archive contains `q_table`, `rewards`, `steps`, `successes`, `epsilons`, `illegal_pickups`, `illegal_dropoffs`, `illegal_actions`, and `config`.

## Training exports

For `results/baseline.npz`, these files are in `results/baseline/`:

| File | Contents |
| --- | --- |
| `reward_curve.png` | Episode rewards and a 100-episode moving average. |
| `reward_curve_early.png` | First 2,000 episode rewards and a 25-episode moving average. |
| `training_metrics.png` | Success rate, steps, epsilon, and illegal pickup/drop-off counts. |
| `episodes.csv` | Per-episode reward, steps, success, epsilon, and illegal attempt counts. |
| `summary.csv` | Aggregate statistics for all episodes and the first and last 100 episodes. |
| `q_table.csv` | State rows with action columns: south, north, east, west, pickup, and drop-off. |
| `config.json` | Saved training parameters. |

Open PNG files with an image viewer and CSV files with a spreadsheet application or text editor. CSV success rates are fractions from 0 to 1. Illegal attempts count pickup/drop-off actions receiving a -10 penalty. Training metrics include exploration.

Regenerate exports from an existing archive without retraining:

```bash
python src/analyze_training.py results/baseline.npz --window 100 --zoom-episodes 2000 --zoom-window 25
```

Use `--output-dir` to choose another export folder. Re-exporting replaces generated tables and plots in that folder. Windows are capped at the available episode count. Older archives without illegal-attempt counters export blank cells for those metrics.

To export the earlier archive under `src/results/`:

```bash
python src/analyze_training.py src/results/training.npz
```

## Run the alpha experiments

After saving the baseline, train the additional learning rates and generate their comparison:

```bash
python src/taxi_qlearning.py --alpha 0.5 --output results/alpha_0.5.npz
python src/taxi_qlearning.py --alpha 0.9 --output results/alpha_0.9.npz
python src/compare_alpha.py results/baseline.npz results/alpha_0.5.npz results/alpha_0.9.npz
```

The comparison script requires distinct alpha values and matching saved settings for every other parameter, including the episode count and seed. It reads saved archives without retraining and writes to `results/alpha_comparison/` by default.

| File | Contents |
| --- | --- |
| `reward_comparison.png` | Overlaid 100-episode mean rewards for the full runs. |
| `reward_comparison_early.png` | Overlaid 25-episode mean rewards for the first 2,000 episodes. |
| `metrics_comparison_early.png` | Early success rates, mean steps, and mean illegal attempts. |
| `stability_comparison.png` | Raw rewards for the last 1,000 episodes and 100-episode rolling reward standard deviation. |
| `final_performance.csv` | Statistics for each run's final 100 training episodes. |
| `episode_blocks.csv` | Statistics for successive 100-episode blocks. |
| `comparison_config.json` | Source archives, training configurations, and comparison settings. |

Use `--output-dir`, `--window`, `--zoom-episodes`, `--zoom-window`, `--final-window`, and `--stability-episodes` to customize the exports. Reward standard deviations use `ddof=0`.

## Record an MP4 demo

```bash
python src/record_agent.py results/baseline.npz --episodes 3
```

By default, this creates `results/baseline/videos/` with one MP4 per episode, named `taxi-episode-0.mp4`, `taxi-episode-1.mp4`, and so on. The folder also contains `episodes.csv` with recording metrics and `recording_config.json` with the source model and settings.

Recording loads the saved Q-table with exploration disabled and does not update it. Defaults are seed 43 and 6 frames per second. No desktop window is needed. Use `--seed`, `--fps`, and `--output-dir` to change the recording settings:

```bash
python src/record_agent.py results/baseline.npz --episodes 3 --seed 43 --fps 10 --output-dir results/baseline/videos_alt
```

Existing MP4 files are protected from replacement; choose another output directory for a new recording. Training and analysis commands do not create videos.
