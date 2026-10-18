# cpsc4420_openaigym

CPSC 4420 — Assignment 2: implement a tabular Q-learning agent for Gymnasium's Taxi environment and compare how hyperparameters affect learning.

## Environment: Taxi

The taxi must navigate a 5 × 5 grid, pick up a passenger, and deliver them to their destination. Four designated locations serve as pickup and destination points.

- **Observations:** 500 discrete states, encoding 25 taxi positions × 5 passenger locations (including inside the taxi) × 4 destinations. There are 400 states reachable during an episode, plus 4 terminal observations after successful delivery, for 404 reachable states total.
- **Actions:** south (0), north (1), east (2), west (3), pickup (4), and drop-off (5).
- **Transitions:** deterministic for a given state and action. Movement follows the grid's walls; blocked moves leave the taxi in place. Initial states are sampled randomly.
- **Rewards:** +20 for successful delivery, −10 for illegal pickup or drop-off, and −1 otherwise.
- **Episode ending:** delivery to the correct destination terminates the episode; reaching 200 steps truncates it.
- **Discretization:** none required; observations directly index a 500 × 6 Q-table.

## Planned environment setup

Use the standard map and time limit, with rain and passenger destination changes disabled:

```python
import gymnasium as gym

env = gym.make("Taxi-v4", is_rainy=False, fickle_passenger=False)
```

Reference: [Gymnasium Taxi documentation](https://gymnasium.farama.org/environments/toy_text/taxi/).

## Python setup

Use Python 3.11. From the repository directory:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Dependencies include Gymnasium and Pygame for Taxi and rendering, NumPy for the Q-table, Matplotlib for experiment plots, and MoviePy with an FFmpeg backend for MP4 recording through Gymnasium's `RecordVideo` wrapper.

Activate the environment with `source .venv/bin/activate` in each new terminal. In your editor, select `.venv/bin/python` as the Python interpreter. Run the script with:

```bash
python src/taxi_qlearning.py
```

## Part B: Q-learning

`src/taxi_qlearning.py` initializes a 500 × 6 Q-table to zero and trains with an epsilon-greedy policy. Exploration samples all six actions; exploitation randomly breaks ties between actions with the highest Q-value. Action masking is not used.

Each transition updates the selected entry using:

```text
Q(s, a) += alpha * (reward + gamma * max(Q(next_state)) - Q(s, a))
```

For successful delivery, the target is just the reward. Both termination and time-limit truncation end the episode, but truncation still includes the next state's estimated value, following [Gymnasium's time-limit guidance](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

Change parameters from the command line:

```bash
python src/taxi_qlearning.py --alpha 0.1 --gamma 0.99 --epsilon 1.0 --epsilon-decay 0.995 --epsilon-min 0.01 --episodes 10000 --seed 42 --output results/baseline.npz
```

The selected baseline uses alpha 0.1, gamma 0.99, 10,000 episodes, and seed 42. Its exploration schedule follows the class Q-learning example: epsilon starts at 1.0 and is multiplied by 0.995 after each episode, down to a minimum of 0.01. The agent initially takes random actions on every step, then gradually chooses highest-valued actions more often. After 919 completed episodes, epsilon reaches its 1% floor. These settings are the script's defaults; Part C should report them and explain the exported reward plot. Set `--epsilon-decay 1.0` to use a fixed epsilon. Use `python src/taxi_qlearning.py --help` for all options.

The output archive contains `q_table`, per-episode `rewards`, `steps`, `successes`, `epsilons`, `illegal_pickups`, `illegal_dropoffs`, `illegal_actions`, and the run's configuration as a JSON string. Use distinct output filenames to preserve experiments. The printed training reward includes exploratory actions; it is not a separate policy evaluation.

## Training plots and tables

Training now automatically exports plots and tables beside the archive. For example, `--output results/baseline.npz` creates a `results/baseline/` folder containing:

| File | What it shows |
| --- | --- |
| `reward_curve.png` | Reward per training episode and a 100-episode moving average. |
| `reward_curve_early.png` | First 2,000 training episodes with a 25-episode moving average, showing early learning in more detail. |
| `training_metrics.png` | Moving-average success rate, steps, and illegal attempts, plus epsilon over training. |
| `episodes.csv` | Every episode's number, reward, steps, success (1 or 0), epsilon, illegal pickups, illegal drop-offs, and total illegal attempts. |
| `summary.csv` | Mean reward, mean steps, success rate, mean illegal pickups/drop-offs/total attempts, and total illegal attempts for the whole run, first 100 episodes, and last 100 episodes. |
| `q_table.csv` | All 500 state rows with labeled action columns. |
| `config.json` | The training configuration used for this run. |

Open PNG files in an image viewer and CSV files in a spreadsheet application or text editor. In `summary.csv`, success rate is a fraction from 0 to 1; the plot displays it as a percentage. A success means delivery completed before the time limit. All these metrics include training exploration. The Q-table also includes rows for unreachable states, which may remain zero. For runs shorter than 100 episodes, the averaging window uses the available episode count.

Illegal attempts are counted during each episode: after `env.step(action)`, a reward of -10 with action 4 increments illegal pickups, and a reward of -10 with action 5 increments illegal drop-offs. Their sum is `illegal_actions`. These counts describe penalized pickup/drop-off attempts, not blocked movement into walls. A permitted drop-off at another designated location is not counted unless Taxi assigns the -10 penalty. The counts help explain reward penalties independently of delivery success and episode length. Residual exploration can still cause illegal attempts late in training.

Archives saved before these counters were added still export successfully, but their illegal-attempt CSV cells are blank and the plot says the metric was not recorded. Counts are never filled in using the aggregate reward formula. Rerun training to collect them directly.

Export an existing archive, or change the averaging window, without retraining:

```bash
python src/analyze_training.py results/baseline.npz --window 100
```

Customize the early-learning view with `--zoom-episodes` and `--zoom-window`:

```bash
python src/analyze_training.py results/baseline.npz --zoom-episodes 2000 --zoom-window 25
```

Both reward plots include the raw episode rewards. The larger averaging window makes the overall trend easier to see; the smaller window shows more local changes in early training. Shorter runs use the available episode count, so the zoom and its averaging window never exceed the saved history. These options only change the exported view and do not alter training or its saved results.

The existing run saved while working from `src/` can be exported from the repository root with:

```bash
python src/analyze_training.py src/results/training.npz
```

Paths are relative to the directory where you run the command. By default, the export folder is the archive path with `.npz` removed. Use `--output-dir` for another location. Re-exporting replaces the generated tables and plots in that folder.

## Part D: learning-rate comparison

Compare alpha 0.1 (the existing baseline), 0.5, and 0.9 while holding gamma 0.99, the exploration schedule (1.0, decay 0.995, minimum 0.01), 10,000 episodes, and seed 42 fixed. Each new run initializes its own Q-table to zero. From the repository root:

```bash
python src/taxi_qlearning.py --alpha 0.5 --output results/alpha_0.5.npz
python src/taxi_qlearning.py --alpha 0.9 --output results/alpha_0.9.npz
python src/compare_alpha.py results/baseline.npz results/alpha_0.5.npz results/alpha_0.9.npz
```

The comparison command reads saved runs without retraining. It checks that all saved configuration settings except alpha match and that each alpha is distinct. It writes these files to `results/alpha_comparison/`:

| File | Use in the report |
| --- | --- |
| `reward_comparison.png` | Compare the overall training trend using overlaid 100-episode reward averages. |
| `reward_comparison_early.png` | Discuss approximate learning speed from the first 2,000 episodes, using 25-episode reward averages. |
| `metrics_comparison_early.png` | Compare early delivery success, episode length, and penalized pickup/drop-off attempts. |
| `stability_comparison.png` | Compare raw reward traces over the last 1,000 episodes on matching scales, plus 100-episode rolling reward standard deviation. |
| `final_performance.csv` | Compare mean reward, reward standard deviation, success rate, steps, and illegal attempts over the same final 100 episodes. |
| `episode_blocks.csv` | Inspect these statistics for successive 100-episode blocks throughout training. |
| `comparison_config.json` | Identify source archives, saved configurations, plot windows, and the standard deviation definition. |

Use `--output-dir`, `--window`, `--zoom-episodes`, `--zoom-window`, `--final-window`, and `--stability-episodes` to adjust the exported views. These affect analysis only. Windows are capped at the available history length. Standard deviations use `ddof=0` and describe observed episode rewards; they are not confidence intervals across repeated training runs. For a different comparison, preserve the original exports by choosing another output directory.

The completed seed-42 experiment produced these final-100 training results:

| Alpha | Mean reward | Reward standard deviation | Success rate | Mean steps | Mean illegal attempts |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.1 | 7.09 | 3.57 | 100% | 13.64 | 0.03 |
| 0.5 | 7.32 | 3.80 | 100% | 13.23 | 0.05 |
| 0.9 | 7.76 | 2.92 | 100% | 13.06 | 0.02 |

In the early reward comparison, alpha 0.5 and 0.9 improve more quickly than alpha 0.1, with 0.9 generally ahead of 0.5 during the initial improvement. The two larger learning rates approach their late reward range roughly around episodes 700–1,000, while alpha 0.1 continues improving toward that range through roughly episodes 1,500–2,000. These are approximate visual descriptions, not measured convergence times. A larger alpha gives each new transition more influence on its Q-value; in this deterministic setting that can help useful estimates propagate sooner.

The late rolling standard-deviation curves overlap substantially and change their ordering. Alpha 0.9 has the highest mean reward and lowest reward standard deviation in the final 100 episodes of this run, but the plots do not show a consistent variability advantage throughout the last 1,000 episodes. Rewards still vary with starting conditions and the remaining 1% exploration. Small final-score differences from one seed do not establish a general ranking of the learning rates. These comparisons describe training with exploration enabled; Part G requires a separate evaluation with exploration disabled.

## Record the trained agent

Load the saved Q-table and record three complete episodes as MP4 files:

```bash
python src/record_agent.py results/baseline.npz --episodes 3
```

For the existing archive:

```bash
python src/record_agent.py src/results/training.npz --episodes 3
```

Videos are saved in the archive's export folder under `videos/`, with one file per episode: `taxi-episode-0.mp4`, `taxi-episode-1.mp4`, and so on. The accompanying `episodes.csv` reports each video's reward, steps, success, and time-limit truncation; `recording_config.json` records the model, training settings, recording seed, and playback speed.

New recordings also count illegal pickups, illegal drop-offs, and their total directly during playback, using the same -10 penalty checks as training.

The policy uses epsilon 0, chooses randomly among tied best actions, and never updates the Q-table. The default recording seed is 43. Recordings use the same deterministic Taxi settings as training. The default playback speed is 6 frames per second; use `--fps 10` to play faster. No desktop window is needed. Existing MP4 files are protected from replacement; choose a different `--output-dir` for another recording.

Three demo episodes illustrate behavior. Part G still requires evaluation over at least 30 episodes with exploration disabled, reporting mean reward and additional metrics and comparing the selected agent with the baseline.
