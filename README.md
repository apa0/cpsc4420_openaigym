# Taxi Q-learning

CPSC 4420 Assignment 2: tabular Q-learning for Gymnasium Taxi. The training code is in [`src/taxi_qlearning.py`](src/taxi_qlearning.py); supporting commands are in [`src/tools/`](src/tools/).

## Setup

Run from the repository root with Python 3.11:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

Train the baseline and the selected agent:

```bash
python -m src.taxi_qlearning --output results/baseline.npz
python -m src.taxi_qlearning --alpha 0.9 --epsilon-decay 0.99 --output results/combined_candidate.npz
```

Each training run saves a `.npz` archive and plots and tables in a folder with the same name. These commands replace existing outputs, so choose another `--output` path to preserve the included results. To evaluate the selected agent, run:

```bash
python -m src.tools.evaluate_agent results/combined_candidate.npz --baseline results/baseline.npz
```

The saved results used in the paper, including hyperparameter comparisons and rain experiments, are indexed in [`results/README.md`](results/README.md). Run any supporting command with `--help` for its options.

## Final agent video

[Watch the trained agent (MP4)](results/combined_candidate/videos/taxi-episode-0.mp4)

For recordings of this same agent with movement noise, see the [rainy videos folder](results/combined_candidate/videos/rain_0.25/).
