# Assignment results

All commands run from the repository root. The `.npz` files contain saved Q-tables and training histories; the adjacent folders contain plots, CSV tables, and settings.

| Assignment topic | Results to open | Command that generated them |
| --- | --- | --- |
| Baseline training | [Reward curve](baseline/reward_curve.png), [training metrics](baseline/training_metrics.png), [summary](baseline/summary.csv) | `python -m src.taxi_qlearning --output results/baseline.npz` |
| Learning rate (alpha) | [Comparison plots and tables](alpha_comparison/) | `python -m src.tools.compare_training alpha results/baseline.npz results/alpha_0.5.npz results/alpha_0.9.npz` |
| Discount factor (gamma) | [Comparison plots and tables](gamma_comparison/) | `python -m src.tools.compare_training gamma results/gamma_0.50.npz results/gamma_0.90.npz results/baseline.npz` |
| Exploration decay | [Comparison plots and tables](epsilon_decay_comparison/) | `python -m src.tools.compare_training epsilon_decay results/epsilon_decay_0.99.npz results/baseline.npz results/epsilon_decay_0.999.npz` |
| Selected configuration (Part G) | [Training curve](combined_candidate/reward_curve.png), [training summary](combined_candidate/summary.csv), [saved model](combined_candidate.npz) | `python -m src.taxi_qlearning --alpha 0.9 --epsilon-decay 0.99 --output results/combined_candidate.npz` |
| Baseline versus selected agent (Part G) | [Evaluation plot](evaluation/evaluation_comparison.png), [evaluation summary](evaluation/summary.csv), [paired differences](evaluation/baseline_comparison.csv) | `python -m src.tools.evaluate_agent results/combined_candidate.npz --baseline results/baseline.npz` |
| Rain and robustness (Part H) | [Training plot](rain_experiment/training_reward_comparison.png), [evaluation plot](rain_experiment/evaluation_comparison.png), [evaluation summary](rain_experiment/evaluation_summary.csv) | `python -m src.tools.run_rain_experiment --model results/combined_candidate.npz --noise-levels 0 0.10 0.25` |
| Rain repeatability check | [Validation plot](rain_validation/validation_comparison.png), [aggregate evaluation](rain_validation/aggregate_evaluation.csv), [per-run evaluation](rain_validation/per_run_evaluation.csv) | `python -m src.tools.validate_rain_experiment --additional-training-seeds 43 44 --evaluation-episodes 500 --evaluation-seed 20000` |

The selected agent has [dry](combined_candidate/videos/taxi-episode-0.mp4) and [25% movement-noise](combined_candidate/videos/rain_0.25/taxi-episode-0.mp4) MP4 recordings. Both use the same saved Q-table with exploration disabled. Earlier exploratory output is kept in [`legacy/`](legacy/).
