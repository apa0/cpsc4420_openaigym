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
