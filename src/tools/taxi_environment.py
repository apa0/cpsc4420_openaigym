"""Build Taxi with a controllable probability of lateral movement in rain."""

import gymnasium as gym


def make_taxi(rain_noise=0.0, render_mode=None):
    """Noise applies to movement; pickup/drop-off and the passenger stay deterministic."""
    if not 0 <= rain_noise <= 1:
        raise ValueError("rain_noise must be in [0, 1].")
    return gym.make(
        "Taxi-v4", is_rainy=rain_noise > 0,
        rainy_probability=1.0 - rain_noise, fickle_passenger=False,
        render_mode=render_mode,
    )
