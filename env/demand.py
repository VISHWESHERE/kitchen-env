"""
Demand simulation for the Kitchen Environment.

Demand is generated in two stages:
  1. A base forecast is drawn from a realistic hourly range.
  2. A "chef skill" multiplier (Uniform(0.8, 1.2)) is applied to produce
     the actual demand. This makes perfect prediction impossible and forces
     the agent to hedge.

The module is intentionally stateless — the caller passes an RNG so that
full episode reproducibility is guaranteed by seeding the RNG once.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import numpy as np

from env.models import DISHES

# ---------------------------------------------------------------------------
# Hourly demand ranges (portions per dish)
# The restaurant operates two services: lunch (11–14) and dinner (17–20).
# Each hour has a realistic [low, high] range.
# ---------------------------------------------------------------------------

HOURLY_DEMAND_RANGE: Dict[int, Tuple[int, int]] = {
    11: (2, 4),    # pre-lunch  — quiet
    12: (8, 14),   # lunch rush — busy
    13: (10, 18),  # peak lunch — very busy
    14: (4, 8),    # post-lunch — winding down
    17: (3, 5),    # pre-dinner — quiet
    18: (9, 15),   # dinner rush — busy
    19: (12, 20),  # peak dinner — very busy
    20: (5, 10),   # late dinner — slowing
}

# Chef skill noise bounds — actual_demand = forecast × Uniform(lo, hi)
CHEF_SKILL_LO: float = 0.8
CHEF_SKILL_HI: float = 1.2

# Demand spike multiplier (used in Task 3 hard)
DEMAND_SPIKE_MULTIPLIER: float = 2.5


def generate_forecast(
    hour: int,
    rng: np.random.Generator,
    dishes: List[str] = DISHES,
) -> Dict[str, int]:
    """
    Generate a demand forecast for *hour* using the provided RNG.

    The forecast is the integer mid-point of the hourly range ± a small
    random offset so that the forecast is a useful-but-imperfect signal.
    The agent sees this value before acting.

    Args:
        hour:   The kitchen hour (must be a key in HOURLY_DEMAND_RANGE).
        rng:    A seeded numpy RNG — caller is responsible for seeding.
        dishes: List of dish names (defaults to global DISHES).

    Returns:
        Dict mapping dish name → forecast portions (int ≥ 0).
    """
    low, high = HOURLY_DEMAND_RANGE.get(hour, (3, 8))
    mid = (low + high) / 2.0
    forecast: Dict[str, int] = {}
    for dish in dishes:
        # Small per-dish noise ±1 around the mid-point
        noise = rng.integers(-1, 2)  # -1, 0, or +1
        forecast[dish] = max(0, int(round(mid + noise)))
    return forecast


def realise_demand(
    forecast: Dict[str, int],
    rng: np.random.Generator,
    spike_active: bool = False,
) -> Dict[str, int]:
    """
    Apply chef-skill noise to a forecast to get actual demand.

    Each dish gets an independent Uniform(0.8, 1.2) multiplier, reflecting
    the inherent unpredictability of a real kitchen. When a demand spike is
    active the result is further multiplied by DEMAND_SPIKE_MULTIPLIER.

    Args:
        forecast:     The forecast dict produced by generate_forecast().
        rng:          The same seeded RNG (advances state deterministically).
        spike_active: If True, multiply final demand by 2.5.

    Returns:
        Dict mapping dish name → actual realised demand (int ≥ 0).
    """
    actual: Dict[str, int] = {}
    for dish, forecast_qty in forecast.items():
        chef_skill = rng.uniform(CHEF_SKILL_LO, CHEF_SKILL_HI)
        raw = forecast_qty * chef_skill
        if spike_active:
            raw *= DEMAND_SPIKE_MULTIPLIER
        actual[dish] = max(0, int(math.floor(raw)))
    return actual


def compute_max_revenue(
    demand: Dict[str, int],
    dish_prices: Dict[str, float],
) -> float:
    """
    Compute the maximum possible revenue if every unit of demand is served.

    Args:
        demand:      Dict of dish → actual demand.
        dish_prices: Dict of dish → selling price per portion.

    Returns:
        Maximum possible revenue for this step (float ≥ 0).
    """
    return sum(demand.get(dish, 0) * dish_prices.get(dish, 0.0)
               for dish in demand)
