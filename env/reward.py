"""
Reward function for the Kitchen Environment.

Design decisions:
  - Revenue component: rewards the agent for serving as much demand as
    possible, scaled to [0, 10] range.
  - Waste penalty: penalises food waste at 1.5× ingredient cost, because
    throwing away prepped food is a real, unrecoverable cost.
  - Stockout penalty: flat 2.0 per stockout event (the kitchen ran out
    of a dish). Slightly less severe than waste — you can apologise for
    being "86'd" but cannot un-waste food.
  - Ordering penalty: sharp -5.0 if any ingredient is ordered at > 3×
    current stock (modelling panic bulk-buying as obviously bad).

The step_reward can be negative. Episode normalisation in the graders
maps total reward to [0.0, 1.0] so that:
  - A purely random agent scores ≈ 0.30
  - A near-perfect oracle agent scores ≈ 0.95
"""

from __future__ import annotations

from typing import Dict

from env.models import (
    DISH_INGREDIENT_COST,
    DISH_PRICES,
    DISHES,
    Reward,
)

# ---------------------------------------------------------------------------
# Normalisation constants for end-of-episode scoring
# Calibrated so random ≈ 0.30, perfect ≈ 0.95.
# ---------------------------------------------------------------------------
SCORE_OFFSET: float = 15.0   # shift the raw cumulative total up
SCORE_SCALE: float = 80.0    # divide to compress into [0, 1] band


def compute_step_reward(
    action_prep: Dict[str, int],
    actual_demand: Dict[str, int],
    prepped_before_service: Dict[str, int],
    reorder: Dict[str, float],
    inventory_before_reorder: Dict[str, float],
) -> Reward:
    """
    Compute the structured reward for a single step.

    Args:
        action_prep:              Portions the agent decided to prep this step.
        actual_demand:            Realised demand (after chef-skill noise).
        prepped_before_service:   Total prepped portions available to serve
                                  (may include carry-over from last step).
        reorder:                  Ingredient reorder quantities from agent action.
        inventory_before_reorder: Inventory levels BEFORE this step's order
                                  is placed (used to detect panic ordering).

    Returns:
        A fully populated Reward model.
    """
    portions_served: Dict[str, int] = {}
    waste_portions: Dict[str, int] = {}
    stockout_events: Dict[str, int] = {}

    revenue_this_step: float = 0.0
    max_possible_revenue: float = 0.0
    waste_cost_total: float = 0.0
    stockout_count: int = 0

    for dish in DISHES:
        prepped = prepped_before_service.get(dish, 0)
        demand = actual_demand.get(dish, 0)
        price = DISH_PRICES.get(dish, 0.0)
        ingredient_cost = DISH_INGREDIENT_COST.get(dish, 0.0)

        served = min(prepped, demand)
        waste = max(0, prepped - demand)   # unsold prepped = wasted
        stockout = max(0, demand - prepped)

        portions_served[dish] = served
        waste_portions[dish] = waste
        stockout_events[dish] = stockout

        revenue_this_step += served * price
        max_possible_revenue += demand * price
        # Waste cost: the raw ingredient cost of every wasted portion
        waste_cost_total += waste * ingredient_cost
        stockout_count += stockout

    # --- Revenue component (0–10) -------------------------------------------
    if max_possible_revenue > 0:
        revenue_ratio = revenue_this_step / max_possible_revenue
    else:
        revenue_ratio = 1.0  # no demand this step → full marks
    revenue_component = revenue_ratio * 10.0

    # --- Waste penalty (≤ 0) -------------------------------------------------
    # Waste is penalised at 1.5× actual ingredient cost: throwing away
    # prepped food hurts more than just the raw material cost.
    waste_penalty = -waste_cost_total * 1.5

    # --- Stockout penalty (≤ 0) ----------------------------------------------
    stockout_penalty = -float(stockout_count) * 2.0

    # --- Over-ordering penalty (≤ 0) -----------------------------------------
    ordering_penalty = 0.0
    for ingredient, order_qty in reorder.items():
        if order_qty <= 0:
            continue
        current_stock = inventory_before_reorder.get(ingredient, 0.0)
        # If ordered quantity > 3× current stock, flag as panic ordering
        if order_qty > 3.0 * max(current_stock, 0.01):
            ordering_penalty = -5.0
            break  # one violation is enough to trigger the penalty

    step_reward = (
        revenue_component
        + waste_penalty
        + stockout_penalty
        + ordering_penalty
    )

    return Reward(
        step_reward=round(step_reward, 4),
        revenue_component=round(revenue_component, 4),
        waste_penalty=round(waste_penalty, 4),
        stockout_penalty=round(stockout_penalty, 4),
        ordering_penalty=round(ordering_penalty, 4),
        portions_served=portions_served,
        actual_demand=actual_demand,
        waste_portions=waste_portions,
        stockout_events=stockout_events,
        revenue_this_step=round(revenue_this_step, 4),
        max_possible_revenue=round(max_possible_revenue, 4),
    )


def normalise_episode_score(total_reward: float) -> float:
    """
    Map raw cumulative episode reward → normalised score in [0.0, 1.0].

    Calibrated so:
      - A random agent (average raw reward ≈ -15) →  score ≈ 0.0–0.30
      - A near-perfect agent (raw ≈ 65)            →  score ≈ 0.90–0.95
      - Formula: score = clip((total + offset) / scale, 0, 1)
    """
    raw = (total_reward + SCORE_OFFSET) / SCORE_SCALE
    return float(max(0.0, min(1.0, raw)))
