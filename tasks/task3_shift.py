"""
Task 3 — Full Dinner Shift with Demand Spikes (Hard)

Difficulty: Hard
Seed: 13
Steps: 8 (hours: 17 → 18 → 19 → 20 → 18 → 19 → 20 → 19)

WHAT IS TESTED:
  The hardest challenge. On top of Task 2's multi-step dynamics, the agent
  must handle:
    - 2 demand spike events (×2.5 at steps 2 and 6 → hours 19 and 20 in
      the schedule), meaning sudden surges the agent cannot fully anticipate
    - 1 supplier delay event (all orders arrive 2 steps late instead of 1),
      forcing the agent to order earlier than it might otherwise
    - Revenue efficiency scoring: not just "did I avoid waste?" but "did I
      maximise the revenue I could have earned?"
    - Ingredient expiry under pressure: high-turnover ingredients like
      beef_patty expire in 1 day, so the agent risks losing stock it does
      not use fast enough

GRADER LOGIC (weighted 40/30/30):
  revenue_score  = total_revenue / theoretical_max_revenue
  waste_score    = 1.0 - (waste_cost / total_ingredient_cost_used)
  stockout_score = 1.0 - (stockout_count / total_demand)

  Final score = 0.4 * revenue_score + 0.3 * waste_score + 0.3 * stockout_score
  Clipped to [0.0, 1.0].

WHY THIS NEEDS REAL REASONING:
  The two spike steps create dilemmas: if the agent doesn't prep aggressively
  it misses a 2.5× revenue opportunity; if it preps aggressively on non-spike
  steps it generates massive waste. The supplier delay means any reactive
  reordering comes too late. Only an agent that thinks ahead (stores up
  supplies, monitors expiry, hedges its prep quantities) can score well.
  A frontier LLM that doesn't reason carefully will typically score 0.30–0.50.
  A thoughtful chain-of-thought LLM can reach 0.70+.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from env.kitchen_env import KitchenEnv, DINNER_SCHEDULE
from env.models import Action, DISH_INGREDIENT_COST, DISH_PRICES, DISHES, INGREDIENT_COSTS

TASK_ID = "task3_full_shift"
TASK_SEED = 13

# Spike fires at these step indices (0-based: step 2 = hour 19, step 6 = hour 20)
SPIKE_STEPS = {2, 6}

# Supplier delay is active for the entire episode
SUPPLIER_DELAY = True


def run_task(agent_fn: Any) -> float:
    """
    Run the Task 3 dinner shift episode using agent_fn.

    Args:
        agent_fn: Callable (Observation) → Action.
    Returns:
        float in [0.0, 1.0].
    """
    env = KitchenEnv(
        seed=TASK_SEED,
        schedule=DINNER_SCHEDULE,
        supplier_delay=SUPPLIER_DELAY,
        spike_steps=SPIKE_STEPS,
    )
    obs = env.reset()
    done = False

    while not done:
        action = agent_fn(obs)
        obs, _reward, done, _info = env.step(action)

    return grade(env)


def grade(env: KitchenEnv) -> float:
    """
    Compute the Task 3 score from a completed episode.

    Three components:
      1. revenue_score  (40%): fraction of maximum possible revenue earned
      2. waste_score    (30%): 1 − (ingredient cost of waste / total ingredient cost used)
      3. stockout_score (30%): 1 − (stockout portions / total demand)

    Deterministic given the same env.history.

    Returns:
        float in [0.0, 1.0].
    """
    if not env.history:
        return 0.0

    total_revenue: float = 0.0
    theoretical_max_revenue: float = 0.0
    total_ingredient_cost_used: float = 0.0
    waste_cost: float = 0.0
    total_demand: float = 0.0
    stockout_count: float = 0.0

    for record in env.history:
        total_revenue += record["revenue"]
        theoretical_max_revenue += record["max_revenue"]

        for dish in DISHES:
            prepped = record["actual_prep"].get(dish, 0)
            waste = record["waste_portions"].get(dish, 0)
            demand = record["actual_demand"].get(dish, 0)
            stockout = record["stockout_events"].get(dish, 0)

            dish_cost_per_portion = DISH_INGREDIENT_COST.get(dish, 0.0)

            # Total ingredient cost incurred for all prepped portions
            total_ingredient_cost_used += prepped * dish_cost_per_portion
            # Waste cost: ingredient cost of portions that could not be sold
            waste_cost += waste * dish_cost_per_portion
            total_demand += demand
            stockout_count += stockout

    # Revenue score: how much of the maximum revenue did the agent capture?
    if theoretical_max_revenue > 0:
        revenue_score = total_revenue / theoretical_max_revenue
    else:
        revenue_score = 1.0

    # Waste score: what fraction of ingredient cost did NOT become waste?
    if total_ingredient_cost_used > 0:
        waste_score = 1.0 - (waste_cost / total_ingredient_cost_used)
    else:
        waste_score = 1.0  # prepped nothing → no waste; stockouts handle the miss

    # Stockout score: what fraction of demand was fulfilled?
    if total_demand > 0:
        stockout_score = 1.0 - (stockout_count / total_demand)
    else:
        stockout_score = 1.0

    score = (
        0.4 * revenue_score
        + 0.3 * waste_score
        + 0.3 * stockout_score
    )
    return float(np.clip(score, 0.0, 1.0))


def run_task_with_actions(actions: List[Action]) -> Dict[str, Any]:
    """
    Run Task 3 by replaying a pre-specified list of Action objects.
    Used by the FastAPI /run_task endpoint.
    """
    env = KitchenEnv(
        seed=TASK_SEED,
        schedule=DINNER_SCHEDULE,
        supplier_delay=SUPPLIER_DELAY,
        spike_steps=SPIKE_STEPS,
    )
    obs = env.reset()
    done = False
    step = 0

    while not done:
        if step < len(actions):
            action = actions[step]
        else:
            action = Action(
                prep_portions={
                    dish: obs.demand_forecast.get(dish, 5) for dish in DISHES
                },
                reorder_ingredients={
                    ing: 1.5
                    for ing, state in obs.inventory.items()
                    if state.quantity < 3.0
                },
            )
        obs, _, done, _ = env.step(action)
        step += 1

    score = grade(env)
    return {
        "task_id": TASK_ID,
        "score": score,
        "history": env.history,
        "cumulative_reward": env.state()["cumulative_reward"],
    }


def run_grader_standalone() -> float:
    """
    Run Task 3 with a simple heuristic agent.
    Returns the grader score.
    """
    def heuristic_agent(obs):
        # Slightly over-prep to hedge against spikes; reorder when low
        return Action(
            prep_portions={
                dish: int(obs.demand_forecast.get(dish, 8) * 1.2) for dish in DISHES
            },
            reorder_ingredients={
                ing: 2.0
                for ing, state in obs.inventory.items()
                if state.quantity < 4.0
            },
        )

    env = KitchenEnv(
        seed=TASK_SEED,
        schedule=DINNER_SCHEDULE,
        supplier_delay=SUPPLIER_DELAY,
        spike_steps=SPIKE_STEPS,
    )
    obs = env.reset()
    done = False

    while not done:
        action = heuristic_agent(obs)
        obs, _, done, _ = env.step(action)

    return grade(env)
