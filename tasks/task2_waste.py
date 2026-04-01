"""
Task 2 — Lunch Service Waste Minimizer (Medium)

Difficulty: Medium
Seed: 7
Steps: 6 (hours: 11 → 12 → 13 → 14 → 12 → 13)

WHAT IS TESTED:
  The agent must manage a full lunch service across 6 steps. Unlike Task 1
  where a single prep decision is trivially easy, here the agent must:
    - Balance prep quantities against demand uncertainty across multiple hours
    - Decide whether and how much to reorder (depleted stock affects later steps)
    - Learn that over-prepping in the lunch rush leads to waste at peak hours
    - Learn that under-prepping loses revenue (stockouts) and customer trust

GRADER LOGIC (50/50 weighted):
  waste_score    = 1.0 - (total_waste_portions / total_prepped_portions)
  stockout_score = 1.0 - (total_stockouts / total_possible_demand)

  Final score = 0.5 * waste_score + 0.5 * stockout_score
  Clipped to [0.0, 1.0].

WHY THIS MEASURES SUCCESS:
  The key challenge is the tension between waste and stockouts. An agent that
  preps nothing scores 0.0 on waste (no waste!) but 0.0 on stockouts (missed
  all demand). Optimal agents must hedge and track cumulative state across steps.
  Reordering adds another dimension: running out of chicken_breast mid-service is
  a recoverable situation only if the agent ordered proactively.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from env.kitchen_env import KitchenEnv, LUNCH_SCHEDULE
from env.models import Action, DISHES

TASK_ID = "task2_waste_minimizer"
TASK_SEED = 7


def run_task(agent_fn: Any) -> float:
    """
    Run the Task 2 lunch service episode using agent_fn.

    Args:
        agent_fn: Callable (Observation) → Action.
    Returns:
        float in [0.0, 1.0].
    """
    env = KitchenEnv(seed=TASK_SEED, schedule=LUNCH_SCHEDULE)
    obs = env.reset()
    done = False

    while not done:
        action = agent_fn(obs)
        obs, _reward, done, _info = env.step(action)

    return grade(env)


def grade(env: KitchenEnv) -> float:
    """
    Compute the Task 2 score from a completed episode.

    Aggregates totals across ALL steps then computes the two component scores.
    Deterministic given the same env.history.

    Returns:
        float in [0.0, 1.0].
    """
    if not env.history:
        return 0.0

    total_prepped: float = 0.0
    total_waste: float = 0.0
    total_demand: float = 0.0
    total_stockouts: float = 0.0

    for record in env.history:
        for dish in DISHES:
            prepped = record["actual_prep"].get(dish, 0)
            waste = record["waste_portions"].get(dish, 0)
            demand = record["actual_demand"].get(dish, 0)
            stockout = record["stockout_events"].get(dish, 0)

            total_prepped += prepped
            total_waste += waste
            total_demand += demand
            total_stockouts += stockout

    # Waste score: proportion of prepped food that was NOT wasted (higher = better)
    if total_prepped > 0:
        waste_score = 1.0 - (total_waste / total_prepped)
    else:
        # Agent prepped nothing → waste_score full (no waste) but stockouts kill it
        waste_score = 1.0

    # Stockout score: proportion of demand that WAS fulfilled (higher = better)
    if total_demand > 0:
        stockout_score = 1.0 - (total_stockouts / total_demand)
    else:
        stockout_score = 1.0

    score = 0.5 * waste_score + 0.5 * stockout_score
    return float(np.clip(score, 0.0, 1.0))


def run_task_with_actions(actions: List[Action]) -> Dict[str, Any]:
    """
    Run Task 2 by replaying a pre-specified list of Action objects.
    Used by the FastAPI /run_task endpoint.
    """
    env = KitchenEnv(seed=TASK_SEED, schedule=LUNCH_SCHEDULE)
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
                reorder_ingredients={},
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
    Run Task 2 with a simple heuristic agent (prep = forecast each step).
    Returns the grader score.
    """
    def heuristic_agent(obs):
        return Action(
            prep_portions={
                dish: obs.demand_forecast.get(dish, 5) for dish in DISHES
            },
            reorder_ingredients={
                ing: 1.0
                for ing, state in obs.inventory.items()
                if state.quantity < 2.0
            },
        )

    env = KitchenEnv(seed=TASK_SEED, schedule=LUNCH_SCHEDULE)
    obs = env.reset()
    done = False

    while not done:
        action = heuristic_agent(obs)
        obs, _, done, _ = env.step(action)

    return grade(env)
