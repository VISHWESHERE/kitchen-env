"""
Task 1 — Stock Check (Easy)

Difficulty: Easy
Seed: 42
Steps: 1

WHAT IS TESTED:
  The agent sees the current inventory and a demand forecast for a single
  lunch-rush hour (hour 12). It must decide how many portions of each dish
  to prep. No reordering is required; the task is purely about reading the
  forecast and mapping it to prep quantities.

GRADER LOGIC:
  Score = 1.0 - MAE(prep_chosen, actual_demand) / max(actual_demand)

  A perfect agent that preps exactly actual_demand portions scores 1.0.
  An agent that preps wildly wrong (e.g. 0 everything) scores close to 0.0.
  Score is clipped to [0.0, 1.0].

WHY THIS MEASURES SUCCESS:
  The simplest possible kitchen skill: mapping a demand signal to prep.
  Even this single step tests whether the agent understands the action space
  and the concept of demand-driven prep.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from env.kitchen_env import KitchenEnv, SINGLE_STEP_SCHEDULE
from env.models import Action, DISHES

TASK_ID = "task1_stock_check"
TASK_SEED = 42


def run_task(agent_fn: Any) -> float:
    """
    Run the Task 1 episode using agent_fn and return the grader score.

    Args:
        agent_fn: A callable with signature (observation: Observation) → Action.
                  Must be deterministic for reproducibility.

    Returns:
        float in [0.0, 1.0].
    """
    env = KitchenEnv(seed=TASK_SEED, schedule=SINGLE_STEP_SCHEDULE)
    obs = env.reset()

    action = agent_fn(obs)
    _obs, reward, done, info = env.step(action)

    return grade(env)


def grade(env: KitchenEnv) -> float:
    """
    Compute the Task 1 score from a completed (or partially completed) episode.

    Uses the first (and only) recorded step in env.history.
    Deterministic: same env.history → same score.

    Returns:
        float in [0.0, 1.0].
    """
    if not env.history:
        return 0.0

    record = env.history[0]
    prep = record["actual_prep"]      # dict[dish → portions actually prepped]
    demand = record["actual_demand"]  # dict[dish → actual realised demand]

    prep_vals = np.array([prep.get(dish, 0) for dish in DISHES], dtype=float)
    demand_vals = np.array([demand.get(dish, 0) for dish in DISHES], dtype=float)

    # Mean Absolute Error between prep and actual demand
    mae = float(np.mean(np.abs(prep_vals - demand_vals)))
    max_demand = float(np.max(demand_vals)) if demand_vals.max() > 0 else 1.0

    score = 1.0 - mae / max_demand
    return float(np.clip(score, 0.0, 1.0))


def run_task_with_actions(actions: list) -> Dict[str, Any]:
    """
    Run Task 1 by replaying a pre-specified list of Action objects.
    Used by the FastAPI /run_task endpoint.

    Args:
        actions: List of Action objects, one per step. If fewer than the
                 episode length, the last action is repeated.

    Returns:
        dict with keys: score, history, cumulative_reward.
    """
    env = KitchenEnv(seed=TASK_SEED, schedule=SINGLE_STEP_SCHEDULE)
    obs = env.reset()
    done = False
    step = 0

    while not done:
        if step < len(actions):
            action = actions[step]
        else:
            # Fallback: prep the forecast quantity for each dish
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
    Run Task 1 with a simple heuristic agent (prep = forecast).
    Used by inference.py for baseline scoring.
    Returns the grader score.
    """
    def heuristic_agent(obs):
        return Action(
            prep_portions={
                dish: obs.demand_forecast.get(dish, 5) for dish in DISHES
            },
            reorder_ingredients={},
        )

    env = KitchenEnv(seed=TASK_SEED, schedule=SINGLE_STEP_SCHEDULE)
    obs = env.reset()
    action = heuristic_agent(obs)
    env.step(action)
    return grade(env)
