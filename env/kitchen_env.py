"""
Core Kitchen Environment — implements the full OpenEnv spec.

Public API:
  reset()         → Observation
  step(action)    → (Observation, Reward, bool, dict)
  state()         → dict  (full internal state)

Episode structure: the caller supplies a schedule (list of hours) which
defines the length of the episode.  Default schedules are used by tasks.

All randomness is seeded through a single numpy RNG so that any episode
is fully reproducible given the same seed and action sequence.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from env.demand import generate_forecast, realise_demand
from env.models import (
    DISH_INGREDIENT_COST,
    DISH_INGREDIENT_USAGE,
    DISH_PRICES,
    DISHES,
    INGREDIENT_COSTS,
    INGREDIENT_EXPIRY_DAYS,
    INGREDIENTS,
    Action,
    IngredientState,
    Observation,
)
from env.reward import compute_step_reward, Reward

# ---------------------------------------------------------------------------
# Default starting inventory (realistic restaurant quantities)
# ---------------------------------------------------------------------------
DEFAULT_INVENTORY: Dict[str, float] = {
    "chicken_breast": 10.0,
    "pasta": 5.0,
    "vegetables": 8.0,
    "cream": 4.0,
    "herbs": 2.0,
    "beef_patty": 6.0,
    "bun": 4.0,
    "lettuce": 3.0,
    "parmesan": 2.0,
    "olive_oil": 3.0,
    "tomatoes": 5.0,
}

# Default hour schedule for each service type
LUNCH_SCHEDULE: List[int] = [11, 12, 13, 14, 12, 13]   # 6 steps
DINNER_SCHEDULE: List[int] = [17, 18, 19, 20, 18, 19, 20, 19]  # 8 steps
SINGLE_STEP_SCHEDULE: List[int] = [12]  # 1 step (Task 1)


class KitchenEnv:
    """
    OpenEnv-compliant environment simulating restaurant kitchen management.

    Attributes:
        seed (int):              RNG seed for full reproducibility.
        schedule (list[int]):    Sequence of kitchen hours for the episode.
        supplier_delay (bool):   If True orders arrive 2 steps late.
        spike_steps (set[int]):  Steps at which demand × 2.5 spike fires.
    """

    def __init__(
        self,
        seed: int = 42,
        schedule: Optional[List[int]] = None,
        supplier_delay: bool = False,
        spike_steps: Optional[set] = None,
    ) -> None:
        self.seed = seed
        self.schedule: List[int] = schedule if schedule is not None else LUNCH_SCHEDULE
        self.supplier_delay: bool = supplier_delay
        self.spike_steps: set = spike_steps if spike_steps is not None else set()

        # Mutable episode state — initialised by reset()
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._step_idx: int = 0
        self._inventory: Dict[str, float] = {}
        self._expiry: Dict[str, int] = {}          # steps until ingredient expires
        self._prepped: Dict[str, int] = {}          # ready-to-serve portions
        self._pending_orders: List[Dict[str, float]] = []  # [step_due → {ing: qty}]
        self._cumulative_reward: float = 0.0
        self._cumulative_revenue: float = 0.0
        self._cumulative_waste_cost: float = 0.0
        self._cumulative_stockouts: int = 0
        self._done: bool = False
        self._last_reward: Optional[Reward] = None

        # Step-level tracking for graders
        self.history: List[Dict[str, Any]] = []

    # -----------------------------------------------------------------------
    # Public OpenEnv API
    # -----------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset the environment to its initial state and return the first observation."""
        self._rng = np.random.default_rng(self.seed)
        self._step_idx = 0
        self._inventory = copy.deepcopy(DEFAULT_INVENTORY)
        self._expiry = copy.deepcopy(INGREDIENT_EXPIRY_DAYS)
        self._prepped = {dish: 0 for dish in DISHES}
        self._pending_orders = []
        self._cumulative_reward = 0.0
        self._cumulative_revenue = 0.0
        self._cumulative_waste_cost = 0.0
        self._cumulative_stockouts = 0
        self._done = False
        self._last_reward = None
        self.history = []
        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Advance the environment by one step.

        Steps performed:
          1. Validate step legality.
          2. Deliver any pending ingredient orders that are due.
          3. Prep portions (consume ingredients).
          4. Reveal actual demand (forecast × chef_skill, with optional spike).
          5. Serve demand up to min(prepped, demand).
          6. Calculate waste (unsold prepped portions).
          7. Place new ingredient orders.
          8. Expire ingredients whose timer has run down.
          9. Compute and return reward.
        """
        if self._done:
            raise RuntimeError("Episode is done. Call reset() before stepping.")

        hour = self.schedule[self._step_idx]
        spike_active = self._step_idx in self.spike_steps

        # --- 1. Deliver due orders -------------------------------------------
        self._deliver_orders()

        # --- 2. Save pre-reorder inventory for penalty check -----------------
        inventory_snapshot = {k: v for k, v in self._inventory.items()}

        # --- 3. Prep portions (consume ingredients) ---------------------------
        actual_prep: Dict[str, int] = {}
        for dish in DISHES:
            requested = max(0, action.prep_portions.get(dish, 0))
            feasible = self._max_feasible_prep(dish, requested)
            actual_prep[dish] = feasible
            self._consume_ingredients(dish, feasible)
            self._prepped[dish] += feasible

        # --- 4. Forecast + realise demand -------------------------------------
        forecast = generate_forecast(hour, self._rng)
        actual_demand = realise_demand(forecast, self._rng, spike_active=spike_active)

        # --- 5 & 6. Serve demand, compute waste and stockouts -----------------
        prepped_snapshot = {d: self._prepped[d] for d in DISHES}

        for dish in DISHES:
            demand = actual_demand.get(dish, 0)
            served = min(self._prepped[dish], demand)
            self._prepped[dish] -= served  # leftover = waste at end of step
            actual_demand[dish] = demand

        # All remaining prepped portions are wasted (could not be sold)
        for dish in DISHES:
            self._prepped[dish] = 0  # clear prep between hours

        # --- 7. Place new ingredient orders ----------------------------------
        order_dict: Dict[str, float] = {}
        delay = 2 if self.supplier_delay else 1
        for ingredient, qty in action.reorder_ingredients.items():
            if ingredient in INGREDIENTS and qty > 0:
                order_dict[ingredient] = float(qty)
        if order_dict:
            arrival_step = self._step_idx + delay
            self._pending_orders.append({"arrives_at": arrival_step, **order_dict})

        # --- 8. Expire ingredients -------------------------------------------
        self._expire_ingredients()

        # --- 9. Compute reward -----------------------------------------------
        reward = compute_step_reward(
            action_prep=actual_prep,
            actual_demand=actual_demand,
            prepped_before_service=prepped_snapshot,
            reorder=action.reorder_ingredients,
            inventory_before_reorder=inventory_snapshot,
        )

        self._cumulative_reward += reward.step_reward
        self._cumulative_revenue += reward.revenue_this_step
        self._cumulative_waste_cost += abs(reward.waste_penalty) / 1.5
        total_stockouts_this_step = sum(reward.stockout_events.values())
        self._cumulative_stockouts += total_stockouts_this_step
        self._last_reward = reward

        # Record step history for graders
        self.history.append({
            "step": self._step_idx,
            "hour": hour,
            "forecast": forecast,
            "actual_demand": dict(actual_demand),
            "actual_prep": dict(actual_prep),
            "portions_served": dict(reward.portions_served),
            "waste_portions": dict(reward.waste_portions),
            "stockout_events": dict(reward.stockout_events),
            "revenue": reward.revenue_this_step,
            "max_revenue": reward.max_possible_revenue,
            "step_reward": reward.step_reward,
            "spike_active": spike_active,
        })

        self._step_idx += 1
        self._done = self._step_idx >= len(self.schedule)

        obs = self._build_observation()
        info: Dict[str, Any] = {
            "step": self._step_idx - 1,
            "hour": hour,
            "cumulative_reward": self._cumulative_reward,
            "cumulative_revenue": self._cumulative_revenue,
            "cumulative_waste_cost": self._cumulative_waste_cost,
            "cumulative_stockouts": self._cumulative_stockouts,
            "actual_demand": actual_demand,
            "actual_prep": actual_prep,
            "spike_active": spike_active,
        }
        return obs, reward, self._done, info

    def state(self) -> Dict[str, Any]:
        """Return the full internal state as a plain dict (for GET /state)."""
        return {
            "step": self._step_idx,
            "done": self._done,
            "schedule": self.schedule,
            "seed": self.seed,
            "supplier_delay": self.supplier_delay,
            "spike_steps": list(self.spike_steps),
            "inventory": {
                ing: {
                    "quantity": round(qty, 4),
                    "expiry_steps_remaining": self._expiry.get(ing, 0),
                }
                for ing, qty in self._inventory.items()
            },
            "prepped_portions": dict(self._prepped),
            "pending_orders": self._pending_orders,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "cumulative_revenue": round(self._cumulative_revenue, 4),
            "cumulative_waste_cost": round(self._cumulative_waste_cost, 4),
            "cumulative_stockouts": self._cumulative_stockouts,
            "history": self.history,
        }

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        """Construct an Observation from current internal state."""
        # Build forecast for the current hour (or last hour if done)
        if self._step_idx < len(self.schedule):
            hour = self.schedule[self._step_idx]
            forecast = generate_forecast(
                hour,
                # Peek-RNG: use a separate ephemeral RNG for display only
                np.random.default_rng(self.seed + self._step_idx * 17),
            )
            spike_active = self._step_idx in self.spike_steps
        else:
            hour = self.schedule[-1]
            forecast = {dish: 0 for dish in DISHES}
            spike_active = False

        # Build IngredientState for each ingredient
        inventory_obs: Dict[str, IngredientState] = {}
        for ing in INGREDIENTS:
            qty = max(0.0, self._inventory.get(ing, 0.0))
            pending_qty = 0.0
            pending_arrives_in = 0
            for order in self._pending_orders:
                if ing in order:
                    pending_qty += order[ing]
                    arrives_at = order.get("arrives_at", self._step_idx)
                    pending_arrives_in = max(
                        0, int(arrives_at) - self._step_idx
                    )
            inventory_obs[ing] = IngredientState(
                quantity=round(qty, 4),
                expiry_steps_remaining=self._expiry.get(ing, 0),
                pending_order=round(pending_qty, 4),
                pending_order_arrives_in=pending_arrives_in,
            )

        return Observation(
            step=self._step_idx,
            hour=hour,
            inventory=inventory_obs,
            prepped_portions=dict(self._prepped),
            demand_forecast=forecast,
            supplier_delay_active=self.supplier_delay,
            demand_spike_active=spike_active,
            cumulative_revenue=round(self._cumulative_revenue, 4),
            cumulative_waste_cost=round(self._cumulative_waste_cost, 4),
            cumulative_stockouts=self._cumulative_stockouts,
            episode_done=self._done,
        )

    def _max_feasible_prep(self, dish: str, requested: int) -> int:
        """How many portions can actually be prepped given current inventory?"""
        if requested <= 0:
            return 0
        usage = DISH_INGREDIENT_USAGE.get(dish, {})
        feasible = requested
        for ingredient, qty_per_portion in usage.items():
            stock = self._inventory.get(ingredient, 0.0)
            if qty_per_portion > 0:
                max_from_this = int(stock / qty_per_portion)
                feasible = min(feasible, max_from_this)
        return max(0, feasible)

    def _consume_ingredients(self, dish: str, portions: int) -> None:
        """Deduct ingredients for `portions` of `dish` from inventory."""
        if portions <= 0:
            return
        for ingredient, qty_per_portion in DISH_INGREDIENT_USAGE.get(dish, {}).items():
            self._inventory[ingredient] = max(
                0.0,
                self._inventory.get(ingredient, 0.0) - qty_per_portion * portions,
            )

    def _deliver_orders(self) -> None:
        """Deliver ingredient orders whose arrival step has been reached."""
        remaining: List[Dict[str, Any]] = []
        for order in self._pending_orders:
            arrives_at = order.get("arrives_at", 0)
            if int(arrives_at) <= self._step_idx:
                for key, val in order.items():
                    if key == "arrives_at":
                        continue
                    if key in INGREDIENTS:
                        self._inventory[key] = (
                            self._inventory.get(key, 0.0) + float(val)
                        )
            else:
                remaining.append(order)
        self._pending_orders = remaining

    def _expire_ingredients(self) -> None:
        """Decrement expiry counters and zero out expired stock."""
        for ing in list(self._expiry.keys()):
            self._expiry[ing] -= 1
            if self._expiry[ing] <= 0:
                self._inventory[ing] = 0.0
                self._expiry[ing] = 0
