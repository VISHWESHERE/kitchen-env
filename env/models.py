"""
Pydantic v2 models for the Kitchen Environment.

Observation: everything the agent can see at each step.
Action:      what the agent decides to do each step.
Reward:      structured reward signal returned from step().
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Dish & Ingredient catalogues (static metadata, not env state)
# ---------------------------------------------------------------------------

DISHES: List[str] = [
    "Grilled Chicken",
    "Pasta Primavera",
    "Caesar Salad",
    "Beef Burger",
    "Tomato Soup",
]

INGREDIENTS: List[str] = [
    "chicken_breast",
    "pasta",
    "vegetables",
    "cream",
    "herbs",
    "beef_patty",
    "bun",
    "lettuce",
    "parmesan",
    "olive_oil",
    "tomatoes",
]

# Selling price per portion (USD)
DISH_PRICES: Dict[str, float] = {
    "Grilled Chicken": 18.0,
    "Pasta Primavera": 14.0,
    "Caesar Salad": 12.0,
    "Beef Burger": 16.0,
    "Tomato Soup": 8.0,
}

# Ingredient cost per unit (kg or L, USD)
INGREDIENT_COSTS: Dict[str, float] = {
    "chicken_breast": 12.0,
    "pasta": 2.0,
    "vegetables": 3.0,
    "cream": 4.0,
    "herbs": 8.0,
    "beef_patty": 15.0,
    "bun": 1.0,
    "lettuce": 2.0,
    "parmesan": 25.0,
    "olive_oil": 6.0,
    "tomatoes": 2.0,
}

# Ingredient expiry in days from baseline (for reference / display)
INGREDIENT_EXPIRY_DAYS: Dict[str, int] = {
    "chicken_breast": 2,
    "pasta": 30,
    "vegetables": 3,
    "cream": 5,
    "herbs": 7,
    "beef_patty": 1,
    "bun": 3,
    "lettuce": 4,
    "parmesan": 14,
    "olive_oil": 365,
    "tomatoes": 5,
}

# Ingredient usage per portion of each dish (kg or L per portion)
DISH_INGREDIENT_USAGE: Dict[str, Dict[str, float]] = {
    "Grilled Chicken": {
        "chicken_breast": 0.25,
        "olive_oil": 0.02,
        "herbs": 0.01,
    },
    "Pasta Primavera": {
        "pasta": 0.15,
        "vegetables": 0.20,
        "cream": 0.10,
    },
    "Caesar Salad": {
        "lettuce": 0.30,
        "chicken_breast": 0.15,
        "parmesan": 0.03,
    },
    "Beef Burger": {
        "beef_patty": 0.20,
        "bun": 0.10,
        "vegetables": 0.05,
    },
    "Tomato Soup": {
        "tomatoes": 0.40,
        "cream": 0.05,
        "herbs": 0.005,
    },
}

# Pre-computed ingredient cost per portion of each dish
DISH_INGREDIENT_COST: Dict[str, float] = {
    dish: sum(
        qty * INGREDIENT_COSTS[ing]
        for ing, qty in ingredients.items()
    )
    for dish, ingredients in DISH_INGREDIENT_USAGE.items()
}


# ---------------------------------------------------------------------------
# Core Pydantic models
# ---------------------------------------------------------------------------


class IngredientState(BaseModel):
    """Stock level and freshness of a single ingredient."""

    quantity: float = Field(..., description="Current stock in kg or L")
    expiry_steps_remaining: int = Field(
        ..., description="Steps until this batch expires (0 = already expired)"
    )
    pending_order: float = Field(
        default=0.0,
        description="Units already ordered but not yet delivered",
    )
    pending_order_arrives_in: int = Field(
        default=0,
        description="Steps until pending order arrives (0 = arrives next step)",
    )


class Observation(BaseModel):
    """
    Full observation the agent receives at every step.

    The agent has access to:
    - Current inventory of every ingredient
    - How many portions of each dish are currently prepped (ready to serve)
    - Demand forecast for this hour per dish
    - Which hour of the shift we are at
    - Step index within the episode
    - Whether a supplier delay is active
    - Whether a demand spike is active this step
    """

    step: int = Field(..., description="Current step index (0-based)")
    hour: int = Field(..., description="Kitchen hour (11–20)")
    inventory: Dict[str, IngredientState] = Field(
        ..., description="Current stock of each ingredient"
    )
    prepped_portions: Dict[str, int] = Field(
        ..., description="Ready-to-serve portions of each dish"
    )
    demand_forecast: Dict[str, int] = Field(
        ...,
        description=(
            "Forecast demand (portions) for this hour per dish. "
            "Actual demand = forecast × chef_skill_noise ~ Uniform(0.8, 1.2)."
        ),
    )
    supplier_delay_active: bool = Field(
        default=False,
        description="If True, new orders arrive 2 steps late instead of 1",
    )
    demand_spike_active: bool = Field(
        default=False,
        description="If True, actual demand is multiplied by 2.5 this step",
    )
    cumulative_revenue: float = Field(
        default=0.0, description="Total revenue earned so far this episode"
    )
    cumulative_waste_cost: float = Field(
        default=0.0, description="Total cost of wasted prepped portions so far"
    )
    cumulative_stockouts: int = Field(
        default=0, description="Total stockout events so far this episode"
    )
    episode_done: bool = Field(
        default=False, description="True when the episode has ended"
    )


class Action(BaseModel):
    """
    The agent's decision at each step.

    prep_portions:
        How many portions of each dish to prepare this hour.
        Prepping consumes ingredients immediately.
        Prep quantities must be non-negative integers.

    reorder_ingredients:
        How many units (kg / L) of each ingredient to order.
        Ordered stock arrives next step (or 2 steps later if supplier delay).
        Quantities must be non-negative floats.
        Ordering > 3× current stock triggers an over-ordering penalty.
    """

    prep_portions: Dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Portions of each dish to prep this step. "
            "Keys are dish names from the DISHES list. "
            "Missing keys default to 0."
        ),
    )
    reorder_ingredients: Dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Units (kg or L) of each ingredient to reorder. "
            "Keys are ingredient names from the INGREDIENTS list. "
            "Missing keys default to 0.0."
        ),
    )


class Reward(BaseModel):
    """
    Structured reward signal returned at every step.

    Reward components:
    - revenue_component:   positive, proportional to revenue earned vs maximum possible
    - waste_penalty:       negative, 1.5× the ingredient cost of wasted food
    - stockout_penalty:    negative, 2.0 per stockout event
    - ordering_penalty:    negative, -5.0 if any ingredient ordered > 3× current stock
    - step_reward:         sum of all components (can be negative)

    Reward is never binary — it always carries per-step signal.
    """

    step_reward: float = Field(..., description="Total reward this step")
    revenue_component: float = Field(
        ..., description="Revenue earned relative to maximum possible (×10)"
    )
    waste_penalty: float = Field(..., description="Penalty for food waste (≤0)")
    stockout_penalty: float = Field(
        ..., description="Penalty for stockout events (≤0)"
    )
    ordering_penalty: float = Field(
        ..., description="Penalty for panic over-ordering (≤0)"
    )
    portions_served: Dict[str, int] = Field(
        ..., description="Portions actually served per dish this step"
    )
    actual_demand: Dict[str, int] = Field(
        ..., description="Actual demand that materialised this step"
    )
    waste_portions: Dict[str, int] = Field(
        ..., description="Portions prepped but not served (wasted) per dish"
    )
    stockout_events: Dict[str, int] = Field(
        ...,
        description="Unmet demand per dish this step (demand − served)",
    )
    revenue_this_step: float = Field(
        ..., description="Revenue generated this step (USD)"
    )
    max_possible_revenue: float = Field(
        ..., description="Revenue if every demand unit was served (USD)"
    )
