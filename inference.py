"""
inference.py — Baseline LLM agent for Kitchen Env (OpenEnv Hackathon)

MANDATORY FILE. Must be in root directory. Uses OpenAI Python client ONLY.

Usage:
    export API_BASE_URL=https://router.huggingface.co/v1
    export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
    export HF_TOKEN=hf_...
    python inference.py

The script:
  1. Initialises the OpenAI client with credentials from environment variables.
  2. Runs Task 1 (easy), Task 2 (medium), Task 3 (hard) sequentially.
  3. For each task: loops through episode steps, calls LLM, parses JSON action.
  4. Handles LLM failures gracefully with a deterministic fallback action.
  5. Prints per-step summaries and final reproducible scores.

Expected output:
    Task 1 (easy)   — score: 0.XX
    Task 2 (medium) — score: 0.XX
    Task 3 (hard)   — score: 0.XX
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from typing import Any, Dict, Optional

from openai import OpenAI

from env.kitchen_env import (
    KitchenEnv,
    LUNCH_SCHEDULE,
    DINNER_SCHEDULE,
    SINGLE_STEP_SCHEDULE,
)
from env.models import (
    Action,
    DISH_INGREDIENT_USAGE,
    DISH_PRICES,
    DISHES,
    INGREDIENTS,
    Observation,
)
from tasks.task1_stock import grade as t1_grade, TASK_SEED as T1_SEED
from tasks.task2_waste import grade as t2_grade, TASK_SEED as T2_SEED
from tasks.task3_shift import (
    grade as t3_grade,
    TASK_SEED as T3_SEED,
    SPIKE_STEPS,
    SUPPLIER_DELAY,
)

# ---------------------------------------------------------------------------
# Credentials — read from environment variables ONLY
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.getenv("MODEL_NAME", "meta-llama/Llama-3.3-70B-Instruct")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
API_KEY: str = HF_TOKEN or os.getenv("API_KEY", "")

# ---------------------------------------------------------------------------
# OpenAI client — initialised exactly as required by competition spec
# ---------------------------------------------------------------------------

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY,
)

# ---------------------------------------------------------------------------
# Menu constants for prompt context
# ---------------------------------------------------------------------------

MENU_CONTEXT = """
== RESTAURANT MENU ==
Dish                | Sell Price | Ingredients Required per Portion
--------------------|------------|----------------------------------
Grilled Chicken     | $18        | chicken_breast 0.25kg, olive_oil 0.02L, herbs 0.01kg
Pasta Primavera     | $14        | pasta 0.15kg, vegetables 0.20kg, cream 0.10L
Caesar Salad        | $12        | lettuce 0.30kg, chicken_breast 0.15kg, parmesan 0.03kg
Beef Burger         | $16        | beef_patty 0.20kg, bun 0.10kg, vegetables 0.05kg
Tomato Soup         | $8         | tomatoes 0.40kg, cream 0.05L, herbs 0.005kg

== INGREDIENT COSTS (USD per kg or L) ==
chicken_breast $12, pasta $2, vegetables $3, cream $4, herbs $8,
beef_patty $15, bun $1, lettuce $2, parmesan $25, olive_oil $6, tomatoes $2
"""

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def build_prompt(obs: Observation, step_num: int, task_description: str) -> str:
    """
    Build an LLM prompt from the current observation.

    The prompt includes:
      - Task context and step number
      - Full observation as JSON
      - Menu with prices and ingredient requirements
      - Exact action schema the LLM must return
      - Guidance on balancing waste vs stockouts
    """
    obs_dict = obs.model_dump()

    warning_lines = []
    if obs.supplier_delay_active:
        warning_lines.append("⚠️  SUPPLIER DELAY ACTIVE: orders arrive 2 steps late, not 1.")
    if obs.demand_spike_active:
        warning_lines.append("⚠️  DEMAND SPIKE ACTIVE THIS STEP: actual demand may be 2.5× forecast!")
    warnings = "\n".join(warning_lines)

    inventory_summary = []
    for ing, state in obs.inventory.items():
        line = f"  {ing}: {state.quantity:.2f} units"
        if state.pending_order > 0:
            line += f" (+ {state.pending_order:.2f} ordered, arrives in {state.pending_order_arrives_in} step(s))"
        if state.expiry_steps_remaining <= 2:
            line += f" ⚠️  EXPIRES IN {state.expiry_steps_remaining} STEP(S)"
        inventory_summary.append(line)

    forecast_lines = [
        f"  {dish}: {qty} portions" for dish, qty in obs.demand_forecast.items()
    ]

    prompt = f"""You are an expert restaurant kitchen manager AI.

TASK: {task_description}
STEP: {step_num} | HOUR: {obs.hour}
{warnings}

== CURRENT OBSERVATION ==
{json.dumps(obs_dict, indent=2)}

== INVENTORY SUMMARY ==
{chr(10).join(inventory_summary)}

== DEMAND FORECAST THIS HOUR ==
{chr(10).join(forecast_lines)}
(Note: actual demand = forecast × chef_skill_noise [Uniform(0.8, 1.2)].
 You cannot predict it exactly — hedge appropriately.)

{MENU_CONTEXT}

== CUMULATIVE STATS SO FAR ==
Revenue earned: ${obs.cumulative_revenue:.2f}
Waste cost incurred: ${obs.cumulative_waste_cost:.2f}
Stockout events: {obs.cumulative_stockouts}

== IMPORTANT REWARD RULES ==
1. Over-prepping is penalised MORE than under-prepping: waste costs 1.5× ingredient cost.
2. Each stockout costs -2 reward points. Missing demand is bad but wastes nothing.
3. Panic bulk-ordering (> 3× current stock) triggers -5 penalty. Order conservatively.
4. Ordered ingredients arrive next step (or 2 steps if supplier delay is active).
5. Ingredients with 0 expiry steps are already worthless — don't cook with expired stock.

== YOUR TASK ==
Decide:
  A) How many portions of each dish to PREP this step.
  B) How many units of each ingredient to REORDER (optional).

Output ONLY valid JSON in this exact format (no markdown, no explanation):
{{
  "prep_portions": {{
    "Grilled Chicken": <int>,
    "Pasta Primavera": <int>,
    "Caesar Salad": <int>,
    "Beef Burger": <int>,
    "Tomato Soup": <int>
  }},
  "reorder_ingredients": {{
    "chicken_breast": <float>,
    "pasta": <float>,
    "vegetables": <float>,
    "cream": <float>,
    "herbs": <float>,
    "beef_patty": <float>,
    "bun": <float>,
    "lettuce": <float>,
    "parmesan": <float>,
    "olive_oil": <float>,
    "tomatoes": <float>
  }}
}}

Rules:
- All prep quantities must be non-negative integers.
- All reorder quantities must be non-negative floats (0.0 means no order).
- Do not include explanations, only the JSON object.
"""
    return prompt


# ---------------------------------------------------------------------------
# Fallback action (used if LLM fails or JSON parse fails)
# ---------------------------------------------------------------------------


def fallback_action(obs: Observation) -> Action:
    """
    Safe deterministic fallback: prep = forecast, no reorder.
    Used whenever the LLM call fails or the response cannot be parsed.
    """
    prep = {dish: max(1, obs.demand_forecast.get(dish, 5)) for dish in DISHES}
    return Action(prep_portions=prep, reorder_ingredients={})


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def call_llm(prompt: str, max_retries: int = 2) -> Optional[str]:
    """
    Call the LLM via OpenAI client and return the raw text response.

    Retries up to max_retries times on transient failures.
    Returns None if all attempts fail.
    """
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert restaurant kitchen manager. "
                            "Always respond with valid JSON only — no markdown, "
                            "no explanation. Follow the schema exactly."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=512,
            )
            return response.choices[0].message.content
        except Exception as exc:
            print(f"  [LLM] Attempt {attempt + 1} failed: {exc}")
            if attempt < max_retries:
                time.sleep(2)
    return None


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


def parse_action(raw: Optional[str], obs: Observation) -> Action:
    """
    Parse LLM text output into an Action object.

    Handles:
    - Missing raw response → fallback
    - JSON parse errors → fallback
    - Missing keys → fill with safe defaults
    - Negative values → clip to 0
    """
    if not raw:
        print("  [PARSE] Empty response — using fallback action.")
        return fallback_action(obs)

    # Strip markdown fences if present
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        cleaned = "\n".join(
            line for line in lines if not line.strip().startswith("```")
        )

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        print(f"  [PARSE] JSON decode error: {exc}. Using fallback action.")
        return fallback_action(obs)

    # Build prep_portions safely
    raw_prep: Dict[str, Any] = data.get("prep_portions", {})
    prep_portions: Dict[str, int] = {}
    for dish in DISHES:
        val = raw_prep.get(dish, obs.demand_forecast.get(dish, 5))
        try:
            prep_portions[dish] = max(0, int(val))
        except (TypeError, ValueError):
            prep_portions[dish] = obs.demand_forecast.get(dish, 5)

    # Build reorder_ingredients safely
    raw_reorder: Dict[str, Any] = data.get("reorder_ingredients", {})
    reorder_ingredients: Dict[str, float] = {}
    for ing in INGREDIENTS:
        val = raw_reorder.get(ing, 0.0)
        try:
            reorder_ingredients[ing] = max(0.0, float(val))
        except (TypeError, ValueError):
            reorder_ingredients[ing] = 0.0

    return Action(
        prep_portions=prep_portions,
        reorder_ingredients=reorder_ingredients,
    )


# ---------------------------------------------------------------------------
# Task runners
# ---------------------------------------------------------------------------


def run_episode(
    env: KitchenEnv,
    task_name: str,
    task_description: str,
    grade_fn: Any,
) -> float:
    """
    Generic episode runner: reset → loop(prompt LLM → step) → grade.

    Args:
        env:              Configured KitchenEnv instance (not yet reset).
        task_name:        Human-readable label for logging.
        task_description: Short description injected into LLM prompt.
        grade_fn:         Function (env) → float score.

    Returns:
        Grader score as float in [0.0, 1.0].
    """
    print(f"\n{'='*60}")
    print(f"  Running: {task_name}")
    print(f"{'='*60}")

    obs = env.reset()
    done = False
    step_num = 0

    while not done:
        step_num += 1
        print(f"\n--- Step {step_num} | Hour {obs.hour} ---")
        if obs.demand_spike_active:
            print("  ⚠️  DEMAND SPIKE ACTIVE")
        if obs.supplier_delay_active:
            print("  ⚠️  SUPPLIER DELAY ACTIVE")

        # Build prompt and call LLM
        prompt = build_prompt(obs, step_num, task_description)
        raw_response = call_llm(prompt)

        if raw_response:
            action = parse_action(raw_response, obs)
        else:
            print("  [AGENT] LLM unavailable — using fallback action.")
            action = fallback_action(obs)

        # Log action
        print(f"  Prep: { {d: v for d, v in action.prep_portions.items() if v > 0} }")
        reorder_active = {k: v for k, v in action.reorder_ingredients.items() if v > 0}
        if reorder_active:
            print(f"  Reorder: {reorder_active}")

        # Step environment
        obs, reward, done, info = env.step(action)

        # Print step summary
        print(f"  Served: { {d: v for d, v in reward.portions_served.items() if v > 0} }")
        wasted = {d: v for d, v in reward.waste_portions.items() if v > 0}
        if wasted:
            print(f"  Waste:  {wasted}")
        stockouts = {d: v for d, v in reward.stockout_events.items() if v > 0}
        if stockouts:
            print(f"  Stockouts: {stockouts}")
        print(
            f"  Reward: {reward.step_reward:.3f} "
            f"(rev={reward.revenue_component:.2f}, "
            f"waste_pen={reward.waste_penalty:.2f}, "
            f"stockout_pen={reward.stockout_penalty:.2f}, "
            f"order_pen={reward.ordering_penalty:.2f})"
        )

    # Compute and return grader score
    score = grade_fn(env)
    print(f"\n  Episode complete. Cumulative reward: {env.state()['cumulative_reward']:.3f}")
    print(f"  Grader score: {score:.4f}")
    return score


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 60)
    print("  Kitchen Env — OpenEnv Hackathon Baseline Agent")
    print(f"  Model: {MODEL_NAME}")
    print(f"  API Base: {API_BASE_URL}")
    print("=" * 60)

    scores: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Task 1 — Easy: single-step stock check
    # ------------------------------------------------------------------
    env1 = KitchenEnv(seed=T1_SEED, schedule=SINGLE_STEP_SCHEDULE)
    scores["task1"] = run_episode(
        env=env1,
        task_name="Task 1 (easy) — Stock Check",
        task_description=(
            "Single step. Prep portions for one lunch-rush hour. "
            "Match your prep to the demand forecast as closely as possible."
        ),
        grade_fn=t1_grade,
    )

    # ------------------------------------------------------------------
    # Task 2 — Medium: 6-step lunch service
    # ------------------------------------------------------------------
    env2 = KitchenEnv(seed=T2_SEED, schedule=LUNCH_SCHEDULE)
    scores["task2"] = run_episode(
        env=env2,
        task_name="Task 2 (medium) — Lunch Service Waste Minimizer",
        task_description=(
            "6-step lunch service. Balance food waste against stockouts. "
            "Reorder ingredients proactively to avoid running out mid-service."
        ),
        grade_fn=t2_grade,
    )

    # ------------------------------------------------------------------
    # Task 3 — Hard: 8-step dinner with spikes and supplier delays
    # ------------------------------------------------------------------
    env3 = KitchenEnv(
        seed=T3_SEED,
        schedule=DINNER_SCHEDULE,
        supplier_delay=SUPPLIER_DELAY,
        spike_steps=SPIKE_STEPS,
    )
    scores["task3"] = run_episode(
        env=env3,
        task_name="Task 3 (hard) — Full Dinner Shift with Spikes",
        task_description=(
            "8-step dinner service. SUPPLIER DELAY ACTIVE (orders arrive 2 steps late). "
            "Demand spikes at steps 3 and 7 (×2.5). "
            "Optimise revenue, waste, and stockouts simultaneously. "
            "Order supplies early or you will run out during the spike."
        ),
        grade_fn=t3_grade,
    )

    # ------------------------------------------------------------------
    # Final scores — required format
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  FINAL SCORES")
    print("=" * 60)
    print(f"  Task 1 (easy)   — score: {scores['task1']:.2f}")
    print(f"  Task 2 (medium) — score: {scores['task2']:.2f}")
    print(f"  Task 3 (hard)   — score: {scores['task3']:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
