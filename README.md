---
title: Kitchen Env
emoji: 🍳
colorFrom: yellow
colorTo: green
sdk: docker
pinned: false
tags:
  - openenv
---

# 🍳 Kitchen Env — OpenEnv Restaurant Kitchen RL Environment

> **OpenEnv Hackathon Round 1 — Submission by vishweshreddy007**

A real-world, OpenEnv-compliant reinforcement learning environment where an AI agent manages a restaurant kitchen — deciding how many portions to prep each hour, when to reorder ingredients, and how to balance food waste against stockouts.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Environment Description](#environment-description)
3. [Observation Space](#observation-space)
4. [Action Space](#action-space)
5. [Reward Function](#reward-function)
6. [Tasks](#tasks)
7. [API Reference](#api-reference)
8. [Running Locally](#running-locally)
9. [Running inference.py](#running-inferencepy)
10. [Deployment](#deployment)

---

## 1. Overview

**Kitchen Env** simulates the operational decisions a head chef makes during a restaurant shift:

- **How many portions** of each dish to prepare each hour
- **When and how much** to reorder ingredients from suppliers
- **How to avoid food waste** — prepped food that expires unsold has a real cost
- **How to avoid stockouts** — running out of a dish during service hurts revenue and reputation

This is not a toy environment. It models the genuine tension between over-preparation (waste) and under-preparation (lost sales) that restaurant managers navigate every day.

---

## 2. Environment Description

### Kitchen Shift

The kitchen operates across two services:
- **Lunch service** (hours 11–14): 4 core hours plus possible additional steps
- **Dinner service** (hours 17–20): 4 core hours plus possible additional steps

Each **step = 1 hour**. At each step the agent:
1. Receives an observation (inventory, prepped portions, demand forecast)
2. Chooses prep quantities and optional ingredient reorders
3. The environment reveals actual demand (forecast × chef_skill_noise)
4. Portions are served up to `min(prepped, actual_demand)`
5. Unsold prepped portions are wasted; unmet demand = stockouts
6. Ordered ingredients arrive next step (or 2 steps if supplier delay active)
7. Reward is calculated and returned

### Dishes

| Dish | Sell Price | Ingredients per Portion |
|------|-----------|------------------------|
| Grilled Chicken | $18 | chicken_breast 0.25kg, olive_oil 0.02L, herbs 0.01kg |
| Pasta Primavera | $14 | pasta 0.15kg, vegetables 0.20kg, cream 0.10L |
| Caesar Salad | $12 | lettuce 0.30kg, chicken_breast 0.15kg, parmesan 0.03kg |
| Beef Burger | $16 | beef_patty 0.20kg, bun 0.10kg, vegetables 0.05kg |
| Tomato Soup | $8 | tomatoes 0.40kg, cream 0.05L, herbs 0.005kg |

### Ingredients

| Ingredient | Starting Stock | Cost/Unit | Expiry |
|------------|---------------|-----------|--------|
| chicken_breast | 10 kg | $12/kg | 2 days |
| pasta | 5 kg | $2/kg | 30 days |
| vegetables | 8 kg | $3/kg | 3 days |
| cream | 4 L | $4/L | 5 days |
| herbs | 2 kg | $8/kg | 7 days |
| beef_patty | 6 kg | $15/kg | 1 day ⚠️ |
| bun | 4 kg | $1/kg | 3 days |
| lettuce | 3 kg | $2/kg | 4 days |
| parmesan | 2 kg | $25/kg | 14 days |
| olive_oil | 3 L | $6/L | 365 days |
| tomatoes | 5 kg | $2/kg | 5 days |

### Demand Patterns

| Hour | Service Phase | Demand Range per Dish |
|------|-------------|----------------------|
| 11 | Pre-lunch | 2–4 portions |
| 12 | Lunch rush | 8–14 portions |
| 13 | Peak lunch | 10–18 portions |
| 14 | Post-lunch | 4–8 portions |
| 17 | Pre-dinner | 3–5 portions |
| 18 | Dinner rush | 9–15 portions |
| 19 | Peak dinner | 12–20 portions |
| 20 | Late dinner | 5–10 portions |

### Chef Skill Multiplier

```
actual_demand = forecast × Uniform(0.8, 1.2)
```

The agent receives the forecast but **cannot perfectly predict actual demand**. Agents must hedge their prep quantities appropriately.

---

## 3. Observation Space

The `Observation` Pydantic model contains:

```json
{
  "step": 0,
  "hour": 12,
  "inventory": {
    "chicken_breast": {
      "quantity": 10.0,
      "expiry_steps_remaining": 2,
      "pending_order": 0.0,
      "pending_order_arrives_in": 0
    }
  },
  "prepped_portions": {
    "Grilled Chicken": 0
  },
  "demand_forecast": {
    "Grilled Chicken": 11
  },
  "supplier_delay_active": false,
  "demand_spike_active": false,
  "cumulative_revenue": 0.0,
  "cumulative_waste_cost": 0.0,
  "cumulative_stockouts": 0,
  "episode_done": false
}
```

---

## 4. Action Space

The `Action` Pydantic model:

```json
{
  "prep_portions": {
    "Grilled Chicken": 10,
    "Pasta Primavera": 8,
    "Caesar Salad": 7,
    "Beef Burger": 9,
    "Tomato Soup": 5
  },
  "reorder_ingredients": {
    "chicken_breast": 3.0,
    "vegetables": 2.0
  }
}
```

**Rules:**
- `prep_portions`: non-negative integers; prepping consumes ingredients immediately
- `reorder_ingredients`: non-negative floats; missing keys = no order placed
- Ordering > 3× current stock triggers a **-5.0 ordering penalty**
- Ordered ingredients arrive **next step** (or 2 steps if `supplier_delay_active`)

---

## 5. Reward Function

Per-step reward:

```
revenue_ratio     = service_revenue / max_possible_revenue
revenue_component = revenue_ratio × 10

waste_penalty     = -(waste_cost) × 1.5      # Over-prepping hurts more than stockouts
stockout_penalty  = -(stockouts) × 2.0
ordering_penalty  = -5.0  if any ingredient ordered > 3× current stock, else 0

step_reward = revenue_component + waste_penalty + stockout_penalty + ordering_penalty
```

**Key design principle:** Waste is penalised at **1.5× ingredient cost** because throwing away prepped food is an unrecoverable loss. A restaurant can apologise for being "86'd" on a dish but cannot un-waste the food.

---

## 6. Tasks

### Task 1 — Stock Check (Easy) `task1_stock_check`
- **Steps:** 1
- **Seed:** 42
- **Goal:** Prep the right portions for a single lunch-rush hour
- **Score:** `1.0 - MAE(prep, actual_demand) / max(actual_demand)`, clipped to [0, 1]

### Task 2 — Lunch Service Waste Minimizer (Medium) `task2_waste_minimizer`
- **Steps:** 6 (hours: 11→12→13→14→12→13)
- **Seed:** 7
- **Goal:** Balance waste and stockouts across full lunch service
- **Score:** `0.5 × waste_score + 0.5 × stockout_score`, clipped to [0, 1]

### Task 3 — Full Dinner Shift with Spikes (Hard) `task3_full_shift`
- **Steps:** 8 (hours: 17→18→19→20→18→19→20→19)
- **Seed:** 13
- **Events:** 2 demand spikes (×2.5 at steps 2 and 6), supplier delay throughout
- **Goal:** Maximise revenue, minimise waste, avoid stockouts under uncertainty
- **Score:** `0.4 × revenue + 0.3 × waste_score + 0.3 × stockout_score`, clipped to [0, 1]

---

## 7. API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Liveness check → `{"status": "ok"}` |
| `POST` | `/reset` | Reset environment, returns initial `Observation` |
| `POST` | `/step` | Submit action → `{obs, reward, done, info}` |
| `GET` | `/state` | Full internal state dict |
| `POST` | `/tasks` | List all task metadata |
| `POST` | `/run_task/{task_id}` | Run full task episode → `{score, history}` |

### Example curl calls

```bash
# Health check
curl https://vishweshreddy007-kitchen-env.hf.space/health

# Reset environment
curl -X POST https://vishweshreddy007-kitchen-env.hf.space/reset

# Step with action
curl -X POST https://vishweshreddy007-kitchen-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"prep_portions": {"Grilled Chicken": 10, "Pasta Primavera": 8, "Caesar Salad": 7, "Beef Burger": 9, "Tomato Soup": 5}, "reorder_ingredients": {}}}'

# Run Task 1 grader
curl -X POST https://vishweshreddy007-kitchen-env.hf.space/run_task/task1_stock_check

# Run Task 2 grader
curl -X POST https://vishweshreddy007-kitchen-env.hf.space/run_task/task2_waste_minimizer

# Run Task 3 grader
curl -X POST https://vishweshreddy007-kitchen-env.hf.space/run_task/task3_full_shift
```

---

## 8. Running Locally

```bash
# Clone the repo
git clone https://huggingface.co/spaces/vishweshreddy007/kitchen-env
cd kitchen-env

# Install dependencies
pip install -r requirements.txt

# Start the server
uvicorn app:app --host 0.0.0.0 --port 7860

# Or with Docker
docker build -t kitchen-env .
docker run -p 7860:7860 kitchen-env
```

---

## 9. Running inference.py

```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export HF_TOKEN=hf_your_token_here

python inference.py
```

Expected output:
```
Task 1 (easy)   — score: 0.XX
Task 2 (medium) — score: 0.XX
Task 3 (hard)   — score: 0.XX
```

---

## 10. Deployment

See [HOW TO DEPLOY](#how-to-deploy) section below for complete deployment commands.

### Environment Variables (HF Space Secrets)

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | OpenAI-compatible API base URL | `https://router.huggingface.co/v1` |
| `MODEL_NAME` | LLM model identifier | `meta-llama/Llama-3.3-70B-Instruct` |
| `HF_TOKEN` | Hugging Face API token | — |

---

## HOW TO DEPLOY

### Step 1 — Install HF CLI and login
```bash
pip install huggingface_hub
huggingface-cli login
# Paste your HF token when prompted
```

### Step 2 — Push all project files to your HF Space
```bash
cd kitchen-env
git init
git add .
git commit -m "initial submission"
huggingface-cli repo create kitchen-env --type space --space-sdk docker
git remote add origin https://huggingface.co/spaces/YOUR_HF_USERNAME/kitchen-env
git push origin main
```

### Step 3 — Set secrets in HF Space settings UI
Go to: `huggingface.co/spaces/YOUR_HF_USERNAME/kitchen-env/settings`

Add these under **"Variables and Secrets"**:
```
API_BASE_URL = https://router.huggingface.co/v1
MODEL_NAME   = meta-llama/Llama-3.3-70B-Instruct
HF_TOKEN     = hf_your_token_here
```

### Step 4 — Test deployment with curl
```bash
curl https://YOUR_HF_USERNAME-kitchen-env.hf.space/health
curl -X POST https://YOUR_HF_USERNAME-kitchen-env.hf.space/reset
curl -X POST https://YOUR_HF_USERNAME-kitchen-env.hf.space/run_task/task1_stock_check
```

### Step 5 — Run inference.py locally before submitting
```bash
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export HF_TOKEN=hf_your_token_here
python inference.py
```

---

## Project Structure

```
kitchen-env/
├── app.py                  # FastAPI server — all HTTP endpoints
├── inference.py            # Baseline agent script (mandatory)
├── openenv.yaml            # OpenEnv metadata
├── Dockerfile              # Container definition
├── requirements.txt        # All Python dependencies
├── README.md               # This file
├── env/
│   ├── __init__.py
│   ├── kitchen_env.py      # Core environment: reset, step, state
│   ├── models.py           # Pydantic models: Observation, Action, Reward
│   ├── demand.py           # Demand simulation + chef skill randomness
│   └── reward.py           # Reward function logic
└── tasks/
    ├── __init__.py
    ├── task1_stock.py      # Easy task + grader
    ├── task2_waste.py      # Medium task + grader
    └── task3_shift.py      # Hard task + grader
```

---

*Built for the OpenEnv Hackathon Round 1. Kitchen Env — where every portion counts. 🍳*
