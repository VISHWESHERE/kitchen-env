"""
FastAPI application — all OpenEnv-compliant HTTP endpoints.

Endpoints:
  GET  /health              → {"status": "ok"}
  POST /reset               → Observation (initial state)
  POST /step                → {obs, reward, done, info}
  GET  /state               → full internal state dict
  POST /tasks               → list of task metadata dicts
  POST /run_task/{task_id}  → {task_id, score, history, cumulative_reward}

The server maintains a single global KitchenEnv instance for interactive
/reset and /step calls. Task runs (/run_task) use isolated instances.

Start server (development):
    uvicorn app:app --host 0.0.0.0 --port 7860

Production (from Dockerfile):
    CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
"""

from __future__ import annotations

import traceback
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env.kitchen_env import KitchenEnv, DINNER_SCHEDULE, LUNCH_SCHEDULE, SINGLE_STEP_SCHEDULE
from env.models import Action, DISHES
from tasks.task1_stock import run_task_with_actions as t1_run, TASK_ID as T1_ID
from tasks.task2_waste import run_task_with_actions as t2_run, TASK_ID as T2_ID
from tasks.task3_shift import (
    run_task_with_actions as t3_run,
    TASK_ID as T3_ID,
    SPIKE_STEPS,
    SUPPLIER_DELAY,
)

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Kitchen Env — OpenEnv API",
    description=(
        "Restaurant kitchen food-prep management RL environment. "
        "OpenEnv-compliant REST API."
    ),
    version="1.0.0",
)

# Global interactive environment (used by /reset, /step, /state)
_env: KitchenEnv = KitchenEnv()
_env.reset()

# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class StepRequest(BaseModel):
    """Request body for POST /step."""
    action: Action


class RunTaskRequest(BaseModel):
    """
    Optional request body for POST /run_task/{task_id}.
    If actions are omitted a heuristic fallback agent is used.
    """
    actions: Optional[List[Action]] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", tags=["system"])
def root() -> Dict[str, Any]:
    """
    Root endpoint — API information and available endpoints.
    """
    return {
        "name": "Kitchen Env — OpenEnv API",
        "version": "1.0.0",
        "description": "Restaurant kitchen food-prep management RL environment",
        "endpoints": {
            "health": "GET /health — Health check",
            "reset": "POST /reset?seed=42 — Reset environment",
            "step": "POST /step — Take action step",
            "state": "GET /state — Get current state",
            "tasks": "POST /tasks — List available tasks",
            "run_task": "POST /run_task/{task_id} — Run a specific task"
        },
        "docs": "/docs",
        "openapi": "/openapi.json"
    }


@app.get("/health", tags=["system"])
def health() -> Dict[str, str]:
    """Liveness check. Required by OpenEnv and HF Spaces."""
    return {"status": "ok"}


@app.post("/reset", tags=["env"])
def reset_env(seed: int = 42) -> Dict[str, Any]:
    """
    Reset the interactive environment and return the initial observation.

    Query params:
      seed (int): RNG seed for deterministic episodes. Default 42.
    """
    global _env
    _env = KitchenEnv(seed=seed, schedule=LUNCH_SCHEDULE)
    obs = _env.reset()
    return obs.model_dump()


@app.post("/step", tags=["env"])
def step_env(body: StepRequest) -> Dict[str, Any]:
    """
    Advance the interactive environment by one step.

    Body: {"action": {"prep_portions": {...}, "reorder_ingredients": {...}}}
    """
    if _env._done:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call POST /reset to start a new episode.",
        )
    try:
        obs, reward, done, info = _env.step(body.action)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "observation": obs.model_dump(),
        "reward": reward.model_dump(),
        "done": done,
        "info": info,
    }


@app.get("/state", tags=["env"])
def get_state() -> Dict[str, Any]:
    """Return the full internal state of the interactive environment."""
    return _env.state()


@app.post("/tasks", tags=["tasks"])
def list_tasks() -> List[Dict[str, Any]]:
    """Return metadata for all available tasks."""
    try:
        with open("openenv.yaml", "r") as f:
            meta = yaml.safe_load(f)
        return meta.get("tasks", [])
    except FileNotFoundError:
        # Fallback if YAML file not accessible
        return [
            {
                "id": T1_ID,
                "name": "Stock Check (Easy)",
                "difficulty": "easy",
                "description": "Single-step demand forecasting and prep task.",
            },
            {
                "id": T2_ID,
                "name": "Lunch Service Waste Minimizer (Medium)",
                "difficulty": "medium",
                "description": "6-step lunch service. Balance waste vs stockouts.",
            },
            {
                "id": T3_ID,
                "name": "Full Dinner Shift with Spikes (Hard)",
                "difficulty": "hard",
                "description": "8-step dinner with demand spikes and supplier delays.",
            },
        ]


@app.post("/run_task/{task_id}", tags=["tasks"])
def run_task(task_id: str, body: Optional[RunTaskRequest] = None) -> Dict[str, Any]:
    """
    Run a complete task episode and return the official grader score.

    Path params:
      task_id: one of task1_stock_check | task2_waste_minimizer | task3_full_shift

    Body (optional):
      actions: list of Action objects to replay. If omitted, a heuristic fallback
               agent is used (preps forecast quantities, reorders when low).

    Returns:
      {task_id, score, history, cumulative_reward}
    """
    actions: List[Action] = []
    if body and body.actions:
        actions = body.actions

    try:
        if task_id == T1_ID:
            result = t1_run(actions)
        elif task_id == T2_ID:
            result = t2_run(actions)
        elif task_id == T3_ID:
            result = t3_run(actions)
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown task_id '{task_id}'. "
                       f"Valid IDs: {T1_ID}, {T2_ID}, {T3_ID}",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Task execution failed: {exc}\n{traceback.format_exc()}",
        ) from exc

    return result


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=7860)
