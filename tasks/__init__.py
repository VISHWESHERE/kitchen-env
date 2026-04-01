"""Tasks package."""

from tasks.task1_stock import (
    run_task_with_actions as task1_run,
    run_grader_standalone as task1_grade,
    TASK_ID as TASK1_ID,
)
from tasks.task2_waste import (
    run_task_with_actions as task2_run,
    run_grader_standalone as task2_grade,
    TASK_ID as TASK2_ID,
)
from tasks.task3_shift import (
    run_task_with_actions as task3_run,
    run_grader_standalone as task3_grade,
    TASK_ID as TASK3_ID,
)

__all__ = [
    "task1_run", "task1_grade", "TASK1_ID",
    "task2_run", "task2_grade", "TASK2_ID",
    "task3_run", "task3_grade", "TASK3_ID",
]
