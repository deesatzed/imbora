"""Task state machine for The Associate.

Manages legal status transitions for tasks and enforces the state graph.
Tasks flow: PENDING → RESEARCHING → CODING → REVIEWING → DONE
with STUCK as a terminal state reachable from CODING or REVIEWING.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.core.exceptions import AgentError
from src.core.models import Task, TaskStatus
from src.db.repository import Repository

logger = logging.getLogger("associate.orchestrator.task_router")

# Legal state transitions — each key maps to the set of states it can move to
VALID_TRANSITIONS: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.PENDING: {TaskStatus.RESEARCHING, TaskStatus.CODING},
    TaskStatus.RESEARCHING: {TaskStatus.CODING, TaskStatus.STUCK},
    TaskStatus.CODING: {
        TaskStatus.RESEARCHING,
        TaskStatus.REVIEWING,
        TaskStatus.CODING,
        TaskStatus.STUCK,
    },
    TaskStatus.REVIEWING: {TaskStatus.DONE, TaskStatus.CODING, TaskStatus.STUCK},
    TaskStatus.STUCK: set(),  # Terminal — requires manual intervention
    TaskStatus.DONE: set(),   # Terminal
}


class TaskRouter:
    """Manages task status transitions with validation.

    All status changes go through this router to ensure legal transitions,
    logging, and database persistence.
    """

    def __init__(self, repository: Repository):
        self.repository = repository

    def transition(self, task: Task, new_status: TaskStatus, reason: Optional[str] = None) -> Task:
        """Move a task to a new status.

        Args:
            task: The task to transition.
            new_status: Target status.
            reason: Optional reason for the transition (logged).

        Returns:
            Updated Task with new status.

        Raises:
            AgentError: If transition is not allowed.
        """
        if not self.can_transition(task.status, new_status):
            raise AgentError(
                f"Invalid transition: {task.status.value} → {new_status.value} "
                f"for task '{task.title}' ({task.id})"
            )

        old_status = task.status
        self.repository.update_task_status(task.id, new_status)
        task.status = new_status

        log_msg = f"Task '{task.title}': {old_status.value} → {new_status.value}"
        if reason:
            log_msg += f" ({reason})"
        logger.info(log_msg)

        return task

    def can_transition(self, from_status: TaskStatus, to_status: TaskStatus) -> bool:
        """Check if a transition is legal."""
        return to_status in VALID_TRANSITIONS.get(from_status, set())

    def mark_stuck(self, task: Task, reason: str) -> Task:
        """Convenience method to mark a task as STUCK with a reason."""
        return self.transition(task, TaskStatus.STUCK, reason=reason)

    def mark_done(self, task: Task) -> Task:
        """Convenience method to mark a task as DONE."""
        return self.transition(task, TaskStatus.DONE, reason="completed successfully")

    def increment_attempt(self, task: Task) -> Task:
        """Increment the task's attempt counter."""
        self.repository.increment_task_attempt(task.id)
        task.attempt_count += 1
        logger.info("Task '%s': attempt count → %d", task.title, task.attempt_count)
        return task

    def increment_council(self, task: Task) -> Task:
        """Increment the task's council invocation counter."""
        self.repository.increment_task_council_count(task.id)
        task.council_count += 1
        logger.info("Task '%s': council count → %d", task.title, task.council_count)
        return task

    def should_trigger_council(self, task: Task, threshold: int) -> bool:
        """Check if the task has enough failures to trigger Council."""
        return task.attempt_count >= threshold and task.council_count == 0

    def should_mark_stuck(self, task: Task, max_council: int) -> bool:
        """Check if the task should be marked STUCK (council exhausted)."""
        return task.council_count >= max_council

    def get_transition_history_summary(self, task: Task) -> str:
        """Build a summary of the task's current state for logging."""
        return (
            f"Task '{task.title}' ({task.id}): "
            f"status={task.status.value}, "
            f"attempts={task.attempt_count}, "
            f"council={task.council_count}"
        )
