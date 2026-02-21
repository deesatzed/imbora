"""State Manager agent for The Associate.

First agent in the pipeline. Responsibilities:
1. Query hypothesis_log for previous failed approaches on this task
2. Build the forbidden approaches list
3. Checkpoint current git state (git stash) before Builder writes code
4. Save a context snapshot to the database
5. Return enriched TaskContext with all context Builder needs

State Manager uses checkpoint() before Builder writes code, and rewind()
when tests fail to restore the working tree.
"""

from __future__ import annotations

import logging
from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.exceptions import CheckpointError
from src.core.models import (
    AgentResult,
    ContextSnapshot,
    PeerReview,
    Task,
    TaskContext,
)
from src.db.repository import Repository
from src.tools.git_ops import checkpoint, is_git_repo

logger = logging.getLogger("associate.agent.state_manager")


class StateManager(BaseAgent):
    """Prepares task context with hypothesis injection and git checkpoint.

    Injected dependencies:
        repository: Database access for hypothesis_log, context_snapshots, peer_reviews.
        repo_path: Path to the git repository for checkpoint operations.
    """

    def __init__(self, repository: Repository, repo_path: str):
        super().__init__(name="StateManager", role="state_manager")
        self.repository = repository
        self.repo_path = repo_path

    def process(self, input_data: Any) -> AgentResult:
        """Build enriched TaskContext from a Task.

        Args:
            input_data: A Task model.

        Returns:
            AgentResult with TaskContext in data["task_context"].
        """
        task: Task = input_data

        # 1. Get failed approaches from hypothesis_log
        forbidden_approaches = self._get_forbidden_approaches(task)

        # 2. Get latest council diagnosis (if any)
        council_diagnosis = self._get_latest_council_diagnosis(task)

        # 3. Checkpoint current git state
        checkpoint_ref = self._checkpoint(task)

        # 4. Save context snapshot to database
        self._save_snapshot(task, checkpoint_ref)

        # 5. Build the enriched context
        task_context = TaskContext(
            task=task,
            forbidden_approaches=forbidden_approaches,
            checkpoint_ref=checkpoint_ref,
            previous_council_diagnosis=council_diagnosis,
        )

        return AgentResult(
            agent_name=self.name,
            status="success",
            data={"task_context": task_context.model_dump()},
        )

    def _get_forbidden_approaches(self, task: Task) -> list[str]:
        """Query hypothesis_log for all failed approaches on this task."""
        failed = self.repository.get_failed_approaches(task.id)
        if not failed:
            logger.info("No previous failed approaches for task '%s'", task.title)
            return []

        forbidden = []
        for entry in failed:
            summary = f"Attempt #{entry.attempt_number}: {entry.approach_summary}"
            if entry.error_signature:
                summary += f" (error: {entry.error_signature})"
            forbidden.append(summary)

        logger.info(
            "Found %d forbidden approaches for task '%s'",
            len(forbidden), task.title,
        )
        return forbidden

    def _get_latest_council_diagnosis(self, task: Task) -> str | None:
        """Get the most recent council peer review diagnosis."""
        reviews: list[PeerReview] = self.repository.get_peer_reviews(task.id)
        if not reviews:
            return None

        # reviews are ordered by created_at DESC — first is latest
        latest = reviews[0]
        diagnosis = f"Council diagnosis ({latest.model_used}): {latest.diagnosis}"
        if latest.recommended_approach:
            diagnosis += f"\nRecommended approach: {latest.recommended_approach}"
        logger.info("Injecting council diagnosis for task '%s'", task.title)
        return diagnosis

    def _checkpoint(self, task: Task) -> str:
        """Create a git checkpoint via stash.

        Returns:
            Stash reference string, or empty string if no changes to stash.

        Raises:
            CheckpointError: If git stash fails.
        """
        if not is_git_repo(self.repo_path):
            logger.warning("Path '%s' is not a git repo — skipping checkpoint", self.repo_path)
            return ""

        try:
            ref = checkpoint(
                self.repo_path,
                message=f"associate-checkpoint-{task.id}-attempt-{task.attempt_count + 1}",
            )
            if ref:
                logger.info("Checkpoint created: %s", ref)
            return ref
        except Exception as e:
            raise CheckpointError(f"Failed to checkpoint for task '{task.title}': {e}") from e

    def _save_snapshot(self, task: Task, git_ref: str) -> None:
        """Save a context snapshot to the database for crash recovery."""
        snapshot = ContextSnapshot(
            task_id=task.id,
            attempt_number=task.attempt_count + 1,
            git_ref=git_ref or "no-checkpoint",
        )
        self.repository.save_context_snapshot(snapshot)
        logger.debug("Context snapshot saved for task '%s'", task.title)

    def rewind_to_checkpoint(self, task: Task, checkpoint_ref: str | None = None) -> bool:
        """Restore working tree from checkpoint after a failed build.

        Called by the orchestrator loop when Builder's tests fail.

        Args:
            task: The task being worked on.
            checkpoint_ref: Specific stash reference, or None for latest.

        Returns:
            True if rewind succeeded, False if nothing to rewind.
        """
        from src.tools.git_ops import rewind

        if not is_git_repo(self.repo_path):
            logger.warning("Path '%s' is not a git repo — skipping rewind", self.repo_path)
            return False

        try:
            success = rewind(self.repo_path, checkpoint_ref)
            if success:
                logger.info("Rewound to checkpoint for task '%s'", task.title)
            else:
                logger.info("No checkpoint to rewind for task '%s'", task.title)
            return success
        except Exception as e:
            raise CheckpointError(
                f"Failed to rewind for task '{task.title}': {e}"
            ) from e
