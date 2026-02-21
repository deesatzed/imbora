"""Base class for all data science pipeline agents.

Extends BaseAgent with DS-specific helpers: LLM calls via ds_analyst/ds_evaluator
roles, experiment tracking via Repository, and config access.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.config import DataScienceConfig
from src.llm.client import LLMMessage

logger = logging.getLogger("associate.datasci")


class BaseDataScienceAgent(BaseAgent):
    """Base agent for the 7-phase data science pipeline.

    Provides:
    - _call_ds_llm(): call LLM via the ds_analyst or ds_evaluator role
    - _save_experiment(): persist experiment record to database
    - Access to DS config, LLM client, model router, repository
    """

    def __init__(
        self,
        name: str,
        role: str,
        llm_client: Any,
        model_router: Any,
        repository: Any,
        ds_config: DataScienceConfig,
    ):
        super().__init__(name=name, role=role)
        self.llm_client = llm_client
        self.model_router = model_router
        self.repository = repository
        self.ds_config = ds_config

    def _call_ds_llm(
        self,
        prompt: str,
        role: str = "ds_analyst",
        system_prompt: str = "",
        max_tokens: int = 4096,
        temperature: float = 0.3,
    ) -> str:
        """Call LLM via the DS-specific model role.

        Args:
            prompt: User message content.
            role: Model router role (ds_analyst or ds_evaluator).
            system_prompt: Optional system prompt override.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.

        Returns:
            LLM response text.
        """
        models = self.model_router.get_model_chain(role)

        messages = []
        if system_prompt:
            messages.append(LLMMessage(role="system", content=system_prompt))
        messages.append(LLMMessage(role="user", content=prompt))

        response = self.llm_client.complete_with_fallback(
            messages=messages,
            models=models,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.content

    def _save_experiment(
        self,
        project_id: uuid.UUID,
        phase: str,
        config: dict[str, Any] | None = None,
        task_id: uuid.UUID | None = None,
        run_id: uuid.UUID | None = None,
        dataset_fingerprint: str | None = None,
        parent_experiment_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        """Create a DS experiment record in the database."""
        return self.repository.create_ds_experiment(
            project_id=project_id,
            experiment_phase=phase,
            experiment_config=config,
            task_id=task_id,
            run_id=run_id,
            dataset_fingerprint=dataset_fingerprint,
            parent_experiment_id=parent_experiment_id,
        )

    def _complete_experiment(
        self,
        experiment_id: uuid.UUID,
        metrics: dict[str, Any] | None = None,
        artifacts_manifest: dict[str, Any] | None = None,
        status: str = "COMPLETED",
    ) -> None:
        """Mark a DS experiment as completed with metrics."""
        self.repository.update_ds_experiment(
            experiment_id=experiment_id,
            status=status,
            metrics=metrics,
            artifacts_manifest=artifacts_manifest,
        )
