"""Council agent for The Associate — multi-model peer review.

Sixth agent in the pipeline. Triggered when the Builder fails
`council_trigger_threshold` consecutive times. Calls a DIFFERENT model
family than the Builder (enforced by ModelRouter) to get a fresh perspective
on why the task is failing and what strategy to try next.

Adapted from:
- Ultrathink (coordinator + specialists pattern)
- DRIFTX (adversarial self-interrogation)
- Architect (conditional outcomes with scope guardrails)
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from src.agents.base_agent import BaseAgent
from src.core.config import PromptLoader
from src.core.models import (
    AgentResult,
    CouncilDiagnosis,
    HypothesisEntry,
    Task,
)
from src.llm.client import LLMMessage, OpenRouterClient
from src.llm.response_parser import extract_json_block
from src.llm.router import ModelRouter

logger = logging.getLogger("associate.agent.council")


class Council(BaseAgent):
    """Multi-model peer review agent.

    Injected dependencies:
        llm_client: OpenRouter client for LLM calls.
        model_router: Resolves the 'council' role to a model ID.
        prompt_loader: Loads system prompt from config/prompts/.
    """

    _DEFAULT_SYSTEM_PROMPT = (
        "You are a senior engineering diagnostician. A coding task has failed "
        "multiple times. Diagnose why and recommend a different strategy. "
        "Respond with a JSON object containing: root_cause_analysis, "
        "failed_approaches_assessment, recommended_strategy, decomposition, "
        "risk_factors, confidence."
    )

    def __init__(
        self,
        llm_client: OpenRouterClient,
        model_router: ModelRouter,
        prompt_loader: PromptLoader | None = None,
    ):
        super().__init__(name="Council", role="council")
        self.llm_client = llm_client
        self.model_router = model_router
        self._prompt_loader = prompt_loader or PromptLoader()

    def process(self, input_data: Any) -> AgentResult:
        """Diagnose why a task is failing and recommend a new strategy.

        Args:
            input_data: Dictionary containing:
                - "task": Task model or dict
                - "hypothesis_log": List of HypothesisEntry dicts (all attempts)
                - "project_rules": Optional string with project constraints

        Returns:
            AgentResult with CouncilDiagnosis in data["council_diagnosis"].
        """
        task_desc, attempts, project_rules = self._extract_inputs(input_data)

        # Compose the user prompt
        user_prompt = self._compose_prompt(task_desc, attempts, project_rules)

        # Get the model for the council role
        model_id = self.model_router.get_model(self.role)
        logger.info("Council calling model '%s' for diagnosis", model_id)

        # Call the LLM
        system_prompt = self._get_system_prompt()
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        response = self.llm_client.complete(messages, model=model_id)
        tokens_used = response.tokens_used

        # Parse the JSON response
        diagnosis = self._parse_response(response.content, model_id)

        if diagnosis is None:
            return AgentResult(
                agent_name=self.name,
                status="failure",
                data={"tokens_used": tokens_used},
                error="Council LLM response could not be parsed as valid JSON diagnosis",
            )

        return AgentResult(
            agent_name=self.name,
            status="success",
            data={
                "council_diagnosis": diagnosis.model_dump(),
                "tokens_used": tokens_used,
            },
        )

    def _get_system_prompt(self) -> str:
        """Load the system prompt from config/prompts/ or fall back to default."""
        return self._prompt_loader.load(
            "council_system.txt", default=self._DEFAULT_SYSTEM_PROMPT
        )

    def _extract_inputs(
        self, input_data: Any
    ) -> tuple[str, list[dict[str, Any]], Optional[str]]:
        """Extract task description, hypothesis log, and project rules."""
        if not isinstance(input_data, dict):
            raise ValueError(f"Council received unexpected input type: {type(input_data)}")

        # Task description
        task = input_data.get("task")
        if isinstance(task, Task):
            task_desc = f"{task.title}: {task.description}"
        elif isinstance(task, dict):
            task_desc = f"{task.get('title', '')}: {task.get('description', '')}"
        else:
            task_desc = str(task) if task else "Unknown task"

        # Hypothesis log entries
        attempts = input_data.get("hypothesis_log", [])

        # Project rules
        project_rules = input_data.get("project_rules")

        return task_desc, attempts, project_rules

    def _compose_prompt(
        self,
        task_desc: str,
        attempts: list[dict[str, Any]],
        project_rules: Optional[str],
    ) -> str:
        """Build the user prompt for the Council LLM call."""
        parts = []

        parts.append(f"## Task\n{task_desc}")

        if attempts:
            parts.append("\n## Failed Attempts")
            for i, attempt in enumerate(attempts, 1):
                summary = attempt.get("approach_summary", "Unknown approach")
                error = attempt.get("error_full", attempt.get("error_signature", "No error info"))
                model = attempt.get("model_used", "unknown model")
                parts.append(f"\n### Attempt {i} (model: {model})")
                parts.append(f"Approach: {summary}")
                if error:
                    parts.append(f"Error: {str(error)[:500]}")

        if project_rules:
            parts.append(f"\n## Project Constraints\n{project_rules}")

        parts.append(
            "\n## Your Task\n"
            "Analyze why these approaches failed and recommend a fundamentally "
            "different strategy. Respond with the JSON format specified in your "
            "system instructions."
        )

        return "\n".join(parts)

    def _parse_response(
        self, response_content: str, model_id: str
    ) -> Optional[CouncilDiagnosis]:
        """Parse the LLM response into a CouncilDiagnosis.

        Tries to extract JSON from the response. Maps the JSON fields
        to CouncilDiagnosis model fields.
        """
        parsed = extract_json_block(response_content)
        if parsed is None:
            logger.warning("Council response could not be parsed as JSON")
            return None

        # Map JSON keys to CouncilDiagnosis fields
        strategy_shift = parsed.get("root_cause_analysis", "")
        new_approach = parsed.get("recommended_strategy", "")
        reasoning = parsed.get("failed_approaches_assessment", "")

        # Include decomposition and risk factors in reasoning if present
        decomposition = parsed.get("decomposition", [])
        risk_factors = parsed.get("risk_factors", [])
        confidence = parsed.get("confidence", "low")

        full_reasoning = reasoning
        if decomposition:
            full_reasoning += f"\nDecomposition: {'; '.join(decomposition)}"
        if risk_factors:
            full_reasoning += f"\nRisks: {'; '.join(risk_factors)}"
        full_reasoning += f"\nConfidence: {confidence}"

        if not new_approach:
            logger.warning("Council response missing 'recommended_strategy'")
            return None

        return CouncilDiagnosis(
            strategy_shift=strategy_shift,
            new_approach=new_approach,
            reasoning=full_reasoning,
            model_used=model_id,
        )
