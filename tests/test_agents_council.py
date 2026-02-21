"""Tests for src/agents/council.py — Council agent (multi-model peer review)."""

import json
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agents.council import Council
from src.core.config import PromptLoader
from src.core.models import CouncilDiagnosis, Task
from src.llm.client import LLMMessage, LLMResponse


class TestCouncilPromptComposition:
    """Test Council prompt building logic without LLM calls."""

    @pytest.fixture
    def council(self):
        c = Council.__new__(Council)
        c.name = "Council"
        c.role = "council"
        c._prompt_loader = PromptLoader()
        return c

    @pytest.fixture
    def task(self):
        return Task(
            project_id=uuid.uuid4(),
            title="Fix auth bug",
            description="JWT tokens expire incorrectly",
        )

    def test_extract_inputs_from_dict(self, council, task):
        input_data = {
            "task": task,
            "hypothesis_log": [
                {"approach_summary": "Tried extending expiry", "error_signature": "token_expired"},
            ],
            "project_rules": "No new dependencies",
        }
        task_desc, attempts, rules = council._extract_inputs(input_data)
        assert "Fix auth bug" in task_desc
        assert "JWT tokens" in task_desc
        assert len(attempts) == 1
        assert rules == "No new dependencies"

    def test_extract_inputs_task_as_dict(self, council):
        input_data = {
            "task": {"title": "Test task", "description": "A description"},
            "hypothesis_log": [],
        }
        task_desc, attempts, rules = council._extract_inputs(input_data)
        assert "Test task" in task_desc
        assert rules is None

    def test_extract_inputs_invalid_type(self, council):
        with pytest.raises(ValueError, match="unexpected input type"):
            council._extract_inputs("not a dict")

    def test_compose_prompt_includes_task(self, council, task):
        prompt = council._compose_prompt(
            "Fix auth bug: JWT tokens expire incorrectly",
            [], None,
        )
        assert "Fix auth bug" in prompt
        assert "JWT tokens" in prompt

    def test_compose_prompt_includes_attempts(self, council):
        attempts = [
            {
                "approach_summary": "Extended token lifetime to 24h",
                "error_signature": "token_expired",
                "model_used": "minimax/minimax-m2.5",
            },
            {
                "approach_summary": "Added refresh token flow",
                "error_full": "TypeError: refresh_token is not callable",
                "model_used": "minimax/minimax-m2.5",
            },
        ]
        prompt = council._compose_prompt("Fix auth", attempts, None)
        assert "Attempt 1" in prompt
        assert "Extended token lifetime" in prompt
        assert "Attempt 2" in prompt
        assert "refresh_token" in prompt
        assert "minimax" in prompt

    def test_compose_prompt_includes_rules(self, council):
        prompt = council._compose_prompt("Task", [], "No new dependencies allowed")
        assert "No new dependencies allowed" in prompt

    def test_compose_prompt_empty_attempts(self, council):
        prompt = council._compose_prompt("Task description", [], None)
        assert "Task description" in prompt
        assert "Failed Attempts" not in prompt


class TestCouncilResponseParsing:
    """Test JSON response parsing without LLM calls."""

    @pytest.fixture
    def council(self):
        c = Council.__new__(Council)
        c.name = "Council"
        c.role = "council"
        c._prompt_loader = PromptLoader()
        return c

    def test_parse_valid_json(self, council):
        response = json.dumps({
            "root_cause_analysis": "Token signing key rotates but cache doesn't invalidate",
            "failed_approaches_assessment": "Both attempts focused on expiry, not key rotation",
            "recommended_strategy": "Implement key rotation listener that clears token cache",
            "decomposition": ["Add key rotation event listener", "Clear cache on event"],
            "risk_factors": ["May affect other services using same key"],
            "confidence": "high",
        })
        result = council._parse_response(response, "z-ai/glm-5")
        assert result is not None
        assert isinstance(result, CouncilDiagnosis)
        assert "key rotates" in result.strategy_shift
        assert "key rotation listener" in result.new_approach
        assert "z-ai/glm-5" == result.model_used
        assert "Confidence: high" in result.reasoning

    def test_parse_json_in_code_fence(self, council):
        response = '```json\n{"root_cause_analysis": "test", "recommended_strategy": "use X", "failed_approaches_assessment": "bad"}\n```'
        result = council._parse_response(response, "test/model")
        assert result is not None
        assert result.new_approach == "use X"

    def test_parse_missing_strategy(self, council):
        response = json.dumps({
            "root_cause_analysis": "test",
            "failed_approaches_assessment": "bad",
        })
        result = council._parse_response(response, "test/model")
        assert result is None

    def test_parse_invalid_json(self, council):
        response = "This is not valid JSON at all."
        result = council._parse_response(response, "test/model")
        assert result is None

    def test_parse_includes_decomposition_in_reasoning(self, council):
        response = json.dumps({
            "root_cause_analysis": "Root cause",
            "recommended_strategy": "New strategy",
            "failed_approaches_assessment": "Assessment",
            "decomposition": ["Step 1", "Step 2"],
            "risk_factors": ["Risk A"],
            "confidence": "medium",
        })
        result = council._parse_response(response, "model/id")
        assert "Step 1; Step 2" in result.reasoning
        assert "Risk A" in result.reasoning
        assert "Confidence: medium" in result.reasoning


class TestCouncilSystemPrompt:
    """Test system prompt loading."""

    def test_loads_from_file(self, tmp_path):
        prompt_file = tmp_path / "council_system.txt"
        prompt_file.write_text("Custom council prompt.")
        loader = PromptLoader(prompts_dir=tmp_path)

        council = Council.__new__(Council)
        council.name = "Council"
        council.role = "council"
        council._prompt_loader = loader

        assert council._get_system_prompt() == "Custom council prompt."

    def test_fallback_default(self, tmp_path):
        loader = PromptLoader(prompts_dir=tmp_path)

        council = Council.__new__(Council)
        council.name = "Council"
        council.role = "council"
        council._prompt_loader = loader

        prompt = council._get_system_prompt()
        assert "diagnostician" in prompt
        assert "JSON" in prompt

    def test_loads_from_project_config(self):
        """Verify the actual config/prompts/council_system.txt is loadable."""
        prompts_dir = Path(__file__).parent.parent / "config" / "prompts"
        if not (prompts_dir / "council_system.txt").exists():
            pytest.skip("council_system.txt not found")
        loader = PromptLoader(prompts_dir=prompts_dir)

        council = Council.__new__(Council)
        council.name = "Council"
        council.role = "council"
        council._prompt_loader = loader

        prompt = council._get_system_prompt()
        assert "diagnostician" in prompt
        assert "root_cause_analysis" in prompt


class TestCouncilProcessIntegration:
    """Test the full process() method with a mocked LLM client."""

    def test_successful_diagnosis(self):
        """Council produces a valid CouncilDiagnosis from LLM response."""
        # Build a valid JSON response
        llm_response_content = json.dumps({
            "root_cause_analysis": "The auth module ignores key rotation events",
            "failed_approaches_assessment": "Both attempts only adjusted token lifetime",
            "recommended_strategy": "Hook into key rotation event and invalidate cache",
            "decomposition": ["Listen for rotation", "Flush cache"],
            "risk_factors": ["Cache flush may cause brief auth disruption"],
            "confidence": "high",
        })

        llm_client = MagicMock()
        llm_client.complete.return_value = LLMResponse(
            content=llm_response_content,
            model="z-ai/glm-5",
            tokens_used=500,
        )

        model_router = MagicMock()
        model_router.get_model.return_value = "z-ai/glm-5"

        council = Council(
            llm_client=llm_client,
            model_router=model_router,
        )

        input_data = {
            "task": {"title": "Fix auth", "description": "Tokens expire wrong"},
            "hypothesis_log": [
                {"approach_summary": "Extended expiry", "error_signature": "expired"},
            ],
        }

        result = council.process(input_data)
        assert result.status == "success"
        assert "council_diagnosis" in result.data
        diag = CouncilDiagnosis(**result.data["council_diagnosis"])
        assert "key rotation" in diag.new_approach
        assert diag.model_used == "z-ai/glm-5"
        assert result.data["tokens_used"] == 500

    def test_unparseable_response(self):
        """Council returns failure when LLM produces non-JSON."""
        llm_client = MagicMock()
        llm_client.complete.return_value = LLMResponse(
            content="I'm not sure, maybe try again?",
            model="z-ai/glm-5",
            tokens_used=100,
        )

        model_router = MagicMock()
        model_router.get_model.return_value = "z-ai/glm-5"

        council = Council(
            llm_client=llm_client,
            model_router=model_router,
        )

        input_data = {
            "task": {"title": "Fix auth", "description": "Broken"},
            "hypothesis_log": [],
        }

        result = council.process(input_data)
        assert result.status == "failure"
        assert "could not be parsed" in result.error
