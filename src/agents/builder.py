"""Builder agent for The Associate.

Composes an LLM prompt with full task context (forbidden approaches,
live docs, past solutions, council diagnosis), calls the LLM to generate
code, writes the output to files, and runs the test suite.

Returns a BuildResult indicating whether tests passed and what changed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.agents.base_agent import BaseAgent
from src.core.config import PromptLoader
from src.core.models import (
    AgentResult,
    BuildResult,
    ContextBrief,
    TaskContext,
)
from src.llm.client import LLMMessage, LLMResponse, OpenRouterClient
from src.llm.router import ModelRouter
from src.security.policy import SecurityPolicy
from src.tools.file_ops import write_file
from src.tools.git_ops import get_full_diff
from src.tools.shell import ShellResult, run_command

logger = logging.getLogger("associate.agent.builder")


class Builder(BaseAgent):
    """Code generation + test execution agent.

    Injected dependencies:
        llm_client: OpenRouter client for LLM calls.
        model_router: Resolves the builder role to a model ID.
        repo_path: Path to the project git repository.
        test_command: Shell command to run project tests (e.g., "pytest tests/").
    """

    # Hardcoded fallback if config/prompts/builder_system.txt doesn't exist
    _DEFAULT_SYSTEM_PROMPT = (
        "You are a code generation agent in an automated software engineering pipeline. "
        "Write clean, complete code with no placeholders, TODOs, or stubs. "
        "Every function must be fully implemented. "
        "Structure your output with: ## Approach, ## Edge Cases Considered, "
        "## Self-Audit, then ## Files using --- FILE: path --- blocks."
    )

    def __init__(
        self,
        llm_client: OpenRouterClient,
        model_router: ModelRouter,
        repo_path: str,
        test_command: str = "pytest tests/ -v",
        prompt_loader: PromptLoader | None = None,
        security_policy: SecurityPolicy | None = None,
    ):
        super().__init__(name="Builder", role="builder")
        self.llm_client = llm_client
        self.model_router = model_router
        self.repo_path = repo_path
        self.test_command = test_command
        self._prompt_loader = prompt_loader or PromptLoader()
        self.security_policy = security_policy

    def process(self, input_data: Any) -> AgentResult:
        """Generate code from context and run tests.

        Args:
            input_data: A TaskContext or ContextBrief with all the context
                        the Builder needs.

        Returns:
            AgentResult with BuildResult in data["build_result"].
        """
        # Extract context depending on input type
        context = self._normalize_input(input_data)

        # 1. Compose the LLM prompt
        prompt = self._compose_prompt(context)

        # 2. Call the LLM
        model_chain = self.model_router.get_model_chain(self.role)
        logger.info("Calling LLM models %s for task '%s'", model_chain, context["title"])

        response = self._call_llm(prompt, model_chain)
        model_used = response.model or model_chain[0]
        tokens_used = response.tokens_used

        # 3. Parse and write code output
        files_changed = self._write_code(response.content, context)

        # Extract structured sections from LLM output
        approach_summary = self._extract_approach_summary(response.content)
        self_audit = self._extract_self_audit(response.content)

        # 3b. Validate that files were actually produced
        if not files_changed:
            failure_text = (
                "LLM output contained no parseable file blocks - no code was written"
            )
            build_result = BuildResult(
                approach_summary=approach_summary,
                model_used=model_used,
                failure_reason="no_parseable_file_blocks",
                failure_detail=failure_text,
            )
            return AgentResult(
                agent_name=self.name,
                status="failure",
                data={
                    "build_result": build_result.model_dump(),
                    "tokens_used": tokens_used,
                    "self_audit": self_audit,
                },
                error=failure_text,
            )

        # 4. Run tests
        test_result = self._run_tests()

        # 5. Get the diff
        diff = get_full_diff(self.repo_path)

        # 6. Build the result
        failure_detail = None
        failure_reason = None
        if not test_result.success:
            failure_reason = "tests_failed"
            failure_detail = f"Tests failed (rc={test_result.return_code})"

        build_result = BuildResult(
            files_changed=files_changed,
            test_output=test_result.stdout + test_result.stderr,
            tests_passed=test_result.success,
            diff=diff,
            approach_summary=approach_summary,
            model_used=model_used,
            failure_reason=failure_reason,
            failure_detail=failure_detail,
        )

        status = "success" if test_result.success else "failure"
        error = build_result.failure_detail

        return AgentResult(
            agent_name=self.name,
            status=status,
            data={
                "build_result": build_result.model_dump(),
                "tokens_used": tokens_used,
                "self_audit": self_audit,
            },
            error=error,
        )

    def _normalize_input(self, input_data: Any) -> dict[str, Any]:
        """Extract a standardized context dict from various input types."""
        if isinstance(input_data, ContextBrief):
            return {
                "title": input_data.task.title,
                "description": input_data.task.description,
                "forbidden_approaches": input_data.forbidden_approaches,
                "past_solutions": [m.model_dump() for m in input_data.past_solutions],
                "live_docs": input_data.live_docs,
                "project_rules": input_data.project_rules,
                "council_diagnosis": input_data.council_diagnosis,
                "retrieval_confidence": input_data.retrieval_confidence,
                "retrieval_conflicts": input_data.retrieval_conflicts,
                "retrieval_strategy_hint": input_data.retrieval_strategy_hint,
                "skill_procedure": input_data.skill_procedure,
                "complexity_tier": input_data.complexity_tier,
            }
        elif isinstance(input_data, TaskContext):
            return {
                "title": input_data.task.title,
                "description": input_data.task.description,
                "forbidden_approaches": input_data.forbidden_approaches,
                "past_solutions": [],
                "live_docs": [],
                "project_rules": None,
                "council_diagnosis": input_data.previous_council_diagnosis,
                "retrieval_confidence": 0.0,
                "retrieval_conflicts": [],
                "retrieval_strategy_hint": None,
            }
        elif isinstance(input_data, dict):
            return input_data
        else:
            raise ValueError(f"Builder received unexpected input type: {type(input_data)}")

    def _compose_prompt(self, context: dict[str, Any]) -> str:
        """Build the full LLM prompt from task context.

        Produces a structured prompt that enforces the output format
        defined in the system message (## Approach, ## Edge Cases Considered,
        ## Self-Audit, ## Files).
        """
        parts = []

        # Task description
        parts.append(f"## Task: {context['title']}\n\n{context['description']}")

        # Forbidden approaches — now with error context
        forbidden = context.get("forbidden_approaches", [])
        if forbidden:
            parts.append("\n## Forbidden Approaches (DO NOT USE THESE — they already failed)")
            for i, approach in enumerate(forbidden, 1):
                if isinstance(approach, dict):
                    summary = approach.get("approach", approach.get("summary", str(approach)))
                    error = approach.get("error_signature", approach.get("error", ""))
                    parts.append(f"{i}. {summary}")
                    if error:
                        parts.append(f"   Error: {error}")
                else:
                    parts.append(f"{i}. {approach}")

        # Council diagnosis
        diagnosis = context.get("council_diagnosis")
        if diagnosis:
            parts.append(f"\n## Council Diagnosis\n{diagnosis}")

        retrieval_hint = context.get("retrieval_strategy_hint")
        retrieval_confidence = float(context.get("retrieval_confidence", 0.0) or 0.0)
        retrieval_conflicts = context.get("retrieval_conflicts", []) or []
        if retrieval_hint or retrieval_confidence > 0 or retrieval_conflicts:
            parts.append("\n## Memory Confidence Signals")
            parts.append(f"- Retrieval confidence: {retrieval_confidence:.2f}")
            if retrieval_conflicts:
                parts.append("- Recall conflicts:")
                for conflict in retrieval_conflicts[:3]:
                    parts.append(f"  - {conflict}")
            if retrieval_hint:
                parts.append(f"- Strategy hint: {retrieval_hint}")

        # Past solutions
        past = context.get("past_solutions", [])
        if past:
            parts.append("\n## Similar Past Solutions (for reference)")
            for sol in past[:3]:  # Limit to top 3
                desc = sol.get("problem_description", "")
                code = sol.get("solution_code", "")
                parts.append(f"### {desc[:100]}\n```\n{code[:500]}\n```")

        # Live documentation
        docs = context.get("live_docs", [])
        if docs:
            parts.append("\n## Live Documentation")
            for doc in docs[:3]:
                title = doc.get("title", "")
                snippet = doc.get("snippet", "")
                parts.append(f"### {title}\n{snippet}")

        # Project rules
        rules = context.get("project_rules")
        if rules:
            parts.append(f"\n## Project Rules\n{rules}")

        # Skill procedure (Item 13)
        skill = context.get("skill_procedure")
        if skill:
            parts.append(f"\n## Matched Skill Procedure\nFollow this procedure:\n{skill}")

        # Complexity tier (Item 7)
        complexity = context.get("complexity_tier")
        if complexity:
            parts.append(f"\n## Task Complexity: {complexity}")

        # Budget-aware guidance (Item 8)
        budget_hint = context.get("budget_hint")
        if budget_hint:
            parts.append(f"\n## Budget Guidance\n{budget_hint}")

        # Output format instructions
        parts.append(
            "\n## Required Output Format\n"
            "Structure your response with these sections in order:\n\n"
            "## Approach\n"
            "(2-3 sentences: what you will do and why)\n\n"
            "## Edge Cases Considered\n"
            "(bulleted list of edge cases handled)\n\n"
            "## Self-Audit\n"
            "(YES/NO for each: fully implemented? error handling? "
            "follows style? no hardcoded secrets? no new deps? imports resolve?)\n\n"
            "## Files\n"
            "For each file use:\n"
            "--- FILE: path/to/file.py ---\n"
            "```python\n"
            "# complete file contents\n"
            "```\n\n"
            "If you are not blocked, include at least one FILE block.\n"
            "If you cannot fully solve this task, explain the blocking issue in "
            "## Approach instead of producing partial code."
        )

        return "\n".join(parts)

    def _get_system_prompt(self) -> str:
        """Load the system prompt from config/prompts/ or fall back to default."""
        return self._prompt_loader.load(
            "builder_system.txt", default=self._DEFAULT_SYSTEM_PROMPT
        )

    def _call_llm(self, prompt: str, model_ids: list[str]) -> LLMResponse:
        """Call the LLM with the composed prompt."""
        system_prompt = self._get_system_prompt()
        messages = [
            LLMMessage(role="system", content=system_prompt),
            LLMMessage(role="user", content=prompt),
        ]
        return self.llm_client.complete_with_fallback(messages, models=model_ids)

    def _write_code(self, llm_output: str, context: dict[str, Any]) -> list[str]:
        """Parse LLM output and write files to disk.

        Expects format:
            --- FILE: path/to/file.py ---
            ```python
            code here
            ```

        Security: All paths are resolved and validated to stay within repo_path.
        """
        files_changed = []
        blocks = _parse_file_blocks(llm_output)
        repo_root = Path(self.repo_path).resolve()
        security_policy = getattr(self, "security_policy", None)

        for file_path, content in blocks:
            if security_policy and not security_policy.is_path_allowed(file_path):
                logger.warning("Path blocked by security policy: %s", file_path)
                continue

            # Resolve and sandbox — path must stay within repo_path
            full_path = (repo_root / file_path).resolve()
            try:
                full_path.relative_to(repo_root)
            except ValueError:
                logger.warning("Path traversal blocked: %s resolves outside repo", file_path)
                continue

            write_file(str(full_path), content, security_policy=security_policy)
            files_changed.append(file_path)
            logger.info("Wrote file: %s (%d bytes)", file_path, len(content))

        if not files_changed:
            logger.warning("LLM output contained no parseable file blocks")

        return files_changed

    def _run_tests(self) -> ShellResult:
        """Execute the project test suite."""
        logger.info("Running tests: %s", self.test_command)
        result = run_command(
            self.test_command,
            cwd=self.repo_path,
            timeout=300,
            security_policy=self.security_policy,
        )

        if result.success:
            logger.info("Tests PASSED")
        else:
            logger.warning("Tests FAILED (rc=%d)", result.return_code)
            if result.stderr:
                logger.debug("Test stderr: %s", result.stderr[:500])

        return result

    def _extract_approach_summary(self, llm_output: str) -> str:
        """Extract the approach summary from structured LLM output.

        Tries in order:
        1. Extract the ## Approach section content (R2 structured output)
        2. Fall back to first 3 non-code lines (legacy heuristic)
        3. Return generic fallback
        """
        return _extract_section(llm_output, "Approach", max_chars=300, fallback="Code generation")

    def _extract_self_audit(self, llm_output: str) -> str:
        """Extract the ## Self-Audit section from structured LLM output.

        Returns the self-audit text if found, empty string otherwise.
        Used by Sentinel for cross-checking claims (R8).
        """
        return _extract_section(llm_output, "Self-Audit", max_chars=500, fallback="")


def _extract_section(llm_output: str, section_name: str, max_chars: int = 300, fallback: str = "") -> str:
    """Extract content of a ## Section from structured LLM output.

    Looks for '## {section_name}' header and captures text until the next
    '## ' header or '--- FILE:' block.

    Falls back to the first 3 non-code lines if the section is not found.

    Args:
        llm_output: Full LLM response text.
        section_name: Name of the section (e.g. "Approach", "Self-Audit").
        max_chars: Maximum characters to return.
        fallback: Value to return if extraction fails.
    """
    import re

    # Try structured section extraction
    pattern = re.compile(
        rf"^##\s+{re.escape(section_name)}\s*\n(.*?)(?=^##\s|\Z|^---\s*FILE:)",
        re.MULTILINE | re.DOTALL,
    )
    match = pattern.search(llm_output)
    if match:
        content = match.group(1).strip()
        if content:
            return content[:max_chars]
        # Section header exists but content is empty — return fallback
        return fallback

    # Fallback: first 3 non-code, non-header lines (legacy heuristic)
    # Only applies when the structured section was not found at all.
    if section_name == "Approach":
        lines = llm_output.strip().split("\n")
        summary_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("--- FILE:") or stripped.startswith("```"):
                break
            if stripped and not stripped.startswith("#"):
                summary_lines.append(stripped)
            if len(summary_lines) >= 3:
                break
        if summary_lines:
            return " ".join(summary_lines)[:max_chars]

    return fallback


def _parse_file_blocks(llm_output: str) -> list[tuple[str, str]]:
    """Parse --- FILE: path --- blocks from LLM output.

    Returns:
        List of (file_path, content) tuples.
    """
    import re

    blocks: list[tuple[str, str]] = []
    # Match: --- FILE: some/path.py ---
    file_pattern = re.compile(r"^---\s*FILE:\s*(.+?)\s*---\s*$", re.MULTILINE)

    matches = list(file_pattern.finditer(llm_output))
    if not matches:
        return blocks

    for i, match in enumerate(matches):
        file_path = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(llm_output)
        raw_content = llm_output[start:end].strip()

        # Strip code fences if present
        content = _strip_code_fences(raw_content)
        if content:
            blocks.append((file_path, content))

    return blocks


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences from content."""
    import re
    # Match ```language\n...\n```
    fence_pattern = re.compile(r"^```\w*\n?(.*?)\n?```$", re.DOTALL)
    match = fence_pattern.match(text.strip())
    if match:
        return match.group(1).strip()
    return text.strip()
