"""Tests for src/agents/builder.py — Builder agent.

Tests the file block parsing and prompt composition logic directly.
Live LLM tests require OPENROUTER_API_KEY.
"""

import uuid

import pytest

from src.agents.builder import Builder, _extract_section, _parse_file_blocks, _strip_code_fences
from src.core.config import PromptLoader
from src.core.models import (
    BuildResult,
    ContextBrief,
    Methodology,
    Task,
    TaskContext,
)

from tests.conftest import requires_openrouter


class TestParseFileBlocks:
    def test_single_file(self):
        output = """Here's the code:

--- FILE: src/utils.py ---
```python
def hello():
    return "hello"
```
"""
        blocks = _parse_file_blocks(output)
        assert len(blocks) == 1
        assert blocks[0][0] == "src/utils.py"
        assert 'def hello():' in blocks[0][1]

    def test_multiple_files(self):
        output = """
--- FILE: src/models.py ---
```python
class User:
    pass
```

--- FILE: tests/test_models.py ---
```python
def test_user():
    u = User()
    assert u is not None
```
"""
        blocks = _parse_file_blocks(output)
        assert len(blocks) == 2
        assert blocks[0][0] == "src/models.py"
        assert blocks[1][0] == "tests/test_models.py"

    def test_no_file_blocks(self):
        output = "Just some text without any file blocks."
        blocks = _parse_file_blocks(output)
        assert blocks == []

    def test_file_with_whitespace_in_path(self):
        output = """
--- FILE:   src/my_module.py   ---
```python
x = 1
```
"""
        blocks = _parse_file_blocks(output)
        assert len(blocks) == 1
        assert blocks[0][0] == "src/my_module.py"


class TestStripCodeFences:
    def test_strips_python_fences(self):
        text = '```python\nprint("hello")\n```'
        assert _strip_code_fences(text) == 'print("hello")'

    def test_strips_plain_fences(self):
        text = '```\nsome code\n```'
        assert _strip_code_fences(text) == 'some code'

    def test_no_fences(self):
        text = 'plain text'
        assert _strip_code_fences(text) == 'plain text'

    def test_preserves_inner_content(self):
        text = '```python\ndef foo():\n    return 42\n```'
        result = _strip_code_fences(text)
        assert 'def foo()' in result
        assert 'return 42' in result


class TestBuilderPromptComposition:
    """Test prompt building logic without LLM calls."""

    @pytest.fixture
    def task(self):
        return Task(
            project_id=uuid.uuid4(),
            title="Add user auth",
            description="Implement JWT authentication",
        )

    def test_context_brief_normalization(self, task):
        brief = ContextBrief(
            task=task,
            forbidden_approaches=["regex parsing", "basic auth"],
            past_solutions=[],
            live_docs=[{"title": "JWT Guide", "snippet": "Use RS256 signing"}],
            project_rules="No wildcard imports",
            council_diagnosis="Try OAuth instead",
        )
        # Create builder with None llm/router (we won't call LLM)
        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"

        context = builder._normalize_input(brief)
        assert context["title"] == "Add user auth"
        assert len(context["forbidden_approaches"]) == 2
        assert context["project_rules"] == "No wildcard imports"
        assert context["council_diagnosis"] == "Try OAuth instead"

    def test_task_context_normalization(self, task):
        tc = TaskContext(
            task=task,
            forbidden_approaches=["approach1"],
            checkpoint_ref="stash@{0}",
            previous_council_diagnosis="diagnosis here",
        )
        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"

        context = builder._normalize_input(tc)
        assert context["title"] == "Add user auth"
        assert context["council_diagnosis"] == "diagnosis here"

    def test_compose_prompt_includes_task(self, task):
        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"

        context = {
            "title": "Add user auth",
            "description": "Implement JWT authentication",
            "forbidden_approaches": [],
            "past_solutions": [],
            "live_docs": [],
            "project_rules": None,
            "council_diagnosis": None,
        }
        prompt = builder._compose_prompt(context)
        assert "Add user auth" in prompt
        assert "JWT authentication" in prompt

    def test_compose_prompt_includes_forbidden(self, task):
        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"

        context = {
            "title": "Test",
            "description": "Test",
            "forbidden_approaches": ["regex parsing", "basic auth"],
            "past_solutions": [],
            "live_docs": [],
            "project_rules": None,
            "council_diagnosis": None,
        }
        prompt = builder._compose_prompt(context)
        assert "Forbidden Approaches" in prompt
        assert "regex parsing" in prompt
        assert "basic auth" in prompt

    def test_compose_prompt_includes_live_docs(self, task):
        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"

        context = {
            "title": "Test",
            "description": "Test",
            "forbidden_approaches": [],
            "past_solutions": [],
            "live_docs": [{"title": "JWT Guide", "snippet": "Use RS256 signing"}],
            "project_rules": None,
            "council_diagnosis": None,
        }
        prompt = builder._compose_prompt(context)
        assert "Live Documentation" in prompt
        assert "JWT Guide" in prompt

    def test_compose_prompt_includes_memory_signals(self, task):
        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"

        context = {
            "title": "Test",
            "description": "Test",
            "forbidden_approaches": [],
            "past_solutions": [],
            "live_docs": [],
            "project_rules": None,
            "council_diagnosis": None,
            "retrieval_confidence": 0.81,
            "retrieval_conflicts": ["conflict A"],
            "retrieval_strategy_hint": "Use synthesis",
        }
        prompt = builder._compose_prompt(context)
        assert "Memory Confidence Signals" in prompt
        assert "0.81" in prompt
        assert "conflict A" in prompt
        assert "Use synthesis" in prompt

    def test_extract_approach_summary_legacy(self):
        """Legacy heuristic: first non-code lines when ## Approach missing."""
        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"

        output = "I'll use JWT with RS256 signing.\nThe approach is...\n--- FILE: src/auth.py ---"
        summary = builder._extract_approach_summary(output)
        assert "JWT" in summary
        assert "FILE:" not in summary

    def test_extract_approach_summary_structured(self):
        """Structured extraction from ## Approach section."""
        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"

        output = (
            "## Approach\n"
            "I will implement JWT auth using RS256 signing with a dedicated middleware.\n\n"
            "## Edge Cases Considered\n"
            "- Expired tokens\n\n"
            "## Files\n"
            "--- FILE: src/auth.py ---\n```python\nx=1\n```"
        )
        summary = builder._extract_approach_summary(output)
        assert "JWT auth" in summary
        assert "RS256" in summary
        assert "Edge Cases" not in summary

    def test_extract_approach_summary_empty(self):
        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"

        summary = builder._extract_approach_summary("--- FILE: x.py ---\n```\ncode\n```")
        assert summary == "Code generation"

    def test_extract_self_audit(self):
        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"

        output = (
            "## Approach\nDoing something.\n\n"
            "## Self-Audit\n"
            "- All functions fully implemented: YES\n"
            "- Error handling: YES\n"
            "- No hardcoded secrets: YES\n\n"
            "## Files\n--- FILE: x.py ---\n```\ncode\n```"
        )
        audit = builder._extract_self_audit(output)
        assert "fully implemented" in audit
        assert "YES" in audit

    def test_extract_self_audit_missing(self):
        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"

        output = "## Approach\nDoing something.\n\n--- FILE: x.py ---\n```\ncode\n```"
        audit = builder._extract_self_audit(output)
        assert audit == ""

    def test_compose_prompt_forbidden_with_error_context(self):
        """R2: Forbidden approaches now include error signatures."""
        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"

        context = {
            "title": "Test",
            "description": "Test",
            "forbidden_approaches": [
                {"approach": "regex parsing", "error_signature": "re.error: bad pattern"},
                "basic auth",  # Plain string still works
            ],
            "past_solutions": [],
            "live_docs": [],
            "project_rules": None,
            "council_diagnosis": None,
        }
        prompt = builder._compose_prompt(context)
        assert "regex parsing" in prompt
        assert "re.error: bad pattern" in prompt
        assert "basic auth" in prompt

    def test_compose_prompt_output_format_instructions(self):
        """R2: Prompt includes structured output format requirements."""
        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"

        context = {
            "title": "Test",
            "description": "Test",
            "forbidden_approaches": [],
            "past_solutions": [],
            "live_docs": [],
            "project_rules": None,
            "council_diagnosis": None,
        }
        prompt = builder._compose_prompt(context)
        assert "## Approach" in prompt
        assert "## Edge Cases Considered" in prompt
        assert "## Self-Audit" in prompt
        assert "## Files" in prompt
        assert "include at least one FILE block" in prompt

    def test_system_prompt_from_file(self, tmp_path):
        """R1: System prompt loaded from config/prompts/builder_system.txt."""
        prompt_file = tmp_path / "builder_system.txt"
        prompt_file.write_text("Custom system prompt for testing.")
        loader = PromptLoader(prompts_dir=tmp_path)

        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"
        builder._prompt_loader = loader

        system_prompt = builder._get_system_prompt()
        assert system_prompt == "Custom system prompt for testing."

    def test_system_prompt_fallback(self, tmp_path):
        """R1: Falls back to hardcoded default if file doesn't exist."""
        loader = PromptLoader(prompts_dir=tmp_path)

        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"
        builder._prompt_loader = loader

        system_prompt = builder._get_system_prompt()
        assert "code generation agent" in system_prompt
        assert "## Approach" in system_prompt

    def test_normalize_input_invalid_type(self):
        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"

        with pytest.raises(ValueError, match="unexpected input type"):
            builder._normalize_input(42)


class TestExtractSection:
    """Test the _extract_section helper used for structured output parsing."""

    def test_extracts_approach_section(self):
        output = "## Approach\nI will do X and Y.\n\n## Edge Cases Considered\n- None\n"
        result = _extract_section(output, "Approach")
        assert result == "I will do X and Y."

    def test_extracts_multiline_section(self):
        output = "## Self-Audit\n- Q1: YES\n- Q2: NO\n- Q3: YES\n\n## Files\n"
        result = _extract_section(output, "Self-Audit")
        assert "Q1: YES" in result
        assert "Q3: YES" in result

    def test_section_stops_at_next_header(self):
        output = "## Approach\nDoing X.\n\n## Edge Cases Considered\n- Edge 1\n"
        result = _extract_section(output, "Approach")
        assert "Doing X" in result
        assert "Edge" not in result

    def test_section_stops_at_file_block(self):
        output = "## Approach\nDoing X.\n\n--- FILE: x.py ---\ncode\n"
        result = _extract_section(output, "Approach")
        assert "Doing X" in result
        assert "FILE:" not in result

    def test_missing_section_returns_fallback(self):
        output = "Some text without sections."
        result = _extract_section(output, "Self-Audit", fallback="not found")
        assert result == "not found"

    def test_missing_approach_falls_back_to_heuristic(self):
        output = "I will use JWT.\nWith RS256.\n--- FILE: x.py ---\ncode\n"
        result = _extract_section(output, "Approach", fallback="Code generation")
        assert "JWT" in result
        assert "RS256" in result

    def test_max_chars_truncation(self):
        long_text = "## Approach\n" + "x" * 500 + "\n## Files\n"
        result = _extract_section(long_text, "Approach", max_chars=100)
        assert len(result) == 100

    def test_empty_section(self):
        output = "## Approach\n\n## Edge Cases\n- one\n"
        result = _extract_section(output, "Approach", fallback="empty")
        assert result == "empty"


class TestBuilderPathTraversal:
    """Verify path traversal protection in _write_code."""

    def test_blocks_parent_directory_traversal(self, tmp_path):
        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"
        builder.repo_path = str(tmp_path / "repo")
        (tmp_path / "repo").mkdir()

        # LLM output with path traversal attempt
        llm_output = '--- FILE: ../../etc/evil.txt ---\n```\nmalicious\n```\n'
        context = {"title": "test"}
        files = builder._write_code(llm_output, context)

        assert files == []
        assert not (tmp_path / "etc" / "evil.txt").exists()

    def test_allows_valid_paths(self, tmp_path):
        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"
        builder.repo_path = str(tmp_path)

        llm_output = '--- FILE: src/module.py ---\n```python\nx = 1\n```\n'
        context = {"title": "test"}
        files = builder._write_code(llm_output, context)

        assert files == ["src/module.py"]
        assert (tmp_path / "src" / "module.py").exists()

    def test_blocks_absolute_path_outside_repo(self, tmp_path):
        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"
        builder.repo_path = str(tmp_path / "repo")
        (tmp_path / "repo").mkdir()

        llm_output = '--- FILE: /tmp/evil.py ---\n```python\nx = 1\n```\n'
        context = {"title": "test"}
        files = builder._write_code(llm_output, context)

        assert files == []

    def test_blocks_prefix_collision_path(self, tmp_path):
        builder = Builder.__new__(Builder)
        builder.name = "Builder"
        builder.role = "builder"
        repo = tmp_path / "repo"
        repo.mkdir()
        sibling = tmp_path / "repo-evil"
        sibling.mkdir()
        target = sibling / "evil.py"

        builder.repo_path = str(repo)

        llm_output = f"--- FILE: {target} ---\n```python\nx = 1\n```\n"
        context = {"title": "test"}
        files = builder._write_code(llm_output, context)

        assert files == []
        assert not target.exists()
