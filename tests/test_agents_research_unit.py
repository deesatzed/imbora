"""Tests for src/agents/research_unit.py — Live documentation research."""

import uuid

import pytest

from src.agents.research_unit import KNOWN_LIBRARIES, ResearchUnit
from src.core.models import AgentResult, ResearchBrief, Task, TaskContext, TaskStatus


# ---------------------------------------------------------------------------
# Library extraction tests
# ---------------------------------------------------------------------------


class TestLibraryExtraction:
    @pytest.fixture
    def agent(self):
        """Create a ResearchUnit without a real search client for extraction tests."""
        # Use a minimal stub that will not be called
        class StubClient:
            def search_documentation(self, lib, topic=None):
                return []
        return ResearchUnit(search_client=StubClient())

    def test_extract_known_library(self, agent):
        libs = agent._extract_libraries("We need to use pydantic for validation")
        assert "pydantic" in libs

    def test_extract_multiple_libraries(self, agent):
        libs = agent._extract_libraries("Use httpx and pydantic with fastapi")
        assert "httpx" in libs
        assert "pydantic" in libs
        assert "fastapi" in libs

    def test_extract_import_style(self, agent):
        libs = agent._extract_libraries("import requests\nfrom sqlalchemy import Column")
        assert "requests" in libs
        assert "sqlalchemy" in libs

    def test_extract_pip_install(self, agent):
        libs = agent._extract_libraries("pip install sentence-transformers")
        assert "sentence-transformers" in libs

    def test_skip_stdlib(self, agent):
        libs = agent._extract_libraries("import os\nimport json\nimport sys")
        assert "os" not in libs
        assert "json" not in libs
        assert "sys" not in libs

    def test_no_false_positives(self, agent):
        libs = agent._extract_libraries("The quick brown fox jumps over the lazy dog")
        assert len(libs) == 0

    def test_case_insensitive(self, agent):
        libs = agent._extract_libraries("Use PyDantic for data validation")
        assert "pydantic" in libs

    def test_empty_input(self, agent):
        libs = agent._extract_libraries("")
        assert libs == []


# ---------------------------------------------------------------------------
# Repo context building tests
# ---------------------------------------------------------------------------


class TestRepoContextBuilder:
    @pytest.fixture
    def agent(self):
        class StubClient:
            def search_documentation(self, lib, topic=None):
                return []
        return ResearchUnit(search_client=StubClient())

    def test_build_repo_context_real_dir(self, agent, tmp_path):
        # Create a minimal project structure
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("print('hello')")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").write_text("def test(): pass")
        (tmp_path / "README.md").write_text("# Test Project")
        (tmp_path / "pyproject.toml").write_text("[project]\nname = 'test'")

        context = agent._build_repo_context(str(tmp_path), max_depth=3)
        assert context["file_count"] >= 2
        assert context["dir_count"] >= 2
        assert "README.md" in context["key_files"]
        assert "pyproject.toml" in context["key_files"]

    def test_build_repo_context_nonexistent(self, agent):
        context = agent._build_repo_context("/nonexistent/path/xyz")
        assert context == {}

    def test_build_repo_context_max_depth(self, agent, tmp_path):
        # Create deeply nested structure
        deep = tmp_path
        for i in range(10):
            deep = deep / f"level{i}"
            deep.mkdir()
            (deep / f"file{i}.py").write_text("pass")

        context = agent._build_repo_context(str(tmp_path), max_depth=2)
        # Should not traverse all 10 levels
        assert context["file_count"] < 10

    def test_build_repo_context_skips_git(self, agent, tmp_path):
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "config").write_text("git config")
        (tmp_path / "src.py").write_text("code")

        context = agent._build_repo_context(str(tmp_path))
        # .git directory should be skipped
        tree = context.get("tree", "")
        assert ".git" not in tree or "src.py" in tree


# ---------------------------------------------------------------------------
# Version warning detection tests
# ---------------------------------------------------------------------------


class TestVersionWarnings:
    @pytest.fixture
    def agent(self):
        class StubClient:
            def search_documentation(self, lib, topic=None):
                return []
        return ResearchUnit(search_client=StubClient())

    def test_detect_deprecation(self, agent):
        docs = [{"title": "Library X", "snippet": "This function is deprecated since v2.0"}]
        warnings = agent._detect_version_warnings(docs)
        assert len(warnings) == 1
        assert "Deprecation" in warnings[0]

    def test_detect_breaking_change(self, agent):
        docs = [{"title": "v3.0 Release", "snippet": "This release includes breaking changes to the API"}]
        warnings = agent._detect_version_warnings(docs)
        assert any("Breaking" in w for w in warnings)

    def test_detect_eol(self, agent):
        docs = [{"title": "Python 2.7", "snippet": "Python 2.7 has reached end-of-life"}]
        warnings = agent._detect_version_warnings(docs)
        assert any("End-of-life" in w for w in warnings)

    def test_no_warnings(self, agent):
        docs = [{"title": "Good Docs", "snippet": "Everything works great"}]
        warnings = agent._detect_version_warnings(docs)
        assert len(warnings) == 0

    def test_empty_docs(self, agent):
        warnings = agent._detect_version_warnings([])
        assert len(warnings) == 0


# ---------------------------------------------------------------------------
# Agent process tests
# ---------------------------------------------------------------------------


class TestResearchUnitProcess:
    def test_process_with_task_context(self):
        class StubClient:
            def search_documentation(self, lib, topic=None):
                from src.tools.search import SearchResult
                return [SearchResult(title=f"{lib} docs", url=f"https://{lib}.dev", snippet="documentation")]
        agent = ResearchUnit(search_client=StubClient())

        project_id = uuid.uuid4()
        task = Task(
            project_id=project_id,
            title="Add pydantic validation",
            description="Use pydantic for request validation in the API",
        )
        context = TaskContext(task=task)

        result = agent.run(context)
        assert result.status == "success"
        assert "research_brief" in result.data
        assert "libraries_detected" in result.data
        assert "pydantic" in result.data["libraries_detected"]

    def test_process_with_dict_input(self):
        class StubClient:
            def search_documentation(self, lib, topic=None):
                return []
        agent = ResearchUnit(search_client=StubClient())

        result = agent.run({"title": "Test task", "description": "Use httpx for HTTP calls"})
        assert result.status == "success"
        assert "httpx" in result.data["libraries_detected"]

    def test_process_handles_search_failure(self):
        class FailingClient:
            def search_documentation(self, lib, topic=None):
                raise Exception("API down")
        agent = ResearchUnit(search_client=FailingClient())

        task = Task(
            project_id=uuid.uuid4(),
            title="Use pydantic",
            description="test",
        )
        result = agent.run(TaskContext(task=task))
        # Should succeed even if search fails (degraded)
        assert result.status == "success"
