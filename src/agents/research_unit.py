"""Research Unit agent for The Associate.

Second agent in the pipeline (after State Manager). Responsibilities:
1. Extract library/framework names from the task description
2. Query Tavily for live documentation of those libraries
3. Build a structured repo context snapshot (P12 from GrokFlow)
4. Return a ResearchBrief with live docs, API signatures, and version warnings

Counters the Knowledge Cutoff failure mode by injecting current documentation.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

from src.agents.base_agent import BaseAgent
from src.core.models import AgentResult, ResearchBrief, TaskContext
from src.tools.search import TavilyClient

logger = logging.getLogger("associate.agent.research_unit")

# Common Python libraries and frameworks that are worth searching for
KNOWN_LIBRARIES = {
    "psycopg", "psycopg2", "psycopg3", "asyncpg", "sqlalchemy",
    "pydantic", "fastapi", "flask", "django", "starlette",
    "httpx", "requests", "aiohttp", "urllib3",
    "pytest", "unittest", "hypothesis",
    "numpy", "pandas", "scipy", "scikit-learn", "sklearn",
    "torch", "pytorch", "tensorflow", "keras",
    "celery", "redis", "rabbitmq",
    "sentence-transformers", "transformers", "huggingface",
    "click", "typer", "argparse", "rich",
    "docker", "kubernetes", "terraform",
    "pgvector", "faiss", "chromadb", "pinecone",
    "openai", "anthropic", "langchain",
    "boto3", "aws", "gcloud", "azure",
    "alembic", "flyway",
}


class ResearchUnit(BaseAgent):
    """Live documentation research agent.

    Injected dependencies:
        search_client: Tavily API client for web search.
        repo_path: Optional path for repo context building.
    """

    def __init__(
        self,
        search_client: TavilyClient,
        repo_path: Optional[str] = None,
    ):
        super().__init__(name="ResearchUnit", role="research")
        self.search_client = search_client
        self.repo_path = repo_path

    def process(self, input_data: Any) -> AgentResult:
        """Research live documentation for the task's dependencies.

        Args:
            input_data: A TaskContext from State Manager.

        Returns:
            AgentResult with ResearchBrief in data["research_brief"].
        """
        task_context = self._extract_context(input_data)
        task_description = task_context.get("description", "")
        task_title = task_context.get("title", "")

        # 1. Extract library names from task
        libraries = self._extract_libraries(f"{task_title} {task_description}")
        logger.info("Detected libraries: %s", libraries)

        # 2. Search for live documentation
        live_docs = []
        for lib in libraries[:5]:  # Limit to avoid excessive API calls
            try:
                results = self.search_client.search_documentation(lib)
                for r in results[:2]:  # Top 2 per library
                    live_docs.append(r.to_dict())
            except Exception as e:
                logger.warning("Search failed for '%s': %s", lib, e)

        # 3. Build repo context (if repo_path is set)
        repo_context = {}
        if self.repo_path:
            repo_context = self._build_repo_context(self.repo_path)

        # 4. Extract version warnings from docs
        version_warnings = self._detect_version_warnings(live_docs)

        # 5. Build the brief
        brief = ResearchBrief(
            live_docs=live_docs,
            api_signatures=[],  # Populated by deeper analysis in future
            version_warnings=version_warnings,
        )

        return AgentResult(
            agent_name=self.name,
            status="success",
            data={
                "research_brief": brief.model_dump(),
                "libraries_detected": libraries,
                "repo_context": repo_context,
            },
        )

    def _extract_context(self, input_data: Any) -> dict[str, str]:
        """Extract task title and description from various input types."""
        if isinstance(input_data, TaskContext):
            return {
                "title": input_data.task.title,
                "description": input_data.task.description,
            }
        if isinstance(input_data, dict):
            task = input_data.get("task", input_data)
            if isinstance(task, dict):
                return {
                    "title": task.get("title", ""),
                    "description": task.get("description", ""),
                }
            return {
                "title": input_data.get("title", ""),
                "description": input_data.get("description", ""),
            }
        if hasattr(input_data, "task"):
            return {
                "title": input_data.task.title,
                "description": input_data.task.description,
            }
        return {"title": "", "description": str(input_data)}

    def _extract_libraries(self, text: str) -> list[str]:
        """Extract library/framework names from task text.

        Uses a combination of known library matching and import pattern detection.

        Args:
            text: Task description or combined text to analyze.

        Returns:
            Sorted list of unique library names found.
        """
        text_lower = text.lower()
        found = set()

        # Match known libraries
        for lib in KNOWN_LIBRARIES:
            # Use word boundary matching to avoid partial matches
            if re.search(r'\b' + re.escape(lib) + r'\b', text_lower):
                found.add(lib)

        # Match import-style references (e.g., "import psycopg3" or "from pydantic import")
        import_pattern = re.compile(r'(?:import|from)\s+(\w+)', re.IGNORECASE)
        for match in import_pattern.finditer(text):
            lib_name = match.group(1).lower()
            if lib_name not in ("os", "sys", "re", "json", "typing", "pathlib", "datetime"):
                found.add(lib_name)

        # Match pip install references
        pip_pattern = re.compile(r'pip\s+install\s+([a-zA-Z0-9_-]+)', re.IGNORECASE)
        for match in pip_pattern.finditer(text):
            found.add(match.group(1).lower())

        return sorted(found)

    def _build_repo_context(self, repo_path: str, max_depth: int = 3, max_files: int = 200) -> dict[str, Any]:
        """Build a structured snapshot of the repository.

        Adapted from GrokFlow's build_repo_context (P12). Traverses the
        directory tree, identifies key files, and returns a structured context.

        Args:
            repo_path: Path to the repository root.
            max_depth: Maximum directory traversal depth.
            max_files: Maximum number of files to include.

        Returns:
            Dictionary with tree structure, key files, and file counts.
        """
        root = Path(repo_path)
        if not root.is_dir():
            return {}

        # Key files to read content from
        key_filenames = {
            "README.md", "readme.md", "README.rst",
            "pyproject.toml", "setup.py", "setup.cfg",
            "requirements.txt", "Pipfile",
            "package.json",
            ".python-version",
            "Makefile",
            "Dockerfile",
        }

        tree_lines = []
        key_contents = {}
        file_count = 0
        dir_count = 0

        skip_dirs = {
            ".git", "__pycache__", "node_modules", ".venv", "venv",
            ".tox", ".mypy_cache", ".pytest_cache", "dist", "build",
            ".eggs", "*.egg-info",
        }

        def _walk(path: Path, depth: int, prefix: str = "") -> None:
            nonlocal file_count, dir_count

            if depth > max_depth or file_count > max_files:
                return

            try:
                entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
            except PermissionError:
                return

            for entry in entries:
                if entry.name.startswith(".") and entry.name not in (".python-version",):
                    if entry.is_dir() and entry.name in skip_dirs:
                        continue
                if entry.name in skip_dirs:
                    continue

                if entry.is_dir():
                    dir_count += 1
                    tree_lines.append(f"{prefix}{entry.name}/")
                    _walk(entry, depth + 1, prefix + "  ")
                else:
                    file_count += 1
                    if file_count <= max_files:
                        tree_lines.append(f"{prefix}{entry.name}")

                    # Read key files
                    if entry.name in key_filenames:
                        try:
                            content = entry.read_text(encoding="utf-8")[:2000]
                            key_contents[str(entry.relative_to(root))] = content
                        except Exception:
                            pass

        _walk(root, 0)

        return {
            "tree": "\n".join(tree_lines[:100]),  # Limit tree output
            "key_files": key_contents,
            "file_count": file_count,
            "dir_count": dir_count,
        }

    def _detect_version_warnings(self, live_docs: list[dict[str, str]]) -> list[str]:
        """Detect version-related warnings in search results.

        Scans snippets for deprecation notices, version requirements,
        and breaking change indicators.

        Args:
            live_docs: List of search result dicts with "snippet" keys.

        Returns:
            List of version warning strings.
        """
        warnings = []
        warning_patterns = [
            (r'deprecat\w+', "Deprecation notice found"),
            (r'breaking\s+change', "Breaking change mentioned"),
            (r'requires?\s+(?:python\s+)?(\d+\.\d+)', "Version requirement detected"),
            (r'end[\s-]of[\s-]life', "End-of-life warning"),
            (r'no\s+longer\s+(?:supported|maintained)', "Unmaintained library warning"),
        ]

        for doc in live_docs:
            snippet = doc.get("snippet", "").lower()
            title = doc.get("title", "").lower()
            combined = f"{title} {snippet}"

            for pattern, message in warning_patterns:
                if re.search(pattern, combined, re.IGNORECASE):
                    warnings.append(f"{message}: {doc.get('title', 'Unknown')}")
                    break  # One warning per doc

        return warnings
