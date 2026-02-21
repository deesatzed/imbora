"""Librarian agent for The Associate.

Third agent in the pipeline (after Research Unit). Responsibilities:
1. Embed the task description
2. Query the methodology store for similar past solutions via hybrid search
3. Merge live docs from Research Unit with past solutions
4. Build a complete ContextBrief for the Builder

The Librarian is the knowledge aggregator — it combines everything
the Builder needs into a single ContextBrief.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from pathlib import Path

from src.agents.base_agent import BaseAgent
from src.core.models import (
    AgentResult,
    ContextBrief,
    Methodology,
    ResearchBrief,
    Task,
    TaskContext,
)
from src.db.repository import Repository
from src.memory.hypothesis_tracker import HypothesisTracker
from src.memory.methodology_store import MethodologyStore

logger = logging.getLogger("associate.agent.librarian")


class Librarian(BaseAgent):
    """Methodology RAG agent that builds complete context for Builder.

    Injected dependencies:
        methodology_store: For finding similar past solutions.
        hypothesis_tracker: For enriched forbidden approaches.
        repository: For project data access.
    """

    def __init__(
        self,
        methodology_store: MethodologyStore,
        hypothesis_tracker: HypothesisTracker,
        repository: Repository,
        skills_dir: Optional[str] = None,
    ):
        super().__init__(name="Librarian", role="librarian")
        self.methodology_store = methodology_store
        self.hypothesis_tracker = hypothesis_tracker
        self.repository = repository
        self._skills_dir = Path(skills_dir) if skills_dir else None

    def process(self, input_data: Any) -> AgentResult:
        """Build a ContextBrief combining all available knowledge.

        Args:
            input_data: Dictionary containing:
                - "task_context": TaskContext from State Manager
                - "research_brief": ResearchBrief from Research Unit (optional)

        Returns:
            AgentResult with ContextBrief in data["context_brief"].
        """
        # Extract components from pipeline data
        task_context, research_brief = self._extract_inputs(input_data)
        task = task_context.task

        # 1. Find similar past solutions
        past_solutions, retrieval_signals = self._find_similar_solutions(task)
        logger.info("Found %d similar past solutions", len(past_solutions))

        # 2. Get enriched forbidden approaches (local + project-wide)
        forbidden = self._get_enriched_forbidden(task)

        # 3. Get project rules
        project_rules = self._get_project_rules(task.project_id)

        # 4. Get live docs from research brief
        live_docs = research_brief.live_docs if research_brief else []

        # 5. Get council diagnosis from task context
        council_diagnosis = task_context.previous_council_diagnosis

        # 6. Look up matching skill procedure (Item 13)
        skill_procedure = self._find_skill(task)

        # 7. Build the complete context brief
        context_brief = ContextBrief(
            task=task,
            past_solutions=past_solutions,
            live_docs=live_docs,
            forbidden_approaches=forbidden,
            project_rules=project_rules,
            council_diagnosis=council_diagnosis,
            retrieval_confidence=float(retrieval_signals.get("retrieval_confidence", 0.0)),
            retrieval_conflicts=list(retrieval_signals.get("conflicts", [])),
            retrieval_strategy_hint=self._build_retrieval_hint(retrieval_signals, len(past_solutions)),
            skill_procedure=skill_procedure,
        )

        return AgentResult(
            agent_name=self.name,
            status="success",
            data={
                "context_brief": context_brief.model_dump(),
                "past_solutions_count": len(past_solutions),
                "forbidden_count": len(forbidden),
                "live_docs_count": len(live_docs),
                "retrieval_confidence": context_brief.retrieval_confidence,
                "retrieval_conflict_count": len(context_brief.retrieval_conflicts),
            },
        )

    def _extract_inputs(self, input_data: Any) -> tuple[TaskContext, Optional[ResearchBrief]]:
        """Extract TaskContext and ResearchBrief from pipeline input."""
        if isinstance(input_data, dict):
            # From pipeline transform
            tc_data = input_data.get("task_context")
            rb_data = input_data.get("research_brief")

            if isinstance(tc_data, TaskContext):
                task_context = tc_data
            elif isinstance(tc_data, dict):
                task_context = TaskContext(**tc_data)
            else:
                raise ValueError(f"Librarian expected task_context, got {type(tc_data)}")

            research_brief = None
            if isinstance(rb_data, ResearchBrief):
                research_brief = rb_data
            elif isinstance(rb_data, dict):
                research_brief = ResearchBrief(**rb_data)

            return task_context, research_brief

        if isinstance(input_data, TaskContext):
            return input_data, None

        raise ValueError(f"Librarian received unexpected input type: {type(input_data)}")

    def _find_similar_solutions(self, task: Task) -> tuple[list[Methodology], dict[str, Any]]:
        """Find similar past solutions using hybrid search.

        Searches project-scoped methodologies first. If results are low-confidence,
        falls back to global-scoped methodologies (Item 2 — cross-project scope).

        Args:
            task: The current task to find solutions for.

        Returns:
            Tuple of methodology list and retrieval signal metadata.
        """
        query = f"{task.title}: {task.description}"
        try:
            if hasattr(self.methodology_store, "find_similar_with_signals"):
                results, signals = self.methodology_store.find_similar_with_signals(query, limit=3)
            else:
                results = self.methodology_store.find_similar(query, limit=3)
                signals = {
                    "retrieval_confidence": 0.0,
                    "conflict_count": 0,
                    "conflicts": [],
                }

            # Item 2: Global scope fallback when local results are low-confidence
            confidence = float(signals.get("retrieval_confidence", 0.0))
            if confidence < 0.40 and len(results) < 2:
                logger.info("Low local confidence (%.2f) — searching global methodologies", confidence)
                try:
                    global_results = self.methodology_store.hybrid_search.search(
                        query=query, limit=3, scope="global",
                    )
                    for gr in global_results:
                        if not any(r.methodology.id == gr.methodology.id for r in results):
                            results.append(gr)
                    if global_results:
                        signals = self.methodology_store.hybrid_search.summarize_signals(results)
                except Exception as e:
                    logger.warning("Global methodology search failed: %s", e)

            return [r.methodology for r in results], signals
        except Exception as e:
            logger.warning("Methodology search failed: %s", e)
            return [], {
                "retrieval_confidence": 0.0,
                "conflict_count": 0,
                "conflicts": [],
            }

    def _get_enriched_forbidden(self, task: Task) -> list[str]:
        """Get forbidden approaches enriched with project-wide patterns.

        Args:
            task: The current task.

        Returns:
            List of forbidden approach strings.
        """
        try:
            return self.hypothesis_tracker.get_enriched_forbidden_approaches(
                task_id=task.id,
                project_id=task.project_id,
            )
        except Exception as e:
            logger.warning("Enriched forbidden lookup failed: %s", e)
            # Fall back to basic forbidden list
            try:
                return self.hypothesis_tracker.get_forbidden_approaches(task.id)
            except Exception:
                return []

    def _get_project_rules(self, project_id) -> Optional[str]:
        """Get project-specific rules.

        Args:
            project_id: The project ID.

        Returns:
            Project rules string, or None.
        """
        try:
            project = self.repository.get_project(project_id)
            if project and project.project_rules:
                return project.project_rules
        except Exception as e:
            logger.warning("Failed to get project rules: %s", e)
        return None

    def _build_retrieval_hint(self, signals: dict[str, Any], hit_count: int) -> str:
        """Produce a concise strategy hint from retrieval confidence/conflicts."""
        confidence = float(signals.get("retrieval_confidence", 0.0))
        conflicts = list(signals.get("conflicts", []))

        if hit_count == 0:
            return "No strong memory matches. Prefer first-principles implementation."
        if conflicts:
            return (
                f"Memory recall has conflicts ({len(conflicts)}). "
                "Treat recalled snippets as partial hints and prioritize test-backed synthesis."
            )
        if confidence >= 0.75:
            return "Memory recall is high-confidence. Reuse patterns directly, then adapt to project rules."
        return "Memory recall is medium-confidence. Reuse structure, but verify APIs and edge cases."

    def _find_skill(self, task: Task) -> Optional[str]:
        """Look up a matching skill procedure file for this task (Item 13).

        Scans the skills directory for markdown files whose name matches
        keywords in the task title/description.

        Returns:
            The skill procedure text, or None if no match found.
        """
        if self._skills_dir is None or not self._skills_dir.is_dir():
            return None

        try:
            task_text = f"{task.title} {task.description}".lower()
            best_match: Optional[Path] = None
            best_score = 0

            for skill_file in self._skills_dir.glob("*.md"):
                # Score by keyword overlap between skill filename and task text
                stem_words = set(skill_file.stem.lower().replace("-", " ").replace("_", " ").split())
                overlap = sum(1 for w in stem_words if w in task_text)
                if overlap > best_score:
                    best_score = overlap
                    best_match = skill_file

            if best_match and best_score > 0:
                content = best_match.read_text().strip()
                if content:
                    logger.info("Matched skill '%s' for task '%s'", best_match.name, task.title)
                    return content
        except Exception as e:
            logger.warning("Skill lookup failed: %s", e)

        return None
