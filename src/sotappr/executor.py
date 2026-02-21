"""Execution bridge from SOTAppR reports into the TaskLoop runtime."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import UUID

from src.core.models import Project, Task
from src.sotappr.models import BuilderRequest, DataScienceTaskSpec, SOTAppRReport
from src.sotappr.observability import SOTAppRObservability

if TYPE_CHECKING:
    from src.core.factory import ComponentBundle


@dataclass
class ExecutionSummary:
    """Summary returned after seeding and running SOTAppR tasks."""

    project_id: str
    tasks_seeded: int
    tasks_processed: int
    run_id: str | None = None
    mode: str = "execute"
    status: str = "completed"
    stop_reason: str | None = None
    estimated_cost_usd: float = 0.0
    elapsed_hours: float = 0.0


class SOTAppRExecutor:
    """Maps SOTAppR phase artifacts into executable task-loop work."""

    def __init__(self, repository: Any, task_loop: Any, progress_callback: Any = None):
        self.repository = repository
        self.task_loop = task_loop
        self.observability: SOTAppRObservability | None = None
        self._progress_callback = progress_callback

    @classmethod
    def from_bundle(
        cls,
        bundle: "ComponentBundle",
        repo_path: str,
        test_command: str = "pytest tests/ -v",
        progress_callback: Any = None,
    ) -> "SOTAppRExecutor":
        """Construct an executor with fully wired production dependencies."""
        from src.agents.builder import Builder
        from src.agents.council import Council
        from src.agents.librarian import Librarian
        from src.agents.research_unit import ResearchUnit
        from src.agents.sentinel import Sentinel
        from src.agents.state_manager import StateManager
        from src.orchestrator.diagnostics import DiagnosticsCollector
        from src.orchestrator.health_monitor import HealthMonitor
        from src.orchestrator.loop import TaskLoop
        from src.orchestrator.metrics import PipelineMetrics
        from src.orchestrator.task_router import TaskRouter

        state_manager = StateManager(repository=bundle.repository, repo_path=repo_path)

        # Override workspace_dir so the security policy allows writes to the
        # target repo, not just the directory where the CLI was launched from.
        if bundle.security_policy is not None:
            bundle.security_policy.workspace_dir = Path(repo_path).resolve()

        builder = Builder(
            llm_client=bundle.llm_client,
            model_router=bundle.model_router,
            repo_path=repo_path,
            test_command=test_command,
            security_policy=bundle.security_policy,
        )
        task_router = TaskRouter(bundle.repository)
        health_monitor = HealthMonitor(
            repository=bundle.repository,
            config=bundle.config.orchestrator,
            max_task_age_minutes=bundle.config.orchestrator.max_task_age_minutes,
            max_tokens_per_task=bundle.config.orchestrator.max_tokens_per_task,
        )
        metrics = PipelineMetrics()
        diagnostics = DiagnosticsCollector()

        research_unit = None
        if bundle.search_client is not None:
            research_unit = ResearchUnit(search_client=bundle.search_client, repo_path=repo_path)

        librarian = None
        if bundle.methodology_store is not None and bundle.hypothesis_tracker is not None:
            librarian = Librarian(
                methodology_store=bundle.methodology_store,
                hypothesis_tracker=bundle.hypothesis_tracker,
                repository=bundle.repository,
            )

        sentinel = Sentinel(
            embedding_engine=bundle.embedding_engine,
            banned_dependencies=[],
            drift_threshold=bundle.config.sentinel.drift_threshold,
            llm_deep_check=bundle.config.sentinel.llm_deep_check,
            llm_client=bundle.llm_client,
            model_router=bundle.model_router,
        )
        council = Council(llm_client=bundle.llm_client, model_router=bundle.model_router)

        task_loop = TaskLoop(
            repository=bundle.repository,
            state_manager=state_manager,
            builder=builder,
            task_router=task_router,
            health_monitor=health_monitor,
            metrics=metrics,
            diagnostics=diagnostics,
            config=bundle.config,
            research_unit=research_unit,
            librarian=librarian,
            sentinel=sentinel,
            council=council,
            progress_callback=progress_callback,
        )
        instance = cls(
            repository=bundle.repository,
            task_loop=task_loop,
            progress_callback=progress_callback,
        )
        observability = SOTAppRObservability(
            jsonl_path=Path(bundle.config.sotappr.observability_jsonl_path),
            metrics_path=Path(bundle.config.sotappr.observability_metrics_path),
        )
        instance.observability = observability
        return instance

    def bootstrap_project(
        self,
        request: BuilderRequest,
        report: SOTAppRReport,
        repo_path: str,
    ) -> tuple[Project, list[Task]]:
        """Create project metadata and seed tasks from Phase 3 backlog actions."""
        project = Project(
            name=request.organism_name,
            repo_path=repo_path,
            tech_stack={
                "builder": "SOTAppR",
                "selected_architecture": report.phase2.selected_architecture,
                "schema_version": request.schema_version,
            },
            project_rules=self._build_project_rules(request, report),
            banned_dependencies=self._rejected_dependencies(report),
        )
        self.repository.create_project(project)

        tasks = self.seed_tasks_from_backlog(
            project_id=project.id,
            backlog_actions=report.phase3.backlog_actions,
        )

        # Seed data science tasks if spec includes a DS task
        if request.data_science_task is not None:
            ds_tasks = self.seed_ds_tasks(
                project_id=project.id,
                ds_spec=request.data_science_task,
            )
            tasks.extend(ds_tasks)

        return project, tasks

    def seed_tasks_from_backlog(self, project_id, backlog_actions: list[str]) -> list[Task]:
        """Convert Phase 3 backlog actions to prioritized task records."""
        tasks: list[Task] = []
        total = len(backlog_actions)
        for index, action in enumerate(backlog_actions):
            title = self._title_from_action(action=action, index=index + 1)
            description = (
                "Seeded from SOTAppR Phase 3 backlog.\n"
                f"Action: {action}\n"
                "Acceptance: tests pass and Sentinel approves."
            )
            task = Task(
                project_id=project_id,
                title=title,
                description=description,
                priority=max(total - index, 1),
            )
            self.repository.create_task(task)
            tasks.append(task)
        return tasks

    def seed_ds_tasks(
        self,
        project_id: UUID,
        ds_spec: DataScienceTaskSpec,
    ) -> list[Task]:
        """Seed data science pipeline tasks from a DataScienceTaskSpec.

        Creates one task per DS phase (7 total), ordered by priority so they
        execute in the correct sequence: audit → eda → feature_eng → training →
        ensemble → evaluation → deployment.
        """
        ds_phases = [
            ("ds_audit", "Data Audit: profile columns, assess label quality"),
            ("ds_eda", "EDA: statistical analysis, correlation, outlier detection"),
            ("ds_feature_eng", "Feature Engineering: LLM-FE evolutionary transforms"),
            ("ds_training", "Model Training: scale-adaptive candidate training with CV"),
            ("ds_ensemble", "Ensemble: stacking, uncertainty routing, Pareto optimization"),
            ("ds_evaluation", "Evaluation: multi-dimensional scoring with LLM narrative"),
            ("ds_deployment", "Deployment: serialize model, API scaffold, monitoring config"),
        ]

        tasks: list[Task] = []
        total = len(ds_phases)
        for index, (task_type, phase_desc) in enumerate(ds_phases):
            title = f"DS-{index + 1}: {phase_desc}"
            description = (
                f"Data science pipeline phase: {task_type}\n"
                f"Dataset: {ds_spec.dataset_path}\n"
                f"Target: {ds_spec.target_column}\n"
                f"Problem type: {ds_spec.problem_type}\n"
            )
            if ds_spec.sensitive_columns:
                description += f"Sensitive columns: {', '.join(ds_spec.sensitive_columns)}\n"

            task = Task(
                project_id=project_id,
                title=title,
                description=description,
                priority=max(total - index, 1),
                task_type=task_type,
            )
            self.repository.create_task(task)
            tasks.append(task)

        return tasks

    def preview_seeded_tasks(self, backlog_actions: list[str]) -> list[dict[str, Any]]:
        """Preview exact task payloads that would be seeded."""
        preview: list[dict[str, Any]] = []
        total = len(backlog_actions)
        for index, action in enumerate(backlog_actions):
            preview.append(
                {
                    "title": self._title_from_action(action=action, index=index + 1),
                    "description": (
                        "Seeded from SOTAppR Phase 3 backlog.\n"
                        f"Action: {action}\n"
                        "Acceptance: tests pass and Sentinel approves."
                    ),
                    "priority": max(total - index, 1),
                }
            )
        return preview

    def execute_project(
        self,
        project_id,
        max_iterations: int = 100,
        budget_contract: dict[str, Any] | None = None,
        run_id: UUID | None = None,
    ) -> int:
        """Run the task loop for seeded project tasks."""
        if hasattr(self.task_loop, "set_run_context"):
            trace_id = str(run_id) if run_id is not None else None
            self.task_loop.set_run_context(run_id, trace_id=trace_id)
        try:
            try:
                return self.task_loop.run_loop(
                    project_id=project_id,
                    max_iterations=max_iterations,
                    budget_contract=budget_contract,
                )
            except TypeError:
                return self.task_loop.run_loop(project_id=project_id, max_iterations=max_iterations)
        finally:
            if hasattr(self.task_loop, "set_run_context"):
                self.task_loop.set_run_context(None, trace_id=None)

    def bootstrap_and_execute(
        self,
        request: BuilderRequest,
        report: SOTAppRReport,
        repo_path: str,
        max_iterations: int = 100,
        *,
        mode: str = "execute",
        governance_pack: str = "balanced",
        report_path: str | None = None,
        execute: bool = True,
        budget_contract: dict[str, Any] | None = None,
    ) -> ExecutionSummary:
        """Seed the project backlog and execute it with TaskLoop."""
        project, tasks = self.bootstrap_project(
            request=request,
            report=report,
            repo_path=repo_path,
        )
        run_id = self._create_run(
            project_id=project.id,
            mode=mode,
            governance_pack=governance_pack,
            request=request,
            report=report,
            repo_path=repo_path,
            report_path=report_path,
        )

        self._update_run(
            run_id=run_id,
            status="seeded",
            tasks_seeded=len(tasks),
            tasks_processed=0,
        )
        self._save_phase_artifacts(run_id=run_id, report=report)
        self._save_seed_artifact(
            run_id=run_id,
            payload={"tasks": self.preview_seeded_tasks(report.phase3.backlog_actions)},
        )
        self._emit_event(
            "sotappr_seeded",
            {
                "run_id": str(run_id) if run_id else None,
                "project_id": str(project.id),
                "tasks_seeded": len(tasks),
                "mode": mode,
            },
        )

        processed = 0
        run_status = "paused" if not execute else "completed"
        stop_reason: str | None = None
        estimated_cost_usd = 0.0
        elapsed_hours = 0.0
        try:
            if execute:
                self._update_run(run_id=run_id, status="running")
                processed = self.execute_project(
                    project.id,
                    max_iterations=max_iterations,
                    budget_contract=budget_contract,
                    run_id=run_id,
                )
                loop_summary = self._task_loop_summary()
                stop_reason = str(loop_summary.get("stop_reason")) if loop_summary else None
                estimated_cost_usd = float(loop_summary.get("estimated_cost_usd", 0.0)) if loop_summary else 0.0
                elapsed_hours = float(loop_summary.get("elapsed_hours", 0.0)) if loop_summary else 0.0
                if stop_reason in {"budget_time_exceeded", "budget_cost_exceeded"}:
                    run_status = "paused"
                    self._update_run(
                        run_id=run_id,
                        status="paused",
                        tasks_processed=processed,
                        last_error=f"Run paused: {stop_reason}",
                        stop_reason=stop_reason,
                        estimated_cost_usd=estimated_cost_usd,
                        elapsed_hours=elapsed_hours,
                    )
                    self._emit_event(
                        "sotappr_budget_paused",
                        {
                            "run_id": str(run_id) if run_id else None,
                            "project_id": str(project.id),
                            "tasks_processed": processed,
                            "stop_reason": stop_reason,
                            "estimated_cost_usd": estimated_cost_usd,
                            "elapsed_hours": elapsed_hours,
                        },
                    )
                else:
                    run_status = "completed"
            else:
                self._update_run(run_id=run_id, status="paused")
                self._emit_event(
                    "sotappr_paused",
                    {
                        "run_id": str(run_id) if run_id else None,
                        "project_id": str(project.id),
                    },
                )
        except Exception as exc:
            self._update_run(
                run_id=run_id,
                status="failed",
                tasks_processed=processed,
                last_error=str(exc)[:2000],
            )
            self._emit_event(
                "sotappr_failed",
                {
                    "run_id": str(run_id) if run_id else None,
                    "project_id": str(project.id),
                    "error": str(exc),
                },
            )
            self._flush_metrics()
            raise
        if execute:
            if run_status == "completed":
                self._update_run(
                    run_id=run_id,
                    status="completed",
                    tasks_processed=processed,
                    stop_reason=stop_reason or "completed",
                    estimated_cost_usd=estimated_cost_usd,
                    elapsed_hours=elapsed_hours,
                    completed=True,
                )
                self._emit_event(
                    "sotappr_executed",
                    {
                        "run_id": str(run_id) if run_id else None,
                        "project_id": str(project.id),
                        "tasks_processed": processed,
                    },
                )
        self._flush_metrics()
        return ExecutionSummary(
            project_id=str(project.id),
            tasks_seeded=len(tasks),
            tasks_processed=processed,
            run_id=str(run_id) if run_id else None,
            mode=mode,
            status=run_status,
            stop_reason=stop_reason,
            estimated_cost_usd=estimated_cost_usd,
            elapsed_hours=elapsed_hours,
        )

    def resume_run(
        self,
        run_id: UUID,
        max_iterations: int = 100,
        budget_contract: dict[str, Any] | None = None,
    ) -> ExecutionSummary:
        """Resume execution for an existing SOTAppR run."""
        run = self._get_run(run_id)
        if run is None:
            raise ValueError(f"SOTAppR run not found: {run_id}")

        project_id_raw = run.get("project_id")
        if project_id_raw is None:
            raise ValueError(f"SOTAppR run missing project_id: {run_id}")
        project_id = UUID(str(project_id_raw))
        existing_processed = int(run.get("tasks_processed") or 0)
        processed = 0
        run_status = "completed"
        stop_reason: str | None = None
        estimated_cost_usd = 0.0
        elapsed_hours = 0.0
        try:
            self._update_run(run_id=run_id, status="running")
            processed = self.execute_project(
                project_id=project_id,
                max_iterations=max_iterations,
                budget_contract=budget_contract,
                run_id=run_id,
            )
            loop_summary = self._task_loop_summary()
            stop_reason = str(loop_summary.get("stop_reason")) if loop_summary else None
            estimated_cost_usd = float(loop_summary.get("estimated_cost_usd", 0.0)) if loop_summary else 0.0
            elapsed_hours = float(loop_summary.get("elapsed_hours", 0.0)) if loop_summary else 0.0

            if stop_reason in {"budget_time_exceeded", "budget_cost_exceeded"}:
                run_status = "paused"
                self._update_run(
                    run_id=run_id,
                    status="paused",
                    tasks_processed=existing_processed + processed,
                    last_error=f"Run paused: {stop_reason}",
                    stop_reason=stop_reason,
                    estimated_cost_usd=estimated_cost_usd,
                    elapsed_hours=elapsed_hours,
                )
                self._emit_event(
                    "sotappr_resume_budget_paused",
                    {
                        "run_id": str(run_id),
                        "project_id": str(project_id),
                        "processed_delta": processed,
                        "stop_reason": stop_reason,
                        "estimated_cost_usd": estimated_cost_usd,
                        "elapsed_hours": elapsed_hours,
                    },
                )
            else:
                self._update_run(
                    run_id=run_id,
                    status="completed",
                    tasks_processed=existing_processed + processed,
                    stop_reason=stop_reason or "completed",
                    estimated_cost_usd=estimated_cost_usd,
                    elapsed_hours=elapsed_hours,
                    completed=True,
                )
                self._emit_event(
                    "sotappr_resumed",
                    {"run_id": str(run_id), "project_id": str(project_id), "processed_delta": processed},
                )
            return ExecutionSummary(
                project_id=str(project_id),
                tasks_seeded=int(run.get("tasks_seeded") or 0),
                tasks_processed=existing_processed + processed,
                run_id=str(run_id),
                mode="resume",
                status=run_status,
                stop_reason=stop_reason,
                estimated_cost_usd=estimated_cost_usd,
                elapsed_hours=elapsed_hours,
            )
        except Exception as exc:
            self._update_run(
                run_id=run_id,
                status="failed",
                tasks_processed=existing_processed + processed,
                last_error=str(exc)[:2000],
            )
            self._emit_event(
                "sotappr_resume_failed",
                {
                    "run_id": str(run_id),
                    "project_id": str(project_id),
                    "error": str(exc),
                },
            )
            raise
        finally:
            self._flush_metrics()

    def _task_loop_summary(self) -> dict[str, Any]:
        if hasattr(self.task_loop, "get_last_run_summary"):
            try:
                summary = self.task_loop.get_last_run_summary()
                if isinstance(summary, dict):
                    return summary
            except Exception:
                return {}
        return {}

    def _build_project_rules(self, request: BuilderRequest, report: SOTAppRReport) -> str:
        top_contract = report.phase1.contracts[0]
        lines = [
            f"Root need: {request.root_need}",
            f"Primary architecture: {report.phase2.selected_architecture}",
            f"Goal metric: {top_contract.goal.get('quantifiable_outcome', '')}",
            f"Latency max: {request.latency_maximum}",
            f"Security threshold: {request.security_threshold}",
            f"Complexity budget used: {report.phase2.complexity_used}/100",
        ]
        return "\n".join(lines)

    def _rejected_dependencies(self, report: SOTAppRReport) -> list[str]:
        rejected = [
            row.dependency
            for row in report.phase2.dependency_tribunal
            if row.verdict == "Rejected"
        ]
        return sorted(set(rejected))

    def _title_from_action(self, action: str, index: int) -> str:
        compact = " ".join(action.split())
        if len(compact) <= 72:
            return f"SOTAppR-{index}: {compact}"
        return f"SOTAppR-{index}: {compact[:69]}..."

    def _create_run(
        self,
        *,
        project_id,
        mode: str,
        governance_pack: str,
        request: BuilderRequest,
        report: SOTAppRReport,
        repo_path: str,
        report_path: str | None,
    ) -> UUID | None:
        if not hasattr(self.repository, "create_sotappr_run"):
            return None
        return self.repository.create_sotappr_run(
            project_id=project_id,
            mode=mode,
            governance_pack=governance_pack,
            spec_json=request.model_dump(mode="json"),
            report_json=report.model_dump(mode="json"),
            repo_path=repo_path,
            report_path=report_path,
            status="planned",
        )

    def _update_run(
        self,
        *,
        run_id: UUID | None,
        status: str | None = None,
        tasks_seeded: int | None = None,
        tasks_processed: int | None = None,
        last_error: str | None = None,
        stop_reason: str | None = None,
        estimated_cost_usd: float | None = None,
        elapsed_hours: float | None = None,
        completed: bool = False,
    ) -> None:
        if run_id is None or not hasattr(self.repository, "update_sotappr_run"):
            return
        self.repository.update_sotappr_run(
            run_id=run_id,
            status=status,
            tasks_seeded=tasks_seeded,
            tasks_processed=tasks_processed,
            last_error=last_error,
            stop_reason=stop_reason,
            estimated_cost_usd=estimated_cost_usd,
            elapsed_hours=elapsed_hours,
            completed=completed,
        )

    def _get_run(self, run_id: UUID) -> dict[str, Any] | None:
        if not hasattr(self.repository, "get_sotappr_run"):
            return None
        return self.repository.get_sotappr_run(run_id)

    def _save_phase_artifacts(self, run_id: UUID | None, report: SOTAppRReport) -> None:
        if run_id is None or not hasattr(self.repository, "save_sotappr_artifact"):
            return
        artifacts = [
            ("alien_goggles", 0, report.alien_goggles.model_dump(mode="json")),
            ("phase1", 1, report.phase1.model_dump(mode="json")),
            ("phase2", 2, report.phase2.model_dump(mode="json")),
            ("phase3", 3, report.phase3.model_dump(mode="json")),
            ("phase4", 4, report.phase4.model_dump(mode="json")),
            ("phase5", 5, report.phase5.model_dump(mode="json")),
            ("phase6", 6, report.phase6.model_dump(mode="json")),
            ("phase7", 7, report.phase7.model_dump(mode="json")),
            ("phase8", 8, report.phase8.model_dump(mode="json")),
        ]
        for artifact_type, phase, payload in artifacts:
            self.repository.save_sotappr_artifact(
                run_id=run_id,
                phase=phase,
                artifact_type=artifact_type,
                payload=payload,
            )
        for replay in report.phase8.experience_replay:
            self.repository.save_sotappr_artifact(
                run_id=run_id,
                phase=8,
                artifact_type="experience_replay",
                payload=replay.model_dump(mode="json"),
            )

    def _save_seed_artifact(self, run_id: UUID | None, payload: dict[str, Any]) -> None:
        if run_id is None or not hasattr(self.repository, "save_sotappr_artifact"):
            return
        self.repository.save_sotappr_artifact(
            run_id=run_id,
            phase=3,
            artifact_type="seeded_tasks",
            payload=payload,
        )

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.observability is None:
            return
        self.observability.emit_event(event_type=event_type, payload=payload)

    def _flush_metrics(self) -> None:
        if self.observability is None:
            return
        self.observability.flush_metrics()
