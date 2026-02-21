"""Main task loop for The Associate.

Fetches the next pending task, runs it through the full agent pipeline:
  State Manager → Research Unit → Librarian → Builder → Sentinel

Handles retry logic with checkpoint/rewind, and manages the task lifecycle.
This is the core orchestration loop that ties everything together.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from src.agents.builder import Builder
from src.agents.state_manager import StateManager
from src.core.config import AppConfig
from src.core.models import (
    AgentResult,
    BuildResult,
    ContextBrief,
    CouncilDiagnosis,
    HypothesisEntry,
    HypothesisOutcome,
    PeerReview,
    ResearchBrief,
    SentinelVerdict,
    Task,
    TaskContext,
    TaskStatus,
    PacketType,
    RunPhase,
)
from src.db.repository import Repository
from src.llm.response_parser import normalize_error_signature
from src.orchestrator.diagnostics import DiagnosticsCollector
from src.orchestrator.health_monitor import HealthMonitor
from src.orchestrator.metrics import PipelineMetrics
from src.orchestrator.pipeline import Pipeline
from src.orchestrator.arbitration import BuildArbiter
from src.orchestrator.budget_hints import generate_budget_hint
from src.orchestrator.complexity import score_task_complexity
from src.orchestrator.task_router import TaskRouter
from src.orchestrator.wrapup import WrapUpWorkflow
from src.protocol.packet_runtime import PacketPolicyError, PacketRuntime
from src.protocol.validation import PacketValidationError, validate_agent_packet
from src.tools.git_ops import commit

logger = logging.getLogger("associate.orchestrator.loop")

# Optional Phase 3 imports — gracefully degrade if not available
try:
    from src.agents.librarian import Librarian
    from src.agents.research_unit import ResearchUnit
    from src.agents.sentinel import Sentinel
    _PHASE3_AVAILABLE = True
except ImportError:
    _PHASE3_AVAILABLE = False


class TaskLoop:
    """Main orchestration loop: fetch task → pipeline → retry/commit.

    Full pipeline (Phase 3): State Manager → Research Unit → Librarian → Builder → Sentinel
    Fallback (Phase 2): State Manager → Builder

    Injected dependencies:
        repository: Database access.
        state_manager: State Manager agent.
        builder: Builder agent.
        task_router: Task state machine.
        health_monitor: Pre-loop health checks.
        metrics: Pipeline metrics collector.
        diagnostics: Diagnostic snapshot collector.
        config: Application configuration.
        research_unit: Optional Research Unit agent (Phase 3).
        librarian: Optional Librarian agent (Phase 3).
        sentinel: Optional Sentinel agent (Phase 3).
    """

    _ARBITRATION_STRATEGY_HINTS: tuple[str, ...] = (
        "Arbitration variant A: prioritize minimal, targeted file changes with strongest test leverage.",
        "Arbitration variant B: prioritize correctness-first edge-case coverage with explicit validation.",
        "Arbitration variant C: prioritize maintainability, clear errors, and conservative dependency use.",
    )

    def __init__(
        self,
        repository: Repository,
        state_manager: StateManager,
        builder: Builder,
        task_router: TaskRouter,
        health_monitor: HealthMonitor,
        metrics: PipelineMetrics,
        diagnostics: DiagnosticsCollector,
        config: AppConfig,
        research_unit: Optional[object] = None,
        librarian: Optional[object] = None,
        sentinel: Optional[object] = None,
        council: Optional[object] = None,
        arbiter: Optional[BuildArbiter] = None,
        progress_callback: Optional[Callable[[str], None]] = None,
    ):
        self.repository = repository
        self.state_manager = state_manager
        self.builder = builder
        self.task_router = task_router
        self.health_monitor = health_monitor
        self.metrics = metrics
        self.diagnostics = diagnostics
        self.config = config
        self.research_unit = research_unit
        self.librarian = librarian
        self.sentinel = sentinel
        self.council = council
        self.arbiter = arbiter or BuildArbiter()
        self._progress_callback = progress_callback
        self._total_tokens_per_task: dict[uuid.UUID, int] = {}
        self._council_diagnosis_cache: dict[uuid.UUID, str] = {}
        self._forced_builder_strategy: dict[uuid.UUID, str] = {}
        self._no_output_attempts: dict[uuid.UUID, int] = {}
        self._active_run_id: uuid.UUID | None = None
        self._active_trace_id: str | None = None
        self._packet_session_id: str = str(uuid.uuid4())
        self._packet_runtime = PacketRuntime()
        self._packet_roots: dict[tuple[uuid.UUID, int], uuid.UUID] = {}
        self._packet_steps: dict[tuple[uuid.UUID, int], int] = {}
        self._pipeline = Pipeline(metrics=metrics, diagnostics=diagnostics)
        self._last_run_summary: dict[str, Any] = {
            "processed": 0,
            "tokens": 0,
            "estimated_cost_usd": 0.0,
            "elapsed_hours": 0.0,
            "stop_reason": "not_started",
        }

    def _notify(self, message: str) -> None:
        """Send a progress notification to the CLI callback, if configured."""
        if self._progress_callback is not None:
            self._progress_callback(message)

    def set_run_context(self, run_id: uuid.UUID | None, trace_id: str | None = None) -> None:
        """Attach active SOTAppR run context so loop artifacts can be persisted."""
        self._active_run_id = run_id
        self._active_trace_id = trace_id

    def _ensure_run_trace_id(self) -> str:
        if self._active_trace_id:
            return self._active_trace_id
        self._active_trace_id = str(uuid.uuid4())
        return self._active_trace_id

    def _build_attempt_trace_id(self, task: Task, attempt: int) -> str:
        root = self._ensure_run_trace_id()
        return f"{root}:task:{str(task.id)[:8]}:attempt:{attempt}"

    def _start_attempt_diagnostics(
        self,
        task: Task,
        attempt: int,
        trace_id: str,
        retrieval_confidence: float | None = None,
        retrieval_conflicts: int | None = None,
    ) -> None:
        run_id = str(self._active_run_id) if self._active_run_id else None
        metadata: dict[str, Any] = {
            "task_title": task.title,
            "task_status": task.status.value,
        }
        if retrieval_confidence is not None:
            metadata["retrieval_confidence"] = float(retrieval_confidence)
        if retrieval_conflicts is not None:
            metadata["retrieval_conflicts"] = int(retrieval_conflicts)
        self.diagnostics.start_run(
            task_id=task.id,
            run_number=attempt,
            run_id=run_id,
            trace_id=trace_id,
            metadata=metadata,
        )

    def _complete_attempt_diagnostics(self, outcome: str, error_summary: str | None = None) -> None:
        self.diagnostics.complete_run(outcome=outcome, error_summary=error_summary)

    @staticmethod
    def _hash_json_payload(payload: Any) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _attempt_packet_key(self, task: Task, attempt: int) -> tuple[uuid.UUID, int]:
        return (task.id, int(attempt))

    def _ensure_packet_root(self, task: Task, attempt: int) -> uuid.UUID:
        key = self._attempt_packet_key(task, attempt)
        if key not in self._packet_roots:
            self._packet_roots[key] = uuid.uuid4()
            self._packet_steps[key] = 0
        return self._packet_roots[key]

    def _next_packet_step(self, task: Task, attempt: int) -> int:
        key = self._attempt_packet_key(task, attempt)
        self._packet_steps[key] = self._packet_steps.get(key, 0) + 1
        return self._packet_steps[key]

    def _emit_agent_packet(
        self,
        *,
        task: Task,
        attempt: int,
        trace_id: str,
        packet_type: PacketType,
        run_phase: RunPhase,
        sender_role: str,
        recipient_roles: list[str],
        payload: dict[str, Any],
        confidence: float = 0.75,
        symbolic_keys: list[str] | None = None,
        capability_tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        lineage: dict[str, Any] | None = None,
        proof_bundle: dict[str, Any] | None = None,
        promote: bool | None = None,
        verification_pass: bool = True,
        generation_override: int | None = None,
    ) -> dict[str, Any] | None:
        root_packet_id = self._ensure_packet_root(task, attempt)
        generation = int(generation_override) if generation_override is not None else int(attempt)
        packet_dict = {
            "protocol_version": "apc/1.0",
            "packet_type": packet_type.value,
            "channel": "INTER_AGENT",
            "run_phase": run_phase.value,
            "sender": {
                "agent_id": f"{sender_role}-1",
                "role": sender_role,
                "swarm_id": "associate-main",
            },
            "recipients": [
                {
                    "agent_id": f"{role}-1",
                    "role": role,
                    "swarm_id": "associate-main",
                }
                for role in recipient_roles
            ],
            "trace": {
                "session_id": self._packet_session_id,
                "run_id": str(self._active_run_id or self._ensure_run_trace_id()),
                "task_id": str(task.id),
                "root_packet_id": str(root_packet_id),
                "generation": generation,
                "step": self._next_packet_step(task, attempt),
            },
            "routing": {
                "delivery_mode": "DIRECT",
                "priority": 5,
                "ttl_ms": 60_000,
                "requires_ack": False,
            },
            "confidence": float(confidence),
            "symbolic_keys": list(symbolic_keys or []),
            "capability_tags": list(capability_tags or []),
            "payload": dict(payload),
            "metadata": {
                "task_title": task.title,
                "attempt": int(attempt),
                **(metadata or {}),
            },
        }
        if proof_bundle is not None:
            packet_dict["proof_bundle"] = dict(proof_bundle)
        if lineage is not None:
            packet_dict["lineage"] = dict(lineage)

        try:
            packet = validate_agent_packet(packet_dict)
        except PacketValidationError as exc:
            logger.warning(
                "Packet validation failed for task %s attempt %d (%s): %s",
                task.id,
                attempt,
                packet_type.value,
                exc,
            )
            self._record_trace_event(
                task=task,
                attempt=attempt,
                trace_id=trace_id,
                event_type="agent_packet_invalid",
                payload={
                    "packet_type": packet_type.value,
                    "run_phase": run_phase.value,
                    "error": str(exc)[:500],
                },
                persist_artifact=False,
            )
            return None

        promote_now = bool(promote) if promote is not None else False
        if promote is None and packet_type == PacketType.EVAL:
            metric_set = payload.get("metric_set", {})
            if isinstance(metric_set, dict):
                promote_now = self._packet_runtime.eval_is_promotable(metric_set)

        try:
            lifecycle_state, lifecycle_history = self._packet_runtime.process_packet(
                packet,
                promote=promote_now,
                verification_pass=verification_pass,
            )
        except PacketPolicyError as exc:
            logger.warning(
                "Packet policy rejected for task %s attempt %d (%s): %s",
                task.id,
                attempt,
                packet_type.value,
                exc,
            )
            self._record_trace_event(
                task=task,
                attempt=attempt,
                trace_id=trace_id,
                event_type="agent_packet_policy_rejected",
                payload={
                    "packet_id": str(packet.packet_id),
                    "packet_type": packet_type.value,
                    "run_phase": run_phase.value,
                    "error": str(exc)[:500],
                },
                persist_artifact=False,
            )
            return None

        self._record_trace_event(
            task=task,
            attempt=attempt,
            trace_id=trace_id,
            event_type="agent_packet_emitted",
            payload={
                "packet_id": str(packet.packet_id),
                "packet_type": packet_type.value,
                "run_phase": run_phase.value,
                "lifecycle_state": lifecycle_state,
            },
            persist_artifact=False,
        )

        if self._active_run_id is not None:
            packet_payload = packet.model_dump(mode="json")
            artifact_payload = {
                "trace_id": trace_id,
                "task_id": str(task.id),
                "task_title": task.title,
                "attempt": int(attempt),
                "packet": packet_payload,
                "packet_hash": self._hash_json_payload(packet_payload),
                "lifecycle_state": lifecycle_state,
                "lifecycle_history": lifecycle_history,
            }
            if hasattr(self.repository, "save_agent_packet"):
                try:
                    self.repository.save_agent_packet(
                        run_id=self._active_run_id,
                        task_id=task.id,
                        attempt_number=int(attempt),
                        trace_id=trace_id,
                        packet=packet_payload,
                        packet_hash=artifact_payload["packet_hash"],
                        lifecycle_state=lifecycle_state,
                        lifecycle_history=lifecycle_history,
                    )
                except Exception as exc:
                    logger.warning("Failed to persist first-class agent packet for task %s: %s", task.id, exc)
            if hasattr(self.repository, "save_sotappr_artifact"):
                try:
                    self.repository.save_sotappr_artifact(
                        run_id=self._active_run_id,
                        phase=8,
                        artifact_type="agent_packet",
                        payload=artifact_payload,
                    )
                except Exception as exc:
                    logger.warning("Failed to persist agent packet for task %s: %s", task.id, exc)

        return {
            "packet_id": str(packet.packet_id),
            "packet_type": packet_type.value,
            "run_phase": run_phase.value,
            "lifecycle_state": lifecycle_state,
            "payload_hash": self._hash_json_payload(packet.payload),
        }

    def _record_trace_event(
        self,
        *,
        task: Task,
        attempt: int,
        trace_id: str,
        event_type: str,
        payload: dict[str, Any],
        persist_artifact: bool = False,
    ) -> None:
        run_id = str(self._active_run_id) if self._active_run_id else None
        full_payload = {
            "task_id": str(task.id),
            "task_title": task.title,
            "attempt": int(attempt),
            **payload,
        }
        self.diagnostics.record_event(
            event_type=event_type,
            payload=full_payload,
            trace_id=trace_id,
            run_id=run_id,
            task_id=task.id,
        )
        if not persist_artifact:
            return
        if self._active_run_id is None or not hasattr(self.repository, "save_sotappr_artifact"):
            return
        try:
            self.repository.save_sotappr_artifact(
                run_id=self._active_run_id,
                phase=8,
                artifact_type="trace_event",
                payload={
                    "trace_id": trace_id,
                    "event_type": event_type,
                    **full_payload,
                },
            )
        except Exception as exc:
            logger.warning("Failed to persist trace event for task %s: %s", task.id, exc)

    def _persist_retry_telemetry(
        self,
        *,
        task: Task,
        attempt: int,
        trace_id: str,
        retry_cause: str,
        retrieval_confidence: float,
        retrieval_conflicts: int,
        details: str | None = None,
    ) -> None:
        payload = {
            "trace_id": trace_id,
            "task_id": str(task.id),
            "task_title": task.title,
            "attempt": int(attempt),
            "retry_cause": retry_cause,
            "retrieval_confidence": float(retrieval_confidence),
            "retrieval_conflicts": int(retrieval_conflicts),
        }
        if details:
            payload["details"] = details[:500]
        self._record_trace_event(
            task=task,
            attempt=attempt,
            trace_id=trace_id,
            event_type="retry_decision",
            payload=payload,
            persist_artifact=True,
        )
        if self._active_run_id is None or not hasattr(self.repository, "save_sotappr_artifact"):
            return
        try:
            self.repository.save_sotappr_artifact(
                run_id=self._active_run_id,
                phase=8,
                artifact_type="retry_telemetry",
                payload=payload,
            )
        except Exception as exc:
            logger.warning("Failed to persist retry telemetry for task %s: %s", task.id, exc)

    def _collect_model_failover_state(self) -> dict[str, Any]:
        snapshots: dict[str, Any] = {}
        agent_map = {
            "builder": self.builder,
            "sentinel": self.sentinel,
            "council": self.council,
            "research_unit": self.research_unit,
            "librarian": self.librarian,
        }
        for role, agent in agent_map.items():
            llm_client = getattr(agent, "llm_client", None) if agent is not None else None
            if llm_client is None or not hasattr(llm_client, "get_model_failover_state"):
                continue
            try:
                state = llm_client.get_model_failover_state()
            except Exception:
                continue
            if state:
                snapshots[role] = state
        return snapshots

    def _persist_model_failover_snapshot(
        self,
        *,
        task: Task,
        attempt: int,
        trace_id: str,
    ) -> None:
        state = self._collect_model_failover_state()
        if not state:
            return
        payload = {
            "trace_id": trace_id,
            "task_id": str(task.id),
            "task_title": task.title,
            "attempt": int(attempt),
            "state": state,
        }
        self._record_trace_event(
            task=task,
            attempt=attempt,
            trace_id=trace_id,
            event_type="model_failover_state",
            payload={"state": state},
            persist_artifact=True,
        )
        if self._active_run_id is None or not hasattr(self.repository, "save_sotappr_artifact"):
            return
        try:
            self.repository.save_sotappr_artifact(
                run_id=self._active_run_id,
                phase=8,
                artifact_type="model_failover_state",
                payload=payload,
            )
        except Exception as exc:
            logger.warning("Failed to persist model failover snapshot for task %s: %s", task.id, exc)

    def run_once(self, project_id: uuid.UUID) -> Optional[Task]:
        """Execute one iteration of the task loop.

        Fetches the next pending task and processes it through the pipeline.
        Handles retries on failure up to max_retries.

        Args:
            project_id: The project to fetch tasks from.

        Returns:
            The processed Task (with updated status), or None if no tasks pending.
        """
        # 1. Health checks
        checks = self.health_monitor.run_checks()
        failed_checks = [c for c in checks if not c.passed]
        if any(c.check_name == "circuit_breaker" and not c.passed for c in checks):
            logger.warning("Circuit breaker open — skipping this iteration")
            self._notify("[WARN] Circuit breaker open — skipping iteration")
            return None

        # 2. Fetch next task
        task = self.repository.get_next_task(project_id)
        if task is None:
            logger.info("No pending tasks for project %s", project_id)
            return None

        logger.info("Processing task: '%s' (%s)", task.title, task.id)
        self._notify(f"[TASK] {task.title}")

        # 3. Route: DS tasks go to the DS pipeline, general tasks go through agent pipeline
        task_type = getattr(task, "task_type", "general") or "general"
        if task_type.startswith("ds_"):
            return self._process_ds_task(task)

        return self._process_task(task)

    def _process_ds_task(self, task: Task) -> Task:
        """Process a data science task through the appropriate DS agent.

        DS tasks are typed (ds_audit, ds_eda, ds_feature_eng, ds_training,
        ds_ensemble, ds_evaluation, ds_deployment) and route to the
        corresponding DS agent rather than the standard agent pipeline.

        The DS agent is instantiated on-the-fly from config; no git operations.
        """
        task_type = getattr(task, "task_type", "general") or "general"
        logger.info("Processing DS task: type=%s, title='%s'", task_type, task.title)
        self._notify(f"  [DS] Phase: {task_type}")

        # Parse dataset metadata from task description
        ds_input = self._parse_ds_task_description(task)
        if ds_input is None:
            logger.error("Could not parse DS task description for '%s'", task.title)
            self.task_router.transition(task, TaskStatus.STUCK, reason="unparseable DS task description")
            return task

        ds_input["task_id"] = task.id
        ds_input["run_id"] = self._active_run_id

        try:
            from src.core.config import DataScienceConfig

            ds_config = self.config.datasci if hasattr(self.config, "datasci") else DataScienceConfig()

            agent = self._create_ds_agent(task_type, ds_config)
            if agent is None:
                logger.error("Unknown DS task type: %s", task_type)
                self.task_router.transition(task, TaskStatus.STUCK, reason=f"unknown DS type: {task_type}")
                return task

            self.task_router.transition(task, TaskStatus.CODING, reason=f"running DS agent: {task_type}")
            result = agent.process(ds_input)

            if result.status == "success":
                self.task_router.transition(task, TaskStatus.DONE, reason="DS phase completed")
                logger.info("DS task '%s' completed successfully", task.title)
                self._notify(f"  [DS] {task_type} completed")
            else:
                logger.warning("DS task '%s' failed: %s", task.title, result.error)
                self.task_router.transition(
                    task, TaskStatus.STUCK,
                    reason=f"DS agent returned error: {result.error}",
                )

        except Exception as exc:
            logger.exception("DS task '%s' failed with exception", task.title)
            self.task_router.transition(
                task, TaskStatus.STUCK,
                reason=f"DS exception: {str(exc)[:500]}",
            )

        return task

    def _parse_ds_task_description(self, task: Task) -> dict[str, Any] | None:
        """Extract dataset_path, target_column, problem_type from DS task description."""
        desc = task.description or ""
        result: dict[str, Any] = {}

        for line in desc.split("\n"):
            line = line.strip()
            if line.startswith("Dataset:"):
                result["dataset_path"] = line.split(":", 1)[1].strip()
            elif line.startswith("Target:"):
                result["target_column"] = line.split(":", 1)[1].strip()
            elif line.startswith("Problem type:"):
                result["problem_type"] = line.split(":", 1)[1].strip()
            elif line.startswith("Sensitive columns:"):
                cols = line.split(":", 1)[1].strip()
                result["sensitive_columns"] = [c.strip() for c in cols.split(",") if c.strip()]

        # project_id comes from the task itself
        result["project_id"] = task.project_id

        if "dataset_path" not in result or "target_column" not in result:
            return None

        return result

    def _create_ds_agent(self, task_type: str, ds_config: Any) -> Any | None:
        """Instantiate the appropriate DS agent for the given task type."""
        agent_map = {
            "ds_audit": "src.datasci.agents.data_audit.DataAuditAgent",
            "ds_eda": "src.datasci.agents.eda.EDAAgent",
            "ds_feature_eng": "src.datasci.agents.feature_engineering.FeatureEngineeringAgent",
            "ds_training": "src.datasci.agents.model_training.ModelTrainingAgent",
            "ds_ensemble": "src.datasci.agents.ensemble.EnsembleAgent",
            "ds_evaluation": "src.datasci.agents.evaluation.EvaluationAgent",
            "ds_deployment": "src.datasci.agents.deployment.DeploymentAgent",
        }
        module_path = agent_map.get(task_type)
        if module_path is None:
            return None

        module_name, class_name = module_path.rsplit(".", 1)
        try:
            import importlib

            mod = importlib.import_module(module_name)
            agent_class = getattr(mod, class_name)
            return agent_class(
                llm_client=self.builder.llm_client if hasattr(self.builder, "llm_client") else None,
                model_router=getattr(self.builder, "model_router", None),
                repository=self.repository,
                ds_config=ds_config,
            )
        except (ImportError, AttributeError) as exc:
            logger.error("Could not load DS agent %s: %s", module_path, exc)
            return None

    def _process_task(self, task: Task) -> Task:
        """Process a single task through the full agent pipeline.

        Full pipeline (Phase 3):
            State Manager → Research Unit → Librarian → Builder → Sentinel
        Fallback (Phase 2):
            State Manager → Builder

        Implements the retry loop with checkpoint/rewind and hypothesis logging.
        """
        max_retries = self.config.orchestrator.max_retries_per_task
        council_threshold = self.config.orchestrator.council_trigger_threshold
        max_council = self.config.orchestrator.council_max_invocations

        # Move to CODING status
        self.task_router.transition(task, TaskStatus.CODING, reason="starting build")

        for attempt in range(1, max_retries + 1):
            logger.info("Task '%s': attempt %d/%d", task.title, attempt, max_retries)
            self._notify(f"  Attempt {attempt}/{max_retries}")

            # Increment attempt counter
            self.task_router.increment_attempt(task)
            current_attempt = task.attempt_count
            attempt_trace_id = self._build_attempt_trace_id(task, current_attempt)
            retrieval_confidence = 0.0
            retrieval_conflicts = 0
            self._start_attempt_diagnostics(
                task=task,
                attempt=current_attempt,
                trace_id=attempt_trace_id,
            )
            self._record_trace_event(
                task=task,
                attempt=current_attempt,
                trace_id=attempt_trace_id,
                event_type="attempt_started",
                payload={"max_retries": max_retries},
                persist_artifact=True,
            )
            self._emit_agent_packet(
                task=task,
                attempt=current_attempt,
                trace_id=attempt_trace_id,
                packet_type=PacketType.SIGNAL,
                run_phase=RunPhase.DISCOVERY,
                sender_role="orchestrator",
                recipient_roles=["state_manager", "builder"],
                payload={
                    "signal": "ATTEMPT_STARTED",
                    "reason": f"attempt={current_attempt}, max_retries={max_retries}",
                },
                symbolic_keys=["attempt", "retry"],
                capability_tags=["orchestration", "control-plane"],
            )

            # Initialize token tracking for this task
            if task.id not in self._total_tokens_per_task:
                self._total_tokens_per_task[task.id] = 0

            # --- State Manager phase ---
            sm_result = self.state_manager.run(task)
            if sm_result.status != "success":
                logger.error("State Manager failed: %s", sm_result.error)
                self._emit_agent_packet(
                    task=task,
                    attempt=current_attempt,
                    trace_id=attempt_trace_id,
                    packet_type=PacketType.ERROR,
                    run_phase=RunPhase.EXECUTION,
                    sender_role="state_manager",
                    recipient_roles=["orchestrator"],
                    payload={
                        "error_code": "state_manager_failed",
                        "message": (sm_result.error or "State manager failed")[:500],
                        "retryable": True,
                        "details": {},
                    },
                    symbolic_keys=["state-manager", "failure"],
                    capability_tags=["state", "error"],
                )
                self._persist_retry_telemetry(
                    task=task,
                    attempt=current_attempt,
                    trace_id=attempt_trace_id,
                    retry_cause="state_manager_failed",
                    retrieval_confidence=retrieval_confidence,
                    retrieval_conflicts=retrieval_conflicts,
                    details=sm_result.error,
                )
                self._log_hypothesis(
                    task,
                    current_attempt,
                    "State Manager failed",
                    sm_result.error,
                )
                self._complete_attempt_diagnostics(outcome="failure", error_summary=sm_result.error)
                continue

            # Extract TaskContext
            task_context = TaskContext(**sm_result.data.get("task_context", {}))
            checkpoint_ref = task_context.checkpoint_ref
            self._emit_agent_packet(
                task=task,
                attempt=current_attempt,
                trace_id=attempt_trace_id,
                packet_type=PacketType.TASK_IR,
                run_phase=RunPhase.DISCOVERY,
                sender_role="state_manager",
                recipient_roles=["builder"],
                payload={
                    "ir_header": {
                        "task_id": str(task.id),
                        "task_title": task.title,
                        "task_description": task.description,
                        "checkpoint_ref": checkpoint_ref or "",
                    },
                    "objects": [
                        {
                            "type": "task",
                            "task_id": str(task.id),
                            "status": task.status.value,
                            "priority": int(task.priority),
                        }
                    ],
                    "normalization": {
                        "attempt_number": int(current_attempt),
                        "forbidden_approaches": list(task_context.forbidden_approaches or []),
                    },
                },
                symbolic_keys=["task_ir", "context"],
                capability_tags=["state", "routing"],
            )

            # --- Research Unit phase (Phase 3, optional) ---
            research_brief = None
            if self.research_unit is not None:
                self._notify("  Research Unit: searching live docs...")
                self.task_router.transition(task, TaskStatus.RESEARCHING, reason="live doc search")
                ru_result = self.research_unit.run(task_context)
                tokens = ru_result.data.get("tokens_used", 0) if ru_result.data else 0
                self._total_tokens_per_task[task.id] += tokens

                if ru_result.status == "success" and ru_result.data:
                    rb_data = ru_result.data.get("research_brief", {})
                    if rb_data:
                        research_brief = ResearchBrief(**rb_data)
                else:
                    logger.warning("Research Unit failed (non-fatal): %s", ru_result.error)

                # Return to CODING status after research
                self.task_router.transition(task, TaskStatus.CODING, reason="research complete")

            # --- Librarian phase (Phase 3, optional) ---
            builder_input = task_context  # Default: pass TaskContext directly to Builder
            if self.librarian is not None:
                self._notify("  Librarian: retrieving past solutions...")
                librarian_input = {
                    "task_context": task_context,
                    "research_brief": research_brief,
                }
                lib_result = self.librarian.run(librarian_input)
                tokens = lib_result.data.get("tokens_used", 0) if lib_result.data else 0
                self._total_tokens_per_task[task.id] += tokens

                if lib_result.status == "success" and lib_result.data:
                    cb_data = lib_result.data.get("context_brief")
                    if cb_data:
                        builder_input = ContextBrief(**cb_data) if isinstance(cb_data, dict) else cb_data
                else:
                    logger.warning("Librarian failed (non-fatal): %s", lib_result.error)

            retrieval_confidence, retrieval_conflicts = self._extract_retrieval_signals(builder_input)
            self._record_trace_event(
                task=task,
                attempt=current_attempt,
                trace_id=attempt_trace_id,
                event_type="retrieval_signals",
                payload={
                    "retrieval_confidence": retrieval_confidence,
                    "retrieval_conflicts": retrieval_conflicts,
                },
                persist_artifact=True,
            )
            self._emit_agent_packet(
                task=task,
                attempt=current_attempt,
                trace_id=attempt_trace_id,
                packet_type=PacketType.SIGNAL,
                run_phase=RunPhase.DISCOVERY,
                sender_role="librarian" if self.librarian is not None else "state_manager",
                recipient_roles=["builder"],
                payload={
                    "signal": "RETRIEVAL_SIGNALS",
                    "reason": (
                        f"confidence={retrieval_confidence:.2f};"
                        f" conflicts={retrieval_conflicts}"
                    ),
                },
                symbolic_keys=["retrieval", "confidence"],
                capability_tags=["retrieval", "context"],
                metadata={
                    "retrieval_confidence": float(retrieval_confidence),
                    "retrieval_conflicts": int(retrieval_conflicts),
                },
            )

            # --- Complexity scoring (Item 7) + budget hints (Item 8) ---
            complexity_tier = score_task_complexity(task)
            budget_hint = generate_budget_hint(
                tokens_used=self._total_tokens_per_task.get(task.id, 0),
                token_budget=self.config.orchestrator.max_tokens_per_task,
                attempt=attempt,
                max_retries=max_retries,
            )
            if isinstance(builder_input, ContextBrief):
                builder_input = builder_input.model_copy(update={
                    "complexity_tier": complexity_tier.value,
                })
            if isinstance(builder_input, dict):
                builder_input["complexity_tier"] = complexity_tier.value
                if budget_hint:
                    builder_input["budget_hint"] = budget_hint

            # --- Builder phase ---
            self._notify("  Builder: generating code...")
            builder_input = self._apply_forced_strategy(task, builder_input)
            builder_input = self._apply_self_correction_guardrail(task, builder_input)
            builder_result, build_result, tokens, selected_builder_input = self._run_builder_with_arbitration(
                task=task,
                builder_input=builder_input,
            )
            build_data = build_result.model_dump()
            build_artifact_uri = f"task://{task.id}/attempt/{current_attempt}/build_result"
            build_artifact_hash = self._hash_json_payload(build_data)
            self._emit_agent_packet(
                task=task,
                attempt=current_attempt,
                trace_id=attempt_trace_id,
                packet_type=PacketType.ARTIFACT,
                run_phase=RunPhase.EXECUTION,
                sender_role="builder",
                recipient_roles=["sentinel" if self.sentinel is not None else "orchestrator"],
                payload={
                    "artifact_id": f"build-result-{str(task.id)[:8]}-{current_attempt}",
                    "artifact_kind": "build_result",
                    "artifact_uri": build_artifact_uri,
                    "artifact_hash": build_artifact_hash,
                    "language": "python",
                    "adapter": "taskloop",
                },
                symbolic_keys=["build_result", "artifact"],
                capability_tags=["execution", "artifact"],
                metadata={
                    "tests_passed": bool(build_result.tests_passed),
                    "files_changed_count": len(build_result.files_changed or []),
                },
            )

            # Track tokens
            self._total_tokens_per_task[task.id] = self._total_tokens_per_task.get(task.id, 0) + tokens

            # Record LLM success/failure for circuit breaker
            if tokens > 0:
                self.health_monitor.record_llm_success()

            # Check token budget
            budget_check = self.health_monitor.check_token_budget(
                self._total_tokens_per_task.get(task.id, 0)
            )
            if not budget_check.passed:
                logger.warning("Token budget exceeded for task '%s'", task.title)
                self._emit_agent_packet(
                    task=task,
                    attempt=current_attempt,
                    trace_id=attempt_trace_id,
                    packet_type=PacketType.ERROR,
                    run_phase=RunPhase.EXECUTION,
                    sender_role="orchestrator",
                    recipient_roles=["builder"],
                    payload={
                        "error_code": "token_budget_exceeded",
                        "message": "Task exceeded max token budget.",
                        "retryable": False,
                        "details": {
                            "tokens_used": self._total_tokens_per_task.get(task.id, 0),
                            "token_budget": self.config.orchestrator.max_tokens_per_task,
                        },
                    },
                    symbolic_keys=["budget", "tokens"],
                    capability_tags=["governance", "budget"],
                )
                self._persist_retry_telemetry(
                    task=task,
                    attempt=current_attempt,
                    trace_id=attempt_trace_id,
                    retry_cause="token_budget_exceeded",
                    retrieval_confidence=retrieval_confidence,
                    retrieval_conflicts=retrieval_conflicts,
                )
                self.task_router.mark_stuck(task, "token_budget_exceeded")
                self._complete_attempt_diagnostics(
                    outcome="failure",
                    error_summary="token_budget_exceeded",
                )
                return task
            cost_ok, estimated_cost = self._check_sotappr_cost_budget(task)
            if not cost_ok:
                logger.warning(
                    "SOTAppR estimated cost budget exceeded for task '%s' (estimated=$%.4f)",
                    task.title,
                    estimated_cost,
                )
                self._emit_agent_packet(
                    task=task,
                    attempt=current_attempt,
                    trace_id=attempt_trace_id,
                    packet_type=PacketType.ERROR,
                    run_phase=RunPhase.EXECUTION,
                    sender_role="orchestrator",
                    recipient_roles=["builder"],
                    payload={
                        "error_code": "sotappr_cost_budget_exceeded",
                        "message": "Task exceeded estimated runtime cost budget.",
                        "retryable": False,
                        "details": {
                            "estimated_cost_usd": float(estimated_cost),
                            "max_cost_usd": self.config.sotappr.max_estimated_cost_per_task_usd,
                        },
                    },
                    symbolic_keys=["budget", "cost"],
                    capability_tags=["governance", "budget"],
                )
                self._persist_retry_telemetry(
                    task=task,
                    attempt=current_attempt,
                    trace_id=attempt_trace_id,
                    retry_cause="sotappr_cost_budget_exceeded",
                    retrieval_confidence=retrieval_confidence,
                    retrieval_conflicts=retrieval_conflicts,
                    details=f"estimated_cost_usd={estimated_cost:.6f}",
                )
                self.task_router.mark_stuck(task, "sotappr_cost_budget_exceeded")
                self._complete_attempt_diagnostics(
                    outcome="failure",
                    error_summary="sotappr_cost_budget_exceeded",
                )
                return task

            if builder_result.status != "success":
                # Tests failed — rewind and log hypothesis
                self._notify("  Builder: FAILED — rewinding to checkpoint")
                retry_detail = self._derive_failure_text(
                    build_result=build_result,
                    agent_error=builder_result.error,
                )
                self._emit_agent_packet(
                    task=task,
                    attempt=current_attempt,
                    trace_id=attempt_trace_id,
                    packet_type=PacketType.ERROR,
                    run_phase=RunPhase.EXECUTION,
                    sender_role="builder",
                    recipient_roles=["orchestrator", "state_manager"],
                    payload={
                        "error_code": build_result.failure_reason or "builder_failed",
                        "message": retry_detail[:500],
                        "retryable": True,
                        "details": {
                            "files_changed": len(build_result.files_changed or []),
                            "tests_passed": bool(build_result.tests_passed),
                        },
                    },
                    symbolic_keys=["builder", "failure"],
                    capability_tags=["execution", "self-correction"],
                )
                self._record_llm_health_signal(
                    build_result=build_result,
                    agent_error=builder_result.error,
                )
                if self._is_no_output_failure(build_result):
                    self._record_no_output_failure(task)
                else:
                    self._clear_no_output_state(task.id)
                retry_cause = build_result.failure_reason or (
                    "no_output_failure" if self._is_no_output_failure(build_result) else "builder_failed"
                )
                self._persist_retry_telemetry(
                    task=task,
                    attempt=current_attempt,
                    trace_id=attempt_trace_id,
                    retry_cause=retry_cause,
                    retrieval_confidence=retrieval_confidence,
                    retrieval_conflicts=retrieval_conflicts,
                    details=retry_detail,
                )
                self._handle_failure(
                    task,
                    build_data,
                    current_attempt,
                    checkpoint_ref,
                    agent_error=builder_result.error,
                )
                self._persist_model_failover_snapshot(
                    task=task,
                    attempt=current_attempt,
                    trace_id=attempt_trace_id,
                )
                self._complete_attempt_diagnostics(outcome="failure", error_summary=retry_detail)
                self._check_escalation(task, current_attempt, council_threshold, max_council)
                if task.status == TaskStatus.STUCK:
                    return task
                continue

            # Builder succeeded — run Sentinel audit (Phase 3, optional)
            self._clear_no_output_state(task.id)
            governance_violation = self._sotappr_governance_violation(task, build_data)
            if governance_violation is not None:
                logger.warning("SOTAppR governance rejected task '%s': %s", task.title, governance_violation)
                self._emit_agent_packet(
                    task=task,
                    attempt=current_attempt,
                    trace_id=attempt_trace_id,
                    packet_type=PacketType.EVAL,
                    run_phase=RunPhase.VALIDATION,
                    sender_role="orchestrator",
                    recipient_roles=["builder"],
                    payload={
                        "metric_set": {
                            "rtf": 1.0,
                            "stability": 1.0,
                            "audit": 1.0,
                            "outcome": 0.0,
                            "learnability": 1.0,
                            "generalization_ratio": 0.0,
                            "drift_risk": 1.0,
                        },
                        "thresholds": {
                            "rtf": 1.0,
                            "stability": 0.9,
                            "audit": 1.0,
                            "outcome": 0.5,
                            "learnability": 0.7,
                            "generalization_ratio": 0.8,
                            "drift_risk": 0.3,
                        },
                        "pass": False,
                        "notes": [f"governance_rejected: {governance_violation}"],
                    },
                    proof_bundle={
                        "evidence_refs": [f"task://{task.id}/attempt/{current_attempt}/governance"],
                        "falsifiers": [],
                        "gate_scores": {"outcome": 0.0},
                        "signatures": [{"agent": "orchestrator", "verdict": "reject"}],
                    },
                    symbolic_keys=["governance", "validation"],
                    capability_tags=["policy", "validation"],
                    verification_pass=False,
                )
                self._persist_retry_telemetry(
                    task=task,
                    attempt=current_attempt,
                    trace_id=attempt_trace_id,
                    retry_cause="sotappr_governance_rejected",
                    retrieval_confidence=retrieval_confidence,
                    retrieval_conflicts=retrieval_conflicts,
                    details=governance_violation,
                )
                self._handle_failure(
                    task,
                    build_data,
                    current_attempt,
                    checkpoint_ref,
                    error_override=f"SOTAppR governance rejected: {governance_violation}",
                )
                self._persist_model_failover_snapshot(
                    task=task,
                    attempt=current_attempt,
                    trace_id=attempt_trace_id,
                )
                self._complete_attempt_diagnostics(
                    outcome="failure",
                    error_summary=f"sotappr_governance_rejected: {governance_violation}",
                )
                self._check_escalation(task, current_attempt, council_threshold, max_council)
                if task.status == TaskStatus.STUCK:
                    return task
                continue

            if self.sentinel is not None:
                self._notify("  Sentinel: auditing diff (6 checks)...")
                self.task_router.transition(task, TaskStatus.REVIEWING, reason="sentinel audit")
                precomputed = builder_result.data.get("precomputed_sentinel_result") if builder_result.data else None
                if precomputed:
                    sentinel_result = AgentResult(**precomputed)
                else:
                    sentinel_input = {
                        "build_result": build_data,
                        "context_brief": selected_builder_input if isinstance(selected_builder_input, dict) else None,
                        "task": task,
                        "self_audit": builder_result.data.get("self_audit", "") if builder_result.data else "",
                    }
                    # If builder_input is a ContextBrief, pass it properly
                    if isinstance(selected_builder_input, ContextBrief):
                        sentinel_input["context_brief"] = selected_builder_input

                    sentinel_result = self.sentinel.run(sentinel_input)

                if sentinel_result.status != "success":
                    self._record_llm_health_signal(
                        build_result=BuildResult(**build_data) if build_data else BuildResult(),
                        agent_error=sentinel_result.error,
                    )
                    # Sentinel rejected — extract violations and feed back
                    verdict_data = sentinel_result.data.get("sentinel_verdict", {})
                    violations = verdict_data.get("violations", [])
                    violation_summary = "; ".join(
                        v.get("detail", "unknown")[:80] for v in violations[:3]
                    )
                    self._notify(f"  Sentinel: REJECTED ({len(violations)} violation(s))")
                    logger.warning(
                        "Sentinel rejected task '%s': %s", task.title, violation_summary
                    )
                    self._emit_agent_packet(
                        task=task,
                        attempt=current_attempt,
                        trace_id=attempt_trace_id,
                        packet_type=PacketType.EVAL,
                        run_phase=RunPhase.VALIDATION,
                        sender_role="sentinel",
                        recipient_roles=["builder", "orchestrator"],
                        payload={
                            "metric_set": {
                                "rtf": 1.0,
                                "stability": 1.0,
                                "audit": 1.0,
                                "outcome": 0.0,
                                "learnability": 1.0,
                                "generalization_ratio": 0.0,
                                "drift_risk": 1.0,
                            },
                            "thresholds": {
                                "rtf": 1.0,
                                "stability": 0.9,
                                "audit": 1.0,
                                "outcome": 0.5,
                                "learnability": 0.7,
                                "generalization_ratio": 0.8,
                                "drift_risk": 0.3,
                            },
                            "pass": False,
                            "notes": [violation_summary or "sentinel_rejected"],
                        },
                        proof_bundle={
                            "evidence_refs": [f"task://{task.id}/attempt/{current_attempt}/sentinel"],
                            "falsifiers": [],
                            "gate_scores": {"outcome": 0.0},
                            "signatures": [{"agent": "sentinel", "verdict": "reject"}],
                        },
                        symbolic_keys=["sentinel", "validation"],
                        capability_tags=["audit", "policy"],
                        verification_pass=False,
                    )
                    self._persist_retry_telemetry(
                        task=task,
                        attempt=current_attempt,
                        trace_id=attempt_trace_id,
                        retry_cause="sentinel_rejected",
                        retrieval_confidence=retrieval_confidence,
                        retrieval_conflicts=retrieval_conflicts,
                        details=violation_summary,
                    )
                    self._handle_failure(
                        task, build_data, current_attempt, checkpoint_ref,
                        error_override=f"Sentinel rejected: {violation_summary}",
                    )
                    self._persist_model_failover_snapshot(
                        task=task,
                        attempt=current_attempt,
                        trace_id=attempt_trace_id,
                    )
                    self._complete_attempt_diagnostics(
                        outcome="failure",
                        error_summary=f"sentinel_rejected: {violation_summary}",
                    )
                    self._check_escalation(task, current_attempt, council_threshold, max_council)
                    if task.status == TaskStatus.STUCK:
                        return task
                    continue

            # All checks passed — commit and mark done
            self._emit_agent_packet(
                task=task,
                attempt=current_attempt,
                trace_id=attempt_trace_id,
                packet_type=PacketType.EVAL,
                run_phase=RunPhase.VALIDATION,
                sender_role="orchestrator" if self.sentinel is None else "sentinel",
                recipient_roles=["builder", "orchestrator"],
                payload={
                    "metric_set": {
                        "rtf": 1.0,
                        "stability": 1.0,
                        "audit": 1.0,
                        "outcome": 1.0,
                        "learnability": 1.0,
                        "generalization_ratio": 1.0,
                        "drift_risk": 0.0,
                    },
                    "thresholds": {
                        "rtf": 1.0,
                        "stability": 0.9,
                        "audit": 1.0,
                        "outcome": 0.5,
                        "learnability": 0.7,
                        "generalization_ratio": 0.8,
                        "drift_risk": 0.3,
                    },
                    "pass": True,
                    "notes": ["build_and_validation_passed"],
                },
                proof_bundle={
                    "evidence_refs": [f"task://{task.id}/attempt/{current_attempt}/build_result"],
                    "falsifiers": [],
                    "gate_scores": {
                        "outcome": 1.0,
                        "stability": 1.0,
                        "audit": 1.0,
                    },
                    "signatures": [{"agent": "orchestrator", "verdict": "pass"}],
                },
                symbolic_keys=["validation", "promotion"],
                capability_tags=["quality-gate", "promotion"],
            )
            self._emit_agent_packet(
                task=task,
                attempt=current_attempt,
                trace_id=attempt_trace_id,
                packet_type=PacketType.DECISION,
                run_phase=RunPhase.PROMOTION,
                sender_role="orchestrator",
                recipient_roles=["builder", "state_manager"],
                payload={
                    "decision_id": f"decision-{str(task.id)[:8]}-{current_attempt}",
                    "recommendation": "promote_and_commit",
                    "priority_distribution": {
                        "correctness": 0.5,
                        "safety": 0.3,
                        "speed": 0.2,
                    },
                    "next_actions": [
                        "commit_changes",
                        "transition_task_to_done",
                    ],
                    "commitment_level": "final",
                },
                symbolic_keys=["decision", "promotion"],
                capability_tags=["governance", "execution"],
                promote=True,
            )
            self._run_transfer_promotion(
                task=task,
                attempt=current_attempt,
                trace_id=attempt_trace_id,
                retrieval_confidence=retrieval_confidence,
            )
            self._persist_model_failover_snapshot(
                task=task,
                attempt=current_attempt,
                trace_id=attempt_trace_id,
            )
            self._record_trace_event(
                task=task,
                attempt=current_attempt,
                trace_id=attempt_trace_id,
                event_type="attempt_completed",
                payload={"result": "success"},
                persist_artifact=True,
            )
            self._emit_agent_packet(
                task=task,
                attempt=current_attempt,
                trace_id=attempt_trace_id,
                packet_type=PacketType.SIGNAL,
                run_phase=RunPhase.VALIDATION,
                sender_role="orchestrator",
                recipient_roles=["builder", "state_manager"],
                payload={
                    "signal": "ATTEMPT_COMPLETED",
                    "reason": "success",
                },
                symbolic_keys=["attempt", "completion"],
                capability_tags=["orchestration", "control-plane"],
            )
            self._complete_attempt_diagnostics(outcome="success")
            return self._handle_success(task, build_data, current_attempt)

        # Exhausted all retries — attempt WrapUp salvage (Item 9)
        logger.warning("Task '%s' exhausted %d retries", task.title, max_retries)
        self._emit_agent_packet(
            task=task,
            attempt=int(task.attempt_count),
            trace_id=self._build_attempt_trace_id(task, int(task.attempt_count)),
            packet_type=PacketType.DECISION,
            run_phase=RunPhase.ARBITRATION,
            sender_role="orchestrator",
            recipient_roles=["builder", "state_manager"],
            payload={
                "decision_id": f"decision-{str(task.id)[:8]}-exhausted",
                "recommendation": "mark_stuck",
                "priority_distribution": {
                    "correctness": 0.6,
                    "safety": 0.4,
                },
                "next_actions": ["mark_task_stuck", "wrapup_salvage_if_enabled"],
                "commitment_level": "terminal",
            },
            symbolic_keys=["decision", "exhausted_retries"],
            capability_tags=["orchestration", "escalation"],
            promote=False,
        )
        if getattr(getattr(self.config, "sotappr", None), "enable_wrapup_workflow", False):
            self._attempt_wrapup_salvage(task)
        self.task_router.mark_stuck(task, f"exhausted {max_retries} retries")
        return task

    def _check_escalation(
        self, task: Task, attempt: int, council_threshold: int, max_council: int
    ) -> None:
        """Check if task should escalate to council or be marked stuck.

        Escalation ladder (Dirac P5 — Broadcast Escalation):
        1. Retry with forbidden approaches (existing, attempts 1-N)
        2. Council peer review (after council_trigger_threshold failures)
        3. Mark STUCK (after council_max_invocations councils)
        """
        # Check if we should trigger council
        if self.task_router.should_trigger_council(task, council_threshold):
            logger.info("Task '%s' triggering council after %d failures", task.title, attempt)
            self._notify("  Council: requesting 2nd-opinion diagnosis...")
            self.task_router.increment_council(task)
            self._run_council(task)

        # Check if we should mark stuck
        if self.task_router.should_mark_stuck(task, max_council):
            logger.warning(
                "Task '%s' marked STUCK after %d council invocations",
                task.title, task.council_count,
            )
            self.diagnostics.dump_to_file(
                Path(self.state_manager.repo_path) / "diagnostics",
                task_id=task.id,
            )
            self.task_router.mark_stuck(task, "council_exhausted")

    def _attempt_wrapup_salvage(self, task: Task) -> None:
        """Attempt to salvage partial outputs on retry exhaustion (Item 9)."""
        # Get the last build result from the most recent hypothesis entry
        try:
            failed = self.repository.get_failed_approaches(task.id)
            if not failed:
                return
            last_entry = failed[-1]
            last_build = BuildResult(
                files_changed=last_entry.files_changed or [],
                approach_summary=last_entry.approach_summary,
            )
        except Exception as e:
            logger.warning("WrapUp: failed to load last build result: %s", e)
            return

        wrapup = WrapUpWorkflow(
            council=self.council,
            sentinel=self.sentinel,
        )
        salvage_result = wrapup.attempt_salvage(task, last_build)
        if salvage_result.get("salvaged"):
            self._notify(f"  WrapUp: salvaged partial output ({len(salvage_result.get('files', []))} files)")
            logger.info(
                "WrapUp salvaged partial output for task '%s': %s",
                task.title, salvage_result.get("reason"),
            )
        else:
            logger.info(
                "WrapUp could not salvage task '%s': %s",
                task.title, salvage_result.get("reason"),
            )

    def _run_council(self, task: Task) -> None:
        """Execute the Council agent for peer review diagnosis.

        Builds input from hypothesis_log entries + task description,
        calls the Council, stores the diagnosis, and caches it for
        the next Builder attempt.
        """
        if self.council is None:
            logger.info("Council agent not configured — skipping peer review")
            return

        # Gather failed approaches from hypothesis log
        try:
            failed_approaches = self.repository.get_failed_approaches(task.id)
        except Exception as e:
            logger.error("Failed to load hypothesis log for council: %s", e)
            failed_approaches = []

        # Build council input
        council_input = {
            "task": task,
            "hypothesis_log": [
                h.model_dump() if hasattr(h, "model_dump") else h
                for h in failed_approaches
            ],
            "project_rules": None,
        }

        # Call council
        council_result = self.council.run(council_input)

        # Track tokens
        tokens = council_result.data.get("tokens_used", 0) if council_result.data else 0
        self._total_tokens_per_task[task.id] = self._total_tokens_per_task.get(task.id, 0) + tokens

        if council_result.status == "success" and council_result.data:
            diag_data = council_result.data.get("council_diagnosis", {})
            if diag_data:
                diagnosis = CouncilDiagnosis(**diag_data)
                logger.info(
                    "Council diagnosis: strategy_shift='%s', new_approach='%s'",
                    diagnosis.strategy_shift[:80], diagnosis.new_approach[:80],
                )

                # Cache for the next Builder attempt
                self._council_diagnosis_cache[task.id] = diagnosis.new_approach

                # Store in database
                try:
                    peer_review = PeerReview(
                        task_id=task.id,
                        model_used=diagnosis.model_used,
                        diagnosis=diagnosis.strategy_shift,
                        recommended_approach=diagnosis.new_approach,
                        reasoning=diagnosis.reasoning,
                    )
                    self.repository.save_peer_review(peer_review)
                except Exception as e:
                    logger.error("Failed to save peer review: %s", e)
        else:
            logger.warning("Council failed: %s", council_result.error)

    def _handle_success(self, task: Task, build_data: dict, attempt: int) -> Task:
        """Handle successful build — commit and mark done."""
        # Log successful hypothesis
        build_result = BuildResult(**build_data) if build_data else BuildResult()
        self._log_hypothesis(
            task, attempt,
            approach=build_result.approach_summary or "Successful build",
            error=None,
            outcome=HypothesisOutcome.SUCCESS,
            files=build_result.files_changed,
            model=build_result.model_used,
        )

        # Git commit
        try:
            commit_hash = commit(
                self.state_manager.repo_path,
                message=f"[associate] {task.title} (attempt #{attempt})",
                files=build_result.files_changed if build_result.files_changed else None,
            )
            if commit_hash:
                logger.info("Committed: %s", commit_hash[:8])
        except Exception as e:
            logger.warning("Git commit failed (non-fatal): %s", e)

        # Mark task as done (or leave in review queue when configured).
        # Sentinel may have already moved the task to REVIEWING.
        if task.status != TaskStatus.REVIEWING:
            self.task_router.transition(task, TaskStatus.REVIEWING, reason="tests passed")
        if self.config.sotappr.require_human_review_before_done:
            logger.info("Task '%s' waiting for human approval (SOTAppR review gate)", task.title)
            return task
        self.task_router.mark_done(task)
        logger.info("Task '%s' DONE on attempt %d", task.title, attempt)

        return task

    def _handle_failure(
        self,
        task: Task,
        build_data: dict,
        attempt: int,
        checkpoint_ref: str | None,
        error_override: str | None = None,
        agent_error: str | None = None,
    ) -> None:
        """Handle failed build — rewind to checkpoint and log hypothesis."""
        build_result = BuildResult(**build_data) if build_data else BuildResult()

        # Extract error signature
        error_text = self._derive_failure_text(
            build_result=build_result,
            error_override=error_override,
            agent_error=agent_error,
        )
        error_sig = normalize_error_signature(error_text[:500])

        # Log hypothesis
        self._log_hypothesis(
            task, attempt,
            approach=build_result.approach_summary or "Failed build attempt",
            error=error_text,
            error_signature=error_sig,
            files=build_result.files_changed,
            model=build_result.model_used,
        )

        # Rewind to checkpoint
        try:
            self.state_manager.rewind_to_checkpoint(task, checkpoint_ref)
        except Exception as e:
            logger.error("Rewind failed: %s", e)

    def _log_hypothesis(
        self,
        task: Task,
        attempt: int,
        approach: str,
        error: str | None,
        outcome: HypothesisOutcome = HypothesisOutcome.FAILURE,
        error_signature: str | None = None,
        files: list[str] | None = None,
        model: str | None = None,
    ) -> None:
        """Record a hypothesis in the database."""
        entry = HypothesisEntry(
            task_id=task.id,
            attempt_number=attempt,
            approach_summary=approach,
            outcome=outcome,
            error_signature=error_signature,
            error_full=error[:2000] if error else None,
            files_changed=files or [],
            model_used=model,
        )
        try:
            self.repository.log_hypothesis(entry)
        except Exception as e:
            logger.error("Failed to log hypothesis: %s", e)

    def _derive_failure_text(
        self,
        build_result: BuildResult,
        error_override: str | None = None,
        agent_error: str | None = None,
    ) -> str:
        if error_override:
            return error_override
        if agent_error:
            return agent_error
        if build_result.failure_detail:
            return build_result.failure_detail
        if build_result.test_output:
            return build_result.test_output
        if not build_result.files_changed:
            reason = build_result.failure_reason or "no_file_changes"
            return f"Builder failed without writing files ({reason})"
        return "Unknown build failure"

    def _is_no_output_failure(self, build_result: BuildResult) -> bool:
        return not build_result.files_changed and not build_result.tests_passed

    def _record_no_output_failure(self, task: Task) -> None:
        streak = self._no_output_attempts.get(task.id, 0) + 1
        self._no_output_attempts[task.id] = streak
        self._forced_builder_strategy[task.id] = (
            "Execution guardrail: the previous attempt produced no file changes. "
            "Use test-contract-first implementation. Produce at least one "
            "'--- FILE: path ---' block with complete code. If blocked, state "
            "the exact blocker in the Approach section and stop."
        )
        logger.warning(
            "Task '%s' had no output artifacts (streak=%d); forcing implementation guardrail",
            task.title,
            streak,
        )

    def _record_llm_health_signal(self, build_result: BuildResult, agent_error: str | None = None) -> None:
        """Update health monitor only for provider/transport-class failures."""
        failure_text = self._derive_failure_text(
            build_result=build_result,
            agent_error=agent_error,
        )
        if not self._looks_like_llm_provider_failure(failure_text):
            return
        self.health_monitor.record_llm_failure()

    def _run_builder_with_arbitration(
        self,
        task: Task,
        builder_input: TaskContext | ContextBrief | dict,
    ) -> tuple[AgentResult, BuildResult, int, TaskContext | ContextBrief | dict]:
        candidate_inputs = [builder_input]
        if self._should_expand_builder_candidates(task, builder_input):
            candidate_inputs = self._build_builder_candidates(task, builder_input)
            self._notify(f"  Builder: evaluating {len(candidate_inputs)} candidates")

        candidate_results: list[tuple[AgentResult, BuildResult, TaskContext | ContextBrief | dict]] = []
        total_tokens = 0
        for idx, candidate_input in enumerate(candidate_inputs, start=1):
            if len(candidate_inputs) > 1:
                self._notify(f"    Candidate {idx}/{len(candidate_inputs)}")
            result = self.builder.run(candidate_input)
            tokens = result.data.get("tokens_used", 0) if result.data else 0
            total_tokens += tokens
            build_data = result.data.get("build_result", {}) if result.data else {}
            build_result = BuildResult(**build_data) if build_data else BuildResult()
            candidate_results.append((result, build_result, candidate_input))

        if len(candidate_results) == 1:
            result, build_result, selected_input = candidate_results[0]
            return result, build_result, total_tokens, selected_input

        retrieval_confidence, _ = self._extract_retrieval_signals(builder_input)
        decision = self.arbiter.choose(
            [(res, br) for res, br, _ in candidate_results],
            retrieval_confidence=retrieval_confidence,
            vetoes=self._collect_candidate_vetoes(task, candidate_results),
        )
        selected_result, selected_build_result, selected_input = candidate_results[decision.selected_index - 1]
        if selected_result.data is None:
            selected_result.data = {}
        selected_result.data["arbitration"] = decision.to_dict()
        selected_result.data["arbitration"]["candidate_count"] = len(candidate_results)
        self._persist_arbitration_event(
            task=task,
            candidate_count=len(candidate_results),
            retrieval_confidence=retrieval_confidence,
            decision_payload=dict(selected_result.data["arbitration"]),
            selected_build_result=selected_build_result,
        )
        logger.info(
            "Arbitration selected candidate %d/%d (score=%.3f)",
            decision.selected_index,
            len(candidate_results),
            decision.scores[decision.selected_index - 1].total,
        )
        return selected_result, selected_build_result, total_tokens, selected_input

    def _clear_no_output_state(self, task_id: uuid.UUID) -> None:
        self._no_output_attempts.pop(task_id, None)
        self._forced_builder_strategy.pop(task_id, None)

    def _apply_forced_strategy(self, task: Task, builder_input: TaskContext | ContextBrief | dict):
        strategy = self._forced_builder_strategy.get(task.id)
        if not strategy:
            return builder_input

        if isinstance(builder_input, TaskContext):
            merged = self._merge_strategy_hint(builder_input.previous_council_diagnosis, strategy)
            return builder_input.model_copy(update={"previous_council_diagnosis": merged})

        if isinstance(builder_input, ContextBrief):
            merged = self._merge_strategy_hint(builder_input.council_diagnosis, strategy)
            return builder_input.model_copy(update={"council_diagnosis": merged})

        if isinstance(builder_input, dict):
            merged = self._merge_strategy_hint(builder_input.get("council_diagnosis"), strategy)
            updated = dict(builder_input)
            updated["council_diagnosis"] = merged
            return updated

        return builder_input

    def _apply_self_correction_guardrail(
        self,
        task: Task,
        builder_input: TaskContext | ContextBrief | dict,
    ) -> TaskContext | ContextBrief | dict:
        cfg = self.config.orchestrator
        if not cfg.enable_prebuild_self_correction:
            return builder_input

        retrieval_confidence, conflict_count = self._extract_retrieval_signals(builder_input)
        low_confidence = (
            retrieval_confidence > 0.0
            and retrieval_confidence < cfg.self_correction_confidence_threshold
        )
        high_conflict = conflict_count >= cfg.self_correction_conflict_threshold
        if not (low_confidence or high_conflict):
            return builder_input

        hint = self._generate_self_correction_hint(
            task=task,
            builder_input=builder_input,
            retrieval_confidence=retrieval_confidence,
            conflict_count=conflict_count,
        )
        self._notify("  Builder: applying self-correction guardrail")
        return self._inject_strategy_hint(builder_input, hint)

    def _generate_self_correction_hint(
        self,
        task: Task,
        builder_input: TaskContext | ContextBrief | dict,
        retrieval_confidence: float,
        conflict_count: int,
    ) -> str:
        existing_strategy = self._extract_strategy_context(builder_input)
        conflicts = self._extract_retrieval_conflicts(builder_input)
        lines = [
            "Self-correction guardrail: retrieval confidence/conflicts indicate uncertainty.",
            f"Task focus: {task.title}.",
            (
                "Before writing code, restate concrete acceptance checks, choose a single canonical API/"
                "version, and implement the smallest test-first change set that validates behavior."
            ),
            (
                f"Signal snapshot: confidence={retrieval_confidence:.2f}, conflicts={conflict_count}."
            ),
        ]
        if conflicts:
            lines.append("Conflicts to resolve first:")
            for conflict in conflicts[:3]:
                lines.append(f"- {conflict}")
        if existing_strategy:
            lines.append("Preserve prior strategy guidance while resolving these conflicts.")
        return "\n".join(lines)

    def _collect_candidate_vetoes(
        self,
        task: Task,
        candidate_results: list[tuple[AgentResult, BuildResult, TaskContext | ContextBrief | dict]],
    ) -> dict[int, list[str]]:
        vetoes: dict[int, list[str]] = {}
        for idx, (agent_result, build_result, candidate_input) in enumerate(candidate_results, start=1):
            reasons: list[str] = []
            if self._should_council_veto_candidate(task, build_result):
                reasons.append("council_veto:no_output_after_council_diagnosis")

            sentinel_reason, sentinel_result = self._sentinel_veto_for_candidate(
                task=task,
                agent_result=agent_result,
                build_result=build_result,
                candidate_input=candidate_input,
            )
            if sentinel_reason:
                reasons.append(sentinel_reason)
            if sentinel_result is not None:
                if agent_result.data is None:
                    agent_result.data = {}
                agent_result.data["precomputed_sentinel_result"] = sentinel_result.model_dump()

            if reasons:
                vetoes[idx] = reasons
        return vetoes

    def _should_council_veto_candidate(self, task: Task, build_result: BuildResult) -> bool:
        if task.id not in self._council_diagnosis_cache:
            return False
        # If council guidance exists, zero-output candidates are vetoed so arbitration
        # is forced toward executable implementation paths.
        return not build_result.files_changed

    def _sentinel_veto_for_candidate(
        self,
        *,
        task: Task,
        agent_result: AgentResult,
        build_result: BuildResult,
        candidate_input: TaskContext | ContextBrief | dict,
    ) -> tuple[str | None, AgentResult | None]:
        if self.sentinel is None:
            return None, None
        if agent_result.status != "success":
            return None, None

        sentinel_input = {
            "build_result": build_result.model_dump(),
            "context_brief": candidate_input if isinstance(candidate_input, dict) else None,
            "task": task,
            "self_audit": agent_result.data.get("self_audit", "") if agent_result.data else "",
        }
        if isinstance(candidate_input, ContextBrief):
            sentinel_input["context_brief"] = candidate_input

        sentinel_result = self.sentinel.run(sentinel_input)
        if sentinel_result.status == "success":
            return None, sentinel_result

        verdict_data = sentinel_result.data.get("sentinel_verdict", {}) if sentinel_result.data else {}
        violations = verdict_data.get("violations", []) if isinstance(verdict_data, dict) else []
        details = "; ".join(v.get("detail", "unknown")[:80] for v in violations[:2])
        summary = details or (sentinel_result.error or "sentinel_rejected")
        return f"sentinel_veto:{summary}", None

    def _should_expand_builder_candidates(
        self,
        task: Task,
        builder_input: TaskContext | ContextBrief | dict,
    ) -> bool:
        cfg = self.config.orchestrator
        if not cfg.enable_multi_candidate_arbitration:
            return False
        if cfg.arbitration_candidate_count < 2:
            return False
        if task.attempt_count > 1:
            return True

        retrieval_confidence, conflict_count = self._extract_retrieval_signals(builder_input)
        if retrieval_confidence < cfg.arbitration_low_confidence_threshold:
            return True
        if conflict_count >= cfg.arbitration_conflict_trigger_count:
            return True
        if self._build_lineage_strategy_hint(task) is not None:
            return True
        return bool(self._extract_strategy_context(builder_input))

    def _build_builder_candidates(
        self,
        task: Task,
        builder_input: TaskContext | ContextBrief | dict,
    ) -> list[TaskContext | ContextBrief | dict]:
        requested = max(1, self.config.orchestrator.arbitration_candidate_count)
        candidate_count = min(requested, len(self._ARBITRATION_STRATEGY_HINTS) + 1)
        candidates: list[TaskContext | ContextBrief | dict] = [builder_input]
        lineage_hint = self._build_lineage_strategy_hint(task)
        for idx in range(candidate_count - 1):
            hint = self._ARBITRATION_STRATEGY_HINTS[idx]
            if idx == 0 and lineage_hint:
                hint = f"{hint}\n{lineage_hint}"
            scoped_hint = (
                f"{hint} Task focus: {task.title}. "
                "Resolve this using a distinct implementation path from other candidates."
            )
            candidates.append(self._inject_strategy_hint(builder_input, scoped_hint))
        if lineage_hint and len(candidates) == 1:
            candidates[0] = self._inject_strategy_hint(builder_input, lineage_hint)
        return candidates

    def _build_lineage_strategy_hint(self, task: Task) -> str | None:
        if not hasattr(self.repository, "get_packet_lineage_summary"):
            return None
        try:
            summary_rows = self.repository.get_packet_lineage_summary(
                project_id=task.project_id,
                limit=5,
            )
        except Exception:
            return None
        if not summary_rows:
            return None

        top = summary_rows[0]
        protocol_id = str(top.get("protocol_id") or "(none)")
        transfer_count = int(top.get("transfer_count") or 0)
        cross_count = int(top.get("cross_use_case_count") or 0)
        mode = "unknown"
        if hasattr(self.repository, "list_protocol_propagation"):
            try:
                propagation = self.repository.list_protocol_propagation(
                    protocol_id=protocol_id if protocol_id != "(none)" else None,
                    project_id=task.project_id,
                    limit=10,
                )
            except Exception:
                propagation = []
            if propagation:
                mode = str(propagation[0].get("transfer_mode") or "unknown")

        return (
            "Lineage strategy bias: prioritize approaches consistent with successful protocol "
            f"'{protocol_id}' (transfers={transfer_count}, cross_use_case={cross_count}, mode={mode}). "
            "Prefer abstractions that preserve transferability and avoid brittle local-only coupling."
        )

    def _inject_strategy_hint(
        self,
        builder_input: TaskContext | ContextBrief | dict,
        hint: str,
    ) -> TaskContext | ContextBrief | dict:
        if isinstance(builder_input, TaskContext):
            merged = self._merge_strategy_hint(builder_input.previous_council_diagnosis, hint)
            return builder_input.model_copy(update={"previous_council_diagnosis": merged})

        if isinstance(builder_input, ContextBrief):
            merged = self._merge_strategy_hint(builder_input.council_diagnosis, hint)
            return builder_input.model_copy(update={"council_diagnosis": merged})

        if isinstance(builder_input, dict):
            merged = self._merge_strategy_hint(builder_input.get("council_diagnosis"), hint)
            updated = dict(builder_input)
            updated["council_diagnosis"] = merged
            return updated

        return builder_input

    @staticmethod
    def _extract_retrieval_signals(builder_input: TaskContext | ContextBrief | dict) -> tuple[float, int]:
        if isinstance(builder_input, ContextBrief):
            return float(builder_input.retrieval_confidence), len(builder_input.retrieval_conflicts)
        if isinstance(builder_input, dict):
            confidence = float(builder_input.get("retrieval_confidence", 0.0) or 0.0)
            conflicts = builder_input.get("retrieval_conflicts") or []
            return confidence, len(conflicts)
        return 0.0, 0

    @staticmethod
    def _extract_strategy_context(builder_input: TaskContext | ContextBrief | dict) -> str | None:
        if isinstance(builder_input, TaskContext):
            return builder_input.previous_council_diagnosis
        if isinstance(builder_input, ContextBrief):
            return builder_input.council_diagnosis
        if isinstance(builder_input, dict):
            return builder_input.get("council_diagnosis")
        return None

    @staticmethod
    def _extract_retrieval_conflicts(builder_input: TaskContext | ContextBrief | dict) -> list[str]:
        if isinstance(builder_input, ContextBrief):
            return list(builder_input.retrieval_conflicts or [])
        if isinstance(builder_input, dict):
            raw_conflicts = builder_input.get("retrieval_conflicts") or []
            if isinstance(raw_conflicts, list):
                return [str(item) for item in raw_conflicts]
        return []

    @staticmethod
    def _merge_strategy_hint(existing: str | None, hint: str) -> str:
        if not existing:
            return hint
        if hint in existing:
            return existing
        return f"{existing}\n\n{hint}"

    def _project_transfer_protocol_id(self, task: Task) -> str:
        return f"project-{str(task.project_id)[:8]}-promotion"

    @staticmethod
    def _effectiveness_score_from_row(row: dict[str, Any]) -> float:
        transfer_count = int(row.get("transfer_count") or 0)
        accepted_count = int(row.get("accepted_count") or 0)
        acceptance_rate = (accepted_count / transfer_count) if transfer_count > 0 else 0.0
        avg_outcome = float(row.get("avg_outcome") or 0.0)
        avg_drift = float(row.get("avg_drift_risk") or 1.0)
        score = (0.5 * acceptance_rate) + (0.35 * avg_outcome) + (0.15 * (1.0 - avg_drift))
        return float(score)

    def _passes_protocol_effectiveness_gate(
        self,
        *,
        task: Task,
        protocol_id: str,
    ) -> tuple[bool, str | None]:
        cfg = self.config.sotappr
        if not getattr(cfg, "enable_protocol_effectiveness_gate", True):
            return True, None
        if not hasattr(self.repository, "get_protocol_effectiveness"):
            return True, None
        try:
            rows = self.repository.get_protocol_effectiveness(
                project_id=task.project_id,
                limit=100,
            )
        except Exception:
            return True, None
        target = next((row for row in rows if str(row.get("protocol_id")) == protocol_id), None)
        if target is None:
            return True, None
        transfer_count = int(target.get("transfer_count") or 0)
        min_samples = int(getattr(cfg, "min_protocol_effectiveness_samples", 3))
        if transfer_count < min_samples:
            return True, None
        score = self._effectiveness_score_from_row(target)
        threshold = float(getattr(cfg, "min_protocol_effectiveness_score", 0.5))
        if score < threshold:
            return (
                False,
                (
                    f"protocol_effectiveness_gate_blocked: protocol={protocol_id}, "
                    f"score={score:.3f}, threshold={threshold:.3f}, samples={transfer_count}"
                ),
            )
        return True, None

    def _build_transfer_candidates(
        self,
        *,
        task: Task,
        attempt: int,
        retrieval_confidence: float,
    ) -> list[dict[str, Any]]:
        protocol_id = self._project_transfer_protocol_id(task)
        sender_score = max(0.0, min(1.0, float(retrieval_confidence or 0.85)))
        candidates = [
            {
                "protocol_version": "apc/1.0",
                "packet_type": "TRANSFER",
                "channel": "INTER_AGENT",
                "run_phase": "PROMOTION",
                "sender": {
                    "agent_id": "orchestrator-1",
                    "role": "orchestrator",
                    "swarm_id": "associate-main",
                },
                "recipients": [
                    {
                        "agent_id": "builder-1",
                        "role": "builder",
                        "swarm_id": "associate-main",
                    }
                ],
                "trace": {
                    "session_id": self._packet_session_id,
                    "run_id": str(self._active_run_id or self._ensure_run_trace_id()),
                    "task_id": str(task.id),
                    "root_packet_id": str(self._ensure_packet_root(task, attempt)),
                    "generation": 25,
                    "step": self._next_packet_step(task, attempt),
                },
                "routing": {
                    "delivery_mode": "DIRECT",
                    "priority": 5,
                    "ttl_ms": 60_000,
                    "requires_ack": False,
                },
                "confidence": min(1.0, sender_score),
                "symbolic_keys": ["transfer", "promotion", "winner_to_losers"],
                "capability_tags": ["knowledge-transfer", "promotion"],
                "lineage": {
                    "protocol_id": protocol_id,
                    "parent_protocol_ids": [],
                    "ancestor_swarms": ["associate-main"],
                    "cross_use_case": False,
                    "transfer_mode": "WINNER_TO_LOSERS",
                },
                "payload": {
                    "protocol_id": protocol_id,
                    "from_swarm": "associate-main",
                    "to_swarm": "associate-main",
                    "sender_score": sender_score,
                    "receiver_score": max(0.0, min(1.0, sender_score * 0.6)),
                    "transfer_policy": {
                        "mode": "WINNER_TO_LOSERS",
                        "top_k": 2,
                    },
                    "accepted": True,
                },
            },
            {
                "protocol_version": "apc/1.0",
                "packet_type": "TRANSFER",
                "channel": "INTER_AGENT",
                "run_phase": "PROMOTION",
                "sender": {
                    "agent_id": "orchestrator-1",
                    "role": "orchestrator",
                    "swarm_id": "associate-main",
                },
                "recipients": [
                    {
                        "agent_id": "builder-1",
                        "role": "builder",
                        "swarm_id": "associate-main",
                    }
                ],
                "trace": {
                    "session_id": self._packet_session_id,
                    "run_id": str(self._active_run_id or self._ensure_run_trace_id()),
                    "task_id": str(task.id),
                    "root_packet_id": str(self._ensure_packet_root(task, attempt)),
                    "generation": 25,
                    "step": self._next_packet_step(task, attempt),
                },
                "routing": {
                    "delivery_mode": "DIRECT",
                    "priority": 5,
                    "ttl_ms": 60_000,
                    "requires_ack": False,
                },
                "confidence": min(1.0, max(0.0, sender_score * 0.95)),
                "symbolic_keys": ["transfer", "promotion", "commons_seed"],
                "capability_tags": ["knowledge-transfer", "promotion"],
                "lineage": {
                    "protocol_id": protocol_id,
                    "parent_protocol_ids": [],
                    "ancestor_swarms": ["associate-main"],
                    "cross_use_case": False,
                    "transfer_mode": "COMMONS_SEED",
                },
                "payload": {
                    "protocol_id": protocol_id,
                    "from_swarm": "associate-main",
                    "to_swarm": "associate-main",
                    "sender_score": max(0.0, min(1.0, sender_score * 0.98)),
                    "receiver_score": max(0.0, min(1.0, sender_score * 0.7)),
                    "transfer_policy": {
                        "mode": "COMMONS_SEED",
                        "top_k": 2,
                    },
                    "accepted": True,
                },
            },
        ]
        candidate_limit = int(getattr(self.config.sotappr, "transfer_arbitration_candidate_limit", 4))
        if candidate_limit > len(candidates):
            extras = self._historical_transfer_protocol_candidates(
                task=task,
                limit=max(0, candidate_limit - len(candidates)),
            )
            for item in extras:
                extra_protocol_id = str(item.get("protocol_id") or "")
                if not extra_protocol_id or extra_protocol_id == protocol_id:
                    continue
                mode = str(item.get("transfer_mode") or "WINNER_TO_LOSERS")
                if mode not in {"WINNER_TO_LOSERS", "COMMONS_SEED"}:
                    mode = "WINNER_TO_LOSERS"
                confidence = min(1.0, max(0.0, sender_score * float(item.get("confidence_multiplier") or 0.92)))
                receiver_ratio = 0.7 if mode == "COMMONS_SEED" else 0.6
                symbolic_mode = "commons_seed" if mode == "COMMONS_SEED" else "winner_to_losers"
                lineage_cross = bool(item.get("cross_use_case", False))
                parent_protocols = item.get("parent_protocol_ids") or []
                if not isinstance(parent_protocols, list):
                    parent_protocols = []
                ancestor_swarms = item.get("ancestor_swarms") or ["associate-main"]
                if not isinstance(ancestor_swarms, list):
                    ancestor_swarms = ["associate-main"]
                candidates.append(
                    {
                        "protocol_version": "apc/1.0",
                        "packet_type": "TRANSFER",
                        "channel": "INTER_AGENT",
                        "run_phase": "PROMOTION",
                        "sender": {
                            "agent_id": "orchestrator-1",
                            "role": "orchestrator",
                            "swarm_id": "associate-main",
                        },
                        "recipients": [
                            {
                                "agent_id": "builder-1",
                                "role": "builder",
                                "swarm_id": "associate-main",
                            }
                        ],
                        "trace": {
                            "session_id": self._packet_session_id,
                            "run_id": str(self._active_run_id or self._ensure_run_trace_id()),
                            "task_id": str(task.id),
                            "root_packet_id": str(self._ensure_packet_root(task, attempt)),
                            "generation": 25,
                            "step": self._next_packet_step(task, attempt),
                        },
                        "routing": {
                            "delivery_mode": "DIRECT",
                            "priority": 5,
                            "ttl_ms": 60_000,
                            "requires_ack": False,
                        },
                        "confidence": confidence,
                        "symbolic_keys": ["transfer", "promotion", symbolic_mode, "lineage_reuse"],
                        "capability_tags": ["knowledge-transfer", "promotion", "lineage-adaptation"],
                        "lineage": {
                            "protocol_id": extra_protocol_id,
                            "parent_protocol_ids": parent_protocols,
                            "ancestor_swarms": ancestor_swarms,
                            "cross_use_case": lineage_cross,
                            "transfer_mode": mode,
                        },
                        "payload": {
                            "protocol_id": extra_protocol_id,
                            "from_swarm": "associate-main",
                            "to_swarm": "associate-main",
                            "sender_score": sender_score,
                            "receiver_score": max(0.0, min(1.0, sender_score * receiver_ratio)),
                            "transfer_policy": {
                                "mode": mode,
                                "top_k": 2,
                            },
                            "accepted": True,
                        },
                    }
                )
        return candidates

    def _historical_transfer_protocol_candidates(
        self,
        *,
        task: Task,
        limit: int,
    ) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        if not hasattr(self.repository, "get_packet_lineage_summary"):
            return []
        try:
            summary_rows = self.repository.get_packet_lineage_summary(
                project_id=task.project_id,
                limit=max(10, limit * 3),
            )
        except Exception:
            return []
        if not summary_rows:
            return []

        effectiveness_by_protocol: dict[str, float] = {}
        if hasattr(self.repository, "get_protocol_effectiveness"):
            try:
                effectiveness_rows = self.repository.get_protocol_effectiveness(
                    project_id=task.project_id,
                    limit=max(20, limit * 4),
                )
            except Exception:
                effectiveness_rows = []
            for row in effectiveness_rows:
                pid = str(row.get("protocol_id") or "")
                if not pid:
                    continue
                effectiveness_by_protocol[pid] = self._effectiveness_score_from_row(row)

        candidates: list[dict[str, Any]] = []
        for row in summary_rows:
            protocol_id = str(row.get("protocol_id") or "")
            if not protocol_id or protocol_id == "(none)":
                continue
            transfer_mode = "WINNER_TO_LOSERS"
            parent_protocol_ids: list[str] = []
            ancestor_swarms: list[str] = ["associate-main"]
            if hasattr(self.repository, "list_protocol_propagation"):
                try:
                    propagation = self.repository.list_protocol_propagation(
                        protocol_id=protocol_id,
                        project_id=task.project_id,
                        limit=1,
                    )
                except Exception:
                    propagation = []
                if propagation:
                    latest = propagation[0]
                    transfer_mode = str(latest.get("transfer_mode") or transfer_mode)
                    raw_parents = latest.get("parent_protocol_ids") or []
                    if isinstance(raw_parents, list):
                        parent_protocol_ids = [str(item) for item in raw_parents]
                    raw_swarms = latest.get("ancestor_swarms") or []
                    if isinstance(raw_swarms, list) and raw_swarms:
                        ancestor_swarms = [str(item) for item in raw_swarms]

            transfer_count = int(row.get("transfer_count") or 0)
            score = effectiveness_by_protocol.get(protocol_id, min(1.0, transfer_count / 10.0))
            candidates.append(
                {
                    "protocol_id": protocol_id,
                    "transfer_mode": transfer_mode,
                    "parent_protocol_ids": parent_protocol_ids,
                    "ancestor_swarms": ancestor_swarms,
                    "cross_use_case": bool(row.get("cross_use_case_count") or 0),
                    "selection_prior": score,
                    "confidence_multiplier": 0.90 + min(0.08, max(0.0, score * 0.1)),
                }
            )

        candidates = sorted(
            candidates,
            key=lambda row: (
                float(row.get("selection_prior") or 0.0),
                bool(row.get("cross_use_case", False)),
            ),
            reverse=True,
        )
        return candidates[:limit]

    def _protocol_decay_blocklist(
        self,
        *,
        task: Task,
    ) -> dict[str, str]:
        cfg = self.config.sotappr
        if not getattr(cfg, "enable_protocol_decay_policy", True):
            return {}
        if not hasattr(self.repository, "get_protocol_effectiveness"):
            return {}
        try:
            rows = self.repository.get_protocol_effectiveness(
                project_id=task.project_id,
                limit=100,
            )
        except Exception:
            return {}
        min_samples = int(getattr(cfg, "protocol_decay_min_samples", 5))
        threshold = float(getattr(cfg, "protocol_decay_effectiveness_threshold", 0.35))
        drift_threshold = float(getattr(cfg, "protocol_decay_drift_risk_threshold", 0.75))
        blocked: dict[str, str] = {}
        for row in rows:
            protocol_id = str(row.get("protocol_id") or "")
            if not protocol_id or protocol_id == "(none)":
                continue
            transfer_count = int(row.get("transfer_count") or 0)
            if transfer_count < min_samples:
                continue
            score = self._effectiveness_score_from_row(row)
            avg_drift = float(row.get("avg_drift_risk") or 1.0)
            if score >= threshold and avg_drift <= drift_threshold:
                continue
            blocked[protocol_id] = (
                "protocol_decay_blocked: "
                f"protocol={protocol_id}, score={score:.3f}, threshold={threshold:.3f}, "
                f"avg_drift={avg_drift:.3f}, drift_threshold={drift_threshold:.3f}, "
                f"samples={transfer_count}"
            )
        return blocked

    def _persist_transfer_arbitration_decision(
        self,
        *,
        task: Task,
        attempt: int,
        trace_id: str,
        payload: dict[str, Any],
    ) -> None:
        self._record_trace_event(
            task=task,
            attempt=attempt,
            trace_id=trace_id,
            event_type="transfer_arbitration_decision",
            payload=payload,
            persist_artifact=False,
        )
        if self._active_run_id is None or not hasattr(self.repository, "save_sotappr_artifact"):
            return
        artifact_payload = {
            "trace_id": trace_id,
            "task_id": str(task.id),
            "task_title": task.title,
            "attempt": int(attempt),
            **payload,
        }
        try:
            self.repository.save_sotappr_artifact(
                run_id=self._active_run_id,
                phase=8,
                artifact_type="transfer_arbitration_decision",
                payload=artifact_payload,
            )
        except Exception as exc:
            logger.warning("Failed to persist transfer arbitration decision for task %s: %s", task.id, exc)

    def _persist_protocol_decay_decision(
        self,
        *,
        task: Task,
        attempt: int,
        trace_id: str,
        blocked_protocols: dict[str, str],
    ) -> None:
        payload = {
            "considered_protocols": len(blocked_protocols),
            "blocked_protocol_count": len(blocked_protocols),
            "blocked_protocols": [
                {"protocol_id": protocol_id, "reason": reason}
                for protocol_id, reason in blocked_protocols.items()
            ],
        }
        self._record_trace_event(
            task=task,
            attempt=attempt,
            trace_id=trace_id,
            event_type="protocol_decay_decision",
            payload=payload,
            persist_artifact=False,
        )
        if self._active_run_id is None or not hasattr(self.repository, "save_sotappr_artifact"):
            return
        artifact_payload = {
            "trace_id": trace_id,
            "task_id": str(task.id),
            "task_title": task.title,
            "attempt": int(attempt),
            **payload,
        }
        try:
            self.repository.save_sotappr_artifact(
                run_id=self._active_run_id,
                phase=8,
                artifact_type="protocol_decay_decision",
                payload=artifact_payload,
            )
        except Exception as exc:
            logger.warning("Failed to persist protocol decay decision for task %s: %s", task.id, exc)

    def _run_transfer_promotion(
        self,
        *,
        task: Task,
        attempt: int,
        trace_id: str,
        retrieval_confidence: float,
    ) -> None:
        raw_candidates = self._build_transfer_candidates(
            task=task,
            attempt=attempt,
            retrieval_confidence=retrieval_confidence,
        )
        validated: list[tuple[dict[str, Any], AgentPacket]] = []
        rejected: list[dict[str, Any]] = []
        for idx, candidate in enumerate(raw_candidates, start=1):
            try:
                packet = validate_agent_packet(candidate)
                validated.append((candidate, packet))
            except PacketValidationError as exc:
                rejected.append(
                    {
                        "index": idx,
                        "reason": f"validation_error:{str(exc)[:300]}",
                    }
                )
        scored: list[dict[str, Any]] = []
        policy_rejected: list[dict[str, Any]] = []
        for idx, (candidate, packet) in enumerate(validated, start=1):
            try:
                ranked = self._packet_runtime.rank_transfer_candidates([packet])
            except PacketPolicyError as exc:
                policy_rejected.append(
                    {
                        "index": idx,
                        "packet_id": str(packet.packet_id),
                        "reason": str(exc),
                    }
                )
                continue
            if not ranked:
                continue
            scored.append(
                {
                    "candidate": candidate,
                    "packet": packet,
                    "score": ranked[0],
                }
            )
        scored = sorted(
            scored,
            key=lambda row: float(row["score"].get("selection_score") or 0.0),
            reverse=True,
        )
        decay_blocklist = self._protocol_decay_blocklist(task=task)
        if decay_blocklist:
            self._persist_protocol_decay_decision(
                task=task,
                attempt=attempt,
                trace_id=trace_id,
                blocked_protocols=decay_blocklist,
            )
        top_k = int(getattr(self.config.sotappr, "transfer_arbitration_top_k", 1))
        decay_blocked: list[dict[str, Any]] = []
        decay_eligible: list[dict[str, Any]] = []
        for row in scored:
            protocol_id = str((row["candidate"].get("payload") or {}).get("protocol_id") or "")
            reason = decay_blocklist.get(protocol_id)
            if reason:
                decay_blocked.append(
                    {
                        "packet_id": str(row["packet"].packet_id),
                        "protocol_id": protocol_id,
                        "reason": reason,
                    }
                )
                continue
            decay_eligible.append(row)

        selected = decay_eligible[: max(0, top_k)]
        blocked: list[dict[str, Any]] = []
        blocked.extend(decay_blocked)
        emitted: list[str] = []
        for item in selected:
            packet = item["packet"]
            candidate = item["candidate"]
            protocol_id = str(candidate["payload"].get("protocol_id"))
            allowed, reason = self._passes_protocol_effectiveness_gate(
                task=task,
                protocol_id=protocol_id,
            )
            if not allowed:
                blocked.append(
                    {
                        "packet_id": str(packet.packet_id),
                        "protocol_id": protocol_id,
                        "reason": reason,
                    }
                )
                continue
            result = self._emit_agent_packet(
                task=task,
                attempt=attempt,
                trace_id=trace_id,
                packet_type=PacketType.TRANSFER,
                run_phase=RunPhase.PROMOTION,
                sender_role="orchestrator",
                recipient_roles=["builder"],
                payload=dict(candidate["payload"]),
                lineage=dict(candidate["lineage"]),
                confidence=float(candidate.get("confidence", 0.75)),
                symbolic_keys=list(candidate.get("symbolic_keys", [])),
                capability_tags=list(candidate.get("capability_tags", [])),
                metadata={"transfer_score": item["score"].get("selection_score")},
                generation_override=int(candidate["trace"].get("generation", 25)),
            )
            if result is not None:
                emitted.append(str(result.get("packet_id")))

        decision_payload = {
            "candidate_count": len(raw_candidates),
            "validated_count": len(validated),
            "rejected_count": len(rejected) + len(policy_rejected),
            "selected_count": len(emitted),
            "blocked_count": len(blocked),
            "decay_blocked_count": len(decay_blocked),
            "selected_packet_ids": emitted,
            "blocked": blocked,
            "rejected": [*rejected, *policy_rejected],
            "ranking": [row["score"] for row in scored],
        }
        self._persist_transfer_arbitration_decision(
            task=task,
            attempt=attempt,
            trace_id=trace_id,
            payload=decision_payload,
        )

    def _persist_arbitration_event(
        self,
        *,
        task: Task,
        candidate_count: int,
        retrieval_confidence: float,
        decision_payload: dict[str, Any],
        selected_build_result: BuildResult,
    ) -> None:
        trace_id = self._build_attempt_trace_id(task, int(task.attempt_count))
        self._record_trace_event(
            task=task,
            attempt=int(task.attempt_count),
            trace_id=trace_id,
            event_type="arbitration_decision",
            payload={
                "candidate_count": int(candidate_count),
                "retrieval_confidence": float(retrieval_confidence),
                "decision": decision_payload,
            },
            persist_artifact=False,
        )
        run_id = self._active_run_id
        if run_id is None or not hasattr(self.repository, "save_sotappr_artifact"):
            return
        payload = {
            "trace_id": trace_id,
            "task_id": str(task.id),
            "task_title": task.title,
            "attempt_count": int(task.attempt_count),
            "candidate_count": int(candidate_count),
            "retrieval_confidence": float(retrieval_confidence),
            "decision": decision_payload,
            "selected_tests_passed": bool(selected_build_result.tests_passed),
            "selected_files_changed": list(selected_build_result.files_changed or []),
        }
        try:
            self.repository.save_sotappr_artifact(
                run_id=run_id,
                phase=8,
                artifact_type="arbitration_decision",
                payload=payload,
            )
        except Exception as exc:
            logger.warning("Failed to persist arbitration artifact for task %s: %s", task.id, exc)

    def _check_sotappr_cost_budget(self, task: Task) -> tuple[bool, float]:
        tokens_used = self._total_tokens_per_task.get(task.id, 0)
        estimated_cost = (
            (tokens_used / 1000.0) * self.config.sotappr.estimated_cost_per_1k_tokens_usd
        )
        return estimated_cost <= self.config.sotappr.max_estimated_cost_per_task_usd, estimated_cost

    @staticmethod
    def _looks_like_llm_provider_failure(error_text: str | None) -> bool:
        if not error_text:
            return False
        lowered = error_text.lower()
        indicators = (
            "request failed after",
            "rate limit",
            "invalid api key",
            "model not found",
            "network error",
            "timeout",
            "connecterror",
            "openrouter",
        )
        return any(token in lowered for token in indicators)

    def _sotappr_governance_violation(self, task: Task, build_data: dict) -> str | None:
        del task
        build_result = BuildResult(**build_data) if build_data else BuildResult()
        files_changed = build_result.files_changed or []

        if len(files_changed) > self.config.sotappr.max_files_changed_per_task:
            return (
                f"changed {len(files_changed)} files "
                f"(limit={self.config.sotappr.max_files_changed_per_task})"
            )

        for changed_path in files_changed:
            if self._is_protected_path(changed_path):
                return f"protected path touched: {changed_path}"
        return None

    def _is_protected_path(self, changed_path: str) -> bool:
        normalized = changed_path.replace("\\", "/")
        normalized = os.path.normpath(normalized).replace("\\", "/")
        normalized = normalized.lstrip("./")
        protected = self.config.sotappr.protected_paths
        for raw_rule in protected:
            rule = raw_rule.replace("\\", "/").strip().lstrip("./")
            if not rule:
                continue
            if rule.endswith("/"):
                prefix = rule.rstrip("/")
                if normalized == prefix or normalized.startswith(prefix + "/"):
                    return True
                continue
            if normalized == rule or normalized.startswith(rule + "/"):
                return True
        return False

    def get_last_run_summary(self) -> dict[str, Any]:
        return dict(self._last_run_summary)

    def run_loop(
        self,
        project_id: uuid.UUID,
        max_iterations: int = 100,
        budget_contract: dict[str, Any] | None = None,
    ) -> int:
        """Run the task loop until no more pending tasks or max iterations.

        Args:
            project_id: The project to process.
            max_iterations: Safety limit on iterations.

        Returns:
            Number of tasks processed.
        """
        processed = 0
        run_tokens = 0
        started = time.monotonic()
        stop_reason = "max_iterations_reached"
        previous_trace_id = self._active_trace_id
        run_trace_id = self._ensure_run_trace_id()
        contract = budget_contract or {}
        max_hours = _safe_budget_number(contract.get("max_hours"))
        max_cost_usd = _safe_budget_number(contract.get("max_cost_usd"))
        cost_per_1k = _safe_budget_number(
            contract.get("estimated_cost_per_1k_tokens_usd"),
            default=self.config.sotappr.estimated_cost_per_1k_tokens_usd,
        )

        for i in range(max_iterations):
            elapsed_hours = (time.monotonic() - started) / 3600.0
            if max_hours is not None and elapsed_hours > max_hours:
                stop_reason = "budget_time_exceeded"
                self._notify(
                    f"[WARN] Runtime budget exceeded (time): {elapsed_hours:.2f}h > {max_hours:.2f}h"
                )
                logger.warning(
                    "Stopping run for project %s due to runtime budget hours %.2f > %.2f",
                    project_id,
                    elapsed_hours,
                    max_hours,
                )
                break

            task = self.run_once(project_id)
            if task is None:
                logger.info("No more tasks — exiting loop after %d iterations", i)
                self._notify("[DONE] No more pending tasks.")
                stop_reason = "no_pending_tasks"
                break
            processed += 1
            tokens = self._total_tokens_per_task.get(task.id, 0)
            run_tokens += tokens
            estimated_cost = (run_tokens / 1000.0) * cost_per_1k
            self._notify(
                f"  Result: {task.status.value} "
                f"(tokens: {tokens:,})"
            )
            logger.info("Iteration %d complete: task '%s' → %s", i + 1, task.title, task.status.value)

            if max_cost_usd is not None and estimated_cost > max_cost_usd:
                stop_reason = "budget_cost_exceeded"
                self._notify(
                    "[WARN] Runtime budget exceeded (cost): "
                    f"${estimated_cost:.4f} > ${max_cost_usd:.4f}"
                )
                logger.warning(
                    "Stopping run for project %s due to runtime budget cost %.4f > %.4f",
                    project_id,
                    estimated_cost,
                    max_cost_usd,
                )
                break

        elapsed_hours = (time.monotonic() - started) / 3600.0
        estimated_cost = (run_tokens / 1000.0) * cost_per_1k
        self._last_run_summary = {
            "processed": processed,
            "tokens": run_tokens,
            "estimated_cost_usd": round(estimated_cost, 6),
            "elapsed_hours": round(elapsed_hours, 6),
            "stop_reason": stop_reason,
            "trace_id": run_trace_id,
            "budget_contract": dict(contract),
        }
        self._active_trace_id = previous_trace_id

        return processed


def _safe_budget_number(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
