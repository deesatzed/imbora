"""Data access layer for The Associate.

All SQL queries live here. Agents never write raw SQL — they call Repository
methods that return Pydantic models. This keeps the SQL in one place and
makes the dual-backend (pgvector vs Python cosine) transparent.
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Any, Optional

from src.core.models import (
    ContextSnapshot,
    HypothesisEntry,
    HypothesisOutcome,
    Methodology,
    PeerReview,
    Project,
    Task,
    TaskStatus,
    TokenCostRecord,
)
from src.db.engine import DatabaseEngine


class Repository:
    """Data access layer wrapping DatabaseEngine with typed methods."""

    def __init__(self, engine: DatabaseEngine):
        self.engine = engine

    # -------------------------------------------------------------------
    # Projects
    # -------------------------------------------------------------------

    def create_project(self, project: Project) -> Project:
        self.engine.execute(
            """INSERT INTO projects (id, name, repo_path, tech_stack, project_rules, banned_dependencies)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            [
                str(project.id),
                project.name,
                project.repo_path,
                json.dumps(project.tech_stack),
                project.project_rules,
                project.banned_dependencies,
            ],
        )
        return project

    def get_project(self, project_id: uuid.UUID) -> Optional[Project]:
        row = self.engine.fetch_one(
            "SELECT * FROM projects WHERE id = %s", [str(project_id)]
        )
        if row is None:
            return None
        return _row_to_project(row)

    # -------------------------------------------------------------------
    # Tasks
    # -------------------------------------------------------------------

    def create_task(self, task: Task) -> Task:
        self.engine.execute(
            """INSERT INTO tasks (id, project_id, title, description, status, priority, task_type)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            [
                str(task.id),
                str(task.project_id),
                task.title,
                task.description,
                task.status.value,
                task.priority,
                task.task_type,
            ],
        )
        return task

    def get_next_task(self, project_id: uuid.UUID) -> Optional[Task]:
        """Get the highest-priority PENDING task for a project."""
        row = self.engine.fetch_one(
            """SELECT * FROM tasks
               WHERE project_id = %s AND status = 'PENDING'
               ORDER BY priority DESC, created_at ASC
               LIMIT 1""",
            [str(project_id)],
        )
        if row is None:
            return None
        return _row_to_task(row)

    def get_task(self, task_id: uuid.UUID) -> Optional[Task]:
        row = self.engine.fetch_one("SELECT * FROM tasks WHERE id = %s", [str(task_id)])
        if row is None:
            return None
        return _row_to_task(row)

    def update_task_status(self, task_id: uuid.UUID, status: TaskStatus) -> None:
        completed_at = datetime.now(UTC) if status == TaskStatus.DONE else None
        self.engine.execute(
            """UPDATE tasks SET status = %s, updated_at = now(), completed_at = %s
               WHERE id = %s""",
            [status.value, completed_at, str(task_id)],
        )

    def increment_task_attempt(self, task_id: uuid.UUID) -> None:
        self.engine.execute(
            "UPDATE tasks SET attempt_count = attempt_count + 1, updated_at = now() WHERE id = %s",
            [str(task_id)],
        )

    def increment_task_council_count(self, task_id: uuid.UUID) -> None:
        self.engine.execute(
            "UPDATE tasks SET council_count = council_count + 1, updated_at = now() WHERE id = %s",
            [str(task_id)],
        )

    def get_tasks_by_status(self, project_id: uuid.UUID, status: TaskStatus) -> list[Task]:
        rows = self.engine.fetch_all(
            "SELECT * FROM tasks WHERE project_id = %s AND status = %s ORDER BY priority DESC",
            [str(project_id), status.value],
        )
        return [_row_to_task(r) for r in rows]

    def get_in_progress_tasks(self) -> list[Task]:
        """Find tasks that were interrupted (for crash recovery)."""
        rows = self.engine.fetch_all(
            "SELECT * FROM tasks WHERE status IN ('RESEARCHING', 'CODING', 'REVIEWING')"
        )
        return [_row_to_task(r) for r in rows]

    def list_tasks(self, project_id: uuid.UUID, include_done: bool = True) -> list[Task]:
        """List all tasks for a project, newest first."""
        if include_done:
            rows = self.engine.fetch_all(
                "SELECT * FROM tasks WHERE project_id = %s ORDER BY created_at DESC",
                [str(project_id)],
            )
        else:
            rows = self.engine.fetch_all(
                """SELECT * FROM tasks
                   WHERE project_id = %s AND status != 'DONE'
                   ORDER BY created_at DESC""",
                [str(project_id)],
            )
        return [_row_to_task(r) for r in rows]

    def approve_task(self, task_id: uuid.UUID) -> Optional[Task]:
        """Approve a REVIEWING task and move it to DONE."""
        task = self.get_task(task_id)
        if task is None:
            return None
        if task.status != TaskStatus.REVIEWING:
            return None
        self.update_task_status(task_id, TaskStatus.DONE)
        return self.get_task(task_id)

    def get_next_hypothesis_attempt(self, task_id: uuid.UUID) -> int:
        """Return the next safe hypothesis attempt number for a task."""
        row = self.engine.fetch_one(
            """
            SELECT COALESCE(MAX(attempt_number), 0) + 1 AS next_attempt
            FROM hypothesis_log
            WHERE task_id = %s
            """,
            [str(task_id)],
        )
        return int(row["next_attempt"]) if row else 1

    def reset_task_for_retry(
        self,
        task_id: uuid.UUID,
        status: TaskStatus = TaskStatus.PENDING,
    ) -> Optional[Task]:
        """Reset task status for retry while preserving hypothesis attempt uniqueness."""
        task = self.get_task(task_id)
        if task is None:
            return None

        next_attempt = self.get_next_hypothesis_attempt(task_id)
        reconciled_attempt_count = max(task.attempt_count, next_attempt - 1)

        self.engine.execute(
            """
            UPDATE tasks
            SET status = %s,
                attempt_count = %s,
                council_count = 0,
                updated_at = now(),
                completed_at = NULL
            WHERE id = %s
            """,
            [status.value, reconciled_attempt_count, str(task_id)],
        )
        return self.get_task(task_id)

    def get_task_status_summary(self, project_id: Optional[uuid.UUID] = None) -> dict[str, int]:
        """Return counts of tasks grouped by status."""
        if project_id is None:
            rows = self.engine.fetch_all(
                """
                SELECT status, COUNT(*) AS cnt
                FROM tasks
                GROUP BY status
                """
            )
        else:
            rows = self.engine.fetch_all(
                """
                SELECT status, COUNT(*) AS cnt
                FROM tasks
                WHERE project_id = %s
                GROUP BY status
                """,
                [str(project_id)],
            )
        return {str(row["status"]): int(row["cnt"]) for row in rows}

    def list_review_queue(
        self,
        limit: int = 50,
        project_id: Optional[uuid.UUID] = None,
    ) -> list[Task]:
        """List tasks waiting for human review."""
        if project_id is None:
            rows = self.engine.fetch_all(
                """
                SELECT * FROM tasks
                WHERE status = 'REVIEWING'
                ORDER BY priority DESC, created_at ASC
                LIMIT %s
                """,
                [limit],
            )
        else:
            rows = self.engine.fetch_all(
                """
                SELECT * FROM tasks
                WHERE project_id = %s AND status = 'REVIEWING'
                ORDER BY priority DESC, created_at ASC
                LIMIT %s
                """,
                [str(project_id), limit],
            )
        return [_row_to_task(r) for r in rows]

    # -------------------------------------------------------------------
    # Hypothesis Log
    # -------------------------------------------------------------------

    def log_hypothesis(self, entry: HypothesisEntry) -> HypothesisEntry:
        self.engine.execute(
            """INSERT INTO hypothesis_log
               (id, task_id, attempt_number, approach_summary, outcome,
                error_signature, error_full, files_changed, duration_seconds, model_used)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            [
                str(entry.id),
                str(entry.task_id),
                entry.attempt_number,
                entry.approach_summary,
                entry.outcome.value,
                entry.error_signature,
                entry.error_full,
                entry.files_changed,
                entry.duration_seconds,
                entry.model_used,
            ],
        )
        return entry

    def get_failed_approaches(self, task_id: uuid.UUID) -> list[HypothesisEntry]:
        """Get all failed hypotheses for a task (for forbidden approach injection)."""
        rows = self.engine.fetch_all(
            """SELECT * FROM hypothesis_log
               WHERE task_id = %s AND outcome = 'FAILURE'
               ORDER BY attempt_number ASC""",
            [str(task_id)],
        )
        return [_row_to_hypothesis(r) for r in rows]

    def get_hypothesis_count(self, task_id: uuid.UUID) -> int:
        row = self.engine.fetch_one(
            "SELECT COUNT(*) as cnt FROM hypothesis_log WHERE task_id = %s",
            [str(task_id)],
        )
        return row["cnt"] if row else 0

    def has_duplicate_error(self, task_id: uuid.UUID, error_signature: str) -> bool:
        """Check if this exact error has been seen before for this task."""
        row = self.engine.fetch_one(
            """SELECT COUNT(*) as cnt FROM hypothesis_log
               WHERE task_id = %s AND error_signature = %s AND outcome = 'FAILURE'""",
            [str(task_id), error_signature],
        )
        return (row["cnt"] if row else 0) > 0

    # -------------------------------------------------------------------
    # Methodologies
    # -------------------------------------------------------------------

    def save_methodology(self, methodology: Methodology) -> Methodology:
        # Use pgvector's vector type for embedding storage
        if methodology.problem_embedding:
            vec_str = "[" + ",".join(str(v) for v in methodology.problem_embedding) + "]"
            self.engine.execute(
                """INSERT INTO methodologies
                   (id, problem_description, problem_embedding, solution_code,
                    methodology_notes, source_task_id, tags, language,
                    scope, methodology_type, files_affected)
                   VALUES (%s, %s, %s::vector, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    str(methodology.id),
                    methodology.problem_description,
                    vec_str,
                    methodology.solution_code,
                    methodology.methodology_notes,
                    str(methodology.source_task_id) if methodology.source_task_id else None,
                    methodology.tags,
                    methodology.language,
                    methodology.scope,
                    methodology.methodology_type,
                    methodology.files_affected,
                ],
            )
        else:
            self.engine.execute(
                """INSERT INTO methodologies
                   (id, problem_description, solution_code, methodology_notes,
                    source_task_id, tags, language,
                    scope, methodology_type, files_affected)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                [
                    str(methodology.id),
                    methodology.problem_description,
                    methodology.solution_code,
                    methodology.methodology_notes,
                    str(methodology.source_task_id) if methodology.source_task_id else None,
                    methodology.tags,
                    methodology.language,
                    methodology.scope,
                    methodology.methodology_type,
                    methodology.files_affected,
                ],
            )
        return methodology

    def find_similar_methodologies(
        self, embedding: list[float], limit: int = 3
    ) -> list[tuple[Methodology, float]]:
        """Find methodologies by vector similarity. Returns (methodology, distance) pairs."""
        vec_str = "[" + ",".join(str(v) for v in embedding) + "]"
        rows = self.engine.fetch_all(
            """SELECT *, problem_embedding <=> %s::vector AS distance
               FROM methodologies
               WHERE problem_embedding IS NOT NULL
               ORDER BY distance ASC
               LIMIT %s""",
            [vec_str, limit],
        )
        results = []
        for row in rows:
            distance = row.pop("distance", 1.0)
            similarity = 1.0 - distance  # cosine distance -> similarity
            results.append((_row_to_methodology(row), similarity))
        return results

    def search_methodologies_text(self, query: str, limit: int = 5) -> list[Methodology]:
        """Full-text search on methodologies."""
        rows = self.engine.fetch_all(
            """SELECT *, ts_rank(search_vector, plainto_tsquery('english', %s)) AS rank
               FROM methodologies
               WHERE search_vector @@ plainto_tsquery('english', %s)
               ORDER BY rank DESC
               LIMIT %s""",
            [query, query, limit],
        )
        return [_row_to_methodology(r) for r in rows]

    # -------------------------------------------------------------------
    # Peer Reviews
    # -------------------------------------------------------------------

    def save_peer_review(self, review: PeerReview) -> PeerReview:
        self.engine.execute(
            """INSERT INTO peer_reviews
               (id, task_id, model_used, diagnosis, recommended_approach, reasoning)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            [
                str(review.id),
                str(review.task_id),
                review.model_used,
                review.diagnosis,
                review.recommended_approach,
                review.reasoning,
            ],
        )
        return review

    def get_peer_reviews(self, task_id: uuid.UUID) -> list[PeerReview]:
        rows = self.engine.fetch_all(
            "SELECT * FROM peer_reviews WHERE task_id = %s ORDER BY created_at DESC",
            [str(task_id)],
        )
        return [_row_to_peer_review(r) for r in rows]

    # -------------------------------------------------------------------
    # Context Snapshots
    # -------------------------------------------------------------------

    def save_context_snapshot(self, snapshot: ContextSnapshot) -> ContextSnapshot:
        self.engine.execute(
            """INSERT INTO context_snapshots
               (id, task_id, attempt_number, git_ref, file_manifest)
               VALUES (%s, %s, %s, %s, %s)""",
            [
                str(snapshot.id),
                str(snapshot.task_id),
                snapshot.attempt_number,
                snapshot.git_ref,
                json.dumps(snapshot.file_manifest) if snapshot.file_manifest else None,
            ],
        )
        return snapshot

    def get_latest_snapshot(self, task_id: uuid.UUID) -> Optional[ContextSnapshot]:
        row = self.engine.fetch_one(
            """SELECT * FROM context_snapshots
               WHERE task_id = %s
               ORDER BY attempt_number DESC
               LIMIT 1""",
            [str(task_id)],
        )
        if row is None:
            return None
        return _row_to_context_snapshot(row)

    # -------------------------------------------------------------------
    # SOTAppR Runs / Artifacts
    # -------------------------------------------------------------------

    def create_sotappr_run(
        self,
        project_id: uuid.UUID,
        mode: str,
        governance_pack: str,
        spec_json: dict[str, Any],
        report_json: dict[str, Any],
        repo_path: str,
        report_path: Optional[str] = None,
        status: str = "planned",
        stop_reason: Optional[str] = None,
        estimated_cost_usd: Optional[float] = None,
        elapsed_hours: Optional[float] = None,
    ) -> uuid.UUID:
        run_id = uuid.uuid4()
        self.engine.execute(
            """INSERT INTO sotappr_runs
               (id, project_id, mode, status, governance_pack, spec_json, report_json, repo_path,
                report_path, stop_reason, estimated_cost_usd, elapsed_hours)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            [
                str(run_id),
                str(project_id),
                mode,
                status,
                governance_pack,
                json.dumps(spec_json),
                json.dumps(report_json),
                repo_path,
                report_path,
                stop_reason,
                estimated_cost_usd,
                elapsed_hours,
            ],
        )
        return run_id

    def update_sotappr_run(
        self,
        run_id: uuid.UUID,
        *,
        status: Optional[str] = None,
        tasks_seeded: Optional[int] = None,
        tasks_processed: Optional[int] = None,
        last_error: Optional[str] = None,
        stop_reason: Optional[str] = None,
        estimated_cost_usd: Optional[float] = None,
        elapsed_hours: Optional[float] = None,
        completed: bool = False,
    ) -> None:
        sets: list[str] = ["updated_at = now()"]
        params: list[Any] = []

        if status is not None:
            sets.append("status = %s")
            params.append(status)
        if tasks_seeded is not None:
            sets.append("tasks_seeded = %s")
            params.append(tasks_seeded)
        if tasks_processed is not None:
            sets.append("tasks_processed = %s")
            params.append(tasks_processed)
        if last_error is not None:
            sets.append("last_error = %s")
            params.append(last_error)
        if stop_reason is not None:
            sets.append("stop_reason = %s")
            params.append(stop_reason)
        if estimated_cost_usd is not None:
            sets.append("estimated_cost_usd = %s")
            params.append(estimated_cost_usd)
        if elapsed_hours is not None:
            sets.append("elapsed_hours = %s")
            params.append(elapsed_hours)
        if completed:
            sets.append("completed_at = now()")

        params.append(str(run_id))
        self.engine.execute(
            f"UPDATE sotappr_runs SET {', '.join(sets)} WHERE id = %s",
            params,
        )

    def get_sotappr_run(self, run_id: uuid.UUID) -> Optional[dict[str, Any]]:
        row = self.engine.fetch_one(
            "SELECT * FROM sotappr_runs WHERE id = %s",
            [str(run_id)],
        )
        return row

    def list_sotappr_runs(
        self,
        limit: int = 20,
        project_id: Optional[uuid.UUID] = None,
    ) -> list[dict[str, Any]]:
        if project_id is None:
            rows = self.engine.fetch_all(
                """SELECT * FROM sotappr_runs
                   ORDER BY created_at DESC
                   LIMIT %s""",
                [limit],
            )
        else:
            rows = self.engine.fetch_all(
                """SELECT * FROM sotappr_runs
                   WHERE project_id = %s
                   ORDER BY created_at DESC
                   LIMIT %s""",
                [str(project_id), limit],
            )
        return rows

    def save_sotappr_artifact(
        self,
        run_id: uuid.UUID,
        phase: int,
        artifact_type: str,
        payload: dict[str, Any],
    ) -> uuid.UUID:
        artifact_id = uuid.uuid4()
        self.engine.execute(
            """INSERT INTO sotappr_artifacts
               (id, run_id, phase, artifact_type, payload)
               VALUES (%s, %s, %s, %s, %s)""",
            [
                str(artifact_id),
                str(run_id),
                phase,
                artifact_type,
                json.dumps(payload),
            ],
        )
        return artifact_id

    def list_sotappr_artifacts(
        self,
        run_id: uuid.UUID,
        phase: Optional[int] = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        if phase is None:
            rows = self.engine.fetch_all(
                """SELECT * FROM sotappr_artifacts
                   WHERE run_id = %s
                   ORDER BY created_at DESC
                   LIMIT %s""",
                [str(run_id), limit],
            )
        else:
            rows = self.engine.fetch_all(
                """SELECT * FROM sotappr_artifacts
                   WHERE run_id = %s AND phase = %s
                   ORDER BY created_at DESC
                   LIMIT %s""",
                [str(run_id), phase, limit],
            )
        return rows

    def save_agent_packet(
        self,
        *,
        run_id: uuid.UUID,
        task_id: Optional[uuid.UUID],
        attempt_number: int,
        trace_id: Optional[str],
        packet: dict[str, Any],
        packet_hash: Optional[str] = None,
        lifecycle_state: Optional[str] = None,
        lifecycle_history: Optional[list[dict[str, Any]]] = None,
    ) -> uuid.UUID:
        """Persist APC/1.0 packet + transition events + lineage as first-class rows."""
        packet_row_id = uuid.uuid4()
        packet_id = str(packet.get("packet_id"))
        trace = packet.get("trace") if isinstance(packet.get("trace"), dict) else {}
        sender = packet.get("sender") if isinstance(packet.get("sender"), dict) else {}
        recipients = packet.get("recipients")
        recipients_json = recipients if isinstance(recipients, list) else []
        payload_json = packet.get("payload") if isinstance(packet.get("payload"), dict) else {}

        self.engine.execute(
            """INSERT INTO agent_packets
               (id, run_id, task_id, attempt_number, packet_id, root_packet_id, parent_packet_id,
                packet_type, channel, run_phase, sender_agent_id, sender_role, recipients_json,
                packet_json, payload_json, packet_hash, confidence, trace_id, lifecycle_state, lifecycle_history)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            [
                str(packet_row_id),
                str(run_id),
                str(task_id) if task_id else None,
                int(attempt_number),
                packet_id,
                str(trace.get("root_packet_id")) if trace.get("root_packet_id") else None,
                str(trace.get("parent_packet_id")) if trace.get("parent_packet_id") else None,
                packet.get("packet_type"),
                packet.get("channel"),
                packet.get("run_phase"),
                sender.get("agent_id"),
                sender.get("role"),
                json.dumps(recipients_json),
                json.dumps(packet),
                json.dumps(payload_json),
                packet_hash,
                float(packet.get("confidence", 0.0) or 0.0),
                trace_id or trace.get("run_id"),
                lifecycle_state,
                json.dumps(lifecycle_history or []),
            ],
        )

        history = lifecycle_history or []
        for item in history:
            if not isinstance(item, dict):
                continue
            self.engine.execute(
                """INSERT INTO packet_events
                   (id, run_id, task_id, packet_row_id, packet_id, event, from_state, to_state, metadata, occurred_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, COALESCE(%s::timestamptz, now()))""",
                [
                    str(uuid.uuid4()),
                    str(run_id),
                    str(task_id) if task_id else None,
                    str(packet_row_id),
                    packet_id,
                    str(item.get("event") or "unknown"),
                    item.get("from"),
                    item.get("to"),
                    json.dumps(item.get("metadata") if isinstance(item.get("metadata"), dict) else {}),
                    item.get("at"),
                ],
            )

        lineage = packet.get("lineage") if isinstance(packet.get("lineage"), dict) else None
        if lineage is not None:
            self.engine.execute(
                """INSERT INTO packet_lineage
                   (id, packet_row_id, packet_id, protocol_id, parent_protocol_ids, ancestor_swarms,
                    cross_use_case, transfer_mode, lineage_json)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (packet_id) DO NOTHING""",
                [
                    str(uuid.uuid4()),
                    str(packet_row_id),
                    packet_id,
                    lineage.get("protocol_id"),
                    lineage.get("parent_protocol_ids") or [],
                    lineage.get("ancestor_swarms") or [],
                    bool(lineage.get("cross_use_case", False)),
                    lineage.get("transfer_mode"),
                    json.dumps(lineage),
                ],
            )

        return packet_row_id

    def list_agent_packets(
        self,
        *,
        run_id: Optional[uuid.UUID] = None,
        task_id: Optional[uuid.UUID] = None,
        packet_id: Optional[uuid.UUID] = None,
        packet_type: Optional[str] = None,
        run_phase: Optional[str] = None,
        trace_id: Optional[str] = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = %s")
            params.append(str(run_id))
        if task_id is not None:
            clauses.append("task_id = %s")
            params.append(str(task_id))
        if packet_id is not None:
            clauses.append("packet_id = %s")
            params.append(str(packet_id))
        if packet_type:
            clauses.append("packet_type = %s")
            params.append(packet_type)
        if run_phase:
            clauses.append("run_phase = %s")
            params.append(run_phase)
        if trace_id:
            clauses.append("trace_id = %s")
            params.append(trace_id)

        where = " AND ".join(clauses) if clauses else "TRUE"
        params.append(limit)
        return self.engine.fetch_all(
            f"""SELECT * FROM agent_packets
                WHERE {where}
                ORDER BY created_at DESC
                LIMIT %s""",
            params,
        )

    def list_packet_events(
        self,
        *,
        run_id: Optional[uuid.UUID] = None,
        task_id: Optional[uuid.UUID] = None,
        packet_id: Optional[uuid.UUID] = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if run_id is not None:
            clauses.append("run_id = %s")
            params.append(str(run_id))
        if task_id is not None:
            clauses.append("task_id = %s")
            params.append(str(task_id))
        if packet_id is not None:
            clauses.append("packet_id = %s")
            params.append(str(packet_id))

        where = " AND ".join(clauses) if clauses else "TRUE"
        params.append(limit)
        return self.engine.fetch_all(
            f"""SELECT * FROM packet_events
                WHERE {where}
                ORDER BY occurred_at DESC
                LIMIT %s""",
            params,
        )

    def get_packet_timeline(
        self,
        *,
        run_id: uuid.UUID,
        task_id: Optional[uuid.UUID] = None,
        limit: int = 500,
    ) -> list[dict[str, Any]]:
        """Fetch packet + transition event timeline for run/task replay."""
        params: list[Any] = [str(run_id)]
        task_filter = ""
        if task_id is not None:
            task_filter = "AND p.task_id = %s"
            params.append(str(task_id))
        params.append(limit)
        return self.engine.fetch_all(
            f"""
            SELECT
              p.packet_id,
              p.packet_type,
              p.run_phase,
              p.lifecycle_state,
              p.trace_id,
              p.created_at AS packet_created_at,
              e.event,
              e.from_state,
              e.to_state,
              e.metadata,
              e.occurred_at
            FROM agent_packets p
            LEFT JOIN packet_events e ON e.packet_row_id = p.id
            WHERE p.run_id = %s {task_filter}
            ORDER BY p.created_at ASC, e.occurred_at ASC
            LIMIT %s
            """,
            params,
        )

    def get_packet_lineage_summary(
        self,
        *,
        project_id: Optional[uuid.UUID] = None,
        run_id: Optional[uuid.UUID] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id is not None:
            clauses.append("r.project_id = %s")
            params.append(str(project_id))
        if run_id is not None:
            clauses.append("p.run_id = %s")
            params.append(str(run_id))

        where = " AND ".join(clauses) if clauses else "TRUE"
        params.append(limit)
        return self.engine.fetch_all(
            f"""
            SELECT
              COALESCE(l.protocol_id, '(none)') AS protocol_id,
              COUNT(*) AS transfer_count,
              COUNT(DISTINCT p.run_id) AS run_count,
              COUNT(DISTINCT p.task_id) AS task_count,
              SUM(CASE WHEN l.cross_use_case THEN 1 ELSE 0 END) AS cross_use_case_count,
              MAX(p.created_at) AS last_seen_at
            FROM packet_lineage l
            JOIN agent_packets p ON p.id = l.packet_row_id
            JOIN sotappr_runs r ON r.id = p.run_id
            WHERE {where}
            GROUP BY COALESCE(l.protocol_id, '(none)')
            ORDER BY transfer_count DESC, last_seen_at DESC
            LIMIT %s
            """,
            params,
        )

    def list_protocol_propagation(
        self,
        *,
        protocol_id: Optional[str] = None,
        project_id: Optional[uuid.UUID] = None,
        run_id: Optional[uuid.UUID] = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if protocol_id:
            clauses.append("l.protocol_id = %s")
            params.append(protocol_id)
        if project_id is not None:
            clauses.append("r.project_id = %s")
            params.append(str(project_id))
        if run_id is not None:
            clauses.append("p.run_id = %s")
            params.append(str(run_id))

        where = " AND ".join(clauses) if clauses else "TRUE"
        params.append(limit)
        return self.engine.fetch_all(
            f"""
            SELECT
              l.protocol_id,
              l.parent_protocol_ids,
              l.ancestor_swarms,
              l.cross_use_case,
              l.transfer_mode,
              p.packet_id,
              p.packet_type,
              p.run_phase,
              p.run_id,
              r.project_id,
              p.task_id,
              p.trace_id,
              p.created_at
            FROM packet_lineage l
            JOIN agent_packets p ON p.id = l.packet_row_id
            JOIN sotappr_runs r ON r.id = p.run_id
            WHERE {where}
            ORDER BY p.created_at DESC
            LIMIT %s
            """,
            params,
        )

    def get_transfer_quality_rollup(
        self,
        *,
        project_id: Optional[uuid.UUID] = None,
        run_id: Optional[uuid.UUID] = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = ["p.packet_type = 'TRANSFER'"]
        params: list[Any] = []
        if project_id is not None:
            clauses.append("r.project_id = %s")
            params.append(str(project_id))
        if run_id is not None:
            clauses.append("p.run_id = %s")
            params.append(str(run_id))

        where = " AND ".join(clauses)
        return self.engine.fetch_all(
            f"""
            SELECT
              COALESCE(
                l.transfer_mode,
                p.payload_json->'transfer_policy'->>'mode',
                p.payload_json->'transfer_policy'->>'transfer_mode',
                'unknown'
              ) AS transfer_mode,
              COUNT(*) AS total_count,
              SUM(
                CASE WHEN COALESCE((p.payload_json->>'accepted')::boolean, FALSE)
                     THEN 1 ELSE 0 END
              ) AS accepted_count,
              SUM(CASE WHEN COALESCE(l.cross_use_case, FALSE) THEN 1 ELSE 0 END) AS cross_use_case_count,
              AVG(NULLIF(p.payload_json->>'sender_score', '')::double precision) AS avg_sender_score,
              AVG(NULLIF(p.payload_json->>'receiver_score', '')::double precision) AS avg_receiver_score
            FROM agent_packets p
            LEFT JOIN packet_lineage l ON l.packet_row_id = p.id
            JOIN sotappr_runs r ON r.id = p.run_id
            WHERE {where}
            GROUP BY 1
            ORDER BY total_count DESC, transfer_mode ASC
            """,
            params,
        )

    def get_transfer_eval_correlation(
        self,
        *,
        project_id: Optional[uuid.UUID] = None,
        run_id: Optional[uuid.UUID] = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id is not None:
            clauses.append("r.project_id = %s")
            params.append(str(project_id))
        if run_id is not None:
            clauses.append("p.run_id = %s")
            params.append(str(run_id))
        condition = " AND ".join(clauses) if clauses else "TRUE"
        return self.engine.fetch_all(
            f"""
            WITH transfer_tasks AS (
              SELECT
                p.run_id,
                p.task_id,
                BOOL_OR(COALESCE(l.cross_use_case, FALSE)) AS has_cross_use_case_transfer
              FROM agent_packets p
              LEFT JOIN packet_lineage l ON l.packet_row_id = p.id
              JOIN sotappr_runs r ON r.id = p.run_id
              WHERE ({condition}) AND p.packet_type = 'TRANSFER'
              GROUP BY p.run_id, p.task_id
            ),
            eval_tasks AS (
              SELECT
                p.run_id,
                p.task_id,
                AVG(NULLIF(p.payload_json->'metric_set'->>'outcome', '')::double precision) AS avg_outcome,
                AVG(NULLIF(p.payload_json->'metric_set'->>'drift_risk', '')::double precision) AS avg_drift_risk
              FROM agent_packets p
              JOIN sotappr_runs r ON r.id = p.run_id
              WHERE ({condition}) AND p.packet_type = 'EVAL'
              GROUP BY p.run_id, p.task_id
            )
            SELECT
              t.has_cross_use_case_transfer,
              COUNT(*) AS task_count,
              AVG(e.avg_outcome) AS avg_outcome,
              AVG(e.avg_drift_risk) AS avg_drift_risk
            FROM transfer_tasks t
            LEFT JOIN eval_tasks e ON e.run_id = t.run_id AND e.task_id = t.task_id
            GROUP BY t.has_cross_use_case_transfer
            ORDER BY t.has_cross_use_case_transfer DESC
            """,
            params + params,
        )

    def get_protocol_effectiveness(
        self,
        *,
        project_id: Optional[uuid.UUID] = None,
        run_id: Optional[uuid.UUID] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        base_params: list[Any] = []
        if project_id is not None:
            clauses.append("r.project_id = %s")
            base_params.append(str(project_id))
        if run_id is not None:
            clauses.append("p.run_id = %s")
            base_params.append(str(run_id))
        condition = " AND ".join(clauses) if clauses else "TRUE"
        params: list[Any] = [*base_params, *base_params, limit]
        return self.engine.fetch_all(
            f"""
            WITH transfer_packets AS (
              SELECT
                COALESCE(l.protocol_id, '(none)') AS protocol_id,
                p.run_id,
                p.task_id,
                COALESCE((p.payload_json->>'accepted')::boolean, FALSE) AS accepted
              FROM agent_packets p
              LEFT JOIN packet_lineage l ON l.packet_row_id = p.id
              JOIN sotappr_runs r ON r.id = p.run_id
              WHERE ({condition}) AND p.packet_type = 'TRANSFER'
            ),
            eval_packets AS (
              SELECT
                p.run_id,
                p.task_id,
                AVG(NULLIF(p.payload_json->'metric_set'->>'outcome', '')::double precision) AS avg_outcome,
                AVG(NULLIF(p.payload_json->'metric_set'->>'drift_risk', '')::double precision) AS avg_drift_risk
              FROM agent_packets p
              JOIN sotappr_runs r ON r.id = p.run_id
              WHERE ({condition}) AND p.packet_type = 'EVAL'
              GROUP BY p.run_id, p.task_id
            )
            SELECT
              t.protocol_id,
              COUNT(*) AS transfer_count,
              SUM(CASE WHEN t.accepted THEN 1 ELSE 0 END) AS accepted_count,
              COUNT(DISTINCT t.run_id) AS run_count,
              COUNT(DISTINCT t.task_id) AS task_count,
              AVG(e.avg_outcome) AS avg_outcome,
              AVG(e.avg_drift_risk) AS avg_drift_risk
            FROM transfer_packets t
            LEFT JOIN eval_packets e ON e.run_id = t.run_id AND e.task_id = t.task_id
            GROUP BY t.protocol_id
            ORDER BY transfer_count DESC, accepted_count DESC
            LIMIT %s
            """,
            params,
        )

    # -------------------------------------------------------------------
    # Token Costs (Item 1)
    # -------------------------------------------------------------------

    def save_token_cost(self, record: TokenCostRecord) -> TokenCostRecord:
        self.engine.execute(
            """INSERT INTO token_costs
               (id, task_id, run_id, agent_role, model_used,
                input_tokens, output_tokens, total_tokens, cost_usd)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)""",
            [
                str(record.id),
                str(record.task_id) if record.task_id else None,
                str(record.run_id) if record.run_id else None,
                record.agent_role,
                record.model_used,
                record.input_tokens,
                record.output_tokens,
                record.total_tokens,
                record.cost_usd,
            ],
        )
        return record

    def get_token_costs_for_task(self, task_id: uuid.UUID) -> list[dict[str, Any]]:
        return self.engine.fetch_all(
            "SELECT * FROM token_costs WHERE task_id = %s ORDER BY created_at ASC",
            [str(task_id)],
        )

    def get_token_costs_for_run(self, run_id: uuid.UUID) -> list[dict[str, Any]]:
        return self.engine.fetch_all(
            "SELECT * FROM token_costs WHERE run_id = %s ORDER BY created_at ASC",
            [str(run_id)],
        )

    def get_token_cost_summary(
        self, run_id: Optional[uuid.UUID] = None
    ) -> dict[str, Any]:
        if run_id:
            row = self.engine.fetch_one(
                """SELECT
                     SUM(input_tokens) AS total_input,
                     SUM(output_tokens) AS total_output,
                     SUM(total_tokens) AS total_tokens,
                     SUM(cost_usd) AS total_cost,
                     COUNT(*) AS call_count
                   FROM token_costs WHERE run_id = %s""",
                [str(run_id)],
            )
        else:
            row = self.engine.fetch_one(
                """SELECT
                     SUM(input_tokens) AS total_input,
                     SUM(output_tokens) AS total_output,
                     SUM(total_tokens) AS total_tokens,
                     SUM(cost_usd) AS total_cost,
                     COUNT(*) AS call_count
                   FROM token_costs"""
            )
        if row is None:
            return {"total_input": 0, "total_output": 0, "total_tokens": 0, "total_cost": 0.0, "call_count": 0}
        return {
            "total_input": int(row["total_input"] or 0),
            "total_output": int(row["total_output"] or 0),
            "total_tokens": int(row["total_tokens"] or 0),
            "total_cost": float(row["total_cost"] or 0.0),
            "call_count": int(row["call_count"] or 0),
        }

    def search_sotappr_replay(self, query: str, limit: int = 20) -> list[dict[str, Any]]:
        rows = self.engine.fetch_all(
            """SELECT a.*, r.project_id
               FROM sotappr_artifacts a
               JOIN sotappr_runs r ON r.id = a.run_id
               WHERE a.artifact_type = 'experience_replay'
                 AND a.payload::text ILIKE %s
               ORDER BY a.created_at DESC
               LIMIT %s""",
            [f"%{query}%", limit],
        )
        return rows

    def get_portfolio_summary(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self.engine.fetch_all(
            """
            SELECT
              p.id,
              p.name,
              p.repo_path,
              COUNT(t.id) AS total_tasks,
              COUNT(*) FILTER (WHERE t.status = 'PENDING') AS pending_tasks,
              COUNT(*) FILTER (WHERE t.status = 'DONE') AS done_tasks,
              COUNT(*) FILTER (WHERE t.status = 'STUCK') AS stuck_tasks,
              MAX(sr.created_at) AS last_run_at
            FROM projects p
            LEFT JOIN tasks t ON t.project_id = p.id
            LEFT JOIN sotappr_runs sr ON sr.project_id = p.id
            GROUP BY p.id
            ORDER BY last_run_at DESC NULLS LAST, p.created_at DESC
            LIMIT %s
            """,
            [limit],
        )
        return rows

    def get_hypothesis_error_stats(
        self, project_id: Optional[uuid.UUID] = None
    ) -> list[dict[str, Any]]:
        if project_id is None:
            rows = self.engine.fetch_all(
                """
                SELECT error_signature, COUNT(*) AS cnt
                FROM hypothesis_log
                WHERE error_signature IS NOT NULL
                GROUP BY error_signature
                ORDER BY cnt DESC
                LIMIT 50
                """
            )
        else:
            rows = self.engine.fetch_all(
                """
                SELECT h.error_signature, COUNT(*) AS cnt
                FROM hypothesis_log h
                JOIN tasks t ON t.id = h.task_id
                WHERE t.project_id = %s AND h.error_signature IS NOT NULL
                GROUP BY h.error_signature
                ORDER BY cnt DESC
                LIMIT 50
                """,
                [str(project_id)],
            )
        return rows


    # -------------------------------------------------------------------
    # Data Science Experiments & Artifacts
    # -------------------------------------------------------------------

    def create_ds_experiment(
        self,
        project_id: uuid.UUID,
        experiment_phase: str,
        experiment_config: dict[str, Any] | None = None,
        task_id: uuid.UUID | None = None,
        run_id: uuid.UUID | None = None,
        dataset_fingerprint: str | None = None,
        parent_experiment_id: uuid.UUID | None = None,
    ) -> uuid.UUID:
        exp_id = uuid.uuid4()
        self.engine.execute(
            """INSERT INTO ds_experiments
               (id, project_id, task_id, run_id, experiment_phase,
                experiment_config, dataset_fingerprint, parent_experiment_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            [
                str(exp_id),
                str(project_id),
                str(task_id) if task_id else None,
                str(run_id) if run_id else None,
                experiment_phase,
                json.dumps(experiment_config or {}),
                dataset_fingerprint,
                str(parent_experiment_id) if parent_experiment_id else None,
            ],
        )
        return exp_id

    def update_ds_experiment(
        self,
        experiment_id: uuid.UUID,
        status: str = "COMPLETED",
        metrics: dict[str, Any] | None = None,
        artifacts_manifest: dict[str, Any] | None = None,
    ) -> None:
        self.engine.execute(
            """UPDATE ds_experiments
               SET status = %s, metrics = %s, artifacts_manifest = %s,
                   completed_at = now()
               WHERE id = %s""",
            [
                status,
                json.dumps(metrics or {}),
                json.dumps(artifacts_manifest or {}),
                str(experiment_id),
            ],
        )

    def get_ds_experiments(
        self,
        project_id: uuid.UUID,
        phase: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if phase:
            return self.engine.fetch_all(
                """SELECT * FROM ds_experiments
                   WHERE project_id = %s AND experiment_phase = %s
                   ORDER BY created_at DESC LIMIT %s""",
                [str(project_id), phase, limit],
            )
        return self.engine.fetch_all(
            """SELECT * FROM ds_experiments
               WHERE project_id = %s
               ORDER BY created_at DESC LIMIT %s""",
            [str(project_id), limit],
        )

    def save_ds_artifact(
        self,
        experiment_id: uuid.UUID,
        artifact_type: str,
        artifact_path: str,
        artifact_hash: str,
        artifact_metadata: dict[str, Any] | None = None,
        size_bytes: int = 0,
    ) -> uuid.UUID:
        art_id = uuid.uuid4()
        self.engine.execute(
            """INSERT INTO ds_artifacts
               (id, experiment_id, artifact_type, artifact_path,
                artifact_hash, artifact_metadata, size_bytes)
               VALUES (%s, %s, %s, %s, %s, %s, %s)""",
            [
                str(art_id),
                str(experiment_id),
                artifact_type,
                artifact_path,
                artifact_hash,
                json.dumps(artifact_metadata or {}),
                size_bytes,
            ],
        )
        return art_id

    def get_ds_artifacts(
        self, experiment_id: uuid.UUID
    ) -> list[dict[str, Any]]:
        return self.engine.fetch_all(
            "SELECT * FROM ds_artifacts WHERE experiment_id = %s ORDER BY created_at",
            [str(experiment_id)],
        )

    def get_tasks_by_type(
        self, project_id: uuid.UUID, task_type: str
    ) -> list[Task]:
        rows = self.engine.fetch_all(
            """SELECT * FROM tasks
               WHERE project_id = %s AND task_type = %s
               ORDER BY priority DESC, created_at ASC""",
            [str(project_id), task_type],
        )
        return [_row_to_task(r) for r in rows]


# ---------------------------------------------------------------------------
# Row-to-model converters
# ---------------------------------------------------------------------------

def _row_to_project(row: dict) -> Project:
    return Project(
        id=uuid.UUID(str(row["id"])),
        name=row["name"],
        repo_path=row["repo_path"],
        tech_stack=row.get("tech_stack") or {},
        project_rules=row.get("project_rules"),
        banned_dependencies=row.get("banned_dependencies") or [],
        created_at=row.get("created_at", datetime.now(UTC)),
        updated_at=row.get("updated_at", datetime.now(UTC)),
    )


def _row_to_task(row: dict) -> Task:
    return Task(
        id=uuid.UUID(str(row["id"])),
        project_id=uuid.UUID(str(row["project_id"])),
        title=row["title"],
        description=row["description"],
        status=TaskStatus(row["status"]),
        priority=row.get("priority", 0),
        context_snapshot_id=(
            uuid.UUID(str(row["context_snapshot_id"])) if row.get("context_snapshot_id") else None
        ),
        attempt_count=row.get("attempt_count", 0),
        council_count=row.get("council_count", 0),
        task_type=row.get("task_type", "general"),
        created_at=row.get("created_at", datetime.now(UTC)),
        updated_at=row.get("updated_at", datetime.now(UTC)),
        completed_at=row.get("completed_at"),
    )


def _row_to_hypothesis(row: dict) -> HypothesisEntry:
    return HypothesisEntry(
        id=uuid.UUID(str(row["id"])),
        task_id=uuid.UUID(str(row["task_id"])),
        attempt_number=row["attempt_number"],
        approach_summary=row["approach_summary"],
        outcome=HypothesisOutcome(row["outcome"]),
        error_signature=row.get("error_signature"),
        error_full=row.get("error_full"),
        files_changed=row.get("files_changed") or [],
        duration_seconds=row.get("duration_seconds"),
        model_used=row.get("model_used"),
        created_at=row.get("created_at", datetime.now(UTC)),
    )


def _row_to_methodology(row: dict) -> Methodology:
    # pgvector returns the embedding as a string or list — normalize
    embedding = row.get("problem_embedding")
    if isinstance(embedding, str):
        embedding = None  # Don't parse back in list context
    return Methodology(
        id=uuid.UUID(str(row["id"])),
        problem_description=row["problem_description"],
        problem_embedding=None,  # Don't roundtrip embeddings through model
        solution_code=row["solution_code"],
        methodology_notes=row.get("methodology_notes"),
        source_task_id=(
            uuid.UUID(str(row["source_task_id"])) if row.get("source_task_id") else None
        ),
        tags=row.get("tags") or [],
        language=row.get("language"),
        scope=row.get("scope", "project"),
        methodology_type=row.get("methodology_type"),
        files_affected=row.get("files_affected") or [],
        created_at=row.get("created_at", datetime.now(UTC)),
    )


def _row_to_peer_review(row: dict) -> PeerReview:
    return PeerReview(
        id=uuid.UUID(str(row["id"])),
        task_id=uuid.UUID(str(row["task_id"])),
        model_used=row["model_used"],
        diagnosis=row["diagnosis"],
        recommended_approach=row.get("recommended_approach"),
        reasoning=row.get("reasoning"),
        created_at=row.get("created_at", datetime.now(UTC)),
    )


def _row_to_context_snapshot(row: dict) -> ContextSnapshot:
    return ContextSnapshot(
        id=uuid.UUID(str(row["id"])),
        task_id=uuid.UUID(str(row["task_id"])),
        attempt_number=row["attempt_number"],
        git_ref=row["git_ref"],
        file_manifest=row.get("file_manifest"),
        created_at=row.get("created_at", datetime.now(UTC)),
    )
