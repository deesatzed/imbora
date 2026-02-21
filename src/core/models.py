"""All Pydantic data models for The Associate.

Defines the data contracts used across all agents, database operations,
and orchestration. Every table row and inter-agent message has a model here.
"""

from __future__ import annotations

import enum
import uuid
from datetime import UTC, datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, model_validator


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()


def _now() -> datetime:
    return datetime.now(UTC)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class TaskStatus(str, enum.Enum):
    PENDING = "PENDING"
    RESEARCHING = "RESEARCHING"
    CODING = "CODING"
    REVIEWING = "REVIEWING"
    STUCK = "STUCK"
    DONE = "DONE"


class HypothesisOutcome(str, enum.Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"


# ---------------------------------------------------------------------------
# Database row models
# ---------------------------------------------------------------------------

class Project(BaseModel):
    id: uuid.UUID = Field(default_factory=_new_uuid)
    name: str
    repo_path: str
    tech_stack: dict[str, Any] = Field(default_factory=dict)
    project_rules: Optional[str] = None
    banned_dependencies: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Task(BaseModel):
    id: uuid.UUID = Field(default_factory=_new_uuid)
    project_id: uuid.UUID
    title: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    priority: int = 0
    context_snapshot_id: Optional[uuid.UUID] = None
    attempt_count: int = 0
    council_count: int = 0
    task_type: str = "general"
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    completed_at: Optional[datetime] = None


class HypothesisEntry(BaseModel):
    id: uuid.UUID = Field(default_factory=_new_uuid)
    task_id: uuid.UUID
    attempt_number: int
    approach_summary: str
    outcome: HypothesisOutcome = HypothesisOutcome.FAILURE
    error_signature: Optional[str] = None
    error_full: Optional[str] = None
    files_changed: list[str] = Field(default_factory=list)
    duration_seconds: Optional[float] = None
    model_used: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)


class Methodology(BaseModel):
    id: uuid.UUID = Field(default_factory=_new_uuid)
    problem_description: str
    problem_embedding: Optional[list[float]] = None  # 384-dim vector
    solution_code: str
    methodology_notes: Optional[str] = None
    source_task_id: Optional[uuid.UUID] = None
    tags: list[str] = Field(default_factory=list)
    language: Optional[str] = None
    scope: str = "project"  # "project" or "global" (Item 2)
    methodology_type: Optional[str] = None  # BUG_FIX/PATTERN/DECISION/GOTCHA (Item 3)
    files_affected: list[str] = Field(default_factory=list)  # Item 4
    created_at: datetime = Field(default_factory=_now)


class PeerReview(BaseModel):
    id: uuid.UUID = Field(default_factory=_new_uuid)
    task_id: uuid.UUID
    model_used: str
    diagnosis: str
    recommended_approach: Optional[str] = None
    reasoning: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)


class ContextSnapshot(BaseModel):
    id: uuid.UUID = Field(default_factory=_new_uuid)
    task_id: uuid.UUID
    attempt_number: int
    git_ref: str
    file_manifest: Optional[dict[str, str]] = None  # {path: hash}
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Inter-agent message models
# ---------------------------------------------------------------------------

class AgentResult(BaseModel):
    """Standardized output from any agent."""
    agent_name: str
    status: str  # "success", "failure", "blocked"
    data: dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    duration_seconds: float = 0.0


class TaskContext(BaseModel):
    """Output of State Manager — enriched task context."""
    task: Task
    forbidden_approaches: list[str] = Field(default_factory=list)
    checkpoint_ref: Optional[str] = None
    previous_council_diagnosis: Optional[str] = None


class ResearchBrief(BaseModel):
    """Output of Research Unit — live documentation context."""
    live_docs: list[dict[str, str]] = Field(default_factory=list)  # [{title, url, snippet}]
    api_signatures: list[str] = Field(default_factory=list)
    version_warnings: list[str] = Field(default_factory=list)


class ContextBrief(BaseModel):
    """Output of Librarian — full context for Builder."""
    task: Task
    past_solutions: list[Methodology] = Field(default_factory=list)
    live_docs: list[dict[str, str]] = Field(default_factory=list)
    forbidden_approaches: list[str] = Field(default_factory=list)
    project_rules: Optional[str] = None
    council_diagnosis: Optional[str] = None
    retrieval_confidence: float = 0.0
    retrieval_conflicts: list[str] = Field(default_factory=list)
    retrieval_strategy_hint: Optional[str] = None
    skill_procedure: Optional[str] = None  # Item 13: matched skill procedure
    complexity_tier: Optional[str] = None  # Item 7: task complexity tier


class BuildResult(BaseModel):
    """Output of Builder — code generation results."""
    files_changed: list[str] = Field(default_factory=list)
    test_output: str = ""
    tests_passed: bool = False
    diff: str = ""
    approach_summary: str = ""
    model_used: Optional[str] = None
    failure_reason: Optional[str] = None
    failure_detail: Optional[str] = None


class SentinelVerdict(BaseModel):
    """Output of Sentinel — audit gate decision."""
    approved: bool = False
    violations: list[dict[str, str]] = Field(default_factory=list)  # [{check, detail}]
    recommendations: list[str] = Field(default_factory=list)
    quality_score: Optional[float] = None  # 0.0-1.0 numeric score (Item 5)


class CouncilDiagnosis(BaseModel):
    """Output of Council — peer review strategy."""
    strategy_shift: str
    new_approach: str
    reasoning: str
    model_used: str


# ---------------------------------------------------------------------------
# Enums for methodology taxonomy (Item 3)
# ---------------------------------------------------------------------------

class MethodologyType(str, enum.Enum):
    BUG_FIX = "BUG_FIX"
    PATTERN = "PATTERN"
    DECISION = "DECISION"
    GOTCHA = "GOTCHA"


# ---------------------------------------------------------------------------
# Token cost tracking (Item 1)
# ---------------------------------------------------------------------------

class TokenCostRecord(BaseModel):
    """Per-call token cost record for LLM usage attribution."""
    id: uuid.UUID = Field(default_factory=_new_uuid)
    task_id: Optional[uuid.UUID] = None
    run_id: Optional[uuid.UUID] = None
    agent_role: str = ""
    model_used: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    created_at: datetime = Field(default_factory=_now)


# ---------------------------------------------------------------------------
# Task complexity scoring (Item 7)
# ---------------------------------------------------------------------------

class ComplexityTier(str, enum.Enum):
    TRIVIAL = "TRIVIAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    VERY_HIGH = "VERY_HIGH"


# ---------------------------------------------------------------------------
# Typed execution state (Item 11)
# ---------------------------------------------------------------------------

class ExecutionState(BaseModel):
    """Shared typed execution state passed through the agent pipeline."""
    task_id: uuid.UUID
    run_id: Optional[uuid.UUID] = None
    trace_id: Optional[str] = None
    current_phase: str = "init"
    attempt_number: int = 0
    token_budget_remaining: int = 100_000
    tokens_used: int = 0
    complexity_tier: Optional[ComplexityTier] = None
    quality_score: Optional[float] = None


# ---------------------------------------------------------------------------
# Inter/Intra-agent packet protocol (APC/1.0)
# ---------------------------------------------------------------------------

class PacketType(str, enum.Enum):
    TASK_IR = "TASK_IR"
    IDEA = "IDEA"
    CLAIM = "CLAIM"
    ARTIFACT = "ARTIFACT"
    SIGNAL = "SIGNAL"
    EVAL = "EVAL"
    TRANSFER = "TRANSFER"
    ARBITRATION = "ARBITRATION"
    DECISION = "DECISION"
    RETRACTION = "RETRACTION"
    HEARTBEAT = "HEARTBEAT"
    ACK = "ACK"
    NACK = "NACK"
    ERROR = "ERROR"


class PacketChannel(str, enum.Enum):
    INTER_AGENT = "INTER_AGENT"
    INTRA_AGENT = "INTRA_AGENT"


class RunPhase(str, enum.Enum):
    RESEARCH = "RESEARCH"
    DISCOVERY = "DISCOVERY"
    REFINEMENT = "REFINEMENT"
    CONVERGENCE = "CONVERGENCE"
    ARBITRATION = "ARBITRATION"
    EXECUTION = "EXECUTION"
    VALIDATION = "VALIDATION"
    PROMOTION = "PROMOTION"


class DeliveryMode(str, enum.Enum):
    DIRECT = "DIRECT"
    MULTICAST = "MULTICAST"
    BROADCAST = "BROADCAST"
    SUBSTRATE = "SUBSTRATE"


class PacketActor(BaseModel):
    agent_id: str
    role: str
    swarm_id: str
    dialect_id: Optional[str] = None
    dialect_hash: Optional[str] = None
    model_id: Optional[str] = None
    capabilities: list[str] = Field(default_factory=list)


class PacketRecipient(BaseModel):
    agent_id: str
    role: Optional[str] = None
    swarm_id: Optional[str] = None


class PacketTrace(BaseModel):
    session_id: str
    run_id: str
    task_id: str
    root_packet_id: uuid.UUID
    parent_packet_id: Optional[uuid.UUID] = None
    generation: int = 0
    step: int = 0
    vector_clock: dict[str, int] = Field(default_factory=dict)


class PacketRouting(BaseModel):
    delivery_mode: DeliveryMode = DeliveryMode.DIRECT
    priority: int = Field(default=5, ge=0, le=9)
    ttl_ms: int = Field(default=60_000, ge=1, le=86_400_000)
    requires_ack: bool = False
    ack_timeout_ms: Optional[int] = Field(default=None, ge=1, le=3_600_000)
    retry_limit: int = Field(default=0, ge=0, le=10)


class FalsifierContract(BaseModel):
    id: str
    target_claim_id: str
    test: str
    fail_if: str
    cost: str


class ProofBundle(BaseModel):
    evidence_refs: list[str] = Field(default_factory=list)
    falsifiers: list[FalsifierContract] = Field(default_factory=list)
    gate_scores: dict[str, float] = Field(default_factory=dict)
    signatures: list[dict[str, str]] = Field(default_factory=list)


class PacketLineage(BaseModel):
    protocol_id: str
    parent_protocol_ids: list[str] = Field(default_factory=list)
    ancestor_swarms: list[str] = Field(default_factory=list)
    cross_use_case: bool = False
    transfer_mode: Optional[str] = None


class FormulaODESystem(BaseModel):
    """Machine-executable ODE system contract embedded in formula bundles."""
    state_variables: list[str]
    equations: list[str]
    parameters: dict[str, float] = Field(default_factory=dict)
    initial_state: dict[str, float] = Field(default_factory=dict)
    time_start: float = 0.0
    time_end: float = 50.0
    num_points: int = Field(default=500, ge=5, le=100_000)

    @model_validator(mode="after")
    def _validate_shape(self) -> "FormulaODESystem":
        if not self.state_variables:
            raise ValueError("state_variables must be non-empty")
        if len(self.equations) != len(self.state_variables):
            raise ValueError("equations must match state_variables cardinality")
        missing = [v for v in self.state_variables if v not in self.initial_state]
        if missing:
            raise ValueError(
                f"initial_state missing variables: {', '.join(sorted(missing))}"
            )
        if self.time_end <= self.time_start:
            raise ValueError("time_end must be greater than time_start")
        return self


class FormulaBundle(BaseModel):
    """Multi-representation formula packet payload (LaTeX + SymPy + AST + ODE)."""
    representation_version: str = "formula_bundle/1.0"
    latex_full: Optional[str] = None
    latex_compact: Optional[str] = None
    sympy_str: Optional[str] = None
    sympy_canonical: Optional[str] = None
    ast_json: Optional[dict[str, Any]] = None
    domains: list[str] = Field(default_factory=list)
    var_ontology: dict[str, str] = Field(default_factory=dict)
    functor_map: Optional[str] = None
    simulation_snippet: Optional[str] = None
    equilibria_sympy: Optional[str] = None
    ode_system: Optional[FormulaODESystem] = None

    @model_validator(mode="after")
    def _validate_representation(self) -> "FormulaBundle":
        if not any([self.latex_full, self.sympy_str, self.ast_json, self.ode_system]):
            raise ValueError(
                "formula bundle requires one of: latex_full, sympy_str, ast_json, ode_system"
            )
        return self


class TaskIRPayload(BaseModel):
    ir_header: dict[str, Any]
    objects: list[dict[str, Any]]
    normalization: dict[str, Any] = Field(default_factory=dict)
    formula_bundle: Optional[FormulaBundle] = None


class IdeaPayload(BaseModel):
    method_id: str
    intent: str
    assumptions: list[str]
    required_capabilities: list[str]
    expected_impact: dict[str, float]
    failure_modes: list[dict[str, str]]
    test_contract_refs: list[str] = Field(default_factory=list)
    language_bindings: list[dict[str, str]] = Field(default_factory=list)
    formula_bundle: Optional[FormulaBundle] = None


class ClaimPayload(BaseModel):
    claim_id: str
    hypothesis_id: str
    text: str
    confidence_pre: float = Field(ge=0, le=1)
    confidence_post: float = Field(ge=0, le=1)


class ArtifactPayload(BaseModel):
    artifact_id: str
    artifact_kind: str
    artifact_uri: str
    artifact_hash: str
    language: Optional[str] = None
    adapter: Optional[str] = None
    formula_bundle: Optional[FormulaBundle] = None


class SignalPayload(BaseModel):
    signal: str
    reason: Optional[str] = None
    expected_response_by: Optional[datetime] = None


class EvalPayload(BaseModel):
    metric_set: dict[str, float]
    thresholds: dict[str, float]
    passed: bool = Field(alias="pass")
    notes: list[str] = Field(default_factory=list)


class TransferPayload(BaseModel):
    protocol_id: str
    from_swarm: str
    to_swarm: str
    sender_score: float = Field(ge=0, le=1)
    receiver_score: float = Field(ge=0, le=1)
    transfer_policy: dict[str, Any]
    accepted: Optional[bool] = None


class ArbitrationPayload(BaseModel):
    arbitration_id: str
    candidates: list[dict[str, Any]]
    winner_candidate_id: str
    scoring: dict[str, float]
    vetoes: list[dict[str, str]] = Field(default_factory=list)


class DecisionPayload(BaseModel):
    decision_id: str
    recommendation: str
    priority_distribution: dict[str, float]
    next_actions: list[str]
    commitment_level: Optional[str] = None


class RetractionPayload(BaseModel):
    target_packet_id: uuid.UUID
    reason: str
    replacement_packet_id: Optional[uuid.UUID] = None


class AckPayload(BaseModel):
    acked_packet_id: uuid.UUID
    ack_status: str
    reason: Optional[str] = None


class ErrorPayload(BaseModel):
    error_code: str
    message: str
    retryable: bool
    details: dict[str, Any] = Field(default_factory=dict)


class AgentPacket(BaseModel):
    """Typed packet envelope for APC/1.0 inter/intra-agent communication."""
    protocol_version: str = "apc/1.0"
    packet_id: uuid.UUID = Field(default_factory=_new_uuid)
    packet_type: PacketType
    channel: PacketChannel
    run_phase: RunPhase
    created_at: datetime = Field(default_factory=_now)
    expires_at: Optional[datetime] = None
    sender: PacketActor
    recipients: list[PacketRecipient]
    trace: PacketTrace
    routing: PacketRouting
    confidence: float = Field(ge=0, le=1)
    symbolic_keys: list[str] = Field(default_factory=list)
    capability_tags: list[str] = Field(default_factory=list)
    lineage: Optional[PacketLineage] = None
    proof_bundle: Optional[ProofBundle] = None
    payload: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_packet(self) -> "AgentPacket":
        if self.protocol_version != "apc/1.0":
            raise ValueError("protocol_version must be 'apc/1.0'")
        if not self.recipients:
            raise ValueError("recipients must be non-empty")
        if self.channel == PacketChannel.INTRA_AGENT and len(self.recipients) > 1:
            raise ValueError("INTRA_AGENT packets may target only one recipient")

        payload_models: dict[PacketType, type[BaseModel]] = {
            PacketType.TASK_IR: TaskIRPayload,
            PacketType.IDEA: IdeaPayload,
            PacketType.CLAIM: ClaimPayload,
            PacketType.ARTIFACT: ArtifactPayload,
            PacketType.SIGNAL: SignalPayload,
            PacketType.HEARTBEAT: SignalPayload,
            PacketType.EVAL: EvalPayload,
            PacketType.TRANSFER: TransferPayload,
            PacketType.ARBITRATION: ArbitrationPayload,
            PacketType.DECISION: DecisionPayload,
            PacketType.RETRACTION: RetractionPayload,
            PacketType.ACK: AckPayload,
            PacketType.NACK: AckPayload,
            PacketType.ERROR: ErrorPayload,
        }
        model_cls = payload_models.get(self.packet_type)
        if model_cls is not None:
            model_cls.model_validate(self.payload)

        if self.packet_type in {PacketType.CLAIM, PacketType.EVAL} and self.proof_bundle is None:
            raise ValueError(f"{self.packet_type} packets require proof_bundle")
        if self.packet_type == PacketType.TRANSFER and self.lineage is None:
            raise ValueError("TRANSFER packets require lineage")
        return self
