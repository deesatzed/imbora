"""Data contracts for the SOTAppR builder lifecycle."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(UTC)


class FeatureInput(BaseModel):
    """Requested feature for the target application."""

    name: str
    description: str
    success_metric: str = "binary"
    threshold: str = "PASS"
    verification_method: str = "integration-test"
    time_to_value: str = "same-session"
    done_definition: str = "passes acceptance tests"
    better_than_existing: str = "lower failure rate and latency"


class DataScienceTaskSpec(BaseModel):
    """Specification for a data science task seeded from SOTAppR."""

    dataset_path: str
    target_column: str
    problem_type: str = "classification"  # classification or regression
    sensitive_columns: list[str] = Field(default_factory=list)


class BuilderRequest(BaseModel):
    """Input request for creating a SOTAppR application plan."""

    organism_name: str
    stated_problem: str
    root_need: str
    features: list[FeatureInput] = Field(default_factory=list)
    resource_ceiling: str = "single-node <= 4 CPU / 8 GB RAM"
    latency_maximum: str = "p95 <= 2000 ms"
    security_threshold: str = "OWASP ASVS L2 baseline"
    privacy_requirements: str = "no secret logging / least privilege"
    regulatory_boundaries: str = "none"
    technical_debt_ceiling: str = "no unowned TODOs in critical path"
    budget_ceiling: str = "$500/month initial ops"
    schema_format: Literal["json", "yaml", "protobuf"] = "json"
    schema_version: str = "v1"
    user_confirmed_phase1: bool = False
    data_science_task: DataScienceTaskSpec | None = None


class AlienGogglesArtifact(BaseModel):
    phase: int = 0
    root_questions: list[str]
    reframed_problem: str
    first_principles_anchor: str
    created_at: datetime = Field(default_factory=_now)


class OrganismDeclaration(BaseModel):
    organism: str
    root_need: str
    metabolism: str
    nervous_system: str
    immune_system: str
    circulatory_system: str
    reproductive_strategy: str
    symbiotic_relationships: str
    lifespan: str
    death_conditions: str


class QuadrupleContract(BaseModel):
    feature_name: str
    goal: dict[str, str]
    constraints: dict[str, str]
    output_format: dict[str, str]
    failure_conditions: dict[str, str]


class Phase1Artifact(BaseModel):
    phase: int = 1
    organism_declaration: OrganismDeclaration
    contracts: list[QuadrupleContract]
    confirmation_required: bool = True
    confirmed: bool = False
    created_at: datetime = Field(default_factory=_now)


class ArchitectureScore(BaseModel):
    feasibility: int
    novelty: int
    resilience: int
    evolvability: int
    simplicity: int

    @property
    def total(self) -> int:
        return (
            self.feasibility
            + self.novelty
            + self.resilience
            + self.evolvability
            + self.simplicity
        )

    @property
    def minimum(self) -> int:
        return min(
            self.feasibility,
            self.novelty,
            self.resilience,
            self.evolvability,
            self.simplicity,
        )


class ArchitectureSketch(BaseModel):
    name: str
    family: str
    summary: str
    score: ArchitectureScore
    contracts_recomputed: bool = True


class ADR(BaseModel):
    decision: str
    context: str
    alternatives_rejected: str
    consequences: str
    reversal_cost: Literal["low", "medium", "high", "catastrophic"]
    migration_path: str


class ComplexityItem(BaseModel):
    component: str
    points: int


class DependencyVerdict(BaseModel):
    dependency: str
    defense: str
    risk_profile: str
    verdict: Literal["Acquitted", "Probationary", "Rejected"]


class Phase2Artifact(BaseModel):
    phase: int = 2
    sketches: list[ArchitectureSketch]
    selected_architecture: str
    selection_rationale: str
    adrs: list[ADR]
    complexity_budget: list[ComplexityItem]
    dependency_tribunal: list[DependencyVerdict]
    created_at: datetime = Field(default_factory=_now)

    @property
    def complexity_used(self) -> int:
        return sum(item.points for item in self.complexity_budget)


class LayerCheckpoint(BaseModel):
    layer: str
    output: str
    verification_gate: str
    contracts_satisfiable: bool
    within_complexity: bool
    mitigation_present: bool


class Phase3Artifact(BaseModel):
    phase: int = 3
    layers: list[LayerCheckpoint]
    backlog_actions: list[str]
    created_at: datetime = Field(default_factory=_now)


class VibeFinding(BaseModel):
    category: str
    finding: str
    remediation: str


class Phase4Artifact(BaseModel):
    phase: int = 4
    emotional_signature: str
    technical_depression_index: int
    findings: list[VibeFinding]
    remediated: bool
    created_at: datetime = Field(default_factory=_now)


class ChaosPlaybook(BaseModel):
    scenario: str
    detection: str
    user_impact: str
    automated_response: str
    manual_escalation: str
    recovery: str
    post_mortem_hook: str


class Phase5Artifact(BaseModel):
    phase: int = 5
    verification_questions: list[str]
    edge_case_coverage: list[str]
    chaos_playbooks: list[ChaosPlaybook]
    created_at: datetime = Field(default_factory=_now)


class BlindSpot(BaseModel):
    rank: int
    blind_spot: str
    current_impact: str
    impact_if_removed: str
    removal_difficulty: str
    addressed: bool


class EvolutionScore(BaseModel):
    component: str
    score: int


class MigrationPath(BaseModel):
    component: str
    trigger: str
    estimated_effort: str
    data_migration_strategy: str
    rollback_plan: str
    parallel_run_capability: str


class Phase6Artifact(BaseModel):
    phase: int = 6
    blind_spots: list[BlindSpot]
    horizon_scan: list[str]
    evolution_scores: list[EvolutionScore]
    migration_paths: list[MigrationPath]
    created_at: datetime = Field(default_factory=_now)

    @property
    def weighted_average(self) -> float:
        if not self.evolution_scores:
            return 0.0
        return sum(score.score for score in self.evolution_scores) / len(
            self.evolution_scores
        )


class EthicalAuditRow(BaseModel):
    dimension: str
    assessment: str
    mitigation: str
    risk_level: Literal["Low", "Medium", "High"]


class Phase7Artifact(BaseModel):
    phase: int = 7
    ethical_audit: list[EthicalAuditRow]
    data_governance: dict[str, str]
    decision_governance: dict[str, str]
    created_at: datetime = Field(default_factory=_now)


class OrganismHealthCard(BaseModel):
    application: str
    sota_confidence: int
    architecture_score_total: int
    evolution_readiness: float
    complexity_budget_used: int
    ethical_risk_level: str
    strongest_organ: str
    weakest_organ: str
    biggest_risk: str
    biggest_blind_spot: str
    recommended_first_iteration: str
    kill_criteria: str


class ExperienceReplay(BaseModel):
    context: str
    options_considered: str
    reasoning: str
    what_we_learned: str
    transferable_insight: str


class Phase8Artifact(BaseModel):
    phase: int = 8
    final_checklist: dict[str, bool]
    health_card: OrganismHealthCard
    experience_replay: list[ExperienceReplay]
    created_at: datetime = Field(default_factory=_now)


class SOTAppRReport(BaseModel):
    """Complete report across all SOTAppR phases."""

    alien_goggles: AlienGogglesArtifact
    phase1: Phase1Artifact
    phase2: Phase2Artifact
    phase3: Phase3Artifact
    phase4: Phase4Artifact
    phase5: Phase5Artifact
    phase6: Phase6Artifact
    phase7: Phase7Artifact
    phase8: Phase8Artifact
    generated_at: datetime = Field(default_factory=_now)
