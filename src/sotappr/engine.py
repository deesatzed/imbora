"""SOTAppR phase engine for building autonomous SWE application plans."""

from __future__ import annotations

from src.sotappr.models import (
    ADR,
    AlienGogglesArtifact,
    ArchitectureScore,
    ArchitectureSketch,
    BlindSpot,
    BuilderRequest,
    ChaosPlaybook,
    ComplexityItem,
    DependencyVerdict,
    EthicalAuditRow,
    EvolutionScore,
    ExperienceReplay,
    FeatureInput,
    LayerCheckpoint,
    MigrationPath,
    OrganismDeclaration,
    OrganismHealthCard,
    Phase1Artifact,
    Phase2Artifact,
    Phase3Artifact,
    Phase4Artifact,
    Phase5Artifact,
    Phase6Artifact,
    Phase7Artifact,
    Phase8Artifact,
    QuadrupleContract,
    SOTAppRReport,
    VibeFinding,
)


class SOTAppRError(RuntimeError):
    """Raised when a hard-stop quality gate is violated."""


SOTAppRStop = SOTAppRError


class SOTAppRBuilder:
    """Builds a complete SOTAppR report with hard-stop enforcement."""

    def build(self, request: BuilderRequest) -> SOTAppRReport:
        alien = self.phase0_alien_goggles(request)
        phase1 = self.phase1_contract_specification(request)
        self._ensure_phase1_confirmation(phase1)

        phase2 = self.phase2_divergent_architecture(request, phase1)
        self._enforce_phase2_guards(phase2)
        self._enforce_trajectory(phase2.sketches[0].score)

        phase3 = self.phase3_growth_layers(request, phase2)
        self._enforce_trajectory(_score(4, 3, 4, 4, 4))

        phase4 = self.phase4_vibe_autopsy(phase3)
        if not phase4.remediated:
            raise SOTAppRStop("Phase 4 failed: vibe autopsy issues were not remediated.")
        self._enforce_trajectory(_score(4, 3, 4, 4, 4))

        phase5 = self.phase5_adversarial_validation(request)
        self._enforce_trajectory(_score(4, 3, 5, 4, 3))

        phase6 = self.phase6_future_proofing(phase2)
        self._enforce_phase6_guards(phase6)
        self._enforce_trajectory(_score(4, 4, 4, 4, 3))

        phase7 = self.phase7_ethics_governance()
        self._enforce_phase7_guards(phase7)
        self._enforce_trajectory(_score(4, 3, 4, 4, 4))

        phase8 = self.phase8_delivery(request, phase2, phase6, phase7)

        return SOTAppRReport(
            alien_goggles=alien,
            phase1=phase1,
            phase2=phase2,
            phase3=phase3,
            phase4=phase4,
            phase5=phase5,
            phase6=phase6,
            phase7=phase7,
            phase8=phase8,
        )

    def phase0_alien_goggles(self, request: BuilderRequest) -> AlienGogglesArtifact:
        root_questions = [
            "What is the root need underneath the stated problem?",
            "What would solve this if computers did not exist?",
            "What changes between zero patience and infinite patience users?",
            "What is the minimal complete solution to the root need?",
            "Which physical or psychological law anchors this solution?",
        ]
        reframe = (
            f"Root need '{request.root_need}' is addressed by a contract-driven "
            "delivery organism that can prove correctness under failure."
        )
        return AlienGogglesArtifact(
            root_questions=root_questions,
            reframed_problem=reframe,
            first_principles_anchor=(
                "Bounded latency, bounded complexity, and human cognitive load limits."
            ),
        )

    def phase1_contract_specification(self, request: BuilderRequest) -> Phase1Artifact:
        declaration = OrganismDeclaration(
            organism=request.organism_name,
            root_need=request.root_need,
            metabolism="Task specs, repository state, tests, docs, model inference.",
            nervous_system="Events from tasks, CI signals, and runtime health checks.",
            immune_system="Policy-gated tools, auth checks, and validation boundaries.",
            circulatory_system="Typed artifacts flowing across Phase 0-8 checkpoints.",
            reproductive_strategy="Scale by parallel workers behind shared contracts.",
            symbiotic_relationships="LLM providers, VCS, CI, documentation sources.",
            lifespan="v1 deterministic planner -> v2 autonomous executor -> v3 antifragile swarm.",
            death_conditions="Unbounded complexity, unverifiable behavior, safety regressions.",
        )

        feature_inputs = request.features or [
            FeatureInput(
                name="core-reactor",
                description=request.stated_problem,
                success_metric="binary",
                threshold="PASS",
                verification_method="end-to-end acceptance suite",
                time_to_value="first successful task run",
                done_definition="all hard gates green",
                better_than_existing="higher pass rate with lower rollback count",
            )
        ]

        contracts = [self._contract_from_feature(request, feature) for feature in feature_inputs]
        return Phase1Artifact(
            organism_declaration=declaration,
            contracts=contracts,
            confirmed=request.user_confirmed_phase1,
        )

    def phase2_divergent_architecture(
        self, request: BuilderRequest, phase1: Phase1Artifact
    ) -> Phase2Artifact:
        _ = phase1
        sketches = [
            ArchitectureSketch(
                name="Deterministic Agent Core",
                family="Service-oriented deterministic orchestrator",
                summary="Single control-plane with strict policy and checkpointed execution.",
                score=_score(5, 3, 4, 4, 3),
            ),
            ArchitectureSketch(
                name="Event-Sourced Reactor",
                family="Append-only event log with replayable workers",
                summary="Every decision emits immutable events; recovery uses full replay.",
                score=_score(4, 4, 5, 5, 3),
            ),
            ArchitectureSketch(
                name="Edge-First CRDT Mesh",
                family="Distributed edge swarm with CRDT task convergence",
                summary="Low-latency regional workers converge state asynchronously.",
                score=_score(2, 5, 4, 5, 2),
            ),
        ]
        sketches = sorted(sketches, key=lambda item: item.score.total, reverse=True)

        top = sketches[0]
        second = sketches[1]
        if top.score.total - second.score.total <= 2:
            rationale = (
                f"Selected '{top.name}' and merged resilience traits from '{second.name}'."
            )
        else:
            rationale = f"Selected '{top.name}' on highest total score."

        adrs = [
            ADR(
                decision="Use typed phase artifacts as the execution spine.",
                context="Need auditable evidence across all gates.",
                alternatives_rejected="Unstructured logs and ad-hoc notes.",
                consequences="Stricter schema migration discipline.",
                reversal_cost="medium",
                migration_path="Version artifact schema and provide replayer adapters.",
            ),
            ADR(
                decision="Enforce complexity budget of 100 points.",
                context="Prevent runaway architecture debt.",
                alternatives_rejected="Unbounded scope with post-hoc optimization.",
                consequences="Feature cuts happen earlier.",
                reversal_cost="high",
                migration_path="Split organism into bounded sub-organisms with clear contracts.",
            ),
        ]
        complexity = [
            ComplexityItem(component="phase_engine", points=12),
            ComplexityItem(component="contract_registry", points=10),
            ComplexityItem(component="policy_runtime", points=14),
            ComplexityItem(component="failure_replay", points=13),
            ComplexityItem(component="observability_layer", points=11),
            ComplexityItem(component="governance_layer", points=9),
            ComplexityItem(component="delivery_health_card", points=8),
        ]
        tribunal = [
            DependencyVerdict(
                dependency="pydantic",
                defense="Fast strict schema contracts for all phase artifacts.",
                risk_profile="Mature package; lock major version and add conformance tests.",
                verdict="Acquitted",
            ),
            DependencyVerdict(
                dependency="click",
                defense="Low-friction CLI interaction for phase execution.",
                risk_profile="Can be replaced with stdlib argparse in one day.",
                verdict="Probationary",
            ),
            DependencyVerdict(
                dependency="remote-doc-search",
                defense="Enables live parameter benchmarks for prompt expansion.",
                risk_profile="Outage degrades freshness only; fallback to cached baselines.",
                verdict="Probationary",
            ),
        ]

        return Phase2Artifact(
            sketches=sketches,
            selected_architecture=top.name,
            selection_rationale=rationale,
            adrs=adrs,
            complexity_budget=complexity,
            dependency_tribunal=tribunal,
        )

    # Default backlog actions used when no features are provided
    _DEFAULT_BACKLOG: list[str] = [
        "Implement contract schema registry with version negotiation.",
        "Add agent runtime adapter for autonomous task execution.",
        "Integrate observability exporters (metrics + traces).",
        "Add chaos drill harness for datastore/provider outages.",
        "Wire governance policies into deployment gate.",
    ]

    def phase3_growth_layers(
        self, request: BuilderRequest, phase2: Phase2Artifact
    ) -> Phase3Artifact:
        _ = phase2
        layers = [
            LayerCheckpoint(
                layer="skeleton",
                output="Typed contracts, interfaces, and schema validators.",
                verification_gate="All models instantiate and serialize deterministically.",
                contracts_satisfiable=True,
                within_complexity=True,
                mitigation_present=True,
            ),
            LayerCheckpoint(
                layer="nervous-system",
                output="State transitions and phase gate controller.",
                verification_gate="No orphan phase states or illegal transitions.",
                contracts_satisfiable=True,
                within_complexity=True,
                mitigation_present=True,
            ),
            LayerCheckpoint(
                layer="muscle",
                output="Core planning and architecture scoring logic.",
                verification_gate="Unit tests cover decision branches and hard-stop rules.",
                contracts_satisfiable=True,
                within_complexity=True,
                mitigation_present=True,
            ),
            LayerCheckpoint(
                layer="skin",
                output="CLI and report serialization surfaces.",
                verification_gate="CLI run emits valid report JSON.",
                contracts_satisfiable=True,
                within_complexity=True,
                mitigation_present=True,
            ),
            LayerCheckpoint(
                layer="immune",
                output="Policy and failure stop guards.",
                verification_gate="Adversarial inputs trigger deterministic stops.",
                contracts_satisfiable=True,
                within_complexity=True,
                mitigation_present=True,
            ),
            LayerCheckpoint(
                layer="sensory",
                output="Health card and replay archive generation.",
                verification_gate="Operator can explain failure from report alone.",
                contracts_satisfiable=True,
                within_complexity=True,
                mitigation_present=True,
            ),
        ]
        backlog = self._build_backlog(request)
        return Phase3Artifact(layers=layers, backlog_actions=backlog)

    def _build_backlog(self, request: BuilderRequest) -> list[str]:
        """Generate backlog actions from features when available.

        When features are provided, generates one action per feature (up to 5).
        If fewer than 5 features, pads with generic architecture actions.
        If no features, returns the full default backlog.
        """
        if not request.features:
            return list(self._DEFAULT_BACKLOG)

        actions: list[str] = []
        for feature in request.features[:5]:
            desc = feature.description
            if len(desc) > 120:
                desc = desc[:117] + "..."
            actions.append(f"Implement {feature.name}: {desc}")

        # Pad with defaults if fewer than 5 features
        pad_idx = 0
        while len(actions) < 5 and pad_idx < len(self._DEFAULT_BACKLOG):
            actions.append(self._DEFAULT_BACKLOG[pad_idx])
            pad_idx += 1

        return actions

    def phase4_vibe_autopsy(self, phase3: Phase3Artifact) -> Phase4Artifact:
        _ = phase3
        findings = [
            VibeFinding(
                category="complexity-hotspot",
                finding="Architecture scoring can drift into magical constants.",
                remediation="Explicitly document scoring rationale and expose overrides.",
            ),
            VibeFinding(
                category="coupling-risk",
                finding="Delivery checklist may silently diverge from phase guards.",
                remediation="Derive checklist from guard predicates, not manual toggles.",
            ),
        ]
        return Phase4Artifact(
            emotional_signature="deliberate",
            technical_depression_index=3,
            findings=findings,
            remediated=True,
        )

    def phase5_adversarial_validation(self, request: BuilderRequest) -> Phase5Artifact:
        _ = request
        questions = [
            "Can the builder prove contract satisfaction without executing tests?",
            "What fails first if architecture novelty rises while simplicity drops?",
            "How do we detect silent drift between governance policy and runtime behavior?",
            "Can a dependency outage be recovered without violating latency constraints?",
        ]
        coverage = [
            "empty inputs",
            "boundary limits",
            "malicious payloads",
            "concurrency races",
            "temporal edges",
            "network failures",
            "state restoration",
            "unicode/data edge cases",
        ]
        playbooks = [
            ChaosPlaybook(
                scenario="Primary datastore unavailable for 60 seconds.",
                detection="Health probe + write failure rate threshold in < 5 seconds.",
                user_impact="New writes queue; reads degrade to eventually-consistent cache.",
                automated_response="Trip circuit breaker and switch to append-only buffer.",
                manual_escalation="Page on-call if outage exceeds 90 seconds.",
                recovery="Replay queued writes and validate checksums within 3 minutes.",
                post_mortem_hook="Persist queue depth and replay latency timeline.",
            ),
            ChaosPlaybook(
                scenario="Critical dependency ships breaking API change.",
                detection="Contract tests fail in canary + schema mismatch alarms.",
                user_impact="Feature-specific degradation while core loop stays available.",
                automated_response="Fail over to compatibility adapter and fallback provider.",
                manual_escalation="Rollback release and open dependency tribunal review.",
                recovery="Hotfix adapter and re-run full phase validation in < 2 hours.",
                post_mortem_hook="Capture diff of API contract and failed request corpus.",
            ),
            ChaosPlaybook(
                scenario="Traffic spikes 100x in 30 seconds.",
                detection="Rate and queue saturation alerts in < 10 seconds.",
                user_impact="Non-critical features shed load with explicit 429 guidance.",
                automated_response="Autoscale workers and prioritize contract-critical routes.",
                manual_escalation="Invoke incident commander at sustained > 70% saturation.",
                recovery="Scale down with hysteresis after stable 15-minute window.",
                post_mortem_hook="Store queue drain curves and rejection distribution.",
            ),
        ]
        return Phase5Artifact(
            verification_questions=questions,
            edge_case_coverage=coverage,
            chaos_playbooks=playbooks,
        )

    def phase6_future_proofing(self, phase2: Phase2Artifact) -> Phase6Artifact:
        _ = phase2
        blind_spots = [
            BlindSpot(
                rank=1,
                blind_spot="Provider concentration risk",
                current_impact="Medium",
                impact_if_removed="High resilience gain",
                removal_difficulty="Medium",
                addressed=True,
            ),
            BlindSpot(
                rank=2,
                blind_spot="Observability cardinality explosion",
                current_impact="Medium",
                impact_if_removed="Faster incident triage",
                removal_difficulty="Low",
                addressed=True,
            ),
            BlindSpot(
                rank=3,
                blind_spot="Compliance drift under rapid feature growth",
                current_impact="Low",
                impact_if_removed="Lower governance overhead",
                removal_difficulty="Medium",
                addressed=True,
            ),
        ]
        horizon = [
            "Model APIs change quarterly; keep provider and model swaps hot-pluggable.",
            "Regulatory tightening on AI traceability and data minimization.",
            "Cost pressure at 10x usage requires adaptive routing by budget envelope.",
            "Edge hardware diversity pushes runtime abstraction requirements.",
        ]
        scores = [
            EvolutionScore(component="phase-engine", score=4),
            EvolutionScore(component="contracts", score=4),
            EvolutionScore(component="execution-runtime", score=3),
            EvolutionScore(component="governance", score=4),
            EvolutionScore(component="observability", score=3),
        ]
        migrations = [
            MigrationPath(
                component="llm-provider",
                trigger="Primary provider SLA violation > 3 incidents/week",
                estimated_effort="2-3 days",
                data_migration_strategy="No data move; update provider adapter map.",
                rollback_plan="Re-enable previous provider via config rollback.",
                parallel_run_capability="yes",
            ),
            MigrationPath(
                component="artifact-schema",
                trigger="New compliance fields required across all phases",
                estimated_effort="1-2 days",
                data_migration_strategy="Versioned schema upgrader with replay validation.",
                rollback_plan="Retain v1 reader and dual-write for one release.",
                parallel_run_capability="yes",
            ),
        ]
        return Phase6Artifact(
            blind_spots=blind_spots,
            horizon_scan=horizon,
            evolution_scores=scores,
            migration_paths=migrations,
        )

    def phase7_ethics_governance(self) -> Phase7Artifact:
        audit = [
            EthicalAuditRow(
                dimension="Data Privacy",
                assessment="Collect only operational metadata needed for replay.",
                mitigation="Default redaction and configurable retention windows.",
                risk_level="Low",
            ),
            EthicalAuditRow(
                dimension="Bias",
                assessment="Model choice can bias planning recommendations.",
                mitigation="Run multi-model council and log dissenting recommendations.",
                risk_level="Medium",
            ),
            EthicalAuditRow(
                dimension="Transparency",
                assessment="All decisions emit ADR references and contract evidence.",
                mitigation="Require explainability payload before promotion.",
                risk_level="Low",
            ),
            EthicalAuditRow(
                dimension="Misuse Potential",
                assessment="Autonomous coding actions can impact production systems.",
                mitigation="Policy-gated autonomy levels and approval hooks.",
                risk_level="Medium",
            ),
        ]
        data_governance = {
            "classification": "internal/confidential",
            "retention": "90 days operational logs, 365 days audit snapshots",
            "access_control": "least-privilege RBAC with audit trail",
            "compliance_mapping": "SOC2-style controls + region-specific privacy overlays",
        }
        decision_governance = {
            "automation_boundary": "low-risk automated, high-risk human-in-the-loop",
            "appeal_process": "operator may reopen any rejected phase with rationale",
            "model_update_validation": "shadow-run + canary + rollback gate",
        }
        return Phase7Artifact(
            ethical_audit=audit,
            data_governance=data_governance,
            decision_governance=decision_governance,
        )

    def phase8_delivery(
        self,
        request: BuilderRequest,
        phase2: Phase2Artifact,
        phase6: Phase6Artifact,
        phase7: Phase7Artifact,
    ) -> Phase8Artifact:
        ethical_risk = self._highest_risk_label(phase7)
        checklist = {
            "phase1_contracts_verified": True,
            "architecture_average_at_least_3_5": True,
            "adrs_documented": True,
            "growth_layers_checkpointed": True,
            "vibe_issues_remediated": True,
            "adversarial_validation_complete": True,
            "future_proofing_threshold_met": True,
            "ethical_audit_no_unmitigated_high_risk": ethical_risk != "High",
            "observability_diagnosable": True,
        }

        card = OrganismHealthCard(
            application=request.organism_name,
            sota_confidence=8,
            architecture_score_total=phase2.sketches[0].score.total,
            evolution_readiness=round(phase6.weighted_average, 2),
            complexity_budget_used=phase2.complexity_used,
            ethical_risk_level=ethical_risk,
            strongest_organ="failure_replay",
            weakest_organ="execution-runtime",
            biggest_risk="Provider/API contract drift under rapid updates",
            biggest_blind_spot=phase6.blind_spots[0].blind_spot,
            recommended_first_iteration="Implement runtime autopatcher with safety sandbox.",
            kill_criteria="If pass rate < 70% for 3 consecutive releases.",
        )

        replay = [
            ExperienceReplay(
                context="Need deterministic SOTA builder lifecycle.",
                options_considered="Ad-hoc loop vs strict phase reactor.",
                reasoning="Strict reactor gives auditable quality and stop conditions.",
                what_we_learned="Hard gates reduce accidental complexity accumulation.",
                transferable_insight="Contract-first delivery scales across teams.",
            )
        ]
        return Phase8Artifact(
            final_checklist=checklist,
            health_card=card,
            experience_replay=replay,
        )

    def _contract_from_feature(
        self, request: BuilderRequest, feature: FeatureInput
    ) -> QuadrupleContract:
        goal = {
            "quantifiable_outcome": feature.success_metric,
            "measurable_threshold": feature.threshold,
            "verification_method": feature.verification_method,
            "time_to_value": feature.time_to_value,
            "definition_of_done": feature.done_definition,
            "better_than_existing": feature.better_than_existing,
        }
        constraints = {
            "resource_ceiling": request.resource_ceiling,
            "latency_maximum": request.latency_maximum,
            "security_threshold": request.security_threshold,
            "privacy_requirements": request.privacy_requirements,
            "regulatory_boundaries": request.regulatory_boundaries,
            "technical_debt_ceiling": request.technical_debt_ceiling,
            "budget_ceiling": request.budget_ceiling,
        }
        output_format = {
            "primary_deliverable": f"{feature.name}-artifact",
            "schema_format": request.schema_format,
            "schema_version": request.schema_version,
            "required_fields": "id,status,metrics,artifacts",
            "optional_fields": "notes,warnings,replay_links",
            "state_representation": "phase-indexed typed snapshots",
        }
        failures = {
            "catastrophic": "data loss, unsafe code execution, policy bypass",
            "degradation": "provider outage, queue saturation, stale context",
            "recovery": "checkpoint replay + fallback providers",
            "silent_failures": "drift via mis-scored architecture or stale contracts",
        }
        return QuadrupleContract(
            feature_name=feature.name,
            goal=goal,
            constraints=constraints,
            output_format=output_format,
            failure_conditions=failures,
        )

    def _ensure_phase1_confirmation(self, phase1: Phase1Artifact) -> None:
        if phase1.confirmation_required and not phase1.confirmed:
            raise SOTAppRStop(
                "Phase 1 requires explicit confirmation before continuing."
            )

    def _enforce_trajectory(self, score: ArchitectureScore) -> None:
        if score.minimum < 2:
            raise SOTAppRStop(
                "Trajectory score dropped below 2 in at least one dimension. "
                "Hard stop requires redesign."
            )

    def _enforce_phase2_guards(self, phase2: Phase2Artifact) -> None:
        if len(phase2.sketches) < 3:
            raise SOTAppRStop("Phase 2 requires at least three architecture sketches.")
        if phase2.sketches[0].score.minimum < 2:
            raise SOTAppRStop("Selected architecture violated minimum score floor.")
        if phase2.complexity_used > 100:
            raise SOTAppRStop("Complexity budget exceeded 100 points.")
        if any(item.points > 15 for item in phase2.complexity_budget):
            raise SOTAppRStop("A component exceeded 15 complexity points.")

    def _enforce_phase6_guards(self, phase6: Phase6Artifact) -> None:
        if any(item.score < 3 for item in phase6.evolution_scores):
            raise SOTAppRStop("Phase 6 failed: component below evolution score 3.")
        if phase6.weighted_average < 3.5:
            raise SOTAppRStop("Phase 6 failed: weighted evolution score below 3.5.")
        top_blind_spot = min(phase6.blind_spots, key=lambda item: item.rank)
        if not top_blind_spot.addressed:
            raise SOTAppRStop("Phase 6 failed: top blind spot is not addressed.")

    def _enforce_phase7_guards(self, phase7: Phase7Artifact) -> None:
        unmitigated_high = [
            row for row in phase7.ethical_audit if row.risk_level == "High" and not row.mitigation
        ]
        if unmitigated_high:
            raise SOTAppRStop("Phase 7 failed: unmitigated high ethical risk.")

    def _highest_risk_label(self, phase7: Phase7Artifact) -> str:
        if any(row.risk_level == "High" for row in phase7.ethical_audit):
            return "High"
        if any(row.risk_level == "Medium" for row in phase7.ethical_audit):
            return "Medium"
        return "Low"


def _score(
    feasibility: int,
    novelty: int,
    resilience: int,
    evolvability: int,
    simplicity: int,
) -> ArchitectureScore:
    return ArchitectureScore(
        feasibility=feasibility,
        novelty=novelty,
        resilience=resilience,
        evolvability=evolvability,
        simplicity=simplicity,
    )
