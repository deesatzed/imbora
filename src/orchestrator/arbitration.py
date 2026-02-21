"""Build candidate arbitration for multi-path orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.core.models import AgentResult, BuildResult


@dataclass
class CandidateScore:
    """Score breakdown for one builder candidate."""

    candidate_index: int
    total: float
    tests_signal: float
    output_signal: float
    reliability_signal: float
    risk_penalty: float
    notes: list[str] = field(default_factory=list)


@dataclass
class ArbitrationDecision:
    """Selected candidate and score evidence."""

    selected_index: int
    selected_result: AgentResult
    selected_build_result: BuildResult
    scores: list[CandidateScore]
    ranked_indices: list[int] = field(default_factory=list)
    tie_break_reason: str | None = None
    vetoed_candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_index": self.selected_index,
            "ranked_indices": list(self.ranked_indices),
            "tie_break_reason": self.tie_break_reason,
            "vetoed_candidates": list(self.vetoed_candidates),
            "scores": [
                {
                    "candidate_index": s.candidate_index,
                    "total": round(s.total, 3),
                    "tests_signal": round(s.tests_signal, 3),
                    "output_signal": round(s.output_signal, 3),
                    "reliability_signal": round(s.reliability_signal, 3),
                    "risk_penalty": round(s.risk_penalty, 3),
                    "notes": list(s.notes),
                }
                for s in self.scores
            ],
        }


class BuildArbiter:
    """Scores builder outputs and selects the strongest candidate."""

    def __init__(self, soft_file_limit: int = 12):
        self.soft_file_limit = soft_file_limit

    def choose(
        self,
        candidates: list[tuple[AgentResult, BuildResult]],
        retrieval_confidence: float = 0.0,
        vetoes: dict[int, list[str]] | None = None,
    ) -> ArbitrationDecision:
        if not candidates:
            raise ValueError("BuildArbiter received no candidates")
        vetoes = vetoes or {}

        scored: list[tuple[CandidateScore, AgentResult, BuildResult]] = []
        for idx, (agent_result, build_result) in enumerate(candidates, start=1):
            score = self._score_candidate(
                candidate_index=idx,
                agent_result=agent_result,
                build_result=build_result,
                retrieval_confidence=retrieval_confidence,
            )
            scored.append((score, agent_result, build_result))

        ranked_all = sorted(scored, key=self._ranking_key, reverse=True)
        ranked_eligible = [
            row for row in ranked_all
            if row[0].candidate_index not in vetoes
        ]
        ranked_pool = ranked_eligible or ranked_all
        best_score, best_result, best_build = ranked_pool[0]
        score_list = [row[0] for row in sorted(scored, key=lambda x: x[0].candidate_index)]
        tie_break_reason = self._tie_break_reason(ranked_pool)
        vetoed_candidates = [
            {
                "candidate_index": idx,
                "reasons": reasons,
            }
            for idx, reasons in sorted(vetoes.items())
        ]
        return ArbitrationDecision(
            selected_index=best_score.candidate_index,
            selected_result=best_result,
            selected_build_result=best_build,
            scores=score_list,
            ranked_indices=[row[0].candidate_index for row in ranked_pool],
            tie_break_reason=tie_break_reason,
            vetoed_candidates=vetoed_candidates,
        )

    def _ranking_key(self, row: tuple[CandidateScore, AgentResult, BuildResult]) -> tuple[float, bool, int, float, int]:
        score, _agent_result, build_result = row
        # Explicit tie-break policy:
        # 1) higher total score
        # 2) tests passed
        # 3) smaller bounded change set
        # 4) lower risk penalty
        # 5) stable deterministic fallback by candidate index
        return (
            score.total,
            build_result.tests_passed,
            -len(build_result.files_changed or []),
            -score.risk_penalty,
            -score.candidate_index,
        )

    @staticmethod
    def _tie_break_reason(ranked: list[tuple[CandidateScore, AgentResult, BuildResult]]) -> str | None:
        if len(ranked) < 2:
            return None
        best = ranked[0]
        second = ranked[1]
        if abs(best[0].total - second[0].total) > 1e-9:
            return None
        if best[2].tests_passed != second[2].tests_passed:
            return "tests_passed_preference"
        if len(best[2].files_changed or []) != len(second[2].files_changed or []):
            return "smaller_change_set_preference"
        if abs(best[0].risk_penalty - second[0].risk_penalty) > 1e-9:
            return "lower_risk_penalty_preference"
        return "stable_candidate_order_preference"

    def _score_candidate(
        self,
        candidate_index: int,
        agent_result: AgentResult,
        build_result: BuildResult,
        retrieval_confidence: float,
    ) -> CandidateScore:
        notes: list[str] = []

        tests_signal = 0.70 if build_result.tests_passed and agent_result.status == "success" else 0.0
        if tests_signal:
            notes.append("tests_passed")

        file_count = len(build_result.files_changed or [])
        output_signal = 0.0
        if file_count > 0:
            output_signal = min(0.20, 0.04 * min(file_count, 5))
            notes.append(f"produced_files={file_count}")
            if file_count <= self.soft_file_limit:
                output_signal += 0.05
                notes.append("bounded_change_set")

        reliability_signal = 0.0
        if agent_result.status == "success":
            reliability_signal += 0.05
            if retrieval_confidence >= 0.75:
                reliability_signal += 0.05
                notes.append("high_retrieval_confidence")

        risk_penalty = 0.0
        if file_count == 0:
            risk_penalty += 0.30
            notes.append("no_artifacts")
        if file_count > self.soft_file_limit:
            risk_penalty += 0.10
            notes.append("large_change_set")
        if build_result.failure_reason:
            risk_penalty += 0.20
            notes.append(f"failure_reason={build_result.failure_reason}")
        if agent_result.status != "success":
            risk_penalty += 0.20
            notes.append(f"agent_status={agent_result.status}")

        total = tests_signal + output_signal + reliability_signal - risk_penalty
        return CandidateScore(
            candidate_index=candidate_index,
            total=total,
            tests_signal=tests_signal,
            output_signal=output_signal,
            reliability_signal=reliability_signal,
            risk_penalty=risk_penalty,
            notes=notes,
        )
