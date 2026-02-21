"""Guided pre-execution workflow artifacts for The Associate."""

from __future__ import annotations

import json
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(UTC)


class IntakeCard(BaseModel):
    session_id: str
    task: str
    target_path: str
    project_mode: str
    created_at: datetime = Field(default_factory=_now)


class AlignmentSpec(BaseModel):
    ux_expectations: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    methodology: str
    acceptance_criteria: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    approved: bool = False


class BudgetContract(BaseModel):
    estimated_hours: float
    estimated_cost_usd: float
    estimated_tokens: int
    max_hours: float
    max_cost_usd: float
    approved: bool = False


class KnowledgeItem(BaseModel):
    source_type: str  # local, url, query
    source: str
    saved_path: Optional[str] = None
    note: Optional[str] = None


class KnowledgeManifest(BaseModel):
    knowledge_dir: str
    items: list[KnowledgeItem] = Field(default_factory=list)


class ExecutionCharter(BaseModel):
    anticipated_end_goal: str
    execution_summary: str
    next_actions: list[str] = Field(default_factory=list)
    approved: bool = False
    created_at: datetime = Field(default_factory=_now)


class RuntimeManifest(BaseModel):
    schema_profile: str
    agent_profiles: dict[str, str] = Field(default_factory=dict)
    prompt_templates: list[str] = Field(default_factory=list)
    knowledge_links: list[str] = Field(default_factory=list)
    mutable_runtime: bool = True


class SessionPlan(BaseModel):
    session_id: str
    intake: IntakeCard
    alignment: AlignmentSpec
    budget: BudgetContract
    knowledge: KnowledgeManifest
    charter: ExecutionCharter
    runtime: RuntimeManifest
    status: str = "draft"


def estimate_budget(task: str, feature_count: int, knowledge_count: int) -> BudgetContract:
    """Heuristic estimate used as a starting contract."""
    word_count = max(1, len(task.split()))
    base_hours = 2.0 + (0.6 * feature_count) + (0.25 * knowledge_count)
    complexity_factor = 1.0 + min(word_count / 220.0, 1.5)
    estimated_hours = round(base_hours * complexity_factor, 1)

    estimated_tokens = int(
        (18_000 + (feature_count * 9_000) + (knowledge_count * 4_000)) * complexity_factor
    )
    estimated_cost_usd = round((estimated_tokens / 1000.0) * 0.02, 2)

    return BudgetContract(
        estimated_hours=estimated_hours,
        estimated_cost_usd=estimated_cost_usd,
        estimated_tokens=estimated_tokens,
        max_hours=estimated_hours,
        max_cost_usd=estimated_cost_usd,
        approved=False,
    )


def materialize_knowledge_items(
    *,
    knowledge_dir: Path,
    local_paths: list[Path],
    urls: list[str],
    query_results: Optional[dict[str, list[dict[str, Any]]]] = None,
) -> KnowledgeManifest:
    """Copy local knowledge and persist URL/query metadata into one knowledge manifest."""
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    items: list[KnowledgeItem] = []

    local_root = knowledge_dir / "local"
    local_root.mkdir(parents=True, exist_ok=True)
    for source in local_paths:
        target = _resolve_copy_target(local_root=local_root, source=source)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        items.append(
            KnowledgeItem(
                source_type="local",
                source=str(source),
                saved_path=str(target),
            )
        )

    if urls:
        url_file = knowledge_dir / "external_urls.json"
        url_payload = [{"url": value, "added_at": _now().isoformat()} for value in urls]
        url_file.write_text(json.dumps(url_payload, indent=2, ensure_ascii=True), encoding="utf-8")
        for value in urls:
            items.append(
                KnowledgeItem(
                    source_type="url",
                    source=value,
                    saved_path=str(url_file),
                )
            )

    if query_results:
        queries_dir = knowledge_dir / "queries"
        queries_dir.mkdir(parents=True, exist_ok=True)
        for query, results in query_results.items():
            filename = _sanitize_filename(query) + ".json"
            out_path = queries_dir / filename
            payload = {
                "query": query,
                "saved_at": _now().isoformat(),
                "results": results,
            }
            out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
            items.append(
                KnowledgeItem(
                    source_type="query",
                    source=query,
                    saved_path=str(out_path),
                    note=f"{len(results)} result(s)",
                )
            )

    return KnowledgeManifest(
        knowledge_dir=str(knowledge_dir),
        items=items,
    )


def build_runtime_manifest(
    *,
    model_roles: dict[str, str],
    prompts_dir: Path,
    knowledge_manifest: KnowledgeManifest,
) -> RuntimeManifest:
    prompt_templates = sorted(
        str(path) for path in prompts_dir.glob("*") if path.is_file()
    )
    knowledge_links = [item.saved_path for item in knowledge_manifest.items if item.saved_path]
    return RuntimeManifest(
        schema_profile="adaptive_sotappr_v1",
        agent_profiles=model_roles,
        prompt_templates=prompt_templates,
        knowledge_links=knowledge_links,
        mutable_runtime=True,
    )


def write_session_artifacts(plan: SessionPlan, session_root: Path) -> dict[str, Path]:
    """Write individual gate artifacts + full session plan JSON."""
    session_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "intake": session_root / "intake_card.json",
        "alignment": session_root / "alignment_spec.json",
        "budget": session_root / "budget_contract.json",
        "knowledge": session_root / "knowledge_manifest.json",
        "charter": session_root / "execution_charter.json",
        "runtime": session_root / "runtime_manifest.json",
        "session": session_root / "session_plan.json",
    }
    outputs["intake"].write_text(plan.intake.model_dump_json(indent=2), encoding="utf-8")
    outputs["alignment"].write_text(plan.alignment.model_dump_json(indent=2), encoding="utf-8")
    outputs["budget"].write_text(plan.budget.model_dump_json(indent=2), encoding="utf-8")
    outputs["knowledge"].write_text(plan.knowledge.model_dump_json(indent=2), encoding="utf-8")
    outputs["charter"].write_text(plan.charter.model_dump_json(indent=2), encoding="utf-8")
    outputs["runtime"].write_text(plan.runtime.model_dump_json(indent=2), encoding="utf-8")
    outputs["session"].write_text(plan.model_dump_json(indent=2), encoding="utf-8")
    return outputs


def _resolve_copy_target(local_root: Path, source: Path) -> Path:
    stem = _sanitize_filename(source.name or source.stem or "knowledge")
    candidate = local_root / stem
    if not candidate.exists():
        return candidate
    suffix = 2
    while True:
        next_candidate = local_root / f"{stem}_{suffix}"
        if not next_candidate.exists():
            return next_candidate
        suffix += 1


def _sanitize_filename(raw: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw.strip())
    cleaned = cleaned.strip("_")
    if not cleaned:
        return "item_" + uuid.uuid4().hex[:8]
    return cleaned[:120]
