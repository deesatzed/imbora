"""Pydantic models for the data science pipeline.

Defines all inter-agent contracts for the 8-phase DS pipeline:
DataAudit -> EDA -> FeatureEngineering -> DataAugmentation -> ModelTraining -> Ensemble -> Evaluation -> Deployment
"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    """Profile of a single dataset column."""
    name: str
    dtype: str  # 'numeric', 'categorical', 'text', 'datetime', 'boolean'
    cardinality: int = 0
    missing_pct: float = 0.0
    distribution_summary: dict[str, Any] = Field(default_factory=dict)
    is_target: bool = False
    text_detected: bool = False
    label_quality_score: Optional[float] = None
    recommended_treatment: str = ""
    semantic_meaning: str = ""
    semantic_dtype: str = ""
    imputation_strategy: str = ""
    encoding_strategy: str = ""
    outlier_strategy: str = ""
    text_processing_strategy: str = ""
    interaction_candidates: list[str] = Field(default_factory=list)
    data_quality_flags: list[str] = Field(default_factory=list)
    importance_prior: str = ""
    correlation_with_target: Optional[float] = None


class DataAuditReport(BaseModel):
    """Output of Phase 1: Data Audit Agent."""
    dataset_path: str
    row_count: int
    column_count: int
    column_profiles: list[ColumnProfile] = Field(default_factory=list)
    overall_quality_score: float = 0.0
    label_issues_count: int = 0
    llm_semantic_analysis: str = ""
    recommended_actions: list[str] = Field(default_factory=list)
    dataset_fingerprint: str = ""


class EDAReport(BaseModel):
    """Output of Phase 2: EDA Agent."""
    questions_generated: list[str] = Field(default_factory=list)
    findings: list[dict[str, Any]] = Field(default_factory=list)
    correlations: dict[str, float] = Field(default_factory=dict)
    outlier_summary: dict[str, Any] = Field(default_factory=dict)
    text_column_analysis: list[dict[str, Any]] = Field(default_factory=list)
    llm_synthesis: str = ""
    recommendations: list[str] = Field(default_factory=list)
    imbalance_ratio: Optional[float] = None
    class_counts: Optional[dict[str, int]] = None
    is_imbalanced: bool = False
    minority_class: Optional[str] = None
    majority_class: Optional[str] = None


class FECategoryApplicability(BaseModel):
    """Assessment of a single feature engineering category."""
    category: str  # e.g. 'cross_column_derivation', 'text_feature_extraction'
    applicable: bool = False
    confidence: float = 0.0  # 0.0-1.0
    rationale: str = ""
    applicable_columns: list[str] = Field(default_factory=list)


class ProposedFeature(BaseModel):
    """A specific feature proposed by the assessment."""
    name: str
    category: str
    source_columns: list[str] = Field(default_factory=list)
    transform_description: str = ""
    expected_impact: str = ""  # 'high', 'medium', 'low'
    rationale: str = ""


class FeatureEngineeringAssessment(BaseModel):
    """Output of Phase 2.5: FE Assessment."""
    dataset_summary: str = ""
    total_columns: int = 0
    numeric_columns: int = 0
    categorical_columns: int = 0
    text_columns: int = 0
    datetime_columns: int = 0
    category_assessments: list[FECategoryApplicability] = Field(default_factory=list)
    proposed_features: list[ProposedFeature] = Field(default_factory=list)
    overall_fe_potential: str = ""  # 'high', 'medium', 'low', 'none'
    llm_reasoning: str = ""


class FeatureEngineeringReport(BaseModel):
    """Output of Phase 3: Feature Engineering Agent."""
    original_feature_count: int = 0
    new_features: list[dict[str, Any]] = Field(default_factory=list)
    feature_ranking: list[dict[str, Any]] = Field(default_factory=list)
    llm_fe_generations_run: int = 0
    best_generation_score: float = 0.0
    transformation_code: str = ""
    dropped_features: list[str] = Field(default_factory=list)
    enhanced_dataset_path: str = ""


class AugmentationReport(BaseModel):
    """Output of Phase 3.5: Data Augmentation Agent."""
    augmented: bool = False
    strategy_used: str = "none"
    original_class_counts: dict[str, int] = Field(default_factory=dict)
    augmented_class_counts: dict[str, int] = Field(default_factory=dict)
    samples_generated: int = 0
    augmented_dataset_path: str = ""
    adversarial_validation_score: float = 0.0
    quality_score: float = 0.0
    llm_quality_review: str = ""


class ModelCandidate(BaseModel):
    """A single model candidate with training results."""
    model_name: str
    model_type: str  # 'tabpfn', 'tabicl', 'xgboost', 'catboost', 'lightgbm', 'nam'
    cv_scores: list[float] = Field(default_factory=list)
    mean_score: float = 0.0
    std_score: float = 0.0
    training_time_seconds: float = 0.0
    prediction_time_ms: float = 0.0
    hyperparameters: dict[str, Any] = Field(default_factory=dict)
    uncertainty_calibration: Optional[float] = None
    conformal_coverage: Optional[float] = None
    calibration_brier_score: Optional[float] = None
    optimal_threshold: Optional[float] = None
    pr_auc: Optional[float] = None
    llm_behavioral_analysis: str = ""


class ModelTrainingReport(BaseModel):
    """Output of Phase 4: Model Training Agent."""
    scale_tier: str = ""  # 'small', 'medium', 'large'
    candidates: list[ModelCandidate] = Field(default_factory=list)
    best_candidate: str = ""
    nam_shape_functions: Optional[dict[str, Any]] = None
    llm_evaluation_narrative: str = ""


class EnsembleReport(BaseModel):
    """Output of Phase 5: Ensemble Agent."""
    stacking_architecture: dict[str, Any] = Field(default_factory=dict)
    routing_rules: list[dict[str, Any]] = Field(default_factory=list)
    pareto_front: list[dict[str, float]] = Field(default_factory=list)
    selected_configuration: str = ""
    llm_ensemble_analysis: str = ""


class EvaluationReport(BaseModel):
    """Output of Phase 6: Evaluation Agent."""
    predictive_scores: dict[str, float] = Field(default_factory=dict)
    robustness_scores: dict[str, Any] = Field(default_factory=dict)
    fairness_scores: Optional[dict[str, float]] = None
    interpretability_scores: dict[str, float] = Field(default_factory=dict)
    feature_importance: list[dict[str, float]] = Field(default_factory=list)
    shap_summary: Optional[dict[str, Any]] = None
    baseline_comparison: dict[str, float] = Field(default_factory=dict)
    llm_evaluation_narrative: str = ""
    overall_grade: str = ""
    recommendations: list[str] = Field(default_factory=list)


class DeploymentPackage(BaseModel):
    """Output of Phase 7: Deployment Agent."""
    model_artifact_path: str = ""
    feature_pipeline_path: str = ""
    api_scaffold_code: str = ""
    monitoring_config: dict[str, Any] = Field(default_factory=dict)
    data_contract: dict[str, Any] = Field(default_factory=dict)
    llm_completeness_review: str = ""


class DSPipelineState(BaseModel):
    """Shared state flowing through the 7-phase DS pipeline."""
    dataset_path: str
    target_column: str
    problem_type: str = "classification"  # 'classification' or 'regression'
    project_id: uuid.UUID
    run_id: Optional[uuid.UUID] = None
    sensitive_columns: list[str] = Field(default_factory=list)
    audit_report: Optional[DataAuditReport] = None
    eda_report: Optional[EDAReport] = None
    fe_assessment: Optional[FeatureEngineeringAssessment] = None
    feature_report: Optional[FeatureEngineeringReport] = None
    augmentation_report: Optional[AugmentationReport] = None
    training_report: Optional[ModelTrainingReport] = None
    ensemble_report: Optional[EnsembleReport] = None
    evaluation_report: Optional[EvaluationReport] = None
    deployment_package: Optional[DeploymentPackage] = None
