"""Tests for data science pipeline Pydantic models.

Validates serialization/deserialization, defaults, and inter-model contracts
for all 7-phase DS pipeline models.
"""

from __future__ import annotations

import uuid

import pytest

from src.datasci.models import (
    ColumnProfile,
    DataAuditReport,
    DeploymentPackage,
    DSPipelineState,
    EDAReport,
    EnsembleReport,
    EvaluationReport,
    FeatureEngineeringReport,
    ModelCandidate,
    ModelTrainingReport,
)


class TestColumnProfile:
    def test_minimal_creation(self):
        cp = ColumnProfile(name="age", dtype="numeric")
        assert cp.name == "age"
        assert cp.dtype == "numeric"
        assert cp.cardinality == 0
        assert cp.missing_pct == 0.0
        assert cp.distribution_summary == {}
        assert cp.is_target is False
        assert cp.text_detected is False
        assert cp.label_quality_score is None
        assert cp.recommended_treatment == ""

    def test_full_creation(self):
        cp = ColumnProfile(
            name="description",
            dtype="text",
            cardinality=500,
            missing_pct=2.5,
            distribution_summary={"avg_length": 120, "unique_ratio": 0.95},
            is_target=False,
            text_detected=True,
            label_quality_score=0.85,
            recommended_treatment="embed_and_cluster",
        )
        assert cp.text_detected is True
        assert cp.label_quality_score == 0.85
        assert cp.distribution_summary["avg_length"] == 120

    def test_roundtrip_json(self):
        cp = ColumnProfile(
            name="price",
            dtype="numeric",
            cardinality=1000,
            missing_pct=0.5,
            distribution_summary={"mean": 50.0, "std": 12.3},
        )
        data = cp.model_dump()
        restored = ColumnProfile(**data)
        assert restored == cp

    def test_roundtrip_json_string(self):
        cp = ColumnProfile(name="status", dtype="categorical", cardinality=5)
        json_str = cp.model_dump_json()
        restored = ColumnProfile.model_validate_json(json_str)
        assert restored == cp


class TestDataAuditReport:
    def test_defaults(self):
        report = DataAuditReport(dataset_path="/data/train.csv", row_count=1000, column_count=15)
        assert report.dataset_path == "/data/train.csv"
        assert report.row_count == 1000
        assert report.column_count == 15
        assert report.column_profiles == []
        assert report.overall_quality_score == 0.0
        assert report.label_issues_count == 0
        assert report.llm_semantic_analysis == ""
        assert report.recommended_actions == []
        assert report.dataset_fingerprint == ""

    def test_with_profiles(self):
        profiles = [
            ColumnProfile(name="age", dtype="numeric", cardinality=80),
            ColumnProfile(name="name", dtype="text", text_detected=True),
        ]
        report = DataAuditReport(
            dataset_path="/data/train.csv",
            row_count=5000,
            column_count=2,
            column_profiles=profiles,
            overall_quality_score=0.87,
            label_issues_count=12,
            llm_semantic_analysis="Dataset contains demographic data with minor quality issues.",
            recommended_actions=["impute_missing", "check_label_noise"],
            dataset_fingerprint="sha256:abc123",
        )
        assert len(report.column_profiles) == 2
        assert report.column_profiles[0].name == "age"
        assert report.overall_quality_score == 0.87

    def test_roundtrip(self):
        report = DataAuditReport(
            dataset_path="/data/train.csv",
            row_count=100,
            column_count=5,
            recommended_actions=["drop_duplicates"],
        )
        restored = DataAuditReport(**report.model_dump())
        assert restored == report


class TestEDAReport:
    def test_defaults(self):
        report = EDAReport()
        assert report.questions_generated == []
        assert report.findings == []
        assert report.correlations == {}
        assert report.outlier_summary == {}
        assert report.text_column_analysis == []
        assert report.llm_synthesis == ""
        assert report.recommendations == []

    def test_populated(self):
        report = EDAReport(
            questions_generated=["What drives target?", "Are there temporal patterns?"],
            findings=[{"type": "correlation", "detail": "age strongly correlates with target"}],
            correlations={"age_target": 0.72, "income_target": 0.55},
            outlier_summary={"age": {"count": 3, "method": "IQR"}},
            text_column_analysis=[{"column": "notes", "avg_length": 200}],
            llm_synthesis="Dataset shows strong age-driven patterns.",
            recommendations=["investigate_age_nonlinearity"],
        )
        assert len(report.questions_generated) == 2
        assert report.correlations["age_target"] == 0.72

    def test_roundtrip(self):
        report = EDAReport(
            questions_generated=["Q1"],
            findings=[{"key": "val"}],
        )
        restored = EDAReport.model_validate_json(report.model_dump_json())
        assert restored == report


class TestFeatureEngineeringReport:
    def test_defaults(self):
        report = FeatureEngineeringReport()
        assert report.original_feature_count == 0
        assert report.new_features == []
        assert report.feature_ranking == []
        assert report.llm_fe_generations_run == 0
        assert report.best_generation_score == 0.0
        assert report.transformation_code == ""
        assert report.dropped_features == []

    def test_populated(self):
        report = FeatureEngineeringReport(
            original_feature_count=20,
            new_features=[
                {"name": "age_squared", "source": "age", "transform": "polynomial"},
                {"name": "name_embedding_0", "source": "name", "transform": "sentence_embedding"},
            ],
            feature_ranking=[{"age_squared": 0.85}, {"name_embedding_0": 0.72}],
            llm_fe_generations_run=5,
            best_generation_score=0.923,
            transformation_code="df['age_squared'] = df['age'] ** 2",
            dropped_features=["unused_id", "constant_col"],
        )
        assert report.original_feature_count == 20
        assert len(report.new_features) == 2
        assert report.best_generation_score == 0.923

    def test_roundtrip(self):
        report = FeatureEngineeringReport(original_feature_count=10, llm_fe_generations_run=3)
        restored = FeatureEngineeringReport(**report.model_dump())
        assert restored == report


class TestModelCandidate:
    def test_defaults(self):
        mc = ModelCandidate(model_name="xgb_v1", model_type="xgboost")
        assert mc.model_name == "xgb_v1"
        assert mc.model_type == "xgboost"
        assert mc.cv_scores == []
        assert mc.mean_score == 0.0
        assert mc.std_score == 0.0
        assert mc.training_time_seconds == 0.0
        assert mc.prediction_time_ms == 0.0
        assert mc.hyperparameters == {}
        assert mc.uncertainty_calibration is None
        assert mc.conformal_coverage is None
        assert mc.llm_behavioral_analysis == ""

    def test_populated(self):
        mc = ModelCandidate(
            model_name="tabpfn_default",
            model_type="tabpfn",
            cv_scores=[0.91, 0.93, 0.90, 0.92, 0.91],
            mean_score=0.914,
            std_score=0.011,
            training_time_seconds=0.5,
            prediction_time_ms=12.0,
            hyperparameters={"n_ensemble_configurations": 16},
            uncertainty_calibration=0.95,
            conformal_coverage=0.90,
            llm_behavioral_analysis="Excellent calibration on small data.",
        )
        assert len(mc.cv_scores) == 5
        assert mc.conformal_coverage == 0.90

    def test_roundtrip(self):
        mc = ModelCandidate(model_name="test", model_type="lightgbm", mean_score=0.85)
        restored = ModelCandidate.model_validate_json(mc.model_dump_json())
        assert restored == mc


class TestModelTrainingReport:
    def test_defaults(self):
        report = ModelTrainingReport()
        assert report.scale_tier == ""
        assert report.candidates == []
        assert report.best_candidate == ""
        assert report.nam_shape_functions is None
        assert report.llm_evaluation_narrative == ""

    def test_with_candidates(self):
        candidates = [
            ModelCandidate(model_name="xgb", model_type="xgboost", mean_score=0.88),
            ModelCandidate(model_name="tabpfn", model_type="tabpfn", mean_score=0.91),
        ]
        report = ModelTrainingReport(
            scale_tier="small",
            candidates=candidates,
            best_candidate="tabpfn",
            llm_evaluation_narrative="TabPFN outperforms XGBoost on this small dataset.",
        )
        assert len(report.candidates) == 2
        assert report.best_candidate == "tabpfn"

    def test_roundtrip(self):
        report = ModelTrainingReport(scale_tier="large", best_candidate="xgb")
        restored = ModelTrainingReport(**report.model_dump())
        assert restored == report


class TestEnsembleReport:
    def test_defaults(self):
        report = EnsembleReport()
        assert report.stacking_architecture == {}
        assert report.routing_rules == []
        assert report.pareto_front == []
        assert report.selected_configuration == ""
        assert report.llm_ensemble_analysis == ""

    def test_populated(self):
        report = EnsembleReport(
            stacking_architecture={
                "level_0": ["xgb", "tabpfn", "catboost"],
                "level_1": "logistic_regression",
            },
            routing_rules=[
                {"condition": "uncertainty > 0.3", "route_to": "tabpfn"},
            ],
            pareto_front=[
                {"accuracy": 0.92, "interpretability": 0.7, "speed": 0.9},
                {"accuracy": 0.95, "interpretability": 0.3, "speed": 0.5},
            ],
            selected_configuration="balanced_knee",
            llm_ensemble_analysis="Stacking improves by 2% over best single model.",
        )
        assert len(report.pareto_front) == 2
        assert report.selected_configuration == "balanced_knee"

    def test_roundtrip(self):
        report = EnsembleReport(selected_configuration="fast")
        restored = EnsembleReport.model_validate_json(report.model_dump_json())
        assert restored == report


class TestEvaluationReport:
    def test_defaults(self):
        report = EvaluationReport()
        assert report.predictive_scores == {}
        assert report.robustness_scores == {}
        assert report.fairness_scores is None
        assert report.interpretability_scores == {}
        assert report.feature_importance == []
        assert report.shap_summary is None
        assert report.baseline_comparison == {}
        assert report.llm_evaluation_narrative == ""
        assert report.overall_grade == ""
        assert report.recommendations == []

    def test_populated(self):
        report = EvaluationReport(
            predictive_scores={"accuracy": 0.93, "f1": 0.91, "auc": 0.96},
            robustness_scores={"perturbation_drop": 0.02, "noise_tolerance": 0.95},
            fairness_scores={"demographic_parity": 0.98, "equalized_odds": 0.97},
            interpretability_scores={"shap_consistency": 0.88, "nam_coverage": 0.92},
            feature_importance=[{"age": 0.35}, {"income": 0.28}],
            shap_summary={"top_features": ["age", "income"]},
            baseline_comparison={"vs_majority": 0.43, "vs_logistic": 0.12},
            llm_evaluation_narrative="Model achieves strong performance with good calibration.",
            overall_grade="A-",
            recommendations=["monitor_drift_on_age"],
        )
        assert report.overall_grade == "A-"
        assert report.fairness_scores["demographic_parity"] == 0.98

    def test_roundtrip(self):
        report = EvaluationReport(overall_grade="B+", predictive_scores={"acc": 0.85})
        restored = EvaluationReport(**report.model_dump())
        assert restored == report


class TestDeploymentPackage:
    def test_defaults(self):
        pkg = DeploymentPackage()
        assert pkg.model_artifact_path == ""
        assert pkg.feature_pipeline_path == ""
        assert pkg.api_scaffold_code == ""
        assert pkg.monitoring_config == {}
        assert pkg.data_contract == {}
        assert pkg.llm_completeness_review == ""

    def test_populated(self):
        pkg = DeploymentPackage(
            model_artifact_path="/artifacts/model.joblib",
            feature_pipeline_path="/artifacts/pipeline.joblib",
            api_scaffold_code="from fastapi import FastAPI\napp = FastAPI()",
            monitoring_config={"drift_threshold": 0.05, "check_interval_hours": 24},
            data_contract={"input_schema": {"age": "float", "name": "str"}},
            llm_completeness_review="All required artifacts present and valid.",
        )
        assert pkg.model_artifact_path == "/artifacts/model.joblib"
        assert pkg.monitoring_config["drift_threshold"] == 0.05

    def test_roundtrip(self):
        pkg = DeploymentPackage(model_artifact_path="/m.joblib")
        restored = DeploymentPackage.model_validate_json(pkg.model_dump_json())
        assert restored == pkg


class TestDSPipelineState:
    def test_minimal_creation(self):
        pid = uuid.uuid4()
        state = DSPipelineState(
            dataset_path="/data/train.csv",
            target_column="label",
            project_id=pid,
        )
        assert state.dataset_path == "/data/train.csv"
        assert state.target_column == "label"
        assert state.problem_type == "classification"
        assert state.project_id == pid
        assert state.run_id is None
        assert state.sensitive_columns == []
        assert state.audit_report is None
        assert state.eda_report is None
        assert state.feature_report is None
        assert state.training_report is None
        assert state.ensemble_report is None
        assert state.evaluation_report is None
        assert state.deployment_package is None

    def test_regression_type(self):
        state = DSPipelineState(
            dataset_path="/data/prices.csv",
            target_column="price",
            problem_type="regression",
            project_id=uuid.uuid4(),
        )
        assert state.problem_type == "regression"

    def test_with_reports_attached(self):
        """Simulate full pipeline flow — each phase attaches its report."""
        pid = uuid.uuid4()
        rid = uuid.uuid4()

        state = DSPipelineState(
            dataset_path="/data/train.csv",
            target_column="label",
            project_id=pid,
            run_id=rid,
            sensitive_columns=["gender", "ethnicity"],
        )

        # Phase 1: attach audit
        state.audit_report = DataAuditReport(
            dataset_path="/data/train.csv",
            row_count=5000,
            column_count=20,
        )
        assert state.audit_report.row_count == 5000

        # Phase 2: attach EDA
        state.eda_report = EDAReport(
            questions_generated=["What predicts the target?"],
        )
        assert len(state.eda_report.questions_generated) == 1

        # Phase 3: attach feature eng
        state.feature_report = FeatureEngineeringReport(
            original_feature_count=20,
            llm_fe_generations_run=5,
        )
        assert state.feature_report.llm_fe_generations_run == 5

        # Phase 4: attach training
        state.training_report = ModelTrainingReport(
            scale_tier="small",
            best_candidate="tabpfn",
        )
        assert state.training_report.best_candidate == "tabpfn"

        # Phase 5: attach ensemble
        state.ensemble_report = EnsembleReport(
            selected_configuration="balanced",
        )
        assert state.ensemble_report.selected_configuration == "balanced"

        # Phase 6: attach evaluation
        state.evaluation_report = EvaluationReport(
            overall_grade="A",
            predictive_scores={"accuracy": 0.95},
        )
        assert state.evaluation_report.overall_grade == "A"

        # Phase 7: attach deployment
        state.deployment_package = DeploymentPackage(
            model_artifact_path="/artifacts/model.joblib",
        )
        assert state.deployment_package.model_artifact_path == "/artifacts/model.joblib"

    def test_roundtrip_full_state(self):
        """Full pipeline state serializes and deserializes correctly."""
        pid = uuid.uuid4()
        state = DSPipelineState(
            dataset_path="/data/train.csv",
            target_column="label",
            project_id=pid,
            audit_report=DataAuditReport(
                dataset_path="/data/train.csv",
                row_count=1000,
                column_count=10,
            ),
            eda_report=EDAReport(questions_generated=["Q1"]),
            training_report=ModelTrainingReport(
                scale_tier="small",
                candidates=[
                    ModelCandidate(model_name="xgb", model_type="xgboost", mean_score=0.88),
                ],
            ),
        )
        json_str = state.model_dump_json()
        restored = DSPipelineState.model_validate_json(json_str)
        assert restored.project_id == pid
        assert restored.audit_report.row_count == 1000
        assert restored.eda_report.questions_generated == ["Q1"]
        assert len(restored.training_report.candidates) == 1
        assert restored.training_report.candidates[0].mean_score == 0.88

    def test_sensitive_columns(self):
        state = DSPipelineState(
            dataset_path="/data/train.csv",
            target_column="target",
            project_id=uuid.uuid4(),
            sensitive_columns=["race", "gender", "age"],
        )
        assert len(state.sensitive_columns) == 3
        assert "race" in state.sensitive_columns
