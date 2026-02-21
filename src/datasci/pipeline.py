"""DS Pipeline Orchestrator — sequences all data science agents.

Orchestrates the full data science pipeline:
  Phase 1:   DataAuditAgent                -> DataAuditReport
  Phase 2:   EDAAgent                      -> EDAReport
  Phase 2.5: FEAssessor                    -> FeatureEngineeringAssessment
  Phase 3:   FeatureEngineeringAgent       -> FeatureEngineeringReport
  Phase 3.5: DataAugmentationAgent         -> AugmentationReport
  Phase 4:   ModelTrainingAgent            -> ModelTrainingReport
  Phase 5:   EnsembleAgent                 -> EnsembleReport
  Phase 6:   EvaluationAgent               -> EvaluationReport
  Phase 7:   DeploymentAgent               -> DeploymentPackage

Each phase result feeds into the next via the input_data dict.
An LLM quality gate reviews each phase output before proceeding.

Phase 2.5 (FE Assessment) analyzes dataset characteristics and uses
LLM reasoning to determine which FE categories are applicable before
Phase 3 runs the expensive LLM-FE evolutionary loop.

Phase 3.5 (Data Augmentation) sits between Feature Engineering and
Model Training. If augmentation occurs, the augmented dataset path
is passed downstream for model fitting, while the original dataset
path is preserved for evaluation/calibration to prevent data leakage.
"""

from __future__ import annotations

import logging
import time
import traceback
import uuid
from typing import Any

from src.core.config import DataScienceConfig
from src.core.models import AgentResult
from src.llm.client import LLMMessage
from src.datasci.agents.data_audit import DataAuditAgent
from src.datasci.agents.data_augmentation import DataAugmentationAgent
from src.datasci.agents.deployment import DeploymentAgent
from src.datasci.agents.eda import EDAAgent
from src.datasci.agents.ensemble import EnsembleAgent
from src.datasci.agents.evaluation import EvaluationAgent
from src.datasci.agents.feature_engineering import FeatureEngineeringAgent
from src.datasci.agents.model_training import ModelTrainingAgent
from src.datasci.models import (
    AugmentationReport,
    DataAuditReport,
    DeploymentPackage,
    DSPipelineState,
    EDAReport,
    EnsembleReport,
    EvaluationReport,
    FeatureEngineeringReport,
    ModelTrainingReport,
)

logger = logging.getLogger("associate.datasci.pipeline")

QUALITY_GATE_SYSTEM_PROMPT = (
    "You are a senior data science reviewer performing a quality gate check.\n"
    "Given the output of a pipeline phase, determine if it passes quality standards.\n"
    "Respond with exactly one of these two lines as your first line:\n"
    "PASS: <brief reason>\n"
    "FAIL: <brief reason>\n\n"
    "Then optionally add 1-3 sentences of detail. Be concise and specific.\n"
    "A phase passes if it produced meaningful, non-empty results without errors.\n"
    "A phase fails only if the output is empty, nonsensical, or contains critical errors."
)

# Total number of pipeline phases (including Phase 2.5 and Phase 3.5)
TOTAL_PHASES = 9


class DSPipelineOrchestrator:
    """Orchestrates the 8-phase data science pipeline.

    Sequences all DS agents, passing results between phases,
    saving experiment records, and running LLM quality gates
    after each phase.
    """

    def __init__(
        self,
        llm_client: Any,
        model_router: Any,
        repository: Any,
        ds_config: DataScienceConfig,
    ):
        self.llm_client = llm_client
        self.model_router = model_router
        self.repository = repository
        self.ds_config = ds_config

        # Shared agent constructor kwargs
        self._agent_kwargs = {
            "llm_client": llm_client,
            "model_router": model_router,
            "repository": repository,
            "ds_config": ds_config,
        }

    def run(self, input_data: dict[str, Any]) -> DSPipelineState:
        """Execute the full 8-phase DS pipeline.

        Args:
            input_data: Dict with required keys:
                - dataset_path: str -- path to the dataset file
                - target_column: str -- name of target variable
                - project_id: UUID -- project identifier
                - problem_type: str -- 'classification' or 'regression'
                - sensitive_columns: list[str] (optional) -- columns for fairness checks
                - task_id: UUID (optional) -- parent task identifier
                - run_id: UUID (optional) -- parent run identifier

        Returns:
            DSPipelineState with all completed phase reports attached.
            On failure in any phase, returns partial state with reports
            from successfully completed phases.
        """
        # Validate required keys
        dataset_path = input_data["dataset_path"]
        target_column = input_data["target_column"]
        project_id = input_data["project_id"]
        problem_type = input_data.get("problem_type", "classification")
        sensitive_columns = input_data.get("sensitive_columns", [])
        task_id = input_data.get("task_id")
        run_id = input_data.get("run_id")

        if isinstance(project_id, str):
            project_id = uuid.UUID(project_id)

        # Store the original dataset path for evaluation/calibration (no leakage)
        original_dataset_path = dataset_path

        # Initialize pipeline state
        state = DSPipelineState(
            dataset_path=dataset_path,
            target_column=target_column,
            problem_type=problem_type,
            project_id=project_id,
            run_id=uuid.UUID(run_id) if isinstance(run_id, str) else run_id,
            sensitive_columns=sensitive_columns,
        )

        logger.info(
            "Starting DS pipeline: dataset=%s, target=%s, problem_type=%s, project_id=%s",
            dataset_path,
            target_column,
            problem_type,
            project_id,
        )

        # Build the base input_data dict that gets enriched phase-by-phase
        pipeline_input: dict[str, Any] = {
            "dataset_path": dataset_path,
            "target_column": target_column,
            "project_id": project_id,
            "problem_type": problem_type,
            "sensitive_columns": sensitive_columns,
        }
        if task_id is not None:
            pipeline_input["task_id"] = task_id
        if run_id is not None:
            pipeline_input["run_id"] = run_id

        # Save a pipeline-level experiment record
        pipeline_experiment_id = self._save_pipeline_experiment(
            project_id=project_id,
            config={
                "dataset_path": dataset_path,
                "target_column": target_column,
                "problem_type": problem_type,
                "sensitive_columns": sensitive_columns,
            },
            task_id=task_id,
            run_id=run_id,
        )

        pipeline_start = time.perf_counter()

        # --- Phase 1: Data Audit ---
        phase_result = self._run_phase(
            phase_number=1,
            phase_name="data_audit",
            agent_class=DataAuditAgent,
            input_data=pipeline_input,
            state=state,
        )
        if phase_result is None:
            state = self._finalize_state(state, "failed")
            self._complete_pipeline_experiment(
                pipeline_experiment_id, state, pipeline_start, "FAILED",
            )
            return state
        state.audit_report = DataAuditReport(**phase_result)
        pipeline_input["audit_report"] = phase_result

        # --- Phase 2: EDA ---
        phase_result = self._run_phase(
            phase_number=2,
            phase_name="eda",
            agent_class=EDAAgent,
            input_data=pipeline_input,
            state=state,
        )
        if phase_result is None:
            state = self._finalize_state(state, "failed")
            self._complete_pipeline_experiment(
                pipeline_experiment_id, state, pipeline_start, "FAILED",
            )
            return state
        state.eda_report = EDAReport(**phase_result)
        pipeline_input["eda_report"] = phase_result

        # --- Phase 2.5: FE Assessment ---
        if self.ds_config.enable_fe_assessment:
            try:
                import pandas as pd  # noqa: E402

                from src.datasci.fe_assessment import FEAssessor

                assessor = FEAssessor(self.llm_client, self.model_router)
                fe_assessment = assessor.assess(
                    df=pd.read_csv(dataset_path)
                    if dataset_path.endswith(".csv")
                    else pd.read_parquet(dataset_path),
                    target_column=target_column,
                    problem_type=problem_type,
                    column_profiles=state.audit_report.column_profiles,
                    eda_report=pipeline_input.get("eda_report", {}),
                )
                state.fe_assessment = fe_assessment
                pipeline_input["fe_assessment"] = fe_assessment.model_dump()
                logger.info(
                    "FE Assessment complete: potential=%s, %d categories applicable, "
                    "%d features proposed",
                    fe_assessment.overall_fe_potential,
                    sum(1 for c in fe_assessment.category_assessments if c.applicable),
                    len(fe_assessment.proposed_features),
                )
            except Exception as e:
                logger.warning(
                    "Phase 2.5 (FE Assessment) failed: %s. "
                    "Proceeding without assessment (non-fatal).",
                    e,
                )

        # --- Phase 3: Feature Engineering ---
        phase_result = self._run_phase(
            phase_number=3,
            phase_name="feature_engineering",
            agent_class=FeatureEngineeringAgent,
            input_data=pipeline_input,
            state=state,
        )
        if phase_result is None:
            state = self._finalize_state(state, "failed")
            self._complete_pipeline_experiment(
                pipeline_experiment_id, state, pipeline_start, "FAILED",
            )
            return state
        state.feature_report = FeatureEngineeringReport(**phase_result)
        pipeline_input["feature_report"] = phase_result

        # If feature engineering produced an enhanced dataset, use it for
        # training/ensemble but keep original for evaluation (leakage prevention).
        enhanced_path = phase_result.get("enhanced_dataset_path", "")
        if enhanced_path:
            pipeline_input["dataset_path"] = enhanced_path
            logger.info(
                "Using enhanced feature dataset at %s for training phases; "
                "original dataset %s preserved for evaluation.",
                enhanced_path, original_dataset_path,
            )

        # --- Phase 3.5: Data Augmentation ---
        # Augmentation needs the original dataset (not enhanced features) to
        # compute class counts and apply resampling on the raw target column.
        augmentation_input = dict(pipeline_input)
        augmentation_input["dataset_path"] = original_dataset_path
        phase_result = self._run_phase(
            phase_number=3,
            phase_name="data_augmentation",
            agent_class=DataAugmentationAgent,
            input_data=augmentation_input,
            state=state,
        )
        if phase_result is None:
            # Data augmentation failure is non-fatal: proceed without augmentation
            logger.warning(
                "Phase 3.5 (data_augmentation) failed; proceeding without augmentation.",
            )
            pipeline_input["augmentation_report"] = {
                "augmented": False,
                "strategy_used": "none",
            }
        else:
            state.augmentation_report = AugmentationReport(**phase_result)
            pipeline_input["augmentation_report"] = phase_result

            # If augmentation produced an augmented dataset, log the intent.
            # The augmentation_report contains the augmented_dataset_path
            # which model_training.py reads for fitting only.
            augmented = phase_result.get("augmented", False)
            augmented_path = phase_result.get("augmented_dataset_path", "")
            if augmented and augmented_path:
                logger.info(
                    "Augmented dataset available at %s; "
                    "model training will use augmented data for fitting "
                    "and original data (%s) for CV scoring.",
                    augmented_path,
                    original_dataset_path,
                )

        # Training and ensemble use the enhanced dataset (if available).
        # Evaluation uses original — switched below before Phase 6.

        # --- Phase 4: Model Training ---
        phase_result = self._run_phase(
            phase_number=4,
            phase_name="model_training",
            agent_class=ModelTrainingAgent,
            input_data=pipeline_input,
            state=state,
        )
        if phase_result is None:
            state = self._finalize_state(state, "failed")
            self._complete_pipeline_experiment(
                pipeline_experiment_id, state, pipeline_start, "FAILED",
            )
            return state
        state.training_report = ModelTrainingReport(**phase_result)
        pipeline_input["training_report"] = phase_result

        # --- Phase 5: Ensemble ---
        phase_result = self._run_phase(
            phase_number=5,
            phase_name="ensemble",
            agent_class=EnsembleAgent,
            input_data=pipeline_input,
            state=state,
        )
        if phase_result is None:
            state = self._finalize_state(state, "failed")
            self._complete_pipeline_experiment(
                pipeline_experiment_id, state, pipeline_start, "FAILED",
            )
            return state
        state.ensemble_report = EnsembleReport(**phase_result)
        pipeline_input["ensemble_report"] = phase_result

        # --- Phase 6: Evaluation ---
        # Evaluation uses the enhanced feature dataset (same original rows,
        # but with engineered features). We only revert to original_dataset_path
        # if no enhanced dataset exists. Note: "leakage prevention" means we
        # don't evaluate on augmented/synthetic rows — the enhanced_features
        # parquet has the same rows as the original with more columns, so it's safe.
        if enhanced_path:
            pipeline_input["dataset_path"] = enhanced_path
        else:
            pipeline_input["dataset_path"] = original_dataset_path
        phase_result = self._run_phase(
            phase_number=6,
            phase_name="evaluation",
            agent_class=EvaluationAgent,
            input_data=pipeline_input,
            state=state,
        )
        if phase_result is None:
            state = self._finalize_state(state, "failed")
            self._complete_pipeline_experiment(
                pipeline_experiment_id, state, pipeline_start, "FAILED",
            )
            return state
        state.evaluation_report = EvaluationReport(**phase_result)
        pipeline_input["evaluation_report"] = phase_result

        # --- Phase 7: Deployment ---
        phase_result = self._run_phase(
            phase_number=7,
            phase_name="deployment",
            agent_class=DeploymentAgent,
            input_data=pipeline_input,
            state=state,
        )
        if phase_result is None:
            state = self._finalize_state(state, "failed")
            self._complete_pipeline_experiment(
                pipeline_experiment_id, state, pipeline_start, "FAILED",
            )
            return state
        state.deployment_package = DeploymentPackage(**phase_result)

        # All phases completed successfully
        state = self._finalize_state(state, "completed")

        total_duration = time.perf_counter() - pipeline_start
        logger.info(
            "DS pipeline completed successfully in %.2f seconds. "
            "All %d phases passed.",
            total_duration,
            TOTAL_PHASES,
        )

        self._complete_pipeline_experiment(
            pipeline_experiment_id, state, pipeline_start, "COMPLETED",
        )

        return state

    def _run_phase(
        self,
        phase_number: int,
        phase_name: str,
        agent_class: type,
        input_data: dict[str, Any],
        state: DSPipelineState,
    ) -> dict[str, Any] | None:
        """Run a single pipeline phase: instantiate agent, execute, quality gate.

        Args:
            phase_number: Phase index (1-7, with 3 reused for Phase 3.5).
            phase_name: Human-readable phase name for logging.
            agent_class: The agent class to instantiate.
            input_data: Current pipeline input dict.
            state: Current pipeline state (for phase tracking).

        Returns:
            Phase result dict (from AgentResult.data) on success,
            or None on failure.
        """
        logger.info(
            "=== Phase %d: %s -- starting ===",
            phase_number,
            phase_name,
        )

        phase_start = time.perf_counter()

        try:
            # Instantiate the agent with shared dependencies
            agent = agent_class(**self._agent_kwargs)

            # Execute the agent
            result: AgentResult = agent.process(input_data)

            phase_duration = time.perf_counter() - phase_start

            if result.status != "success":
                logger.error(
                    "Phase %d (%s) returned non-success status: %s. Error: %s",
                    phase_number,
                    phase_name,
                    result.status,
                    result.error,
                )
                return None

            if not result.data:
                logger.error(
                    "Phase %d (%s) returned empty data dict.",
                    phase_number,
                    phase_name,
                )
                return None

            logger.info(
                "Phase %d (%s) completed in %.2f seconds with status=%s",
                phase_number,
                phase_name,
                phase_duration,
                result.status,
            )

            # Run LLM quality gate
            gate_passed = self._quality_gate(
                phase_number=phase_number,
                phase_name=phase_name,
                phase_output=result.data,
            )

            if not gate_passed:
                logger.warning(
                    "Phase %d (%s) quality gate returned FAIL. "
                    "Continuing pipeline (non-blocking gate).",
                    phase_number,
                    phase_name,
                )
                # Quality gate failure is non-blocking: log and continue

            return result.data

        except Exception as e:
            phase_duration = time.perf_counter() - phase_start
            logger.error(
                "Phase %d (%s) raised an exception after %.2f seconds: %s\n%s",
                phase_number,
                phase_name,
                phase_duration,
                e,
                traceback.format_exc(),
            )
            return None

    def _quality_gate(
        self,
        phase_number: int,
        phase_name: str,
        phase_output: dict[str, Any],
    ) -> bool:
        """Run an LLM quality gate on a phase's output.

        Makes a real LLM call to review the phase output and determine
        if it passes quality standards. Returns True for pass, False for fail.

        On LLM call failure (network, timeout, etc.), logs the error and
        returns True (assumes pass to avoid blocking the pipeline on
        infrastructure issues).
        """
        try:
            models = self.model_router.get_model_chain("ds_evaluator")
        except Exception as e:
            logger.warning(
                "Could not resolve ds_evaluator model for quality gate: %s. "
                "Skipping quality gate for phase %d (%s).",
                e,
                phase_number,
                phase_name,
            )
            return True

        # Build a concise summary of the phase output to avoid sending
        # excessively large payloads to the LLM
        output_summary = self._summarize_phase_output(phase_output)

        # Add context for phases where intentional pass-through is valid
        context_note = ""
        if phase_name == "data_augmentation":
            context_note = (
                "\nNote: Data augmentation is OPTIONAL. If the dataset is not "
                "imbalanced (imbalance_ratio below threshold), returning "
                "augmented=False with strategy_used='none' is a valid PASS. "
                "Only FAIL if there's an actual error or corrupted output.\n"
            )

        prompt = (
            f"Phase {phase_number}: {phase_name}\n"
            f"{'=' * 40}\n"
            f"{context_note}"
            f"Output summary:\n{output_summary}\n\n"
            f"Does this phase output pass the quality gate? "
            f"Check that the output contains meaningful, non-empty results "
            f"OR is a valid intentional skip with clear explanation."
        )

        messages = [
            LLMMessage(role="system", content=QUALITY_GATE_SYSTEM_PROMPT),
            LLMMessage(role="user", content=prompt),
        ]

        try:
            response = self.llm_client.complete_with_fallback(
                messages=messages,
                models=models,
                max_tokens=512,
                temperature=0.2,
            )

            response_text = response.content.strip()
            first_line = response_text.split("\n")[0].strip().upper()

            if first_line.startswith("PASS"):
                logger.info(
                    "Quality gate PASSED for phase %d (%s): %s",
                    phase_number,
                    phase_name,
                    response_text.split("\n")[0],
                )
                return True
            elif first_line.startswith("FAIL"):
                logger.warning(
                    "Quality gate FAILED for phase %d (%s): %s",
                    phase_number,
                    phase_name,
                    response_text,
                )
                return False
            else:
                # Ambiguous response -- log and assume pass
                logger.warning(
                    "Quality gate returned ambiguous response for phase %d (%s): %s. "
                    "Assuming PASS.",
                    phase_number,
                    phase_name,
                    response_text[:200],
                )
                return True

        except Exception as e:
            logger.warning(
                "Quality gate LLM call failed for phase %d (%s): %s. "
                "Assuming PASS to avoid blocking pipeline.",
                phase_number,
                phase_name,
                e,
            )
            return True

    def _summarize_phase_output(
        self,
        phase_output: dict[str, Any],
        max_length: int = 3000,
    ) -> str:
        """Create a concise text summary of a phase's output dict.

        Truncates large values and nested structures to stay within
        token budget for the quality gate LLM call.
        """
        lines: list[str] = []

        for key, value in phase_output.items():
            if isinstance(value, str):
                # Truncate long strings
                display = value[:300] + "..." if len(value) > 300 else value
                lines.append(f"- {key}: {display}")
            elif isinstance(value, list):
                lines.append(f"- {key}: list with {len(value)} items")
                # Show first 2 items as preview
                for i, item in enumerate(value[:2]):
                    item_str = str(item)
                    if len(item_str) > 150:
                        item_str = item_str[:150] + "..."
                    lines.append(f"    [{i}]: {item_str}")
                if len(value) > 2:
                    lines.append(f"    ... and {len(value) - 2} more")
            elif isinstance(value, dict):
                lines.append(f"- {key}: dict with {len(value)} keys")
                # Show first 3 keys
                for i, (k, v) in enumerate(value.items()):
                    if i >= 3:
                        lines.append(
                            f"    ... and {len(value) - 3} more keys"
                        )
                        break
                    v_str = str(v)
                    if len(v_str) > 100:
                        v_str = v_str[:100] + "..."
                    lines.append(f"    {k}: {v_str}")
            elif value is None:
                lines.append(f"- {key}: None")
            else:
                lines.append(f"- {key}: {value}")

        summary = "\n".join(lines)

        # Final truncation guard
        if len(summary) > max_length:
            summary = summary[:max_length] + "\n... (truncated)"

        return summary

    def _save_pipeline_experiment(
        self,
        project_id: uuid.UUID,
        config: dict[str, Any],
        task_id: Any = None,
        run_id: Any = None,
    ) -> uuid.UUID | None:
        """Save a pipeline-level experiment record.

        Returns the experiment ID, or None if saving fails.
        """
        try:
            experiment_id = self.repository.create_ds_experiment(
                project_id=project_id,
                experiment_phase="pipeline",
                experiment_config=config,
                task_id=task_id,
                run_id=run_id,
            )
            logger.info(
                "Created pipeline experiment record: %s", experiment_id,
            )
            return experiment_id
        except Exception as e:
            logger.warning(
                "Failed to save pipeline experiment record: %s. "
                "Pipeline will continue without experiment tracking.",
                e,
            )
            return None

    def _complete_pipeline_experiment(
        self,
        experiment_id: uuid.UUID | None,
        state: DSPipelineState,
        start_time: float,
        status: str,
    ) -> None:
        """Mark the pipeline experiment as completed with summary metrics."""
        if experiment_id is None:
            return

        total_duration = time.perf_counter() - start_time

        # Count how many phases completed (including 2.5 and 3.5 = 9 total)
        completed_phases = sum(
            1 for report in [
                state.audit_report,
                state.eda_report,
                state.fe_assessment,
                state.feature_report,
                state.augmentation_report,
                state.training_report,
                state.ensemble_report,
                state.evaluation_report,
                state.deployment_package,
            ]
            if report is not None
        )

        metrics: dict[str, Any] = {
            "completed_phases": completed_phases,
            "total_phases": TOTAL_PHASES,
            "total_duration_seconds": round(total_duration, 2),
            "problem_type": state.problem_type,
        }

        # Add evaluation grade if available
        if state.evaluation_report is not None:
            metrics["overall_grade"] = state.evaluation_report.overall_grade

        # Add FE assessment info if available
        if state.fe_assessment is not None:
            metrics["fe_potential"] = state.fe_assessment.overall_fe_potential
            metrics["fe_categories_applicable"] = sum(
                1 for c in state.fe_assessment.category_assessments if c.applicable
            )
            metrics["fe_features_proposed"] = len(state.fe_assessment.proposed_features)

        # Add augmentation info if available
        if state.augmentation_report is not None:
            metrics["augmented"] = state.augmentation_report.augmented
            metrics["augmentation_strategy"] = (
                state.augmentation_report.strategy_used
            )

        try:
            self.repository.update_ds_experiment(
                experiment_id=experiment_id,
                status=status,
                metrics=metrics,
            )
            logger.info(
                "Pipeline experiment %s completed: status=%s, phases=%d/%d, "
                "duration=%.2fs",
                experiment_id,
                status,
                completed_phases,
                TOTAL_PHASES,
                total_duration,
            )
        except Exception as e:
            logger.warning(
                "Failed to update pipeline experiment record %s: %s",
                experiment_id,
                e,
            )

    @staticmethod
    def _finalize_state(
        state: DSPipelineState,
        status: str,
    ) -> DSPipelineState:
        """Set the final status on the pipeline state.

        Since DSPipelineState is a Pydantic model, we return a copy
        with updated fields rather than mutating in place (though
        Pydantic v2 models are mutable by default).
        """
        # Count completed phases by checking all report slots
        completed = sum(
            1 for report in [
                state.audit_report,
                state.eda_report,
                state.fe_assessment,
                state.feature_report,
                state.augmentation_report,
                state.training_report,
                state.ensemble_report,
                state.evaluation_report,
                state.deployment_package,
            ]
            if report is not None
        )

        # DSPipelineState does not have current_phase/status fields
        # in the model definition, but we log them for observability
        logger.info(
            "Pipeline finalized: status=%s, phases_completed=%d/%d",
            status,
            completed,
            TOTAL_PHASES,
        )

        return state
