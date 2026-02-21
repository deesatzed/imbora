"""Phase 3.5: Data Augmentation Agent.

Applies synthetic minority generation for imbalanced classification datasets.
Sits between Feature Engineering (Phase 3) and Model Training (Phase 4).

Strategy selection:
- Not imbalanced or regression -> pass-through
- imbalance_ratio >= threshold, numeric only -> SMOTE
- imbalance_ratio >= threshold, has categoricals -> SMOTENC
- imbalance_ratio >= 10.0, has text -> LLM synth (EPIC-style)
- imbalance_ratio >= 10.0, no text -> ADASYN
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.config import DataScienceConfig
from src.core.models import AgentResult
from src.datasci.agents.base_ds_agent import BaseDataScienceAgent
from src.datasci.models import AugmentationReport

logger = logging.getLogger("associate.datasci.data_augmentation")

AUGMENTATION_SYSTEM_PROMPT = (
    "You are an expert data scientist reviewing data augmentation results.\n"
    "Given original and augmented class distributions plus adversarial validation\n"
    "scores, assess the quality of synthetic data generation.\n"
    "Focus on: balance achieved, quality indicators, and potential risks.\n"
    "Be concise."
)


def select_augmentation_strategy(
    imbalance_ratio: float,
    has_text_columns: bool,
    has_categorical_columns: bool,
    config: DataScienceConfig,
) -> str:
    """Select appropriate augmentation strategy based on data characteristics.

    Args:
        imbalance_ratio: Ratio of majority to minority class count.
        has_text_columns: Whether dataset has text columns.
        has_categorical_columns: Whether dataset has categorical columns.
        config: DataScience config with thresholds.

    Returns:
        Strategy string: 'none', 'smote', 'smotenc', 'adasyn', 'llm_synth'.
    """
    if not config.enable_augmentation:
        return "none"

    if config.augmentation_strategy != "auto":
        return config.augmentation_strategy

    if imbalance_ratio < config.imbalance_threshold:
        return "none"

    if imbalance_ratio >= 10.0 and has_text_columns:
        return "llm_synth"

    if imbalance_ratio >= 10.0:
        return "adasyn"

    if has_categorical_columns:
        return "smotenc"

    return "smote"


class DataAugmentationAgent(BaseDataScienceAgent):
    """Agent for Phase 3.5: Data Augmentation.

    Reads imbalance signals from EDA report, selects an augmentation
    strategy, generates synthetic minority samples, and saves the
    augmented dataset for downstream training.
    """

    def __init__(
        self,
        llm_client: Any,
        model_router: Any,
        repository: Any,
        ds_config: DataScienceConfig,
    ):
        super().__init__(
            name="DataAugmentationAgent",
            role="ds_analyst",
            llm_client=llm_client,
            model_router=model_router,
            repository=repository,
            ds_config=ds_config,
        )

    def process(self, input_data: dict[str, Any]) -> AgentResult:
        """Run the data augmentation pipeline.

        Args:
            input_data: Dict with keys:
                - dataset_path: str
                - target_column: str
                - project_id: UUID
                - problem_type: str ('classification' or 'regression')
                - eda_report: dict (EDAReport)
                - audit_report: dict (DataAuditReport)
                - task_id: UUID (optional)
                - run_id: UUID (optional)

        Returns:
            AgentResult with AugmentationReport in data.
        """
        dataset_path = input_data["dataset_path"]
        target_column = input_data["target_column"]
        project_id = input_data["project_id"]
        problem_type = input_data.get("problem_type", "classification")
        eda_data = input_data.get("eda_report", {})
        audit_data = input_data.get("audit_report", {})

        experiment_id = self._save_experiment(
            project_id=project_id,
            phase="data_augmentation",
            config={
                "target_column": target_column,
                "problem_type": problem_type,
            },
            task_id=input_data.get("task_id"),
            run_id=input_data.get("run_id"),
        )

        try:
            # Regression doesn't need augmentation
            if problem_type != "classification":
                report = AugmentationReport(augmented=False, strategy_used="none")
                self._complete_experiment(
                    experiment_id,
                    metrics={"augmented": False, "reason": "regression"},
                )
                return AgentResult(
                    agent_name=self.name,
                    status="success",
                    data=report.model_dump(),
                )

            # Get imbalance signal from EDA report
            imbalance_ratio = eda_data.get("imbalance_ratio", 1.0)
            is_imbalanced = eda_data.get("is_imbalanced", False)

            if not is_imbalanced or imbalance_ratio is None:
                ratio_str = f"{imbalance_ratio:.2f}" if imbalance_ratio is not None else "N/A"
                report = AugmentationReport(
                    augmented=False,
                    strategy_used="none",
                    llm_quality_review=(
                        f"Augmentation skipped: imbalance_ratio={ratio_str} "
                        f"is below threshold. Dataset is balanced enough for "
                        f"class-weight-only mitigation."
                    ),
                )
                self._complete_experiment(
                    experiment_id,
                    metrics={"augmented": False, "reason": "not_imbalanced"},
                )
                return AgentResult(
                    agent_name=self.name,
                    status="success",
                    data=report.model_dump(),
                )

            # Detect column types from audit report
            profiles = audit_data.get("column_profiles", [])
            has_text = any(p.get("text_detected", False) for p in profiles)
            has_categorical = any(
                p.get("dtype") == "categorical"
                for p in profiles
                if p.get("name") != target_column
            )

            # Select strategy
            strategy = select_augmentation_strategy(
                imbalance_ratio=imbalance_ratio,
                has_text_columns=has_text,
                has_categorical_columns=has_categorical,
                config=self.ds_config,
            )

            if strategy == "none":
                report = AugmentationReport(augmented=False, strategy_used="none")
                self._complete_experiment(
                    experiment_id,
                    metrics={"augmented": False, "reason": "strategy_none"},
                )
                return AgentResult(
                    agent_name=self.name,
                    status="success",
                    data=report.model_dump(),
                )

            # Load dataset
            df = self._load_dataset(dataset_path)
            logger.info(
                "Loaded dataset for augmentation: %d rows, imbalance_ratio=%.2f",
                len(df), imbalance_ratio,
            )

            # Run augmentation
            from src.datasci.synth_generator import SynthGenerator

            generator = SynthGenerator(
                llm_client=self.llm_client,
                model_router=self.model_router,
            )

            result = generator.run(
                df=df,
                target_column=target_column,
                strategy=strategy,
                imbalance_ratio=imbalance_ratio,
                max_augmentation_ratio=self.ds_config.max_augmentation_ratio,
            )

            augmented = result["strategy_used"] != "none" and result["samples_generated"] > 0

            # Save augmented dataset if augmentation occurred
            augmented_path = ""
            if augmented:
                artifacts_dir = Path(self.ds_config.artifacts_dir)
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                augmented_path = str(
                    artifacts_dir / f"augmented_{Path(dataset_path).stem}.parquet"
                )
                result["augmented_df"].to_parquet(augmented_path, index=False)
                logger.info(
                    "Saved augmented dataset (%d rows) to %s",
                    len(result["augmented_df"]), augmented_path,
                )

            # LLM quality review
            llm_review = ""
            if augmented:
                llm_review = self._quality_review(
                    strategy=result["strategy_used"],
                    original_counts=result["original_counts"],
                    augmented_counts=result["augmented_counts"],
                    samples_generated=result["samples_generated"],
                    adv_score=result["adversarial_validation_score"],
                )

            report = AugmentationReport(
                augmented=augmented,
                strategy_used=result["strategy_used"],
                original_class_counts=result["original_counts"],
                augmented_class_counts=result["augmented_counts"],
                samples_generated=result["samples_generated"],
                augmented_dataset_path=augmented_path,
                adversarial_validation_score=result["adversarial_validation_score"],
                quality_score=result["quality_score"],
                llm_quality_review=llm_review,
            )

            self._complete_experiment(
                experiment_id,
                metrics={
                    "augmented": augmented,
                    "strategy": result["strategy_used"],
                    "samples_generated": result["samples_generated"],
                    "quality_score": result["quality_score"],
                },
            )

            return AgentResult(
                agent_name=self.name,
                status="success",
                data=report.model_dump(),
            )

        except Exception:
            self._complete_experiment(experiment_id, status="FAILED")
            raise

    def _load_dataset(self, path: str) -> pd.DataFrame:
        """Load dataset from supported formats."""
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(p)
        elif suffix == ".tsv":
            return pd.read_csv(p, sep="\t")
        elif suffix in (".parquet", ".pq"):
            return pd.read_parquet(p)
        else:
            raise ValueError(f"Unsupported dataset format: {suffix}")

    def _quality_review(
        self,
        strategy: str,
        original_counts: dict[str, int],
        augmented_counts: dict[str, int],
        samples_generated: int,
        adv_score: float,
    ) -> str:
        """Use LLM to review augmentation quality."""
        prompt = (
            f"Data Augmentation Review\n"
            f"========================\n"
            f"Strategy: {strategy}\n"
            f"Original class distribution: {original_counts}\n"
            f"Augmented class distribution: {augmented_counts}\n"
            f"Synthetic samples generated: {samples_generated}\n"
            f"Adversarial validation AUC: {adv_score:.4f}\n"
            f"(Lower adversarial AUC = better quality synthetic data)\n\n"
            f"Assess the quality and risks of this augmentation."
        )

        return self._call_ds_llm(
            prompt=prompt,
            role="ds_evaluator",
            system_prompt=AUGMENTATION_SYSTEM_PROMPT,
        )
