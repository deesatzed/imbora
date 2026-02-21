"""Scale-adaptive model routing.

Determines the dataset scale tier (small/medium/large) and selects
appropriate model candidates. Smaller datasets get TabPFN/TabICL
(tabular foundation models); larger datasets use gradient boosters.
When imbalance is severe, imbens specialists are included.
"""

from __future__ import annotations

import logging
from typing import Any

from src.core.config import DataScienceConfig

logger = logging.getLogger("associate.datasci.scale_router")

# Model candidate registry with metadata
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "tabpfn": {
        "type": "tabpfn",
        "max_rows": 10_000,
        "max_features": 100,
        "supports_regression": True,
        "supports_classification": True,
        "interpretable": False,
        "training_speed": "fast",
        "description": "TabPFN v2 — zero-shot tabular foundation model",
    },
    "tabicl": {
        "type": "tabicl",
        "max_rows": 600_000,
        "max_features": 100,
        "supports_regression": True,
        "supports_classification": True,
        "interpretable": False,
        "training_speed": "fast",
        "description": "TabICL v2 — in-context learning for tabular data",
    },
    "xgboost": {
        "type": "xgboost",
        "max_rows": None,
        "max_features": None,
        "supports_regression": True,
        "supports_classification": True,
        "interpretable": False,
        "training_speed": "medium",
        "description": "XGBoost — gradient boosted trees",
    },
    "catboost": {
        "type": "catboost",
        "max_rows": None,
        "max_features": None,
        "supports_regression": True,
        "supports_classification": True,
        "interpretable": False,
        "training_speed": "medium",
        "description": "CatBoost — gradient boosted trees with native categorical support",
    },
    "lightgbm": {
        "type": "lightgbm",
        "max_rows": None,
        "max_features": None,
        "supports_regression": True,
        "supports_classification": True,
        "interpretable": False,
        "training_speed": "fast",
        "description": "LightGBM — gradient boosted trees with histogram binning",
    },
    "nam": {
        "type": "nam",
        "max_rows": None,
        "max_features": 100,
        "supports_regression": True,
        "supports_classification": True,
        "interpretable": True,
        "training_speed": "slow",
        "description": "Neural Additive Model (dnamite) — per-feature shape functions",
    },
    "self_paced_ensemble": {
        "type": "self_paced_ensemble",
        "max_rows": None,
        "max_features": None,
        "supports_regression": False,
        "supports_classification": True,
        "interpretable": False,
        "training_speed": "medium",
        "imbalance_specialist": True,
        "description": "SelfPacedEnsemble — SOTA undersampling ensemble",
    },
    "easy_ensemble": {
        "type": "easy_ensemble",
        "max_rows": None,
        "max_features": None,
        "supports_regression": False,
        "supports_classification": True,
        "interpretable": False,
        "training_speed": "medium",
        "imbalance_specialist": True,
        "description": "EasyEnsemble — adaptive boosting with balanced subsets",
    },
}


def determine_scale_tier(
    row_count: int,
    config: DataScienceConfig,
) -> str:
    """Determine the dataset scale tier.

    Args:
        row_count: Number of rows in the dataset.
        config: DS config with threshold settings.

    Returns:
        One of 'small', 'medium', 'large'.
    """
    if row_count <= config.small_dataset_threshold:
        return "small"
    elif row_count <= config.medium_dataset_threshold:
        return "medium"
    else:
        return "large"


def select_model_candidates(
    scale_tier: str,
    problem_type: str,
    feature_count: int,
    config: DataScienceConfig,
    imbalance_ratio: float = 1.0,
) -> list[str]:
    """Select model candidates appropriate for the scale tier.

    Args:
        scale_tier: 'small', 'medium', or 'large'.
        problem_type: 'classification' or 'regression'.
        feature_count: Number of features in the dataset.
        config: DS config with feature flags.
        imbalance_ratio: Ratio of majority to minority class.
            When exceeding config.imbalance_threshold and problem_type
            is classification, imbens specialist models are included.

    Returns:
        List of model names to train.
    """
    candidates: list[str] = []

    # Determine imbalance threshold from config (default 3.0)
    imbalance_threshold = getattr(config, "imbalance_threshold", 3.0)

    for name, meta in MODEL_REGISTRY.items():
        # Skip imbalance specialists — handled separately below
        if meta.get("imbalance_specialist", False):
            continue

        # Check problem type support
        if problem_type == "classification" and not meta["supports_classification"]:
            continue
        if problem_type == "regression" and not meta["supports_regression"]:
            continue

        # Check feature flag
        if name == "tabpfn" and not config.enable_tabpfn:
            continue
        if name == "tabicl" and not config.enable_tabicl:
            continue
        if name == "nam" and not config.enable_nam:
            continue

        # Check feature count limits
        if meta["max_features"] is not None and feature_count > meta["max_features"]:
            continue

        # Check scale-tier compatibility
        if scale_tier == "small":
            # Small: all models are candidates
            candidates.append(name)
        elif scale_tier == "medium":
            # Medium: skip TabPFN (row limit 10K), keep everything else
            if meta["max_rows"] is not None and meta["max_rows"] < 50_000:
                continue
            candidates.append(name)
        elif scale_tier == "large":
            # Large: skip TabPFN and TabICL if over their limits, keep gradient boosters + NAM
            if name == "tabpfn":
                continue  # TabPFN hard-limited to ~10K
            if name == "tabicl":
                continue  # Skip for large by default; technically supports 600K but slow
            candidates.append(name)

    # Always include at least one gradient booster
    if not any(c in candidates for c in ("xgboost", "catboost", "lightgbm")):
        candidates.append("xgboost")

    # Include imbens imbalance specialists when imbalance is severe
    enable_imbens = getattr(config, "enable_imbens", True)
    if (
        enable_imbens
        and problem_type == "classification"
        and imbalance_ratio > imbalance_threshold
    ):
        candidates.append("self_paced_ensemble")
        candidates.append("easy_ensemble")
        logger.info(
            "Imbalance ratio %.2f exceeds threshold %.2f; "
            "adding imbens specialists: self_paced_ensemble, easy_ensemble",
            imbalance_ratio, imbalance_threshold,
        )

    return candidates


def get_model_metadata(model_name: str) -> dict[str, Any] | None:
    """Get metadata for a registered model."""
    return MODEL_REGISTRY.get(model_name)


def get_interpretable_models(candidates: list[str]) -> list[str]:
    """Filter candidates to only interpretable models."""
    return [c for c in candidates if MODEL_REGISTRY.get(c, {}).get("interpretable", False)]
