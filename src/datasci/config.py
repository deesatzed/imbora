"""Data science configuration utilities."""

from __future__ import annotations

from pathlib import Path

from src.core.config import DataScienceConfig


def validate_dataset_path(path: str) -> Path:
    """Validate that a dataset path exists and is a supported format."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if p.suffix.lower() not in {".csv", ".parquet", ".pq", ".tsv"}:
        raise ValueError(
            f"Unsupported dataset format: {p.suffix}. "
            "Supported: .csv, .tsv, .parquet, .pq"
        )
    return p


def validate_ds_config(config: DataScienceConfig) -> list[str]:
    """Validate DS config and return list of warnings (empty = OK)."""
    warnings: list[str] = []
    if config.cv_folds < 2:
        warnings.append("cv_folds < 2: cross-validation requires at least 2 folds")
    if config.conformal_alpha <= 0 or config.conformal_alpha >= 1:
        warnings.append("conformal_alpha must be in (0, 1)")
    if config.llm_fe_generations < 1:
        warnings.append("llm_fe_generations must be >= 1")
    return warnings
