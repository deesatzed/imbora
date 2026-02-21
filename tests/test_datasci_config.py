"""Tests for data science config validation utilities."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from src.core.config import DataScienceConfig
from src.datasci.config import validate_dataset_path, validate_ds_config


class TestValidateDatasetPath:
    def test_valid_csv(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        result = validate_dataset_path(str(f))
        assert result == f
        assert isinstance(result, Path)

    def test_valid_tsv(self, tmp_path):
        f = tmp_path / "data.tsv"
        f.write_text("a\tb\n1\t2\n")
        result = validate_dataset_path(str(f))
        assert result == f

    def test_valid_parquet(self, tmp_path):
        f = tmp_path / "data.parquet"
        f.write_bytes(b"PAR1fake")  # Just needs to exist
        result = validate_dataset_path(str(f))
        assert result == f

    def test_valid_pq(self, tmp_path):
        f = tmp_path / "data.pq"
        f.write_bytes(b"PAR1fake")
        result = validate_dataset_path(str(f))
        assert result == f

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Dataset not found"):
            validate_dataset_path("/nonexistent/path/data.csv")

    def test_unsupported_format_json(self, tmp_path):
        f = tmp_path / "data.json"
        f.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported dataset format"):
            validate_dataset_path(str(f))

    def test_unsupported_format_xlsx(self, tmp_path):
        f = tmp_path / "data.xlsx"
        f.write_bytes(b"fake")
        with pytest.raises(ValueError, match="Unsupported dataset format"):
            validate_dataset_path(str(f))

    def test_unsupported_format_txt(self, tmp_path):
        f = tmp_path / "data.txt"
        f.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported dataset format"):
            validate_dataset_path(str(f))


class TestValidateDSConfig:
    def test_valid_defaults(self):
        config = DataScienceConfig()
        warnings = validate_ds_config(config)
        assert warnings == []

    def test_cv_folds_too_low(self):
        config = DataScienceConfig(cv_folds=1)
        warnings = validate_ds_config(config)
        assert any("cv_folds" in w for w in warnings)

    def test_cv_folds_zero(self):
        config = DataScienceConfig(cv_folds=0)
        warnings = validate_ds_config(config)
        assert any("cv_folds" in w for w in warnings)

    def test_conformal_alpha_zero(self):
        config = DataScienceConfig(conformal_alpha=0.0)
        warnings = validate_ds_config(config)
        assert any("conformal_alpha" in w for w in warnings)

    def test_conformal_alpha_one(self):
        config = DataScienceConfig(conformal_alpha=1.0)
        warnings = validate_ds_config(config)
        assert any("conformal_alpha" in w for w in warnings)

    def test_conformal_alpha_negative(self):
        config = DataScienceConfig(conformal_alpha=-0.1)
        warnings = validate_ds_config(config)
        assert any("conformal_alpha" in w for w in warnings)

    def test_llm_fe_generations_zero(self):
        config = DataScienceConfig(llm_fe_generations=0)
        warnings = validate_ds_config(config)
        assert any("llm_fe_generations" in w for w in warnings)

    def test_multiple_warnings(self):
        config = DataScienceConfig(cv_folds=1, conformal_alpha=0.0, llm_fe_generations=0)
        warnings = validate_ds_config(config)
        assert len(warnings) == 3

    def test_valid_custom_config(self):
        config = DataScienceConfig(
            cv_folds=10,
            conformal_alpha=0.05,
            llm_fe_generations=10,
        )
        warnings = validate_ds_config(config)
        assert warnings == []
