"""Artifact management utilities for the data science pipeline.

Provides model serialization, API scaffold generation, monitoring
configuration, and data contract generation for deployment packaging.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("associate.datasci.artifacts")


class ArtifactManager:
    """Manages model artifacts, API scaffolds, and deployment configs.

    Handles serialization/deserialization of trained models, generates
    FastAPI scaffold code for serving predictions, produces monitoring
    configurations with drift thresholds, and generates data contracts
    defining input schemas.
    """

    def save_model(self, model: Any, path: str) -> str:
        """Serialize a trained model to disk using joblib.

        Args:
            model: A fitted sklearn-compatible estimator.
            path: Absolute or relative path to write the model file.

        Returns:
            The resolved path where the model was saved.

        Raises:
            ImportError: If joblib is not installed.
            OSError: If the target directory cannot be created.
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                "joblib is required for model serialization. "
                "Install it with: pip install joblib"
            ) from exc

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, target)
        logger.info("Model saved to %s", target)
        return str(target.resolve())

    def load_model(self, path: str) -> Any:
        """Deserialize a trained model from disk using joblib.

        Args:
            path: Path to the serialized model file.

        Returns:
            The deserialized model object.

        Raises:
            ImportError: If joblib is not installed.
            FileNotFoundError: If the model file does not exist.
        """
        try:
            import joblib
        except ImportError as exc:
            raise ImportError(
                "joblib is required for model deserialization. "
                "Install it with: pip install joblib"
            ) from exc

        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        model = joblib.load(target)
        logger.info("Model loaded from %s", target)
        return model

    def generate_api_scaffold(
        self,
        model_name: str,
        feature_names: list[str],
        target_column: str,
    ) -> str:
        """Generate a FastAPI scaffold for serving model predictions.

        Produces a complete, runnable FastAPI application as a Python
        source string. The scaffold includes input validation via
        Pydantic, model loading via joblib, and a /predict endpoint.

        Args:
            model_name: Human-readable name for the model.
            feature_names: List of input feature column names.
            target_column: Name of the prediction target.

        Returns:
            Python source code string for the FastAPI application.
        """
        # Build the Pydantic field definitions
        feature_fields = "\n".join(
            f"    {name}: float" for name in feature_names
        )

        # Build the feature extraction list
        feature_extraction = ", ".join(
            f"request.{name}" for name in feature_names
        )

        scaffold = f'''"""Auto-generated FastAPI prediction service for {model_name}.

Serves predictions for target: {target_column}
Input features: {len(feature_names)} numeric columns

Usage:
    uvicorn app:app --host 0.0.0.0 --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(
    title="{model_name} Prediction Service",
    description="Serves predictions for {target_column}",
    version="1.0.0",
)

# Load model at startup
MODEL_PATH = "model.joblib"
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    model = None


class PredictionRequest(BaseModel):
    """Input features for prediction."""
{feature_fields}


class PredictionResponse(BaseModel):
    """Prediction output."""
    prediction: float
    model_name: str = "{model_name}"
    target: str = "{target_column}"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Check service health and model availability."""
    return HealthResponse(
        status="healthy" if model is not None else "degraded",
        model_loaded=model is not None,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Generate a prediction from input features."""
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Ensure model.joblib is available.",
        )

    features = np.array([[{feature_extraction}]])

    try:
        prediction = float(model.predict(features)[0])
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {{str(e)}}",
        )

    return PredictionResponse(prediction=prediction)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
'''
        logger.info(
            "Generated API scaffold for %s with %d features",
            model_name, len(feature_names),
        )
        return scaffold

    def generate_monitoring_config(
        self,
        feature_stats: dict[str, dict[str, float]],
        training_metrics: dict[str, float],
    ) -> dict[str, Any]:
        """Generate a drift monitoring configuration.

        Produces a configuration dict with per-feature drift thresholds
        derived from training-time statistics, plus model performance
        thresholds based on training metrics.

        Args:
            feature_stats: Per-feature statistics from training data.
                Expected keys per feature: mean, std, min, max.
            training_metrics: Model performance metrics from training
                (e.g. accuracy, f1, r2, rmse).

        Returns:
            Dict with drift_thresholds, performance_thresholds,
            and monitoring settings.
        """
        drift_thresholds: dict[str, dict[str, float]] = {}
        for feature_name, stats in feature_stats.items():
            mean = stats.get("mean", 0.0)
            std = stats.get("std", 1.0)
            feat_min = stats.get("min", mean - 3 * std)
            feat_max = stats.get("max", mean + 3 * std)

            # Drift thresholds: alert if mean shifts by > 2 std
            # or if range expands by > 50%
            feature_range = feat_max - feat_min
            drift_thresholds[feature_name] = {
                "expected_mean": round(mean, 6),
                "expected_std": round(std, 6),
                "mean_drift_threshold": round(2.0 * std, 6),
                "range_min": round(feat_min, 6),
                "range_max": round(feat_max, 6),
                "range_expansion_threshold": round(
                    feature_range * 0.5, 6,
                ),
            }

        # Performance thresholds: alert if metric drops by > 10%
        performance_thresholds: dict[str, dict[str, float]] = {}
        for metric_name, value in training_metrics.items():
            performance_thresholds[metric_name] = {
                "training_value": round(value, 6),
                "alert_threshold": round(value * 0.9, 6),
                "critical_threshold": round(value * 0.8, 6),
            }

        config: dict[str, Any] = {
            "drift_thresholds": drift_thresholds,
            "performance_thresholds": performance_thresholds,
            "monitoring": {
                "check_interval_minutes": 60,
                "window_size_samples": 1000,
                "drift_test": "ks_test",
                "significance_level": 0.05,
                "alert_channels": ["log"],
            },
        }

        logger.info(
            "Generated monitoring config for %d features, "
            "%d performance metrics",
            len(drift_thresholds), len(performance_thresholds),
        )
        return config

    def generate_data_contract(
        self,
        feature_names: list[str],
        feature_types: dict[str, str],
    ) -> dict[str, Any]:
        """Generate a data contract defining the input schema.

        Produces a schema dict specifying required input features,
        their types, and validation rules for production inference.

        Args:
            feature_names: Ordered list of input feature names.
            feature_types: Mapping of feature name to type string
                (e.g. 'float64', 'int64', 'object').

        Returns:
            Dict with schema version, fields, and constraints.
        """
        fields: list[dict[str, Any]] = []
        for name in feature_names:
            dtype = feature_types.get(name, "float64")

            # Map pandas dtypes to contract types
            if "int" in dtype:
                contract_type = "integer"
            elif "float" in dtype:
                contract_type = "number"
            elif dtype in ("object", "string", "str"):
                contract_type = "string"
            elif "bool" in dtype:
                contract_type = "boolean"
            elif "datetime" in dtype:
                contract_type = "datetime"
            else:
                contract_type = "number"

            fields.append({
                "name": name,
                "type": contract_type,
                "required": True,
                "nullable": False,
                "source_dtype": dtype,
            })

        contract: dict[str, Any] = {
            "schema_version": "1.0",
            "description": "Input data contract for model inference",
            "fields": fields,
            "constraints": {
                "min_features_required": len(feature_names),
                "allow_extra_fields": True,
                "null_handling": "reject",
            },
        }

        logger.info(
            "Generated data contract with %d fields",
            len(fields),
        )
        return contract
