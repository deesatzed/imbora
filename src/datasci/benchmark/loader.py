"""Benchmark dataset loader — fetches and caches datasets as local CSV."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from src.datasci.benchmark.catalog import DatasetEntry

logger = logging.getLogger("associate.datasci.benchmark.loader")

_DEFAULT_CACHE_DIR = Path.home() / ".associate" / "benchmark_cache"


class BenchmarkLoader:
    """Fetches benchmark datasets and caches as local CSV files.

    Each dataset is downloaded once and cached at:
        <cache_dir>/<dataset_name>.csv

    Subsequent calls return the cached path without re-downloading.
    """

    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or _DEFAULT_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(self, entry: DatasetEntry) -> Path:
        """Fetch a dataset, cache as CSV, and return the file path.

        Dispatches to the appropriate loader based on entry.source.
        Returns the path to the cached CSV file.

        Raises:
            ValueError: If the source type is unknown.
            RuntimeError: If the dataset cannot be loaded.
        """
        cached = self.cache_dir / f"{entry.name}.csv"
        if cached.exists():
            logger.info("Cache hit for %s: %s", entry.name, cached)
            return cached

        logger.info(
            "Loading %s from %s (id=%s)...",
            entry.name, entry.source, entry.source_id,
        )

        if entry.source == "sklearn":
            df = self._load_sklearn(entry)
        elif entry.source == "openml":
            df = self._load_openml(entry)
        elif entry.source == "imblearn":
            df = self._load_imblearn(entry)
        else:
            raise ValueError(
                f"Unknown source '{entry.source}' for {entry.name}"
            )

        df.to_csv(cached, index=False)
        logger.info(
            "Cached %s (%d rows, %d cols) at %s",
            entry.name, len(df), len(df.columns), cached,
        )
        return cached

    def _load_sklearn(self, entry: DatasetEntry) -> pd.DataFrame:
        """Load a dataset from sklearn.datasets."""
        from sklearn.datasets import load_breast_cancer

        loaders = {
            "breast_cancer": load_breast_cancer,
        }

        loader_fn = loaders.get(str(entry.source_id))
        if loader_fn is None:
            raise RuntimeError(
                f"No sklearn loader for source_id={entry.source_id}"
            )

        bunch = loader_fn()
        df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
        df[entry.target_column] = bunch.target
        return df

    def _load_openml(self, entry: DatasetEntry) -> pd.DataFrame:
        """Load a dataset from OpenML by dataset ID."""
        try:
            from sklearn.datasets import fetch_openml
        except ImportError as exc:
            raise RuntimeError(
                "sklearn required for OpenML loading"
            ) from exc

        dataset = fetch_openml(
            data_id=int(entry.source_id),
            as_frame=True,
            parser="auto",
        )
        df = dataset.frame
        if df is None:
            raise RuntimeError(
                f"OpenML returned no frame for id={entry.source_id}"
            )

        # Ensure target column exists
        if entry.target_column not in df.columns:
            # OpenML sometimes puts target in dataset.target
            if dataset.target is not None:
                df[entry.target_column] = dataset.target

        # Convert categorical columns to numeric codes
        cat_cols = df.select_dtypes(include=["category"]).columns
        for col in cat_cols:
            df[col] = df[col].cat.codes.replace(-1, pd.NA)

        # Convert object columns to numeric where possible
        obj_cols = df.select_dtypes(include=["object"]).columns
        for col in obj_cols:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                # Label-encode string columns
                codes = df[col].astype("category").cat.codes
                df[col] = codes.replace(-1, pd.NA)

        return df

    def _load_imblearn(self, entry: DatasetEntry) -> pd.DataFrame:
        """Load a dataset from imblearn.datasets."""
        try:
            from imblearn.datasets import fetch_datasets
        except ImportError as exc:
            raise RuntimeError(
                "imbalanced-learn required for imblearn loading"
            ) from exc

        all_datasets = fetch_datasets(
            filter_data=(str(entry.source_id),),
        )
        key = str(entry.source_id)
        if key not in all_datasets:
            raise RuntimeError(
                f"imblearn dataset '{key}' not found. "
                f"Available: {list(all_datasets.keys())}"
            )

        bunch = all_datasets[key]
        n_features = bunch.data.shape[1]
        feature_names = [
            f"feature_{i}" for i in range(n_features)
        ]
        df = pd.DataFrame(bunch.data, columns=feature_names)
        df[entry.target_column] = bunch.target
        return df

    def is_cached(self, entry: DatasetEntry) -> bool:
        """Check if a dataset is already cached."""
        return (self.cache_dir / f"{entry.name}.csv").exists()

    def cache_status(self) -> dict[str, bool]:
        """Return cache status for all CSV files in cache dir."""
        result = {}
        if self.cache_dir.exists():
            for csv_file in self.cache_dir.glob("*.csv"):
                result[csv_file.stem] = True
        return result

    def clear_cache(self) -> None:
        """Remove all cached datasets."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "Cleared benchmark cache at %s", self.cache_dir
            )
