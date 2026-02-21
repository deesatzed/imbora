"""Benchmark system for running the DS pipeline across standard datasets.

Provides a catalog of imbalanced classification benchmark datasets,
a loader that fetches and caches them locally, a runner that executes
the pipeline across datasets, and a reporter that aggregates results.
"""

from src.datasci.benchmark.catalog import (
    BENCHMARK_CATALOG,
    DatasetEntry,
    get_by_source,
    get_by_tier,
    get_catalog,
)
from src.datasci.benchmark.loader import BenchmarkLoader
from src.datasci.benchmark.report import BenchmarkReport, BenchmarkReporter
from src.datasci.benchmark.runner import BenchmarkRunner, DatasetResult

__all__ = [
    "BENCHMARK_CATALOG",
    "BenchmarkLoader",
    "BenchmarkReport",
    "BenchmarkReporter",
    "BenchmarkRunner",
    "DatasetEntry",
    "DatasetResult",
    "get_by_source",
    "get_by_tier",
    "get_catalog",
]
