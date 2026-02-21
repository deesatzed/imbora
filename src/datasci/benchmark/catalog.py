"""Dataset registry for imbalanced classification benchmarks.

Contains ~27-30 curated datasets across four imbalance tiers
(mild, moderate, severe, extreme) sourced from sklearn, OpenML,
and imblearn. Each entry records provenance, approximate statistics,
and published SOTA metrics where available.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DatasetEntry:
    """A single benchmark dataset descriptor."""

    name: str
    source: str  # "sklearn", "openml", "imblearn"
    source_id: str | int  # OpenML ID, sklearn func, imblearn key
    target_column: str
    problem_type: str = "classification"
    imbalance_tier: str = "moderate"  # mild/moderate/severe/extreme
    approximate_ir: float = 1.0
    sota_auc: float | None = None
    sota_f1: float | None = None
    n_rows: int = 0
    n_features: int = 0
    notes: str = ""


# ── Mild (IR 1.5-3:1) ──────────────────────────────────────────

_MILD: dict[str, DatasetEntry] = {
    "breast_cancer_wisconsin": DatasetEntry(
        name="Breast Cancer Wisconsin",
        source="sklearn",
        source_id="breast_cancer",
        target_column="target",
        imbalance_tier="mild",
        approximate_ir=1.7,
        sota_auc=0.995,
        sota_f1=0.98,
        n_rows=569,
        n_features=30,
    ),
    "german_credit": DatasetEntry(
        name="German Credit",
        source="openml",
        source_id=31,
        target_column="class",
        imbalance_tier="mild",
        approximate_ir=2.3,
        sota_auc=0.81,
        n_rows=1000,
        n_features=20,
    ),
    "pima_diabetes": DatasetEntry(
        name="Pima Indians Diabetes",
        source="openml",
        source_id=37,
        target_column="class",
        imbalance_tier="mild",
        approximate_ir=1.9,
        sota_auc=0.84,
        n_rows=768,
        n_features=8,
    ),
    "haberman_survival": DatasetEntry(
        name="Haberman Survival",
        source="openml",
        source_id=43,
        target_column="Survival",
        imbalance_tier="mild",
        approximate_ir=2.8,
        sota_auc=0.72,
        n_rows=306,
        n_features=3,
    ),
    "ionosphere": DatasetEntry(
        name="Ionosphere",
        source="openml",
        source_id=59,
        target_column="class",
        imbalance_tier="mild",
        approximate_ir=1.8,
        sota_auc=0.98,
        n_rows=351,
        n_features=34,
    ),
}

# ── Moderate (IR 3-9:1) ────────────────────────────────────────

_MODERATE: dict[str, DatasetEntry] = {
    "yeast_me2": DatasetEntry(
        name="Yeast ME2",
        source="imblearn",
        source_id="yeast_me2",
        target_column="target",
        imbalance_tier="moderate",
        approximate_ir=5.2,
        sota_f1=0.48,
        n_rows=1484,
        n_features=8,
    ),
    "glass_7": DatasetEntry(
        name="Glass 7 (vehicle windows)",
        source="imblearn",
        source_id="glass2",
        target_column="target",
        imbalance_tier="moderate",
        approximate_ir=6.4,
        sota_auc=0.94,
        n_rows=214,
        n_features=9,
        notes="imblearn key is 'glass2'",
    ),
    "ecoli_imu": DatasetEntry(
        name="E. coli IMU",
        source="imblearn",
        source_id="ecoli4",
        target_column="target",
        imbalance_tier="moderate",
        approximate_ir=8.6,
        sota_f1=0.76,
        n_rows=336,
        n_features=7,
        notes="imblearn key is 'ecoli4'",
    ),
    "thyroid_sick": DatasetEntry(
        name="Thyroid Sick",
        source="openml",
        source_id=38,
        target_column="Class",
        imbalance_tier="moderate",
        approximate_ir=7.4,
        sota_auc=0.99,
        n_rows=3772,
        n_features=29,
    ),
    "car_eval_good": DatasetEntry(
        name="Car Evaluation (good)",
        source="openml",
        source_id=40975,
        target_column="class",
        imbalance_tier="moderate",
        approximate_ir=4.0,
        sota_auc=0.99,
        n_rows=1728,
        n_features=6,
    ),
    "vehicle_insurance": DatasetEntry(
        name="Vehicle Insurance",
        source="openml",
        source_id=1067,
        target_column="Class",
        imbalance_tier="moderate",
        approximate_ir=6.0,
        sota_auc=0.93,
        n_rows=846,
        n_features=18,
        notes="Updated SOTA for binary classification setting",
    ),
    "wine_quality_red": DatasetEntry(
        name="Wine Quality (Red)",
        source="openml",
        source_id=40691,
        target_column="quality",
        imbalance_tier="moderate",
        approximate_ir=5.0,
        sota_auc=0.95,
        n_rows=1599,
        n_features=11,
        notes="Updated SOTA for binary classification (3-vs-rest), not multi-class",
    ),
}

# ── Severe (IR 10-50:1) ────────────────────────────────────────

_SEVERE: dict[str, DatasetEntry] = {
    "abalone_19": DatasetEntry(
        name="Abalone 19",
        source="imblearn",
        source_id="abalone_19",
        target_column="target",
        imbalance_tier="severe",
        approximate_ir=16.4,
        sota_f1=0.12,
        n_rows=4177,
        n_features=8,
        notes="Hardest imbalanced benchmark",
    ),
    "satimage": DatasetEntry(
        name="Satimage",
        source="openml",
        source_id=182,
        target_column="class",
        imbalance_tier="severe",
        approximate_ir=10.0,
        sota_auc=0.93,
        n_rows=6435,
        n_features=36,
    ),
    "oil_spill": DatasetEntry(
        name="Oil Spill",
        source="openml",
        source_id=1461,
        target_column="class",
        imbalance_tier="severe",
        approximate_ir=22.0,
        sota_auc=0.94,
        n_rows=937,
        n_features=49,
    ),
    "pc1_software": DatasetEntry(
        name="PC1 Software Defects",
        source="openml",
        source_id=1068,
        target_column="defects",
        imbalance_tier="severe",
        approximate_ir=14.0,
        sota_auc=0.81,
        n_rows=1109,
        n_features=21,
    ),
    "mammography": DatasetEntry(
        name="Mammography",
        source="openml",
        source_id=310,
        target_column="class",
        imbalance_tier="severe",
        approximate_ir=42.0,
        sota_auc=0.94,
        n_rows=11183,
        n_features=6,
    ),
    "phoneme": DatasetEntry(
        name="Phoneme",
        source="openml",
        source_id=1489,
        target_column="Class",
        imbalance_tier="severe",
        approximate_ir=2.4,
        sota_auc=0.96,
        n_rows=5404,
        n_features=5,
        notes=(
            "Actually mild IR but commonly "
            "benchmarked with severe group"
        ),
    ),
    "us_crime": DatasetEntry(
        name="US Crime",
        source="imblearn",
        source_id="us_crime",
        target_column="target",
        imbalance_tier="severe",
        approximate_ir=12.0,
        n_rows=1994,
        n_features=100,
    ),
    "letter_img": DatasetEntry(
        name="Letter Image Recognition",
        source="openml",
        source_id=6,
        target_column="class",
        imbalance_tier="severe",
        approximate_ir=26.0,
        sota_auc=0.99,
        n_rows=20000,
        n_features=16,
    ),
}

# ── Extreme (IR 50+:1) ─────────────────────────────────────────

_EXTREME: dict[str, DatasetEntry] = {
    "credit_card_fraud": DatasetEntry(
        name="Credit Card Fraud",
        source="openml",
        source_id=1597,
        target_column="Class",
        imbalance_tier="extreme",
        approximate_ir=577.0,
        sota_auc=0.98,
        n_rows=284807,
        n_features=30,
        notes="Very large, slow",
    ),
    "ozone_level": DatasetEntry(
        name="Ozone Level Detection",
        source="openml",
        source_id=1487,
        target_column="Class",
        imbalance_tier="extreme",
        approximate_ir=34.0,
        sota_auc=0.90,
        n_rows=2536,
        n_features=72,
    ),
    "solar_flare_m": DatasetEntry(
        name="Solar Flare M-class",
        source="imblearn",
        source_id="solar_flare_m",
        target_column="target",
        imbalance_tier="extreme",
        approximate_ir=19.0,
        sota_f1=0.30,
        n_rows=1066,
        n_features=12,
    ),
    "webpage": DatasetEntry(
        name="Webpage Classification",
        source="openml",
        source_id=350,
        target_column="class",
        imbalance_tier="extreme",
        approximate_ir=33.0,
        sota_auc=0.98,
        n_rows=34780,
        n_features=300,
    ),
    "coil_2000": DatasetEntry(
        name="COIL 2000 Insurance",
        source="openml",
        source_id=955,
        target_column="class",
        imbalance_tier="extreme",
        approximate_ir=16.0,
        sota_auc=0.78,
        n_rows=9822,
        n_features=85,
    ),
}

# ── Assembled catalog ──────────────────────────────────────────

BENCHMARK_CATALOG: dict[str, DatasetEntry] = {
    **_MILD,
    **_MODERATE,
    **_SEVERE,
    **_EXTREME,
}


# ── Helper functions ───────────────────────────────────────────


def get_catalog() -> dict[str, DatasetEntry]:
    """Return the full benchmark catalog."""
    return dict(BENCHMARK_CATALOG)


def get_by_tier(tier: str) -> list[DatasetEntry]:
    """Return datasets in a specific imbalance tier.

    Args:
        tier: One of "mild", "moderate", "severe", "extreme".

    Returns:
        List of matching DatasetEntry objects.
    """
    return [
        e
        for e in BENCHMARK_CATALOG.values()
        if e.imbalance_tier == tier
    ]


def get_by_source(source: str) -> list[DatasetEntry]:
    """Return datasets from a specific source.

    Args:
        source: One of "sklearn", "openml", "imblearn".

    Returns:
        List of matching DatasetEntry objects.
    """
    return [
        e
        for e in BENCHMARK_CATALOG.values()
        if e.source == source
    ]
