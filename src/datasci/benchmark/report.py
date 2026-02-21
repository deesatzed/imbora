"""Benchmark report — aggregates results and produces formatted output.

Compiles a list of DatasetResult objects into a BenchmarkReport
with tier grouping, average gaps, and best/worst identification.
Supports JSON serialization and CLI-friendly table rendering.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field

from src.datasci.benchmark.runner import DatasetResult

logger = logging.getLogger(
    "associate.datasci.benchmark.report"
)


@dataclass
class BenchmarkReport:
    """Aggregated benchmark results across all datasets."""

    results: list[DatasetResult]
    total_datasets: int = 0
    successful: int = 0
    failed: int = 0
    skipped: int = 0
    by_tier: dict[str, list[DatasetResult]] = field(
        default_factory=dict,
    )
    avg_auc_gap: float | None = None
    avg_f1_gap: float | None = None
    best_relative_performance: str = ""  # dataset name
    worst_relative_performance: str = ""  # dataset name


class BenchmarkReporter:
    """Compiles benchmark results into reports."""

    def compile(
        self,
        results: list[DatasetResult],
    ) -> BenchmarkReport:
        """Compile raw results into an aggregated report.

        Counts totals by status, groups by tier, computes
        average AUC/F1 gaps across successful datasets that
        have gap data, and identifies best/worst performers
        by auc_gap.

        Args:
            results: List of DatasetResult from a benchmark
                run.

        Returns:
            BenchmarkReport with all aggregated fields.
        """
        report = BenchmarkReport(results=list(results))
        report.total_datasets = len(results)

        # Count by status
        report.successful = sum(
            1 for r in results if r.status == "success"
        )
        report.failed = sum(
            1 for r in results if r.status == "failed"
        )
        report.skipped = sum(
            1 for r in results if r.status == "skipped"
        )

        # Group by tier
        by_tier: dict[str, list[DatasetResult]] = {}
        for r in results:
            tier = r.tier
            if tier not in by_tier:
                by_tier[tier] = []
            by_tier[tier].append(r)
        report.by_tier = by_tier

        # Compute average AUC gap (successful with data)
        auc_gaps = [
            r.auc_gap
            for r in results
            if r.status == "success" and r.auc_gap is not None
        ]
        if auc_gaps:
            report.avg_auc_gap = round(
                sum(auc_gaps) / len(auc_gaps), 6
            )

        # Compute average F1 gap (successful with data)
        f1_gaps = [
            r.f1_gap
            for r in results
            if r.status == "success" and r.f1_gap is not None
        ]
        if f1_gaps:
            report.avg_f1_gap = round(
                sum(f1_gaps) / len(f1_gaps), 6
            )

        # Best/worst relative performance by auc_gap
        results_with_auc_gap = [
            r
            for r in results
            if r.status == "success" and r.auc_gap is not None
        ]
        if results_with_auc_gap:
            best = max(
                results_with_auc_gap,
                key=lambda r: r.auc_gap,  # type: ignore[arg-type]
            )
            worst = min(
                results_with_auc_gap,
                key=lambda r: r.auc_gap,  # type: ignore[arg-type]
            )
            report.best_relative_performance = (
                best.dataset_name
            )
            report.worst_relative_performance = (
                worst.dataset_name
            )

        logger.info(
            "Compiled report: %d total, %d success, "
            "%d failed, %d skipped, avg_auc_gap=%s, "
            "avg_f1_gap=%s",
            report.total_datasets,
            report.successful,
            report.failed,
            report.skipped,
            report.avg_auc_gap,
            report.avg_f1_gap,
        )

        return report

    def to_json(self, report: BenchmarkReport) -> str:
        """Serialize report to JSON string.

        Uses dataclasses.asdict for full conversion, then
        json.dumps with indent for readability. None values
        are preserved as JSON null.

        Args:
            report: The BenchmarkReport to serialize.

        Returns:
            JSON string representation of the report.
        """
        data = asdict(report)
        return json.dumps(data, indent=2, default=str)

    def to_table(self, report: BenchmarkReport) -> str:
        """Format report as a CLI-friendly table.

        Produces a fixed-width table grouped by tier with
        columns for dataset, tier, status, AUC, SOTA AUC,
        gap, F1, SOTA F1, gap, phases, and elapsed time.
        Includes a summary footer section.

        Args:
            report: The BenchmarkReport to format.

        Returns:
            Multi-line string suitable for terminal output.
        """
        lines: list[str] = []

        # Column headers
        header = (
            f"{'Dataset':<28} "
            f"{'Tier':<10} "
            f"{'Status':<8} "
            f"{'AUC':>6} "
            f"{'SOTA':>6} "
            f"{'Gap':>7} "
            f"{'F1':>6} "
            f"{'SOTA':>6} "
            f"{'Gap':>7} "
            f"{'Phases':>6} "
            f"{'Time':>8}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        # Determine tier ordering for grouped output
        tier_order = [
            "mild", "moderate", "severe", "extreme",
        ]
        # Include any tiers not in the standard list
        seen_tiers = set(tier_order)
        for tier in report.by_tier:
            if tier not in seen_tiers:
                tier_order.append(tier)
                seen_tiers.add(tier)

        for tier in tier_order:
            tier_results = report.by_tier.get(tier, [])
            if not tier_results:
                continue

            # Tier header
            lines.append(
                f"  [{tier.upper()}] "
                f"({len(tier_results)} datasets)"
            )

            for r in tier_results:
                auc_str = self._fmt_metric(r.best_auc)
                sota_auc_str = self._fmt_metric(r.sota_auc)
                auc_gap_str = self._fmt_gap(r.auc_gap)
                f1_str = self._fmt_metric(r.best_f1)
                sota_f1_str = self._fmt_metric(r.sota_f1)
                f1_gap_str = self._fmt_gap(r.f1_gap)
                phases_str = str(r.phases_completed)
                time_str = self._fmt_time(r.elapsed_seconds)

                row = (
                    f"  {r.dataset_name:<26} "
                    f"{r.tier:<10} "
                    f"{r.status:<8} "
                    f"{auc_str:>6} "
                    f"{sota_auc_str:>6} "
                    f"{auc_gap_str:>7} "
                    f"{f1_str:>6} "
                    f"{sota_f1_str:>6} "
                    f"{f1_gap_str:>7} "
                    f"{phases_str:>6} "
                    f"{time_str:>8}"
                )
                lines.append(row)

        # Summary footer
        lines.append("")

        sep = (
            "======================================="
            "===="
        )
        lines.append(sep)

        summary_line = (
            f"Summary: {report.successful}/"
            f"{report.total_datasets} datasets successful "
            f"({report.failed} failed, "
            f"{report.skipped} skipped)"
        )
        lines.append(summary_line)

        avg_auc_str = self._fmt_gap(report.avg_auc_gap)
        avg_f1_str = self._fmt_gap(report.avg_f1_gap)
        lines.append(
            f"Avg AUC gap: {avg_auc_str}  |  "
            f"Avg F1 gap: {avg_f1_str}"
        )

        # Best/worst lines
        if report.best_relative_performance:
            best_gap = self._find_auc_gap(
                report, report.best_relative_performance,
            )
            lines.append(
                f"Best:  {report.best_relative_performance}"
                f" (AUC gap: {self._fmt_gap(best_gap)})"
            )
        else:
            lines.append("Best:  N/A")

        if report.worst_relative_performance:
            worst_gap = self._find_auc_gap(
                report,
                report.worst_relative_performance,
            )
            lines.append(
                f"Worst: "
                f"{report.worst_relative_performance}"
                f" (AUC gap: {self._fmt_gap(worst_gap)})"
            )
        else:
            lines.append("Worst: N/A")

        return "\n".join(lines)

    def _find_auc_gap(
        self,
        report: BenchmarkReport,
        dataset_name: str,
    ) -> float | None:
        """Find the auc_gap for a named dataset.

        Args:
            report: The report containing results.
            dataset_name: Name of the dataset to look up.

        Returns:
            The auc_gap value, or None if not found.
        """
        for r in report.results:
            if r.dataset_name == dataset_name:
                return r.auc_gap
        return None

    @staticmethod
    def _fmt_metric(value: float | None) -> str:
        """Format a metric value to 3 decimal places.

        Args:
            value: Metric value or None.

        Returns:
            Formatted string or "N/A".
        """
        if value is None:
            return "N/A"
        return f"{value:.3f}"

    @staticmethod
    def _fmt_gap(value: float | None) -> str:
        """Format a gap value with sign to 3 decimal places.

        Args:
            value: Gap value or None.

        Returns:
            Formatted string with +/- sign, or "N/A".
        """
        if value is None:
            return "N/A"
        return f"{value:+.3f}"

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        """Format elapsed time as human-readable string.

        Renders as "Xs" for durations under 60 seconds,
        or "Xm Ys" for longer durations.

        Args:
            seconds: Elapsed time in seconds.

        Returns:
            Formatted time string.
        """
        if seconds < 60.0:
            return f"{seconds:.1f}s"
        minutes = int(seconds // 60)
        remaining = seconds - (minutes * 60)
        return f"{minutes}m {remaining:.0f}s"
