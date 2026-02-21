"""WrapUp workflow for salvaging partial outputs on Builder timeout (Item 9).

Adapted from ClawWork's wrapup_workflow.py concept. When the Builder
exhausts all retries without a passing build, this workflow:
1. Collects any partial file outputs from the last attempt
2. Asks the Council to evaluate which partial artifacts are worth keeping
3. Scores them against the Sentinel at a relaxed threshold
4. Records the partial attempt for future learning

This is NOT a LangGraph dependency — it is a pure Python state machine.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from src.core.models import BuildResult, Task

logger = logging.getLogger("associate.orchestrator.wrapup")


class WrapUpWorkflow:
    """Salvage partial outputs when Builder exhausts retries."""

    def __init__(
        self,
        council: Optional[Any] = None,
        sentinel: Optional[Any] = None,
        relaxed_quality_threshold: float = 0.40,
    ):
        self.council = council
        self.sentinel = sentinel
        self.relaxed_quality_threshold = relaxed_quality_threshold

    def attempt_salvage(
        self,
        task: Task,
        last_build_result: BuildResult,
    ) -> dict[str, Any]:
        """Attempt to salvage partial work from a failed task.

        Returns a dict with:
            - salvaged: bool — whether any partial output was saved
            - files: list[str] — files from the partial output
            - quality_score: float — score from relaxed Sentinel check
            - reason: str — explanation of the salvage decision
        """
        result: dict[str, Any] = {
            "salvaged": False,
            "files": [],
            "quality_score": 0.0,
            "reason": "no_partial_output",
        }

        # Step 1: Check if there are any partial file outputs
        if not last_build_result.files_changed:
            logger.info("No partial files to salvage for task '%s'", task.title)
            return result

        result["files"] = list(last_build_result.files_changed)

        # Step 2: Ask Council to evaluate partial output (if available)
        if self.council is not None:
            try:
                council_input = {
                    "task": task,
                    "hypothesis_log": [],
                    "project_rules": None,
                    "partial_output": {
                        "files_changed": last_build_result.files_changed,
                        "approach_summary": last_build_result.approach_summary,
                        "test_output": last_build_result.test_output[:500],
                    },
                    "mode": "salvage_evaluation",
                }
                council_result = self.council.run(council_input)
                if council_result.status == "success" and council_result.data:
                    diag = council_result.data.get("council_diagnosis", {})
                    if isinstance(diag, dict) and diag.get("new_approach", "").lower().startswith("discard"):
                        result["reason"] = "council_recommends_discard"
                        return result
            except Exception as e:
                logger.warning("Council salvage evaluation failed: %s", e)

        # Step 3: Score against Sentinel at relaxed threshold
        if self.sentinel is not None:
            try:
                sentinel_input = {
                    "build_result": last_build_result.model_dump(),
                    "task": task,
                    "self_audit": "",
                }
                sentinel_result = self.sentinel.run(sentinel_input)
                verdict_data = sentinel_result.data.get("sentinel_verdict", {}) if sentinel_result.data else {}
                quality = float(verdict_data.get("quality_score", 0.0) or 0.0)
                result["quality_score"] = quality

                if quality >= self.relaxed_quality_threshold:
                    result["salvaged"] = True
                    result["reason"] = f"partial_output_acceptable (quality={quality:.2f})"
                    logger.info(
                        "Salvaged partial output for task '%s' (quality=%.2f)",
                        task.title, quality,
                    )
                else:
                    result["reason"] = f"partial_output_below_threshold (quality={quality:.2f} < {self.relaxed_quality_threshold:.2f})"
            except Exception as e:
                logger.warning("Sentinel salvage check failed: %s", e)
                result["reason"] = f"sentinel_check_failed: {e}"
        else:
            # No sentinel — accept partial output if files exist
            result["salvaged"] = bool(last_build_result.files_changed)
            result["reason"] = "no_sentinel_available_accepting_partial"

        return result
