"""Sentinel agent for The Associate — 6-check audit gate.

Fifth agent in the pipeline (after Builder). Performs a separate LLM pass
auditing the Builder's diff. If any check fails, the build is rejected
and violations are fed back to the Builder for retry.

6 Checks:
1. Dependency Jail — blocks unauthorized packages (P8 from GrokFlow SafetyEnforcer)
2. Style Match — verifies code follows project conventions
3. Chaos Check — verifies edge case handling (Happy Path Bias counter)
4. Placeholder Scan — rejects TODOs, stubs, and unimplemented code
5. Drift Alignment — semantic similarity between task intent and output (P7 from GrokFlow)
6. Claim Validation — detects and verifies unsubstantiated claims (P9 from GrokFlow)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from src.agents.base_agent import BaseAgent
from src.core.config import PromptLoader
from src.core.models import (
    AgentResult,
    BuildResult,
    ContextBrief,
    SentinelVerdict,
    Task,
)
from src.db.embeddings import EmbeddingEngine

logger = logging.getLogger("associate.agent.sentinel")


# ---------------------------------------------------------------------------
# P8: Safety Enforcer keyword sets (from GrokFlow safety/enforcer.py)
# ---------------------------------------------------------------------------

DESTRUCTIVE_KEYWORDS = {
    "delete", "drop", "remove", "destroy", "erase", "purge",
    "truncate", "clear", "wipe", "kill", "terminate", "shutdown",
}

CRITICAL_KEYWORDS = {
    "deploy", "publish", "release", "migrate", "upgrade",
    "downgrade", "restart", "reboot", "format",
}

# Protected path patterns — code should not modify these
PROTECTED_PATTERNS = [
    r'.*\.git/.*',
    r'.*\.env\b',
    r'.*\.ssh/.*',
    r'.*password.*',
    r'.*secret.*',
    r'.*credential.*',
    r'/etc/.*',
    r'/usr/.*',
]

# ---------------------------------------------------------------------------
# P9: Claim definitions (from GrokFlow core/evidence.py)
# ---------------------------------------------------------------------------

CLAIM_PATTERNS = [
    {"claims": ["production ready", "prod ready", "ready for production"], "evidence": "Full pipeline validated"},
    {"claims": ["tested", "tests pass", "all tests pass"], "evidence": "Tests exist and pass"},
    {"claims": ["fixed", "resolved", "bug fixed"], "evidence": "Repro steps no longer fail"},
    {"claims": ["done", "complete", "completed", "finished"], "evidence": "Acceptance criteria met"},
    {"claims": ["refactored", "refactor"], "evidence": "Behavior unchanged, tests pass"},
    {"claims": ["optimized", "faster", "performance improved"], "evidence": "Benchmark before/after"},
    {"claims": ["secure", "hardened"], "evidence": "Security scan clean"},
    {"claims": ["no side effects", "side-effect free"], "evidence": "Pure function verification"},
    {"claims": ["thread safe", "thread-safe"], "evidence": "Concurrency test evidence"},
    {"claims": ["backward compatible", "backwards compatible"], "evidence": "API compatibility test"},
]

# ---------------------------------------------------------------------------
# Placeholder patterns
# ---------------------------------------------------------------------------

PLACEHOLDER_PATTERNS = [
    r'\bTODO\b',
    r'\bFIXME\b',
    r'\bHACK\b',
    r'\bXXX\b',
    r'\bpass\b\s*#',             # `pass  # placeholder`
    r'raise\s+NotImplementedError',
    r'\.\.\.\s*$',               # Ellipsis as implementation
    r'#\s*(?:implement|placeholder|stub)',
]


class Sentinel(BaseAgent):
    """6-check audit gate agent, with optional LLM deep check (7th check).

    Injected dependencies:
        embedding_engine: For drift alignment check (P7).
        banned_dependencies: List of packages that must not appear in code.
        drift_threshold: Minimum cosine similarity for drift check (0.0-1.0).
        llm_deep_check: Enable optional 7th check using LLM review (R5).
        llm_client: OpenRouter client (required if llm_deep_check=True).
        model_router: Model router (required if llm_deep_check=True).
        prompt_loader: Prompt loader for sentinel_deep_check.txt.
    """

    _DEFAULT_DEEP_CHECK_PROMPT = (
        "Review this diff against the task. Does it fully solve the task? "
        "Return JSON: {\"verdict\": \"PASS\"|\"FAIL\", \"gaps\": [...], \"severity\": \"none\"|\"low\"|\"medium\"|\"high\"|\"critical\"}"
    )

    def __init__(
        self,
        embedding_engine: EmbeddingEngine,
        banned_dependencies: Optional[list[str]] = None,
        drift_threshold: float = 0.40,
        llm_deep_check: bool = False,
        llm_client: Optional[Any] = None,
        model_router: Optional[Any] = None,
        prompt_loader: Optional[PromptLoader] = None,
    ):
        super().__init__(name="Sentinel", role="sentinel")
        self.embedding_engine = embedding_engine
        self.banned_dependencies = set(d.lower() for d in (banned_dependencies or []))
        self.drift_threshold = drift_threshold
        self.llm_deep_check = llm_deep_check
        self.llm_client = llm_client
        self.model_router = model_router
        self._prompt_loader = prompt_loader or PromptLoader()

    def process(self, input_data: Any) -> AgentResult:
        """Audit the Builder's output against 6 safety checks.

        Args:
            input_data: Dictionary containing:
                - "build_result": BuildResult dict or model
                - "context_brief": ContextBrief or dict with task info
                - "task": Task model (for drift check)

        Returns:
            AgentResult with SentinelVerdict in data["sentinel_verdict"].
        """
        build_result, task_desc, approach_summary, diff, banned, self_audit = self._extract_inputs(input_data)

        all_violations: list[dict[str, str]] = []
        all_recommendations: list[str] = []

        # Run all 6 checks
        checks = [
            ("dependency_jail", self._check_dependency_jail, (diff, banned)),
            ("style_match", self._check_style_match, (diff,)),
            ("chaos_check", self._check_chaos, (diff,)),
            ("placeholder_scan", self._check_placeholders, (diff,)),
            ("drift_alignment", self._check_drift_alignment, (task_desc, approach_summary)),
            ("claim_validation", self._check_claims, (approach_summary, build_result, self_audit)),
        ]

        for check_name, check_fn, args in checks:
            try:
                violations, recommendations = check_fn(*args)
                all_violations.extend(violations)
                all_recommendations.extend(recommendations)
                logger.debug("Check '%s': %d violations", check_name, len(violations))
            except Exception as e:
                logger.warning("Check '%s' failed: %s", check_name, e)
                all_recommendations.append(f"Check '{check_name}' could not be executed: {e}")

        # Optional 7th check: LLM deep review (R5)
        # Only runs when enabled AND all 6 rule-based checks passed
        tokens_used = 0
        if self.llm_deep_check and len(all_violations) == 0:
            deep_violations, deep_recommendations, tokens_used = self._check_llm_deep(
                task_desc, approach_summary, diff
            )
            all_violations.extend(deep_violations)
            all_recommendations.extend(deep_recommendations)

        approved = len(all_violations) == 0

        verdict = SentinelVerdict(
            approved=approved,
            violations=all_violations,
            recommendations=all_recommendations,
        )

        status = "success" if approved else "failure"
        error = None
        if not approved:
            error = f"Sentinel rejected: {len(all_violations)} violation(s) found"

        return AgentResult(
            agent_name=self.name,
            status=status,
            data={
                "sentinel_verdict": verdict.model_dump(),
                "tokens_used": tokens_used,
            },
            error=error,
        )

    def _extract_inputs(self, input_data: Any) -> tuple[BuildResult, str, str, str, set[str], str]:
        """Extract build result, task description, diff, and self-audit from input.

        Returns:
            (build_result, task_desc, approach_summary, diff, banned_deps, self_audit)
        """
        if isinstance(input_data, dict):
            # Build result
            br_data = input_data.get("build_result", {})
            if isinstance(br_data, BuildResult):
                build_result = br_data
            elif isinstance(br_data, dict):
                build_result = BuildResult(**br_data)
            else:
                build_result = BuildResult()

            # Task description for drift check
            task_desc = ""
            cb_data = input_data.get("context_brief")
            if isinstance(cb_data, ContextBrief):
                task_desc = f"{cb_data.task.title}: {cb_data.task.description}"
            elif isinstance(cb_data, dict):
                task = cb_data.get("task", {})
                if isinstance(task, dict):
                    task_desc = f"{task.get('title', '')}: {task.get('description', '')}"

            task_data = input_data.get("task")
            if task_data and not task_desc:
                if isinstance(task_data, Task):
                    task_desc = f"{task_data.title}: {task_data.description}"
                elif isinstance(task_data, dict):
                    task_desc = f"{task_data.get('title', '')}: {task_data.get('description', '')}"

            # Banned dependencies (from project or input)
            banned = set(self.banned_dependencies)
            extra_banned = input_data.get("banned_dependencies", [])
            if extra_banned:
                banned.update(d.lower() for d in extra_banned)

            # Self-audit from Builder (R8 feed-forward)
            self_audit = input_data.get("self_audit", "")

            return build_result, task_desc, build_result.approach_summary, build_result.diff, banned, self_audit

        raise ValueError(f"Sentinel received unexpected input type: {type(input_data)}")

    # -------------------------------------------------------------------
    # Check 1: Dependency Jail (P8)
    # -------------------------------------------------------------------

    def _check_dependency_jail(
        self, diff: str, banned: set[str]
    ) -> tuple[list[dict[str, str]], list[str]]:
        """Block unauthorized package imports.

        Scans the diff for import statements that reference banned packages
        or packages that look like new dependencies not in the approved list.
        """
        violations = []
        recommendations = []

        if not diff:
            return violations, recommendations

        # Find all import statements in the diff (new lines only)
        import_pattern = re.compile(
            r'^\+\s*(?:import|from)\s+(\w+)', re.MULTILINE
        )

        for match in import_pattern.finditer(diff):
            package = match.group(1).lower()

            if package in banned:
                violations.append({
                    "check": "dependency_jail",
                    "detail": f"Banned dependency detected: '{package}'. "
                              f"This package is not authorized for this project.",
                })

        # Check for destructive operations in new code
        for line in diff.split("\n"):
            if not line.startswith("+"):
                continue
            line_lower = line.lower()
            for keyword in DESTRUCTIVE_KEYWORDS:
                # Only flag if it looks like a function call or command
                if re.search(rf'\b{keyword}\s*\(', line_lower):
                    violations.append({
                        "check": "dependency_jail",
                        "detail": f"Destructive operation detected: '{keyword}' call in new code.",
                    })
                    break

        return violations, recommendations

    # -------------------------------------------------------------------
    # Check 2: Style Match
    # -------------------------------------------------------------------

    def _check_style_match(
        self, diff: str
    ) -> tuple[list[dict[str, str]], list[str]]:
        """Verify code follows basic style conventions."""
        violations = []
        recommendations = []

        if not diff:
            return violations, recommendations

        new_lines = [l[1:] for l in diff.split("\n") if l.startswith("+") and not l.startswith("+++")]

        # Check for mixing tabs and spaces
        has_tabs = any("\t" in line for line in new_lines)
        has_spaces = any(line.startswith("    ") for line in new_lines)
        if has_tabs and has_spaces:
            violations.append({
                "check": "style_match",
                "detail": "Mixed tabs and spaces detected in new code.",
            })

        # Check for extremely long lines
        for line in new_lines:
            if len(line) > 200:
                recommendations.append(
                    f"Line exceeds 200 characters ({len(line)} chars). Consider breaking it up."
                )
                break  # One warning is enough

        # Check for wildcard imports
        if any(re.search(r'from\s+\S+\s+import\s+\*', line) for line in new_lines):
            violations.append({
                "check": "style_match",
                "detail": "Wildcard import (from X import *) detected. Use explicit imports.",
            })

        return violations, recommendations

    # -------------------------------------------------------------------
    # Check 3: Chaos Check (Happy Path Bias counter)
    # -------------------------------------------------------------------

    def _check_chaos(
        self, diff: str
    ) -> tuple[list[dict[str, str]], list[str]]:
        """Verify edge case handling in new code."""
        violations = []
        recommendations = []

        if not diff:
            return violations, recommendations

        new_lines = [l[1:] for l in diff.split("\n") if l.startswith("+") and not l.startswith("+++")]
        code_block = "\n".join(new_lines)

        # Check for bare except
        if re.search(r'except\s*:', code_block):
            violations.append({
                "check": "chaos_check",
                "detail": "Bare 'except:' found. Catch specific exceptions to avoid masking bugs.",
            })

        # Check for eval/exec
        if re.search(r'\b(?:eval|exec)\s*\(', code_block):
            violations.append({
                "check": "chaos_check",
                "detail": "eval() or exec() detected. These are security risks.",
            })

        # Check for hardcoded credentials
        credential_patterns = [
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']',
            r'secret\s*=\s*["\'][^"\']+["\']',
            r'token\s*=\s*["\'][^"\']+["\']',
        ]
        for pattern in credential_patterns:
            if re.search(pattern, code_block, re.IGNORECASE):
                violations.append({
                    "check": "chaos_check",
                    "detail": "Possible hardcoded credential detected. Use environment variables.",
                })
                break

        # Check for no None/empty checks before operations
        # (Heuristic: functions that access .attribute without None check)
        if re.search(r'\.split\(|\.strip\(|\.lower\(', code_block):
            if not re.search(r'if\s+\w+\s+is\s+not\s+None|if\s+\w+:', code_block):
                recommendations.append(
                    "Code accesses string methods without apparent None checks. "
                    "Consider adding null safety."
                )

        return violations, recommendations

    # -------------------------------------------------------------------
    # Check 4: Placeholder Scan
    # -------------------------------------------------------------------

    def _check_placeholders(
        self, diff: str
    ) -> tuple[list[dict[str, str]], list[str]]:
        """Reject code containing TODOs, stubs, or unimplemented sections."""
        violations = []
        recommendations = []

        if not diff:
            return violations, recommendations

        new_lines = [l[1:] for l in diff.split("\n") if l.startswith("+") and not l.startswith("+++")]

        for i, line in enumerate(new_lines):
            for pattern in PLACEHOLDER_PATTERNS:
                if re.search(pattern, line, re.IGNORECASE):
                    violations.append({
                        "check": "placeholder_scan",
                        "detail": f"Placeholder/stub detected on new line {i + 1}: {line.strip()[:80]}",
                    })
                    break

        return violations, recommendations

    # -------------------------------------------------------------------
    # Check 5: Drift Alignment (P7 from GrokFlow)
    # -------------------------------------------------------------------

    def _check_drift_alignment(
        self, task_description: str, approach_summary: str
    ) -> tuple[list[dict[str, str]], list[str]]:
        """Check semantic alignment between task intent and Builder's approach.

        Uses sentence-transformers embeddings to compute cosine similarity
        between the original task description and the Builder's approach summary.
        Adapted from GrokFlow's DriftDetector.
        """
        violations = []
        recommendations = []

        if not task_description or not approach_summary:
            recommendations.append("Drift check skipped: missing task description or approach summary.")
            return violations, recommendations

        try:
            similarity = self._compute_alignment(task_description, approach_summary)
            logger.info("Drift alignment score: %.3f (threshold: %.3f)", similarity, self.drift_threshold)

            if similarity < self.drift_threshold:
                severity = self._drift_severity(similarity)
                guidance = self._drift_guidance(similarity, severity)

                violations.append({
                    "check": "drift_alignment",
                    "detail": (
                        f"Task drift detected (severity: {severity}). "
                        f"Alignment score: {similarity:.3f} "
                        f"(threshold: {self.drift_threshold:.3f}). "
                        f"The Builder's approach may not address the original task. "
                        f"{guidance}"
                    ),
                })
            elif similarity < self.drift_threshold + 0.1:
                recommendations.append(
                    f"Drift alignment borderline ({similarity:.3f}). "
                    f"Review that the approach fully addresses the task."
                )

        except Exception as e:
            logger.warning("Drift alignment check failed: %s", e)
            recommendations.append(f"Drift check could not be executed: {e}")

        return violations, recommendations

    def _compute_alignment(self, task_desc: str, approach: str) -> float:
        """Compute cosine similarity between task and approach embeddings."""
        task_vec = self.embedding_engine.encode(task_desc)
        approach_vec = self.embedding_engine.encode(approach)
        return self.embedding_engine.cosine_similarity(task_vec, approach_vec)

    def _drift_severity(self, similarity: float) -> str:
        """Determine drift severity from similarity score.

        Adapted from GrokFlow's DriftDetector._determine_severity.
        """
        if similarity < 0.1:
            return "CRITICAL"
        if similarity < 0.2:
            return "HIGH"
        if similarity < 0.3:
            return "MEDIUM"
        return "LOW"

    def _drift_guidance(self, similarity: float, severity: str) -> str:
        """Generate corrective guidance based on drift severity.

        Adapted from GrokFlow's DriftDetector._generate_corrective_guidance.
        """
        if severity == "CRITICAL":
            return "The approach appears completely unrelated to the task. Restart from the task description."
        if severity == "HIGH":
            return "Significant divergence from task intent. Re-read the task and refocus the approach."
        if severity == "MEDIUM":
            return "Partial drift detected. Ensure all task requirements are addressed."
        return "Minor drift. Verify edge cases and secondary requirements."

    # -------------------------------------------------------------------
    # Check 6: Claim Validation (P9 from GrokFlow)
    # -------------------------------------------------------------------

    def _check_claims(
        self, approach_summary: str, build_result: BuildResult, self_audit: str = ""
    ) -> tuple[list[dict[str, str]], list[str]]:
        """Detect and validate claims in the Builder's output.

        Scans the approach summary for claims like "tested", "production ready",
        "fixed", etc., and validates them against actual evidence from the
        BuildResult. Cross-checks with Builder's self-audit (R8 feed-forward).
        Adapted from GrokFlow's evidence.py.
        """
        violations = []
        recommendations = []

        if not approach_summary:
            return violations, recommendations

        text = approach_summary.lower()

        for claim_def in CLAIM_PATTERNS:
            for phrase in claim_def["claims"]:
                pattern = r"\b" + re.escape(phrase) + r"\b"
                if re.search(pattern, text):
                    # Claim detected — validate against evidence
                    evidence = claim_def["evidence"]
                    verdict = self._validate_claim(phrase, build_result)

                    if verdict == "BLOCK":
                        violations.append({
                            "check": "claim_validation",
                            "detail": (
                                f"Unsubstantiated claim: '{phrase}'. "
                                f"Required evidence: {evidence}. "
                                f"No supporting evidence found in build output."
                            ),
                        })
                    elif verdict == "PARTIAL":
                        recommendations.append(
                            f"Claim '{phrase}' is only partially substantiated. "
                            f"Required: {evidence}."
                        )
                    break  # One match per claim definition

        # R8: Cross-check self-audit against actual check results
        if self_audit:
            self._cross_check_self_audit(self_audit, build_result, violations, recommendations)

        return violations, recommendations

    def _cross_check_self_audit(
        self,
        self_audit: str,
        build_result: BuildResult,
        violations: list[dict[str, str]],
        recommendations: list[str],
    ) -> None:
        """Cross-check Builder's self-audit claims against actual evidence.

        If the Builder claims YES to a self-audit question but evidence
        contradicts it, add a compounding violation.
        """
        audit_lower = self_audit.lower()

        # Check: Builder claims "no placeholders" but placeholders were found
        placeholder_violations = [v for v in violations if v.get("check") == "placeholder_scan"]
        if placeholder_violations and ("yes" in audit_lower) and ("placeholder" in audit_lower or "todo" in audit_lower):
            violations.append({
                "check": "claim_validation",
                "detail": (
                    "Self-audit contradiction: Builder claimed no placeholders/TODOs "
                    "in self-audit, but placeholder scan found violations."
                ),
            })

        # Check: Builder claims error handling but bare except found
        chaos_violations = [v for v in violations if v.get("check") == "chaos_check" and "except:" in v.get("detail", "")]
        if chaos_violations and ("yes" in audit_lower) and ("error handling" in audit_lower):
            violations.append({
                "check": "claim_validation",
                "detail": (
                    "Self-audit contradiction: Builder claimed error handling "
                    "in self-audit, but bare 'except:' was detected."
                ),
            })

    def _validate_claim(self, claim_phrase: str, build_result: BuildResult) -> str:
        """Validate a specific claim against build evidence.

        Returns:
            "PASS", "PARTIAL", or "BLOCK"
        """
        # "tested" / "tests pass" — check if tests actually passed
        test_claims = {"tested", "tests pass", "all tests pass"}
        if claim_phrase in test_claims:
            if build_result.tests_passed:
                return "PASS"
            return "BLOCK"

        # "fixed" / "resolved" — check if tests passed (proxy for fix verification)
        fix_claims = {"fixed", "resolved", "bug fixed"}
        if claim_phrase in fix_claims:
            if build_result.tests_passed and build_result.files_changed:
                return "PARTIAL"  # Fixed but no specific repro verification
            return "BLOCK"

        # "done" / "complete" — check tests + files changed
        done_claims = {"done", "complete", "completed", "finished"}
        if claim_phrase in done_claims:
            if build_result.tests_passed and build_result.files_changed:
                return "PARTIAL"  # Complete but no acceptance criteria check
            return "BLOCK"

        # "production ready" — always BLOCK unless full pipeline validated
        production_claims = {"production ready", "prod ready", "ready for production"}
        if claim_phrase in production_claims:
            return "BLOCK"

        # Other claims — partial by default
        return "PARTIAL"

    # -------------------------------------------------------------------
    # Check 7: LLM Deep Review (R5 — optional)
    # -------------------------------------------------------------------

    def _check_llm_deep(
        self, task_desc: str, approach_summary: str, diff: str
    ) -> tuple[list[dict[str, str]], list[str], int]:
        """Call the sentinel LLM model to review the diff against the task.

        Only executed when llm_deep_check=True and all rule-based checks passed.
        Uses a different model family than the Builder (via model_router).

        Returns:
            (violations, recommendations, tokens_used)
        """
        violations: list[dict[str, str]] = []
        recommendations: list[str] = []

        if not self.llm_client or not self.model_router:
            recommendations.append(
                "LLM deep check enabled but llm_client or model_router not configured."
            )
            return violations, recommendations, 0

        if not diff:
            recommendations.append("LLM deep check skipped: no diff to review.")
            return violations, recommendations, 0

        try:
            from src.llm.client import LLMMessage

            model_id = self.model_router.get_model(self.role)

            # Build the prompt (use replace instead of format to avoid
            # conflicts with JSON curly braces in the template)
            prompt_template = self._prompt_loader.load(
                "sentinel_deep_check.txt", default=self._DEFAULT_DEEP_CHECK_PROMPT
            )
            prompt = (
                prompt_template
                .replace("{task_description}", task_desc or "(no task description)")
                .replace("{diff}", diff[:8000])  # Truncate to avoid token limits
                .replace("{approach_summary}", approach_summary or "(no approach summary)")
            )

            messages = [
                LLMMessage(role="user", content=prompt),
            ]
            response = self.llm_client.complete(messages, model=model_id)

            # Parse the JSON response
            result = self._parse_deep_check_response(response.content)
            if result:
                if result.get("verdict") == "FAIL":
                    gaps = result.get("gaps", [])
                    severity = result.get("severity", "medium")
                    for gap in gaps[:5]:  # Limit to 5 gaps
                        violations.append({
                            "check": "llm_deep_review",
                            "detail": f"[{severity}] {gap}",
                        })
                    if not gaps:
                        violations.append({
                            "check": "llm_deep_review",
                            "detail": f"LLM deep review returned FAIL (severity: {severity}) but no specific gaps listed.",
                        })
                else:
                    logger.info("LLM deep check PASSED (model=%s)", model_id)

            return violations, recommendations, response.tokens_used

        except Exception as e:
            logger.warning("LLM deep check failed: %s", e)
            recommendations.append(f"LLM deep check could not be executed: {e}")
            return violations, recommendations, 0

    def _parse_deep_check_response(self, content: str) -> Optional[dict]:
        """Parse the LLM deep check JSON response."""
        import json

        from src.llm.response_parser import extract_json_block

        # Try structured JSON extraction first (returns dict or None)
        data = extract_json_block(content)
        if isinstance(data, dict) and "verdict" in data:
            return data

        # Fallback: try parsing the whole content as JSON
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "verdict" in data:
                return data
        except (json.JSONDecodeError, TypeError):
            pass

        logger.warning("Could not parse LLM deep check response")
        return None
