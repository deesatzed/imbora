"""Tests for src/agents/sentinel.py — 6-check audit gate.

Tests all 6 sentinel checks:
1. Dependency Jail (P8)
2. Style Match
3. Chaos Check
4. Placeholder Scan
5. Drift Alignment (P7)
6. Claim Validation (P9)
"""

import uuid

import pytest

from src.agents.sentinel import (
    CLAIM_PATTERNS,
    CRITICAL_KEYWORDS,
    DESTRUCTIVE_KEYWORDS,
    PLACEHOLDER_PATTERNS,
    PROTECTED_PATTERNS,
    Sentinel,
)
from src.core.config import EmbeddingsConfig
from src.core.models import BuildResult, ContextBrief, SentinelVerdict, Task, TaskContext


# ---------------------------------------------------------------------------
# Test fixture: Sentinel with real embeddings
# ---------------------------------------------------------------------------


@pytest.fixture
def embedding_engine():
    from src.db.embeddings import EmbeddingEngine
    return EmbeddingEngine(EmbeddingsConfig())


@pytest.fixture
def sentinel(embedding_engine):
    return Sentinel(
        embedding_engine=embedding_engine,
        banned_dependencies=["flask", "django"],
        drift_threshold=0.40,
    )


# ---------------------------------------------------------------------------
# Check 1: Dependency Jail tests
# ---------------------------------------------------------------------------


class TestDependencyJail:
    def test_blocked_import(self, sentinel):
        diff = "+import flask\n+from flask import Flask"
        violations, _ = sentinel._check_dependency_jail(diff, {"flask"})
        assert len(violations) >= 1
        assert any("flask" in v["detail"] for v in violations)

    def test_allowed_import(self, sentinel):
        diff = "+import pydantic\n+from pydantic import BaseModel"
        violations, _ = sentinel._check_dependency_jail(diff, {"flask"})
        assert len(violations) == 0

    def test_destructive_call(self, sentinel):
        diff = "+    database.drop()\n+    os.remove('/tmp/file')"
        violations, _ = sentinel._check_dependency_jail(diff, set())
        assert len(violations) >= 1
        assert any("Destructive" in v["detail"] for v in violations)

    def test_no_diff(self, sentinel):
        violations, _ = sentinel._check_dependency_jail("", set())
        assert len(violations) == 0

    def test_only_new_lines_checked(self, sentinel):
        diff = "-import flask\n old line import flask"
        violations, _ = sentinel._check_dependency_jail(diff, {"flask"})
        assert len(violations) == 0


class TestDestructiveKeywords:
    def test_all_keywords_defined(self):
        assert len(DESTRUCTIVE_KEYWORDS) >= 10
        assert "delete" in DESTRUCTIVE_KEYWORDS
        assert "drop" in DESTRUCTIVE_KEYWORDS

    def test_critical_keywords_defined(self):
        assert len(CRITICAL_KEYWORDS) >= 7
        assert "deploy" in CRITICAL_KEYWORDS
        assert "migrate" in CRITICAL_KEYWORDS


# ---------------------------------------------------------------------------
# Check 2: Style Match tests
# ---------------------------------------------------------------------------


class TestStyleMatch:
    def test_mixed_tabs_spaces(self, sentinel):
        diff = "+\tindented with tab\n+    indented with spaces"
        violations, _ = sentinel._check_style_match(diff)
        assert any("Mixed tabs" in v["detail"] for v in violations)

    def test_consistent_spaces(self, sentinel):
        diff = "+    line 1\n+    line 2\n+    line 3"
        violations, _ = sentinel._check_style_match(diff)
        tab_violations = [v for v in violations if "tabs" in v["detail"].lower()]
        assert len(tab_violations) == 0

    def test_wildcard_import(self, sentinel):
        diff = "+from os import *"
        violations, _ = sentinel._check_style_match(diff)
        assert any("Wildcard" in v["detail"] for v in violations)

    def test_long_line_recommendation(self, sentinel):
        diff = "+" + "x" * 250
        _, recommendations = sentinel._check_style_match(diff)
        assert any("200 characters" in r for r in recommendations)

    def test_empty_diff(self, sentinel):
        violations, recommendations = sentinel._check_style_match("")
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# Check 3: Chaos Check tests
# ---------------------------------------------------------------------------


class TestChaosCheck:
    def test_bare_except(self, sentinel):
        diff = "+try:\n+    do_something()\n+except:\n+    pass"
        violations, _ = sentinel._check_chaos(diff)
        assert any("Bare" in v["detail"] for v in violations)

    def test_specific_except_ok(self, sentinel):
        diff = "+try:\n+    do_something()\n+except ValueError:\n+    pass"
        violations, _ = sentinel._check_chaos(diff)
        bare_violations = [v for v in violations if "Bare" in v["detail"]]
        assert len(bare_violations) == 0

    def test_eval_detected(self, sentinel):
        diff = "+result = eval(user_input)"
        violations, _ = sentinel._check_chaos(diff)
        assert any("eval" in v["detail"] for v in violations)

    def test_exec_detected(self, sentinel):
        diff = "+exec(code_string)"
        violations, _ = sentinel._check_chaos(diff)
        assert any("exec" in v["detail"] for v in violations)

    def test_hardcoded_password(self, sentinel):
        diff = "+password = 'my_secret_pass_123'"
        violations, _ = sentinel._check_chaos(diff)
        assert any("credential" in v["detail"].lower() for v in violations)

    def test_hardcoded_api_key(self, sentinel):
        diff = "+API_KEY = 'sk-12345abcdef'"
        violations, _ = sentinel._check_chaos(diff)
        assert any("credential" in v["detail"].lower() for v in violations)

    def test_env_var_ok(self, sentinel):
        diff = "+password = os.getenv('DB_PASSWORD')"
        violations, _ = sentinel._check_chaos(diff)
        credential_violations = [v for v in violations if "credential" in v["detail"].lower()]
        assert len(credential_violations) == 0

    def test_empty_diff(self, sentinel):
        violations, _ = sentinel._check_chaos("")
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# Check 4: Placeholder Scan tests
# ---------------------------------------------------------------------------


class TestPlaceholderScan:
    def test_todo_detected(self, sentinel):
        diff = "+# TODO: implement this later"
        violations, _ = sentinel._check_placeholders(diff)
        assert len(violations) >= 1

    def test_fixme_detected(self, sentinel):
        diff = "+# FIXME: this is broken"
        violations, _ = sentinel._check_placeholders(diff)
        assert len(violations) >= 1

    def test_not_implemented_detected(self, sentinel):
        diff = "+    raise NotImplementedError"
        violations, _ = sentinel._check_placeholders(diff)
        assert len(violations) >= 1

    def test_ellipsis_body(self, sentinel):
        diff = "+def my_func():\n+    ..."
        violations, _ = sentinel._check_placeholders(diff)
        assert len(violations) >= 1

    def test_stub_comment(self, sentinel):
        diff = "+# placeholder for real implementation"
        violations, _ = sentinel._check_placeholders(diff)
        assert len(violations) >= 1

    def test_real_code_ok(self, sentinel):
        diff = "+def real_function():\n+    return calculate_value(x, y)"
        violations, _ = sentinel._check_placeholders(diff)
        assert len(violations) == 0

    def test_todo_in_comment_is_violation(self, sentinel):
        diff = "+    # TODO: fix this edge case"
        violations, _ = sentinel._check_placeholders(diff)
        assert len(violations) >= 1

    def test_hack_detected(self, sentinel):
        diff = "+# HACK: workaround for bug"
        violations, _ = sentinel._check_placeholders(diff)
        assert len(violations) >= 1


# ---------------------------------------------------------------------------
# Check 5: Drift Alignment tests
# ---------------------------------------------------------------------------


class TestDriftAlignment:
    def test_aligned_task(self, sentinel):
        """Task and approach are semantically similar — should pass."""
        violations, _ = sentinel._check_drift_alignment(
            task_description="Implement user authentication with JWT tokens",
            approach_summary="Added JWT-based authentication using PyJWT library with token refresh",
        )
        assert len(violations) == 0

    def test_drifted_task(self, sentinel):
        """Task and approach are completely unrelated — should fail."""
        violations, _ = sentinel._check_drift_alignment(
            task_description="Implement user authentication with JWT tokens",
            approach_summary="Created a beautiful CSS animation for the landing page hero section",
        )
        assert len(violations) >= 1
        assert any("drift" in v["detail"].lower() for v in violations)

    def test_missing_description(self, sentinel):
        """Missing task description — should skip with recommendation."""
        violations, recommendations = sentinel._check_drift_alignment(
            task_description="",
            approach_summary="Some approach",
        )
        assert len(violations) == 0
        assert any("skipped" in r.lower() for r in recommendations)

    def test_missing_approach(self, sentinel):
        violations, recommendations = sentinel._check_drift_alignment(
            task_description="Some task",
            approach_summary="",
        )
        assert len(violations) == 0
        assert any("skipped" in r.lower() for r in recommendations)


class TestDriftSeverity:
    def test_critical(self, sentinel):
        assert sentinel._drift_severity(0.05) == "CRITICAL"

    def test_high(self, sentinel):
        assert sentinel._drift_severity(0.15) == "HIGH"

    def test_medium(self, sentinel):
        assert sentinel._drift_severity(0.25) == "MEDIUM"

    def test_low(self, sentinel):
        assert sentinel._drift_severity(0.35) == "LOW"


class TestDriftGuidance:
    def test_critical_guidance(self, sentinel):
        guidance = sentinel._drift_guidance(0.05, "CRITICAL")
        assert "unrelated" in guidance.lower() or "restart" in guidance.lower()

    def test_low_guidance(self, sentinel):
        guidance = sentinel._drift_guidance(0.35, "LOW")
        assert "minor" in guidance.lower() or "verify" in guidance.lower()


# ---------------------------------------------------------------------------
# Check 6: Claim Validation tests
# ---------------------------------------------------------------------------


class TestClaimValidation:
    def test_tested_claim_with_passing_tests(self, sentinel):
        br = BuildResult(tests_passed=True, files_changed=["main.py"])
        violations, _ = sentinel._check_claims("All tests pass and code is tested", br)
        assert len(violations) == 0

    def test_tested_claim_without_passing_tests(self, sentinel):
        br = BuildResult(tests_passed=False)
        violations, _ = sentinel._check_claims("Code is tested", br)
        assert any("tested" in v["detail"].lower() for v in violations)

    def test_production_ready_always_blocked(self, sentinel):
        br = BuildResult(tests_passed=True, files_changed=["main.py"])
        violations, _ = sentinel._check_claims("This is production ready", br)
        assert any("production ready" in v["detail"].lower() for v in violations)

    def test_fixed_claim_partial(self, sentinel):
        br = BuildResult(tests_passed=True, files_changed=["bugfix.py"])
        violations, recommendations = sentinel._check_claims("Bug is fixed", br)
        # Should be PARTIAL (no repro steps verification)
        assert any("fixed" in r.lower() for r in recommendations)

    def test_done_claim_blocked_without_tests(self, sentinel):
        br = BuildResult(tests_passed=False)
        violations, _ = sentinel._check_claims("Task is complete", br)
        assert any("complete" in v["detail"].lower() for v in violations)

    def test_no_claims(self, sentinel):
        br = BuildResult()
        violations, _ = sentinel._check_claims("Implemented the feature", br)
        assert len(violations) == 0

    def test_empty_summary(self, sentinel):
        br = BuildResult()
        violations, _ = sentinel._check_claims("", br)
        assert len(violations) == 0


class TestSelfAuditCrossCheck:
    """R8: Test cross-checking Builder's self-audit claims against check results."""

    def test_placeholder_contradiction(self, sentinel):
        """Builder claims no placeholders but placeholder scan finds TODOs."""
        br = BuildResult(
            tests_passed=True,
            files_changed=["main.py"],
            diff="+# TODO: finish implementation",
            approach_summary="Implemented feature",
        )
        self_audit = "All functions fully implemented (no TODOs, stubs, placeholders)? YES"
        violations, _ = sentinel._check_claims("Implemented feature", br, self_audit)
        # _check_claims alone won't find placeholders — they come from _check_placeholders
        # So we test the cross-check directly
        placeholder_violations = [{"check": "placeholder_scan", "detail": "TODO found"}]
        recommendations: list[str] = []
        sentinel._cross_check_self_audit(self_audit, br, placeholder_violations, recommendations)
        assert any("contradiction" in v["detail"].lower() for v in placeholder_violations if v["check"] == "claim_validation")

    def test_no_contradiction_when_no_self_audit(self, sentinel):
        """No self-audit means no cross-check."""
        br = BuildResult(tests_passed=True)
        violations, _ = sentinel._check_claims("Implemented feature", br, "")
        # Should not add any cross-check violations
        contradiction = [v for v in violations if "contradiction" in v.get("detail", "").lower()]
        assert len(contradiction) == 0

    def test_error_handling_contradiction(self, sentinel):
        """Builder claims error handling but bare except found."""
        self_audit = "Error handling for external calls? YES"
        violations: list[dict[str, str]] = [
            {"check": "chaos_check", "detail": "Bare 'except:' found. Catch specific exceptions."},
        ]
        recommendations: list[str] = []
        sentinel._cross_check_self_audit(self_audit, BuildResult(), violations, recommendations)
        assert any("contradiction" in v["detail"].lower() for v in violations if v["check"] == "claim_validation")

    def test_no_false_positive_without_yes(self, sentinel):
        """Self-audit with NO answers should not trigger contradiction."""
        self_audit = "All functions fully implemented (no TODOs, stubs, placeholders)? NO"
        violations: list[dict[str, str]] = [
            {"check": "placeholder_scan", "detail": "TODO found"},
        ]
        recommendations: list[str] = []
        sentinel._cross_check_self_audit(self_audit, BuildResult(), violations, recommendations)
        contradiction = [v for v in violations if "contradiction" in v.get("detail", "").lower()]
        assert len(contradiction) == 0

    def test_self_audit_passed_through_process(self, sentinel):
        """Self-audit field is extracted and used in full process() call."""
        result = sentinel.run({
            "build_result": BuildResult(
                files_changed=["main.py"],
                tests_passed=True,
                diff="+def real_code():\n+    return 42",
                approach_summary="Implemented calculation",
            ),
            "task": Task(
                project_id=uuid.uuid4(),
                title="Add calculation",
                description="Add a calculation function",
            ),
            "self_audit": "All functions fully implemented? YES\nError handling? YES",
        })
        # Clean code + correct self-audit — should pass
        assert result.status == "success"


class TestClaimPatterns:
    def test_patterns_all_have_claims(self):
        for pattern in CLAIM_PATTERNS:
            assert len(pattern["claims"]) > 0
            assert pattern["evidence"]


# ---------------------------------------------------------------------------
# Full Sentinel process tests
# ---------------------------------------------------------------------------


class TestSentinelProcess:
    def test_approved_clean_code(self, sentinel):
        """Clean code with passing tests should be approved."""
        result = sentinel.run({
            "build_result": BuildResult(
                files_changed=["auth.py"],
                tests_passed=True,
                diff="+def authenticate(token):\n+    return validate_jwt(token)",
                approach_summary="Implemented JWT validation",
            ),
            "task": Task(
                project_id=uuid.uuid4(),
                title="Implement JWT auth",
                description="Add JWT token validation",
            ),
        })
        assert result.status == "success"
        verdict = SentinelVerdict(**result.data["sentinel_verdict"])
        assert verdict.approved is True
        assert len(verdict.violations) == 0

    def test_rejected_with_todo(self, sentinel):
        """Code with TODO should be rejected."""
        result = sentinel.run({
            "build_result": BuildResult(
                files_changed=["main.py"],
                tests_passed=True,
                diff="+def handler():\n+    # TODO: implement this\n+    pass",
                approach_summary="Started implementation",
            ),
            "task": Task(
                project_id=uuid.uuid4(),
                title="Implement handler",
                description="Create request handler",
            ),
        })
        assert result.status == "failure"
        verdict = SentinelVerdict(**result.data["sentinel_verdict"])
        assert verdict.approved is False
        assert len(verdict.violations) >= 1

    def test_rejected_banned_dependency(self, sentinel):
        """Importing banned dependency should be rejected."""
        result = sentinel.run({
            "build_result": BuildResult(
                files_changed=["app.py"],
                tests_passed=True,
                diff="+import flask\n+app = flask.Flask(__name__)",
                approach_summary="Built with Flask",
            ),
            "task": Task(
                project_id=uuid.uuid4(),
                title="Build API",
                description="Create REST API",
            ),
        })
        assert result.status == "failure"
        verdict = SentinelVerdict(**result.data["sentinel_verdict"])
        assert any("flask" in v["detail"].lower() for v in verdict.violations)

    def test_multiple_violations(self, sentinel):
        """Multiple issues should all be reported."""
        result = sentinel.run({
            "build_result": BuildResult(
                files_changed=["main.py"],
                tests_passed=False,
                diff="+import flask\n+# TODO: finish\n+from os import *\n+except:\n+    pass",
                approach_summary="Code is production ready and tested",
            ),
            "task": Task(
                project_id=uuid.uuid4(),
                title="Build feature",
                description="Implement new feature",
            ),
        })
        verdict = SentinelVerdict(**result.data["sentinel_verdict"])
        assert verdict.approved is False
        assert len(verdict.violations) >= 3  # Flask, TODO, wildcard, bare except, claims


# ---------------------------------------------------------------------------
# Check 7: LLM Deep Review tests (R5)
# ---------------------------------------------------------------------------


class TestLLMDeepCheck:
    """Test optional LLM deep review (R5)."""

    def test_disabled_by_default(self, sentinel):
        """Deep check should not run when llm_deep_check=False (default)."""
        assert sentinel.llm_deep_check is False
        result = sentinel.run({
            "build_result": BuildResult(
                files_changed=["auth.py"],
                tests_passed=True,
                diff="+def authenticate(token):\n+    return validate_jwt(token)",
                approach_summary="Implemented JWT validation",
            ),
            "task": Task(
                project_id=uuid.uuid4(),
                title="Implement JWT auth",
                description="Add JWT token validation",
            ),
        })
        assert result.status == "success"
        assert result.data.get("tokens_used", 0) == 0

    def test_skipped_when_rule_checks_fail(self, embedding_engine):
        """Deep check should not run if rule-based checks already found violations."""
        from unittest.mock import MagicMock

        llm_client = MagicMock()
        model_router = MagicMock()

        sentinel = Sentinel(
            embedding_engine=embedding_engine,
            banned_dependencies=["flask"],
            llm_deep_check=True,
            llm_client=llm_client,
            model_router=model_router,
        )

        result = sentinel.run({
            "build_result": BuildResult(
                files_changed=["app.py"],
                tests_passed=True,
                diff="+import flask",
                approach_summary="Used Flask",
            ),
            "task": Task(
                project_id=uuid.uuid4(),
                title="Build API",
                description="Create REST API",
            ),
        })
        # Rule-based check fails (banned dep) — LLM should NOT be called
        assert result.status == "failure"
        llm_client.complete.assert_not_called()

    def test_no_llm_client_graceful(self, embedding_engine):
        """Missing LLM client should add recommendation, not crash."""
        sentinel = Sentinel(
            embedding_engine=embedding_engine,
            llm_deep_check=True,
            llm_client=None,
            model_router=None,
        )
        violations, recommendations, tokens = sentinel._check_llm_deep(
            "Task description", "Approach summary", "+code diff"
        )
        assert len(violations) == 0
        assert any("not configured" in r for r in recommendations)
        assert tokens == 0

    def test_empty_diff_skipped(self, embedding_engine):
        """Empty diff should skip deep check."""
        from unittest.mock import MagicMock

        sentinel = Sentinel(
            embedding_engine=embedding_engine,
            llm_deep_check=True,
            llm_client=MagicMock(),
            model_router=MagicMock(),
        )
        violations, recommendations, tokens = sentinel._check_llm_deep(
            "Task", "Approach", ""
        )
        assert len(violations) == 0
        assert any("no diff" in r for r in recommendations)

    def test_parse_pass_response(self, sentinel):
        """PASS verdict from LLM should not produce violations."""
        import json
        result = sentinel._parse_deep_check_response(json.dumps({
            "verdict": "PASS",
            "gaps": [],
            "severity": "none",
        }))
        assert result is not None
        assert result["verdict"] == "PASS"

    def test_parse_fail_response(self, sentinel):
        """FAIL verdict should be parsed correctly."""
        import json
        result = sentinel._parse_deep_check_response(json.dumps({
            "verdict": "FAIL",
            "gaps": ["Missing error handling for null input"],
            "severity": "medium",
        }))
        assert result is not None
        assert result["verdict"] == "FAIL"
        assert len(result["gaps"]) == 1

    def test_parse_json_in_code_fence(self, sentinel):
        """Response wrapped in code fences should be parsed."""
        result = sentinel._parse_deep_check_response(
            '```json\n{"verdict": "PASS", "gaps": [], "severity": "none"}\n```'
        )
        assert result is not None
        assert result["verdict"] == "PASS"

    def test_parse_invalid_json(self, sentinel):
        """Non-JSON response should return None."""
        result = sentinel._parse_deep_check_response("I think the code looks good.")
        assert result is None

    def test_deep_check_fail_adds_violations(self, embedding_engine):
        """LLM FAIL verdict should add violations to the result."""
        import json
        from unittest.mock import MagicMock
        from src.llm.client import LLMResponse

        llm_client = MagicMock()
        llm_client.complete.return_value = LLMResponse(
            content=json.dumps({
                "verdict": "FAIL",
                "gaps": ["Missing null check", "No error handling for timeout"],
                "severity": "high",
            }),
            model="qwen/qwen3-max-thinking",
            tokens_used=300,
        )

        model_router = MagicMock()
        model_router.get_model.return_value = "qwen/qwen3-max-thinking"

        sentinel = Sentinel(
            embedding_engine=embedding_engine,
            llm_deep_check=True,
            llm_client=llm_client,
            model_router=model_router,
        )

        violations, _, tokens = sentinel._check_llm_deep(
            "Fix null handling", "Added null checks", "+def check(x):\n+    return x"
        )
        assert len(violations) == 2
        assert tokens == 300
        assert any("null check" in v["detail"].lower() for v in violations)
        assert all(v["check"] == "llm_deep_review" for v in violations)

    def test_deep_check_pass_no_violations(self, embedding_engine):
        """LLM PASS verdict should not add violations."""
        import json
        from unittest.mock import MagicMock
        from src.llm.client import LLMResponse

        llm_client = MagicMock()
        llm_client.complete.return_value = LLMResponse(
            content=json.dumps({
                "verdict": "PASS",
                "gaps": [],
                "severity": "none",
            }),
            model="qwen/qwen3-max-thinking",
            tokens_used=200,
        )

        model_router = MagicMock()
        model_router.get_model.return_value = "qwen/qwen3-max-thinking"

        sentinel = Sentinel(
            embedding_engine=embedding_engine,
            llm_deep_check=True,
            llm_client=llm_client,
            model_router=model_router,
        )

        violations, _, tokens = sentinel._check_llm_deep(
            "Add calculation", "Added calculation function", "+def calc(x):\n+    return x * 2"
        )
        assert len(violations) == 0
        assert tokens == 200
