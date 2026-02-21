"""Tests for src/core/exceptions.py — exception hierarchy."""

import pytest

from src.core.exceptions import (
    AgentError,
    AssociateError,
    AuthenticationError,
    CheckpointError,
    ConfigError,
    ConnectionError,
    CouncilExhaustionError,
    DatabaseError,
    GitOperationError,
    LLMError,
    ModelNotFoundError,
    RateLimitError,
    ResponseParseError,
    SchemaInitError,
    SearchError,
    SentinelRejectionError,
    ShellTimeoutError,
    ToolError,
)


class TestExceptionHierarchy:
    def test_base_exception(self):
        with pytest.raises(AssociateError):
            raise AssociateError("test")

    def test_database_errors_inherit_from_associate(self):
        assert issubclass(DatabaseError, AssociateError)
        assert issubclass(SchemaInitError, DatabaseError)
        assert issubclass(ConnectionError, DatabaseError)

    def test_llm_errors_inherit_from_associate(self):
        assert issubclass(LLMError, AssociateError)
        assert issubclass(RateLimitError, LLMError)
        assert issubclass(AuthenticationError, LLMError)
        assert issubclass(ModelNotFoundError, LLMError)
        assert issubclass(ResponseParseError, LLMError)

    def test_agent_errors_inherit_from_associate(self):
        assert issubclass(AgentError, AssociateError)
        assert issubclass(CheckpointError, AgentError)
        assert issubclass(SentinelRejectionError, AgentError)
        assert issubclass(CouncilExhaustionError, AgentError)

    def test_tool_errors_inherit_from_associate(self):
        assert issubclass(ToolError, AssociateError)
        assert issubclass(ShellTimeoutError, ToolError)
        assert issubclass(GitOperationError, ToolError)
        assert issubclass(SearchError, ToolError)

    def test_config_error_inherits_from_associate(self):
        assert issubclass(ConfigError, AssociateError)


class TestSentinelRejectionError:
    def test_violations_stored(self):
        violations = [
            {"check": "dependency_jail", "detail": "flask is banned"},
            {"check": "placeholder_scan", "detail": "# TODO found"},
        ]
        err = SentinelRejectionError(violations=violations)
        assert err.violations == violations
        assert len(err.violations) == 2
        assert str(err) == "Sentinel rejected build"

    def test_custom_message(self):
        err = SentinelRejectionError(violations=[], message="Custom rejection")
        assert str(err) == "Custom rejection"


class TestCouncilExhaustionError:
    def test_attributes(self):
        err = CouncilExhaustionError(task_id="task-123", council_count=3)
        assert err.task_id == "task-123"
        assert err.council_count == 3
        assert "task-123" in str(err)
        assert "3" in str(err)


class TestExceptionCatching:
    def test_catch_broad(self):
        """Callers can catch all Associate errors broadly."""
        with pytest.raises(AssociateError):
            raise RateLimitError("too many requests")

    def test_catch_narrow(self):
        """Callers can catch specific error types."""
        with pytest.raises(RateLimitError):
            raise RateLimitError("too many requests")

    def test_database_catch_broad(self):
        with pytest.raises(DatabaseError):
            raise SchemaInitError("missing table")
