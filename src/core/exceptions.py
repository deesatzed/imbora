"""Custom exception hierarchy for The Associate.

All exceptions inherit from AssociateError so callers can catch broadly
or narrowly as needed.
"""


class AssociateError(Exception):
    """Base exception for all Associate errors."""


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

class DatabaseError(AssociateError):
    """Failed database operation."""


class SchemaInitError(DatabaseError):
    """Failed to initialize database schema."""


class ConnectionError(DatabaseError):
    """Failed to connect to database."""


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

class LLMError(AssociateError):
    """Failed LLM operation."""


class RateLimitError(LLMError):
    """Hit API rate limit."""


class AuthenticationError(LLMError):
    """Invalid API key or unauthorized."""


class ModelNotFoundError(LLMError):
    """Requested model not available."""


class ResponseParseError(LLMError):
    """Failed to parse LLM response."""


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

class AgentError(AssociateError):
    """Agent processing failure."""


class CheckpointError(AgentError):
    """Failed to create or restore git checkpoint."""


class SentinelRejectionError(AgentError):
    """Sentinel rejected the build output."""

    def __init__(self, violations: list[dict[str, str]], message: str = "Sentinel rejected build"):
        self.violations = violations
        super().__init__(message)


class CouncilExhaustionError(AgentError):
    """Council invoked too many times — task is STUCK."""

    def __init__(self, task_id: str, council_count: int):
        self.task_id = task_id
        self.council_count = council_count
        super().__init__(f"Task {task_id} stuck after {council_count} council invocations")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

class ToolError(AssociateError):
    """Tool execution failure."""


class ShellTimeoutError(ToolError):
    """Shell command exceeded timeout."""


class GitOperationError(ToolError):
    """Git operation failed."""


class SearchError(ToolError):
    """Web search API failure."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class ConfigError(AssociateError):
    """Invalid or missing configuration."""
