"""Shell command execution for The Associate.

Runs subprocesses with timeout, captures stdout/stderr, and provides
structured results for Builder's test execution and git operations.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from src.core.exceptions import ShellTimeoutError, ToolError
from src.security.policy import SecurityPolicy

logger = logging.getLogger("associate.tools.shell")

DEFAULT_TIMEOUT = 120  # seconds
MAX_OUTPUT_BYTES = 1_048_576


@dataclass
class ShellResult:
    """Structured result from a shell command."""
    command: str
    return_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.return_code == 0 and not self.timed_out


def run_command(
    command: str | list[str],
    cwd: Optional[str] = None,
    timeout: int = DEFAULT_TIMEOUT,
    env: Optional[dict[str, str]] = None,
    security_policy: Optional[SecurityPolicy] = None,
) -> ShellResult:
    """Execute a shell command with timeout and output capture.

    Args:
        command: Command string or list of args.
        cwd: Working directory for the command.
        timeout: Max seconds before killing the process.
        env: Optional environment variables (merged with current env).
        security_policy: Optional execution policy (allowlist, workspace scoping).

    Returns:
        ShellResult with return code, stdout, stderr.

    Raises:
        ShellTimeoutError: If command exceeds timeout.
        ToolError: If command can't be started.
    """
    cmd_str = command if isinstance(command, str) else " ".join(command)
    logger.debug("Running: %s (cwd=%s, timeout=%ds)", cmd_str, cwd, timeout)

    try:
        if security_policy is not None:
            _enforce_policy(command=command, cwd=cwd, security_policy=security_policy)

        run_env = _build_env(env=env, security_policy=security_policy)

        result = subprocess.run(
            command,
            shell=isinstance(command, str),
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env,
        )

        stdout = _truncate_output(result.stdout)
        stderr = _truncate_output(result.stderr)
        shell_result = ShellResult(
            command=cmd_str,
            return_code=result.returncode,
            stdout=stdout,
            stderr=stderr,
        )
        logger.debug(
            "Command finished: rc=%d stdout=%d chars stderr=%d chars",
            result.returncode, len(stdout), len(stderr),
        )
        return shell_result

    except subprocess.TimeoutExpired:
        logger.warning("Command timed out after %ds: %s", timeout, cmd_str)
        raise ShellTimeoutError(f"Command timed out after {timeout}s: {cmd_str}")

    except FileNotFoundError as e:
        raise ToolError(f"Command not found: {e}") from e

    except Exception as e:
        raise ToolError(f"Failed to run command: {e}") from e


def _enforce_policy(
    command: str | list[str],
    cwd: Optional[str],
    security_policy: SecurityPolicy,
) -> None:
    if security_policy.is_rate_limited():
        raise ToolError("Security policy rate limit exceeded")

    if isinstance(command, str):
        if not security_policy.is_command_allowed(command):
            raise ToolError(f"Command not allowed by security policy: {command}")
    else:
        if not command:
            raise ToolError("Command not allowed by security policy: empty command")
        if not security_policy.can_act():
            raise ToolError("Command not allowed by security policy: read-only mode")
        base_cmd = Path(command[0]).name
        if base_cmd not in security_policy.allowed_commands:
            raise ToolError(f"Command not allowed by security policy: {base_cmd}")

    if cwd:
        resolved_cwd = Path(cwd).resolve()
        if security_policy.workspace_only and not security_policy.is_resolved_path_allowed(resolved_cwd):
            raise ToolError(f"Working directory escapes workspace: {resolved_cwd}")

    if not security_policy.record_action():
        raise ToolError("Security policy action budget exceeded")


def _truncate_output(text: str) -> str:
    if len(text.encode("utf-8")) <= MAX_OUTPUT_BYTES:
        return text

    encoded = text.encode("utf-8")[:MAX_OUTPUT_BYTES]
    truncated = encoded.decode("utf-8", errors="ignore")
    return truncated + "\n... [output truncated]"


def _build_env(
    env: Optional[dict[str, str]],
    security_policy: Optional[SecurityPolicy],
) -> dict[str, str]:
    if security_policy is None:
        base_env = dict(os.environ)
    else:
        base_env = security_policy.build_subprocess_env()

    if env:
        base_env.update(env)

    return base_env
