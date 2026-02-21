"""Tests for src/tools/shell.py — subprocess execution."""

import pytest

from src.core.exceptions import ShellTimeoutError, ToolError
from src.security.policy import AutonomyLevel, SecurityPolicy
from src.tools.shell import ShellResult, _truncate_output, run_command


class TestShellResult:
    def test_success_property(self):
        result = ShellResult(command="echo hi", return_code=0, stdout="hi\n", stderr="")
        assert result.success is True

    def test_failure_property(self):
        result = ShellResult(command="false", return_code=1, stdout="", stderr="error")
        assert result.success is False

    def test_timeout_property(self):
        result = ShellResult(command="sleep", return_code=0, stdout="", stderr="", timed_out=True)
        assert result.success is False


class TestRunCommand:
    def test_echo_command(self):
        result = run_command("echo hello")
        assert result.success
        assert result.stdout.strip() == "hello"
        assert result.return_code == 0

    def test_captures_stderr(self):
        result = run_command("echo error >&2")
        assert "error" in result.stderr

    def test_captures_return_code(self):
        result = run_command("exit 42")
        assert result.return_code == 42
        assert result.success is False

    def test_command_string(self):
        result = run_command("echo hello")
        assert result.command == "echo hello"

    def test_command_list(self):
        result = run_command(["echo", "hello"])
        assert result.success
        assert "hello" in result.stdout

    def test_cwd_parameter(self, tmp_path):
        result = run_command("pwd", cwd=str(tmp_path))
        assert result.success
        assert str(tmp_path) in result.stdout

    def test_timeout_raises(self):
        with pytest.raises(ShellTimeoutError, match="timed out"):
            run_command("sleep 10", timeout=1)

    def test_nonexistent_command_raises(self):
        with pytest.raises(ToolError, match="Command not found"):
            run_command(["completely_nonexistent_binary_xyz"])

    def test_multiline_output(self):
        result = run_command("echo 'line1\nline2\nline3'")
        assert result.success
        lines = result.stdout.strip().split("\n")
        assert len(lines) == 3

    def test_env_parameter(self):
        import os
        env = os.environ.copy()
        env["TEST_VAR_XYZ"] = "test_value_123"
        result = run_command("echo $TEST_VAR_XYZ", env=env)
        assert "test_value_123" in result.stdout

    def test_pipe_commands(self):
        result = run_command("echo 'hello world' | wc -w")
        assert result.success
        assert result.stdout.strip() == "2"

    def test_security_policy_blocks_disallowed_command(self, tmp_path):
        policy = SecurityPolicy(
            workspace_dir=tmp_path,
            allowed_commands=["echo"],
        )
        with pytest.raises(ToolError, match="not allowed"):
            run_command("ls", security_policy=policy)

    def test_security_policy_readonly_blocks_commands(self, tmp_path):
        policy = SecurityPolicy(
            workspace_dir=tmp_path,
            autonomy=AutonomyLevel.READ_ONLY,
            allowed_commands=["echo"],
        )
        with pytest.raises(ToolError, match="not allowed"):
            run_command("echo hello", security_policy=policy)

    def test_security_policy_blocks_cwd_escape(self, tmp_path):
        policy = SecurityPolicy(
            workspace_dir=tmp_path,
            allowed_commands=["pwd"],
            workspace_only=True,
        )
        with pytest.raises(ToolError, match="escapes workspace"):
            run_command("pwd", cwd="/tmp", security_policy=policy)

    def test_security_policy_sanitizes_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("API_KEY", "secret-value-123")
        policy = SecurityPolicy(
            workspace_dir=tmp_path,
            allowed_commands=["env"],
            safe_env_vars=["PATH"],
            sanitize_env=True,
        )
        result = run_command("env", security_policy=policy)
        assert result.success
        assert "secret-value-123" not in result.stdout

    def test_security_policy_rate_limited(self, tmp_path):
        policy = SecurityPolicy(
            workspace_dir=tmp_path,
            allowed_commands=["echo"],
            max_actions_per_hour=0,
        )
        with pytest.raises(ToolError, match="rate limit"):
            run_command("echo hi", security_policy=policy)

    def test_security_policy_list_cmd_empty(self, tmp_path):
        policy = SecurityPolicy(
            workspace_dir=tmp_path,
            allowed_commands=["echo"],
        )
        with pytest.raises(ToolError, match="empty command"):
            run_command([], security_policy=policy)

    def test_security_policy_list_cmd_readonly_mode(self, tmp_path):
        policy = SecurityPolicy(
            workspace_dir=tmp_path,
            autonomy=AutonomyLevel.READ_ONLY,
            allowed_commands=["echo"],
        )
        with pytest.raises(ToolError, match="read-only"):
            run_command(["echo", "hi"], security_policy=policy)

    def test_security_policy_list_cmd_not_allowed(self, tmp_path):
        policy = SecurityPolicy(
            workspace_dir=tmp_path,
            allowed_commands=["echo"],
        )
        with pytest.raises(ToolError, match="not allowed"):
            run_command(["rm", "-rf", "/"], security_policy=policy)

    def test_security_policy_action_budget_exceeded(self, tmp_path):
        policy = SecurityPolicy(
            workspace_dir=tmp_path,
            allowed_commands=["echo"],
            max_actions_per_hour=1,
        )
        # First call uses the budget
        run_command("echo first", security_policy=policy)
        # Second call should exceed budget
        with pytest.raises(ToolError, match="(budget exceeded|rate limit)"):
            run_command("echo second", security_policy=policy)


class TestTruncateOutput:
    def test_short_output_unchanged(self):
        assert _truncate_output("hello") == "hello"

    def test_long_output_truncated(self):
        big = "x" * 2_000_000
        result = _truncate_output(big)
        assert len(result.encode("utf-8")) <= 1_048_576 + 100
        assert "[output truncated]" in result
