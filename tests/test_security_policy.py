"""Tests for src/security/policy.py."""

from pathlib import Path

from src.security.policy import ActionTracker, AutonomyLevel, SecurityPolicy


class TestActionTracker:
    def test_record_and_count(self):
        tracker = ActionTracker(window_seconds=3600)
        assert tracker.count() == 0
        assert tracker.record() == 1
        assert tracker.count() == 1


class TestSecurityPolicyCommands:
    def test_allows_configured_commands(self, tmp_path):
        policy = SecurityPolicy(workspace_dir=tmp_path, allowed_commands=["echo", "wc"])
        assert policy.is_command_allowed("echo hello | wc -c")

    def test_blocks_subshell_and_redirect(self, tmp_path):
        policy = SecurityPolicy(workspace_dir=tmp_path, allowed_commands=["echo"])
        assert not policy.is_command_allowed("echo $(uname)")
        assert not policy.is_command_allowed("echo hello > out.txt")

    def test_blocks_unlisted_command(self, tmp_path):
        policy = SecurityPolicy(workspace_dir=tmp_path, allowed_commands=["echo"])
        assert not policy.is_command_allowed("ls")

    def test_readonly_cannot_act(self, tmp_path):
        policy = SecurityPolicy(
            workspace_dir=tmp_path,
            autonomy=AutonomyLevel.READ_ONLY,
            allowed_commands=["echo"],
        )
        assert not policy.is_command_allowed("echo hi")


class TestSecurityPolicyPaths:
    def test_blocks_traversal_but_allows_absolute_workspace_targets(self, tmp_path):
        policy = SecurityPolicy(workspace_dir=tmp_path, workspace_only=True)
        assert not policy.is_path_allowed("../etc/passwd")
        assert policy.is_path_allowed(str((tmp_path / "file.txt").resolve()))

    def test_resolved_path_boundary(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        policy = SecurityPolicy(workspace_dir=workspace)

        inside = (workspace / "a.txt").resolve()
        outside = (tmp_path / "outside.txt").resolve()

        assert policy.is_resolved_path_allowed(inside)
        assert not policy.is_resolved_path_allowed(outside)

    def test_forbidden_path_is_blocked(self, tmp_path):
        policy = SecurityPolicy(
            workspace_dir=tmp_path,
            workspace_only=False,
            forbidden_paths=["/etc", "~/.ssh"],
        )
        assert not policy.is_path_allowed("/etc/passwd")
        assert not policy.is_path_allowed(str(Path("~/.ssh/id_rsa").expanduser()))
