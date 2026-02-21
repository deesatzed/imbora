"""Tests for src/tools/git_ops.py — git operations.

Uses real git repos created in tmp_path — no mocks.
"""

import pytest

from src.core.exceptions import GitOperationError
from src.tools.git_ops import (
    checkpoint,
    commit,
    get_diff,
    get_full_diff,
    get_status,
    is_git_repo,
    rewind,
)
from src.tools.shell import run_command


def _init_git_repo(path):
    """Initialize a fresh git repo with an initial commit."""
    run_command("git init", cwd=str(path))
    run_command("git config user.email 'test@test.com'", cwd=str(path))
    run_command("git config user.name 'Test'", cwd=str(path))
    (path / "initial.txt").write_text("initial content")
    run_command("git add -A", cwd=str(path))
    run_command("git commit -m 'initial commit'", cwd=str(path))


class TestIsGitRepo:
    def test_true_for_git_repo(self, tmp_path):
        _init_git_repo(tmp_path)
        assert is_git_repo(str(tmp_path)) is True

    def test_false_for_non_repo(self, tmp_path):
        assert is_git_repo(str(tmp_path)) is False


class TestCheckpointAndRewind:
    def test_checkpoint_creates_stash(self, tmp_path):
        _init_git_repo(tmp_path)
        # Make a change
        (tmp_path / "change.txt").write_text("new file")
        ref = checkpoint(str(tmp_path))
        assert ref  # Should have a stash reference
        # File should be gone after stash
        assert not (tmp_path / "change.txt").exists()

    def test_checkpoint_no_changes(self, tmp_path):
        _init_git_repo(tmp_path)
        ref = checkpoint(str(tmp_path))
        assert ref == ""  # Nothing to stash

    def test_rewind_restores_stash(self, tmp_path):
        _init_git_repo(tmp_path)
        # Make and checkpoint a change
        (tmp_path / "change.txt").write_text("stashed content")
        ref = checkpoint(str(tmp_path))
        assert not (tmp_path / "change.txt").exists()

        # Rewind should restore it
        success = rewind(str(tmp_path), ref)
        assert success is True
        assert (tmp_path / "change.txt").exists()
        assert (tmp_path / "change.txt").read_text() == "stashed content"

    def test_rewind_no_stash(self, tmp_path):
        _init_git_repo(tmp_path)
        success = rewind(str(tmp_path))
        assert success is False

    def test_rewind_empty_ref(self, tmp_path):
        _init_git_repo(tmp_path)
        success = rewind(str(tmp_path), "")
        assert success is False

    def test_checkpoint_rewind_cycle(self, tmp_path):
        """Full cycle: modify → checkpoint → verify gone → rewind → verify restored."""
        _init_git_repo(tmp_path)

        # Create two files
        (tmp_path / "file_a.txt").write_text("content A")
        (tmp_path / "file_b.txt").write_text("content B")

        # Checkpoint
        ref = checkpoint(str(tmp_path), message="test-checkpoint")
        assert ref
        assert not (tmp_path / "file_a.txt").exists()
        assert not (tmp_path / "file_b.txt").exists()

        # Rewind
        rewind(str(tmp_path), ref)
        assert (tmp_path / "file_a.txt").read_text() == "content A"
        assert (tmp_path / "file_b.txt").read_text() == "content B"


class TestGetDiff:
    def test_unstaged_diff(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "initial.txt").write_text("modified content")

        diff = get_diff(str(tmp_path), staged=False)
        assert "modified content" in diff
        assert "initial content" in diff

    def test_staged_diff(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "initial.txt").write_text("staged change")
        run_command("git add initial.txt", cwd=str(tmp_path))

        diff = get_diff(str(tmp_path), staged=True)
        assert "staged change" in diff

    def test_no_changes_empty_diff(self, tmp_path):
        _init_git_repo(tmp_path)
        diff = get_diff(str(tmp_path))
        assert diff.strip() == ""


class TestGetFullDiff:
    def test_combines_staged_and_unstaged(self, tmp_path):
        _init_git_repo(tmp_path)

        # Staged change
        (tmp_path / "initial.txt").write_text("staged")
        run_command("git add initial.txt", cwd=str(tmp_path))

        # Unstaged change (new file)
        (tmp_path / "new.txt").write_text("unstaged")
        run_command("git add new.txt", cwd=str(tmp_path))
        (tmp_path / "new.txt").write_text("unstaged modified")

        diff = get_full_diff(str(tmp_path))
        assert len(diff) > 0


class TestCommit:
    def test_commit_all(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "new_file.py").write_text("print('hello')")

        commit_hash = commit(str(tmp_path), "test commit")
        assert len(commit_hash) == 40  # Full SHA

    def test_commit_specific_files(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "included.py").write_text("included")
        (tmp_path / "excluded.py").write_text("excluded")

        commit_hash = commit(str(tmp_path), "partial commit", files=["included.py"])
        assert commit_hash

        # excluded.py should still be untracked
        status = get_status(str(tmp_path))
        assert "excluded.py" in status

    def test_nothing_to_commit(self, tmp_path):
        _init_git_repo(tmp_path)
        result = commit(str(tmp_path), "empty commit")
        assert result == ""


class TestGetStatus:
    def test_clean_repo(self, tmp_path):
        _init_git_repo(tmp_path)
        status = get_status(str(tmp_path))
        assert status.strip() == ""

    def test_modified_file(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "initial.txt").write_text("modified")
        status = get_status(str(tmp_path))
        assert "initial.txt" in status

    def test_new_file(self, tmp_path):
        _init_git_repo(tmp_path)
        (tmp_path / "new.txt").write_text("new")
        status = get_status(str(tmp_path))
        assert "new.txt" in status
