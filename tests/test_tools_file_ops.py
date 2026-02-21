"""Tests for src/tools/file_ops.py — safe file operations."""

import hashlib
from pathlib import Path

import pytest

from src.core.exceptions import ToolError
from src.security.policy import SecurityPolicy
from src.tools.file_ops import file_hash, list_files, read_file, write_file


class TestReadFile:
    def test_reads_existing_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world", encoding="utf-8")
        assert read_file(str(f)) == "hello world"

    def test_reads_with_path_object(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("content", encoding="utf-8")
        assert read_file(f) == "content"

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(ToolError, match="File not found"):
            read_file(str(tmp_path / "nonexistent.txt"))

    def test_reads_utf8(self, tmp_path):
        f = tmp_path / "unicode.txt"
        f.write_text("café ñ 日本語", encoding="utf-8")
        assert read_file(str(f)) == "café ñ 日本語"


class TestWriteFile:
    def test_writes_new_file(self, tmp_path):
        target = tmp_path / "output.txt"
        result = write_file(str(target), "new content")
        assert result == target
        assert target.read_text(encoding="utf-8") == "new content"

    def test_creates_parent_directories(self, tmp_path):
        target = tmp_path / "deep" / "nested" / "file.txt"
        write_file(str(target), "nested content")
        assert target.exists()
        assert target.read_text(encoding="utf-8") == "nested content"

    def test_creates_backup_on_overwrite(self, tmp_path):
        target = tmp_path / "existing.py"
        target.write_text("original", encoding="utf-8")

        write_file(str(target), "updated", backup=True)

        assert target.read_text(encoding="utf-8") == "updated"
        backup = tmp_path / "existing.py.bak"
        assert backup.exists()
        assert backup.read_text(encoding="utf-8") == "original"

    def test_no_backup_when_disabled(self, tmp_path):
        target = tmp_path / "existing.py"
        target.write_text("original", encoding="utf-8")

        write_file(str(target), "updated", backup=False)

        assert target.read_text(encoding="utf-8") == "updated"
        assert not (tmp_path / "existing.py.bak").exists()

    def test_no_backup_for_new_file(self, tmp_path):
        target = tmp_path / "brand_new.py"
        write_file(str(target), "content", backup=True)
        assert not (tmp_path / "brand_new.py.bak").exists()


class TestListFiles:
    def test_lists_all_files(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        (tmp_path / "c.txt").write_text("c")

        result = list_files(str(tmp_path))
        assert len(result) == 3

    def test_lists_with_pattern(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        (tmp_path / "c.txt").write_text("c")

        result = list_files(str(tmp_path), pattern="*.py")
        assert len(result) == 2
        assert all(p.suffix == ".py" for p in result)

    def test_recursive_search(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (tmp_path / "top.py").write_text("top")
        (sub / "nested.py").write_text("nested")

        flat = list_files(str(tmp_path), pattern="*.py", recursive=False)
        assert len(flat) == 1

        deep = list_files(str(tmp_path), pattern="*.py", recursive=True)
        assert len(deep) == 2

    def test_returns_empty_for_nonexistent_dir(self):
        result = list_files("/nonexistent/directory")
        assert result == []


class TestFileHash:
    def test_hashes_file(self, tmp_path):
        f = tmp_path / "test.txt"
        content = "hash me"
        f.write_text(content, encoding="utf-8")

        expected = hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        assert file_hash(str(f)) == expected

    def test_returns_none_for_missing_file(self):
        assert file_hash("/nonexistent/file.txt") is None

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content A")
        f2.write_text("content B")

        assert file_hash(str(f1)) != file_hash(str(f2))

    def test_same_content_same_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("same content")
        f2.write_text("same content")

        assert file_hash(str(f1)) == file_hash(str(f2))


class TestFileOpsSecurityPolicy:
    def test_read_blocks_outside_workspace(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("secret", encoding="utf-8")

        policy = SecurityPolicy(workspace_dir=workspace)
        with pytest.raises(ToolError, match="(not allowed|escapes workspace)"):
            read_file(str(outside), security_policy=policy)

    def test_write_blocks_path_traversal(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        policy = SecurityPolicy(workspace_dir=workspace)

        with pytest.raises(ToolError, match="not allowed"):
            write_file("../evil.txt", "bad", security_policy=policy)

    @pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="symlink not supported")
    def test_write_blocks_symlink_target(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("outside", encoding="utf-8")
        symlink = workspace / "link.txt"
        symlink.symlink_to(outside)

        policy = SecurityPolicy(workspace_dir=workspace)
        with pytest.raises(ToolError, match="symlink"):
            write_file("link.txt", "new", security_policy=policy)

    @pytest.mark.skipif(not hasattr(Path, "symlink_to"), reason="symlink not supported")
    def test_write_blocks_symlink_no_policy(self, tmp_path):
        """write_file with backup=True refuses to write through symlinks even without policy."""
        target = tmp_path / "real.txt"
        target.write_text("original", encoding="utf-8")
        symlink = tmp_path / "link.txt"
        symlink.symlink_to(target)

        with pytest.raises(ToolError, match="symlink"):
            write_file(str(symlink), "evil", backup=True)

    def test_list_files_with_security_policy(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "a.py").write_text("a")
        policy = SecurityPolicy(workspace_dir=workspace)
        result = list_files(".", security_policy=policy)
        assert len(result) >= 1

    def test_read_file_with_security_policy(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        f = workspace / "test.txt"
        f.write_text("secured", encoding="utf-8")
        policy = SecurityPolicy(workspace_dir=workspace)
        content = read_file("test.txt", security_policy=policy)
        assert content == "secured"

    def test_read_file_nonexistent_with_security_policy(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        policy = SecurityPolicy(workspace_dir=workspace)
        with pytest.raises(ToolError, match="File not found"):
            read_file("nope.txt", security_policy=policy)

    def test_write_file_with_security_policy(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        policy = SecurityPolicy(workspace_dir=workspace)
        result = write_file("new.txt", "content", security_policy=policy)
        assert result.exists()
        assert result.read_text(encoding="utf-8") == "content"

    def test_list_files_nonexistent_dir_with_policy(self, tmp_path):
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        policy = SecurityPolicy(workspace_dir=workspace)
        result = list_files("nope", security_policy=policy)
        assert result == []
