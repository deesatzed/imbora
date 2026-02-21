"""Safe file operations for The Associate.

Provides read/write with automatic backup creation so that Builder's
code writes can be reversed by State Manager's rewind.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Optional

from src.core.exceptions import ToolError
from src.security.policy import SecurityPolicy

logger = logging.getLogger("associate.tools.file_ops")


def read_file(path: str | Path, security_policy: SecurityPolicy | None = None) -> str:
    """Read a file and return its contents.

    Args:
        path: Absolute or relative file path.

    Returns:
        File contents as string.

    Raises:
        ToolError: If file doesn't exist or can't be read.
    """
    p = Path(path)
    if security_policy is not None:
        p = _resolve_read_path(path, security_policy)

    if not p.exists():
        raise ToolError(f"File not found: {p}")
    try:
        return p.read_text(encoding="utf-8")
    except Exception as e:
        raise ToolError(f"Failed to read {p}: {e}") from e


def write_file(
    path: str | Path,
    content: str,
    backup: bool = True,
    security_policy: SecurityPolicy | None = None,
) -> Path:
    """Write content to a file, optionally creating a backup.

    Args:
        path: Target file path.
        content: Content to write.
        backup: If True and file exists, copy to path.bak first.

    Returns:
        The Path that was written.

    Raises:
        ToolError: If write fails.
    """
    p = Path(path)
    try:
        if security_policy is not None:
            p = _resolve_write_path(path, security_policy)

        p.parent.mkdir(parents=True, exist_ok=True)

        if backup and p.exists():
            if p.is_symlink():
                raise ToolError(f"Refusing to write through symlink: {p}")
            backup_path = p.with_suffix(p.suffix + ".bak")
            shutil.copy2(p, backup_path)
            logger.debug("Backup created: %s", backup_path)

        p.write_text(content, encoding="utf-8")
        logger.debug("Wrote %d bytes to %s", len(content), p)
        return p
    except Exception as e:
        raise ToolError(f"Failed to write {p}: {e}") from e


def list_files(
    directory: str | Path,
    pattern: str = "*",
    recursive: bool = False,
    security_policy: SecurityPolicy | None = None,
) -> list[Path]:
    """List files in a directory matching a glob pattern.

    Args:
        directory: Directory to search.
        pattern: Glob pattern (e.g., "*.py").
        recursive: If True, search subdirectories.

    Returns:
        List of matching file paths.
    """
    d = Path(directory)
    if security_policy is not None:
        d = _resolve_list_directory(directory, security_policy)

    if not d.is_dir():
        return []
    if recursive:
        return sorted(d.rglob(pattern))
    return sorted(d.glob(pattern))


def file_hash(path: str | Path) -> Optional[str]:
    """Compute a quick hash of file contents for manifest comparison."""
    import hashlib

    p = Path(path)
    if not p.exists():
        return None
    try:
        content = p.read_bytes()
        return hashlib.sha256(content).hexdigest()[:16]
    except Exception:
        return None


def _resolve_read_path(path: str | Path, security_policy: SecurityPolicy) -> Path:
    path_str = str(path)
    if not security_policy.is_path_allowed(path_str):
        raise ToolError(f"Path not allowed by security policy: {path_str}")

    candidate = security_policy.resolved_target(path)
    if not candidate.exists():
        return candidate

    resolved = candidate.resolve()
    if not security_policy.is_resolved_path_allowed(resolved):
        raise ToolError(f"Resolved path escapes workspace: {resolved}")

    return resolved


def _resolve_write_path(path: str | Path, security_policy: SecurityPolicy) -> Path:
    path_str = str(path)
    if not security_policy.is_path_allowed(path_str):
        raise ToolError(f"Path not allowed by security policy: {path_str}")

    candidate = security_policy.resolved_target(path)
    parent = candidate.parent
    parent.mkdir(parents=True, exist_ok=True)

    resolved_parent = parent.resolve()
    if not security_policy.is_resolved_path_allowed(resolved_parent):
        raise ToolError(f"Resolved path escapes workspace: {resolved_parent}")

    target = resolved_parent / candidate.name
    if target.exists() and target.is_symlink():
        raise ToolError(f"Refusing to write through symlink: {target}")

    return target


def _resolve_list_directory(directory: str | Path, security_policy: SecurityPolicy) -> Path:
    directory_str = str(directory)
    if not security_policy.is_path_allowed(directory_str):
        raise ToolError(f"Path not allowed by security policy: {directory_str}")

    resolved = security_policy.resolved_target(directory)
    if not resolved.exists():
        return resolved

    resolved = resolved.resolve()
    if not security_policy.is_resolved_path_allowed(resolved):
        raise ToolError(f"Resolved path escapes workspace: {resolved}")

    return resolved
