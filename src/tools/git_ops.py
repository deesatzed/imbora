"""Git operations for The Associate.

Provides checkpoint/rewind via git stash, diff retrieval, and commit.
State Manager uses checkpoint() before Builder writes code, and rewind()
when tests fail to restore the working tree.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.core.exceptions import GitOperationError
from src.tools.shell import ShellResult, run_command

logger = logging.getLogger("associate.tools.git_ops")


def checkpoint(repo_path: str, message: str = "associate-checkpoint") -> str:
    """Save current working tree state via git stash.

    Args:
        repo_path: Path to the git repository.
        message: Stash message for identification.

    Returns:
        Git stash reference (e.g., "stash@{0}").

    Raises:
        GitOperationError: If stash fails.
    """
    # Include untracked files in the stash (list-form to prevent shell injection)
    result = run_command(
        ["git", "stash", "push", "--include-untracked", "-m", message],
        cwd=repo_path,
    )
    if not result.success:
        raise GitOperationError(f"git stash failed: {result.stderr}")

    # Check if anything was actually stashed
    if "No local changes" in result.stdout:
        logger.info("No changes to checkpoint")
        return ""

    # Get the stash reference
    ref_result = run_command(["git", "stash", "list", "--max-count=1"], cwd=repo_path)
    ref = ref_result.stdout.strip().split(":")[0] if ref_result.stdout.strip() else "stash@{0}"
    logger.info("Checkpoint created: %s", ref)
    return ref


def rewind(repo_path: str, stash_ref: Optional[str] = None) -> bool:
    """Restore working tree from a git stash.

    Args:
        repo_path: Path to the git repository.
        stash_ref: Specific stash reference (default: latest stash).

    Returns:
        True if rewind succeeded, False if nothing to rewind.

    Raises:
        GitOperationError: If stash pop fails.
    """
    if stash_ref and not stash_ref.strip():
        logger.info("No stash ref to rewind — nothing was checkpointed")
        return False

    cmd = ["git", "stash", "pop"] + ([stash_ref] if stash_ref else [])
    result = run_command(cmd, cwd=repo_path)

    if "No stash entries" in result.stderr or "No stash entries" in result.stdout:
        logger.info("No stash entries to rewind")
        return False

    if not result.success:
        raise GitOperationError(f"git stash pop failed: {result.stderr}")

    logger.info("Rewound to checkpoint: %s", stash_ref or "latest")
    return True


def get_diff(repo_path: str, staged: bool = False) -> str:
    """Get the current diff.

    Args:
        repo_path: Path to the git repository.
        staged: If True, show staged changes. If False, show unstaged.

    Returns:
        Diff text.
    """
    cmd = ["git", "diff", "--staged"] if staged else ["git", "diff"]
    result = run_command(cmd, cwd=repo_path)
    return result.stdout


def get_full_diff(repo_path: str) -> str:
    """Get both staged and unstaged changes combined."""
    staged = get_diff(repo_path, staged=True)
    unstaged = get_diff(repo_path, staged=False)
    parts = []
    if staged:
        parts.append(staged)
    if unstaged:
        parts.append(unstaged)
    return "\n".join(parts)


def commit(repo_path: str, message: str, files: Optional[list[str]] = None) -> str:
    """Stage files and create a git commit.

    Args:
        repo_path: Path to the git repository.
        message: Commit message.
        files: Specific files to stage. If None, stages all changes.

    Returns:
        Commit hash.

    Raises:
        GitOperationError: If commit fails.
    """
    # Stage files (list-form to prevent shell injection)
    if files:
        for f in files:
            result = run_command(["git", "add", f], cwd=repo_path)
            if not result.success:
                raise GitOperationError(f"git add failed for {f}: {result.stderr}")
    else:
        result = run_command(["git", "add", "-A"], cwd=repo_path)
        if not result.success:
            raise GitOperationError(f"git add -A failed: {result.stderr}")

    # Commit (list-form to prevent shell injection via message)
    result = run_command(["git", "commit", "-m", message], cwd=repo_path)
    if not result.success:
        if "nothing to commit" in result.stdout:
            logger.info("Nothing to commit")
            return ""
        raise GitOperationError(f"git commit failed: {result.stderr}")

    # Get commit hash
    hash_result = run_command(["git", "rev-parse", "HEAD"], cwd=repo_path)
    commit_hash = hash_result.stdout.strip()
    logger.info("Committed: %s", commit_hash[:8])
    return commit_hash


def get_status(repo_path: str) -> str:
    """Get git status output."""
    result = run_command(["git", "status", "--short"], cwd=repo_path)
    return result.stdout


def is_git_repo(path: str) -> bool:
    """Check if the given path is inside a git repository."""
    result = run_command(["git", "rev-parse", "--is-inside-work-tree"], cwd=path)
    return result.success and result.stdout.strip() == "true"
