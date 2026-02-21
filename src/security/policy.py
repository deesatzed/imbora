"""Security policy adapted from ZeroClaw for tool sandboxing.

Provides:
- Command allowlisting with separator-aware parsing
- Path allowlisting with traversal checks
- Workspace boundary checks for resolved paths
- Simple sliding-window action rate limiting
"""

from __future__ import annotations

import os
import time
from collections import deque
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class AutonomyLevel(StrEnum):
    READ_ONLY = "readonly"
    SUPERVISED = "supervised"
    FULL = "full"


@dataclass
class ActionTracker:
    """Sliding-window action tracker (default: last hour)."""

    window_seconds: int = 3600
    _timestamps: deque[float] = field(default_factory=deque)

    def _prune(self) -> None:
        cutoff = time.time() - self.window_seconds
        while self._timestamps and self._timestamps[0] <= cutoff:
            self._timestamps.popleft()

    def record(self) -> int:
        self._prune()
        self._timestamps.append(time.time())
        return len(self._timestamps)

    def count(self) -> int:
        self._prune()
        return len(self._timestamps)


@dataclass
class SecurityPolicy:
    """Tool-execution policy for shell and filesystem operations."""

    autonomy: AutonomyLevel = AutonomyLevel.SUPERVISED
    workspace_dir: Path = field(default_factory=lambda: Path(".").resolve())
    workspace_only: bool = True
    allowed_commands: list[str] = field(
        default_factory=lambda: [
            "git",
            "npm",
            "cargo",
            "ls",
            "cat",
            "grep",
            "find",
            "echo",
            "pwd",
            "wc",
            "head",
            "tail",
            "pytest",
            "python",
            "python3",
        ]
    )
    forbidden_paths: list[str] = field(
        default_factory=lambda: [
            "/etc",
            "/root",
            "/home",
            "/usr",
            "/bin",
            "/sbin",
            "/lib",
            "/opt",
            "/boot",
            "/dev",
            "/proc",
            "/sys",
            "/var",
            "/tmp",
            "~/.ssh",
            "~/.gnupg",
            "~/.aws",
            "~/.config",
        ]
    )
    max_actions_per_hour: int = 200
    sanitize_env: bool = True
    safe_env_vars: list[str] = field(
        default_factory=lambda: [
            "PATH",
            "HOME",
            "TERM",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "USER",
            "SHELL",
            "TMPDIR",
            "VIRTUAL_ENV",
            "PYTHONPATH",
        ]
    )
    tracker: ActionTracker = field(default_factory=ActionTracker)

    def can_act(self) -> bool:
        return self.autonomy != AutonomyLevel.READ_ONLY

    def record_action(self) -> bool:
        count = self.tracker.record()
        return count <= self.max_actions_per_hour

    def is_rate_limited(self) -> bool:
        return self.tracker.count() >= self.max_actions_per_hour

    def is_command_allowed(self, command: str) -> bool:
        """Validate the entire shell command, not just the first token."""
        if not self.can_act():
            return False

        if "`" in command or "$(" in command or "${" in command:
            return False

        if ">" in command:
            return False

        normalized = command
        for sep in ("&&", "||"):
            normalized = normalized.replace(sep, "\x00")
        for sep in ("\n", ";", "|"):
            normalized = normalized.replace(sep, "\x00")

        has_cmd = False
        for segment in normalized.split("\x00"):
            segment = segment.strip()
            if not segment:
                continue

            cmd_part = _skip_env_assignments(segment)
            base = _base_command(cmd_part)
            if not base:
                continue

            has_cmd = True
            if base not in self.allowed_commands:
                return False

        return has_cmd

    def is_path_allowed(self, path: str) -> bool:
        """Path pre-check before any filesystem operation."""
        if "\x00" in path:
            return False

        path_obj = Path(path)
        if any(part == ".." for part in path_obj.parts):
            return False

        lowered = path.lower()
        if "..%2f" in lowered or "%2f.." in lowered:
            return False

        expanded = Path(path).expanduser()

        for forbidden in self.forbidden_paths:
            forbidden_path = Path(forbidden).expanduser()
            if expanded == forbidden_path or _starts_with_path(expanded, forbidden_path):
                return False

        return True

    def is_resolved_path_allowed(self, resolved: Path) -> bool:
        workspace_root = self.workspace_dir.resolve()
        try:
            resolved.relative_to(workspace_root)
            return True
        except ValueError:
            return False

    def resolved_target(self, path: str | Path) -> Path:
        """Resolve a path against workspace while preserving relative inputs."""
        p = Path(path)
        if p.is_absolute():
            return p.resolve()
        return (self.workspace_dir / p).resolve()

    def build_subprocess_env(self, extra_env: dict[str, str] | None = None) -> dict[str, str]:
        """Build execution env with optional sanitization."""
        if self.sanitize_env:
            env = {k: os.environ[k] for k in self.safe_env_vars if k in os.environ}
        else:
            env = dict(os.environ)

        if extra_env:
            env.update(extra_env)

        return env


def _skip_env_assignments(command_segment: str) -> str:
    rest = command_segment.strip()
    while rest:
        parts = rest.split(maxsplit=1)
        word = parts[0]
        if "=" in word and (word[0].isalpha() or word[0] == "_"):
            rest = parts[1] if len(parts) > 1 else ""
            rest = rest.lstrip()
            continue
        return rest
    return ""


def _base_command(command_segment: str) -> str:
    if not command_segment:
        return ""
    head = command_segment.split(maxsplit=1)[0]
    return Path(head).name


def _starts_with_path(path: Path, prefix: Path) -> bool:
    try:
        path.relative_to(prefix)
        return True
    except ValueError:
        return False
