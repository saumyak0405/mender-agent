"""Authorisation policy: what the agent may read, write and run.

The policy is enforced in code at the tool boundary, not requested in the
prompt. A model that decides to "fix" a failing test by editing the test,
or that follows an instruction planted in a file, is refused here no matter
what it says. Every denial is traced and fed back to the model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path

DEFAULT_PROTECTED = (
    "tests/**", "test/**", "test_*.py", "**/test_*.py", "*_test.py", "**/*_test.py", "conftest.py",
    "**/conftest.py", ".git/**", ".github/**", "pyproject.toml", "setup.py", "setup.cfg", "tox.ini",
    "pytest.ini", "noxfile.py", "requirements*.txt", "*.lock", ".env*",
)


def _match(path: str, pattern: str) -> bool:
    return fnmatch(path, pattern) or (pattern.startswith("**/") and fnmatch(path, pattern[3:]))


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str = ""


@dataclass
class Policy:
    writable: tuple[str, ...] = ("*.py", "**/*.py")
    protected: tuple[str, ...] = DEFAULT_PROTECTED
    unreadable: tuple[str, ...] = (".env*", "**/.env*", "*.pem", "**/*.pem", "**/id_rsa*", ".git/**")
    max_files_touched: int = 3
    max_lines_per_edit: int = 40
    max_read_bytes: int = 400_000
    notes: list[str] = field(default_factory=list)

    def resolve(self, root: Path, path: str) -> tuple[str | None, str]:
        """Return the repo-relative POSIX path, or None with a reason if it escapes the repo."""
        if not path or "\x00" in path:
            return None, "empty or invalid path"
        try:
            full = (root / path).resolve()
            rel = full.relative_to(root.resolve())
        except (ValueError, OSError):
            return None, f"'{path}' is outside the repository"
        return rel.as_posix(), ""

    def check_read(self, rel: str) -> Decision:
        if any(_match(rel, p) for p in self.unreadable):
            return Decision(False, f"reading '{rel}' is not permitted (secrets / VCS internals)")
        return Decision(True)

    def check_write(self, rel: str, lines_changed: int, touched: set[str]) -> Decision:
        if any(_match(rel, p) for p in self.protected):
            return Decision(False, f"'{rel}' is protected (tests, config and CI cannot be modified). Fix the source code instead.")
        if not any(_match(rel, p) for p in self.writable):
            return Decision(False, f"'{rel}' is not a writable file type")
        if rel not in touched and len(touched) >= self.max_files_touched:
            return Decision(False, f"edit would touch more than {self.max_files_touched} files; keep the fix focused")
        if lines_changed > self.max_lines_per_edit:
            return Decision(False, f"edit changes {lines_changed} lines (limit {self.max_lines_per_edit}); make smaller edits")
        return Decision(True)
