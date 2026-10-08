"""Isolated workspace: a private git-tracked copy of the target repo.

The agent never touches the user's repository. It works on a copy that is
committed as a baseline; every attempt can be rolled back to that commit
exactly, and the deliverable is a patch the user chooses to apply.

Tests run in a subprocess with a scrubbed environment (no credentials, proxy
pointed at a dead port to discourage network access), a wall-clock timeout,
CPU / file-size rlimits where the OS supports them, and results parsed from
JUnit XML rather than scraped from console text.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

IGNORE = shutil.ignore_patterns(".git", "__pycache__", "*.pyc", ".venv", "venv", "node_modules",
                                ".mender", ".pytest_cache", ".mypy_cache", "*.egg-info")
MAX_REPO_BYTES = 50 * 1024 * 1024


class SandboxError(RuntimeError):
    pass


@dataclass
class TestReport:
    passed: set[str] = field(default_factory=set)
    failed: set[str] = field(default_factory=set)
    timed_out: bool = False
    returncode: int = 0
    output: str = ""

    @property
    def collected(self) -> int:
        return len(self.passed) + len(self.failed)

    def summary(self) -> str:
        if self.timed_out:
            return "test run TIMED OUT"
        if self.collected == 0:
            return f"no tests collected (exit {self.returncode})"
        return f"{len(self.passed)} passed, {len(self.failed)} failed"

    def failure_text(self, limit: int = 3500) -> str:
        out = self.output
        cut = max(out.find("= FAILURES ="), out.find("= ERRORS ="))
        if cut > 0:
            out = out[out.rfind("\n", 0, cut) + 1:]
        return out[:limit] + ("\n...[truncated]" if len(out) > limit else "")


def _limits(cpu_s: int):  # pragma: no cover - runs in the child process
    def apply() -> None:
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
            resource.setrlimit(resource.RLIMIT_FSIZE, (100 * 1024 * 1024,) * 2)
        except (ImportError, ValueError, OSError):
            pass
    return apply


class Workspace:
    def __init__(self, root: Path, scratch: Path):
        self.root = root.resolve()
        self.scratch = scratch.resolve()

    # --------------------------------------------------------- creation
    @classmethod
    def create(cls, source: Path | str, dest: Path | str | None = None) -> "Workspace":
        source = Path(source).resolve()
        if not source.is_dir():
            raise SandboxError(f"{source} is not a directory")
        size = sum(f.stat().st_size for f in source.rglob("*") if f.is_file() and ".git" not in f.parts)
        if size > MAX_REPO_BYTES:
            raise SandboxError(f"repo too large for sandbox ({size >> 20} MB)")
        base = Path(dest) if dest else Path(tempfile.mkdtemp(prefix="mender-"))
        root, scratch = base / "repo", base / "scratch"
        if root.exists():
            shutil.rmtree(root)
        shutil.copytree(source, root, ignore=IGNORE, symlinks=True)
        scratch.mkdir(parents=True, exist_ok=True)
        ws = cls(root, scratch)
        ws.git("init", "-q")
        ws.commit("baseline")
        return ws

    @classmethod
    def attach(cls, base: Path | str) -> "Workspace":
        base = Path(base)
        if not (base / "repo" / ".git").exists():
            raise SandboxError(f"no workspace at {base}")
        return cls(base / "repo", base / "scratch")

    # --------------------------------------------------------- git
    def git(self, *args: str, check: bool = True) -> str:
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(self.scratch), "GIT_CONFIG_NOSYSTEM": "1",
               "GIT_AUTHOR_NAME": "mender", "GIT_AUTHOR_EMAIL": "mender@localhost",
               "GIT_COMMITTER_NAME": "mender", "GIT_COMMITTER_EMAIL": "mender@localhost"}
        p = subprocess.run(["git", *args], cwd=self.root, env=env, capture_output=True, text=True)
        if check and p.returncode != 0:
            raise SandboxError(f"git {' '.join(args)} failed: {p.stderr.strip()}")
        return p.stdout

    def commit(self, label: str) -> str:
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "--no-verify", "-m", label)
        return self.git("rev-parse", "HEAD").strip()

    def rollback(self, sha: str) -> None:
        self.git("reset", "-q", "--hard", sha)
        self.git("clean", "-qfd")

    def diff(self, sha: str) -> str:
        self.git("add", "-A")
        return self.git("diff", "--cached", sha)

    def changed_files(self, sha: str) -> list[str]:
        self.git("add", "-A")
        return [l for l in self.git("diff", "--cached", "--name-only", sha).splitlines() if l]

    # --------------------------------------------------------- tests
    def run_tests(self, tests: list[str] | None = None, timeout: int = 90) -> TestReport:
        xml = self.scratch / "junit.xml"
        xml.unlink(missing_ok=True)
        cmd = [sys.executable, "-m", "pytest", "-q", "--tb=short", "-p", "no:cacheprovider",
               "-o", "junit_family=xunit1", f"--junitxml={xml}", f"--rootdir={self.root}"]
        if not any((self.root / f).exists() for f in ("pytest.ini", "pyproject.toml", "tox.ini", "setup.cfg")):
            # repo has no pytest config: don't let pytest adopt one from a parent directory
            ini = self.scratch / "pytest.ini"
            ini.write_text("[pytest]\n")
            cmd += ["-c", str(ini)]
        cmd += list(tests or [])
        dead = "http://127.0.0.1:9"
        env = {"PATH": os.environ.get("PATH", ""), "HOME": str(self.scratch), "LANG": "C.UTF-8",
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0",
               "http_proxy": dead, "https_proxy": dead, "HTTP_PROXY": dead, "HTTPS_PROXY": dead}
        if (self.root / "src").is_dir():
            env["PYTHONPATH"] = str(self.root / "src")
        try:
            p = subprocess.run(cmd, cwd=self.root, env=env, capture_output=True, text=True,
                               timeout=timeout, preexec_fn=_limits(timeout + 5) if os.name == "posix" else None)
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or b"")
            out = out.decode(errors="replace") if isinstance(out, bytes) else out
            return TestReport(timed_out=True, returncode=-1, output=out[-4000:])
        report = TestReport(returncode=p.returncode, output=(p.stdout + p.stderr)[-12000:])
        if xml.exists():
            self._parse_junit(xml, report)
        return report

    @staticmethod
    def _parse_junit(xml: Path, report: TestReport) -> None:
        for case in ET.parse(xml).getroot().iter("testcase"):
            file, cls_, name = case.get("file", ""), case.get("classname", ""), case.get("name", "")
            module = file[:-3].replace("/", ".") if file.endswith(".py") else ""
            rest = cls_[len(module) + 1:] if module and cls_.startswith(module) else ""
            node = "::".join(p for p in [file, *(rest.split(".") if rest else []), name] if p)
            if case.find("skipped") is not None:
                continue
            bad = case.find("failure") is not None or case.find("error") is not None
            (report.failed if bad else report.passed).add(node)
