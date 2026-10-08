"""Run state, episodic memory across attempts, and SQLite checkpoints.

Two kinds of memory:
- *Working memory* for the current attempt: a step log whose older entries
  are compacted to one-liners under a hard budget, plus the last observation
  verbatim.
- *Episodic memory* across attempts: when an attempt fails verification the
  workspace is rolled back, the working memory is cleared, and a short lesson
  (what was changed, what the hypothesis was, what broke) is kept. The next
  attempt starts with a clean context but knows what not to repeat.

Disk and state can never disagree after a crash: the state holds the exact
list of edits applied in the current attempt, and resume rolls the workspace
back to the baseline commit and replays them.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LogEntry:
    turn: int
    tool: str
    summary: str


@dataclass
class RunState:
    run_id: str
    task: str
    workdir: str
    phase: str = "reproduce"   # reproduce | act | verify | done
    targets: list[str] = field(default_factory=list)
    baseline_sha: str = ""
    baseline_passed: list[str] = field(default_factory=list)
    baseline_failed: list[str] = field(default_factory=list)
    failure_text: str = ""
    attempt: int = 1
    attempt_turns: int = 0
    edits: list[dict[str, Any]] = field(default_factory=list)   # applied this attempt, in order
    lessons: list[str] = field(default_factory=list)
    log: list[LogEntry] = field(default_factory=list)
    last_observation: str = ""
    submit_summary: str = ""
    outcome: str = ""          # resolved | gave_up | exhausted | cannot_reproduce | error
    outcome_detail: str = ""
    patch: str = ""
    counters: dict[str, int] = field(default_factory=lambda: {
        "turns": 0, "llm_calls": 0, "tool_calls": 0, "test_runs": 0, "edits": 0, "edit_errors": 0,
        "policy_denials": 0, "rollbacks": 0, "verify_runs": 0, "injection_flags": 0, "invalid_actions": 0,
    })
    started_at: float = field(default_factory=time.time)

    @classmethod
    def new(cls, task: str, workdir: str, run_id: str | None = None) -> "RunState":
        return cls(run_id=run_id or uuid.uuid4().hex[:10], task=task, workdir=workdir)

    @property
    def touched(self) -> set[str]:
        return {e["path"] for e in self.edits}

    def working_memory(self, keep_recent: int = 8, max_chars: int = 3000) -> str:
        older, recent = self.log[:-keep_recent], self.log[-keep_recent:]
        lines = []
        if older:
            reads = sorted({e.summary.split()[1] for e in older if e.tool == "read_file" and len(e.summary.split()) > 1})
            lines.append(f"[{len(older)} earlier actions compacted; files already read: {', '.join(reads) or 'none'}]")
        lines += [f"turn {e.turn} {e.tool}: {e.summary}" for e in recent]
        out = "\n".join(lines)
        return out if len(out) <= max_chars else out[:200] + "\n...\n" + out[-(max_chars - 210):]

    def lessons_text(self) -> str:
        if not self.lessons:
            return ""
        return "Previous attempts FAILED verification and were rolled back. Do not repeat them:\n" + \
               "\n".join(f"- {l}" for l in self.lessons)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "RunState":
        d = json.loads(raw)
        d["log"] = [LogEntry(**e) for e in d["log"]]
        return cls(**d)


class Checkpointer:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as c:
            c.execute("CREATE TABLE IF NOT EXISTS checkpoints (run_id TEXT, seq INTEGER, phase TEXT, ts REAL,"
                      " state TEXT, PRIMARY KEY (run_id, seq))")

    def save(self, s: RunState) -> int:
        with sqlite3.connect(self.path) as c:
            seq = c.execute("SELECT COALESCE(MAX(seq),0)+1 FROM checkpoints WHERE run_id=?", (s.run_id,)).fetchone()[0]
            c.execute("INSERT INTO checkpoints VALUES (?,?,?,?,?)", (s.run_id, seq, s.phase, time.time(), s.to_json()))
        return seq

    def load(self, run_id: str) -> RunState | None:
        with sqlite3.connect(self.path) as c:
            row = c.execute("SELECT state FROM checkpoints WHERE run_id=? ORDER BY seq DESC LIMIT 1", (run_id,)).fetchone()
        return RunState.from_json(row[0]) if row else None

    def runs(self) -> list[tuple[str, str, int]]:
        with sqlite3.connect(self.path) as c:
            return c.execute("SELECT run_id, phase, seq FROM checkpoints c1 WHERE seq = "
                             "(SELECT MAX(seq) FROM checkpoints c2 WHERE c2.run_id=c1.run_id)").fetchall()
