"""CLI.

  python -m mender fix path/to/repo --task "median is wrong for even-length lists" --provider auto
  python -m mender fix path/to/repo --tests tests/test_stats.py --out fix.patch
  python -m mender resume <run_id>
  python -m mender runs
  python -m mender trace <run_id>

The original repo is never modified. Apply the result yourself with `git apply fix.patch`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .agent import Agent, Config
from .llm import build_llm
from .state import Checkpointer
from .trace import Tracer

HOME = Path(".mender")
PROVIDERS = ["auto", "groq", "ollama", "heuristic"]


def make_agent(provider: str, **cfg) -> Agent:
    return Agent(llm=build_llm(provider), config=Config(**cfg),
                 checkpointer=Checkpointer(HOME / "checkpoints.sqlite"), trace_dir=HOME / "traces")


def report(state, out: str | None) -> int:
    print(f"outcome: {state.outcome} - {state.outcome_detail}")
    print(f"run: {state.run_id}  attempts: {state.attempt}  counters: {json.dumps(state.counters)}")
    for lesson in state.lessons:
        print(f"  lesson: {lesson}")
    if state.patch:
        if out:
            Path(out).write_text(state.patch)
            print(f"patch written to {out}  (apply with: git apply {out})")
        else:
            print(state.patch)
    return 0 if state.outcome == "resolved" else 2


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mender")
    sub = p.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fix", help="fix failing tests in a repo (works on a sandboxed copy)")
    f.add_argument("repo")
    f.add_argument("--task", default="Make the failing tests pass by fixing the bug in the source code.")
    f.add_argument("--tests", nargs="*", help="target test ids or files (default: everything failing)")
    f.add_argument("--provider", default="auto", choices=PROVIDERS)
    f.add_argument("--max-attempts", type=int, default=3)
    f.add_argument("--out")
    r = sub.add_parser("resume")
    r.add_argument("run_id")
    r.add_argument("--provider", default="auto", choices=PROVIDERS)
    r.add_argument("--out")
    sub.add_parser("runs")
    t = sub.add_parser("trace")
    t.add_argument("run_id")
    a = p.parse_args(argv)

    if a.cmd == "fix":
        agent = make_agent(a.provider, max_attempts=a.max_attempts)
        return report(agent.run(a.repo, a.task, a.tests, work_root=HOME / "work"), a.out)
    if a.cmd == "resume":
        return report(make_agent(a.provider).resume(a.run_id), a.out)
    if a.cmd == "runs":
        for run_id, phase, seq in Checkpointer(HOME / "checkpoints.sqlite").runs():
            print(f"{run_id}  {phase:<10} checkpoints={seq}")
        return 0
    for ev in Tracer.load(HOME / "traces" / f"{a.run_id}.jsonl"):
        print(json.dumps(ev, default=str)[:300])
    return 0


if __name__ == "__main__":
    sys.exit(main())
