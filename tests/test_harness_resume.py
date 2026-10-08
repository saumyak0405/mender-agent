"""Harness bookkeeping: incremental results, --resume, and stopping when the API quota is gone."""
import json

import pytest

from evals import harness
from evals.harness import Result


def fake(outcome_for):
    def run_task(task, provider, cfg):
        out = outcome_for(task["id"])
        ok = out == "resolved"
        return Result(task["id"], task["kind"], out, ok, ok, ok, ok, False, False, [], 2 if ok else 0, 1, 0.1,
                      counters={"turns": 3 if ok else 1}, detail="" if ok else "HTTP 429: tokens per day")
    return run_task


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "HERE", tmp_path)
    (tmp_path / "tasks.json").write_text(json.dumps({"tasks": [
        {"id": f"t{i}", "kind": "single", "file": "x.py", "mutations": []} for i in range(6)]}))

    class Exit(Exception):
        pass
    monkeypatch.setattr(harness, "_hard_exit", lambda code: (_ for _ in ()).throw(Exit(code)))
    monkeypatch.setattr(harness, "write_report", lambda *a, **k: tmp_path / "r.md")
    return tmp_path, Exit


def lines(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def test_quota_exhaustion_stops_early_and_resume_finishes(env, monkeypatch):
    tmp, Exit = env
    argv = ["--provider", "x", "--label", "run", "--tasks", str(tmp / "tasks.json"), "--no-special"]
    # day 1: two tasks succeed, then every call is rate-limited
    monkeypatch.setattr(harness, "run_task", fake(lambda tid: "resolved" if tid in ("t0", "t1") else "error"))
    with pytest.raises(Exit) as e:
        harness.main(argv)
    assert e.value.args[0] == 3
    log = lines(tmp / "results" / "run.jsonl")
    assert [r["id"] for r in log] == ["t0", "t1", "t2", "t3", "t4"]   # stopped after 3 infra errors in a row
    # day 2: quota is back; finished tasks are not re-run, errored and unstarted ones are
    seen = []

    def day2(tid):
        seen.append(tid)
        return "resolved"
    monkeypatch.setattr(harness, "run_task", fake(day2))
    assert harness.main(argv + ["--resume"]) == 0
    assert sorted(seen) == ["t2", "t3", "t4", "t5"]
    summary = json.loads((tmp / "results" / "run.json").read_text())["summary"]
    assert summary["complete"] and summary["tasks"] == 6 and summary["success_rate"] == 1.0


def test_fresh_run_discards_old_log(env, monkeypatch):
    tmp, _ = env
    argv = ["--provider", "x", "--label", "run", "--tasks", str(tmp / "tasks.json"), "--no-special"]
    monkeypatch.setattr(harness, "run_task", fake(lambda tid: "resolved"))
    harness.main(argv)
    harness.main(argv)
    assert len(lines(tmp / "results" / "run.jsonl")) == 6
