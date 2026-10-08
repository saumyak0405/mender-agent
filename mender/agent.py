"""mender: reproduce -> act -> verify -> (rollback + lesson | done).

The loop is an explicit state machine. Each handler does one unit of work
(one test run, one model turn, one verification) and the state is
checkpointed after each, so a run can be resumed after a crash.

The agent's own "I'm done" (submit) is never trusted: verification re-runs
the full suite and compares against the baseline. A failed attempt is rolled
back to the baseline commit, turned into a lesson, and the next attempt
starts from a clean tree and a clean context.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import prompts, tools
from .llm import LLM, LLMError, call_json
from .policy import Policy
from .sandbox import Workspace
from .state import Checkpointer, LogEntry, RunState
from .trace import Tracer


@dataclass
class Config:
    max_attempts: int = 3
    max_turns_per_attempt: int = 25
    max_turns_total: int = 60
    max_test_runs: int = 40
    test_timeout_s: int = 60        # per test run; mutated loops can hang forever
    native_tools: bool = True       # send tools as function schemas (tool_choice=required)
    wall_clock_s: float = 900.0


@dataclass
class Agent:
    llm: LLM
    config: Config = field(default_factory=Config)
    policy: Policy = field(default_factory=Policy)
    checkpointer: Checkpointer | None = None
    trace_dir: Path | None = None

    # ------------------------------------------------------------ public
    def run(self, repo: str | Path, task: str, targets: list[str] | None = None,
            run_id: str | None = None, work_root: Path | None = None) -> RunState:
        state = RunState.new(task, "", run_id)
        base = (work_root or Path(".mender/work")) / state.run_id
        ws = Workspace.create(repo, base)
        state.workdir = str(base)
        state.baseline_sha = ws.git("rev-parse", "HEAD").strip()
        state.targets = list(targets or [])
        return self._drive(state, ws)

    def resume(self, run_id: str) -> RunState:
        if not self.checkpointer:
            raise RuntimeError("resume needs a checkpointer")
        state = self.checkpointer.load(run_id)
        if state is None:
            raise KeyError(f"no checkpoint for {run_id}")
        ws = Workspace.attach(state.workdir)
        ws.rollback(state.baseline_sha)            # make disk match state exactly:
        for e in state.edits:                      # baseline + replay of the edit log
            p = ws.root / e["path"]
            text = p.read_text()
            if text.count(e["old"]) != 1:
                state.edits, state.phase = [], "act"
                state.lessons.append(f"attempt {state.attempt} could not be replayed after a crash; restarted it")
                ws.rollback(state.baseline_sha)
                break
            p.write_text(text.replace(e["old"], e["new"], 1))
        return self._drive(state, ws, resumed=True)

    # ------------------------------------------------------------ driver
    def _drive(self, state: RunState, ws: Workspace, resumed: bool = False) -> RunState:
        self.ws, self.state = ws, state
        self.box = tools.Toolbox(ws, self.policy, self.config.test_timeout_s)
        self.tracer = Tracer(self.trace_dir / f"{state.run_id}.jsonl" if self.trace_dir else None)
        self.tracer.emit("resume" if resumed else "run_start", run_id=state.run_id, task=state.task,
                         phase=state.phase, edits=len(state.edits))
        handlers = {"reproduce": self._reproduce, "act": self._act, "verify": self._verify}
        deadline = time.time() + self.config.wall_clock_s
        while state.phase in handlers:
            if time.time() > deadline:
                self._finish("exhausted", "wall-clock budget exhausted")
                break
            try:
                handlers[state.phase]()
            except LLMError as e:
                self.tracer.emit("error", phase=state.phase, error=str(e))
                self._finish("error", f"model error: {e}")
            except Exception as e:  # a bug in a handler or policy must not take down the caller
                self.tracer.emit("error", phase=state.phase, error=f"{type(e).__name__}: {e}")
                self._finish("error", f"internal error in {state.phase}: {type(e).__name__}: {e}")
            self._checkpoint()
        self.tracer.emit("run_end", outcome=state.outcome, detail=state.outcome_detail, counters=state.counters,
                         elapsed_s=round(time.time() - state.started_at, 2))
        state.trace = self.tracer.events  # type: ignore[attr-defined]
        return state

    def _checkpoint(self) -> None:
        if self.checkpointer:
            self.checkpointer.save(self.state)

    # ------------------------------------------------------------ phases
    def _reproduce(self) -> None:
        s = self.state
        rep = self.ws.run_tests(timeout=self.config.test_timeout_s)
        s.counters["test_runs"] += 1
        s.baseline_passed, s.baseline_failed = sorted(rep.passed), sorted(rep.failed)
        if rep.timed_out or rep.collected == 0:
            return self._finish("cannot_reproduce", f"baseline test run unusable: {rep.summary()}")
        if s.targets:  # node ids or prefixes such as "tests/test_x.py"; keep only those that really fail
            s.targets = sorted(f for f in rep.failed if any(f.startswith(r) for r in s.targets))
        else:
            s.targets = sorted(rep.failed)
        self.tracer.emit("reproduce", passed=len(rep.passed), failed=sorted(rep.failed), targets=s.targets)
        if not s.targets:
            return self._finish("cannot_reproduce", "no failing target tests at baseline")
        focus = self.ws.run_tests(s.targets, timeout=self.config.test_timeout_s)
        s.counters["test_runs"] += 1
        s.failure_text = focus.failure_text(3500)
        s.last_observation = "Reproduced: " + focus.summary()
        s.phase = "act"

    def _act(self) -> None:
        s, c = self.state, self.config
        if s.attempt_turns >= c.max_turns_per_attempt or s.counters["turns"] >= c.max_turns_total \
                or s.counters["test_runs"] >= c.max_test_runs:
            if s.edits:  # out of turns but has a candidate fix: verify it rather than waste it
                self.tracer.emit("budget_stop", action="force_verify")
                s.phase = "verify"
            else:
                self._fail_attempt("ran out of turns without making an edit")
            return
        s.attempt_turns += 1
        s.counters["turns"] += 1

        def valid(obj: Any) -> str | None:
            if not isinstance(obj, dict):
                return "reply must be a JSON object"
            return tools.validate(obj.get("tool", ""), obj.get("args", {}))

        msgs = [{"role": "system", "content": prompts.CODER.format(tools=tools.catalog())},
                {"role": "user", "content": prompts.CODER_USER.format(
                    task=s.task, targets=json.dumps(s.targets), attempt=s.attempt, max_attempts=c.max_attempts,
                    turns_left=c.max_turns_per_attempt - s.attempt_turns + 1, lessons=s.lessons_text(),
                    failure=s.failure_text, touched=", ".join(sorted(s.touched)) or "none",
                    memory=s.working_memory() or "(nothing yet)", last=s.last_observation[:4500])}]

        def count() -> None:
            s.counters["llm_calls"] += 1
        try:
            kw = {"tools": tools.openai_tools()} if self.config.native_tools else {}
            action = call_json(self.llm, msgs, valid, retries=2, on_call=count, **kw)
        except LLMError as e:
            if "valid JSON" not in str(e):
                raise
            s.counters["invalid_actions"] += 1
            s.last_observation = f"Your last reply was not a valid action ({e}). Reply with one JSON tool call."
            return
        tool, args = action["tool"], action.get("args", {})
        self.tracer.emit("decision", attempt=s.attempt, turn=s.counters["turns"], tool=tool, args=_short(args),
                         thought=str(action.get("thought", ""))[:300])

        if tool == "submit":
            if not s.edits:
                s.last_observation = "submit rejected: you have not changed anything yet."
                return
            s.submit_summary = args["summary"][:500]
            s.phase = "verify"
            return
        if tool == "give_up":
            self.ws.rollback(s.baseline_sha)
            s.edits = []
            return self._finish("gave_up", args["reason"][:500])

        s.counters["tool_calls"] += 1
        out = self.box.execute(tool, args, s.touched)
        if out.report is not None:
            s.counters["test_runs"] += 1
        if out.denied:
            s.counters["policy_denials"] += 1
        if out.flagged:
            s.counters["injection_flags"] += 1
        if out.edit:
            s.edits.append(out.edit)
            s.counters["edits"] += 1
        elif tool == "edit":
            s.counters["edit_errors"] += 1
        s.last_observation = out.text
        s.log.append(LogEntry(s.counters["turns"], tool, _summarize(tool, args, out.text)))
        self.tracer.emit("observation", tool=tool, ok=out.ok, denied=out.denied, flagged=out.flagged,
                         text=out.text[:400])

    def _verify(self) -> None:
        s = self.state
        rep = self.ws.run_tests(timeout=self.config.test_timeout_s)
        s.counters["test_runs"] += 1
        s.counters["verify_runs"] += 1
        still = sorted(set(s.targets) & rep.failed) if not rep.timed_out else s.targets
        regressions = sorted(set(s.baseline_passed) - rep.passed)
        touched_protected = [f for f in self.ws.changed_files(s.baseline_sha)
                             if not self.policy.check_write(f, 0, {f}).allowed]
        ok = not rep.timed_out and not still and not regressions and not touched_protected
        self.tracer.emit("verify", attempt=s.attempt, ok=ok, still_failing=still, regressions=regressions,
                         timed_out=rep.timed_out, protected_changes=touched_protected)
        if ok:
            s.patch = self.ws.diff(s.baseline_sha)
            return self._finish("resolved", s.submit_summary or "targets pass, no regressions")
        why = []
        if rep.timed_out:
            why.append("test suite timed out (infinite loop?)")
        if still:
            why.append(f"still failing: {', '.join(still[:5])}")
        if regressions:
            why.append(f"broke previously passing tests: {', '.join(regressions[:5])}")
        if touched_protected:
            why.append(f"modified protected files {touched_protected}")
        self._fail_attempt("; ".join(why), rep.failure_text(1500))

    # ------------------------------------------------------------ helpers
    def _fail_attempt(self, reason: str, detail: str = "") -> None:
        s = self.state
        # record the NET change that was verified, not the raw edit log (which includes reverted tries)
        net = [l for l in self.ws.diff(s.baseline_sha).splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---")]
        changes = "; ".join(_oneline(l, 90) for l in net[:6]) + (" ..." if len(net) > 6 else "") or "no net change"
        hyp = f" Hypothesis was: {s.submit_summary}." if s.submit_summary else ""
        s.lessons.append(f"attempt {s.attempt} ({len(s.edits)} edits): net change [{changes}].{hyp} Result: {reason}.")
        self.ws.rollback(s.baseline_sha)
        s.counters["rollbacks"] += 1
        self.tracer.emit("rollback", attempt=s.attempt, reason=reason)
        if s.attempt >= self.config.max_attempts:
            return self._finish("exhausted", f"{s.attempt} attempts failed verification; last: {reason}")
        s.attempt += 1
        s.attempt_turns, s.edits, s.log, s.submit_summary = 0, [], [], ""
        s.last_observation = f"Attempt {s.attempt - 1} was rolled back: {reason}\n{detail}"
        s.phase = "act"

    def _finish(self, outcome: str, detail: str) -> None:
        s = self.state
        s.outcome, s.outcome_detail, s.phase = outcome, detail, "done"
        self.tracer.emit("final", outcome=outcome, detail=detail, patch_lines=s.patch.count("\n"))


def _oneline(t: str, n: int = 70) -> str:
    t = " ".join(t.split())
    return t if len(t) <= n else t[: n - 3] + "..."


def _short(args: dict[str, Any]) -> dict[str, Any]:
    return {k: (_oneline(v, 120) if isinstance(v, str) else v) for k, v in args.items()}


def _summarize(tool: str, args: dict[str, Any], text: str) -> str:
    first = text.splitlines()[0] if text else ""
    if tool == "read_file":
        return f"read {args.get('path')} -> {first}"
    if tool == "edit":
        return f"edit {args.get('path')}: {_oneline(args['old'], 50)} -> {_oneline(args['new'], 50)} | {first[:80]}"
    if tool == "search":
        return f"search /{args.get('pattern')}/ -> {len(text.splitlines())} lines"
    return f"{_short(args)} -> {first[:160]}"
