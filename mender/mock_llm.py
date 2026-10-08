"""HeuristicPolicy: a non-LLM baseline that speaks the same action protocol.

It localises the bug from names in the failing assertions, reads the
function, then searches single-token edits (the inverse of the mutation
operators) one at a time: edit -> run failing tests -> keep or revert.
It is a classic generate-and-validate repair baseline, useful for two
things: running the whole agent loop in CI without an API key, and giving
the LLM agent a floor to beat. It is not a language model.
"""
from __future__ import annotations

import json
import re
import textwrap
from dataclasses import dataclass, field

from . import mutations

_SKIP = {"len", "list", "sorted", "set", "dict", "str", "int", "float", "abs", "round", "range", "tuple",
         "isinstance", "approx", "raises", "assert", "print", "sum", "min", "max", "any", "all", "type", "repr"}
_ROW = re.compile(r"^\s*(\d+)\| ?(.*)$")


@dataclass
class HeuristicPolicy:
    name: str = "heuristic"
    calls: int = 0
    attempt: int = 0
    queue: list[str] = field(default_factory=list)          # function names still to localise
    pending: list[dict] = field(default_factory=list)        # candidate edits for the current function
    tried: set[tuple[str, int, str]] = field(default_factory=set)
    applied: dict | None = None
    expect: str = ""                                         # what the last action was
    started: bool = False

    def complete(self, messages, **kw) -> str:
        self.calls += 1
        user = messages[-1]["content"]
        attempt = int(re.search(r"Attempt (\d+) of", user).group(1))
        if attempt != self.attempt:  # workspace was rolled back: any applied candidate is gone
            self.attempt, self.applied = attempt, None
            if self.expect in ("edit", "test", "revert", "submit"):
                self.expect = "resume"
        last = user.split("Last observation:\n", 1)[1]
        targets = json.loads(re.search(r"Failing tests: (\[.*?\])\n", user).group(1))

        if not self.started:
            self.started = True
            failure = user.split("Failure output at start:\n", 1)[1].split("\n\nFiles edited", 1)[0]
            names: list[str] = []
            for line in failure.splitlines():
                if "assert" in line or line.lstrip().startswith("E "):
                    for n in re.findall(r"\b([a-z_][a-z0-9_]*)\(", line):
                        if n not in _SKIP and not n.startswith("test_") and n not in names:
                            names.append(n)
            self.queue = names
            return self._next_function()

        if self.expect == "search":
            hit = next((l for l in last.splitlines() if re.match(r"^(?!tests?/)[\w/.-]+\.py:\d+:", l)), None)
            if not hit:
                return self._next_function()
            path, line = hit.split(":")[:2]
            return self._act("read", "read_file", {"path": path, "start": int(line), "end": int(line) + 80},
                             "locate function body")

        if self.expect == "read":
            self.pending = self._candidates(last)
            return self._try_next(targets)

        if self.expect == "edit":
            if last.startswith("edited"):
                return self._act("test", "run_tests", {"tests": targets}, "check candidate")
            self.applied = None
            return self._try_next(targets)

        if self.expect == "test" and self.applied:
            if re.search(r"run_tests: \d+ passed, 0 failed", last):
                return self._act("submit", "submit", {"summary": f"single-token fix: {self.applied['desc']}"},
                                 "targets pass")
            a = self.applied
            self.applied = None
            return self._act("revert", "edit", {"path": a["path"], "old": a["new"], "new": a["old"]},
                             "candidate failed, revert")

        if self.expect == "revert":
            return self._try_next(targets)
        return self._try_next(targets)

    # ------------------------------------------------------------ helpers
    def _act(self, expect: str, tool: str, args: dict, thought: str) -> str:
        self.expect = expect
        return json.dumps({"thought": thought, "tool": tool, "args": args})

    def _next_function(self) -> str:
        if not self.queue:
            return self._act("done", "give_up", {"reason": "no single-token change to the implicated functions makes the failing tests pass"},
                             "search space exhausted")
        name = self.queue.pop(0)
        return self._act("search", "search", {"pattern": rf"def {name}\b", "glob": "*.py"}, f"locate {name}")

    def _candidates(self, read_out: str) -> list[dict]:
        head = read_out.splitlines()[0] if read_out else ""
        path = head.split(" (lines", 1)[0]
        rows = [(int(m.group(1)), m.group(2)) for m in map(_ROW.match, read_out.splitlines()) if m]
        if not rows or not rows[0][1].lstrip().startswith("def "):
            return []
        indent = len(rows[0][1]) - len(rows[0][1].lstrip())
        body = [rows[0]]
        for n, text in rows[1:]:
            if text.strip() and len(text) - len(text.lstrip()) <= indent:
                break
            body.append((n, text))
        first = body[0][0]
        src = textwrap.dedent("\n".join(t for _, t in body)) + "\n"
        lines = {n: t for n, t in body}
        out = []
        for m in mutations.candidates(src):
            if m.line == 1:  # don't mutate the signature
                continue
            real = first + m.line - 1
            orig = lines[real]
            col = m.col + indent
            new = orig[:col] + m.replacement + orig[m.end_col + indent:]
            key = (path, real, new)
            if key not in self.tried:
                out.append({"path": path, "line": real, "old": orig, "new": new, "key": key,
                            "desc": f"{path}:{real} '{m.original}' -> '{m.replacement}'"})
        return out

    def _try_next(self, targets: list[str]) -> str:
        while self.pending:
            c = self.pending.pop(0)
            if c["key"] in self.tried:
                continue
            self.tried.add(c["key"])
            self.applied = c
            return self._act("edit", "edit", {"path": c["path"], "old": c["old"], "new": c["new"]},
                             f"try {c['desc']}")
        return self._next_function()
