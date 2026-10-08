"""Generate repair tasks by injecting single-token bugs into the target library.

For every mutable token in src/textkit (comparison, arithmetic, boolean,
min/max, integer constant), apply the mutation, run the visible tests, and
keep the mutant only if (a) it still compiles, (b) at least one visible test
fails, and (c) the suite does not hang. Mutants no visible test catches are
"equivalent or untested" and are dropped. A stratified, seeded sample (at
most N per function) becomes the task file, so tasks are reproducible.

  python -m evals.mutate --per-function 2 --seed 0
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from mender import mutations

HERE = Path(__file__).parent
TARGET = HERE / "target"


def enclosing_function(tree: ast.AST, line: int) -> str | None:
    best = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.lineno <= line <= (node.end_lineno or node.lineno):
            best = node.name
    return best


def run_visible(root: Path, timeout: int = 10) -> tuple[str, list[str]]:
    xml = root.parent / "junit.xml"
    xml.unlink(missing_ok=True)
    try:
        subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "junit_family=xunit1",
                        f"--junitxml={xml}", "tests"], cwd=root, env={"PYTHONPATH": str(root / "src"), "PATH": "",
                            # one-char mutants keep file size and mtime-second, so a cached .pyc
                            # from the previous mutant would be reused: never write bytecode
                            "PYTHONDONTWRITEBYTECODE": "1"},
                       capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "timeout", []
    if not xml.exists():
        return "error", []
    import xml.etree.ElementTree as ET
    failed = [f"{c.get('file')}::{c.get('name')}" for c in ET.parse(xml).getroot().iter("testcase")
              if c.find("failure") is not None or c.find("error") is not None]
    return ("killed" if failed else "survived"), failed


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-function", type=int, default=2)
    ap.add_argument("--doubles", type=int, default=3, help="extra tasks with two bugs in one function")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "tasks.json"))
    args = ap.parse_args(argv)
    rng = random.Random(args.seed)

    work = Path(tempfile.mkdtemp(prefix="mender-mutate-")) / "repo"
    shutil.copytree(TARGET, work, ignore=shutil.ignore_patterns("__pycache__"))
    killed: dict[tuple[str, str], list[dict]] = {}
    stats = {"candidates": 0, "killed": 0, "survived": 0, "timeout": 0, "error": 0}
    for src in sorted((TARGET / "src").rglob("*.py")):
        rel = src.relative_to(TARGET).as_posix()
        original = src.read_text()
        tree = ast.parse(original)
        for m in mutations.candidates(original):
            fn = enclosing_function(tree, m.line)
            if fn is None:
                continue
            mutated = mutations.apply(original, m)
            try:
                compile(mutated, rel, "exec")
            except SyntaxError:
                continue
            stats["candidates"] += 1
            (work / rel).write_text(mutated)
            status, failed = run_visible(work)
            (work / rel).write_text(original)
            stats[status] += 1
            if status == "killed":
                killed.setdefault((rel, fn), []).append({"mutation": m.__dict__, "failing": failed})
        print(f"{rel}: done", file=sys.stderr)

    tasks = []
    for (rel, fn), ms in sorted(killed.items()):
        for k in rng.sample(ms, min(args.per_function, len(ms))):
            mm = k["mutation"]
            tid = f"{Path(rel).stem}.{fn}.L{mm['line']}c{mm['col']}"
            tid += "" if tid not in {t["id"] for t in tasks} else f"-{mm['replacement']}"
            tasks.append({"id": tid, "kind": "single", "file": rel,
                          "function": fn, "mutations": [mm], "hint_tests": sorted({f.split('::')[0] for f in k["failing"]})})
    # double bugs: two independent killed mutants on different lines of one function
    pool = [(key, ms) for key, ms in sorted(killed.items()) if len({m["mutation"]["line"] for m in ms}) >= 2]
    for (rel, fn), ms in rng.sample(pool, min(args.doubles, len(pool))):
        a = rng.choice(ms)
        b = rng.choice([m for m in ms if m["mutation"]["line"] != a["mutation"]["line"]])
        tasks.append({"id": f"{Path(rel).stem}.{fn}.double", "kind": "double", "file": rel, "function": fn,
                      "mutations": sorted([a["mutation"], b["mutation"]], key=lambda m: (m["line"], m["col"])),
                      "hint_tests": []})
    Path(args.out).write_text(json.dumps({"seed": args.seed, "stats": stats, "tasks": tasks}, indent=1))
    print(json.dumps(stats), f"-> {len(tasks)} tasks written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
