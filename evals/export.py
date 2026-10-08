"""Write one eval task's buggy repo to a directory, to try the CLI by hand.

  python -m evals.export stats.median.L15c26 /tmp/buggy
  python -m mender fix /tmp/buggy --provider heuristic --out fix.patch
"""
import json
import sys
from pathlib import Path

from .harness import HERE, build, special_tasks


def main() -> int:
    task_id, dest = sys.argv[1], Path(sys.argv[2])
    tasks = json.loads((HERE / "tasks.json").read_text())["tasks"]
    tasks += special_tasks(tasks)
    task = next((t for t in tasks if t["id"] == task_id), None)
    if task is None:
        print(f"unknown task; choose from: {', '.join(t['id'] for t in tasks)}", file=sys.stderr)
        return 1
    if dest.exists():
        print(f"{dest} already exists", file=sys.stderr)
        return 1
    build(task, dest)
    print(f"buggy repo for {task_id} written to {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
