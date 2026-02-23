#!/usr/bin/env python3
"""
task_manager.py — Lightweight Kanban Task Manager
==================================================

Purpose:
<<<<<<< HEAD
    Markdown-backed task board with lane directories: backlog, todo, in-progress, done.
    Each task is a Markdown file (NNNN-title.md) stored in its lane directory.
    Consolidates create_task.py and board.py into a single manager.
=======
    Simple JSON-backed task board with lanes: backlog, todo, in-progress, done.
    Designed as a standalone replacement for the heavier cli.py task subsystem.
>>>>>>> origin/main

Layer: Plugin / Task-Manager

Usage:
<<<<<<< HEAD
    python3 plugins/task-manager/skills/task-agent/scripts/task_manager.py create "Fix login bug" --lane todo
    python3 plugins/task-manager/skills/task-agent/scripts/task_manager.py list
    python3 plugins/task-manager/skills/task-agent/scripts/task_manager.py list --lane in-progress
    python3 plugins/task-manager/skills/task-agent/scripts/task_manager.py move 3 in-progress
    python3 plugins/task-manager/skills/task-agent/scripts/task_manager.py get 3
    python3 plugins/task-manager/skills/task-agent/scripts/task_manager.py search "login"
    python3 plugins/task-manager/skills/task-agent/scripts/task_manager.py board
=======
    python3 task_manager.py create "Fix login bug" --objective "Resolve 401 errors"
    python3 task_manager.py list
    python3 task_manager.py list --status in-progress
    python3 task_manager.py move 3 in-progress
    python3 task_manager.py get 3
    python3 task_manager.py board
>>>>>>> origin/main
"""

import os
import sys
<<<<<<< HEAD
import re
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

SCRIPT_DIR = Path(__file__).parent.resolve()
# scripts/ → task-agent/ → skills/ → task-manager/
PLUGIN_ROOT = SCRIPT_DIR.parents[2]

VALID_LANES = ["backlog", "todo", "in-progress", "done"]
LANE_ICONS = {
    "backlog": "📋",
    "todo": "📝",
    "in-progress": "🔨",
    "done": "✅",
}
TASK_PATTERN = re.compile(r"^(\d{4})-(.*?)\.md$")


def _find_project_root() -> Path:
    """Walk up from PLUGIN_ROOT to find the project root."""
=======
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).parent.resolve()
PLUGIN_ROOT = SCRIPT_DIR.parent.resolve()

# Find project root
def _find_project_root() -> Path:
>>>>>>> origin/main
    p = PLUGIN_ROOT
    for _ in range(10):
        if (p / ".git").exists() or (p / ".agent").exists():
            return p
        p = p.parent
    return Path.cwd()

<<<<<<< HEAD

PROJECT_ROOT = _find_project_root()
TEMPLATE_PATH = PLUGIN_ROOT / "templates" / "task-template.md"


def _sanitize_title(title: str) -> str:
    """Convert title to kebab-case filename segment."""
    return re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')


def _get_next_number(tasks_dir: Path) -> int:
    """Scan all lanes for the highest task number."""
    highest = 0
    for lane in VALID_LANES:
        lane_dir = tasks_dir / lane
        if not lane_dir.exists():
            continue
        for f in lane_dir.iterdir():
            match = TASK_PATTERN.match(f.name)
            if match:
                highest = max(highest, int(match.group(1)))
    return highest + 1


def _find_task(tasks_dir: Path, task_number: int) -> Optional[Tuple[Path, str]]:
    """Find a task file across all lanes. Returns (filepath, lane) or None."""
    num_str = f"{task_number:04d}"
    for lane in VALID_LANES:
        lane_dir = tasks_dir / lane
        if not lane_dir.exists():
            continue
        for f in lane_dir.iterdir():
            if f.name.startswith(num_str) and f.suffix == '.md':
                return (f, lane)
    return None


def _get_all_tasks(tasks_dir: Path, lane_filter: str = None) -> List[Tuple[Path, str, int, str]]:
    """Get all tasks as (filepath, lane, number, title). Optionally filter by lane."""
    tasks = []
    lanes = [lane_filter] if lane_filter else VALID_LANES
    for lane in lanes:
        lane_dir = tasks_dir / lane
        if not lane_dir.exists():
            continue
        for f in sorted(lane_dir.iterdir()):
            match = TASK_PATTERN.match(f.name)
            if match:
                num = int(match.group(1))
                title = match.group(2).replace("-", " ").title()
                tasks.append((f, lane, num, title))
    tasks.sort(key=lambda t: t[2])
    return tasks


# ─── Commands ────────────────────────────────────────

def cmd_create(tasks_dir: Path, title: str, lane: str = "todo",
               objective: str = "", acceptance: str = ""):
    """Create a new task in the specified lane."""
    if lane not in VALID_LANES:
        print(f"❌ Invalid lane '{lane}'. Must be: {', '.join(VALID_LANES)}")
        return

    lane_dir = tasks_dir / lane
    lane_dir.mkdir(parents=True, exist_ok=True)

    next_num = _get_next_number(tasks_dir)
    num_str = f"{next_num:04d}"
    safe_title = _sanitize_title(title)
    filename = f"{num_str}-{safe_title}.md"
    filepath = lane_dir / filename

    # Load template or use fallback
    if TEMPLATE_PATH.exists():
        content = TEMPLATE_PATH.read_text(encoding='utf-8')
        content = content.replace("NNNN", num_str)
        content = content.replace("[Title]", title)
    else:
        content = f"# Task {num_str}: {title}\n\n## Objective\n{objective or 'TBD'}\n\n## Acceptance Criteria\n{acceptance or 'TBD'}\n\n## Notes\n"

    filepath.write_text(content, encoding='utf-8')
    print(f"✅ Created task #{num_str}: {title} [{lane}]")
    print(f"   Path: {filepath}")


def cmd_list(tasks_dir: Path, lane: str = None):
    """List tasks, optionally filtered by lane."""
    tasks = _get_all_tasks(tasks_dir, lane)
    if not tasks:
        lane_msg = f" in '{lane}'" if lane else ""
        print(f"📂 No tasks found{lane_msg}.")
        return

    print(f"\n📋 Tasks ({len(tasks)}):\n")
    for filepath, task_lane, num, title in tasks:
        icon = LANE_ICONS.get(task_lane, "📌")
        print(f"  {icon} #{num:04d} {title:40} [{task_lane}]")


def cmd_get(tasks_dir: Path, task_number: int):
    """Print the full content of a specific task."""
    result = _find_task(tasks_dir, task_number)
    if not result:
        print(f"❌ Task #{task_number:04d} not found.")
        return

    filepath, lane = result
    icon = LANE_ICONS.get(lane, "📌")
    print(f"{icon} [{lane}] {filepath.name}\n")
    print(filepath.read_text(encoding='utf-8'))


def cmd_move(tasks_dir: Path, task_number: int, new_lane: str, note: str = None):
    """Move a task from one lane to another."""
    if new_lane not in VALID_LANES:
        print(f"❌ Invalid lane '{new_lane}'. Must be: {', '.join(VALID_LANES)}")
        return

    result = _find_task(tasks_dir, task_number)
    if not result:
        print(f"❌ Task #{task_number:04d} not found.")
        return

    filepath, old_lane = result
    if old_lane == new_lane:
        print(f"ℹ️  Task #{task_number:04d} is already in '{new_lane}'.")
        return

    # Move file
    new_dir = tasks_dir / new_lane
    new_dir.mkdir(parents=True, exist_ok=True)
    new_path = new_dir / filepath.name
    filepath.rename(new_path)

    # Append note if provided
    if note:
        with open(new_path, 'a', encoding='utf-8') as f:
            f.write(f"\n\n---\n**Moved {old_lane} → {new_lane}:** {note}\n")

    print(f"✅ Task #{task_number:04d} moved: {old_lane} → {new_lane}")
    print(f"   Path: {new_path}")


def cmd_search(tasks_dir: Path, query: str):
    """Search task contents by keyword."""
    query_lower = query.lower()
    results = []
    for filepath, lane, num, title in _get_all_tasks(tasks_dir):
        try:
            content = filepath.read_text(encoding='utf-8')
            if query_lower in content.lower():
                results.append((num, title, lane))
        except Exception:
            continue

    if not results:
        print(f"❌ No tasks matching '{query}'.")
    else:
        print(f"\n🔍 {len(results)} task(s) matching '{query}':\n")
        for num, title, lane in results:
            icon = LANE_ICONS.get(lane, "📌")
            print(f"  {icon} #{num:04d} {title:40} [{lane}]")


def cmd_board(tasks_dir: Path):
    """Print kanban board view."""
    print(f"\n{'='*60}")
    print(f"  📋 KANBAN BOARD")
    print(f"{'='*60}")

    total = 0
    done_count = 0

    for lane in VALID_LANES:
        tasks = _get_all_tasks(tasks_dir, lane)
        icon = LANE_ICONS.get(lane, "📌")
        print(f"\n{icon} {lane.upper()} ({len(tasks)})")
        print(f"{'─'*40}")

        if not tasks:
            print(f"   (empty)")
        else:
            for _, _, num, title in tasks:
                print(f"   #{num:04d} {title}")

        total += len(tasks)
        if lane == "done":
            done_count = len(tasks)

    print(f"\n{'='*60}")
    print(f"  Total: {total}  |  Done: {done_count}/{total}")
    print(f"{'='*60}\n")
=======
PROJECT_ROOT = _find_project_root()
DEFAULT_TASKS_FILE = PROJECT_ROOT / "tasks" / "tasks.json"

VALID_LANES = ["backlog", "todo", "in-progress", "done"]

LANE_ICONS = {
    "backlog": "📋",
    "todo": "📝",
    "in-progress": "🔨",
    "done": "✅",
}


class TaskManager:
    """JSON-backed kanban task manager."""

    def __init__(self, tasks_file: Path = None):
        self.tasks_file = tasks_file or DEFAULT_TASKS_FILE
        self.tasks_file.parent.mkdir(parents=True, exist_ok=True)
        self.data = self._load()

    def _load(self) -> Dict:
        if not self.tasks_file.exists():
            return {"next_id": 1, "tasks": []}
        with open(self.tasks_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save(self):
        with open(self.tasks_file, 'w', encoding='utf-8') as f:
            json.dump(self.data, f, indent=2, ensure_ascii=False)

    def create(self, title: str, objective: str = "", deliverables: List[str] = None,
               acceptance_criteria: List[str] = None, status: str = "todo") -> Dict:
        """Create a new task."""
        if status not in VALID_LANES:
            print(f"❌ Invalid status '{status}'. Must be: {', '.join(VALID_LANES)}")
            return {}

        task_id = self.data["next_id"]
        self.data["next_id"] = task_id + 1

        task = {
            "id": task_id,
            "title": title,
            "objective": objective,
            "deliverables": deliverables or [],
            "acceptance_criteria": acceptance_criteria or [],
            "status": status,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "notes": [],
        }

        self.data["tasks"].append(task)
        self._save()
        print(f"✅ Created task #{task_id}: {title} [{status}]")
        return task

    def list_tasks(self, status: str = None) -> List[Dict]:
        """List tasks, optionally filtered by status."""
        tasks = self.data["tasks"]
        if status:
            tasks = [t for t in tasks if t["status"] == status]
        return tasks

    def get_task(self, task_id: int) -> Optional[Dict]:
        """Get a single task by ID."""
        for t in self.data["tasks"]:
            if t["id"] == task_id:
                return t
        return None

    def move(self, task_id: int, new_status: str, note: str = None) -> bool:
        """Move a task to a new lane."""
        if new_status not in VALID_LANES:
            print(f"❌ Invalid status '{new_status}'. Must be: {', '.join(VALID_LANES)}")
            return False

        task = self.get_task(task_id)
        if not task:
            print(f"❌ Task #{task_id} not found.")
            return False

        old_status = task["status"]
        task["status"] = new_status
        task["updated_at"] = datetime.now().isoformat()
        if note:
            task["notes"].append({
                "text": note,
                "timestamp": datetime.now().isoformat(),
                "transition": f"{old_status} → {new_status}",
            })

        self._save()
        print(f"✅ Task #{task_id} moved: {old_status} → {new_status}")
        return True

    def update(self, task_id: int, title: str = None, objective: str = None, note: str = None) -> bool:
        """Update task fields."""
        task = self.get_task(task_id)
        if not task:
            print(f"❌ Task #{task_id} not found.")
            return False

        if title:
            task["title"] = title
        if objective:
            task["objective"] = objective
        if note:
            task["notes"].append({"text": note, "timestamp": datetime.now().isoformat()})
        task["updated_at"] = datetime.now().isoformat()

        self._save()
        print(f"✅ Task #{task_id} updated.")
        return True

    def delete(self, task_id: int) -> bool:
        """Delete a task."""
        task = self.get_task(task_id)
        if not task:
            print(f"❌ Task #{task_id} not found.")
            return False

        self.data["tasks"] = [t for t in self.data["tasks"] if t["id"] != task_id]
        self._save()
        print(f"✅ Task #{task_id} deleted.")
        return True

    def board(self):
        """Print kanban board view."""
        print(f"\n{'='*60}")
        print(f"  📋 KANBAN BOARD")
        print(f"{'='*60}")

        for lane in VALID_LANES:
            tasks_in_lane = [t for t in self.data["tasks"] if t["status"] == lane]
            icon = LANE_ICONS.get(lane, "📌")
            print(f"\n{icon} {lane.upper()} ({len(tasks_in_lane)})")
            print(f"{'─'*40}")

            if not tasks_in_lane:
                print(f"   (empty)")
            else:
                for t in tasks_in_lane:
                    print(f"   #{t['id']:03d} {t['title']}")

        total = len(self.data["tasks"])
        done = len([t for t in self.data["tasks"] if t["status"] == "done"])
        print(f"\n{'='*60}")
        print(f"  Total: {total}  |  Done: {done}/{total}")
        print(f"{'='*60}\n")


def _print_task(task: Dict):
    """Pretty-print a single task."""
    icon = LANE_ICONS.get(task["status"], "📌")
    print(f"\n{icon} Task #{task['id']:03d}: {task['title']}")
    print(f"   Status:    {task['status']}")
    print(f"   Objective: {task.get('objective', 'N/A')}")
    print(f"   Created:   {task['created_at']}")
    print(f"   Updated:   {task['updated_at']}")

    if task.get("deliverables"):
        print(f"   Deliverables:")
        for d in task["deliverables"]:
            print(f"     - {d}")

    if task.get("acceptance_criteria"):
        print(f"   Acceptance Criteria:")
        for a in task["acceptance_criteria"]:
            print(f"     - {a}")

    if task.get("notes"):
        print(f"   Notes ({len(task['notes'])}):")
        for n in task["notes"][-3:]:  # Show last 3
            ts = n.get("timestamp", "")[:16]
            transition = f" [{n['transition']}]" if "transition" in n else ""
            print(f"     [{ts}]{transition} {n['text']}")
>>>>>>> origin/main


def main():
    parser = argparse.ArgumentParser(description="Lightweight Kanban Task Manager")
<<<<<<< HEAD
    parser.add_argument("--dir", default=None, help="Root tasks directory (default: tasks/)")
=======
    parser.add_argument("--file", default=None, help="Path to tasks JSON file")
>>>>>>> origin/main
    subparsers = parser.add_subparsers(dest="command")

    # create
    create_p = subparsers.add_parser("create", help="Create a new task")
    create_p.add_argument("title", help="Task title")
<<<<<<< HEAD
    create_p.add_argument("--lane", default="todo", choices=VALID_LANES, help="Target lane (default: todo)")
    create_p.add_argument("--objective", default="", help="Task objective")
    create_p.add_argument("--acceptance", default="", help="Acceptance criteria")

    # list
    list_p = subparsers.add_parser("list", help="List tasks")
    list_p.add_argument("--lane", choices=VALID_LANES, help="Filter by lane")

    # get
    get_p = subparsers.add_parser("get", help="View a specific task")
    get_p.add_argument("task_number", type=int, help="Task number")

    # move
    move_p = subparsers.add_parser("move", help="Move task to a lane")
    move_p.add_argument("task_number", type=int, help="Task number")
    move_p.add_argument("new_lane", choices=VALID_LANES, help="Target lane")
    move_p.add_argument("--note", help="Transition note")

    # search
    search_p = subparsers.add_parser("search", help="Search tasks by keyword")
    search_p.add_argument("query", help="Search query")
=======
    create_p.add_argument("--objective", default="", help="Task objective")
    create_p.add_argument("--deliverables", nargs="+", default=[], help="Deliverable items")
    create_p.add_argument("--acceptance-criteria", nargs="+", default=[], help="Acceptance criteria")
    create_p.add_argument("--status", default="todo", choices=VALID_LANES, help="Initial status")

    # list
    list_p = subparsers.add_parser("list", help="List tasks")
    list_p.add_argument("--status", choices=VALID_LANES, help="Filter by status")

    # get
    get_p = subparsers.add_parser("get", help="Get task details")
    get_p.add_argument("task_id", type=int, help="Task ID")

    # move
    move_p = subparsers.add_parser("move", help="Move task to a lane")
    move_p.add_argument("task_id", type=int, help="Task ID")
    move_p.add_argument("new_status", choices=VALID_LANES, help="Target lane")
    move_p.add_argument("--note", help="Transition note")

    # update
    update_p = subparsers.add_parser("update", help="Update task fields")
    update_p.add_argument("task_id", type=int, help="Task ID")
    update_p.add_argument("--title", help="New title")
    update_p.add_argument("--objective", help="New objective")
    update_p.add_argument("--note", help="Add a note")

    # delete
    delete_p = subparsers.add_parser("delete", help="Delete a task")
    delete_p.add_argument("task_id", type=int, help="Task ID")
>>>>>>> origin/main

    # board
    subparsers.add_parser("board", help="Show kanban board")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

<<<<<<< HEAD
    tasks_dir = Path(args.dir) if args.dir else PROJECT_ROOT / "tasks"

    if args.command == "create":
        cmd_create(tasks_dir, args.title, args.lane, args.objective, args.acceptance)
    elif args.command == "list":
        cmd_list(tasks_dir, getattr(args, 'lane', None))
    elif args.command == "get":
        cmd_get(tasks_dir, args.task_number)
    elif args.command == "move":
        cmd_move(tasks_dir, args.task_number, args.new_lane, getattr(args, 'note', None))
    elif args.command == "search":
        cmd_search(tasks_dir, args.query)
    elif args.command == "board":
        cmd_board(tasks_dir)


if __name__ == "__main__":
    main()
=======
    tasks_file = Path(args.file) if args.file else None
    tm = TaskManager(tasks_file)

    if args.command == "create":
        tm.create(args.title, args.objective, args.deliverables, args.acceptance_criteria, args.status)

    elif args.command == "list":
        tasks = tm.list_tasks(args.status)
        if not tasks:
            status_msg = f" with status '{args.status}'" if args.status else ""
            print(f"📂 No tasks found{status_msg}.")
        else:
            for t in tasks:
                _print_task(t)

    elif args.command == "get":
        task = tm.get_task(args.task_id)
        if task:
            _print_task(task)
        else:
            print(f"❌ Task #{args.task_id} not found.")

    elif args.command == "move":
        tm.move(args.task_id, args.new_status, args.note)

    elif args.command == "update":
        tm.update(args.task_id, args.title, args.objective, args.note)

    elif args.command == "delete":
        tm.delete(args.task_id)

    elif args.command == "board":
        tm.board()


if __name__ == "__main__":
    main()
>>>>>>> origin/main
