#!/usr/bin/env python3
"""
task_manager.py — Lightweight Kanban Task Manager
==================================================

Purpose:
    Simple JSON-backed task board with lanes: backlog, todo, in-progress, done.
    Designed as a standalone replacement for the heavier cli.py task subsystem.

Layer: Plugin / Task-Manager

Usage:
    python3 task_manager.py create "Fix login bug" --objective "Resolve 401 errors"
    python3 task_manager.py list
    python3 task_manager.py list --status in-progress
    python3 task_manager.py move 3 in-progress
    python3 task_manager.py get 3
    python3 task_manager.py board
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).parent.resolve()
PLUGIN_ROOT = SCRIPT_DIR.parent.resolve()

# Find project root
def _find_project_root() -> Path:
    p = PLUGIN_ROOT
    for _ in range(10):
        if (p / ".git").exists() or (p / ".agent").exists():
            return p
        p = p.parent
    return Path.cwd()

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


def main():
    parser = argparse.ArgumentParser(description="Lightweight Kanban Task Manager")
    parser.add_argument("--file", default=None, help="Path to tasks JSON file")
    subparsers = parser.add_subparsers(dest="command")

    # create
    create_p = subparsers.add_parser("create", help="Create a new task")
    create_p.add_argument("title", help="Task title")
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

    # board
    subparsers.add_parser("board", help="Show kanban board")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

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
