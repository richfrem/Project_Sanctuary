#!/usr/bin/env python3
"""
track_budget.py - Real-World Token & Quota Budget Governance CLI
==============================================================

Purpose:
    Monitors and bounds token and turn usage across model backends
    (ChatGPT Plus, Claude Pro, Gemini Pro) to respect the Human Partner's
    real-world subscription quotas and prevent runaway loop consumption.

Usage:
    python3 track_budget.py status
    python3 track_budget.py record --task-id task-001 --backend codex --model gpt-6-astra --tokens-est 55658 --turns 1 --summary "Automaton architectural review"
    python3 track_budget.py check --profile claude-pro --planned-turns 5
"""

import os
import sys
import json
import uuid
import argparse
from pathlib import Path
from datetime import datetime, timezone


BUDGET_PROFILES = {
    "claude-pro": {
        "description": "Claude Pro ($20/mo) - ~45 messages / 5 hours",
        "soft_turn_limit_per_task": 12,
        "max_tokens_per_task": 60000,
        "recommended_batch_size": 3
    },
    "chatgpt-plus": {
        "description": "ChatGPT Plus ($20/mo) - ~40-80 messages / 3 hours",
        "soft_turn_limit_per_task": 15,
        "max_tokens_per_task": 80000,
        "recommended_batch_size": 4
    },
    "gemini-pro": {
        "description": "Gemini Pro / AGY ($20/mo) - 1500 RPD / 60 RPM",
        "soft_turn_limit_per_task": 25,
        "max_tokens_per_task": 150000,
        "recommended_batch_size": 5
    },
    "default": {
        "description": "Standard Balanced Budget",
        "soft_turn_limit_per_task": 15,
        "max_tokens_per_task": 80000,
        "recommended_batch_size": 3
    }
}


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def load_budget_ledger(root: Path) -> tuple[dict, Path]:
    ledger_path = root / ".agent" / "learning" / "budget_ledger.json"
    ledger_path.parent.mkdir(parents=True, exist_ok=True)

    default_data = {
        "schema_version": 1,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "total_tokens_consumed": 0,
        "total_turns": 0,
        "tasks": {},
        "recent_records": []
    }

    if ledger_path.exists():
        try:
            with open(ledger_path, "r", encoding="utf-8") as f:
                return json.load(f), ledger_path
        except Exception:
            pass

    return default_data, ledger_path


def main():
    parser = argparse.ArgumentParser(description="Token Budget & Quota Governance")
    subparsers = parser.add_subparsers(dest="subcommand", help="Budget subcommand")

    # Status
    status_parser = subparsers.add_parser("status", help="Show budget status and consumption")
    status_parser.add_argument("--format", choices=["text", "json"], default="text")

    # Record
    record_parser = subparsers.add_parser("record", help="Record token expenditure")
    record_parser.add_argument("--task-id", required=True, help="Task identifier")
    record_parser.add_argument("--backend", default="unknown", help="Model backend: claude, codex, agy, local")
    record_parser.add_argument("--model", default="default", help="Model name")
    record_parser.add_argument("--tokens-est", type=int, required=True, help="Estimated or reported tokens used")
    record_parser.add_argument("--turns", type=int, default=1, help="Turns consumed")
    record_parser.add_argument("--summary", default="", help="Task turn summary")
    record_parser.add_argument("--format", choices=["text", "json"], default="text")

    # Check
    check_parser = subparsers.add_parser("check", help="Check planned dispatch against budget")
    check_parser.add_argument("--profile", choices=list(BUDGET_PROFILES.keys()), default="default", help="Subscription profile")
    check_parser.add_argument("--planned-turns", type=int, default=1, help="Planned execution turns")
    check_parser.add_argument("--format", choices=["text", "json"], default="text")

    args = parser.parse_args()
    root = get_project_root()
    data, ledger_path = load_budget_ledger(root)

    if args.subcommand == "record":
        task_id = args.task_id
        tasks = data.setdefault("tasks", {})
        task_stat = tasks.setdefault(task_id, {
            "tokens_consumed": 0,
            "turns": 0,
            "last_backend": args.backend,
            "last_model": args.model
        })
        task_stat["tokens_consumed"] += args.tokens_est
        task_stat["turns"] += args.turns
        task_stat["last_backend"] = args.backend
        task_stat["last_model"] = args.model

        data["total_tokens_consumed"] = data.get("total_tokens_consumed", 0) + args.tokens_est
        data["total_turns"] = data.get("total_turns", 0) + args.turns
        data["updated_at"] = datetime.now(timezone.utc).isoformat()

        record_entry = {
            "record_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "task_id": task_id,
            "backend": args.backend,
            "model": args.model,
            "tokens_est": args.tokens_est,
            "turns": args.turns,
            "summary": args.summary
        }

        recent = data.setdefault("recent_records", [])
        recent.insert(0, record_entry)
        data["recent_records"] = recent[:30]

        with open(ledger_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        # Append to events.jsonl
        events_file = root / ".agent" / "learning" / "events.jsonl"
        events_file.parent.mkdir(parents=True, exist_ok=True)
        with open(events_file, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "event_type": "BUDGET_CONSUMED",
                **record_entry
            }) + "\n")

        if args.format == "json":
            print(json.dumps({"status": "success", "record": record_entry, "task_totals": task_stat}, indent=2))
        else:
            print(f"Recorded Budget Consumption for Task [{task_id}]:")
            print(f"  Backend/Model:   {args.backend} ({args.model})")
            print(f"  Tokens Added:    {args.tokens_est:,}")
            print(f"  Task Total:      {task_stat['tokens_consumed']:,} tokens across {task_stat['turns']} turns")
            print(f"  Global Total:    {data['total_tokens_consumed']:,} tokens")

    elif args.subcommand == "check":
        prof = BUDGET_PROFILES.get(args.profile, BUDGET_PROFILES["default"])
        soft_limit = prof["soft_turn_limit_per_task"]
        allowed = args.planned_turns <= soft_limit
        warning = "" if allowed else f"Planned turns ({args.planned_turns}) exceeds soft limit ({soft_limit}) for {args.profile}."

        result = {
            "profile": args.profile,
            "description": prof["description"],
            "planned_turns": args.planned_turns,
            "soft_turn_limit": soft_limit,
            "status": "APPROVED" if allowed else "WARNING",
            "warning": warning
        }

        if args.format == "json":
            print(json.dumps(result, indent=2))
        else:
            print(f"Budget Check [{args.profile}]: {result['status']}")
            print(f"  {prof['description']}")
            print(f"  Planned: {args.planned_turns} turns (Soft Limit: {soft_limit})")
            if warning:
                print(f"  [!] {warning}")

    else:  # status (default)
        if getattr(args, "format", "text") == "json":
            print(json.dumps(data, indent=2))
        else:
            print("=== Project Sanctuary Token & Budget Governance ===")
            print(f"Cumulative Tokens Consumed: {data.get('total_tokens_consumed', 0):,}")
            print(f"Cumulative Execution Turns: {data.get('total_turns', 0)}")
            print(f"Active Tasks Tracked:       {len(data.get('tasks', {}))}")
            tasks = data.get("tasks", {})
            if tasks:
                print("\nTask Breakdown:")
                for tid, tinfo in list(tasks.items())[:5]:
                    print(f"  - [{tid}] {tinfo['tokens_consumed']:,} tokens ({tinfo['turns']} turns) - {tinfo['last_backend']}/{tinfo['last_model']}")


if __name__ == "__main__":
    main()
