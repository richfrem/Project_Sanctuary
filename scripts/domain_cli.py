#!/usr/bin/env python3
"""
domain_cli.py (CLI)
=====================================

Purpose:
    CLI Orchestrator for Sanctuary Domain Logic (Chronicle, Task, ADR, Protocol).
    Replaces generic MCP servers with a direct Python CLI interface for
    CRUD operations and file management.

Layer: Orchestrator

Usage Examples:
    python scripts/domain_cli.py chronicle list --limit 15
    python scripts/domain_cli.py task create "Fix Bug" --objective "Fix X" --deliverables "Code" --acceptance-criteria "Test Pass"
    python scripts/domain_cli.py adr get 85
    python scripts/domain_cli.py protocol search "Mnemonic"

CLI Arguments:
    chronicle       : Manage Chronicle Entries
    task            : Manage Tasks
    adr             : Manage Architecture Decision Records
    protocol        : Manage Protocols

Input Files:
    - 00_CHRONICLE/ENTRIES/
    - tasks/
    - ADRs/
    - 01_PROTOCOLS/

Output:
    - Markdown files in respective directories

Key Functions:
    - ChronicleOperations.create_entry()
    - TaskOperations.create_task()
    - ADROperations.create_adr()
    - ProtocolOperations.create_protocol()

Script Dependencies:
    - mcp_servers/chronicle/operations.py
    - mcp_servers/task/operations.py
    - mcp_servers/adr/operations.py
    - mcp_servers/protocol/operations.py

Consumed by:
    - User (Manual CLI)
    - Agent (via Tool Calls)
"""

import argparse
import sys
import os
from pathlib import Path

# 🛠️ PATH INJECTION: Fixes ModuleNotFoundError
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from mcp_servers.lib.env_helper import get_env_variable
from mcp_servers.lib.path_utils import find_project_root
from mcp_servers.task.operations import TaskOperations
from mcp_servers.task.models import taskstatus, TaskPriority
from mcp_servers.adr.operations import ADROperations
from mcp_servers.chronicle.operations import ChronicleOperations
from mcp_servers.protocol.operations import ProtocolOperations

PROJECT_ROOT = get_env_variable("PROJECT_ROOT", required=False) or find_project_root()
PROJECT_ROOT_PATH = Path(PROJECT_ROOT)

def main():
    parser = argparse.ArgumentParser(description="Sanctuary Domain CLI")
    subparsers = parser.add_subparsers(dest="subcommand", help="Available Clusters")

    # --- CHRONICLE ---
    chron_parser = subparsers.add_parser("chronicle", help="Manage the Chronicle")
    chron_subs = chron_parser.add_subparsers(dest="action")
    
    # list
    list_chron = chron_subs.add_parser("list")
    list_chron.add_argument("--limit", type=int, default=10)
    
    # search
    search_chron = chron_subs.add_parser("search")
    search_chron.add_argument("query")
    
    # get
    get_chron = chron_subs.add_parser("get")
    get_chron.add_argument("number", type=int)

    # create
    create_chron = chron_subs.add_parser("create")
    create_chron.add_argument("title")
    create_chron.add_argument("--content", required=True)
    create_chron.add_argument("--author", default="AI Assistant")
    create_chron.add_argument("--status", default="draft")
    create_chron.add_argument("--classification", default="internal")

    # --- TASK ---
    task_parser = subparsers.add_parser("task", help="Manage tasks")
    task_subs = task_parser.add_subparsers(dest="action")
    
    # list
    list_task = task_subs.add_parser("list")
    list_task.add_argument("--status", help="Task status (backlog, todo, in-progress, done)")
    
    # update-status
    update_task = task_subs.add_parser("update-status")
    update_task.add_argument("number", type=int)
    update_task.add_argument("new_status")
    update_task.add_argument("--notes", required=True)

    # get
    get_task = task_subs.add_parser("get")
    get_task.add_argument("number", type=int)

    # create
    create_task = task_subs.add_parser("create")
    create_task.add_argument("title")
    create_task.add_argument("--objective", required=True)
    create_task.add_argument("--deliverables", nargs="+", required=True)
    create_task.add_argument("--acceptance-criteria", nargs="+", required=True)
    create_task.add_argument("--priority", default="MEDIUM")
    create_task.add_argument("--status", default="TODO")
    create_task.add_argument("--lead", default="Unassigned")

    # --- ADR ---
    adr_parser = subparsers.add_parser("adr", help="Manage ADRs")
    adr_subs = adr_parser.add_subparsers(dest="action")
    list_adr = adr_subs.add_parser("list")
    list_adr.add_argument("--status")
    search_adr = adr_subs.add_parser("search")
    search_adr.add_argument("query")
    
    get_adr = adr_subs.add_parser("get")
    get_adr.add_argument("number", type=int)
    
    create_adr = adr_subs.add_parser("create")
    create_adr.add_argument("title")
    create_adr.add_argument("--context", required=True)
    create_adr.add_argument("--decision", required=True)
    create_adr.add_argument("--consequences", required=True)
    create_adr.add_argument("--status", default="proposed")

    # --- PROTOCOL ---
    prot_parser = subparsers.add_parser("protocol", help="Manage Protocols")
    prot_subs = prot_parser.add_subparsers(dest="action")
    list_prot = prot_subs.add_parser("list")
    list_prot.add_argument("--status")
    search_prot = prot_subs.add_parser("search")
    search_prot.add_argument("query")
    
    get_prot = prot_subs.add_parser("get")
    get_prot.add_argument("number", type=int)
    
    create_prot = prot_subs.add_parser("create")
    create_prot.add_argument("title", help="Protocol Title (e.g., 'Git Flow')")
    create_prot.add_argument("--content", required=True, help="Full markdown content")
    create_prot.add_argument("--version", default="1.0")
    create_prot.add_argument("--status", default="PROPOSED")
    create_prot.add_argument("--authority", default="Council")
    create_prot.add_argument("--classification", default="Blue")

    args = parser.parse_args()
    if not args.subcommand:
        parser.print_help()
        return

    try:
        if args.subcommand == "chronicle":
            ops = ChronicleOperations(os.path.join(PROJECT_ROOT, "00_CHRONICLE/ENTRIES"))
            if args.action == "list":
                res = ops.list_entries(limit=args.limit)
                for e in res: print(f"[{e['number']:03d}] {e['title']} ({e['date']})")
            elif args.action == "search":
                res = ops.search_entries(args.query)
                for e in res: print(f"[{e['number']:03d}] {e['title']}")
            elif args.action == "get":
                res = ops.get_entry(args.number)
                print(f"[{res['number']:03d}] {res['title']}")
                print("-" * 40)
                print(res['content'])
            elif args.action == "create":
                res = ops.create_entry(
                    title=args.title, 
                    content=str(args.content).replace("\\n", "\n"), 
                    author=args.author, 
                    status=args.status,
                    classification=args.classification
                )
                print(f"✅ Created Chronicle Entry #{res['entry_number']:03d}: {res['file_path']}")

        elif args.subcommand == "task":
            ops = TaskOperations(PROJECT_ROOT_PATH)
            if args.action == "list":
                # FIXED: Argument is 'status', not 'status_filter'
                status_obj = taskstatus(args.status) if args.status else None
                res = ops.list_tasks(status=status_obj)
                for t in res: print(f"[{t['number']:03d}] {t['title']} ({t['status']})")
            elif args.action == "update-status":
                ops.update_task_status(args.number, taskstatus(args.new_status), args.notes)
                print(f"✅ Task {args.number} moved to {args.new_status}")
            elif args.action == "get":
                res = ops.get_task(args.number)
                if not res:
                    print(f"❌ Task {args.number} not found")
                    sys.exit(1)
                print(f"[{res['number']:03d}] {res['title']}")
                print(f"Status: {res['status']} | Priority: {res['priority']} | Lead: {res['lead']}")
                print("-" * 40)
                print(res['content'])
            elif args.action == "create":
                res = ops.create_task(
                    title=args.title,
                    objective=str(args.objective).replace("\\n", "\n"),
                    deliverables=args.deliverables,
                    acceptance_criteria=args.acceptance_criteria,
                    priority=TaskPriority(args.priority.capitalize()),
                    status=taskstatus(args.status.lower()),
                    lead=args.lead
                )
                if res.status == "success":
                     print(f"✅ Created Task #{res.task_number:03d} at {res.file_path}")
                else:
                     print(f"❌ Creation failed: {res.message}")


        elif args.subcommand == "adr":
            ops = ADROperations(os.path.join(PROJECT_ROOT, "ADRs"))
            if args.action == "list":
                res = ops.list_adrs(status=args.status.upper() if args.status else None)
                for a in res: print(f"[{a['number']:03d}] {a['title']} [{a['status']}]")
            elif args.action == "search":
                res = ops.search_adrs(args.query)
                for a in res: print(f"[{a['number']:03d}] {a['title']}")
            elif args.action == "get":
                res = ops.get_adr(args.number)
                print(f"ADR-{res['number']:03d}: {res['title']}")
                print(f"Status: {res['status']}")
                print("-" * 40)
                print(f"# Context\n{res['context']}\n")
                print(f"# Decision\n{res['decision']}\n")
                print(f"# Consequences\n{res['consequences']}")
            elif args.action == "create":
                res = ops.create_adr(
                    title=args.title,
                    context=str(args.context).replace("\\n", "\n"),
                    decision=str(args.decision).replace("\\n", "\n"),
                    consequences=str(args.consequences).replace("\\n", "\n"),
                    status=args.status
                )
                print(f"✅ Created ADR-{res['adr_number']:03d} at {res['file_path']}")

        elif args.subcommand == "protocol":
            ops = ProtocolOperations(os.path.join(PROJECT_ROOT, "01_PROTOCOLS"))
            if args.action == "list":
                res = ops.list_protocols(status=args.status.upper() if args.status else None)
                for p in res: print(f"[{p['number']:03d}] {p['title']} [{p['status']}]")
            elif args.action == "search":
                res = ops.search_protocols(args.query)
                for p in res: print(f"[{p['number']:03d}] {p['title']}")
            elif args.action == "get":
                res = ops.get_protocol(args.number)
                print(f"Protocol-{res['number']:03d}: {res['title']}")
                print(f"v{res['version']} | {res['status']} | {res['classification']}")
                print("-" * 40)
                print(res['content'])
            elif args.action == "create":
                res = ops.create_protocol(
                    number=None, # Auto-generate
                    title=args.title,
                    status=args.status,
                    classification=args.classification,
                    version=args.version,
                    authority=args.authority,
                    content=str(args.content).replace("\\n", "\n")
                )
                print(f"✅ Created Protocol-{res['protocol_number']:03d} at {res['file_path']}")

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()