#============================================
# scripts/domain_cli.py
# Purpose: CLI Orchestrator for Sanctuary Domain Logic.
# Role: Administrative Utility for direct file-system operations.
# Reference: ADR-066 / Protocol 122 / Protocol 128
#
# CHRONICLE EXAMPLES:
#   python3 scripts/domain_cli.py chronicle list --limit 15
#   python3 scripts/domain_cli.py chronicle search "Egyptian Labyrinth"
#
# TASK EXAMPLES:
#   python3 scripts/domain_cli.py task list --status in-progress
#   python3 scripts/domain_cli.py task list --status done
#   python3 scripts/domain_cli.py task update-status 42 done --notes "RAG audit pass"
#
# ADR EXAMPLES:
#   python3 scripts/domain_cli.py adr list --status ACCEPTED
#   python3 scripts/domain_cli.py adr search "Gateway SSE"
#
# PROTOCOL EXAMPLES:
#   python3 scripts/domain_cli.py protocol list --status CANONICAL
#   python3 scripts/domain_cli.py protocol search "Mnemonic Cortex"
#============================================

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
from mcp_servers.task.models import taskstatus
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
    list_chron = chron_subs.add_parser("list")
    list_chron.add_argument("--limit", type=int, default=10)
    search_chron = chron_subs.add_parser("search")
    search_chron.add_argument("query")

    # --- TASK ---
    task_parser = subparsers.add_parser("task", help="Manage tasks")
    task_subs = task_parser.add_subparsers(dest="action")
    list_task = task_subs.add_parser("list")
    list_task.add_argument("--status", help="Task status (backlog, todo, in-progress, done)")
    update_task = task_subs.add_parser("update-status")
    update_task.add_argument("number", type=int)
    update_task.add_argument("new_status")
    update_task.add_argument("--notes", required=True)

    # --- ADR ---
    adr_parser = subparsers.add_parser("adr", help="Manage ADRs")
    adr_subs = adr_parser.add_subparsers(dest="action")
    list_adr = adr_subs.add_parser("list")
    list_adr.add_argument("--status")
    search_adr = adr_subs.add_parser("search")
    search_adr.add_argument("query")

    # --- PROTOCOL ---
    prot_parser = subparsers.add_parser("protocol", help="Manage Protocols")
    prot_subs = prot_parser.add_subparsers(dest="action")
    list_prot = prot_subs.add_parser("list")
    list_prot.add_argument("--status")
    search_prot = prot_subs.add_parser("search")
    search_prot.add_argument("query")

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

        elif args.subcommand == "adr":
            ops = ADROperations(os.path.join(PROJECT_ROOT, "ADRs"))
            if args.action == "list":
                res = ops.list_adrs(status=args.status.upper() if args.status else None)
                for a in res: print(f"[{a['number']:03d}] {a['title']} [{a['status']}]")
            elif args.action == "search":
                res = ops.search_adrs(args.query)
                for a in res: print(f"[{a['number']:03d}] {a['title']}")

        elif args.subcommand == "protocol":
            ops = ProtocolOperations(os.path.join(PROJECT_ROOT, "01_PROTOCOLS"))
            if args.action == "list":
                res = ops.list_protocols(status=args.status.upper() if args.status else None)
                for p in res: print(f"[{p['number']:03d}] {p['title']} [{p['status']}]")
            elif args.action == "search":
                res = ops.search_protocols(args.query)
                for p in res: print(f"[{p['number']:03d}] {p['title']}")

    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()