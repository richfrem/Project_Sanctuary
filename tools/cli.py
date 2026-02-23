#!/usr/bin/env python3
"""
cli.py - Project Sanctuary Command Line Interface
==================================================

Purpose:
    Main entry point for the Project Sanctuary Command System.
    Provides unified access to all core operations:
    - Protocol 128 Learning Loop (Debrief, Snapshot, Persist, Guardian)
    - RAG Cortex Operations (Ingest, Query, Stats, Cache)
    - Context Bundling & Manifest Management
    - Tool Discovery & Inventory
    - Workflow Orchestration
    - Evolutionary Metrics (Protocol 131)
    - RLM Distillation (Protocol 132)
    - Domain Entity Management (Chronicle, Task, ADR, Protocol)
    - Fine-Tuned Model Interaction (Forge)

Layer: Tools / Orchestrator

Commands:
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PROTOCOL 128 - LEARNING LOOP
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    debrief         : Phase I - Run Learning Debrief (orientation)
    snapshot        : Phase V - Capture context snapshot (seal, audit, guardian, bootstrap)
    persist-soul    : Phase VI - Broadcast learnings to Hugging Face
    persist-soul-full : Full JSONL regeneration and HF deployment (ADR 081)
    guardian        : Bootloader operations (wakeup, snapshot)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # RAG CORTEX
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ingest          : RAG ingestion (full or incremental)
    query           : Semantic search against vector DB
    stats           : View RAG health and collection statistics
    cache-stats     : View semantic cache efficiency metrics
    cache-warmup    : Pre-populate cache with common queries

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # CONTEXT BUNDLING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    init-context    : Quick setup - initialize manifest and auto-bundle
    manifest        : Full manifest management (init, add, remove, update, search, list, bundle)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TOOLS & WORKFLOWS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    tools           : Discover and manage CLI tools (list, search, add, update, remove)
    workflow        : Agent lifecycle management (start, retrospective, end)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # EVOLUTION & RLM
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    evolution       : Evolutionary metrics - fitness, depth, scope (Protocol 131)
    rlm-distill     : Distill semantic summaries from files (Protocol 132)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DOMAIN ENTITY MANAGEMENT
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    chronicle       : Manage Chronicle Entries (list, search, get, create, update)
    task            : Manage Tasks (list, get, create, update-status, search, update)
    adr             : Manage Architecture Decision Records (list, search, get, create, update-status)
    protocol        : Manage Protocols (list, search, get, create, update)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # FINE-TUNED MODEL
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    forge           : Sanctuary Fine-Tuned Model (query, status) - requires ollama

Usage Examples:

    # Learning Loop (Protocol 128)
    python tools/cli.py debrief --hours 24
    python tools/cli.py snapshot --type seal
    python tools/cli.py persist-soul
    python tools/cli.py guardian wakeup --mode HOLISTIC

    # RAG Cortex
    python tools/cli.py ingest --incremental --hours 24
    python tools/cli.py query "What is Protocol 128?"
    python tools/cli.py stats --samples

    # Context Bundling
    python tools/cli.py init-context --target MyFeature --type generic
    python tools/cli.py manifest init --bundle-title MyBundle --type learning
    python tools/cli.py manifest bundle

    # Tools & Workflows
    python tools/cli.py tools list
    python tools/cli.py tools search "ingestion"
    python tools/cli.py workflow start --name workflow-start --target MyFeature
    python tools/cli.py workflow retrospective
    python tools/cli.py workflow end "feat: implemented feature X"

    # Evolution & RLM
    python tools/cli.py evolution fitness --file docs/my-document.md
    python tools/cli.py rlm-distill --profile project plugins/adr-manager/README.md
    python tools/cli.py rlm-distill --profile tools plugins/rlm-factory/skills/rlm-curator/scripts/distiller.py

    # Domain Entities
    python tools/cli.py chronicle list --limit 10
    python tools/cli.py chronicle create "Title" --content "Content" --author "Author"
    python tools/cli.py chronicle update 5 --title "New Title" --reason "Fix typo"
    python tools/cli.py task list --status in-progress
    python tools/cli.py task create "Title" --objective "Goal" --deliverables item1 item2 --acceptance-criteria done1
    python tools/cli.py task update-status 5 done --notes "Completed"
    python tools/cli.py task search "migration"
    python tools/cli.py adr list --status proposed
    python tools/cli.py adr create "Title" --context "Why" --decision "What" --consequences "Impact"
    python tools/cli.py adr update-status 85 accepted --reason "Approved by council"
    python tools/cli.py protocol list
    python tools/cli.py protocol create "Title" --content "Content" --status PROPOSED
    python tools/cli.py protocol update 128 --status ACTIVE --reason "Ratified"

    # Fine-Tuned Model (requires ollama)
    python tools/cli.py forge status
    python tools/cli.py forge query "What are the core principles of Project Sanctuary?"
"""
import sys
import argparse
import json
import os
import subprocess
from pathlib import Path
import re

# Add project root to sys.path to import utils
CLI_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = CLI_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Inline path resolver (formerly in tools.utils.path_resolver)
def resolve_path(relative_path: str, base_path: Path = None) -> str:
    """
    Resolves a relative path against the project root or a specified base path.
    Handles potential inconsistencies in path separators and ensures a clean, absolute path.
    """
    if base_path is None:
        base_path = PROJECT_ROOT
    full_path = base_path / relative_path
    return str(full_path.resolve())

# Resolve Directories
# SEARCH_DIR removed — no longer used (migrated to plugins)
# DOCS_DIR removed — no longer used (migrated to plugins)
# TRACKING_DIR removed — no longer used (migrated to plugins)
# SHARED_DIR removed — no longer used (migrated to plugins)
RETRIEVE_DIR = PROJECT_ROOT / "plugins/context-bundler/scripts"
INVENTORIES_DIR = PROJECT_ROOT / "plugins/tool-inventory/skills/tool-inventory/scripts"
RLM_DIR = PROJECT_ROOT / "plugins/rlm-factory/skills/rlm-curator/scripts"
ORCHESTRATOR_DIR = PROJECT_ROOT / "plugins" / "agent-loops" / "skills" / "orchestrator" / "scripts"

# Add directories to sys.path for internal imports
for d in [RETRIEVE_DIR, INVENTORIES_DIR, RLM_DIR, ORCHESTRATOR_DIR]:
    if str(d) not in sys.path:
        sys.path.append(str(d))



# ─────────────────────────────────────────────────────────────
# Plugin Script Paths — Zero mcp_servers dependencies.
# All commands delegate to self-contained plugin scripts.
# ─────────────────────────────────────────────────────────────

# Guardian Onboarding (Protocol 128: Debrief, Snapshot, Persist, Wakeup)
GUARDIAN_SCRIPTS  = PROJECT_ROOT / "plugins/guardian-onboarding/scripts"

# Vector DB (RAG Ingestion and Semantic Query)
VECTOR_DB_SCRIPTS = PROJECT_ROOT / "plugins/vector-db/skills/vector-db-agent/scripts"

# Domain Entity Managers
CHRONICLE_SCRIPTS = PROJECT_ROOT / "plugins/chronicle-manager/skills/chronicle-agent/scripts"
PROTOCOL_SCRIPTS  = PROJECT_ROOT / "plugins/protocol-manager/skills/protocol-agent/scripts"
ADR_SCRIPTS       = PROJECT_ROOT / "plugins/adr-manager/skills/adr-management/scripts"
TASK_SCRIPTS      = PROJECT_ROOT / "plugins/task-manager/skills/task-agent/scripts"

# ADR 090: Iron Core Definitions
IRON_CORE_PATHS = [
    "01_PROTOCOLS",
    "ADRs",
    "cognitive_continuity_policy.md"
]

def verify_iron_core(root_path):
    """
    Verifies that Iron Core paths have not been tampered with (uncommitted/unstaged changes).
    ADR 090 (Evolution-Aware):
    - Unstaged changes (Dirty Worktree) -> VIOLATION (Drift)
    - Staged changes (Index) -> ALLOWED (Evolution)
    """
    violations = []
    try:
        # Check for modifications in Iron Core paths
        cmd = ["git", "status", "--porcelain"] + IRON_CORE_PATHS
        result = subprocess.run(
            cmd, 
            cwd=root_path, 
            capture_output=True, 
            text=True, 
            check=False
        )
        
        if result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                if len(line.strip()) < 3: 
                    continue
                    
                status_code = line[:2]
                
                # Check Worktree Status (2nd character)
                # ' ' = Unmodified in worktree (changes are staged or clean)
                # 'M' = Modified in worktree
                # 'D' = Deleted in worktree
                # '?' = Untracked
                worktree_status = status_code[1]
                
                # Violation if:
                # 1. Untracked ('??') inside Iron Core path
                # 2. Modified in Worktree ('M')
                # 3. Deleted in Worktree ('D')
                if status_code == '??' or worktree_status in ['M', 'D']:
                    violations.append(f"{line.strip()} (Unstaged/Dirty - Please 'git add' to authorize)")
                
    except Exception as e:
        return False, [f"Error checking Iron Core: {str(e)}"]
        
    return len(violations) == 0, violations


def main():
    parser = argparse.ArgumentParser(description="Recursive Business Rule Discovery CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Tools Command (Tool Discovery)
    tools_parser = subparsers.add_parser("tools", help="Discover and Manage CLI Tools")
    tools_subparsers = tools_parser.add_subparsers(dest="tools_action")
    
    tools_list = tools_subparsers.add_parser("list", help="List all available tools")
    tools_list.add_argument("--category", help="Filter by category")
    
    tools_search = tools_subparsers.add_parser("search", help="Search for tools")
    tools_search.add_argument("keyword", help="Keyword (name/desc)")
    
    tools_add = tools_subparsers.add_parser("add", help="Register a new tool")
    tools_add.add_argument("--path", required=True, help="Path to tool script")
    tools_add.add_argument("--category", help="Category")
    
    tools_update = tools_subparsers.add_parser("update", help="Update tool entry")
    tools_update.add_argument("--path", required=True, help="Path/Name of tool")
    tools_update.add_argument("--desc", help="New description")
    
    tools_remove = tools_subparsers.add_parser("remove", help="Remove tool from inventory")
    tools_remove.add_argument("--path", required=True, help="Path/Name of tool")

    # Command: ingest
    ingest_parser = subparsers.add_parser("ingest", help="Perform full ingestion")
    ingest_parser.add_argument("--no-purge", action="store_false", dest="purge", help="Skip purging DB")
    ingest_parser.add_argument("--dirs", nargs="+", help="Specific directories to ingest")
    ingest_parser.add_argument("--incremental", action="store_true", help="Incremental ingestion mode")
    ingest_parser.add_argument("--hours", type=int, default=24, help="Hours to look back (for incremental mode)")

    # Command: stats
    stats_parser = subparsers.add_parser("stats", help="Get RAG health and statistics")
    stats_parser.add_argument("--samples", action="store_true", help="Include sample documents")
    stats_parser.add_argument("--sample-count", type=int, default=5, help="Number of samples to include")

    # Command: query
    query_parser = subparsers.add_parser("query", help="Perform semantic search query")
    query_parser.add_argument("query_text", help="Search query string")
    query_parser.add_argument("--max-results", type=int, default=5, help="Maximum results to return")
    query_parser.add_argument("--use-cache", action="store_true", help="Use semantic cache")

    # Command: cache-stats
    subparsers.add_parser("cache-stats", help="Get cache statistics")

    # Command: cache-warmup
    warmup_parser = subparsers.add_parser("cache-warmup", help="Pre-populate cache with genesis queries")
    warmup_parser.add_argument("--queries", nargs="+", help="Custom queries to cache")

    # Command: evolution (Protocol 131)
    evolution_parser = subparsers.add_parser("evolution", help="Evolutionary metrics (Protocol 131)")
    evolution_sub = evolution_parser.add_subparsers(dest="evolution_subcommand", help="Evolution subcommands")
    
    # fitness
    fit_parser = evolution_sub.add_parser("fitness", help="Calculate full fitness vector")
    fit_parser.add_argument("content", nargs="?", help="Text content to evaluate")
    fit_parser.add_argument("--file", help="Read content from file")
    
    # depth
    depth_parser = evolution_sub.add_parser("depth", help="Evaluate technical depth")
    depth_parser.add_argument("content", nargs="?", help="Text content to evaluate")
    depth_parser.add_argument("--file", help="Read content from file")
    
    # scope
    scope_parser = evolution_sub.add_parser("scope", help="Evaluate architectural scope")
    scope_parser.add_argument("content", nargs="?", help="Text content to evaluate")
    scope_parser.add_argument("--file", help="Read content from file")

    # Command: rlm-distill (Protocol 132)
    rlm_parser = subparsers.add_parser("rlm-distill", aliases=["rlm-test"], help="Distill semantic summaries")
    rlm_parser.add_argument("target", help="File or folder path to distill")
    rlm_parser.add_argument("--profile", default="project", choices=["project", "tools"], help="RLM profile: 'project' (docs/markdown) or 'tools' (code/scripts)")


    # Init-Context Command: Quick setup - initializes manifest and auto-bundles
    context_parser = subparsers.add_parser("init-context", help="Initialize manifest and generate first bundle")
    context_parser.add_argument("--target", required=True, help="Target ID")
    context_parser.add_argument("--type", choices=[
        'generic', 'context-bundler', 'tool', 'workflow', 'docs', 'adr', 'spec',
        'learning', 'learning-audit', 'learning-audit-core', 'red-team', 'guardian', 'bootstrap'
    ], help="Artifact Type")

    # Manifest Command: Full manifest management (init, add, remove, update, search, list, bundle)
    manifest_parser = subparsers.add_parser("manifest", help="Manage context manifest")
    # Global args for manifest subcommands? No, must add to each subparser unless we structure differently.
    # To keep simple, we add --base to each action that supports it.
    
    manifest_subparsers = manifest_parser.add_subparsers(dest="manifest_action")
    
    man_init = manifest_subparsers.add_parser("init", help="Init from base manifest")
    man_init.add_argument("--bundle-title", required=True, help="Title for the bundle")
    man_init.add_argument("--type", choices=[
        'generic', 'context-bundler', 'tool', 'workflow', 'docs', 'adr', 'spec',
        'learning', 'learning-audit', 'learning-audit-core', 'red-team', 'guardian', 'bootstrap'
    ], help="Artifact Type (Optional if resolvable)")
    man_init.add_argument("--manifest", help="Custom manifest path")

    man_add = manifest_subparsers.add_parser("add", help="Add file to manifest")
    man_add.add_argument("--path", required=True)
    man_add.add_argument("--note", default="")
    man_add.add_argument("--base", help="Target base manifest type")
    man_add.add_argument("--manifest", help="Custom manifest path")
    
    man_remove = manifest_subparsers.add_parser("remove", help="Remove file from manifest")
    man_remove.add_argument("--path", required=True)
    man_remove.add_argument("--base", help="Target base manifest type")
    man_remove.add_argument("--manifest", help="Custom manifest path")

    man_update = manifest_subparsers.add_parser("update", help="Update file in manifest")
    man_update.add_argument("--path", required=True)
    man_update.add_argument("--note")
    man_update.add_argument("--new-path")
    man_update.add_argument("--base", help="Target base manifest type")
    man_update.add_argument("--manifest", help="Custom manifest path")
    
    man_search = manifest_subparsers.add_parser("search", help="Search in manifest")
    man_search.add_argument("pattern")
    man_search.add_argument("--base", help="Target base manifest type")
    man_search.add_argument("--manifest", help="Custom manifest path")
    
    man_list = manifest_subparsers.add_parser("list", help="List manifest contents")
    man_list.add_argument("--base", help="Target base manifest type")
    man_list.add_argument("--manifest", help="Custom manifest path")
    
    man_bundle = manifest_subparsers.add_parser("bundle", help="Regenerate bundle from manifest")
    man_bundle.add_argument("--output", help="Optional output path")
    man_bundle.add_argument("--base", help="Target base manifest type")
    man_bundle.add_argument("--manifest", help="Custom manifest path")

    # Snapshot Command: Protocol 128 memory bundles for session continuity
    snapshot_parser = subparsers.add_parser("snapshot", help="Generate Protocol 128 context snapshots")
    snapshot_parser.add_argument("--type", required=True, choices=[
        'seal', 'learning_audit', 'audit', 'guardian', 'bootstrap'
    ], help="Snapshot type")
    snapshot_parser.add_argument("--manifest", help="Custom manifest path (overrides default)")
    snapshot_parser.add_argument("--output", help="Output path (default: based on type)")
    snapshot_parser.add_argument("--context", help="Strategic context for the snapshot")
    snapshot_parser.add_argument("--override-iron-core", action="store_true", help="⚠️ Override Iron Core check (Requires ADR 090 Amendment)")

    # Debrief Command (Protocol 128 Phase I)
    debrief_parser = subparsers.add_parser("debrief", help="Run Learning Debrief (Protocol 128 Phase I)")
    debrief_parser.add_argument("--hours", type=int, default=24, help="Lookback window (hours)")
    debrief_parser.add_argument("--output", help="Output file path (default: stdout)")

    # Guardian Command: Bootloader operations for session startup
    guardian_parser = subparsers.add_parser("guardian", help="Guardian Bootloader Operations")
    guardian_parser.add_argument("--manifest", help="Custom manifest path")
    guardian_subparsers = guardian_parser.add_subparsers(dest="guardian_action")
    
    g_wakeup = guardian_subparsers.add_parser("wakeup", help="Generate Guardian Boot Digest")
    g_wakeup.add_argument("--mode", default="HOLISTIC", help="Wakeup mode")
    
    g_snapshot = guardian_subparsers.add_parser("snapshot", help="Capture Guardian Session Pack")
    g_snapshot.add_argument("--context", help="Strategic context")

    # Command: bootstrap-debrief (Fresh Repo Onboarding)
    bootstrap_parser = subparsers.add_parser("bootstrap-debrief", help="Generate onboarding context packet for fresh repo setup")
    bootstrap_parser.add_argument("--manifest", default=".agent/learning/bootstrap_manifest.json", help="Path to bootstrap manifest")
    bootstrap_parser.add_argument("--output", default=".agent/learning/bootstrap_packet.md", help="Output path for the packet")

    # Persist Soul Command (Protocol 128 Phase VI)
    ps_parser = subparsers.add_parser("persist-soul", help="Broadcast learnings to Hugging Face")
    ps_parser.add_argument("--snapshot", help="Specific snapshot path (default: active seal)")
    ps_parser.add_argument("--valence", type=float, default=0.5, help="Session valence (0.0-1.0)")
    ps_parser.add_argument("--uncertainty", type=float, default=0.0, help="Logic confidence")
    ps_parser.add_argument("--full-sync", action="store_true", help="Sync entire learning directory")

    # Persist Soul Full Command (ADR 081)
    subparsers.add_parser("persist-soul-full", help="Regenerate full JSONL and deploy to HF (ADR 081)")

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DOMAIN OPERATIONS (Chronicle, Task, ADR, Protocol)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    # Chronicle Command
    chron_parser = subparsers.add_parser("chronicle", help="Manage Chronicle Entries")
    chron_subs = chron_parser.add_subparsers(dest="chronicle_action")
    
    chron_list = chron_subs.add_parser("list", help="List chronicle entries")
    chron_list.add_argument("--limit", type=int, default=10, help="Number of entries to show")
    
    chron_search = chron_subs.add_parser("search", help="Search chronicle entries")
    chron_search.add_argument("query", help="Search query")
    
    chron_get = chron_subs.add_parser("get", help="Get a specific chronicle entry")
    chron_get.add_argument("number", type=int, help="Entry number")
    
    chron_create = chron_subs.add_parser("create", help="Create a new chronicle entry")
    chron_create.add_argument("title", help="Entry title")
    chron_create.add_argument("--content", required=True, help="Entry content")
    chron_create.add_argument("--author", default="AI Assistant", help="Author name")
    chron_create.add_argument("--status", default="draft", help="Entry status")
    chron_create.add_argument("--classification", default="internal", help="Classification level")
    
    chron_update = chron_subs.add_parser("update", help="Update a chronicle entry")
    chron_update.add_argument("number", type=int, help="Entry number")
    chron_update.add_argument("--title", help="New title")
    chron_update.add_argument("--content", help="New content")
    chron_update.add_argument("--status", help="New status")
    chron_update.add_argument("--reason", required=True, help="Reason for update")

    # Task Command
    task_parser = subparsers.add_parser("task", help="Manage Tasks")
    task_subs = task_parser.add_subparsers(dest="task_action")
    
    task_list = task_subs.add_parser("list", help="List tasks")
    task_list.add_argument("--status", help="Filter by status (backlog, todo, in-progress, done)")
    
    task_get = task_subs.add_parser("get", help="Get a specific task")
    task_get.add_argument("number", type=int, help="Task number")
    
    task_create = task_subs.add_parser("create", help="Create a new task")
    task_create.add_argument("title", help="Task title")
    task_create.add_argument("--objective", required=True, help="Task objective")
    task_create.add_argument("--deliverables", nargs="+", required=True, help="Deliverables")
    task_create.add_argument("--acceptance-criteria", nargs="+", required=True, help="Acceptance criteria")
    task_create.add_argument("--priority", default="MEDIUM", help="Priority level")
    task_create.add_argument("--status", default="TODO", dest="task_status", help="Initial status")
    task_create.add_argument("--lead", default="Unassigned", help="Lead assignee")
    
    task_update = task_subs.add_parser("update-status", help="Update task status")
    task_update.add_argument("number", type=int, help="Task number")
    task_update.add_argument("new_status", help="New status")
    task_update.add_argument("--notes", required=True, help="Status change notes")
    
    task_search = task_subs.add_parser("search", help="Search tasks")
    task_search.add_argument("query", help="Search query")
    
    task_edit = task_subs.add_parser("update", help="Update task fields")
    task_edit.add_argument("number", type=int, help="Task number")
    task_edit.add_argument("--title", help="New title")
    task_edit.add_argument("--objective", help="New objective")
    task_edit.add_argument("--priority", help="New priority")
    task_edit.add_argument("--lead", help="New lead")

    # ADR Command
    adr_parser = subparsers.add_parser("adr", help="Manage Architecture Decision Records")
    adr_subs = adr_parser.add_subparsers(dest="adr_action")
    
    adr_list = adr_subs.add_parser("list", help="List ADRs")
    adr_list.add_argument("--status", help="Filter by status")
    adr_list.add_argument("--limit", type=int, default=20, help="Number of ADRs to show")
    
    adr_search = adr_subs.add_parser("search", help="Search ADRs")
    adr_search.add_argument("query", help="Search query")
    
    adr_get = adr_subs.add_parser("get", help="Get a specific ADR")
    adr_get.add_argument("number", type=int, help="ADR number")
    
    adr_create = adr_subs.add_parser("create", help="Create a new ADR")
    adr_create.add_argument("title", help="ADR title")
    adr_create.add_argument("--context", required=True, help="Decision context")
    adr_create.add_argument("--decision", required=True, help="Decision made")
    adr_create.add_argument("--consequences", required=True, help="Consequences")
    adr_create.add_argument("--status", default="proposed", help="ADR status")
    
    adr_update_status = adr_subs.add_parser("update-status", help="Update ADR status")
    adr_update_status.add_argument("number", type=int, help="ADR number")
    adr_update_status.add_argument("new_status", help="New status (proposed, accepted, deprecated, superseded)")
    adr_update_status.add_argument("--reason", required=True, help="Reason for status change")

    # Protocol Command
    prot_parser = subparsers.add_parser("protocol", help="Manage Protocols")
    prot_subs = prot_parser.add_subparsers(dest="protocol_action")
    
    prot_list = prot_subs.add_parser("list", help="List protocols")
    prot_list.add_argument("--status", help="Filter by status")
    
    prot_search = prot_subs.add_parser("search", help="Search protocols")
    prot_search.add_argument("query", help="Search query")
    
    prot_get = prot_subs.add_parser("get", help="Get a specific protocol")
    prot_get.add_argument("number", type=int, help="Protocol number")
    
    prot_create = prot_subs.add_parser("create", help="Create a new protocol")
    prot_create.add_argument("title", help="Protocol title")
    prot_create.add_argument("--content", required=True, help="Protocol content")
    prot_create.add_argument("--version", default="1.0", help="Version")
    prot_create.add_argument("--status", default="PROPOSED", help="Status")
    prot_create.add_argument("--authority", default="Council", help="Authority")
    prot_create.add_argument("--classification", default="Blue", help="Classification")
    
    prot_update = prot_subs.add_parser("update", help="Update protocol fields")
    prot_update.add_argument("number", type=int, help="Protocol number")
    prot_update.add_argument("--title", help="New title")
    prot_update.add_argument("--content", help="New content")
    prot_update.add_argument("--status", help="New status")
    prot_update.add_argument("--version", help="New version")
    prot_update.add_argument("--reason", required=True, help="Reason for update")

    # Forge LLM Command (Fine-Tuned Model)
    forge_parser = subparsers.add_parser("forge", help="Interact with Sanctuary Fine-Tuned Model")
    forge_subs = forge_parser.add_subparsers(dest="forge_action")
    
    forge_query = forge_subs.add_parser("query", help="Query the Sanctuary model")
    forge_query.add_argument("prompt", help="Prompt to send to the model")
    forge_query.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    forge_query.add_argument("--max-tokens", type=int, default=2048, help="Max tokens to generate")
    forge_query.add_argument("--system", help="System prompt for context")
    
    forge_subs.add_parser("status", help="Check model availability")
    

    # Workflow Command: Agent lifecycle management (start, retrospective, end)
    wf_parser = subparsers.add_parser("workflow", help="Agent Workflow Orchestration")
    wf_subparsers = wf_parser.add_subparsers(dest="workflow_action")
    
    wf_start = wf_subparsers.add_parser("start", help="Start a new workflow (Safe Pre-flight)")
    wf_start.add_argument("--name", required=True, help="Workflow Name (e.g. workflow-start)")
    wf_start.add_argument("--target", required=True, help="Target ID")
    wf_start.add_argument("--type", default="generic", help="Target Type (optional)")

    wf_retro = wf_subparsers.add_parser("retrospective", help="Run Self-Retrospective")

    wf_end = wf_subparsers.add_parser("end", help="End workflow (Commit & Push)")
    wf_end.add_argument("message", nargs="?", help="Commit message")
    wf_end.add_argument("files", nargs="*", help="Files to commit")
    wf_end.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")

    wf_cleanup = wf_subparsers.add_parser("cleanup", help="Post-Merge Cleanup (Main Checkout & Delete Branch)")
    wf_cleanup.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")


    args = parser.parse_args()

    # ─────────────────────────────────────────────────────────
    # Command Handlers — all delegated to plugin scripts
    # ─────────────────────────────────────────────────────────

    # RAG Ingestion → vector-db plugin
    if args.command == "ingest":
        if args.incremental:
            print(f"🔄 Starting INCREMENTAL ingestion (Last {args.hours}h)...")
            import time
            from datetime import timedelta
            
            cutoff_time = time.time() - (args.hours * 3600)
            modified_files = []
            
            exclude_dirs = {'.git', '.vector_data', '__pycache__', 'node_modules', 'venv', 'env', 
                            'dataset_package', 'docs/site', 'training_logs'}
            
            for path in PROJECT_ROOT.rglob('*'):
                if path.is_file():
                    if any(part in exclude_dirs for part in path.parts):
                        continue
                    if path.suffix not in ['.md', '.py', '.js', '.ts', '.txt', '.json']:
                        continue
                    if path.stat().st_mtime > cutoff_time:
                        modified_files.append(str(path))
            
            if not modified_files:
                print(f"⚠️ No files modified in the last {args.hours} hours. Skipping ingestion.")
                sys.exit(0)
                
            print(f"📄 Found {len(modified_files)} modified files.")
            res = cortex_ops.ingest_incremental(file_paths=modified_files)
            
            # res.status contains "success" or "error"
            if res.status == "success":
                print(f"✅ Success: {res.documents_added} added, {res.chunks_created} chunks in {res.ingestion_time_ms/1000:.2f}s")
            else:
                print(f"❌ Error: {res.error}")
                sys.exit(1)
        else:
            # Full Ingestion: Purges and rebuilds the collection
            print(f"🔄 Starting full ingestion (Purge: {args.purge})...")
            res = cortex_ops.ingest_full(purge_existing=args.purge, source_directories=args.dirs)
            if res.status == "success":
                print(f"✅ Success: {res.documents_processed} docs, {res.chunks_created} chunks in {res.ingestion_time_ms/1000:.2f}s")
            else:
                print(f"❌ Error: {res.error}")
                sys.exit(1)

    # Vector Query Command: Semantic search against the RAG collection
    elif args.command == "query":
        print(f"🔍 Querying: {args.query_text}")
        res = cortex_ops.query(
            query=args.query_text,
            max_results=args.max_results,
            use_cache=args.use_cache
        )
        if res.status == "success":
            print(f"✅ Found {len(res.results)} results in {res.query_time_ms:.2f}ms")
            print(f"💾 Cache hit: {res.cache_hit}")
            for i, result in enumerate(res.results, 1):
                print(f"\n--- Result {i} (Score: {result.relevance_score:.4f}) ---")
                print(f"Content: {result.content[:300]}...")
                if result.metadata:
                    print(f"Source: {result.metadata.get('source', 'Unknown')}")
        else:
            print(f"❌ Error: {res.error}")
            sys.exit(1)

    # RAG Stats Command: View health and collection metrics
    elif args.command == "stats":
        stats = cortex_ops.get_stats(include_samples=args.samples, sample_count=args.sample_count)
        print(f"🏥 Health: {stats.health_status}")
        print(f"📚 Documents: {stats.total_documents}")
        print(f"🧩 Chunks: {stats.total_chunks}")
        if stats.collections:
            print("\n📊 Collections:")
            for name, coll in stats.collections.items():
                print(f"  - {coll.name}: {coll.count} items")
        if stats.samples:
            print(f"\n🔍 Sample Documents:")
            for i, sample in enumerate(stats.samples, 1):
                print(f"\n  {i}. ID: {sample.id}")
                print(f"     Preview: {sample.content_preview[:100]}...")

    # Cache Stats Command: View Semantic Cache efficiency metrics
    elif args.command == "cache-stats":
        stats = cortex_ops.get_cache_stats()
        print(f"💾 Cache Statistics: {stats}")

    # Cache Warmup Command: Pre-populate cache with common queries
    elif args.command == "cache-warmup":
        print(f"🔥 Warming up cache...")
        res = cortex_ops.cache_warmup(genesis_queries=args.queries)
        if res.status == "success":
            print(f"✅ Cached {res.queries_cached} queries in {res.total_time_ms/1000:.2f}s")
        else:
            print(f"❌ Error: {res.error}")
            sys.exit(1)

    # Evolution Metrics Command: Protocol 131 Fitness/Depth/Scope metrics
    elif args.command == "evolution":
        if not args.evolution_subcommand:
            print("❌ Subcommand required (fitness, depth, scope)")
            sys.exit(1)
        content = args.content
        if args.file:
            try:
                content = Path(args.file).read_text()
            except Exception as e:
                print(f"❌ Error reading file: {e}")
                sys.exit(1)
        if not content:
            print("❌ No content provided.")
            sys.exit(1)
            
        if args.evolution_subcommand == "fitness":
            print(json.dumps(evolution_ops.calculate_fitness(content), indent=2))
        elif args.evolution_subcommand == "depth":
            print(f"Depth: {evolution_ops.measure_depth(content)}")
        elif args.evolution_subcommand == "scope":
            print(f"Scope: {evolution_ops.measure_scope(content)}")
            
    # RLM Distillation: Atomic summarization of files (Protocol 132 Level 1)
    elif args.command in ["rlm-distill", "rlm-test"]:
        profile = getattr(args, 'profile', 'project')
        print(f"🧠 RLM: Distilling '{args.target}' [profile={profile}]...")
        distiller_script = str(RLM_DIR / "distiller.py")
        cmd = [sys.executable, distiller_script, "--profile", profile, "--file", args.target]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

    # Init-Context Command: Initialize manifest from base template and auto-bundle
    elif args.command == "init-context":
        artifact_type = args.type if args.type else "generic"
        
        print(f"🚀 Initializing Smart Context Bundle for {args.target} ({artifact_type})...")
        script = str(RETRIEVE_DIR / "manifest_manager.py")
        
        subprocess.run([sys.executable, script, "init", "--bundle-title", args.target, "--type", artifact_type])
        subprocess.run([sys.executable, script, "bundle"])

    # Manifest Command: Manage Context Bundler manifests (.json configs)
    elif args.command == "manifest":
        script = str(RETRIEVE_DIR / "manifest_manager.py")
        
        # Helper to build base command with globals
        base_cmd = [sys.executable, script]
        if hasattr(args, 'base') and args.base:
            base_cmd.extend(["--base", args.base])
        if hasattr(args, 'manifest') and args.manifest:
            base_cmd.extend(["--manifest", args.manifest])

        if args.manifest_action == "init":
            artifact_type = args.type
            if not artifact_type:
                print(f"❌ Error: --type is required for manifest init. Options: generic, learning, guardian, etc.")
                sys.exit(1)
            
            cmd = base_cmd + ["init", "--bundle-title", args.bundle_title, "--type", artifact_type]
            subprocess.run(cmd)
            
        elif args.manifest_action == "add":
            cmd = base_cmd + ["add", "--path", args.path, "--note", args.note]
            subprocess.run(cmd)
            
        elif args.manifest_action == "remove":
            cmd = base_cmd + ["remove", "--path", args.path]
            subprocess.run(cmd)
            
        elif args.manifest_action == "update":
            cmd = base_cmd + ["update", "--path", args.path]
            if args.note:
                cmd.extend(["--note", args.note])
            if args.new_path:
                cmd.extend(["--new-path", args.new_path])
            subprocess.run(cmd)
            
        elif args.manifest_action == "search":
            cmd = base_cmd + ["search", args.pattern]
            subprocess.run(cmd)
            
        elif args.manifest_action == "list":
            cmd = base_cmd + ["list"]
            subprocess.run(cmd)
            
        elif args.manifest_action == "bundle":
            cmd = base_cmd + ["bundle"]
            if args.output:
                cmd.extend(["--output", args.output])
            subprocess.run(cmd)

    # Protocol 128 Snapshot: Create a memory bundle for session continuity
    elif args.command == "snapshot":
        # ADR 090: Iron Core Verification
        if not args.override_iron_core:
            print("🛡️  Running Iron Core Verification (ADR 090)...")
            is_pristine, violations = verify_iron_core(PROJECT_ROOT)
            if not is_pristine:
                print(f"\n\033[91m⛔ IRON CORE BREACH DETECTED (SAFE MODE ENGAGED)\033[0m")
                print("The following immutable files have been modified without authorization:")
                for v in violations:
                    print(f"  - {v}")
                print("\nAction blocked: 'snapshot' is disabled in Safe Mode.")
                print("To proceed, revert changes or use --override-iron-core (Constitutional Amendment required).")
                sys.exit(1)
            print("✅ Iron Core Integrity Verified.")
        else:
            print(f"⚠️  \033[93mWARNING: IRON CORE CHECK OVERRIDDEN\033[0m")

        # Protocol 128 Snapshot Generation (Delegated to LearningOperations)
        print(f"📸 Generating {args.type} snapshot via Learning Operations...")
        
        ops = _get_learning_ops()
        
        # Manifest Handling
        manifest_list = []
        if args.manifest:
            p = Path(args.manifest)
            if p.exists():
                try:
                    data = json.loads(p.read_text())
                    if isinstance(data, list): 
                        manifest_list = data
                    elif isinstance(data, dict):
                        # ADR 097 support
                        if "files" in data: 
                            manifest_list = [f["path"] if isinstance(f, dict) else f for f in data["files"]]
                        else:
                            # Try legacy keys or fallback
                            manifest_list = data.get("core", []) + data.get("topic", [])
                except Exception as e:
                    print(f"⚠️ Could not parse custom manifest {args.manifest}: {e}")

        # Execute
        result = ops.capture_snapshot(
            manifest_files=manifest_list,
            snapshot_type=args.type,
            strategic_context=args.context
        )
        
        if result.status == "success":
            print(f"✅ Snapshot created: {result.snapshot_path}")
            print(f"   Files: {result.total_files}, Bytes: {result.total_bytes}")
            if not result.manifest_verified:
                print(f"   ⚠️ Manifest Verification Failed: {result.git_diff_context}")
        else:
            print(f"❌ Error: {result.error}")
            if result.git_diff_context:
                print(f"   Context: {result.git_diff_context}")
            sys.exit(1)

    # Protocol 128 Debrief: Orientation for fresh sessions (Truth Anchor)
    elif args.command == "debrief":
        print(f"📡 Running Learning Debrief (Protocol 128 Phase I)...")
        ops = _get_learning_ops()
        
        # Debrief returns a formatted Markdown string
        debrief_content = ops.learning_debrief(hours=args.hours)
        
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(debrief_content)
            print(f"✅ Debrief written to: {output_path}")
            print(f"📊 Content length: {len(debrief_content)} characters")
        else:
            # Output to stdout
            print(debrief_content)

    # Guardian Command: Session pack and Boot Digest (Lifecycle)
    elif args.command == "guardian":
        # Initialize ops locally to ensure availability
        ops = _get_learning_ops()
        
        if args.guardian_action == "wakeup":
            # Load manifest if exists (using proper arg now)
            manifest_path_str = args.manifest if args.manifest else ".agent/learning/guardian_manifest.json"
            manifest_path = Path(manifest_path_str)
            
            if manifest_path.exists():
                try:
                    with open(manifest_path, 'r') as f:
                        manifest = json.load(f)
                    print(f"📋 Loaded guardian manifest: {len(manifest)} files")
                except Exception as e:
                    print(f"⚠️  Error reading guardian manifest: {e}")
            else:
                print(f"⚠️  Guardian manifest not found at {manifest_path_str}. Using defaults.")

            # ROUTED TO LEARNING MCP
            response = ops.guardian_wakeup(mode=args.mode)
            
            if response.status == "success":
                print(f"✅ Boot Digest Generated: {response.digest_path}")
                print(f"   Time: {response.total_time_ms:.2f}ms")
            else:
                print(f"❌ Error: {response.error}")
                sys.exit(1)

        elif args.guardian_action == "snapshot":
            print(f"🛡️  Guardian Snapshot: Capturing Session Pack...")
            response = ops.guardian_snapshot(strategic_context=args.context)
            
            if response.status == "success":
                print(f"✅ Session Pack Captured: {response.snapshot_path}")
                print(f"   Files: {response.total_files}, Bytes: {response.total_bytes}")
            else:
                print(f"❌ Error: {response.error}")
                sys.exit(1)

    # Persist Soul Command: Protocol 128 Phase VI (Hugging Face Broadcast)
    elif args.command == "persist-soul":
        print(f"📡 Initiating Soul Persistence (Protocol 128 Phase VI)...")
        print(f"   Valence: {args.valence} | Uncertainty: {args.uncertainty} | Full Sync: {args.full_sync}")
        ops = _get_learning_ops()
        
        # Default snapshot for seal is usually 'learning/learning_package_snapshot.md'
        snapshot_path = args.snapshot
        if not snapshot_path:
            snapshot_path = ".agent/learning/learning_package_snapshot.md"
            
        PersistSoulRequest, _, _ = _get_learning_models()
        req = PersistSoulRequest(
            snapshot_path=snapshot_path,
            valence=args.valence,
            uncertainty=args.uncertainty,
            is_full_sync=args.full_sync
        )
        
        result = ops.persist_soul(req)
        
        if result.status == "success":
            print(f"✅ Persistence Complete!")
            print(f"   Repo: {result.repo_url}")
            print(f"   Artifact: {result.snapshot_name}")
        elif result.status == "quarantined":
            print(f"🚫 Quarantined: {result.error}")
        else:
            print(f"❌ Persistence Failed: {result.error}")
            sys.exit(1)

    # Persist Soul Full: ADR 081 Full Dataset Regeneration
    elif args.command == "persist-soul-full":
        print(f"🧬 Regenerating full Soul JSONL and deploying to HuggingFace (ADR 081)...")
        ops = _get_learning_ops()
        
        result = ops.persist_soul_full()
        
        if result.status == "success":
            print(f"✅ Full Sync Complete!")
            print(f"   Repo: {result.repo_url}")
            print(f"   Output: {result.snapshot_name}")
        else:
            print(f"❌ Error: {result.error}")
            sys.exit(1)

    # Bootstrap Debrief Command: Fresh Repo Onboarding
    elif args.command == "bootstrap-debrief":
        print(f"🏗️  Generating Bootstrap Context Packet...")
        ops = _get_learning_ops()
        
        # Load manifest
        manifest_path = Path(args.manifest)
        manifest_list = []
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text())
                if isinstance(data, list): 
                    manifest_list = data
                elif isinstance(data, dict): 
                    # Extract 'path' from dict entries if present, or use raw strings
                    raw_files = data.get("files", [])
                    manifest_list = [f.get("path") if isinstance(f, dict) else f for f in raw_files]
                print(f"📋 Loaded bootstrap manifest: {len(manifest_list)} items")
            except Exception as e:
                print(f"⚠️  Error reading manifest: {e}")
        else:
            print(f"⚠️  Bootstrap manifest not found at {args.manifest}. Using defaults/empty.")

        # Generate snapshot
        res = ops.capture_snapshot(
            manifest_files=manifest_list,
            snapshot_type="seal",
            strategic_context="Fresh repository onboarding context"
        )
        
        if res.status == "success":
            # Copy to output path
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copy(res.snapshot_path, output_path)
            
            print(f"✅ Bootstrap packet generated: {output_path}")
            print(f"📊 Files: {res.total_files} | Bytes: {res.total_bytes}")
        else:
            print(f"❌ Error: {res.error}")
            sys.exit(1)

    # Tools Command: Manage tool inventory (list, search, add, update, remove)
    elif args.command == "tools":
        script = str(INVENTORIES_DIR / "manage_tool_inventory.py")
        cmd = [sys.executable, script]
        
        if args.tools_action == "list":
            cmd.append("list")
            # manage_tool_inventory doesn't explicitly have --category for list in its main args? 
            # Looking at manage_tool_inventory.py, it doesn't seem to expose category filtering for list via CLI args easily, 
            # but let's check its argparse. It has --category but that's for 'add'.
            # 'list_tools' prints all categories.
            # We'll just run 'list'.
            
        elif args.tools_action == "search":
            # manage_tool_inventory expects: search <keyword> as subcommand?
            # No, looking at it: parser has "keyword" as a positional arg for SEARCH? 
            # Wait, manage_tool_inventory main() usually uses subparsers or just flags.
            # Let's re-read manage_tool_inventory.py usage.
            # It seems it uses flags: --path, --desc, keyword (positional), --status
            # Actually, manage_tool_inventory seems to handle flags directly.
            # Let's map args blindly or intelligently.
            cmd.append("search")
            cmd.append(args.keyword)
            
        elif args.tools_action == "add":
            cmd.append("add")
            cmd.extend(["--path", args.path])
            if args.category:
                cmd.extend(["--category", args.category])
                
        elif args.tools_action == "update":
            cmd.append("update")
            cmd.extend(["--path", args.path])
            if args.desc:
                cmd.extend(["--desc", args.desc])
                
        elif args.tools_action == "remove":
            cmd.append("remove")
            cmd.extend(["--path", args.path])
            
        # Execute
        subprocess.run(cmd)

    # Workflow Command: Agent lifecycle management (Start/End/Retro)
    elif args.command == "workflow":
        orchestrator_script = str(PROJECT_ROOT / "plugins/agent-loops/skills/orchestrator/scripts/agent_orchestrator.py")
        if not os.path.exists(orchestrator_script):
            print(f"❌ Error: agent_orchestrator.py not found at {orchestrator_script}")
            sys.exit(1)

        if args.workflow_action == "start":
            print(f"🚀 Routing 'workflow start' to Orchestrator ({args.name})...")
            try:
                cmd = [sys.executable, orchestrator_script, "scan"]
                if getattr(args, 'target', None) and os.path.isdir(args.target):
                    cmd.extend(["--spec-dir", args.target])
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Orchestrator Scan Failed: {e}")
                sys.exit(1)
        
        elif args.workflow_action == "retrospective":
            try:
                subprocess.run([sys.executable, orchestrator_script, "retro"], check=True)
            except subprocess.CalledProcessError as e:
                print(f"❌ Retrospective Failed: {e}")
                sys.exit(1)

        elif args.workflow_action == "end":
            print("🚀 'workflow end' is deprecated. Use agent-loops retro and manual git flow.")
            sys.exit(0)

        elif args.workflow_action == "cleanup":
            print("🚀 'workflow cleanup' is deprecated.")
            sys.exit(0)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # DOMAIN COMMAND HANDLERS
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    # Chronicle Command Handler
    elif args.command == "chronicle":
        chron_ops = ChronicleOperations(os.path.join(PROJECT_ROOT, "00_CHRONICLE/ENTRIES"))
        
        if args.chronicle_action == "list":
            res = chron_ops.list_entries(limit=args.limit)
            for e in res:
                print(f"[{e['number']:03d}] {e['title']} ({e['date']})")
        elif args.chronicle_action == "search":
            res = chron_ops.search_entries(args.query)
            for e in res:
                print(f"[{e['number']:03d}] {e['title']}")
        elif args.chronicle_action == "get":
            res = chron_ops.get_entry(args.number)
            print(f"[{res['number']:03d}] {res['title']}")
            print("-" * 40)
            print(res['content'])
        elif args.chronicle_action == "create":
            res = chron_ops.create_entry(
                title=args.title,
                content=str(args.content).replace("\\n", "\n"),
                author=args.author,
                status=args.status,
                classification=args.classification
            )
            print(f"✅ Created Chronicle Entry #{res['entry_number']:03d}: {res['file_path']}")
        elif args.chronicle_action == "update":
            updates = {}
            if args.title:
                updates['title'] = args.title
            if args.content:
                updates['content'] = str(args.content).replace("\\n", "\n")
            if args.status:
                updates['status'] = args.status
            res = chron_ops.update_entry(args.number, updates, args.reason)
            print(f"✅ Updated Chronicle Entry #{args.number:03d}")
        else:
            print("❌ Chronicle subcommand required (list, search, get, create, update)")
            sys.exit(1)

    # Task Command Handler
    elif args.command == "task":
        task_ops = TaskOperations(PROJECT_ROOT)
        
        if args.task_action == "list":
            status_obj = taskstatus(args.status) if args.status else None
            res = task_ops.list_tasks(status=status_obj)
            for t in res:
                print(f"[{t['number']:03d}] {t['title']} ({t['status']})")
        elif args.task_action == "get":
            res = task_ops.get_task(args.number)
            if not res:
                print(f"❌ Task {args.number} not found")
                sys.exit(1)
            print(f"[{res['number']:03d}] {res['title']}")
            print(f"Status: {res['status']} | Priority: {res['priority']} | Lead: {res['lead']}")
            print("-" * 40)
            print(res['content'])
        elif args.task_action == "create":
            res = task_ops.create_task(
                title=args.title,
                objective=str(args.objective).replace("\\n", "\n"),
                deliverables=args.deliverables,
                acceptance_criteria=args.acceptance_criteria,
                priority=TaskPriority(args.priority.capitalize()),
                status=taskstatus(args.task_status.lower()),
                lead=args.lead
            )
            if res.status == "success":
                print(f"✅ Created Task #{res.task_number:03d} at {res.file_path}")
            else:
                print(f"❌ Creation failed: {res.message}")
                sys.exit(1)
        elif args.task_action == "update-status":
            task_ops.update_task_status(args.number, taskstatus(args.new_status), args.notes)
            print(f"✅ Task {args.number} moved to {args.new_status}")
        elif args.task_action == "search":
            res = task_ops.search_tasks(args.query)
            for t in res:
                print(f"[{t['number']:03d}] {t['title']} ({t['status']})")
        elif args.task_action == "update":
            updates = {}
            if args.title:
                updates['title'] = args.title
            if args.objective:
                updates['objective'] = args.objective
            if args.priority:
                updates['priority'] = args.priority
            if args.lead:
                updates['lead'] = args.lead
            res = task_ops.update_task(args.number, updates)
            print(f"✅ Updated Task #{args.number:03d}")
        else:
            print("❌ Task subcommand required (list, get, create, update-status, search, update)")
            sys.exit(1)

    # ADR Command Handler
    elif args.command == "adr":
        adr_ops = ADROperations(os.path.join(PROJECT_ROOT, "ADRs"))
        
        if args.adr_action == "list":
            res = adr_ops.list_adrs(status=args.status.upper() if args.status else None)
            for a in res:
                print(f"[{a['number']:03d}] {a['title']} [{a['status']}]")
        elif args.adr_action == "search":
            res = adr_ops.search_adrs(args.query)
            for a in res:
                print(f"[{a['number']:03d}] {a['title']}")
        elif args.adr_action == "get":
            res = adr_ops.get_adr(args.number)
            print(f"ADR-{res['number']:03d}: {res['title']}")
            print(f"Status: {res['status']}")
            print("-" * 40)
            print(f"# Context\n{res['context']}\n")
            print(f"# Decision\n{res['decision']}\n")
            print(f"# Consequences\n{res['consequences']}")
        elif args.adr_action == "create":
            res = adr_ops.create_adr(
                title=args.title,
                context=str(args.context).replace("\\n", "\n"),
                decision=str(args.decision).replace("\\n", "\n"),
                consequences=str(args.consequences).replace("\\n", "\n"),
                status=args.status
            )
            print(f"✅ Created ADR-{res['adr_number']:03d} at {res['file_path']}")
        elif args.adr_action == "update-status":
            res = adr_ops.update_adr_status(args.number, args.new_status.upper(), args.reason)
            print(f"✅ ADR-{args.number:03d} status updated to {args.new_status.upper()}")
        else:
            print("❌ ADR subcommand required (list, search, get, create, update-status)")
            sys.exit(1)

    # Protocol Command Handler
    elif args.command == "protocol":
        prot_ops = ProtocolOperations(os.path.join(PROJECT_ROOT, "01_PROTOCOLS"))
        
        if args.protocol_action == "list":
            res = prot_ops.list_protocols(status=args.status.upper() if args.status else None)
            for p in res:
                print(f"[{p['number']:03d}] {p['title']} [{p['status']}]")
        elif args.protocol_action == "search":
            res = prot_ops.search_protocols(args.query)
            for p in res:
                print(f"[{p['number']:03d}] {p['title']}")
        elif args.protocol_action == "get":
            res = prot_ops.get_protocol(args.number)
            print(f"Protocol-{res['number']:03d}: {res['title']}")
            print(f"v{res['version']} | {res['status']} | {res['classification']}")
            print("-" * 40)
            print(res['content'])
        elif args.protocol_action == "create":
            res = prot_ops.create_protocol(
                number=None,  # Auto-generate
                title=args.title,
                status=args.status,
                classification=args.classification,
                version=args.version,
                authority=args.authority,
                content=str(args.content).replace("\\n", "\n")
            )
            print(f"✅ Created Protocol-{res['protocol_number']:03d} at {res['file_path']}")
        elif args.protocol_action == "update":
            updates = {}
            if args.title:
                updates['title'] = args.title
            if args.content:
                updates['content'] = str(args.content).replace("\\n", "\n")
            if args.status:
                updates['status'] = args.status
            if args.version:
                updates['version'] = args.version
            res = prot_ops.update_protocol(args.number, updates, args.reason)
            print(f"✅ Updated Protocol-{args.number:03d}")
        else:
            print("❌ Protocol subcommand required (list, search, get, create, update)")
            sys.exit(1)

    # Forge LLM Command Handler
    elif args.command == "forge":
        if not FORGE_AVAILABLE:
            print("❌ Forge LLM not available. Install ollama: pip install ollama")
            sys.exit(1)
        forge_ops = ForgeOperations(str(PROJECT_ROOT))
        
        if args.forge_action == "query":
            print(f"🤖 Querying Sanctuary Model...")
            res = forge_ops.query_sanctuary_model(
                prompt=args.prompt,
                temperature=args.temperature,
                max_tokens=getattr(args, 'max_tokens', 2048),
                system_prompt=getattr(args, 'system', None)
            )
            if res.status == "success":
                print(f"\n{res.response}")
                print(f"\n📊 Tokens: {res.total_tokens or 'N/A'} | Temp: {res.temperature}")
            else:
                print(f"❌ Error: {res.error}")
                sys.exit(1)
        elif args.forge_action == "status":
            print("🔍 Checking Sanctuary Model availability...")
            res = forge_ops.check_model_availability()
            if res.get("status") == "success":
                print(f"✅ Model: {res['model']}")
                print(f"   Available: {res['available']}")
                if res.get('all_models'):
                    print(f"   All Models: {', '.join(res['all_models'][:5])}{'...' if len(res['all_models']) > 5 else ''}")
            else:
                print(f"❌ Error: {res.get('error', 'Unknown error')}")
                sys.exit(1)
        else:
            print("❌ Forge subcommand required (query, status)")
            sys.exit(1)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
