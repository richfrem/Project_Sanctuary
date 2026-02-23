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
        script = str(VECTOR_DB_SCRIPTS / "ingest.py")
        cmd = [sys.executable, script]
        if args.incremental:
            cmd.extend(["--since", str(args.hours)])
        else:
            cmd.append("--full")
        if hasattr(args, 'profile') and args.profile:
            cmd.extend(["--profile", args.profile])
        print(f"🔄 Delegating ingestion to vector-db plugin...")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

    # Vector Query → vector-db plugin
    elif args.command == "query":
        script = str(VECTOR_DB_SCRIPTS / "query.py")
        cmd = [sys.executable, script, args.query_text, "--limit", str(args.max_results)]
        if hasattr(args, 'profile') and args.profile:
            cmd.extend(["--profile", args.profile])
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

    # RAG Stats → vector-db plugin operations
    elif args.command == "stats":
        print("ℹ️  Use the vector-db plugin directly for stats:")
        print(f"   python3 {VECTOR_DB_SCRIPTS / 'query.py'} --help")

    # Cache Stats — DEPRECATED (semantic cache not in vector-db plugin)
    elif args.command == "cache-stats":
        print("⚠️  'cache-stats' is deprecated. The semantic cache layer was removed.")
        print("   Use the vector-db plugin for RAG operations instead.")

    # Cache Warmup — DEPRECATED
    elif args.command == "cache-warmup":
        print("⚠️  'cache-warmup' is deprecated. The semantic cache layer was removed.")
        print("   Use the vector-db plugin for RAG operations instead.")

    # Evolution Metrics → guardian-onboarding plugin
    elif args.command == "evolution":
        script = str(GUARDIAN_SCRIPTS / "evolution_metrics.py")
        if not args.evolution_subcommand:
            print("❌ Subcommand required (fitness, depth, scope)")
            sys.exit(1)
        cmd = [sys.executable, script, args.evolution_subcommand]
        if args.file:
            cmd.extend(["--file", args.file])
        elif args.content:
            cmd.append(args.content)
        else:
            print("❌ No content provided. Use --file or pass text.")
            sys.exit(1)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)
            
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

        # Protocol 128 Snapshot Generation → guardian-onboarding plugin
        print(f"📸 Generating {args.type} snapshot...")
        script = str(GUARDIAN_SCRIPTS / "capture_snapshot.py")
        cmd = [sys.executable, script, "--type", args.type]
        if args.context:
            cmd.extend(["--context", args.context])
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr if result.stderr else "❌ Snapshot failed")
            sys.exit(1)
        # capture_snapshot.py outputs JSON
        print(result.stdout)

    # Protocol 128 Debrief → guardian-onboarding plugin
    elif args.command == "debrief":
        script = str(GUARDIAN_SCRIPTS / "learning_debrief.py")
        cmd = [sys.executable, script, "--hours", str(args.hours)]
        if args.output:
            cmd.extend(["--output", args.output])
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

    # Guardian Command → guardian-onboarding plugin
    elif args.command == "guardian":
        if args.guardian_action == "wakeup":
            script = str(GUARDIAN_SCRIPTS / "guardian_wakeup.py")
            cmd = [sys.executable, script, "--mode", args.mode]
            result = subprocess.run(cmd)
            if result.returncode != 0:
                sys.exit(result.returncode)

        elif args.guardian_action == "snapshot":
            script = str(GUARDIAN_SCRIPTS / "capture_snapshot.py")
            cmd = [sys.executable, script, "--type", "audit"]
            if hasattr(args, 'context') and args.context:
                cmd.extend(["--context", args.context])
            result = subprocess.run(cmd)
            if result.returncode != 0:
                sys.exit(result.returncode)

    # Persist Soul → guardian-onboarding plugin
    elif args.command == "persist-soul":
        script = str(GUARDIAN_SCRIPTS / "persist_soul.py")
        cmd = [sys.executable, script]
        snapshot = getattr(args, 'snapshot', None) or ".agent/learning/learning_package_snapshot.md"
        cmd.extend(["--snapshot", snapshot])
        if hasattr(args, 'valence'):
            cmd.extend(["--valence", str(args.valence)])
        if getattr(args, 'full_sync', False):
            cmd.append("--full-sync")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

    # Persist Soul Full → persist_soul.py --full-sync
    elif args.command == "persist-soul-full":
        script = str(GUARDIAN_SCRIPTS / "persist_soul.py")
        cmd = [sys.executable, script, "--full-sync"]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

    # Bootstrap Debrief → capture_snapshot.py (seal type)
    elif args.command == "bootstrap-debrief":
        script = str(GUARDIAN_SCRIPTS / "capture_snapshot.py")
        cmd = [sys.executable, script, "--type", "seal", "--context", "Fresh repository onboarding context"]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

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
    
    # Chronicle → chronicle-manager plugin
    elif args.command == "chronicle":
        script = str(CHRONICLE_SCRIPTS / "chronicle_manager.py")
        cmd = [sys.executable, script]

        if args.chronicle_action == "list":
            cmd.append("list")
            if hasattr(args, 'limit') and args.limit:
                cmd.extend(["--limit", str(args.limit)])
        elif args.chronicle_action == "search":
            cmd.extend(["search", args.query])
        elif args.chronicle_action == "get":
            cmd.extend(["get", str(args.number)])
        elif args.chronicle_action == "create":
            cmd.extend(["create", args.title])
            if args.content:
                cmd.extend(["--content", str(args.content).replace("\\n", "\n")])
            if hasattr(args, 'author') and args.author:
                cmd.extend(["--author", args.author])
        else:
            print("❌ Chronicle subcommand required (list, search, get, create)")
            sys.exit(1)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

    # Task → task-manager plugin
    elif args.command == "task":
        script = str(TASK_SCRIPTS / "task_manager.py")
        cmd = [sys.executable, script]

        if args.task_action == "list":
            cmd.append("list")
            if args.status:
                cmd.extend(["--lane", args.status])
        elif args.task_action == "get":
            cmd.extend(["get", str(args.number)])
        elif args.task_action == "create":
            cmd.extend(["create", args.title])
            if hasattr(args, 'objective') and args.objective:
                cmd.extend(["--objective", str(args.objective)])
            if hasattr(args, 'task_status') and args.task_status:
                cmd.extend(["--lane", args.task_status])
        elif args.task_action == "update-status":
            cmd.extend(["move", str(args.number), args.new_status])
            if hasattr(args, 'notes') and args.notes:
                cmd.extend(["--note", args.notes])
        elif args.task_action == "search":
            cmd.extend(["search", args.query])
        else:
            print("❌ Task subcommand required (list, get, create, update-status, search)")
            sys.exit(1)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

    # ADR → adr-manager plugin
    elif args.command == "adr":
        script = str(ADR_SCRIPTS / "adr_manager.py")
        cmd = [sys.executable, script]

        if args.adr_action == "list":
            cmd.append("list")
            if hasattr(args, 'limit') and args.limit:
                cmd.extend(["--limit", str(args.limit)])
        elif args.adr_action == "search":
            cmd.extend(["search", args.query])
        elif args.adr_action == "get":
            cmd.extend(["get", str(args.number)])
        elif args.adr_action == "create":
            cmd.extend(["create", args.title])
            if args.context:
                cmd.extend(["--context", str(args.context).replace("\\n", "\n")])
            if args.decision:
                cmd.extend(["--decision", str(args.decision).replace("\\n", "\n")])
            if args.consequences:
                cmd.extend(["--consequences", str(args.consequences).replace("\\n", "\n")])
        elif args.adr_action == "update-status":
            print("⚠️  ADR update-status: use adr_manager.py directly (not yet wired).")
        else:
            print("❌ ADR subcommand required (list, search, get, create)")
            sys.exit(1)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

    # Protocol → protocol-manager plugin
    elif args.command == "protocol":
        script = str(PROTOCOL_SCRIPTS / "protocol_manager.py")
        cmd = [sys.executable, script]

        if args.protocol_action == "list":
            cmd.append("list")
            if hasattr(args, 'limit') and args.limit:
                cmd.extend(["--limit", str(args.limit)])
            if args.status:
                cmd.extend(["--status", args.status])
        elif args.protocol_action == "search":
            cmd.extend(["search", args.query])
        elif args.protocol_action == "get":
            cmd.extend(["get", str(args.number)])
        elif args.protocol_action == "create":
            cmd.extend(["create", args.title])
            if args.content:
                cmd.extend(["--content", str(args.content).replace("\\n", "\n")])
            if args.status:
                cmd.extend(["--status", args.status])
        elif args.protocol_action == "update":
            cmd.extend(["update", str(args.number)])
            if args.status:
                cmd.extend(["--status", args.status])
            if hasattr(args, 'reason') and args.reason:
                cmd.extend(["--reason", args.reason])
        else:
            print("❌ Protocol subcommand required (list, search, get, create, update)")
            sys.exit(1)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

    # Forge → guardian-onboarding plugin
    elif args.command == "forge":
        script = str(GUARDIAN_SCRIPTS / "forge_llm.py")
        cmd = [sys.executable, script]

        if args.forge_action == "query":
            cmd.extend(["query", args.prompt])
            if hasattr(args, 'temperature'):
                cmd.extend(["--temperature", str(args.temperature)])
        elif args.forge_action == "status":
            cmd.append("status")
        else:
            print("❌ Forge subcommand required (query, status)")
            sys.exit(1)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            sys.exit(result.returncode)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
