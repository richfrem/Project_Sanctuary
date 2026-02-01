#!/usr/bin/env python3
"""
cli.py (CLI)
=====================================

Purpose:
    Main entry point for the Antigravity Command System. Orchestrates sub-tools 
    for context bundling, tool discovery, workflow management, and queries.

Layer: Tools / CLI

Usage Examples:
    python tools/cli.py --help
    python tools/cli.py tools list
    python tools/cli.py query "search term"
    python tools/cli.py workflow start --name codify --target MyTarget

Supported Object Types:
    - Generic
    - Workflow

CLI Arguments:
    --target        : Target ID
    --type          : Artifact Type
    --json          : Output in JSON format
    --manifest      : Custom manifest path
    --output        : Output directory

Key Functions:
    - resolve_type_from_inventory(): Placeholder for artifact type resolution.
    - main(): CLI entry point.

Script Dependencies:
    - tools/investigate/utils/path_resolver.py
    - tools/orchestrator/workflow_manager.py
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

# Import path resolver
try:
    from tools.utils.path_resolver import resolve_path
except ImportError:
    # Fallback if running from root without package structure
    sys.path.append(str(PROJECT_ROOT))
    from tools.utils.path_resolver import resolve_path

# Resolve Directories
MINERS_DIR = Path(resolve_path("tools/investigate/miners"))
SEARCH_DIR = Path(resolve_path("tools/investigate/search"))
MENU_DIR = Path(resolve_path("tools/investigate/menu"))
DOCS_DIR = Path(resolve_path("tools/codify/documentation"))
TRACKING_DIR = Path(resolve_path("tools/codify/tracking"))
SHARED_DIR = Path(resolve_path("tools/shared"))
RETRIEVE_DIR = Path(resolve_path("tools/retrieve/bundler"))
VECTOR_TOOLS_DIR = Path(resolve_path("tools/retrieve/vector"))
INVENTORIES_DIR = Path(resolve_path("tools/curate/inventories"))
HYGIENE_DIR = Path(resolve_path("tools/curate/hygiene"))
RLM_DIR = Path(resolve_path("tools/retrieve/rlm"))
ORCHESTRATOR_DIR = Path(resolve_path("tools/orchestrator"))

# Add directories to sys.path for internal imports
for d in [MINERS_DIR, SEARCH_DIR, MENU_DIR, DOCS_DIR, TRACKING_DIR, SHARED_DIR, RETRIEVE_DIR, INVENTORIES_DIR, RLM_DIR, ORCHESTRATOR_DIR]:
    if str(d) not in sys.path:
        sys.path.append(str(d))

from tools.utils.path_resolver import resolve_path
from workflow_manager import WorkflowManager
try:
    from mcp_servers.learning.operations import LearningOperations, PersistSoulRequest
except ImportError:
    # Fallback/Bootstrap if pathing is tricky
    sys.path.append(str(PROJECT_ROOT))
    from mcp_servers.learning.operations import LearningOperations, PersistSoulRequest

# ADR 090: Iron Core Definitions
IRON_CORE_PATHS = [
    "01_PROTOCOLS",
    "ADRs",
    "cognitive_continuity_policy.md",
    "founder_seed.json"
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

def resolve_type_from_inventory(target_id: str) -> str:
    """
    Placeholder for artifact type resolution.
    Project Sanctuary does not use a legacy object inventory.
    Returns None - callers should handle type inference differently.
    """
    return None

def _enrich_manifest(target_id: str, manifest_manager_path: str):
    """
    Enriches the manifest by automatically adding downstream dependencies
    discovered by dependencies.py.
    """
    print(f"✨ Auto-enriching manifest with dependencies for {target_id}...")
    
    # 1. Query Dependencies (Downstream + JSON)
    dep_script = str(SEARCH_DIR / "dependencies.py")
    cmd = [sys.executable, dep_script, "--target", target_id, "--direction", "downstream", "--json"]
    
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(res.stdout)
        
        downstream = data.get("downstream", {})
        count = 0
        
        # 2. Iterate and Add
        for category, items in downstream.items():
            # FILTER: Skip Forms and Reports (Focus on Logic/DB)
            if category in ["Forms", "Reports", "Menus"]:
                continue
                
            for item in items:
                # Item structure from dependencies.py: {"id": "NAME", "FilePath": "path/to/file"}
                file_path = item.get("FilePath")
                if file_path:
                    # Construct note
                    note = f"Auto-Dep: {category} ({item['id']})"
                    
                    # Call manifest add
                    # We use subprocess to keep manifest logic encapsulated in manager
                    # Note: manifest_manager expects relative or absolute path. 
                    # dependencies.py returns path relative to PROJECT_ROOT usually.
                    
                    add_cmd = [sys.executable, manifest_manager_path, "add", "--path", file_path, "--note", note]
                    subprocess.run(add_cmd, capture_output=True) # Silence output to avoid noise
                    count += 1
        
        print(f"✅ Added {count} downstream dependencies to manifest.")
        
    except subprocess.CalledProcessError as e:
        print(f"⚠️  Dependency enrichment failed (Command Error): {e}")
    except json.JSONDecodeError:
        print(f"⚠️  Dependency enrichment failed (Invalid JSON output from dependencies.py)")
    except Exception as e:
        print(f"⚠️  Dependency enrichment failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Recursive Business Rule Discovery CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Scan Command
    scan_parser = subparsers.add_parser("scan", help="Recursively scan an artifact")
    scan_parser.add_argument("--target", required=True, help="Target ID")
    scan_parser.add_argument("--depth", type=int, default=1, help="Recursion depth")

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

    # Candidates Command (Wrapper)
    cand_parser = subparsers.add_parser("candidates", help="Manage priority candidates")
    cand_parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for priority_candidates.py")

    # Business Rules Command (Wrapper to business_rules_inventory_manager.py)
    br_parser = subparsers.add_parser("br", help="Business Rules Inventory (search, register, investigate)")
    br_parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for business_rules_inventory_manager.py")

    # Business Workflows Command (Wrapper to business_workflows_inventory_manager.py)
    bw_parser = subparsers.add_parser("bw", help="Business Workflows Inventory (search, register, investigate)")
    bw_parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for business_workflows_inventory_manager.py")


    # Dependency Command
    dep_parser = subparsers.add_parser("dependencies", help="Query Artifact Dependencies")
    dep_parser.add_argument("--target", required=True, help="Target Artifact ID")
    dep_parser.add_argument("--deep", action="store_true", help="Deep search across all source files")
    dep_parser.add_argument("--json", action="store_true", help="Output in JSON format")
    dep_parser.add_argument("--direction", choices=['upstream', 'downstream', 'both'], default='both', help="Analysis direction (upstream/downstream/both)")

    # Bundle Command
    bundle_parser = subparsers.add_parser("bundle", help="Gather comprehensive context (XML + Deps + Menu)")
    bundle_parser.add_argument("--target", required=True, help="Target Artifact ID")

    # Context Command (Initial Reconstruction)
    context_parser = subparsers.add_parser("init-context", help="Initialize manifest and generate first bundle")
    context_parser.add_argument("--target", required=True, help="Target ID")
    context_parser.add_argument("--type", choices=[
        'generic', 'context-bundler', 'tool', 'workflow', 'docs', 'adr', 'spec',
        'learning', 'learning-audit', 'learning-audit-core', 'red-team', 'guardian', 'bootstrap'
    ], help="Artifact Type")

    # Manifest Command (Manual Management)
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
    man_add.add_argument("--base", help="Target base manifest type (e.g. form)")
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

    # Snapshot Command (Protocol 128 - uses bundler directly, no MCP required)
    snapshot_parser = subparsers.add_parser("snapshot", help="Generate Protocol 128 context snapshots")
    snapshot_parser.add_argument("--type", required=True, choices=[
        'seal', 'learning_audit', 'audit', 'guardian', 'bootstrap'
    ], help="Snapshot type")
    snapshot_parser.add_argument("--manifest", help="Custom manifest path (overrides default)")
    snapshot_parser.add_argument("--output", help="Output path (default: based on type)")
    snapshot_parser.add_argument("--context", help="Strategic context for the snapshot")
    snapshot_parser.add_argument("--override-iron-core", action="store_true", help="⚠️ Override Iron Core check (Requires ADR 090 Amendment)")

    # Persist Soul Command (Protocol 128 Phase VI)
    ps_parser = subparsers.add_parser("persist-soul", help="Broadcast learnings to Hugging Face")
    ps_parser.add_argument("--snapshot", help="Specific snapshot path (default: active seal)")
    ps_parser.add_argument("--valence", type=float, default=0.5, help="Session valence (0.0-1.0)")
    ps_parser.add_argument("--uncertainty", type=float, default=0.0, help="Logic confidence")
    ps_parser.add_argument("--full-sync", action="store_true", help="Sync entire learning directory")

    # Persist Soul Full Command (ADR 081)
    subparsers.add_parser("persist-soul-full", help="Regenerate full JSONL and deploy to HF (ADR 081)")


    # Config Command Group (TS Rules Manager)
    config_parser = subparsers.add_parser("config", help="Update Form Rules (TS)")
    config_subparsers = config_parser.add_subparsers(dest="config_action")
    
    cfg_query = config_subparsers.add_parser("query", help="Get rules for an item")
    cfg_query.add_argument("target", help="Form ID or App Code")
    cfg_query.add_argument("item", help="Item ID")
    
    cfg_update = config_subparsers.add_parser("update", help="Update rules for an item")
    cfg_update.add_argument("target", help="Form ID or App Code")
    cfg_update.add_argument("item", help="Item ID")
    cfg_update.add_argument("--visible", help="Comma-sep roles, or *")
    cfg_update.add_argument("--enabled", help="Comma-sep roles, or *")

    # Workflow Command (New Python Orchestrator)
    wf_parser = subparsers.add_parser("workflow", help="Agent Workflow Orchestration")
    wf_subparsers = wf_parser.add_subparsers(dest="workflow_action")
    
    wf_start = wf_subparsers.add_parser("start", help="Start a new workflow (Safe Pre-flight)")
    wf_start.add_argument("--name", required=True, help="Workflow Name (e.g. codify-form)")
    wf_start.add_argument("--target", required=True, help="Target ID")
    wf_start.add_argument("--type", default="generic", help="Target Type (optional)")

    wf_retro = wf_subparsers.add_parser("retrospective", help="Run Self-Retrospective")

    wf_end = wf_subparsers.add_parser("end", help="End workflow (Commit & Push)")
    wf_end.add_argument("message", nargs="?", help="Commit message")
    wf_end.add_argument("files", nargs="*", help="Files to commit")
    wf_end.add_argument("--force", "-f", action="store_true", help="Skip confirmation prompt")


    args = parser.parse_args()

    if args.command == "roles":
        if args.roles_action == "verify":
            role_name = args.role_name.upper()
            inventory_path = PROJECT_ROOT / ".agent" / "learning" / "roles_inventory.json"
            
            if not inventory_path.exists():
                print(f"❌ Error: Role inventory not found at {inventory_path}")
                sys.exit(1)
                
            try:
                with open(inventory_path, "r", encoding="utf-8") as f:
                    inventory = json.load(f)
                
                if role_name in inventory:
                    status = inventory[role_name].get("status", "Active")
                    print(f"✅ ACTIVE: {role_name} found in inventory (Status: {status})")
                else:
                    print(f"⚠️  LEGACY / DEPRECATED: {role_name} NOT found in inventory.")
            except Exception as e:
                print(f"❌ Error reading inventory: {e}")

        elif args.roles_action == "scan":
            target = args.target
            print(f"🔎 Scanning {target} for Role usage...")
            
            # 1. Load Inventory
            inventory_path = PROJECT_ROOT / ".agent" / "learning" / "roles_inventory.json"
            if not inventory_path.exists():
                print("❌ Inventory not found.")
                sys.exit(1)
            
            with open(inventory_path, "r", encoding="utf-8") as f:
                inventory = json.load(f)
            
            # 2. Locate Source (Simplified - assume XML or text dump exists in context)
            # We'll use the find_xml_file logic from xml_miner if available, or simple glob
            found_roles = {}
            
            # Search strategy: Try to find the bundle first (text), then raw XML
            bundle_path = PROJECT_ROOT / "temp" / "context-bundles" / f"{target.lower()}_context.md"
            if not bundle_path.exists():
                bundle_path = PROJECT_ROOT / "temp" / "context-bundles" / f"{target.lower()}_context_bundle.md"
            source_content = ""
            
            if bundle_path.exists():
                 source_content = bundle_path.read_text(encoding="utf-8", errors="ignore")
            else:
                 # Fallback to recursively searching docs for the file
                 # Simple exact match grep
                 print(f"⚠️ Context bundle not found. Searching source files...")
                 files = list(PROJECT_ROOT.glob(f"docs/**/{target.lower()}*.*"))
                 for p in files:
                     if p.suffix in ['.xml', '.txt', '.md', '.sql']:
                         source_content += p.read_text(encoding="utf-8", errors="ignore")
            
            if not source_content:
                print(f"❌ No source content found for {target}")
                sys.exit(1)
                
            source_content_upper = source_content.upper()
            
            # 3. Scan
            print(f"   Analyzing {len(source_content)} bytes of source...")
            match_count = 0
            for role, meta in inventory.items():
                # Basic whole word check logic usually preferred, but simple substring for now
                # to mimic "grep" rigor.
                # Using regex for word boundary is safer: \bROLE\b
                if re.search(r'\b' + re.escape(role) + r'\b', source_content_upper):
                    status = meta.get("status", "Active")
                    found_roles[role] = status
                    match_count += 1
                    
            # 4. Report
            if found_roles:
                print(f"\n✅ Found {match_count} roles referenced in {target}:")
                print(f"{'ROLE':<30} | {'STATUS':<15}")
                print("-" * 50)
                for r, s in found_roles.items():
                    icon = "🟢" if s == "Active" else "cx🔴"
                    print(f"{icon} {r:<28} | {s}")
            else:
                print(f"\n⚪ No known roles detected in {target}.")

    elif args.command == "scan":
        print(f"Scanning target: {args.target}...")
        try:
             # Route to XML Miner Directly
             from tools.business_rule_extraction.scripts.xml_miner import find_xml_file, mine_declarative_rules
             
             xml_path = find_xml_file(args.target, str(PROJECT_ROOT / "docs" / "source" / "XML"))
             if xml_path and os.path.exists(xml_path):
                 mined_data = mine_declarative_rules(xml_path)
                 print(json.dumps(mined_data, indent=2))
             else:
                 print(f"❌ XML source file not found for {args.target}")
                 
        except ImportError:
             # Fallback if direct import fails (path issues)
             # Fallback if direct import fails (path issues)
             cmd = [sys.executable, str(MINERS_DIR / "xml_miner.py"), "--target", args.target]
             subprocess.run(cmd)

    elif args.command == "investigate":
        if args.investigate_action == "code":
            print(f"🔍 Deep Code Search for pattern: '{args.pattern}' in {args.target}")
            
            # Use search_plsql (enhanced to search bundle or direct file)
            # Strategy: Find the Context Bundle or Markdown Source 
            # We will use the same logic as 'roles scan' to find content, or pass file to search_plsql if it supports it
            
            # Better strategy: Wrap search_plsql structure
            if str(SEARCH_DIR) not in sys.path:
                sys.path.append(str(SEARCH_DIR))
            
            from search_plsql import search_text
            
            # Locate file (Try Markdown first, then XML, then bundle)
            target_lower = args.target.lower()
            candidate_files = []
            
            # 1. Context Bundle (Best for holistic search)
            bundle_path = PROJECT_ROOT / "temp" / "context-bundles" / f"{target_lower}_context.md"
            if bundle_path.exists():
                candidate_files.append(bundle_path)
            
            # 2. Form Module Markdown
            md_path = PROJECT_ROOT / "docs" / "source" / "markdown" / f"{target_lower}-FormModule.md"
            if md_path.exists():
                candidate_files.append(md_path)
                
            # 3. XML Source
            xml_path = PROJECT_ROOT / "docs" / "source" / "XML" / f"{target_lower}_fmb.xml"
            if xml_path.exists():
                candidate_files.append(xml_path)

            if not candidate_files:
                print(f"❌ No source files found for {args.target}")
                return

            print(f"   Scanning {len(candidate_files)} source files...")
            
            all_matches = []
            for fpath in candidate_files:
                matches = search_text(str(fpath), args.pattern, is_regex=True, context=1)
                for m in matches:
                    m['File'] = fpath.name
                    all_matches.append(m)
            
            if all_matches:
                 print(f"✅ Found {len(all_matches)} occurrences:\n")
                 for m in all_matches:
                     print(f"   [{m['File']}:{m['Line']}]  {m['Content'].strip()[:100]}")
            else:
                 print("   No matches found.")

        elif args.investigate_action == "lineage":
            print(f"📉 Analyzing lineage/reachability for {args.target}...")
            cmd = [sys.executable, str(SEARCH_DIR / "reachability.py"), "--target", args.target]
            subprocess.run(cmd)

        elif args.investigate_action == "library":
             print(f"📚 Investigating Library (PLL) {args.target}...")
             try:
                 if str(MINERS_DIR) not in sys.path:
                     sys.path.append(str(MINERS_DIR))
                 from pll_miner import PllMiner
                 miner = PllMiner()
                 miner.scan_plls(target=args.target)
                 
                 # Output JSON to stdout so it can be captured or read
                 print(json.dumps(miner.rules, indent=2))
             except ImportError as e:
                 print(f"❌ Error: Could not import pll_miner: {e}")
             except Exception as e:
                 print(f"❌ Error during library investigation: {e}")

    elif args.command == "candidates":
        pass_args = args.args
        if pass_args and pass_args[0] == "--":
            pass_args = pass_args[1:]
        cmd = [sys.executable, str(SEARCH_DIR / "priority_candidates.py")] + pass_args
        subprocess.run(cmd)

    elif args.command == "br":
        # Wrapper to business_rules_inventory_manager.py
        pass_args = args.args
        if pass_args and pass_args[0] == "--":
            pass_args = pass_args[1:]
        cmd = [sys.executable, str(INVENTORIES_DIR / "business_rules_inventory_manager.py")] + pass_args
        subprocess.run(cmd)

    elif args.command == "bw":
        # Wrapper to business_workflows_inventory_manager.py
        pass_args = args.args
        if pass_args and pass_args[0] == "--":
            pass_args = pass_args[1:]
        cmd = [sys.executable, str(INVENTORIES_DIR / "business_workflows_inventory_manager.py")] + pass_args
        subprocess.run(cmd)


    elif args.command == "dependencies":
        cmd_dep = [sys.executable, str(SEARCH_DIR / "dependencies.py"), "--target", args.target]
        if args.deep:
            cmd_dep.append("--deep")
        if args.json:
            cmd_dep.append("--json")
        if args.direction:
            cmd_dep.extend(["--direction", args.direction])
        subprocess.run(cmd_dep)

    elif args.command == "bundle":
        # New "Super Command" to gather all context in one shot
        data = {"Target": args.target, "Analysis": {}}
        
        # 0. RLM Cache Lookup (Instant Context)

        
        # 1. Miners (Declarative Rules & Logic)
        # XML Miner (Forms)
        try:
            if str(MINERS_DIR) not in sys.path:
                sys.path.append(str(MINERS_DIR))
            
            from xml_miner import find_xml_file, mine_declarative_rules
            
            xml_path = find_xml_file(args.target, str(PROJECT_ROOT / "docs" / "source" / "XML"))
            
            if xml_path and os.path.exists(xml_path):
                mined_data = mine_declarative_rules(xml_path)
                data["Analysis"]["XML_Miner"] = mined_data
            else:
                data["Analysis"]["XML_Miner"] = {"status": "skipped", "reason": "No XML source found"}
                
        except ImportError as e:
             data["Analysis"]["XML_Miner"] = {"error": f"ImportError: {e}"}
        except Exception as e:
            data["Analysis"]["XML_Miner"] = {"error": str(e)}

        # PLL Miner (Libraries)
        try:
            from pll_miner import PllMiner
            miner = PllMiner()
            # PllMiner prints to stdout, we need to capture or modify it. 
            # Ideally PllMiner should return data. Looking at PllMiner.scan_plls, it populates self.rules.
            # We will use scan_plls(target) then access miner.rules.
            
            # Check if PLL exists effectively before running to avoid noise
            miner.scan_plls(target=args.target)
            if any(miner.rules.values()):
                data["Analysis"]["PLL_Miner"] = miner.rules
            else:
                data["Analysis"]["PLL_Miner"] = {"status": "skipped", "reason": "No PLL content found"}
                
        except ImportError:
            data["Analysis"]["PLL_Miner"] = {"error": "ImportError: pll_miner"}
        except Exception as e:
            data["Analysis"]["PLL_Miner"] = {"error": str(e)}

        # MMB Miner (Menus)
        try:
            # Only run if we suspect it is a menu (Target ID pattern or just check file)
            # Simplest approach: Try finding the file
            if str(MINERS_DIR) not in sys.path:
                 sys.path.append(str(MINERS_DIR))
            from mmb_miner import MmbMiner
            
            # MmbMiner logic usually requires path.
            # We can use find_xml_file logic but look for .mmb.xml or standard name
            mmb_xml_path = PROJECT_ROOT / "docs" / "source" / "XML" / f"{args.target.lower()}_mmb.xml"
            
            if mmb_xml_path.exists():
                miner = MmbMiner(str(mmb_xml_path))
                raw_structure = miner.mine_structure()
                data["Analysis"]["MMB_Miner"] = raw_structure
            else:
                 # It might be in 'Forms' if misclassified, or just not a menu.
                 # If we explicitly checked type in a real scenario we'd error, but here we just skip.
                 pass
        except ImportError:
             data["Analysis"]["MMB_Miner"] = {"error": "ImportError: mmb_miner"}
        except Exception as e:
            # Don't clutter unless it's a menu target
            if "MENU" in args.target.upper():
                 data["Analysis"]["MMB_Miner"] = {"error": str(e)}

        # OLB Miner (Object Libraries)
        try:
             if str(MINERS_DIR) not in sys.path:
                 sys.path.append(str(MINERS_DIR))
             from olb_miner import OlbMiner
             
             olb_xml_path = PROJECT_ROOT / "docs" / "source" / "XML" / f"{args.target.lower()}_olb.xml"
             
             if olb_xml_path.exists():
                 miner = OlbMiner(str(olb_xml_path))
                 data["Analysis"]["OLB_Miner"] = miner.mine()
        except ImportError:
             pass
        except Exception as e:
             if "OLB" in args.target.upper() or "LIB" in args.target.upper():
                  data["Analysis"]["OLB_Miner"] = {"error": str(e)}

        # 2. Dependencies (Code Detected + CSV)
        try:
            if str(SEARCH_DIR) not in sys.path:
                 sys.path.append(str(SEARCH_DIR))

            from dependencies import load_dependency_map, find_upstream, find_downstream, deep_search
            
            dep_map = load_dependency_map()
            downstream = find_downstream(args.target, dep_map)
            upstream = find_upstream(args.target, dep_map)
            
            # Perform deep search? bundle usually implies thoroughness
            deep_results = deep_search(args.target)
            
            # Merge upstream logic (similar to dependencies.py main)
            merged_callers = {
                'Forms': list(set(upstream) | set(deep_results.get('Forms', []))),
                'Libraries': deep_results.get('Libraries', []),
                'Packages': deep_results.get('Packages', []),
                'Procedures': deep_results.get('Procedures', []),
                'Functions': deep_results.get('Functions', []),
                'Views': deep_results.get('Views', [])
            }
            
            # Structure for JSON output
            data["Analysis"]["Dependencies"] = {
                "Downstream": downstream,
                "Upstream": merged_callers
            }
            
        except ImportError as e:
             print(f"⚠️  Could not import dependencies: {e}")
             data["Analysis"]["Dependencies"] = {"error": f"ImportError: {e}"}
        except Exception as e:
            data["Analysis"]["Dependencies"] = {"error": str(e)}

        # 3. APPL4 Menu Rules
        try:
            res = subprocess.run([sys.executable, str(MENU_DIR / "appl4_lookup.py"), "--target", args.target], capture_output=True, text=True)
            if res.returncode == 0:
                 data["Analysis"]["APPL4_Menu"] = json.loads(res.stdout)
        except Exception as e:
            data["Analysis"]["APPL4_Menu"] = str(e)

        print(json.dumps(data, indent=2))

    elif args.command == "init-context":
        # Auto-resolve type if not provided
        artifact_type = args.type
        if not artifact_type:
            artifact_type = resolve_type_from_inventory(args.target)
            if not artifact_type:
                # Fallback to 'form' for backward compatibility or error?
                # Ideally error, but 'form' was the old default.
                # Let's default to 'form' but warn.
                print(f"⚠️  Could not resolve type for '{args.target}'. Defaulting to 'form'.")
                artifact_type = "form"
        
        print(f"🚀 Initializing Smart Context Bundle for {args.target} ({artifact_type})...")
        script = str(RETRIEVE_DIR / "manifest_manager.py")
        
        # Note: manifest init now expects --bundle-title, not --target
        subprocess.run([sys.executable, script, "init", "--bundle-title", args.target, "--type", artifact_type])
        
        # Auto-Enrichment
        _enrich_manifest(args.target, script)
        
        subprocess.run([sys.executable, script, "bundle"])

    elif args.command == "manifest":
        script = str(RETRIEVE_DIR / "manifest_manager.py")
        
        # Helper to build base command with globals
        base_cmd = [sys.executable, script]
        if hasattr(args, 'base') and args.base:
            base_cmd.extend(["--base", args.base])
        if hasattr(args, 'manifest') and args.manifest:
            base_cmd.extend(["--manifest", args.manifest])

        if args.manifest_action == "init":
            # Auto-resolve type if not provided
            artifact_type = args.type
            if not artifact_type:
                # Use bundle_title as target ID for resolution
                artifact_type = resolve_type_from_inventory(args.bundle_title)
                if not artifact_type:
                    print(f"❌ Error: Could not resolve type for '{args.bundle_title}'. Please specify --type.")
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
        
        ops = LearningOperations(project_root=str(PROJECT_ROOT))
        
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

    elif args.command == "persist-soul":
        print(f"📡 Initiating Soul Persistence (Protocol 128 Phase VI)...")
        print(f"   Valence: {args.valence} | Uncertainty: {args.uncertainty} | Full Sync: {args.full_sync}")
        ops = LearningOperations(project_root=str(PROJECT_ROOT))
        
        # Default snapshot for seal is usually 'learning/learning_package_snapshot.md'
        snapshot_path = args.snapshot
        if not snapshot_path:
            snapshot_path = ".agent/learning/learning_package_snapshot.md"
            
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

    elif args.command == "persist-soul-full":
        print(f"🧬 Regenerating full Soul JSONL and deploying to HuggingFace (ADR 081)...")
        ops = LearningOperations(project_root=str(PROJECT_ROOT))
        
        result = ops.persist_soul_full()
        
        if result.status == "success":
            print(f"✅ Full Sync Complete!")
            print(f"   Repo: {result.repo_url}")
            print(f"   Output: {result.snapshot_name}")
        else:
            print(f"❌ Error: {result.error}")
            sys.exit(1)

    elif args.command == "applications":
        target = args.target.upper()
        print(f"\n📱 Application Trace for: {target}")
        print("=" * 60)
        
        try:
            if str(SCRIPTS_DIR) not in sys.path:
                sys.path.append(str(SCRIPTS_DIR))
            
            from dependencies import load_dependency_map, trace_applications
            
            dep_map = load_dependency_map()
            apps = trace_applications(target, dep_map, max_depth=3)
            
            if args.json:
                print(json.dumps({'target': target, 'applications': apps}, indent=2))
            else:
                if apps:
                    print("   Reachable from Main Modules:")
                    for module, app_name in sorted(apps.items()):
                        print(f"      - [{app_name}] {module}")
                else:
                    print("   ⚠️  No Main Modules found within 3 levels upstream.")
                    print("   (Form may be orphaned or called via library indirection)")
                print("=" * 60)
                
        except ImportError as e:
            print(f"❌ Error: Could not import dependencies module: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")

    elif args.command == "config":
        # Dynamic import to avoid earlier errors if file didn't exist
        if str(SCRIPTS_DIR) not in sys.path:
            sys.path.append(str(SCRIPTS_DIR))
            
        # config_manager is in tools/business-rule-extraction/ so we use relative path logic or update sys.path
        # Actually SCRIPTS_DIR points to tools/business-rule-extraction/scripts
        # config_manager is one level up
        
        bre_dir = SCRIPTS_DIR.parent
        if str(bre_dir) not in sys.path:
             sys.path.append(str(bre_dir))
             
        try:
            import config_manager as cm
            
            if args.config_action == "query":
                res = cm.query_item(args.target, args.item)
                print(json.dumps(res, indent=2))
                
            elif args.config_action == "update":
                vis_list = args.visible.split(",") if args.visible else None
                en_list = args.enabled.split(",") if args.enabled else None
                
                # Handle wildcards
                if vis_list and "*" in vis_list: vis_list = ["*"]
                if en_list and "*" in en_list: en_list = ["*"]
                
                res = cm.update_item(args.target, args.item, vis_list, en_list)
                print(json.dumps(res, indent=2))
                
        except ImportError as e:
            print(f"❌ Error importing config_manager: {e}")
        except Exception as e:
            print(f"❌ Config Error: {e}")

    elif args.command == "menu":
        # Dynamic imports
        if str(SCRIPTS_DIR) not in sys.path:
            sys.path.append(str(SCRIPTS_DIR))
        
        try:
            from menu_builder import MenuBuilder, save_inventory
            import menu_query as mq
        except ImportError as e:
            print(f"❌ Error importing menu tools: {e}")
            sys.exit(1)

        mb = MenuBuilder()

        if args.menu_action == "rebuild":
            # Rebuild menu inventory from APPL4
            print("🔄 Rebuilding menu inventory from APPL4...")
            try:
                # Build new from APPL4 + Static XML
                new_inventory = mb.build_full_inventory()
                
                # TODO: Implement merge logic if we have manual/discovered items in the future
                # For now, we trust the builder completely.
                
                # Save
                path = save_inventory(new_inventory)
                
                # Also save to React path
                react_path = PROJECT_ROOT / "sandbox" / "ui" / "public" / "config" / "menu_inventory.json"
                if react_path.parent.exists():
                    with open(react_path, "w", encoding="utf-8") as f:
                        json.dump(new_inventory, f, indent=2)
                    print(f"   ✅ Saved to React config: {react_path}")
                
            except Exception as e:
                print(f"❌ Error rebuilding menu: {e}")
        
        elif args.menu_action == "find":
            # Check if a menu item exists
            try:
                inventory = mb.load_inventory()
                if not inventory:
                    print("❌ Inventory not found. Run: cli.py menu rebuild")
                    sys.exit(1)
                
                item_id = args.item_id.upper()
                found_any = False
                
                apps_to_search = [args.app.upper()] if args.app else inventory.get("applications", {}).keys()
                
                for app in apps_to_search:
                    app_data = inventory.get("applications", {}).get(app)
                    if not app_data: continue
                    
                    sections = app_data.get("sections", [])
                    found_items = mq.search_by_id(sections, item_id)
                    
                    if found_items:
                        found_any = True
                        print(f"\n✅ Matches in {app}:")
                        for match in found_items:
                             # match has: id, label, path, roles
                             print(f"   - {match['label']} (ID: {match['id']})")
                             print(f"     Path: {match['path']}")
                             print(f"     Roles (Visible): {match['roles'].get('visible', [])[:5]}...")

                if not found_any:
                     print(f"❌ Not found: {item_id}")
                    
            except Exception as e:
                print(f"❌ Error during find: {e}")
        
        elif args.menu_action == "add":
            print("⚠️ 'menu add' is temporarily disabled pending schema update.")
            print("   Please edit 'Appl4 Menu Item Rules.csv' or XML definitions directly.")
        
        elif args.menu_action == "query":
            # Query with filtering
            try:
                inventory = mb.load_inventory()
                if not inventory:
                    print("❌ Inventory not found. Run: cli.py menu rebuild")
                    sys.exit(1)
                
                # If app is inferred from form
                app = args.app
                if not app and args.form:
                    app = args.form[:3].upper()
                
                if not app:
                    print("❌ Please provide --app or --form")
                    sys.exit(1)
                
                app_data = inventory.get("applications", {}).get(app.upper())
                if not app_data:
                    print(f"❌ Application '{app}' not found in inventory.")
                    sys.exit(1)

                sections = app_data.get("sections", [])
                # Correct signature: nodes, role, form=None
                results = mq.traverse_and_filter(sections, args.role, args.form)
                
                if args.json:
                    print(json.dumps(results, indent=2))
                else:
                    print(f"\n📂 Menu Query Results ({len(results)} items) for {app} " + (f"Form={args.form}" if args.form else "") + ":")
                    print("=" * 80)
                    print(f"{'LABEL':<30} | {'ID':<35} | {'ENABLED'}")
                    print("-" * 80)
                    for item in results:
                        status = "Yes" if item['enabled'] else "No"
                        print(f"{item['label']:<30} | {item['id']:<35} | {status}")
                        
            except Exception as e:
                print(f"❌ Error during query: {e}")
        
        elif args.menu_action == "list":
            # List all for an application
            try:
                inventory = mb.load_inventory()
                if not inventory:
                     print("❌ Inventory not found.")
                     sys.exit(1)
                
                app = args.app.upper()
                app_data = inventory.get("applications", {}).get(app)
                if not app_data:
                     print(f"❌ Application {app} not found.")
                     sys.exit(1)
                
                sections = app_data.get("sections", [])
                results = mq.traverse_and_filter(sections, role=None, form=None)
                
                print(f"\n📂 All Menu Items for {app} ({len(results)} items):")
                print("=" * 70)
                for item in results:
                    print(f"   {item['label']:<30}  ID: {item['id']}")
                    
            except Exception as e:
                print(f"❌ Error listing items: {e}")
        
        elif args.menu_action == "export":
            # Export for React
            try:
                inventory = mb.load_inventory()
                if not inventory:
                    print("❌ Inventory not found.")
                    sys.exit(1)
                
                # Determine App
                app = args.app
                if not app and args.form:
                    app = args.form[:3].upper()
                
                if not app or app not in inventory["applications"]:
                    print(f"❌ Application '{app}' not found in inventory.")
                    sys.exit(1)
                
                # Get Sections
                sections = inventory["applications"][app].get("sections", [])
                
                # Filter Hierarchically
                filtered_sections = mb.filter_sections(sections, args.role, args.form)
                
                # Prepare Output
                output_dir = Path(args.output)
                if not output_dir.exists():
                    os.makedirs(output_dir, exist_ok=True)
                
                filename = f"{app}_{args.form if args.form else 'MENU'}.json"
                output_path = output_dir / filename
                
                with open(output_path, "w", encoding="utf-8") as f:
                     json.dump({"sections": filtered_sections}, f, indent=2)
                
                print(f"✅ Exported menu config to: {output_path}")
                
                # Type Script Generation
                if args.format == "ts":
                    ts_path = output_path.with_suffix('.ts')
                    ts_content = f"""// Auto-generated menu configuration
import {{ MenuSection }} from '../../../types/MenuConfig';

export const menuConfig: {{ sections: MenuSection[] }} = {json.dumps({"sections": filtered_sections}, indent=2)};
"""
                    with open(ts_path, "w", encoding="utf-8") as f:
                        f.write(ts_content)
                    print(f"✅ Exported TypeScript wrapper: {ts_path}")
                    
            except Exception as e:
                print(f"❌ Error exporting menu: {e}")
        
        else:
            print("Usage: cli.py menu [rebuild|find|query|list|export] ...")

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

    elif args.command == "workflow":
        if args.workflow_action == "start":
            try:
                # WorkflowManager is already imported at the top
                manager = WorkflowManager()
                success = manager.start_workflow(args.name, args.target, args.type)
                if not success:
                    sys.exit(1)
            except Exception as e:
                print(f"❌ Workflow Start Failed: {e}")
                sys.exit(1)
        
        elif args.workflow_action == "retrospective":
            try:
                manager = WorkflowManager()
                success = manager.run_retrospective()
                if not success:
                    sys.exit(1)
            except Exception as e:
                print(f"❌ Retrospective Failed: {e}")
                sys.exit(1)

        elif args.workflow_action == "end":
            try:
                manager = WorkflowManager()
                force = getattr(args, 'force', False)
                
                message = args.message
                if not message:
                    # Interactive prompt if running in TTY
                    if sys.stdin.isatty():
                        try:
                            message = input("📝 Enter Commit Message: ").strip()
                        except EOFError:
                            pass
                    
                    if not message:
                        print("❌ Error: Commit message is required.")
                        sys.exit(1)

                success = manager.end_workflow_with_confirmation(message, args.files, force=force)
                if not success:
                    sys.exit(1)
            except Exception as e:
                print(f"❌ Workflow End Failed: {e}")
                sys.exit(1)

    elif args.command == "rules":
        manager_script = str(INVENTORIES_DIR / "business_rules_inventory_manager.py")
        
        if args.rules_action == "search":
            print(f"🔎 Searching Business Rules for: '{args.keyword}'")
            cmd = [sys.executable, manager_script, "--search", args.keyword]
            subprocess.run(cmd)
            
        elif args.rules_action == "register":
            print(f"📝 Registering Rule: {args.title}")
            cmd = [sys.executable, manager_script, "--register", 
                   "--title", args.title, 
                   "--source", args.source, 
                   "--priority", args.priority]
            subprocess.run(cmd)
            
        elif args.rules_action == "update":
            print(f"📝 Updating Rule Summary: {args.id}")
            cmd = [sys.executable, manager_script, "--update-summary", args.id, "--new-summary", args.summary]
            subprocess.run(cmd)

        elif args.rules_action == "candidates":
            # priority_candidates.py arguments reconstruction
            cmd = [sys.executable, str(SEARCH_DIR / "priority_candidates.py")]
            if args.summary: cmd.append("--summary")
            if args.top: cmd.extend(["--top", str(args.top)])
            if args.source: cmd.extend(["--source", args.source])
            if args.type: cmd.extend(["--type", args.type])
            if args.export: cmd.extend(["--export", args.export])
            
            subprocess.run(cmd)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
