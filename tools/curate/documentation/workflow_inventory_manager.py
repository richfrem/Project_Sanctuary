
"""
tools/curate/documentation/workflow_inventory_manager.py
=========================================================

Purpose:
    Manages the workflow inventory for agent workflows (.agent/workflows/*.md).
    Provides search, scan, add, and update capabilities.

Output:
    - docs/antigravity/workflow/workflow_inventory.json
    - docs/antigravity/workflow/WORKFLOW_INVENTORY.md

Usage:
    # Scan and regenerate inventory
    python tools/curate/documentation/workflow_inventory_manager.py --scan
    
    # Search workflows
    python tools/curate/documentation/workflow_inventory_manager.py --search "keyword"
    
    # List all workflows
    python tools/curate/documentation/workflow_inventory_manager.py --list
    
    # Show workflow details
    python tools/curate/documentation/workflow_inventory_manager.py --show "workflow-name"
"""

import sys
import os
import re
import json
import datetime
import argparse
from pathlib import Path

# Add project root to sys.path to enable imports
current = os.path.abspath(os.getcwd())
# Traverse up to find .agent marker
while not os.path.exists(os.path.join(current, ".agent")):
    parent = os.path.dirname(current)
    if parent == current:
        break # Hit root
    current = parent

REPO_ROOT = Path(current)
WORKFLOWS_DIR = REPO_ROOT / ".agent" / "workflows"
OUTPUT_DIR = REPO_ROOT / "docs" / "antigravity" / "workflow"
JSON_PATH = OUTPUT_DIR / "workflow_inventory.json"
MD_PATH = OUTPUT_DIR / "WORKFLOW_INVENTORY.md"


def parse_frontmatter(content):
    """Extract description, inputs, tier, and track from YAML frontmatter."""
    meta = {"description": "No description", "inputs": [], "tier": None, "track": None}
    
    # Match content between --- and ---
    match = re.search(r"^---\s*\n(.*?)\n---", content, re.DOTALL)
    if match:
        yaml_block = match.group(1)
        
        for line in yaml_block.split('\n'):
            line = line.strip()
            if line.startswith('description:'):
                try:
                    meta['description'] = line.split(':', 1)[1].strip()
                except IndexError:
                    pass
            elif line.startswith('inputs:'):
                try:
                    inputs_str = line.split(':', 1)[1].strip()
                    inputs_clean = inputs_str.strip("[]")
                    if inputs_clean:
                        meta['inputs'] = [x.strip() for x in inputs_clean.split(',')]
                except IndexError:
                    pass
            elif line.startswith('tier:'):
                try:
                    meta['tier'] = int(line.split(':', 1)[1].strip())
                except (IndexError, ValueError):
                    pass
            elif line.startswith('track:'):
                try:
                    meta['track'] = line.split(':', 1)[1].strip()
                except IndexError:
                    pass
    
    return meta


def extract_called_by(content):
    """Extract 'Called By' workflows from content."""
    called_by = []
    match = re.search(r"\*\*Called By:\*\*\s*(.*?)(?:\n\n|\n---|\Z)", content, re.DOTALL)
    if match:
        block = match.group(1)
        # Find workflow references like `/codify-form`
        refs = re.findall(r"`(/[a-z\-]+)`", block)
        called_by = refs
    return called_by


def scan_workflows():
    """Scan .agent/workflows/ and return workflow metadata."""
    if not WORKFLOWS_DIR.exists():
        print(f"Error: Workflows dir not found: {WORKFLOWS_DIR}")
        return []

    workflows = []
    for f in sorted(WORKFLOWS_DIR.glob("*.md")):
        try:
            content = f.read_text(encoding='utf-8')
            meta = parse_frontmatter(content)
            called_by = extract_called_by(content)
            
            # Detect tier from content if not in frontmatter
            tier = meta.get('tier')
            if tier is None:
                if "ATOMIC workflow" in content or "Tier 1" in content:
                    tier = 1
                elif "COMPOUND" in content or "Tier 2" in content:
                    tier = 2
                elif "ORCHESTRAT" in content or "Tier 3" in content:
                    tier = 3
            
            # Detect track from filename if not in frontmatter
            track = meta.get('track')
            if track is None:
                if f.name.startswith("spec-kitty."):
                    track = "Discovery"
                else:
                    track = "Factory"

            workflows.append({
                "name": f.stem,
                "filename": f.name,
                "path": str(f.relative_to(REPO_ROOT)).replace("\\", "/"),
                "description": meta["description"],
                "inputs": meta["inputs"],
                "tier": tier,
                "track": track,
                "called_by": called_by
            })
        except Exception as e:
            print(f"Error reading {f.name}: {e}")
    
    return workflows


def load_inventory():
    """Load existing inventory from JSON."""
    if JSON_PATH.exists():
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"metadata": {}, "workflows": []}


def save_inventory(data):
    """Save inventory to JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"Saved: {JSON_PATH}")


def generate_json(workflows):
    """Generate JSON inventory file."""
    data = {
        "metadata": {
            "generatedAt": datetime.datetime.now().isoformat(),
            "count": len(workflows),
            "source": str(WORKFLOWS_DIR.relative_to(REPO_ROOT)).replace("\\", "/")
        },
        "workflows": workflows
    }
    save_inventory(data)


def generate_markdown(workflows):
    """Generate Markdown inventory file."""
    lines = []
    lines.append("# Antigravity Workflow Inventory")
    lines.append(f"\n> **Generated:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> **Total Workflows:** {len(workflows)}\n")
    
    # Define Tracks
    tracks = ["Discovery", "Factory"]
    # Identify shared/meta ops if any (can be added later, for now we stick to 2 tracks)
    
    for track in tracks:
        track_workflows = [w for w in workflows if w.get('track') == track]
        if track_workflows:
            lines.append(f"\n## Track: {track}\n")
            lines.append("| Command | Tier | Description | Called By |")
            lines.append("| :--- | :--- | :--- | :--- |")
            
            for wf in track_workflows:
                cmd = f"`/{wf['name']}`"
                tier = str(wf.get('tier') or '-')
                desc = wf['description'].replace("|", "\\|")
                called_by = ", ".join(wf.get('called_by', [])) or "-"
                lines.append(f"| {cmd} | {tier} | {desc} | {called_by} |")
    
    lines.append("\n## Quick Reference (All)\n")
    lines.append("| Command | Track | Description |")
    lines.append("| :--- | :--- | :--- |")
    
    for wf in sorted(workflows, key=lambda x: x['name']):
        cmd = f"`/{wf['name']}`"
        track = wf.get('track', '-')
        desc = wf['description'].replace("|", "\\|")
        lines.append(f"| {cmd} | {track} | {desc} |")

    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Generated: {MD_PATH}")


def search_workflows(query, workflows=None):
    """Search workflows by keyword."""
    if workflows is None:
        inv = load_inventory()
        workflows = inv.get('workflows', [])
    
    query_lower = query.lower()
    results = []
    
    for wf in workflows:
        # Search in name, description, inputs
        searchable = f"{wf['name']} {wf['description']} {' '.join(wf.get('inputs', []))}".lower()
        if query_lower in searchable:
            results.append(wf)
    
    return results


def list_workflows():
    """List all workflows."""
    inv = load_inventory()
    workflows = inv.get('workflows', [])
    
    if not workflows:
        print("No workflows in inventory. Run --scan first.")
        return
    
    print(f"\n{'='*60}")
    print(f"WORKFLOW INVENTORY ({len(workflows)} workflows)")
    print(f"{'='*60}\n")
    
    # Group by tier
    for tier in [1, 2, 3, None]:
        tier_wfs = [w for w in workflows if w.get('tier') == tier]
        if tier_wfs:
            tier_label = f"Tier {tier}" if tier else "Uncategorized"
            print(f"--- {tier_label} ({len(tier_wfs)}) ---")
            for wf in tier_wfs:
                print(f"  /{wf['name']}: {wf['description'][:50]}...")
            print()


def show_workflow(name):
    """Show detailed info for a workflow."""
    inv = load_inventory()
    workflows = inv.get('workflows', [])
    
    # Find by name (with or without leading /)
    name_clean = name.lstrip('/')
    wf = next((w for w in workflows if w['name'] == name_clean), None)
    
    if not wf:
        print(f"Workflow not found: {name}")
        return
    
    print(f"\n{'='*60}")
    print(f"/{wf['name']}")
    print(f"{'='*60}")
    print(f"Description: {wf['description']}")
    print(f"Tier:        {wf.get('tier', 'Unknown')}")
    print(f"Inputs:      {', '.join(wf['inputs']) if wf['inputs'] else 'None'}")
    print(f"Called By:   {', '.join(wf.get('called_by', [])) or 'None'}")
    print(f"Path:        {wf['path']}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Workflow Inventory Manager - Search, Scan, and Manage Agent Workflows",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  --scan              Scan .agent/workflows/ and regenerate inventory
  --search "form"     Search workflows containing 'form'
  --list              List all workflows grouped by tier
  --show "codify-form"  Show details for a specific workflow
        """
    )
    
    parser.add_argument('--scan', action='store_true',
                        help='Scan workflows dir and regenerate inventory')
    parser.add_argument('--search', type=str, metavar='QUERY',
                        help='Search workflows by keyword')
    parser.add_argument('--list', action='store_true',
                        help='List all workflows')
    parser.add_argument('--show', type=str, metavar='NAME',
                        help='Show details for a workflow')
    
    args = parser.parse_args()
    
    # Default to scan if no args
    if not any([args.scan, args.search, args.list, args.show]):
        args.scan = True
    
    if args.scan:
        print(f"Scanning workflows in {WORKFLOWS_DIR}...")
        workflows = scan_workflows()
        
        if workflows:
            generate_json(workflows)
            generate_markdown(workflows)
            print(f"Done. {len(workflows)} workflows indexed.")
        else:
            print("No workflows found.")
    
    if args.search:
        results = search_workflows(args.search)
        if results:
            print(f"\nFound {len(results)} workflow(s) matching '{args.search}':\n")
            for wf in results:
                tier_label = f"[T{wf.get('tier', '?')}]" if wf.get('tier') else ""
                print(f"  {tier_label} /{wf['name']}: {wf['description']}")
        else:
            print(f"No workflows found matching '{args.search}'")
    
    if args.list:
        list_workflows()
    
    if args.show:
        show_workflow(args.show)


if __name__ == "__main__":
    main()
