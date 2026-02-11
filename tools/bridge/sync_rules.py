
#!/usr/bin/env python3
"""
Sync Rules Bridge
-----------------
Synchronizes specific rule files from .agent/rules/ into the monolithic 
configuration files for other agents (Claude, Gemini, Copilot).

Usage:
    python3 tools/bridge/sync_rules.py --rule standard-workflow-rules.md
    python3 tools/bridge/sync_rules.py --all

Target Files:
    - .claude/CLAUDE.md
    - .github/copilot-instructions.md
    - GEMINI.md
"""

import os
import sys
import argparse
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RULES_DIR = PROJECT_ROOT / ".agent" / "rules"
TARGETS = {
    "CLAUDE": PROJECT_ROOT / ".claude" / "CLAUDE.md",
    "COPILOT": PROJECT_ROOT / ".github" / "copilot-instructions.md",
    "GEMINI": PROJECT_ROOT / "GEMINI.md"
}

MARKER_START = "<!-- RULES_SYNC_START -->"
MARKER_END = "<!-- RULES_SYNC_END -->"

def read_rule(rule_filename):
    """Reads a specific rule file content."""
    rule_path = RULES_DIR / rule_filename
    if not rule_path.exists():
        print(f"Error: Rule file {rule_filename} not found in {RULES_DIR}")
        sys.exit(1)
    return rule_path.read_text()

def get_all_rules_content():
    """Reads all markdown files in the rules directory."""
    content = []
    if not RULES_DIR.exists():
        return ""
        
    for rule_file in sorted(RULES_DIR.glob("*.md")):
        content.append(f"\n\n--- RULE: {rule_file.name} ---\n\n")
        content.append(rule_file.read_text())
    return "".join(content)

def update_file(target_name, target_path, new_content):
    """Updates the target file between markers or appends if not present."""
    if not target_path.exists():
        print(f"[{target_name}] File not found: {target_path}, skipping.")
        return

    original_content = target_path.read_text()
    
    # Construct the block
    injection_block = f"{MARKER_START}\n# SHARED RULES FROM .agent/rules/\n{new_content}\n{MARKER_END}"

    if MARKER_START in original_content and MARKER_END in original_content:
        # Replace existing block
        pre = original_content.split(MARKER_START)[0]
        post = original_content.split(MARKER_END)[1]
        updated_content = pre + injection_block + post
        print(f"[{target_name}] Updating existing rules block.")
    else:
        # Append to end
        updated_content = original_content + "\n\n" + injection_block
        print(f"[{target_name}] Appending new rules block.")

    target_path.write_text(updated_content)

def main():
    parser = argparse.ArgumentParser(description="Sync agent rules to monolithic files.")
    parser.add_argument("--rule", help="Specific rule filename to sync (e.g., standard-workflow-rules.md)")
    parser.add_argument("--all", action="store_true", help="Sync ALL rules from .agent/rules/")
    args = parser.parse_args()

    if args.rule:
        content = read_rule(args.rule)
    elif args.all:
        content = get_all_rules_content()
    else:
        print("Usage: provide --rule [filename] or --all")
        sys.exit(1)

    print(f"Syncing rules to: {list(TARGETS.keys())}")
    for name, path in TARGETS.items():
        update_file(name, path, content)
    print("Done.")

if __name__ == "__main__":
    main()
