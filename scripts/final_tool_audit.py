#!/usr/bin/env python3
"""
Final Audit of Broken Tool References
=====================================

Purpose:
    Generates a unique list of broken 'tools/' references and the files that contain them.
    Follows strict exclusion rules from the user.

Exclusions:
    - Files: *bundle*.md, *.jsonl
    - Directories: .agent/, .github/, .claude/, .kittify/, old_version*/
"""

import os
import re
from pathlib import Path

def get_repo_root():
    return Path(__file__).resolve().parent.parent

def audit():
    root = get_repo_root()
    tools_dir = root / "tools"
    
    # 1. Valid tools inventory
    valid_tools = set()
    if tools_dir.exists():
        for f in tools_dir.rglob("*"):
            if f.is_file():
                valid_tools.add(str(f.relative_to(root)))
    
    # 2. Exclude rules
    exclude_dirs = {'.agent', '.github', '.claude', '.kittify'}
    exclude_file_patterns = [re.compile(r'.*bundle.*\.md$'), re.compile(r'.*\.jsonl$')]
    
    # 3. Find references
    pattern = re.compile(r'tools/[\w\-/.]+')
    broken_to_sources = {}
    source_to_brokens = {}

    print(f"--- Hardened Scan for Broken 'tools/' References ---")

    for f in root.rglob("*"):
        # Check Directory Exclusions
        if any(part in exclude_dirs for part in f.parts) or any(part.startswith('old_version') for part in f.parts):
            continue
            
        if f.is_dir():
            continue
            
        # Check File Exclusions
        if any(pat.match(f.name) for pat in exclude_file_patterns):
            continue
            
        # Only scan readable text formats
        if f.suffix not in {'.md', '.mmd', '.py', '.js', '.ts', '.tsx', '.json', '.sh', '.txt'}:
            continue

        try:
            content = f.read_text(encoding='utf-8', errors='ignore')
            matches = pattern.findall(content)
            
            for match in matches:
                ref = match.strip()
                if ref.endswith('.'): ref = ref[:-1]
                
                # If not a valid file in current tools/
                if ref not in valid_tools:
                    source_rel = str(f.relative_to(root))
                    
                    # Map both ways
                    broken_to_sources.setdefault(ref, set()).add(source_rel)
                    source_to_brokens.setdefault(source_rel, set()).add(ref)
        except Exception:
            continue

    # 4. Generate MD Report
    report_lines = [
        "# Broken Tool References Audit",
        f"**Date:** {os.popen('date').read().strip()}",
        "\n## Summary",
        f"- Found **{len(broken_to_sources)}** unique broken `tools/` references.",
        f"- Found **{len(source_to_brokens)}** unique source files containing these references.",
        "\n---",
        "\n## Unique Broken References (The 'What')",
    ]
    
    for ref in sorted(broken_to_sources.keys()):
        report_lines.append(f"- `[ ]` `{ref}`")

    report_lines.append("\n## Source Files to Review (The 'Where')")
    for src in sorted(source_to_brokens.keys()):
        refs = ", ".join([f"`{r}`" for r in sorted(source_to_brokens[src])])
        report_lines.append(f"- **{src}**")
        report_lines.append(f"  - References: {refs}")

    output_path = root / "broken_tool_references.md"
    output_path.write_text("\n".join(report_lines))
    print(f"Report generated: {output_path}")

if __name__ == "__main__":
    audit()
