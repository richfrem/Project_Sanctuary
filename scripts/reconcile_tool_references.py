#!/usr/bin/env python3
"""
Reconcile Tool References
=========================

Purpose:
    Identifies broken references to 'tools/' in the codebase and maps them to their 
    new locations in the 'plugins/' directory. 

Logic:
    1. Inventory current valid files in tools/.
    2. Search codebase for "tools/" strings.
    3. Filter out references that still exist in tools/.
    4. Search plugins/ for matching basenames of broken references.
    5. Generate a mapping report.

Usage:
    python3 reconcile_tool_references.py
"""

import os
import re
from pathlib import Path

def get_repo_root():
    return Path(__file__).resolve().parent.parent

def reconcile():
    root = get_repo_root()
    tools_dir = root / "tools"
    plugins_dir = root / "plugins"
    
    # 1. Valid tools
    # We still check this to be safe, but we primarily use physical existence check
    valid_tools = set()
    if tools_dir.exists():
        for f in tools_dir.rglob("*"):
            if f.is_file():
                valid_tools.add(str(f.relative_to(root)))
    
    # 2. Plugins inventory (basename -> relative_path)
    plugin_map = {}
    if plugins_dir.exists():
        for f in plugins_dir.rglob("*"):
            if f.is_file():
                # Store full path for mapping, but also basename for discovery
                plugin_map[f.name] = str(f.relative_to(root))

    # 3. Find references
    broken_refs = {}
    pattern = re.compile(r'tools/[\w\-/.]+')
    
    # Exclusions
    exclude_dirs = {'.agent', '.github', '.claude', '.kittify', '.vendor', '.venv', 'dataset_package', '.vector_data'}
    exclude_file_patterns = [re.compile(r'.*bundle.*\.md$'), re.compile(r'.*\.jsonl$')]
    
    print(f"--- Scanning codebase for 'tools/' references ---")
    
    # Extensions to scan
    extensions = {'.md', '.mmd', '.py', '.js', '.ts', '.tsx', '.json', '.sh', '.txt', '.jsonl'}
    
    for f in root.rglob("*"):
        # Check Directory Exclusions
        if any(part in exclude_dirs for part in f.parts) or any(part.startswith('old_version') for part in f.parts):
            continue
            
        if f.is_dir() or f.suffix not in extensions:
            continue
        
        # Check File Exclusions
        if any(pat.match(f.name) for pat in exclude_file_patterns):
            continue

        try:
            content = f.read_text(encoding='utf-8', errors='ignore')
            matches = pattern.findall(content)
            
            for match in matches:
                # Normalize reference
                ref = match.strip()
                if ref.endswith('.'): ref = ref[:-1] # Clean up trailing dots
                
                # USER OVERRIDE: Ignore tools/ai-resources
                if ref.startswith("tools/ai-resources/"):
                    continue

                # Check if it actually exists in tools/
                phys_path = root / ref
                if phys_path.exists():
                    continue

                if ref not in broken_refs:
                    broken_refs[ref] = set()
                broken_refs[ref].add(str(f.relative_to(root)))
        except Exception:
            continue

    # 4. Attempt mapping
    print(f"\n--- Mapping {len(broken_refs)} Broken References ---")
    mapping = {}
    unmapped = []
    
    for ref, files in broken_refs.items():
        ref_lower = ref.lower()
        basename = os.path.basename(ref)
        target_path = None

        # RULE 1: RLM
        if "rlm" in ref_lower:
            target_path = "plugins/rlm-factory/"
            # Try to find specific script match
            for p_name, p_path in plugin_map.items():
                if "rlm-factory" in p_path and p_name == basename:
                    target_path = p_path
                    break
        
        # RULE 2: Vector
        elif "vector" in ref_lower:
            target_path = "plugins/vector-db/"
            for p_name, p_path in plugin_map.items():
                if "vector-db" in p_path and p_name == basename:
                    target_path = p_path
                    break
        
        # RULE 3: Bundler / Context Bundler
        elif "context-bundler" in ref_lower or "capture-code-snapshot.js" in ref:
            target_path = "plugins/context-bundler/"
            for p_name, p_path in plugin_map.items():
                if "context-bundler" in p_path and p_name == basename:
                    target_path = p_path
                    break
        
        # RULE 4: Bridge
        elif "tools/bridge/" in ref or ref == "tools/bridge":
            target_path = "plugins/plugin-mapper/"

        # RULE 5: Link Checker
        elif "link-checker" in ref_lower:
            target_path = "plugins/link-checker/"
            for p_name, p_path in plugin_map.items():
                if "link-checker" in p_path and p_name == basename:
                    target_path = p_path
                    break
        
        # RULE 6: Basename fallback
        elif basename in plugin_map:
            target_path = plugin_map[basename]
            
        if target_path:
            mapping[ref] = {"new_path": target_path, "found_in": list(files)}
        else:
            unmapped.append(ref)

    # 5. Output Report
    print(f"\nFound {len(mapping)} broken references that can be mapped.")

    # Generate the Markdown Report (UNMAPPED ONLY)
    report_path = root / "broken_tool_references.md"
    report_lines = [
        "# Broken Tool References Audit (Unmapped Only)",
        f"**Date:** {os.popen('date').read().strip()}\n",
        "## Summary",
        f"- Unique broken `tools/` references found: **{len(broken_refs)}**",
        f"- Unmapped references: **{len(unmapped)}**",
        f"- Source files containing unmapped references: **{len(set().union(*[broken_refs[r] for r in unmapped])) if unmapped else 0}**\n",
        "---",
        "\n## Unmapped Broken References (The 'What')"
    ]
    
    if not unmapped:
        report_lines.append("- **No unmapped references found!** All identified broken references have migration targets.")
    else:
        for ref in sorted(unmapped):
            report_lines.append(f"- `[ ]` `{ref}`")
            # List source files for this reference
            source_files = sorted(list(broken_refs[ref]))
            for sf in source_files[:3]:
                report_lines.append(f"  - `{sf}`")
            if len(source_files) > 3:
                report_lines.append(f"  - ... and {len(source_files)-3} more files")

    report_lines.append("\n## Source Files Containing Unmapped References (The 'Where')")
    # Group by file for readability
    file_to_refs = {}
    for ref in unmapped:
        files = broken_refs[ref]
        for f in files:
            if f not in file_to_refs: file_to_refs[f] = set()
            file_to_refs[f].add(ref)
    
    for f in sorted(file_to_refs.keys()):
        report_lines.append(f"- **{f}**")
        report_lines.append(f"  - References: {', '.join(sorted(list(file_to_refs[f])))}")

    report_path.write_text("\n".join(report_lines))
    print(f"\nGenerated UNMAPPED audit report at: {report_path.relative_to(root)}")

    # Generate a replacement script
    if mapping:
        script_path = root / "scripts" / "apply_tool_migration.sh"
        lines = ["#!/bin/bash", "# Auto-generated tool migration script\n"]
        for old, data in mapping.items():
            new = data["new_path"]
            old_esc = old.replace("/", "\\/").replace(".", "\\.")
            new_esc = new.replace("/", "\\/").replace(".", "\\.")
            lines.append(f"echo 'Updating {old} -> {new}...'")
            for f in data["found_in"]:
                lines.append(f"sed -i '' 's/{old_esc}/{new_esc}/g' \"{f}\"")
        
        script_path.write_text("\n".join(lines))
        os.chmod(script_path, 0o755)
        print(f"Created migration script at: {script_path.relative_to(root)}")

if __name__ == "__main__":
    reconcile()
