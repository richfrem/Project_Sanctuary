#!/usr/bin/env python3
"""
Learning Debrief Script
=======================
Scans the project for technical state changes (Protocol 128 Phase I).
Extracted from legacy MCP server implementation.
"""

import os
import sys
import re
import json
import time
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists() or (parent / "README.md").exists():
            return parent
    return Path.cwd()

def get_git_diff_summary(project_root: Path, file_path: str) -> str:
    try:
        result = subprocess.run(
            ["git", "diff", "--shortstat", "HEAD", file_path],
            cwd=str(project_root), capture_output=True, text=True, timeout=1
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    except Exception:
        pass
    return ""

def get_recency_delta(project_root: Path, hours: int = 48) -> str:
    try:
        delta = timedelta(hours=hours)
        cutoff_time = time.time() - delta.total_seconds()
        now = time.time()
        
        recent_files = []
        scan_dirs = ["00_CHRONICLE/ENTRIES", "01_PROTOCOLS", "plugins", "02_USER_REFLECTIONS"]
        allowed_extensions = {".md", ".py", ".ts", ".tsx"}
        
        for directory in scan_dirs:
            dir_path = project_root / directory
            if not dir_path.exists(): continue
            
            for file_path in dir_path.rglob("*"):
                if not file_path.is_file(): continue
                if file_path.suffix not in allowed_extensions: continue
                if "__pycache__" in str(file_path): continue
                
                mtime = file_path.stat().st_mtime
                if mtime > cutoff_time:
                    recent_files.append((file_path, mtime))
        
        if not recent_files:
            return "* **Recent Files Modified (48h):** None"
            
        recent_files.sort(key=lambda x: x[1], reverse=True)
        
        git_info = "[Git unavailable]"
        try:
            result = subprocess.run(
                ["git", "log", "-1", "--oneline"],
                cwd=str(project_root), capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0: git_info = result.stdout.strip()
        except Exception:
            pass
        
        lines = [f"* **Most Recent Commit:** {git_info}", f"* **Recent Files Modified ({hours}h):**"]
        
        recent_files_list = list(recent_files)[:10]
        for file_path, mtime in recent_files_list:
            relative_path = file_path.relative_to(project_root)
            age_seconds = now - mtime
            if age_seconds < 3600: age_str = f"{int(age_seconds / 60)}m ago"
            elif age_seconds < 86400: age_str = f"{int(age_seconds / 3600)}h ago"
            else: age_str = f"{int(age_seconds / 86400)}d ago"
            
            context = ""
            try:
                with open(file_path, 'r', errors='ignore') as f:
                    content = f.read(500)
                    if file_path.suffix == ".md":
                        title_match = re.search(r"^#\s+(.+)$", content, re.MULTILINE)
                        if title_match: context = f" → {title_match.group(1)}"
                    elif file_path.suffix in {".py", ".ts", ".tsx"}:
                        if "def " in content or "class " in content or "function " in content:
                            context = " → Implementation changes"
            except Exception:
                pass
            
            diff_summary = get_git_diff_summary(project_root, str(relative_path))
            if diff_summary: context += f" [{diff_summary}]"
            
            lines.append(f"    * `{relative_path}` ({age_str}){context}")
        
        return "\n".join(lines)
    except Exception as e:
        return f"Error generating recency delta: {str(e)}"

def generate_debrief(hours: int) -> str:
    project_root = get_project_root()
    
    # 1. Seek Truth (Git)
    git_evidence = "Git Not Available"
    try:
        result = subprocess.run(
            ["git", "diff", "--stat", "HEAD"],
            capture_output=True, text=True, cwd=str(project_root)
        )
        git_evidence = result.stdout if result.stdout else "No uncommitted code changes found."
    except Exception as e:
        git_evidence = f"Git Error: {e}"

    # 2. Scan Recency (Filesystem)
    recency_summary = get_recency_delta(project_root, hours=hours)
    
    # 3. Read Core Documents
    primer_content = "[MISSING] plugins/guardian-onboarding/resources/cognitive_primer.md"
    protocol_content = "[MISSING] plugins/guardian-onboarding/resources/protocols/128_Hardened_Learning_Loop.md"
    
    try:
        p_path = project_root / ".agent" / "learning" / "cognitive_primer.md"
        if p_path.exists(): primer_content = p_path.read_text(errors='ignore')
        
        pr_path = project_root / "plugins" / "guardian-onboarding" / "resources" / "protocols" / "128_Hardened_Learning_Loop.md"
        if pr_path.exists(): protocol_content = pr_path.read_text(errors='ignore')
    except Exception as e:
        print(f"Warning: Error reading sovereignty docs: {e}", file=sys.stderr)

    # 4. Strategic Context (Learning Package Snapshot)
    last_package_content = "⚠️ No active Learning Package Snapshot found."
    package_path = project_root / ".agent" / "learning" / "learning_package_snapshot.md"
    package_status = "ℹ️ No `.agent/learning/learning_package_snapshot.md` detected."
    
    if package_path.exists():
        try:
            mtime = package_path.stat().st_mtime
            delta_hours = (datetime.now().timestamp() - mtime) / 3600
            if delta_hours <= hours:
                last_package_content = package_path.read_text(errors='ignore')
                package_status = f"✅ Loaded Learning Package Snapshot from {delta_hours:.1f}h ago."
            else:
                last_package_content = package_path.read_text(errors='ignore')
                package_status = f"⚠️ Snapshot found but too old ({delta_hours:.1f}h)."
        except Exception as e:
            package_status = f"❌ Error reading snapshot: {e}"

    # 5. Create Draft
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    
    primer_slice = primer_content[:1000] if len(primer_content) > 1000 else primer_content
    protocol_slice = protocol_content[:1000] if len(protocol_content) > 1000 else protocol_content
    package_slice = last_package_content[:2000] if len(last_package_content) > 2000 else last_package_content
    
    lines = [
        f"# [HARDENED] Learning Package Snapshot v5.0",
        f"**Scan Time:** {timestamp} (Window: {hours}h)",
        f"**Strategic Status:** ✅ Successor Context Active",
        "",
        "## I. The Truth (System State)",
        f"**Git Status:**\n```\n{git_evidence}\n```",
        "",
        f"## II. The Change (Recency Delta - {hours}h)",
        recency_summary,
        "",
        "## III. The Law (Protocol 128 - Cognitive Continuity)",
        "> *\"We do not restart. We reload.\"*",
        "### A. The Cognitive Primer (Constitution)",
        f"```markdown\n{primer_slice}...\n```",
        "",
        "### B. Protocol 128 Extract",
        f"```markdown\n{protocol_slice}...\n```",
        "",
        "## IV. The Strategy (Successor Context)",
        f"**Snapshot Status:** {package_status}",
        "### Active Context (Previous Cycle):",
        f"```markdown\n{package_slice}...\n```",
    ]
    
    return "\n".join(lines)

def main():
    parser = argparse.ArgumentParser(description="Scan project for technical state changes")
    parser.add_argument("--hours", type=int, default=24, help="Hours to look back for changes")
    parser.add_argument("--output", type=str, help="Optional output file to write debrief to")
    args = parser.parse_args()
    
    try:
        debrief_text = generate_debrief(args.hours)
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(debrief_text)
            print(json.dumps({"status": "success", "file": args.output}, indent=2))
        else:
            print(debrief_text)
    except Exception as e:
        print(json.dumps({"status": "error", "error": str(e)}, indent=2), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
