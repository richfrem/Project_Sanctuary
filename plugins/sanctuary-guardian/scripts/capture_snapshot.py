#!/usr/bin/env python3
"""
Capture Snapshot Script
=======================
Generates a consolidated snapshot of the project state (Protocol 128 v3.5).
Calls the context-bundler plugin for actual generation.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import hashlib
from datetime import datetime
from pathlib import Path

def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists() or (parent / "README.md").exists():
            return parent
    return Path.cwd()

def get_git_state(project_root: Path) -> dict:
    try:
        git_status_proc = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=str(project_root)
        )
        git_lines = git_status_proc.stdout.splitlines()
        changed_files = set()
        for line in git_lines:
            status_bits = line[:2]
            path = line[3:].split(" -> ")[-1].strip()
            if not path:
                path = line[2:].strip()
            if 'D' not in status_bits: changed_files.add(path)
        
        state_str = "".join(sorted(git_lines))
        state_hash = hashlib.sha256(state_str.encode()).hexdigest()
        return {"lines": git_lines, "changed_files": changed_files, "hash": state_hash}
    except Exception:
        return {"lines": [], "changed_files": set(), "hash": "error"}

def main():
    parser = argparse.ArgumentParser(description="Capture Snapshot (Protocol 128)")
    parser.add_argument("--type", choices=["audit", "learning_audit", "seal"], default="audit", help="Snapshot type")
    parser.add_argument("--context", type=str, default="", help="Strategic context string (for seal)")
    args = parser.parse_args()

    project_root = get_project_root()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    learning_dir = project_root / ".agent" / "learning"
    
    if args.type == "audit":
        output_dir = learning_dir / "red_team"
    elif args.type == "learning_audit":
        output_dir = learning_dir / "learning_audit"
    else:
        output_dir = learning_dir
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Identify manifest
    if args.type == "seal":
        manifest_file = learning_dir / "learning_manifest.json"
    elif args.type == "learning_audit":
        manifest_file = output_dir / "learning_audit_manifest.json"
    else:
        manifest_file = output_dir / "red_team_manifest.json"
        
    if not manifest_file.exists():
        print(json.dumps({"status": "error", "error": f"Manifest not found: {manifest_file}"}, indent=2), file=sys.stderr)
        sys.exit(1)
        
    # Check Git State
    git_state = get_git_state(project_root)
    git_diff_context = git_state["hash"]
    manifest_verified = True
    
    if args.type == "audit":
        try:
            with open(manifest_file, 'r') as f:
                manifest_data = json.load(f)
            
            effective_manifest = set()
            if isinstance(manifest_data, dict) and "files" in manifest_data:
                for item in manifest_data["files"]:
                    val = item if isinstance(item, str) else item.get("path")
                    if val: effective_manifest.add(val)
            elif isinstance(manifest_data, dict):
                effective_manifest.update(manifest_data.get("core", []))
                effective_manifest.update(manifest_data.get("topic", []))
            elif isinstance(manifest_data, list):
                effective_manifest.update(manifest_data)
                
            untracked_changes = git_state["changed_files"] - effective_manifest
            untracked_changes = {f for f in untracked_changes if not any(p in f for p in ["logs/", "temp/", ".temp", ".agent/learning/"])}
            
            untracked_changes_list = list(untracked_changes)[:5] if len(untracked_changes) > 5 else list(untracked_changes)
            if untracked_changes:
                print(json.dumps({
                    "snapshot_path": "",
                    "manifest_verified": False,
                    "git_diff_context": f"REJECTED: Untracked changes in {untracked_changes_list}",
                    "snapshot_type": args.type,
                    "status": "error",
                    "error": "Strict manifestation failed: drift detected"
                }, indent=2))
                return
        except Exception as e:
            print(f"Failed strict manifest verification: {e}", file=sys.stderr)

    # Output file name
    if args.type == "audit": 
        snapshot_filename = "red_team_audit_packet.md"
    elif args.type == "learning_audit": 
        snapshot_filename = "learning_audit_packet.md"
    elif args.type == "seal":
        snapshot_filename = "learning_package_snapshot.md"
    else:
        snapshot_filename = f"{args.type}_snapshot_{timestamp}.md"
        
    final_snapshot_path = output_dir / snapshot_filename

    # Call Context Bundler
    bundler_script = project_root / "plugins" / "context-bundler" / "scripts" / "bundle.py"
    if not bundler_script.exists():
        print(json.dumps({"status": "error", "error": "Context bundler script not found"}, indent=2), file=sys.stderr)
        sys.exit(1)
        
    bundler_cmd = [
        sys.executable,
        str(bundler_script),
        str(manifest_file),
        "-o", str(final_snapshot_path)
    ]
    
    result = subprocess.run(bundler_cmd, cwd=str(project_root), capture_output=True, text=True)
    
    if result.returncode != 0:
        err_msg = result.stderr[:200] if result.stderr and len(result.stderr) > 200 else result.stderr
        print(json.dumps({
            "status": "error",
            "error": f"Bundler failed: {err_msg}"
        }, indent=2), file=sys.stderr)
        sys.exit(1)
        
    if not final_snapshot_path.exists():
        print(json.dumps({
            "status": "error",
            "error": "Snapshot generation failed (file not created)"
        }, indent=2), file=sys.stderr)
        sys.exit(1)

    # Inject Strategic Context Hologram
    if args.type == "seal":
        # Check for RLM synthesized hologram
        rlm_script = project_root / "plugins" / "rlm-factory" / "scripts" / "cortex_synthesis.py"
        strategic_context = args.context
        
        if not strategic_context and rlm_script.exists():
            try:
                rlm_res = subprocess.run(
                    [sys.executable, str(rlm_script)],
                    cwd=str(project_root), capture_output=True, text=True, timeout=120
                )
                if rlm_res.returncode == 0:
                    try:
                        rlm_data = json.loads(rlm_res.stdout)
                        strategic_context = rlm_data.get("hologram", "")
                    except:
                        strategic_context = rlm_res.stdout.strip()
            except Exception:
                pass
                
        if strategic_context:
            try:
                existing_content = final_snapshot_path.read_text(encoding='utf-8')
                final_snapshot_path.write_text(strategic_context + "\n\n---\n\n" + existing_content, encoding='utf-8')
            except Exception as e:
                print(f"Warning: Failed to inject strategic context: {e}", file=sys.stderr)

    file_stat = final_snapshot_path.stat()
    print(json.dumps({
        "snapshot_path": str(final_snapshot_path.relative_to(project_root)),
        "manifest_verified": manifest_verified,
        "git_diff_context": git_diff_context,
        "snapshot_type": args.type,
        "status": "success",
        "total_bytes": file_stat.st_size
    }, indent=2))

if __name__ == "__main__":
    main()
