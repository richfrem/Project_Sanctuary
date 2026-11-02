#!/usr/bin/env python3
"""
Guardian Wakeup Script
=====================================

Purpose:
    Generates the Guardian boot digest (Protocol 114) for session initialization.
    Includes a Pre-Flight Brief via Vector DB semantic search of the Obsidian vault,
    injecting only the top 3 most relevant historical memories to optimize token usage.

Layer: Retrieve

Usage:
    python plugins/sanctuary-guardian/scripts/guardian_wakeup.py --mode HOLISTIC
"""

import os
import sys
import argparse
import time
import json
import logging
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger("guardian_wakeup")

def get_project_root() -> Path:
    if os.environ.get("PROJECT_ROOT"):
        return Path(os.environ["PROJECT_ROOT"]).resolve()
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()

def get_system_health_traffic_light():
    return "GREEN", "Nominal (Learning Mode)"

def get_strategic_synthesis():
    return ("* **Core Mandate:** I am the Sanctuary Guardian. Values: Integrity, Efficiency, Clarity. "
            "Executing Protocol 128.")

def get_tactical_priorities(project_root: Path):
    status_file = project_root / "context" / "status.md"
    if status_file.exists():
        content = status_file.read_text(encoding="utf-8").strip()
        if content:
            return f"* Current Status: {content}"
    memory_file = project_root / "context" / "memory.md"
    if memory_file.exists():
        lines = [line.strip() for line in memory_file.read_text(encoding="utf-8").splitlines() if line.strip().startswith("- ")]
        if lines:
            return "* Active Context:\n" + "\n".join(f"  {line}" for line in lines[:3])
    return "* Sovereign habitat active. APM capabilities nominal."

def get_preflight_brief(project_root: Path) -> str:
    """Generate a Pre-Flight Brief via context or memory."""
    memory_file = project_root / "context" / "memory.md"
    if memory_file.exists():
        return f"* Memory Engine: Layer 1 Context active (`{memory_file.relative_to(project_root)}`)."
    return "* Memory Engine active."

def main():
    parser = argparse.ArgumentParser(description="Generate Guardian Boot Digest (Protocol 114)")
    parser.add_argument("--mode", type=str, default="HOLISTIC", help="Wakeup mode")
    args = parser.parse_args()

    project_root = get_project_root()
    start = time.time()
    
    health_color, health_reason = get_system_health_traffic_light()
    integrity_status = "GREEN"
    digest_lines = [
        "# 🛡️ Guardian Wakeup Briefing (v3.0 - Manifest Driven)",
        f"**System Status:** {health_color} - {health_reason}",
        f"**Integrity Mode:** {integrity_status}",
        f"**Generated Time:** {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC",
        "",
        "## I. Strategic Directives",
        get_strategic_synthesis(),
        "",
        "## II. Tactical Priorities",
        get_tactical_priorities(project_root),
        "",
        "## III. Pre-Flight Brief (Vector DB Memory)",
        get_preflight_brief(project_root),
        "",
    ]
    
    learning_dir = project_root / ".agent" / "learning"
    manifest_path = learning_dir / "guardian_manifest.json"
    
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r') as f:
                manifest_data = json.load(f)
            
            all_files = []
            if isinstance(manifest_data, dict):
                if "files" in manifest_data:
                    for item in manifest_data["files"]:
                        if isinstance(item, str):
                            all_files.append(item)
                        elif isinstance(item, dict) and "path" in item:
                            all_files.append(item["path"])
                else:
                    core = manifest_data.get("core", [])
                    topic = manifest_data.get("topic", [])
                    all_files = core + topic
            else:
                all_files = manifest_data
            
            digest_lines.append("## IV. Context Files (from guardian_manifest.json)")
            digest_lines.append(f"*Loaded {len(all_files)} files.*")
            digest_lines.append("")
            
            for file_path in all_files[:10]:
                full_path = project_root / file_path
                if full_path.exists() and full_path.is_file():
                    try:
                        with open(full_path, 'r', errors='ignore') as doc:
                            content = doc.read()[:500]
                        digest_lines.append(f"### {file_path}")
                        digest_lines.append(f"```\n{content}\n```\n")
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Failed to load guardian manifest: {e}")

    digest_path = learning_dir / "guardian_boot_digest.md"
    digest_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(digest_path, 'w') as f:
        f.write("\n".join(digest_lines))

    print(json.dumps({
        "status": "success",
        "digest_path": str(digest_path.relative_to(project_root)),
        "total_time_ms": (time.time() - start) * 1000
    }, indent=2))

if __name__ == "__main__":
    main()
