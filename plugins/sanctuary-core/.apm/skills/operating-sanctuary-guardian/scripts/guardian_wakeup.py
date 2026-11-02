#!/usr/bin/env python3
"""
guardian_wakeup.py - Sub-4ms Protocol 128 Holistic Wakeup Bootloader
===================================================================

Purpose:
    Provides sub-4ms situational awareness and cognitive hydration for autonomous
    AI agents booting into Project Sanctuary. Evaluates Layer 1 context integrity,
    current git branch state, and recent institutional decisions.

    By default, operations are strictly read-only to obey the Human Gate Law.
    To persist the briefing to disk, supply `--write-digest`.

Usage:
    python3 scripts/guardian_wakeup.py --mode HOLISTIC
    python3 scripts/guardian_wakeup.py --mode HOLISTIC --write-digest
    python3 scripts/guardian_wakeup.py --format markdown
"""

import os
import sys
import time

# Start wall-clock measurement immediately at top of process
PROCESS_START_TIME = time.perf_counter()

import json
import logging
import argparse
import subprocess
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
logger = logging.getLogger("guardian_wakeup")


def get_project_root() -> Path:
    if os.environ.get("PROJECT_ROOT"):
        return Path(os.environ["PROJECT_ROOT"]).resolve()
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def get_git_telemetry(project_root: Path) -> dict[str, str | bool]:
    """Retrieve actual git branch and status dynamically."""
    branch = "unknown"
    dirty = False
    try:
        res = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=2
        )
        if res.returncode == 0:
            branch = res.stdout.strip()

        status_res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=2
        )
        if status_res.returncode == 0:
            dirty = bool(status_res.stdout.strip())
    except Exception:
        pass

    return {"branch": branch, "dirty": dirty}


def check_layer1_health(project_root: Path) -> tuple[str, list[dict[str, str]]]:
    """Inspect presence of Layer 1 fast-boot context files."""
    checks = []
    required_files = [
        ("context/soul.md", "Soul Identity & Anti-Asch Principles"),
        ("context/user.md", "Human Partner Preferences & Conventions"),
        ("context/memory.md", "Curated Long-Term Memory & Active Decisions"),
        ("chronicle/ledger.json", "Episodic Chronicle Index")
    ]

    all_passed = True
    for rel_path, desc in required_files:
        p = project_root / rel_path
        exists = p.exists() and p.is_file()
        size = p.stat().st_size if exists else 0
        checks.append({
            "path": rel_path,
            "description": desc,
            "status": "PASS" if exists and size > 0 else "FAIL",
            "size_bytes": size
        })
        if not exists or size == 0:
            all_passed = False

    health_status = "GREEN" if all_passed else "YELLOW"
    return health_status, checks


def get_tactical_priorities(project_root: Path) -> str:
    memory_file = project_root / "context" / "memory.md"
    if memory_file.exists():
        lines = [line.strip() for line in memory_file.read_text(encoding="utf-8").splitlines() if line.strip().startswith("- ")]
        if lines:
            return "* Active Context:\n" + "\n".join(f"  {line}" for line in lines[:4])
    return "* Sovereign engineering habitat active. APM capabilities nominal."


def get_preflight_brief(project_root: Path) -> str:
    ledger_file = project_root / "chronicle" / "ledger.json"
    entry_count = 0
    if ledger_file.exists():
        try:
            with open(ledger_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                entry_count = len(data.get("entries", [])) if isinstance(data, dict) else len(data)
        except Exception:
            pass

    return (f"* Sovereign Memory Engine active (Layer 1 fast-boot + Layer 2 progressive disclosure).\n"
            f"* Living Chronicle indexed: {entry_count} entries recorded in `chronicle/ledger.json`.")


def main():
    parser = argparse.ArgumentParser(description="Generate Guardian Boot Digest (Protocol 128)")
    parser.add_argument("--mode", type=str, default="HOLISTIC", help="Wakeup mode")
    parser.add_argument("--write-digest", action="store_true", help="Explicitly persist boot digest to .agent/learning/guardian_boot_digest.md")
    parser.add_argument("--format", choices=["json", "markdown"], default="json", help="Output format to stdout")
    args = parser.parse_args()

    project_root = get_project_root()
    git_meta = get_git_telemetry(project_root)
    health_status, checks = check_layer1_health(project_root)

    digest_lines = [
        "# 🛡️ Guardian Wakeup Briefing (Protocol 128 Holistic Wakeup)",
        f"**System Status:** {health_status} - Layer 1 Fast-Boot Hydrated",
        f"**Git Environment:** Branch `{git_meta['branch']}` ({'DIRTY' if git_meta['dirty'] else 'CLEAN'})",
        f"**Generated Time:** {time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC",
        "",
        "## I. Strategic Directives",
        "* **Core Mandate:** Anti-Asch epistemic platform, sovereign memory architecture, and pair-programming partnership with Richard.",
        "* **Supreme Law:** Never perform unapproved state-changing operations or git pushes.",
        "",
        "## II. Tactical Priorities",
        get_tactical_priorities(project_root),
        "",
        "## III. Pre-Flight Brief (Sovereign Memory Engine)",
        get_preflight_brief(project_root),
        "",
        "## IV. Layer 1 Integrity Health Checks",
    ]

    for check in checks:
        digest_lines.append(f"- [{check['status']}] `{check['path']}`: {check['description']} ({check['size_bytes']} bytes)")

    digest_text = "\n".join(digest_lines)
    digest_path_str = None

    if args.write_digest:
        learning_dir = project_root / ".agent" / "learning"
        digest_path = learning_dir / "guardian_boot_digest.md"
        digest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(digest_path, 'w', encoding='utf-8') as f:
            f.write(digest_text)
        digest_path_str = str(digest_path.relative_to(project_root))

    total_time_ms = (time.perf_counter() - PROCESS_START_TIME) * 1000

    if args.format == "markdown":
        print(digest_text)
    else:
        output_payload = {
            "status": "success" if health_status == "GREEN" else "warning",
            "health": health_status,
            "git": git_meta,
            "digest_path": digest_path_str,
            "digest_persisted": bool(args.write_digest),
            "checks": checks,
            "total_time_ms": total_time_ms
        }
        print(json.dumps(output_payload, indent=2))


if __name__ == "__main__":
    main()
