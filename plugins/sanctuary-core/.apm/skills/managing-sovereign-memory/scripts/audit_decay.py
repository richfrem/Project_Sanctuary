#!/usr/bin/env python3
"""audit_decay.py - 30-Day Confidence Decay Auditor for Sovereign Memory Layer 2.
Purpose: Enforces the Confidence Decay Protocol across LEARNING/concepts/: Knowledge not re-verified within 30 days decays from CONFIRMED to OBSERVED.
Layer: sanctuary-core / managing-sovereign-memory (Layer 2 Epistemic Lifecycle & Decay).
Key Inputs: Markdown concept files in LEARNING/concepts/*.md.
Key Outputs: Decay detection audit reports and in-place status downgrades (CONFIRMED -> OBSERVED).
Usage:
    python3 plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/scripts/audit_decay.py --check
    python3 plugins/sanctuary-core/.apm/skills/managing-sovereign-memory/scripts/audit_decay.py --apply
Key Functions: get_project_root(), parse_date(), last_touched(), audit_concepts(), main().
"""

import re
import sys
import argparse
import subprocess
from datetime import datetime
from pathlib import Path


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def parse_date(date_str: str) -> datetime | None:
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d-%m-%Y"):
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def last_touched(md_file: Path, root: Path) -> datetime:
    """When the dossier last changed: its last git commit date, else (untracked or no git) its mtime.

    The mtime of a checked-out file is the checkout time, which made every undated dossier look new and the decay audit vacuous.
    """
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cs", "--", str(md_file)], cwd=str(root), capture_output=True,
                             text=True, timeout=10).stdout.strip()
        if out:
            return datetime.strptime(out, "%Y-%m-%d")
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return datetime.fromtimestamp(md_file.stat().st_mtime)


def audit_concepts(root: Path, apply_fixes: bool = False) -> int:
    concepts_dir = root / "LEARNING" / "concepts"
    if not concepts_dir.exists():
        print("No LEARNING/concepts directory found.")
        return 0

    now = datetime.now()
    decay_count = 0
    total_checked = 0

    status_pattern = re.compile(r"\*\*Status:\*\*\s*(CONFIRMED|OBSERVED|HYPOTHESIS)", re.IGNORECASE)
    date_pattern = re.compile(r"\*\*Last Verified:\*\*\s*([0-9\-/]+)", re.IGNORECASE)

    for md_file in sorted(concepts_dir.glob("*.md")):
        total_checked += 1
        content = md_file.read_text(encoding="utf-8")
        status_match = status_pattern.search(content)
        if not status_match:
            continue

        status = status_match.group(1).upper()
        if status != "CONFIRMED":
            continue

        date_match = date_pattern.search(content)
        if not date_match:
            # No verification date: age from the last git commit (mtime only for untracked files or outside git)
            age_days = (now - last_touched(md_file, root)).days
        else:
            v_date = parse_date(date_match.group(1))
            age_days = (now - v_date).days if v_date else 0

        if age_days > 30:
            decay_count += 1
            print(f"[DECAY DETECTED] {md_file.name}: Status CONFIRMED is {age_days} days old (> 30 days).")
            if apply_fixes:
                new_content = status_pattern.sub("**Status:** OBSERVED", content)
                md_file.write_text(new_content, encoding="utf-8")
                print(f"  -> Downgraded {md_file.name} to OBSERVED")

    print(f"\nAudit complete: Checked {total_checked} files. Found {decay_count} decayed entries.")
    return decay_count


def main():
    parser = argparse.ArgumentParser(description="Audit confidence decay in LEARNING/ concepts")
    parser.add_argument("--check", action="store_true", help="Report decayed entries without modifying")
    parser.add_argument("--apply", action="store_true", help="Apply decay changes in-place")
    args = parser.parse_args()

    root = get_project_root()
    apply_fixes = args.apply
    decay_count = audit_concepts(root, apply_fixes=apply_fixes)

    if args.check and decay_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
