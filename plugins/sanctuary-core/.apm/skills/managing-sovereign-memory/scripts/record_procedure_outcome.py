#!/usr/bin/env python3
"""
record_procedure_outcome.py - Procedural Memory Outcome Tracking CLI
===================================================================

Purpose:
    Records deterministic execution outcomes (success, failure, blocked)
    for procedural recipes in `LEARNING/procedures/`. Updates frontmatter
    counters bound to specific versions and runtime environments, and
    emits an immutable event to `.agent/learning/events.jsonl`.

Usage:
    python3 record_procedure_outcome.py --id proc-git-feature-branch --outcome success
    python3 record_procedure_outcome.py --file LEARNING/procedures/git/create_feature_branch.md --outcome failure --reason "Merge conflict"
"""

import os
import sys
import json
import uuid
import yaml
import platform
import argparse
from pathlib import Path
from datetime import datetime, timezone


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def find_procedure_file(root: Path, proc_id: str | None, file_path: str | None) -> Path | None:
    if file_path:
        p = (root / file_path).resolve()
        if p.exists() and p.is_file():
            return p
        return None

    if not proc_id:
        return None

    proc_dir = root / "LEARNING" / "procedures"
    if not proc_dir.exists():
        return None

    for f in proc_dir.rglob("*.md"):
        if f.name == "README.md":
            continue
        try:
            content = f.read_text(encoding="utf-8")
            if f"id: {proc_id}" in content or f'id: "{proc_id}"' in content or f"id: '{proc_id}'" in content:
                return f
        except Exception:
            continue
    return None


def parse_frontmatter(content: str) -> tuple[dict, str, str]:
    """Extract frontmatter dict, raw frontmatter str, and body."""
    lines = content.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        return {}, "", content

    fm_lines = []
    body_idx = 1
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            body_idx = i + 1
            break
        fm_lines.append(lines[i])

    fm_raw = "".join(fm_lines)
    body = "".join(lines[body_idx:])
    try:
        data = yaml.safe_load(fm_raw) or {}
    except Exception:
        # Fallback simple parser if yaml safe_load encounters issues
        data = {}
        for line in fm_lines:
            if ":" in line and not line.strip().startswith("-"):
                k, v = line.split(":", 1)
                data[k.strip()] = v.strip().strip('"').strip("'")
    return data, fm_raw, body


def main():
    parser = argparse.ArgumentParser(description="Record Procedural Memory Execution Outcomes")
    parser.add_argument("--id", help="Procedure ID (e.g. proc-git-feature-branch)")
    parser.add_argument("--file", help="Path to procedure markdown file")
    parser.add_argument("--outcome", choices=["success", "failure", "blocked"], required=True, help="Execution outcome")
    parser.add_argument("--environment", default=f"{platform.system().lower()}-{platform.machine().lower()}", help="Runtime environment")
    parser.add_argument("--reason", default="", help="Optional context or error reason")
    parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")
    args = parser.parse_args()

    root = get_project_root()
    proc_file = find_procedure_file(root, args.id, args.file)

    if not proc_file:
        err_msg = f"Procedure not found for id='{args.id}' file='{args.file}'"
        if args.format == "json":
            print(json.dumps({"status": "error", "message": err_msg}, indent=2))
        else:
            print(f"ERROR: {err_msg}", file=sys.stderr)
        sys.exit(1)

    content = proc_file.read_text(encoding="utf-8")
    fm_data, _, body = parse_frontmatter(content)

    if not fm_data:
        err_msg = f"Procedure file '{proc_file.name}' lacks valid YAML frontmatter"
        if args.format == "json":
            print(json.dumps({"status": "error", "message": err_msg}, indent=2))
        else:
            print(f"ERROR: {err_msg}", file=sys.stderr)
        sys.exit(1)

    outcomes = fm_data.get("outcomes", {})
    if not isinstance(outcomes, dict):
        outcomes = {}

    current_env = outcomes.get("environment", args.environment)
    success_count = int(outcomes.get("success_count", 0))
    failure_count = int(outcomes.get("failure_count", 0))
    blocked_count = int(outcomes.get("blocked_count", 0))

    if args.outcome == "success":
        success_count += 1
    elif args.outcome == "failure":
        failure_count += 1
    elif args.outcome == "blocked":
        blocked_count += 1

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    outcomes["environment"] = current_env
    outcomes["success_count"] = success_count
    outcomes["failure_count"] = failure_count
    outcomes["blocked_count"] = blocked_count
    outcomes["last_verified"] = today
    fm_data["outcomes"] = outcomes

    # Re-serialize YAML frontmatter
    new_fm_str = yaml.dump(fm_data, sort_keys=False, default_flow_style=False)
    new_content = f"---\n{new_fm_str}---\n{body}"
    proc_file.write_text(new_content, encoding="utf-8")

    # Append event to .agent/learning/events.jsonl
    events_file = root / ".agent" / "learning" / "events.jsonl"
    events_file.parent.mkdir(parents=True, exist_ok=True)
    event_payload = {
        "event_id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event_type": "PROCEDURE_OUTCOME",
        "procedure_id": fm_data.get("id", proc_file.stem),
        "procedure_version": fm_data.get("version", "1.0.0"),
        "outcome": args.outcome,
        "environment": current_env,
        "reason": args.reason,
        "counts": {
            "success": success_count,
            "failure": failure_count,
            "blocked": blocked_count
        }
    }

    with open(events_file, "a", encoding="utf-8") as ef:
        ef.write(json.dumps(event_payload) + "\n")

    rel_path = str(proc_file.relative_to(root))
    if args.format == "json":
        print(json.dumps({
            "status": "success",
            "procedure": rel_path,
            "procedure_id": fm_data.get("id"),
            "outcome": args.outcome,
            "counts": outcomes
        }, indent=2))
    else:
        print(f"Recorded '{args.outcome.upper()}' for {fm_data.get('id', proc_file.name)} in {rel_path}")
        print(f"Updated counters: Success: {success_count} | Failure: {failure_count} | Blocked: {blocked_count}")


if __name__ == "__main__":
    main()
