#!/usr/bin/env python3
"""
setup_git_hooks.py - Install Project Sanctuary Git Governance Hooks
===================================================================

Purpose:
    Installs and configures Git pre-commit hooks to enforce the Cognitive
    Investment Gate, Backlog Integrity, and Zero-Trust Governance.
"""

import os
import stat
import subprocess
from pathlib import Path


def get_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


def install_hooks() -> None:
    root = get_project_root()
    hooks_dir = root / ".git" / "hooks"
    agent_hooks_dir = root / ".agent" / "hooks"

    if not hooks_dir.exists():
        print("No .git/hooks directory found. Not a git repository.")
        return

    agent_hooks_dir.mkdir(parents=True, exist_ok=True)
    cognitive_script = agent_hooks_dir / "pre-commit-cognitive-check.sh"
    if cognitive_script.exists():
        cognitive_script.chmod(cognitive_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    pre_commit = hooks_dir / "pre-commit"
    hook_call = 'REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"\nif [ -x "$REPO_ROOT/.agent/hooks/pre-commit-cognitive-check.sh" ]; then\n    "$REPO_ROOT/.agent/hooks/pre-commit-cognitive-check.sh" || exit 1\nfi\n'

    if pre_commit.exists():
        content = pre_commit.read_text(encoding="utf-8")
        if "pre-commit-cognitive-check.sh" not in content:
            # Insert before exit 0
            if "exit 0" in content:
                content = content.replace("exit 0", f"{hook_call}\nexit 0")
            else:
                content += f"\n{hook_call}\n"
            pre_commit.write_text(content, encoding="utf-8")
            print("✓ Updated existing .git/hooks/pre-commit with cognitive hook.")
        else:
            print("✓ .git/hooks/pre-commit already configured.")
    else:
        new_content = f"#!/usr/bin/env bash\nset -eo pipefail\n{hook_call}\nexit 0\n"
        pre_commit.write_text(new_content, encoding="utf-8")
        print("✓ Created new .git/hooks/pre-commit with cognitive hook.")

    pre_commit.chmod(pre_commit.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print("✓ Project Sanctuary cognitive pre-commit hooks successfully verified and installed.")


if __name__ == "__main__":
    install_hooks()
