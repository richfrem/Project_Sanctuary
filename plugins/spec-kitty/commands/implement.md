---
description: Create isolated worktree for a work package
argument-hint: "<WP-ID>"
---

# Implement Work Package

Create an isolated worktree (`.worktrees/WP-xx`) for implementing a specific WP.

## Usage
```bash
# Create worktree
spec-kitty agent workflow implement --task-id <WP-ID> --agent "<AGENT-NAME>"

# CRITICAL: cd into the worktree
cd .worktrees/<WP-ID>

# Verify you're in the right place
pwd
```

## Kanban Update
```bash
python3 .kittify/scripts/tasks/tasks_cli.py update <FEATURE> <WP-ID> doing \
  --agent "<AGENT-NAME>" --note "Starting implementation"
```

## ⛔ Rules
- **NEVER** edit files in root repo while implementing a WP
- **ALWAYS** verify `pwd` shows the worktree path
- **ALWAYS** commit to the local feature branch, not main
