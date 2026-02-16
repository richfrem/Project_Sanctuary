---
description: Automated batch merge of all WP worktrees into main
argument-hint: "[--feature <SLUG>] [--strategy squash] [--push]"
---

# Merge Feature

Run from **Main Repo Root**. Merges ALL WP worktrees into local `main`, cleans up worktrees and branches.

## Usage
```bash
# Version check first
spec-kitty upgrade --dry-run

# State preservation (backup untracked files)
cp .worktrees/<WP-ID>/.env .env.backup  # Example

# Automated merge
spec-kitty merge --feature <FEATURE-SLUG> --strategy squash --push

# Final kanban update
python3 .kittify/scripts/tasks/tasks_cli.py update <FEATURE> <WP-ID> done \
  --agent "<AGENT-NAME>" --note "Feature fully integrated"
```

## ⚠️ Caution
- Merge tool **deletes worktrees** — backup untracked state FIRST
- **NEVER** manually delete worktrees before merge

## Manual Fallback
If `spec-kitty merge` fails:
```bash
git merge <WP-BRANCH-NAME>
git worktree remove .worktrees/<WP-FOLDER>
git branch -d <WP-BRANCH-NAME>
```
