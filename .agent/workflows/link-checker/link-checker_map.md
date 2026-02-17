---
description: Index all repository files to create the link resolution inventory
argument-hint: "[target_directory]"
---

# Map Repository Files

Create a `file_inventory.json` that indexes every filename to its path(s).
This inventory is required before running `check` or `fix`.

## Usage
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/map_repository_files.py
```

## Behavior
1. Walks the current working directory recursively
2. Excludes `.git`, `node_modules`, `.venv`, `.next`, `bin`, `obj`
3. Maps each filename → list of relative paths (handles duplicates)
4. Saves to `file_inventory.json` next to the script

## Important
**This must be run first** — the `fix` and `check` commands depend on this inventory.
