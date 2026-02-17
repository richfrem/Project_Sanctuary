---
description: Run the full Map → Fix → Verify workflow after moving or renaming files
argument-hint: "[target_directory]"
---

# Post-Move Link Check Workflow

When moving or renaming files or folders, run this workflow to fix all broken links.
**Must be performed BEFORE git commit.**

## Quick Pre-Commit Check
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/check_broken_paths.py
```
If `Found 0 broken references` → safe to commit.

---

## Full Workflow

### 1. Complete the file/folder move
- Use `git mv` for tracked files to preserve history

### 2. Build file inventory (Map)
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/map_repository_files.py
```

### 3. Auto-repair broken links (Fix)
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/smart_fix_links.py
```

### 4. Verify repairs (Check)
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/check_broken_paths.py
```

### 5. Repeat steps 2-4 until clean (0 broken references)

### 6. Proceed with git workflow
```bash
git add .
git commit -m "docs: fix broken links after file restructure"
```

## Script Reference

| Script | Purpose |
|:---|:---|
| `check_broken_paths.py` | Scans markdown files for broken links |
| `map_repository_files.py` | Builds file inventory for auto-fixer |
| `smart_fix_links.py` | Auto-repairs broken links using inventory |

## Output Files

| File | Description |
|:---|:---|
| `broken_links.log` | Report from check_broken_paths.py |
| `file_inventory.json` | File index for smart_fix_links.py |
