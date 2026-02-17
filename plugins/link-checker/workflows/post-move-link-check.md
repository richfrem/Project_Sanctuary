---
description: Run link checker after moving or renaming files/folders
---

# Post-Move Link Checker Workflow

When moving or renaming files or folders in the repository, run this workflow to ensure all internal documentation links remain valid.

**MUST be performed BEFORE git commit and push operations.**

## Quick Check (Pre-Commit)

// turbo
```bash
python scripts/link-checker/verify_links.py
```

If `Found issues in 0 files` → safe to commit.

---

## Full Workflow Steps

### 1. Complete the file/folder move or rename operation
- Use `git mv` for tracked files to preserve history
- Or use standard move/rename commands

### 2. Run the comprehensive link checker
// turbo
```bash
python scripts/link-checker/verify_links.py
```
This scans both markdown files AND manifest JSON files.

### 3. Review the report
```bash
cat scripts/link-checker/invalid_links_report.json
```

### 4. If broken links are found, use the auto-fixer

// turbo
```bash
# Build file inventory
python scripts/link-checker/map_repository_files.py

# Preview fixes (dry run)
python scripts/link-checker/smart_fix_links.py --dry-run

# Apply fixes
python scripts/link-checker/smart_fix_links.py
```

### 5. Re-run verification
// turbo
```bash
python scripts/link-checker/verify_links.py
```

### 6. Repeat steps 4-5 until clean (0 files with issues)

### 7. Proceed with git workflow
```bash
git add .
git commit -m "docs: fix broken links after file restructure"
git push
```

---

## Script Reference

| Script | Purpose |
|--------|---------|
| `verify_links.py` | **Primary** - Scans markdown + manifest JSON files |
| `check_broken_paths.py` | Quick markdown-only check |
| `map_repository_files.py` | Builds file inventory for auto-fixer |
| `smart_fix_links.py` | Auto-repairs broken links using inventory |

## Output Files

| File | Description |
|------|-------------|
| `invalid_links_report.json` | Comprehensive report (verify_links.py) |
| `broken_links.log` | Quick report (check_broken_paths.py) |
| `file_inventory.json` | File index for smart_fix_links.py |

All outputs are saved to `scripts/link-checker/`.