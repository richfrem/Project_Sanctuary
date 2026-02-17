---
description: Auto-repair broken documentation links using fuzzy matching against the file inventory
argument-hint: "[target_directory]"
---

# Fix Broken Links

Automatically repair broken links by finding each file's new location via the inventory.

## Usage
```bash
python3 ${CLAUDE_PLUGIN_ROOT}/scripts/smart_fix_links.py
```

## Prerequisites
Run `/link-checker:map` first to create the file inventory.

## Behavior
1. Scans all `.md` files for broken `[text](path)` links
2. Extracts the basename from the broken path
3. Looks up the basename in `file_inventory.json`
4. If unique match found → rewrites the link with the correct relative path
5. If ambiguous (multiple files with same name) → skips with warning
6. If not found → marks as `(Reference Missing: filename)`

## Safety
- Only modifies files with actual broken links
- Skips `README.md` basename matches (too ambiguous)
- Preserves anchor fragments (`#section`)
