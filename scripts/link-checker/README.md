# Link Checker - Project Sanctuary

Documentation link validation and auto-repair tools for maintaining repository health.

## Workflow

![link_checker_workflow](../../docs/architecture_diagrams/workflows/link_checker_workflow.png)

*[Source: link_checker_workflow.mmd](../../docs/architecture_diagrams/workflows/link_checker_workflow.mmd)*

---

## 🛠️ Tools

| Script | Description |
| :--- | :--- |
| `verify_links.py` | **Primary Inspector**. Scans markdown files AND manifest JSON files for broken links. Uses the manifest registry. Outputs `invalid_links_report.json`. |
| `check_broken_paths.py` | **Quick Checker**. Lightweight scan of markdown files only. Outputs `broken_links.log`. |
| `map_repository_files.py` | **The Mapper**. Indexes all files in the repo to `file_inventory.json` for smart fixing. |
| `smart_fix_links.py` | **Auto-Fixer**. Uses the file inventory to automatically repair broken links by matching filenames. |

---

## Usage

### Option A: Full Verification (Recommended)

```bash
# From project root
python scripts/link-checker/verify_links.py
```

This scans all markdown files AND manifest JSON files (using `.agent/learning/manifest_registry.json`).

### Option B: Quick Check + Auto-Fix Workflow

```bash
# Step 1: Find broken links
python scripts/link-checker/check_broken_paths.py

# Step 2: Build file inventory 
python scripts/link-checker/map_repository_files.py

# Step 3: Auto-fix (dry run first)
python scripts/link-checker/smart_fix_links.py --dry-run

# Step 4: Apply fixes
python scripts/link-checker/smart_fix_links.py

# Step 5: Verify clean
python scripts/link-checker/verify_links.py
```

---

## Output Files

| File | Description |
| :--- | :--- |
| `invalid_links_report.json` | JSON report from `verify_links.py` (comprehensive) |
| `broken_links.log` | Markdown report from `check_broken_paths.py` |
| `file_inventory.json` | File index for `smart_fix_links.py` |

---

## Integration

These tools are part of the **Protocol 128 Hardened Learning Loop** verification phase. Run `verify_links.py` before sealing a session to ensure documentation integrity.
