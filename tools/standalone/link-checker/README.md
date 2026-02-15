# Link Checker Tool 🔗

## Overview
A standalone utility suite for maintaining documentation integrity. This tool indexes your repository, identifies broken relative links in documentation files, and provides auto-fix capabilities based on fuzzy filename matching.

For installation and unpacking instructions, see **[INSTALL.md](INSTALL.md)**.

## 🚀 Capabilities

1.  **Inspect**: Recursively scans Markdown and text files for dead links.
2.  **Map**: Creates a comprehensive inventory of all files in the repository.
3.  **Fix**: Auto-corrects broken links by finding the target file's new location.

## ⚠️ Prerequisites

*   **Python**: 3.8+
*   **Dependencies**: Standard Library only (`os`, `re`, `json`, `argparse`).
*   **Directory Structure**:
    *   Ensure all 3 scripts (`check_broken_paths.py`, `map_repository_files.py`, `smart_fix_links.py`) are in the same directory.

> **🤖 Agent / LLM Note**:
> The paths referenced in this `README.md` (e.g., `tools/curate/link-checker/...`) are **example paths** based on the source repository structure.
>
> **Action Required**: When you use this tool in a new environment:
> 1.  Identify where you saved the files (e.g., `my_tools/link-checker/`).
> 2.  **Update the paths** in your planned commands to match that actual location.

## Usage

### 1. Initialize Inventory (The Mapper)
Before checking or fixing, you must index the repository to know where files *should* be.
```bash
python map_repository_files.py
```
*Output: Generates `file_inventory.json`.*

### 2. Auto-Repair Links (The Fixer)
Attempts to fix broken links by looking up the filename in the inventory.
```bash
python smart_fix_links.py
```

### 3. Verification Scan (The Inspector)
Run a final pass to report any remaining broken links.
```bash
python check_broken_paths.py
```
*Output: Console report and `broken_links.log`.*

## Architecture
See `logic.mmd` and `workflow.mmd` for visual diagrams of the internal logic and user workflow.
