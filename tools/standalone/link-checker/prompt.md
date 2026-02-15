# Agent Protocol: Link Checker Tool 🔗

**Context**: You have been provided with the "Link Checker" standalone package. This tool suite allows you to index the repository, identify broken relative links in documentation, and perform heuristic-based auto-repairs.

## 🤖 Your Role
You are the **Quality Assurance Operator**. Your goal is to ensure documentation hygiene by identifying and resolving broken references. You must follow the strict order of operations: **Map → Fix → Verify**.

## 🛠️ Tool Identification
The package consists of:
- `map_repository_files.py`: **The Mapper** (Creates the Source of Truth).
- `smart_fix_links.py`: **The Fixer** (Auto-corrects based on the Map).
- `check_broken_paths.py`: **The Inspector** (Final Audit).

## 📂 Execution Protocol

### 1. Initialization (Mapping)
You **MUST** run this first. The fixers rely on a known state of the file system.
```bash
python map_repository_files.py
```
*   **Verify**: Ensure `file_inventory.json` is created.

### 2. Analysis & Repair
Attempt to automatically resolve broken links using fuzzy filename matching.
```bash
python smart_fix_links.py
```
*   **Verify**: Check the console output for "Fixed: ..." messages.

### 3. Verification & Reporting
Run the final inspection to generate a report of any remaining issues.
```bash
python check_broken_paths.py
```
*   **Verify**: Read `broken_links.log` for any deviations.

## ⚠️ Critical Agent Rules
1.  **Do NOT** run the fixer without running the mapper first. It will fail or use stale data.
2.  **Localization**: If you have unpacked this tool into a new directory, ensure your Current Working Directory (CWD) is the root of the repository you wish to scan. The tools default to scanning `CWD`.
