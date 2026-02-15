# Context Bundle
**Generated:** 2026-01-24T10:05:54.035812

Standalone utility for validating and fixing documentation links.

---

## 📑 Table of Contents
1. [UNPACK_INSTRUCTIONS.md](#file-1)
2. [README.md](#file-2)
3. [prompt.md](#file-3)
4. [INSTALL.md](#file-4)
5. [unpacking.mmd](#file-5)
6. [logic.mmd](#file-6)
7. [workflow.mmd](#file-7)
8. [check_broken_paths.py](#file-8)
9. [map_repository_files.py](#file-9)
10. [smart_fix_links.py](#file-10)

---

<a id='file-1'></a>
## 1. UNPACK_INSTRUCTIONS.md
**Path:** `UNPACK_INSTRUCTIONS.md`
**Note:** CRITICAL: How to hydrate this tool

```markdown
# 📦 Bundle Unpacking Protocol
> **🛑 STOP & READ**: Use this protocol to hydrate the tools in this bundle.

## Extraction Logic
1.  **Scan** this document for sections marked with **Path:** metadata (e.g., `**Path:** scripts/run.py`).
2.  **Extract** the code block content immediately following the path.
3.  **Save** the content to the specified filename (relative to your desired tool root).

## ⚠️ Critical Setup Step
After extracting the files, you **MUST** run the mapper script to initialize the inventory:
```bash
python map_repository_files.py
```
*Without this step, the auto-fixer will not work.*

## 📄 File List
The following files are included in this bundle:
- `README.md`
- `INSTALL.md`
- `check_broken_paths.py`
- `map_repository_files.py`
- `smart_fix_links.py`
- *Diagrams for context*

*(See standard Agent Unpacking Process diagram in the bundle for visuals)*

```

---

<a id='file-2'></a>
## 2. README.md
**Path:** `README.md`
**Note:** Documentation

```markdown
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

```

---

<a id='file-3'></a>
## 3. prompt.md
**Path:** `tools/standalone/link-checker/prompt.md`
**Note:** Agent Protocol

> ❌ File not found: C:\Users\RICHFREM\source\repos\Oracle-Forms-Analysis\tools\standalone\link-checker\tools\standalone\link-checker\prompt.md

---

<a id='file-4'></a>
## 4. INSTALL.md
**Path:** `INSTALL.md`
**Note:** Installation Guide

```markdown
# Installation & Unpacking Guide 📂

This guide explains how to restore the "Link Checker" tool from its Markdown distribution bundle into a working directory structure.

## 1. Directory Structure

For optimal usage by an Agent/LLM or in a clean environment, unpack the files into the following standard directory structure:

```text
<your_tool_root>/
├── link-checker/          # Tool Root
│   ├── check_broken_paths.py    # The Inspector
│   ├── map_repository_files.py  # The Mapper
│   ├── smart_fix_links.py       # The Fixer
│   ├── UNPACK_INSTRUCTIONS.md
│   ├── README.md
│   ├── INSTALL.md
│   ├── file_inventory.json      # Generated at runtime
│   └── docs/                    # Context/Diagrams
│       ├── logic.mmd
│       ├── workflow.mmd
│       └── unpacking.mmd
```

## 2. Configuration: Post-Unpack

### Initialization
Unlike other tools, the Link Checker relies on a fresh index of the *current* environment.
**You must run the mapper immediately after unpacking or moving files.**

```bash
cd link-checker
python map_repository_files.py
```

## 4. Localization (Agent / LLM Task) 🤖

**Context**: You (the Agent) have just unpacked this tool into a new location.
**Action**: You must now "localize" the documentation and state.

1.  **Read the `README.md`**: Identify the example command paths.
2.  **Rewrite `README.md`**: Search and replace the example paths to match your actual structure.
3.  **Run Mapper**: Do not trust any pre-existing `file_inventory.json`. Run `map_repository_files.py` to build a map of *your* current file system.

> **Why?** The auto-fixer relies entirely on the `file_inventory.json`. If it's stale or empty, the fixer will do nothing.

```

---

<a id='file-5'></a>
## 5. unpacking.mmd
**Path:** `docs/tools/standalone/link-checker/unpacking.mmd`
**Note:** Diagram: Unpacking Process

> ❌ File not found: C:\Users\RICHFREM\source\repos\Oracle-Forms-Analysis\tools\standalone\link-checker\docs\tools\standalone\link-checker\unpacking.mmd

---

<a id='file-6'></a>
## 6. logic.mmd
**Path:** `docs/tools/standalone/link-checker/logic.mmd`
**Note:** Diagram: Internal Logic

> ❌ File not found: C:\Users\RICHFREM\source\repos\Oracle-Forms-Analysis\tools\standalone\link-checker\docs\tools\standalone\link-checker\logic.mmd

---

<a id='file-7'></a>
## 7. workflow.mmd
**Path:** `docs/tools/standalone/link-checker/workflow.mmd`
**Note:** Diagram: Workflow

> ❌ File not found: C:\Users\RICHFREM\source\repos\Oracle-Forms-Analysis\tools\standalone\link-checker\docs\tools\standalone\link-checker\workflow.mmd

---

<a id='file-8'></a>
## 8. check_broken_paths.py
**Path:** `tools/curate/link-checker/check_broken_paths.py`
**Note:** Script: The Inspector

> ❌ File not found: C:\Users\RICHFREM\source\repos\Oracle-Forms-Analysis\tools\standalone\link-checker\tools\curate\link-checker\check_broken_paths.py

---

<a id='file-9'></a>
## 9. map_repository_files.py
**Path:** `tools/curate/link-checker/map_repository_files.py`
**Note:** Script: The Mapper

> ❌ File not found: C:\Users\RICHFREM\source\repos\Oracle-Forms-Analysis\tools\standalone\link-checker\tools\curate\link-checker\map_repository_files.py

---

<a id='file-10'></a>
## 10. smart_fix_links.py
**Path:** `tools/curate/link-checker/smart_fix_links.py`
**Note:** Script: The Fixer

> ❌ File not found: C:\Users\RICHFREM\source\repos\Oracle-Forms-Analysis\tools\standalone\link-checker\tools\curate\link-checker\smart_fix_links.py

---

