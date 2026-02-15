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
